"""Shared route helpers.

Keeps the translation from core exceptions to HTTP status codes in one place,
so that adding a pipeline stage does not mean re-deriving the mapping.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import HTTPException, UploadFile

from gausscapture.config import get_settings
from gausscapture.errors import (
    CaptureFormatError,
    DependencyMissingError,
    GaussCaptureError,
    PipelineStateError,
)
from gausscapture.project import Project, ProjectStore

store = ProjectStore()


def get_project(project_id: str) -> Project:
    try:
        return store.get(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def project_path(project_id: str) -> Path:
    return get_project(project_id).path


def to_http(exc: GaussCaptureError) -> HTTPException:
    """Map a core error to a status code a client can act on.

    412 for "you skipped a step", 424 for "install this first", 400 for "your
    file is malformed" -- distinguishable in the UI without string matching.
    """
    if isinstance(exc, PipelineStateError):
        return HTTPException(status_code=412, detail=str(exc))
    if isinstance(exc, DependencyMissingError):
        return HTTPException(status_code=424, detail=str(exc))
    if isinstance(exc, CaptureFormatError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


def save_upload(file: UploadFile, allowed_suffixes: set[str]) -> Path:
    """Stream an upload to a temporary file, validating its extension."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed_suffixes:
        raise HTTPException(
            status_code=400,
            detail=f"Upload must be one of: {', '.join(sorted(allowed_suffixes))}",
        )
    temp_dir = Path(get_settings().projects_dir).parent / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    target = temp_dir / f"upload_{Path(file.filename or 'file').name}"
    if not target.suffix:
        target = target.with_suffix(suffix)
    with target.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    return target
