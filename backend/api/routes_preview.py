from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse

from backend.deps import get_project, store, to_http
from gausscapture.errors import GaussCaptureError
from gausscapture.export.preview import (
    build_preview_from_latest_training,
    preview_model_path,
    preview_status,
)
from gausscapture.project import STATUS_PREVIEW

router = APIRouter(prefix="/api/projects/{project_id}/preview", tags=["preview"])
files_router = APIRouter(prefix="/api/projects/{project_id}/files", tags=["files"])


@router.get("/status")
def status(project_id: str):
    return preview_status(get_project(project_id).path)


@router.post("/build")
def build(project_id: str):
    project = get_project(project_id)
    try:
        result = build_preview_from_latest_training(project.path)
    except GaussCaptureError as exc:
        raise to_http(exc) from exc
    store.update(project_id, status=STATUS_PREVIEW, last_step="Preview built")
    return result


@router.get("/model")
def model(project_id: str):
    project = get_project(project_id)
    try:
        path = preview_model_path(project.path)
    except GaussCaptureError as exc:
        raise to_http(exc) from exc
    return FileResponse(path, filename=path.name)


@files_router.get("/quality-thumbnail/{name}")
def quality_thumbnail(project_id: str, name: str):
    project = get_project(project_id)
    # Resolve and confine to the thumbnails directory: `name` is user input.
    root = (project.path / "quality" / "thumbnails").resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.exists():
        raise to_http(GaussCaptureError("Thumbnail not found"))
    return FileResponse(path)
