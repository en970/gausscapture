from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.config import DATA_DIR
from backend.core.capturepack import create_minimal_capturepack, import_capturepack_archive, validate_capturepack
from backend.core.preview_builder import import_training_result
from backend.core.project_manager import PROJECT_STATUS_IMPORTED, PROJECT_STATUS_PREVIEW, PROJECT_STATUS_TRAINED, project_manager
from backend.core.video_importer import is_video_file

router = APIRouter(prefix="/api/projects/{project_id}/import", tags=["import"])


@router.post("/video")
def import_video(project_id: str, file: UploadFile = File(...)):
    project = _project(project_id)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}:
        raise HTTPException(status_code=400, detail="Upload must be a video file")
    temp = _save_upload(file, suffix)
    try:
        manifest = create_minimal_capturepack(Path(project["path"]), temp, project["name"], project.get("target_type", "unknown"))
        project = project_manager.update_project(project_id, status=PROJECT_STATUS_IMPORTED, last_step="Video imported as minimal CapturePack")
        return {"project": project, "manifest": manifest, "warnings": ["Minimal CapturePack created from video. Camera pose metadata is not available."], "errors": []}
    finally:
        temp.unlink(missing_ok=True)


@router.post("/capturepack")
def import_capturepack(project_id: str, file: UploadFile = File(...)):
    project = _project(project_id)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".capturepack", ".zip"}:
        raise HTTPException(status_code=400, detail="Upload must be .capturepack or .zip")
    temp = _save_upload(file, suffix)
    try:
        result = import_capturepack_archive(Path(project["path"]), temp)
        if not result["valid"]:
            raise HTTPException(status_code=400, detail={"errors": result["errors"], "warnings": result["warnings"]})
        project = project_manager.update_project(project_id, status=PROJECT_STATUS_IMPORTED, last_step="CapturePack imported")
        return {"project": project, **result}
    finally:
        temp.unlink(missing_ok=True)


@router.post("/training-result")
def import_training(project_id: str, file: UploadFile = File(...)):
    project = _project(project_id)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".zip", ".ply", ".splat", ".ksplat"}:
        raise HTTPException(status_code=400, detail="Upload must be .zip, .ply, .splat, or .ksplat")
    temp = _save_upload(file, suffix)
    try:
        result = import_training_result(Path(project["path"]), temp)
        project = project_manager.update_project(project_id, status=PROJECT_STATUS_PREVIEW, last_step="Training result imported")
        return {"project": project, **result}
    finally:
        temp.unlink(missing_ok=True)


@router.get("/validate")
def validate_current_capturepack(project_id: str):
    project = _project(project_id)
    return validate_capturepack(Path(project["path"]) / "capturepack")


def _project(project_id: str) -> dict:
    try:
        return project_manager.get_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _save_upload(file: UploadFile, suffix: str) -> Path:
    temp_dir = DATA_DIR / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    target = temp_dir / f"upload_{file.filename or 'file'}"
    if not target.suffix:
        target = target.with_suffix(suffix)
    with target.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    return target
