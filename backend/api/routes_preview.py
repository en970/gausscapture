from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.core.preview_builder import build_preview_from_latest_training, preview_status
from backend.core.project_manager import PROJECT_STATUS_PREVIEW, project_manager

router = APIRouter(prefix="/api/projects/{project_id}/preview", tags=["preview"])
files_router = APIRouter(prefix="/api/projects/{project_id}/files", tags=["files"])


@router.get("/status")
def status(project_id: str):
    project = _project(project_id)
    return preview_status(Path(project["path"]))


@router.post("/build")
def build(project_id: str):
    project = _project(project_id)
    result = build_preview_from_latest_training(Path(project["path"]))
    project_manager.update_project(project_id, status=PROJECT_STATUS_PREVIEW, last_step="Preview built")
    return result


@router.get("/model")
def model(project_id: str):
    project = _project(project_id)
    config_path = Path(project["path"]) / "preview" / "preview_config.json"
    if not config_path.exists():
        raise HTTPException(status_code=404, detail="Preview config not found")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_path = Path(project["path"]) / "preview" / config["model_file"]
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Preview model not found")
    return FileResponse(model_path, filename=model_path.name)


@router.get("/config")
def config(project_id: str):
    project = _project(project_id)
    config_path = Path(project["path"]) / "preview" / "preview_config.json"
    if not config_path.exists():
        raise HTTPException(status_code=404, detail="Preview config not found")
    return json.loads(config_path.read_text(encoding="utf-8"))


@files_router.get("/{relative_path:path}")
def project_file(project_id: str, relative_path: str):
    project = _project(project_id)
    root = Path(project["path"]).resolve()
    target = (root / relative_path).resolve()
    if root not in target.parents and target != root:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target)


def _project(project_id: str) -> dict:
    try:
        return project_manager.get_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

