from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.config import AppSettings, load_settings, save_settings
from backend.core.project_manager import project_manager
from backend.schemas.project import ProjectCreate

router = APIRouter(prefix="/api/projects", tags=["projects"])
settings_router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("")
def list_projects():
    return project_manager.list_projects()


@router.post("")
def create_project(payload: ProjectCreate):
    return project_manager.create_project(payload.name, payload.target_type)


@router.get("/{project_id}")
def get_project(project_id: str):
    try:
        return project_manager.get_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/{project_id}")
def delete_project(project_id: str):
    try:
        project_manager.delete_project(project_id)
        return {"ok": True}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@settings_router.get("")
def get_settings():
    return load_settings()


@settings_router.put("")
def update_settings(payload: AppSettings):
    Path(payload.projects_dir).mkdir(parents=True, exist_ok=True)
    save_settings(payload)
    return payload

