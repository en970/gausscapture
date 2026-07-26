from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from backend.deps import get_project, store
from backend.schemas.project import ProjectCreate
from gausscapture.config import Settings, get_settings, save_settings

router = APIRouter(prefix="/api/projects", tags=["projects"])
settings_router = APIRouter(prefix="/api/settings", tags=["settings"])


class SettingsPayload(BaseModel):
    """Pydantic mirror of the core settings dataclass, for request validation.

    The core deliberately uses a plain dataclass so the library does not depend
    on a web framework's validation layer; this model is the web layer's own
    concern.
    """

    projects_dir: str
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    colmap_path: str = "colmap"
    python_path: str = "python"
    gaussian_trainer_path: str = ""
    default_preview_port: int = 7860
    default_frame_preset: str = "balanced"
    default_training_preset: str = "draft"


@router.get("")
def list_projects():
    return [p.to_dict() for p in store.list()]


@router.post("")
def create_project(payload: ProjectCreate):
    return store.create(payload.name, payload.target_type).to_dict()


@router.get("/{project_id}")
def read_project(project_id: str):
    return get_project(project_id).to_dict()


@router.delete("/{project_id}")
def delete_project(project_id: str):
    get_project(project_id)  # 404s if absent
    store.delete(project_id)
    return {"ok": True}


@settings_router.get("")
def read_settings():
    return get_settings().to_dict()


@settings_router.put("")
def update_settings(payload: SettingsPayload):
    Path(payload.projects_dir).mkdir(parents=True, exist_ok=True)
    save_settings(Settings(**payload.model_dump()))
    return payload.model_dump()
