from __future__ import annotations

from pydantic import BaseModel


class TrainingRequest(BaseModel):
    mode: str = "dry_run"
    preset: str = "draft"
    scene_type: str = "object"


class ColabPackageRequest(BaseModel):
    preset: str = "draft"
    scene_type: str = "object"
    notes: str = ""

