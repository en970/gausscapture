from __future__ import annotations

from pydantic import BaseModel


class ProjectCreate(BaseModel):
    name: str
    target_type: str = "unknown"


class ProjectOut(BaseModel):
    id: str
    name: str
    target_type: str = "unknown"
    status: str
    created_at: str
    updated_at: str
    path: str
    last_step: str | None = None

