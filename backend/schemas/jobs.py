from __future__ import annotations

from pydantic import BaseModel


class JobOut(BaseModel):
    id: str
    kind: str
    project_id: str | None = None
    status: str
    progress: int
    current_step: str
    created_at: str
    updated_at: str
    error: str | None = None
    result: dict | None = None

