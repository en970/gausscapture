from __future__ import annotations

from pydantic import BaseModel


class ImportResult(BaseModel):
    project: dict
    manifest: dict | None = None
    warnings: list[str] = []
    errors: list[str] = []

