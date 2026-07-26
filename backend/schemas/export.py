from __future__ import annotations

from pydantic import BaseModel


class ExportRequest(BaseModel):
    export_type: str

