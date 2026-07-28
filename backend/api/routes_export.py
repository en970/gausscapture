from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.deps import get_project, store, to_http
from backend.schemas.export import ExportRequest
from gausscapture.errors import GaussCaptureError
from gausscapture.export.bundles import create_export, export_path, list_exports
from gausscapture.project import STATUS_EXPORTED

router = APIRouter(prefix="/api/projects/{project_id}/export", tags=["export"])


@router.get("")
def exports(project_id: str):
    return list_exports(get_project(project_id).path)


@router.post("")
def create(project_id: str, payload: ExportRequest):
    project = get_project(project_id)
    try:
        result = create_export(project.path, payload.export_type)
    except GaussCaptureError as exc:
        raise to_http(exc) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=412, detail=str(exc)) from exc
    store.update(project_id, status=STATUS_EXPORTED, last_step=f"Exported {payload.export_type}")
    return result


@router.get("/{export_id}/download")
def download(project_id: str, export_id: str):
    project = get_project(project_id)
    try:
        path = export_path(project.path, export_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Export not found") from exc
    return FileResponse(path, filename=path.name)
