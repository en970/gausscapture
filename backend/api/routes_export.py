from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.core.export_manager import create_export, export_path, list_exports
from backend.core.job_manager import job_manager
from backend.core.project_manager import PROJECT_STATUS_EXPORTED, project_manager
from backend.schemas.export import ExportRequest

router = APIRouter(prefix="/api/projects/{project_id}/export", tags=["export"])


@router.post("")
def export(project_id: str, payload: ExportRequest):
    project = _project(project_id)
    path = Path(project["path"])

    def task(job):
        job.set_progress(20, f"Creating {payload.export_type} export")
        result = create_export(path, payload.export_type)
        project_manager.update_project(project_id, status=PROJECT_STATUS_EXPORTED, last_step=f"{payload.export_type} export created")
        return result

    job = job_manager.start("export", task, project_id=project_id, log_path=path / "logs" / "export.log")
    return job.to_dict()


@router.get("/list")
def list_project_exports(project_id: str):
    project = _project(project_id)
    return list_exports(Path(project["path"]))


@router.get("/download/{export_id}")
def download(project_id: str, export_id: str):
    project = _project(project_id)
    try:
        path = export_path(Path(project["path"]), export_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, filename=path.name)


def _project(project_id: str) -> dict:
    try:
        return project_manager.get_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

