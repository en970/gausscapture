from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.core.colab_packager import create_colab_package
from backend.core.gaussian_runner import run_local_training, trainer_status
from backend.core.job_manager import job_manager
from backend.core.project_manager import PROJECT_STATUS_TRAINED, PROJECT_STATUS_TRAINING, project_manager
from backend.schemas.training import ColabPackageRequest, TrainingRequest

router = APIRouter(prefix="/api/projects/{project_id}/training", tags=["training"])


@router.get("/status")
def status(project_id: str):
    project = _project(project_id)
    return {"project_status": project["status"], "trainer": trainer_status()}


@router.post("/local")
def local_training(project_id: str, payload: TrainingRequest):
    project = _project(project_id)
    path = Path(project["path"])

    def task(job):
        project_manager.update_project(project_id, status=PROJECT_STATUS_TRAINING, last_step="Local training running")
        result = run_local_training(job, path, payload.preset, payload.scene_type)
        project_manager.update_project(project_id, status=PROJECT_STATUS_TRAINED, last_step="Local training completed")
        return result

    job = job_manager.start("local_training", task, project_id=project_id, log_path=path / "logs" / "training.log")
    return job.to_dict()


@router.post("/create-colab-package")
def colab_package(project_id: str, payload: ColabPackageRequest):
    project = _project(project_id)
    path = Path(project["path"])

    def task(job):
        return create_colab_package(job, path, payload.model_dump())

    job = job_manager.start("colab_package", task, project_id=project_id, log_path=path / "logs" / "training.log")
    return job.to_dict()


@router.get("/download-colab-package")
def download_colab(project_id: str):
    project = _project(project_id)
    path = Path(project["path"]) / "colab_export" / "dataset.zip"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Colab package not found")
    return FileResponse(path, filename="dataset.zip")


def _project(project_id: str) -> dict:
    try:
        return project_manager.get_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

