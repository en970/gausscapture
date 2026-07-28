from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.deps import get_project, store
from backend.jobs import job_manager
from backend.schemas.training import ColabPackageRequest, TrainingRequest
from gausscapture.export.colab import create_colab_package
from gausscapture.project import STATUS_TRAINED, STATUS_TRAINING
from gausscapture.recon.dataset import build_dataset
from gausscapture.recon.external import ExternalTrainer, trainer_status

router = APIRouter(prefix="/api/projects/{project_id}/training", tags=["training"])


@router.get("/status")
def status(project_id: str):
    project = get_project(project_id)
    return {"project_status": project.status, "trainer": trainer_status()}


@router.post("/local")
def local_training(project_id: str, payload: TrainingRequest):
    project = get_project(project_id)

    def task(job):
        store.update(project_id, status=STATUS_TRAINING, last_step="Local training running")
        dataset_dir = build_dataset(project.path, progress=job)
        run_dir = _next_run_dir(project)
        summary = ExternalTrainer().fit(
            dataset_dir, run_dir / "output", preset=payload.preset, progress=job
        )
        store.update(project_id, status=STATUS_TRAINED, last_step="Local training completed")
        return summary

    job = job_manager.start(
        "local_training",
        task,
        project_id=project_id,
        log_path=project.path / "logs" / "training.log",
    )
    return job.to_dict()


@router.post("/create-colab-package")
def colab_package(project_id: str, payload: ColabPackageRequest):
    project = get_project(project_id)

    def task(job):
        return create_colab_package(project.path, payload.model_dump(), progress=job)

    job = job_manager.start(
        "colab_package",
        task,
        project_id=project_id,
        log_path=project.path / "logs" / "training.log",
    )
    return job.to_dict()


@router.get("/download-colab-package")
def download_colab(project_id: str):
    path = get_project(project_id).path / "colab_export" / "dataset.zip"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Colab package not found")
    return FileResponse(path, filename="dataset.zip")


def _next_run_dir(project):
    training_dir = project.path / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    index = len([p for p in training_dir.glob("run_*") if p.is_dir()]) + 1
    run_dir = training_dir / f"run_{index:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
