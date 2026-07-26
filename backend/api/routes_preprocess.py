from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.core.colmap_runner import run_colmap
from backend.core.frame_extractor import extract_frames
from backend.core.job_manager import job_manager
from backend.core.project_manager import PROJECT_STATUS_PREPROCESSED, PROJECT_STATUS_READY, project_manager

router = APIRouter(prefix="/api/projects/{project_id}/preprocess", tags=["preprocess"])


class FrameExtractionRequest(BaseModel):
    target_fps: int | str = 2
    max_frames: int | str | None = 600
    resize_max_side: int | str | None = 1920
    blur_filter: bool = True
    duplicate_filter: bool = True


@router.post("/extract-frames")
def extract(project_id: str, payload: FrameExtractionRequest):
    project = _project(project_id)
    path = Path(project["path"])

    def task(job):
        result = extract_frames(job, path, payload.model_dump())
        project_manager.update_project(project_id, status=PROJECT_STATUS_PREPROCESSED, last_step="Frames extracted")
        return result

    job = job_manager.start("frame_extraction", task, project_id=project_id, log_path=path / "logs" / "preprocess.log")
    return job.to_dict()


@router.get("/frame-index")
def frame_index(project_id: str):
    project = _project(project_id)
    path = Path(project["path"]) / "frames" / "frame_index.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Frame index not found")
    return json.loads(path.read_text(encoding="utf-8"))


@router.post("/run-colmap")
def colmap(project_id: str):
    project = _project(project_id)
    path = Path(project["path"])

    def task(job):
        result = run_colmap(job, path)
        project_manager.update_project(project_id, status=PROJECT_STATUS_READY, last_step="COLMAP completed")
        return result

    job = job_manager.start("colmap", task, project_id=project_id, log_path=path / "logs" / "colmap.log")
    return job.to_dict()


@router.get("/pose-report")
def pose_report(project_id: str):
    project = _project(project_id)
    path = Path(project["path"]) / "colmap" / "pose_report.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Pose report not found")
    return json.loads(path.read_text(encoding="utf-8"))


def _project(project_id: str) -> dict[str, Any]:
    try:
        return project_manager.get_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

