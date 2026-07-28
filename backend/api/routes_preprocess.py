from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.deps import get_project, store
from backend.jobs import job_manager
from gausscapture.ingest.frames import extract_frames
from gausscapture.pose.colmap import colmap_available, run_colmap
from gausscapture.project import STATUS_PREPROCESSED, STATUS_READY
from gausscapture.recon.dataset import build_dataset

router = APIRouter(prefix="/api/projects/{project_id}/preprocess", tags=["preprocess"])


class FrameExtractionRequest(BaseModel):
    target_fps: int | str = 2
    max_frames: int | str | None = 600
    resize_max_side: int | str | None = 1920
    blur_filter: bool = True
    duplicate_filter: bool = True
    #: Reject frames softer than this fraction of the clip's rolling median.
    blur_relative_threshold: float = 0.60


@router.post("/extract-frames")
def extract(project_id: str, payload: FrameExtractionRequest):
    project = get_project(project_id)

    def task(job):
        index = extract_frames(project.path, payload.model_dump(), progress=job)
        store.update(project_id, status=STATUS_PREPROCESSED, last_step="Frames extracted")
        return {
            "frame_index": str(project.frame_index_path),
            **{k: v for k, v in index.to_dict().items() if k != "frames"},
        }

    job = job_manager.start(
        "frame_extraction",
        task,
        project_id=project_id,
        log_path=project.path / "logs" / "preprocess.log",
    )
    return job.to_dict()


@router.get("/frame-index")
def frame_index(project_id: str):
    path = get_project(project_id).frame_index_path
    if not path.exists():
        raise HTTPException(status_code=404, detail="Frame index not found")
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/colmap-available")
def colmap_check():
    return {"available": colmap_available()}


@router.post("/run-colmap")
def colmap(project_id: str, matcher: str = "sequential"):
    project = get_project(project_id)

    def task(job):
        report = run_colmap(project.path, matcher=matcher, progress=job)
        # A usable reconstruction is immediately staged into the layout a
        # trainer expects, so the next step needs no extra call.
        dataset = None
        if report.status != "bad":
            dataset = str(build_dataset(project.path, progress=job))
            store.update(project_id, status=STATUS_READY, last_step="Poses estimated")
        return {"pose_report": report.to_dict(), "dataset": dataset}

    job = job_manager.start(
        "colmap", task, project_id=project_id, log_path=project.path / "logs" / "colmap.log"
    )
    return job.to_dict()


@router.get("/pose-report")
def pose_report(project_id: str):
    path = get_project(project_id).colmap_dir / "pose_report.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Pose report not found")
    return json.loads(path.read_text(encoding="utf-8"))
