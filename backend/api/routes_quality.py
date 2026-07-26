from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.core.job_manager import job_manager
from backend.core.project_manager import project_manager
from backend.core.quality_analyzer import analyze_quality

router = APIRouter(prefix="/api/projects/{project_id}/quality", tags=["quality"])


@router.post("/analyze")
def analyze(project_id: str):
    project = _project(project_id)
    path = Path(project["path"])

    def task(job):
        result = analyze_quality(job, path)
        project_manager.update_project(project_id, last_step="Quality analysis completed")
        return result

    job = job_manager.start("quality_analysis", task, project_id=project_id, log_path=path / "logs" / "quality.log")
    return job.to_dict()


@router.get("/report")
def report(project_id: str):
    project = _project(project_id)
    path = Path(project["path"]) / "quality" / "quality_report.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Quality report not found")
    return json.loads(path.read_text(encoding="utf-8"))


def _project(project_id: str) -> dict:
    try:
        return project_manager.get_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

