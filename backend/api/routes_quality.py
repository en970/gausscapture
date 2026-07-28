from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from backend.deps import get_project, store
from backend.jobs import job_manager
from gausscapture.telemetry import analyze_capture

router = APIRouter(prefix="/api/projects/{project_id}/quality", tags=["quality"])


@router.post("/analyze")
def analyze(project_id: str):
    project = get_project(project_id)

    def task(job):
        report = analyze_capture(project.path, progress=job)
        store.update(project_id, last_step="Quality analysis completed")
        # Per-frame signals are large; the UI reads them from the report file
        # when it needs them rather than through the job payload.
        return {
            "quality_report": str(project.quality_report_path),
            "report": report.to_dict(include_signals=False),
        }

    job = job_manager.start(
        "quality_analysis",
        task,
        project_id=project_id,
        log_path=project.path / "logs" / "quality.log",
    )
    return job.to_dict()


@router.get("/report")
def report(project_id: str):
    path = get_project(project_id).quality_report_path
    if not path.exists():
        raise HTTPException(status_code=404, detail="Quality report not found")
    return json.loads(path.read_text(encoding="utf-8"))
