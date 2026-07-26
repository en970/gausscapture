from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from backend.core.job_manager import job_manager

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("/{job_id}")
def get_job(job_id: str):
    try:
        return job_manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


@router.get("/{job_id}/logs")
def get_logs(job_id: str):
    try:
        return PlainTextResponse(job_manager.logs(job_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc

