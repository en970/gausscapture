from __future__ import annotations

import json
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable

from backend.config import DATA_DIR
from backend.core.logging_utils import append_log, utc_now


class Job:
    def __init__(self, kind: str, project_id: str | None = None, log_path: Path | None = None):
        self.id = str(uuid.uuid4())
        self.kind = kind
        self.project_id = project_id
        self.status = "queued"
        self.progress = 0
        self.current_step = "Queued"
        self.created_at = utc_now()
        self.updated_at = utc_now()
        self.error: str | None = None
        self.result: dict[str, Any] | None = None
        self.logs: list[str] = []
        self.log_path = log_path

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "project_id": self.project_id,
            "status": self.status,
            "progress": self.progress,
            "current_step": self.current_step,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "result": self.result,
        }

    def log(self, message: str) -> None:
        line = f"{message.rstrip()}"
        self.logs.append(line)
        self.updated_at = utc_now()
        if self.log_path:
            append_log(self.log_path, line)

    def set_progress(self, progress: int, step: str | None = None) -> None:
        self.progress = max(0, min(100, int(progress)))
        if step:
            self.current_step = step
            self.log(step)
        self.updated_at = utc_now()


class JobManager:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.history_dir = DATA_DIR / "jobs"
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def start(
        self,
        kind: str,
        func: Callable[..., dict[str, Any] | None],
        *,
        project_id: str | None = None,
        log_path: Path | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Job:
        job = Job(kind=kind, project_id=project_id, log_path=log_path)
        with self.lock:
            self.jobs[job.id] = job
        thread = threading.Thread(target=self._run, args=(job, func, args, kwargs or {}), daemon=True)
        thread.start()
        return job

    def _run(self, job: Job, func: Callable[..., dict[str, Any] | None], args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        job.status = "running"
        job.set_progress(1, "Started")
        self._persist(job)
        try:
            result = func(job, *args, **kwargs)
            job.result = result or {}
            job.status = "success"
            job.set_progress(100, "Completed")
        except Exception as exc:
            job.status = "error"
            job.error = str(exc)
            job.current_step = "Error"
            job.log(str(exc))
            job.log(traceback.format_exc())
        finally:
            self._persist(job)

    def get(self, job_id: str) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if job:
            return job.to_dict()
        path = self.history_dir / f"{job_id}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        raise KeyError(job_id)

    def logs(self, job_id: str) -> str:
        job = self.jobs.get(job_id)
        if job:
            return "\n".join(job.logs)
        data = self.get(job_id)
        return "\n".join(data.get("logs", []))

    def _persist(self, job: Job) -> None:
        data = job.to_dict()
        data["logs"] = job.logs[-1000:]
        (self.history_dir / f"{job.id}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


job_manager = JobManager()

