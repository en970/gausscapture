from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from backend.config import settings
from backend.core.logging_utils import utc_now


PROJECT_STATUS_IMPORTED = "Imported"
PROJECT_STATUS_PREPROCESSED = "Preprocessed"
PROJECT_STATUS_READY = "Ready for Training"
PROJECT_STATUS_TRAINING = "Training"
PROJECT_STATUS_TRAINED = "Trained"
PROJECT_STATUS_PREVIEW = "Preview Ready"
PROJECT_STATUS_EXPORTED = "Exported"
PROJECT_STATUS_ERROR = "Error"


class ProjectManager:
    def __init__(self, projects_dir: Path | None = None):
        self.projects_dir = projects_dir or Path(settings.projects_dir)
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    def project_path(self, project_id: str) -> Path:
        return self.projects_dir / project_id

    def metadata_path(self, project_id: str) -> Path:
        return self.project_path(project_id) / "project.json"

    def create_project(self, name: str, target_type: str = "unknown") -> dict[str, Any]:
        project_id = str(uuid.uuid4())
        path = self.project_path(project_id)
        for child in ["capturepack", "video", "frames", "quality", "colmap", "training", "preview", "export", "logs"]:
            (path / child).mkdir(parents=True, exist_ok=True)
        project = {
            "id": project_id,
            "name": name.strip() or "Untitled Capture",
            "target_type": target_type,
            "status": "Created",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "path": str(path),
            "last_step": "Project created",
        }
        self.save_project(project)
        return project

    def save_project(self, project: dict[str, Any]) -> None:
        project["updated_at"] = utc_now()
        path = self.metadata_path(project["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(project, indent=2), encoding="utf-8")

    def update_project(self, project_id: str, **updates: Any) -> dict[str, Any]:
        project = self.get_project(project_id)
        project.update(updates)
        self.save_project(project)
        return project

    def get_project(self, project_id: str) -> dict[str, Any]:
        path = self.metadata_path(project_id)
        if not path.exists():
            raise FileNotFoundError(f"Project not found: {project_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def list_projects(self) -> list[dict[str, Any]]:
        projects: list[dict[str, Any]] = []
        for item in sorted(self.projects_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            meta = item / "project.json"
            if meta.exists():
                projects.append(json.loads(meta.read_text(encoding="utf-8")))
        return projects

    def delete_project(self, project_id: str) -> None:
        path = self.project_path(project_id)
        if not path.exists():
            raise FileNotFoundError(f"Project not found: {project_id}")
        shutil.rmtree(path)


project_manager = ProjectManager()

