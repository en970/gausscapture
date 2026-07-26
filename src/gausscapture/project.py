"""Projects: a directory with a known layout and a JSON metadata file.

A project is deliberately just a directory. Anything the pipeline produces can
be inspected, diffed, deleted, or fed to another tool without going through
this package -- which is a requirement for the benchmark work, where 180
reconstructions need to be processed by scripts that were never written against
our API.
"""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gausscapture.config import Settings, get_settings
from gausscapture.util.log import utc_now

STATUS_CREATED = "Created"
STATUS_IMPORTED = "Imported"
STATUS_PREPROCESSED = "Preprocessed"
STATUS_READY = "Ready for Training"
STATUS_TRAINING = "Training"
STATUS_TRAINED = "Trained"
STATUS_PREVIEW = "Preview Ready"
STATUS_EXPORTED = "Exported"
STATUS_ERROR = "Error"

#: Subdirectories created for every project.
SUBDIRS = (
    "capturepack",
    "frames",
    "quality",
    "colmap",
    "dataset",
    "training",
    "preview",
    "export",
    "logs",
)


@dataclass
class Project:
    """Metadata for one capture project."""

    id: str
    name: str
    path: Path
    target_type: str = "unknown"
    status: str = STATUS_CREATED
    created_at: str = ""
    updated_at: str = ""
    last_step: str | None = None

    # --- convenient derived paths, so callers never hardcode layout ---

    @property
    def capturepack_dir(self) -> Path:
        return self.path / "capturepack"

    @property
    def frames_dir(self) -> Path:
        return self.path / "frames" / "images"

    @property
    def frame_index_path(self) -> Path:
        return self.path / "frames" / "frame_index.json"

    @property
    def quality_report_path(self) -> Path:
        return self.path / "quality" / "quality_report.json"

    @property
    def colmap_dir(self) -> Path:
        return self.path / "colmap"

    @property
    def dataset_dir(self) -> Path:
        """Trainer-convention dataset root: ``images/`` plus ``sparse/0/``."""
        return self.path / "dataset"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "target_type": self.target_type,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "path": str(self.path),
            "last_step": self.last_step,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Project:
        return cls(
            id=data["id"],
            name=data.get("name", "Untitled Capture"),
            path=Path(data["path"]),
            target_type=data.get("target_type", "unknown"),
            status=data.get("status", STATUS_CREATED),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            last_step=data.get("last_step"),
        )


class ProjectStore:
    """Filesystem-backed collection of projects."""

    def __init__(self, projects_dir: Path | str | None = None, settings: Settings | None = None):
        if projects_dir is None:
            projects_dir = (settings or get_settings()).projects_dir
        self.projects_dir = Path(projects_dir).expanduser()
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, project_id: str) -> Path:
        return self.projects_dir / project_id

    def _metadata_path(self, project_id: str) -> Path:
        return self.path_for(project_id) / "project.json"

    def create(self, name: str, target_type: str = "unknown") -> Project:
        project_id = str(uuid.uuid4())
        path = self.path_for(project_id)
        for child in SUBDIRS:
            (path / child).mkdir(parents=True, exist_ok=True)
        now = utc_now()
        project = Project(
            id=project_id,
            name=name.strip() or "Untitled Capture",
            path=path,
            target_type=target_type,
            status=STATUS_CREATED,
            created_at=now,
            updated_at=now,
            last_step="Project created",
        )
        self.save(project)
        return project

    def save(self, project: Project) -> None:
        project.updated_at = utc_now()
        path = self._metadata_path(project.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(project.to_dict(), indent=2), encoding="utf-8")

    def update(self, project_id: str, **updates: Any) -> Project:
        project = self.get(project_id)
        for key, value in updates.items():
            setattr(project, key, value)
        self.save(project)
        return project

    def get(self, project_id: str) -> Project:
        path = self._metadata_path(project_id)
        if not path.exists():
            raise FileNotFoundError(f"Project not found: {project_id}")
        return Project.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list(self) -> list[Project]:
        projects: list[Project] = []
        if not self.projects_dir.exists():
            return projects
        entries = [p for p in self.projects_dir.iterdir() if (p / "project.json").exists()]
        for item in sorted(entries, key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads((item / "project.json").read_text(encoding="utf-8"))
                projects.append(Project.from_dict(data))
            except (json.JSONDecodeError, KeyError):
                continue  # Skip unreadable projects rather than failing the listing.
        return projects

    def delete(self, project_id: str) -> None:
        path = self.path_for(project_id)
        if not path.exists():
            raise FileNotFoundError(f"Project not found: {project_id}")
        shutil.rmtree(path)
