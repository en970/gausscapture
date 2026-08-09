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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gausscapture.config import Settings, get_settings
from gausscapture.errors import PipelineStateError
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

# The fixed-camera 4D path runs alongside the static one rather than replacing
# it: the same capture can yield a static splat and a bullet-time scene, so its
# stages get their own labels instead of overloading "Trained".
STATUS_MASKED = "Dynamic Region Measured"
STATUS_PREPARED_4D = "Prepared (4D)"
STATUS_PACKAGED_4D = "Packaged for GPU (4D)"
STATUS_TRAINED_4D = "Trained (4D)"
STATUS_EXPORTED_4D = "Exported (4D)"

#: Subdirectories created for every project.
SUBDIRS = (
    "capturepack",
    "frames",
    # Dynamic-region masks. They are inputs to COLMAP and to the canonical
    # dynamic/static split, not products of either, so they get their own home
    # rather than living inside colmap/ or frames/.
    "masks",
    "quality",
    "colmap",
    "dataset",
    "training",
    "preview",
    "export",
    "logs",
)

# --------------------------------------------------------------------------
# 4D layout
#
# Written down once, here, because five modules and the CLI all need to agree
# on where a 4D project keeps its artefacts, and a layout that is spelled out
# at each call site is a layout that drifts.
# --------------------------------------------------------------------------

MASKS_SUBDIR = "masks"
SCENE4D_INIT_RELPATH = Path("training") / "scene4d_init.npz"
DATASET4D_RELPATH = Path("colab_export") / "dataset4d.zip"
SCENE4D_RELPATH = Path("training") / "scene4d.npz"
SCENE4D_G4D_RELPATH = Path("export") / "scene.g4d"
VIEWER4D_RELPATH = Path("export") / "viewer4d"
FOURD_STATE_RELPATH = Path("training") / "fourd.json"

#: The order 4D artefacts appear in. :meth:`FourDState.require` compares
#: positions in this tuple, so it is the single written-down shape of the path.
FOURD_STAGES = (
    "none",
    "masked",
    "prepared",
    "packaged",
    "trained",
    "exported",
    "viewable",
)

#: What to run next from each stage. Reported by `project show` so a stalled
#: project answers "what now" without anyone reading the roadmap.
FOURD_NEXT_COMMAND = {
    "none": "prep4d",
    "masked": "prep4d",
    "prepared": "colab4d",
    "packaged": "train4d",
    "trained": "export4d",
    "exported": "viewer4d",
    "viewable": None,
}


@dataclass
class FourDState:
    """What a 4D project has actually produced, read off the disk.

    The stage is derived from the artefacts rather than from a status string
    that a previous run wrote. That is deliberate: a project whose training
    output was deleted to free space, or that was copied without its export
    directory, has to report what it can do *next*, and a recorded status would
    confidently say otherwise. The notes each step leaves -- the measured cone,
    which tier the fixed pose came from -- annotate the stage and never decide
    it.
    """

    project_path: Path
    stage: str = "none"
    notes: dict[str, Any] = field(default_factory=dict)

    # --- artefact locations, so no caller hardcodes the layout ---

    @property
    def masks_dir(self) -> Path:
        return self.project_path / MASKS_SUBDIR

    @property
    def init_path(self) -> Path:
        return self.project_path / SCENE4D_INIT_RELPATH

    @property
    def dataset_zip(self) -> Path:
        return self.project_path / DATASET4D_RELPATH

    @property
    def scene_path(self) -> Path:
        return self.project_path / SCENE4D_RELPATH

    @property
    def g4d_path(self) -> Path:
        return self.project_path / SCENE4D_G4D_RELPATH

    @property
    def viewer_dir(self) -> Path:
        return self.project_path / VIEWER4D_RELPATH

    @property
    def is_fourd(self) -> bool:
        return self.stage != "none"

    @classmethod
    def observe(cls, project_path: Path) -> FourDState:
        project_path = Path(project_path)
        state = cls(project_path=project_path)
        state.notes = _read_fourd_notes(project_path)
        # Descending, so the stage is the furthest point actually reached.
        if (state.viewer_dir / "index.html").exists():
            state.stage = "viewable"
        elif state.g4d_path.exists():
            state.stage = "exported"
        elif state.scene_path.exists():
            state.stage = "trained"
        elif state.dataset_zip.exists():
            state.stage = "packaged"
        elif state.init_path.exists():
            state.stage = "prepared"
        elif state.masks_dir.is_dir() and any(state.masks_dir.glob("*.png")):
            state.stage = "masked"
        return state

    def require(self, stage: str, hint: str) -> None:
        """Refuse to run a step whose inputs do not exist yet.

        ``hint`` names the command that produces them, because "not ready" on
        its own is the least useful thing a pipeline can say.
        """
        if stage not in FOURD_STAGES:
            raise ValueError(f"Unknown 4D stage: {stage}")
        if FOURD_STAGES.index(self.stage) < FOURD_STAGES.index(stage):
            raise PipelineStateError(
                f"This project has reached the '{self.stage}' stage of the 4D path, "
                f"which is short of '{stage}'. {hint}"
            )

    def to_dict(self) -> dict[str, Any]:
        def present(path: Path) -> str | None:
            return str(path) if path.exists() else None

        # masks/ is created for every project, so its mere existence says
        # nothing; only a directory with masks in it is an artefact.
        has_masks = self.masks_dir.is_dir() and any(self.masks_dir.glob("*.png"))
        return {
            "stage": self.stage,
            "next_command": FOURD_NEXT_COMMAND[self.stage],
            "masks_dir": str(self.masks_dir) if has_masks else None,
            "init": present(self.init_path),
            "dataset_zip": present(self.dataset_zip),
            "scene": present(self.scene_path),
            "g4d": present(self.g4d_path),
            "viewer": present(self.viewer_dir / "index.html"),
            "notes": self.notes,
        }


def _read_fourd_notes(project_path: Path) -> dict[str, Any]:
    path = Path(project_path) / FOURD_STATE_RELPATH
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # Unreadable notes are an annotation lost, not a project lost.
        return {}
    return data if isinstance(data, dict) else {}


def record_fourd_stage(project_path: Path, stage: str, **notes: Any) -> FourDState:
    """Append what a 4D step measured to the project's notes.

    The stage argument is recorded for a human reading the file, never read
    back by :meth:`FourDState.observe` -- see that method's docstring.
    """
    if stage not in FOURD_STAGES:
        raise ValueError(f"Unknown 4D stage: {stage}")
    project_path = Path(project_path)
    merged = _read_fourd_notes(project_path)
    merged.update({k: v for k, v in notes.items() if v is not None})
    merged["stage_recorded"] = stage
    merged["updated_at"] = utc_now()
    path = project_path / FOURD_STATE_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2, default=str), encoding="utf-8")
    return FourDState.observe(project_path)


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

    # --- the fixed-camera 4D path ---

    @property
    def masks_dir(self) -> Path:
        return self.path / MASKS_SUBDIR

    @property
    def scene4d_init_path(self) -> Path:
        return self.path / SCENE4D_INIT_RELPATH

    @property
    def dataset4d_zip(self) -> Path:
        return self.path / DATASET4D_RELPATH

    @property
    def scene4d_path(self) -> Path:
        return self.path / SCENE4D_RELPATH

    @property
    def scene4d_g4d_path(self) -> Path:
        return self.path / SCENE4D_G4D_RELPATH

    @property
    def viewer4d_dir(self) -> Path:
        return self.path / VIEWER4D_RELPATH

    def fourd_state(self) -> FourDState:
        return FourDState.observe(self.path)

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
        # Resolved, never relative. Project paths are handed to subprocesses --
        # COLMAP, ffmpeg, a trainer -- that run with their own working
        # directory, so a relative path here resolves against the wrong root
        # and the tool reports a missing file that plainly exists.
        self.projects_dir = Path(projects_dir).expanduser().resolve()
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
