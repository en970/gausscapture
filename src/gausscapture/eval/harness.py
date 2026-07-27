"""Unattended batch runs.

One capture in, one row out. The row carries every capture-time signal
alongside what the pipeline actually managed to do with it, which is the
pairing the research question needs and which no existing dataset provides.

Design constraints that come from the science rather than from convenience:

* **Failures are data.** A capture that COLMAP cannot register is not an
  excluded sample -- it is the outcome we most want to predict. It is recorded
  with ``registered = False`` and the run continues.
* **Settings are pinned per batch and recorded in the output**, so a results
  file is interpretable a year later without archaeology.
* **Every stage is timed**, because "does this predict wasted compute" is a
  question a user cares about even when reconstruction quality is unmeasurable.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from gausscapture import __version__
from gausscapture.errors import GaussCaptureError
from gausscapture.ingest.frames import FRAME_PRESETS, extract_frames
from gausscapture.ingest.video import copy_video_into_pack, is_video_file, probe_video
from gausscapture.pack import manifest as pack_manifest
from gausscapture.pose.colmap import colmap_available, run_colmap
from gausscapture.progress import NullProgress, Progress
from gausscapture.project import ProjectStore
from gausscapture.telemetry import analyze_capture

#: The capture-time signals recorded for every run. Named explicitly rather
#: than dumped wholesale so that adding a signal is a deliberate act and the
#: results schema stays stable across a study.
SIGNAL_FIELDS = (
    "vol_mean",
    "vol_p10",
    "vol_median",
    "blurry_frame_ratio",
    "brightness_mean",
    "overexposed_ratio",
    "underexposed_ratio",
    "too_dark_ratio",
    "too_bright_ratio",
    "motion_mean",
    "too_fast_ratio",
    "duplicate_ratio",
    "score",
)


@dataclass
class CaptureRecord:
    """One capture's telemetry paired with what the pipeline achieved."""

    name: str
    project_id: str = ""
    source: str = ""
    label: str = ""

    # capture-time signals
    signals: dict[str, float] = field(default_factory=dict)
    frames_used: int = 0
    frames_sampled: int = 0
    frames_skipped_blur: int = 0
    frames_skipped_duplicate: int = 0
    duration_sec: float | None = None
    resolution: str = ""

    # outcomes
    registered: bool = False
    registered_ratio: float = 0.0
    images_registered: int = 0
    sparse_points: int = 0
    pose_status: str = "not_run"

    # cost
    seconds_telemetry: float = 0.0
    seconds_frames: float = 0.0
    seconds_pose: float = 0.0

    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # Flatten signals so the row is a flat record in CSV and JSONL alike.
        data.pop("signals")
        data.update({f"signal_{k}": v for k, v in self.signals.items()})
        return data


def run_one(
    source: Path,
    store: ProjectStore,
    label: str = "",
    frame_preset: str = "balanced",
    matcher: str = "sequential",
    skip_pose: bool = False,
    progress: Progress | None = None,
) -> CaptureRecord:
    """Take one video or pack through telemetry, frames, and pose estimation.

    Never raises for a capture-level failure: an unregisterable capture is the
    outcome under study, so it is recorded rather than propagated.
    """
    progress = progress or NullProgress()
    record = CaptureRecord(name=source.stem, source=str(source), label=label)

    try:
        project = store.create(source.stem, target_type=label or "unknown")
        record.project_id = project.id

        if is_video_file(source):
            for name in pack_manifest.REQUIRED_DIRS:
                (project.capturepack_dir / name).mkdir(parents=True, exist_ok=True)
            video_path = copy_video_into_pack(source, project.path)
            info = probe_video(video_path)
            pack_manifest.write_manifest(
                project.capturepack_dir,
                pack_manifest.create_minimal_manifest(
                    f"video/{video_path.name}", info, project.name, label or "unknown"
                ),
            )
        else:
            from gausscapture.pack.archive import import_archive

            result = import_archive(project.path, source)
            if not result["valid"]:
                record.error = "; ".join(result["errors"])[:300]
                return record

        started = time.perf_counter()
        report = analyze_capture(project.path, progress=progress)
        record.seconds_telemetry = time.perf_counter() - started
        record.signals = {name: float(getattr(report, name)) for name in SIGNAL_FIELDS}
        record.duration_sec = report.video.duration_sec
        record.resolution = f"{report.video.width}x{report.video.height}"

        started = time.perf_counter()
        index = extract_frames(project.path, dict(FRAME_PRESETS[frame_preset]), progress=progress)
        record.seconds_frames = time.perf_counter() - started
        record.frames_used = index.frames_used
        record.frames_sampled = index.frames_total_sampled
        record.frames_skipped_blur = index.frames_skipped_blur
        record.frames_skipped_duplicate = index.frames_skipped_duplicate

        if skip_pose:
            record.pose_status = "skipped"
            return record
        if not colmap_available():
            record.pose_status = "colmap_missing"
            return record

        started = time.perf_counter()
        pose = run_colmap(project.path, matcher=matcher, progress=progress)
        record.seconds_pose = time.perf_counter() - started
        record.pose_status = pose.status
        record.registered_ratio = pose.registered_ratio
        record.images_registered = pose.images_registered
        record.sparse_points = pose.sparse_points
        # "Registered" means a reconstruction a trainer could actually use, not
        # merely that COLMAP produced some output.
        record.registered = pose.status in {"good", "warning"}

    except GaussCaptureError as exc:
        record.error = str(exc)[:300]
        record.pose_status = "error"
    except Exception as exc:  # noqa: BLE001 - a crashed capture is still a data point
        record.error = f"{type(exc).__name__}: {exc}"[:300]
        record.pose_status = "error"

    return record


def run_batch(
    sources: list[Path],
    out_dir: Path,
    labels: dict[str, str] | None = None,
    frame_preset: str = "balanced",
    matcher: str = "sequential",
    skip_pose: bool = False,
    progress: Progress | None = None,
) -> list[CaptureRecord]:
    """Run every source and write ``results.jsonl`` plus ``run_config.json``.

    Results are appended after each capture rather than at the end, so a batch
    interrupted after six hours still yields six hours of data.
    """
    progress = progress or NullProgress()
    out_dir.mkdir(parents=True, exist_ok=True)
    store = ProjectStore(out_dir / "projects")

    config = {
        "gausscapture_version": __version__,
        "frame_preset": frame_preset,
        "frame_settings": FRAME_PRESETS[frame_preset],
        "matcher": matcher,
        "skip_pose": skip_pose,
        "colmap_available": colmap_available(),
        "platform": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "sources": [str(s) for s in sources],
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    results_path = out_dir / "results.jsonl"
    results_path.write_text("", encoding="utf-8")

    records: list[CaptureRecord] = []
    for i, source in enumerate(sources):
        progress.update(
            int(100 * i / max(1, len(sources))), f"[{i + 1}/{len(sources)}] {source.name}"
        )
        record = run_one(
            source,
            store,
            label=(labels or {}).get(source.name, ""),
            frame_preset=frame_preset,
            matcher=matcher,
            skip_pose=skip_pose,
            progress=progress,
        )
        records.append(record)
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict()) + "\n")

    progress.update(100, f"{len(records)} captures processed")
    return records


def load_results(path: Path) -> list[dict[str, Any]]:
    """Read a ``results.jsonl`` written by :func:`run_batch`."""
    if not path.exists():
        raise FileNotFoundError(f"No results file at {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_csv(records: list[CaptureRecord] | list[dict[str, Any]], path: Path) -> Path:
    """Write results as CSV, for people who would rather use a spreadsheet."""
    import csv

    rows = [r.to_dict() if isinstance(r, CaptureRecord) else r for r in records]
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    columns = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path
