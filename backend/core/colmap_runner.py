from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from backend.config import settings
from backend.core.job_manager import Job


def colmap_available() -> bool:
    configured = Path(settings.colmap_path)
    return configured.exists() or shutil.which(settings.colmap_path) is not None


def run_colmap(job: Job, project_path: Path, matcher: str = "sequential") -> dict[str, Any]:
    if not colmap_available():
        raise RuntimeError("COLMAP not found. Install COLMAP or configure colmap_path in settings.json.")
    images = project_path / "frames" / "images"
    if not images.exists() or not list(images.glob("*.jpg")):
        raise RuntimeError("No extracted frames found. Run frame extraction first.")

    colmap_dir = project_path / "colmap"
    sparse_dir = colmap_dir / "sparse"
    database = colmap_dir / "database.db"
    colmap_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    commands = [
        [settings.colmap_path, "feature_extractor", "--database_path", str(database), "--image_path", str(images)],
        [settings.colmap_path, f"{matcher}_matcher", "--database_path", str(database)],
        [settings.colmap_path, "mapper", "--database_path", str(database), "--image_path", str(images), "--output_path", str(sparse_dir)],
    ]
    steps = ["Extracting features", "Matching features", "Mapping sparse reconstruction"]
    for i, (step, cmd) in enumerate(zip(steps, commands)):
        job.set_progress(5 + i * 30, step)
        _run_logged(job, cmd, cwd=project_path)

    model_dirs = [p for p in sparse_dir.iterdir() if p.is_dir()]
    images_total = len(list(images.glob("*.jpg")))
    images_registered = _registered_images(model_dirs[0]) if model_dirs else 0
    points = _sparse_points(model_dirs[0]) if model_dirs else 0
    ratio = images_registered / images_total if images_total else 0
    status = "good" if ratio >= 0.75 else "warning" if ratio >= 0.35 else "bad"
    report = {
        "method": "colmap",
        "images_total": images_total,
        "images_registered": images_registered,
        "registered_ratio": ratio,
        "sparse_points": points,
        "status": status,
        "warnings": [] if status == "good" else ["COLMAP registered too few images for reliable training."],
    }
    (colmap_dir / "pose_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"pose_report": report}


def _run_logged(job: Job, cmd: list[str], cwd: Path) -> None:
    job.log("$ " + " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        job.log(line.rstrip())
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Command failed with exit code {code}: {' '.join(cmd)}")


def _registered_images(model_dir: Path) -> int:
    txt = model_dir / "images.txt"
    if not txt.exists():
        return 0
    count = 0
    for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line and not line.startswith("#") and ".jpg" in line.lower():
            count += 1
    return count


def _sparse_points(model_dir: Path) -> int:
    txt = model_dir / "points3D.txt"
    if not txt.exists():
        return 0
    return len([line for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines() if line and not line.startswith("#")])

