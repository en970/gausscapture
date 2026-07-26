from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from backend.config import settings
from backend.core.job_manager import Job


PRESETS = {
    "draft": {"iterations": 5000, "resolution": 1024},
    "balanced": {"iterations": 15000, "resolution": 1600},
    "high": {"iterations": 30000, "resolution": 2048},
}


def trainer_status() -> dict[str, Any]:
    trainer = Path(settings.gaussian_trainer_path).expanduser() if settings.gaussian_trainer_path else None
    return {
        "configured": bool(trainer),
        "exists": bool(trainer and trainer.exists()),
        "path": str(trainer) if trainer else "",
    }


def run_local_training(job: Job, project_path: Path, preset: str, scene_type: str) -> dict[str, Any]:
    status = trainer_status()
    if not status["exists"]:
        raise RuntimeError("Gaussian trainer path is not configured or does not exist.")
    images = project_path / "frames" / "images"
    sparse = project_path / "colmap" / "sparse"
    if not images.exists() or not list(images.glob("*.jpg")):
        raise RuntimeError("No extracted frames found. Run preprocessing first.")
    if not sparse.exists() or not any(sparse.iterdir()):
        raise RuntimeError("COLMAP sparse output is missing. Run COLMAP or import pose metadata first.")

    run_dir = _next_run_dir(project_path / "training")
    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "preset": preset,
        "scene_type": scene_type,
        "trainer_path": status["path"],
        "parameters": PRESETS.get(preset, PRESETS["draft"]),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    cmd = [
        settings.python_path if hasattr(settings, "python_path") else "python",
        "train.py",
        "-s",
        str(project_path),
        "-m",
        str(output_dir),
        "--iterations",
        str(config["parameters"]["iterations"]),
    ]
    if config["parameters"].get("resolution"):
        cmd.extend(["--resolution", str(config["parameters"]["resolution"])])
    job.set_progress(5, "Starting external Gaussian trainer")
    proc = subprocess.Popen(cmd, cwd=status["path"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    with (run_dir / "logs.txt").open("a", encoding="utf-8") as log_file:
        for line in proc.stdout:
            stripped = line.rstrip()
            log_file.write(stripped + "\n")
            job.log(stripped)
            _parse_progress(job, stripped, config["parameters"]["iterations"])
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Gaussian training failed with exit code {code}")
    models = list(output_dir.rglob("*.ply")) + list(output_dir.rglob("*.splat")) + list(output_dir.rglob("*.ksplat"))
    if not models:
        raise RuntimeError("Training completed but no .ply/.splat/.ksplat model output was found.")
    summary = {"status": "success", "run_dir": str(run_dir), "models": [str(p) for p in models]}
    (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _next_run_dir(training_dir: Path) -> Path:
    training_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in training_dir.glob("run_*") if p.is_dir()]
    idx = len(existing) + 1
    run_dir = training_dir / f"run_{idx:03d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _parse_progress(job: Job, line: str, iterations: int) -> None:
    for token in line.replace("/", " ").split():
        if token.isdigit():
            value = int(token)
            if 0 < value <= iterations:
                job.progress = max(job.progress, min(95, int(value / iterations * 95)))
                return

