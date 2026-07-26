from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from gausscapture.errors import CaptureFormatError, PipelineStateError

#: Model containers we can ingest. ``.ksplat`` is read for backwards
#: compatibility but is not an export target: it is tied to a viewer whose
#: author has retired it and now points users elsewhere. New work should use
#: ``.spz`` or ``.sog``, which are an order of magnitude smaller than ``.ply``
#: and actively maintained.
MODEL_SUFFIXES = (".ply", ".splat", ".spz", ".sog", ".ksplat")


def import_training_result(project_path: Path, source: Path) -> dict[str, Any]:
    """Bring a trained model into the project's preview directory.

    Accepts a bare model file or a zip from a Colab or cloud run.
    """
    preview_dir = project_path / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        raise CaptureFormatError(f"Training result does not exist: {source}")

    if source.is_file() and source.suffix.lower() == ".zip":
        result_dir = project_path / "training" / "imported_result"
        if result_dir.exists():
            shutil.rmtree(result_dir)
        result_dir.mkdir(parents=True)
        with zipfile.ZipFile(source, "r") as archive:
            archive.extractall(result_dir)
        candidates = sorted(
            p for p in result_dir.rglob("*") if p.suffix.lower() in MODEL_SUFFIXES
        )
    elif source.is_file() and source.suffix.lower() in MODEL_SUFFIXES:
        candidates = [source]
    else:
        raise CaptureFormatError(
            f"Training result must be a .zip or one of {', '.join(MODEL_SUFFIXES)}"
        )

    if not candidates:
        raise CaptureFormatError("No supported model file found in the training result")

    # Prefer the largest model: a run directory often contains small
    # intermediate checkpoints alongside the final output.
    model = max(candidates, key=lambda p: p.stat().st_size)
    target = preview_dir / model.name
    shutil.copy2(model, target)

    config = {
        "model_file": target.name,
        "model_type": target.suffix.lower().lstrip("."),
        "size_bytes": target.stat().st_size,
        "scene_scale": 1.0,
        "background": "#111318",
    }
    (preview_dir / "preview_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return {"model": str(target), "config": config}


def preview_status(project_path: Path) -> dict[str, Any]:
    config_path = project_path / "preview" / "preview_config.json"
    if not config_path.exists():
        return {"ready": False, "message": "No preview model imported or built yet."}
    data = json.loads(config_path.read_text(encoding="utf-8"))
    model = project_path / "preview" / data["model_file"]
    return {
        "ready": model.exists(),
        "config": data,
        "model_url": f"/api/projects/{project_path.name}/preview/model",
    }


def build_preview_from_latest_training(project_path: Path) -> dict[str, Any]:
    candidates = [
        p for p in (project_path / "training").rglob("*") if p.suffix.lower() in MODEL_SUFFIXES
    ]
    if not candidates:
        raise PipelineStateError("No trained model found. Train or import a result first.")
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return import_training_result(project_path, latest)


def preview_model_path(project_path: Path) -> Path:
    """Resolve the configured preview model, raising if it is absent."""
    config_path = project_path / "preview" / "preview_config.json"
    if not config_path.exists():
        raise PipelineStateError("No preview model configured. Import a trained result first.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = project_path / "preview" / config["model_file"]
    if not model.exists():
        raise CaptureFormatError("The configured preview model file is missing.")
    return model
