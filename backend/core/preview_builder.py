from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


MODEL_EXTENSIONS = {".ply", ".splat", ".ksplat"}


def import_training_result(project_path: Path, source: Path) -> dict[str, Any]:
    preview_dir = project_path / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    if source.is_file() and source.suffix.lower() == ".zip":
        import zipfile

        result_dir = project_path / "training" / "imported_result"
        if result_dir.exists():
            shutil.rmtree(result_dir)
        result_dir.mkdir(parents=True)
        with zipfile.ZipFile(source, "r") as archive:
            archive.extractall(result_dir)
        candidates = [p for p in result_dir.rglob("*") if p.suffix.lower() in MODEL_EXTENSIONS]
    elif source.is_file() and source.suffix.lower() in MODEL_EXTENSIONS:
        candidates = [source]
    else:
        raise ValueError("Training result must be a .zip, .ply, .splat, or .ksplat file")
    if not candidates:
        raise FileNotFoundError("No supported model file found in training result")
    model = candidates[0]
    target = preview_dir / model.name
    shutil.copy2(model, target)
    copied.append(target)
    config = {
        "model_file": target.name,
        "model_type": target.suffix.lower().lstrip("."),
        "scene_scale": 1.0,
        "background": "#111318",
    }
    (preview_dir / "preview_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return {"model": str(target), "config": config, "copied": [str(p) for p in copied]}


def preview_status(project_path: Path) -> dict[str, Any]:
    config = project_path / "preview" / "preview_config.json"
    if not config.exists():
        return {"ready": False, "message": "No preview model imported or built yet."}
    data = json.loads(config.read_text(encoding="utf-8"))
    model = project_path / "preview" / data["model_file"]
    return {"ready": model.exists(), "config": data, "model_url": f"/api/projects/{project_path.name}/preview/model"}


def build_preview_from_latest_training(project_path: Path) -> dict[str, Any]:
    candidates = list((project_path / "training").rglob("*.ply")) + list((project_path / "training").rglob("*.splat")) + list((project_path / "training").rglob("*.ksplat"))
    if not candidates:
        raise FileNotFoundError("No trained model found. Import a training result first.")
    return import_training_result(project_path, candidates[-1])

