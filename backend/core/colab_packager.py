from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from backend.core.job_manager import Job


def create_colab_package(job: Job, project_path: Path, training_config: dict[str, Any]) -> dict[str, Any]:
    images = project_path / "frames" / "images"
    if not images.exists() or not list(images.glob("*.jpg")):
        raise RuntimeError("No extracted frames found. Create frames before building a Colab package.")
    export_dir = project_path / "colab_export"
    export_dir.mkdir(parents=True, exist_ok=True)
    config_path = export_dir / "run_config.json"
    config_path.write_text(json.dumps(training_config, indent=2), encoding="utf-8")
    readme_path = export_dir / "README_COLAB.md"
    readme_path.write_text(
        "# GaussCapture Colab Package\n\nUpload `dataset.zip` in the notebook, use a CUDA GPU runtime, and run the cells. TPU is not supported.\n",
        encoding="utf-8",
    )
    dataset_zip = export_dir / "dataset.zip"
    if dataset_zip.exists():
        dataset_zip.unlink()
    with zipfile.ZipFile(dataset_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        include_roots = [
            project_path / "frames" / "images",
            project_path / "colmap" / "sparse",
            project_path / "capturepack" / "manifest.json",
            project_path / "quality" / "quality_report.json",
            config_path,
        ]
        files = []
        for root in include_roots:
            if root.is_file():
                files.append(root)
            elif root.exists():
                files.extend([p for p in root.rglob("*") if p.is_file()])
        for i, file in enumerate(files):
            archive.write(file, file.relative_to(project_path))
            job.set_progress(5 + int(85 * ((i + 1) / max(1, len(files)))), f"Packed {file.name}")
    return {"dataset_zip": str(dataset_zip), "readme": str(readme_path), "run_config": str(config_path)}

