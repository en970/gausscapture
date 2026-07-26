from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from gausscapture.errors import PipelineStateError
from gausscapture.progress import NullProgress, Progress
from gausscapture.recon.dataset import build_dataset

COLAB_README = """# GaussCapture Colab package

`dataset.zip` is laid out the way every 3D Gaussian Splatting trainer expects:

    images/
    sparse/0/{cameras,images,points3D}.bin

so it can be handed straight to a trainer with `-s <extracted dir>`.

Use a CUDA GPU runtime. TPU is not supported. We recommend gsplat
(https://github.com/nerfstudio-project/gsplat), which is Apache-2.0 licensed;
the original reference implementation is licensed for non-commercial research
only, so anything trained with it cannot be used commercially.
"""


def create_colab_package(
    project_path: Path,
    training_config: dict[str, Any] | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Zip a trainer-ready dataset for upload to Colab, Kaggle, or a rented GPU.

    The archive contains the conventional ``images/`` + ``sparse/0/`` layout
    rather than GaussCapture's internal directory names, so the receiving
    notebook needs no knowledge of this project.
    """
    progress = progress or NullProgress()

    dataset_dir = build_dataset(project_path, progress=progress)
    export_dir = project_path / "colab_export"
    export_dir.mkdir(parents=True, exist_ok=True)

    config_path = export_dir / "run_config.json"
    config_path.write_text(json.dumps(training_config or {}, indent=2), encoding="utf-8")
    readme_path = export_dir / "README_COLAB.md"
    readme_path.write_text(COLAB_README, encoding="utf-8")

    dataset_zip = export_dir / "dataset.zip"
    if dataset_zip.exists():
        dataset_zip.unlink()

    files = [p for p in sorted(dataset_dir.rglob("*")) if p.is_file()]
    if not files:
        raise PipelineStateError("Dataset is empty; nothing to package.")

    with zipfile.ZipFile(dataset_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for i, file in enumerate(files):
            archive.write(file, file.relative_to(dataset_dir))
            if i % 25 == 0:
                progress.update(
                    5 + int(85 * ((i + 1) / len(files))), f"Packed {i + 1}/{len(files)} files"
                )
        # Context that helps a human but that a trainer ignores.
        for extra, arcname in (
            (project_path / "capturepack" / "manifest.json", "gausscapture/manifest.json"),
            (project_path / "quality" / "quality_report.json", "gausscapture/quality_report.json"),
            (config_path, "gausscapture/run_config.json"),
            (readme_path, "README_COLAB.md"),
        ):
            if extra.exists():
                archive.write(extra, arcname)

    progress.update(100, "Colab package ready")
    return {
        "dataset_zip": str(dataset_zip),
        "size_bytes": dataset_zip.stat().st_size,
        "readme": str(readme_path),
        "run_config": str(config_path),
        "file_count": len(files),
    }
