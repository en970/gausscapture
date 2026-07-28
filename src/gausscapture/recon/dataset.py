"""Assemble a dataset directory in the layout every 3DGS trainer expects.

This module exists because of a real bug. The pipeline used to hand a trainer
the *project* root with ``-s``, while keeping images in ``frames/images`` and
COLMAP output in ``colmap/sparse``. Every trainer in this ecosystem -- the
Inria reference implementation, gsplat's ``simple_trainer``, Brush, Postshot --
expects::

    <source>/
      images/
      sparse/0/{cameras,images,points3D}.{bin,txt}

so nothing could ever load. Rather than pass more flags, we materialise the
conventional layout once and point the trainer at that.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from gausscapture.errors import PipelineStateError
from gausscapture.progress import NullProgress, Progress

SPARSE_FILES = ("cameras", "images", "points3D")


def build_dataset(
    project_path: Path,
    dataset_dir: Path | None = None,
    link: bool = True,
    progress: Progress | None = None,
) -> Path:
    """Create ``<dataset>/images`` and ``<dataset>/sparse/0`` for a project.

    ``link`` hardlinks images instead of copying them. A 600-frame capture at
    1920px is roughly 300 MB; hardlinking keeps a project from doubling in size
    for what is really a second view of the same files. It falls back to
    copying when the filesystem refuses, which happens across volumes.
    """
    progress = progress or NullProgress()
    dataset_dir = dataset_dir or (project_path / "dataset")

    images_src = project_path / "frames" / "images"
    if not images_src.exists() or not any(images_src.glob("*.jpg")):
        raise PipelineStateError("No extracted frames found. Run frame extraction first.")

    sparse_src = _find_sparse_model(project_path)
    if sparse_src is None:
        raise PipelineStateError(
            "No COLMAP sparse model found. Run pose estimation before building a dataset."
        )

    images_dst = dataset_dir / "images"
    sparse_dst = dataset_dir / "sparse" / "0"
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    images_dst.mkdir(parents=True, exist_ok=True)
    sparse_dst.mkdir(parents=True, exist_ok=True)

    frames = sorted(images_src.glob("*.jpg"))
    for i, frame in enumerate(frames):
        _place(frame, images_dst / frame.name, link=link)
        if i % 25 == 0:
            progress.update(int(80 * (i + 1) / len(frames)), f"Staged {i + 1}/{len(frames)} images")

    copied = 0
    for stem in SPARSE_FILES:
        for suffix in (".bin", ".txt"):
            candidate = sparse_src / f"{stem}{suffix}"
            if candidate.exists():
                shutil.copy2(candidate, sparse_dst / candidate.name)
                copied += 1
    if copied == 0:
        raise PipelineStateError(f"COLMAP model at {sparse_src} contains no cameras/images/points3D")

    # A transforms.json alongside the COLMAP model widens what can read this
    # directory: gsplat's examples take the COLMAP layout, while nerfstudio,
    # Brush and most research code take transforms.json. Writing both costs a
    # few kilobytes and removes a conversion step either way.
    _write_transforms(sparse_dst, dataset_dir, {frame.name for frame in frames}, progress)

    progress.update(100, f"Dataset ready at {dataset_dir}")
    return dataset_dir


def _write_transforms(
    model_dir: Path, dataset_dir: Path, image_names: set[str], progress: Progress
) -> None:
    """Emit transforms.json, treating failure as non-fatal.

    The COLMAP layout is the primary contract; transforms.json is a
    convenience. An unreadable model should surface when a trainer reads it,
    not by aborting dataset assembly.
    """
    from gausscapture.errors import GaussCaptureError
    from gausscapture.pack.transforms import write_transforms

    try:
        write_transforms(
            model_dir, dataset_dir / "transforms.json", applies_to=image_names
        )
    except (GaussCaptureError, OSError, ValueError) as exc:
        progress.log(f"Could not write transforms.json: {exc}")


def _find_sparse_model(project_path: Path) -> Path | None:
    """Locate the sparse model, tolerating both ``sparse/`` and ``sparse/0/``."""
    sparse_root = project_path / "colmap" / "sparse"
    if not sparse_root.exists():
        return None
    if any((sparse_root / f"{s}.bin").exists() or (sparse_root / f"{s}.txt").exists() for s in SPARSE_FILES):
        return sparse_root
    candidates = [
        p
        for p in sorted(sparse_root.iterdir())
        if p.is_dir()
        and any((p / f"{s}.bin").exists() or (p / f"{s}.txt").exists() for s in SPARSE_FILES)
    ]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    # COLMAP numbers disconnected models in reconstruction order, not by size,
    # so sparse/0 is frequently a small fragment. Train on the largest.
    from gausscapture.errors import CaptureFormatError
    from gausscapture.pose.model import read_model

    def registered(path: Path) -> int:
        try:
            return len(read_model(path).images)
        except CaptureFormatError:
            return 0

    return max(candidates, key=registered)


def _place(src: Path, dst: Path, link: bool) -> None:
    if dst.exists():
        dst.unlink()
    if link:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass  # Cross-device or unsupported; fall through to a copy.
    shutil.copy2(src, dst)
