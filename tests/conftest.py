"""Shared fixtures.

Tests synthesise their own video so the suite needs no fixture binaries, no
network, and no GPU -- it runs on a standard GitHub runner in seconds, which is
the gate the CI smoke test has to clear.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from gausscapture.pack import manifest
from gausscapture.project import ProjectStore


@pytest.fixture()
def projects_dir(tmp_path: Path, monkeypatch) -> Path:
    """Isolate every test from the user's real projects directory."""
    directory = tmp_path / "projects"
    directory.mkdir()
    monkeypatch.setenv("GAUSSCAPTURE_PROJECTS_DIR", str(directory))
    monkeypatch.setenv("GAUSSCAPTURE_SETTINGS_PATH", str(tmp_path / "settings.json"))
    return directory


@pytest.fixture()
def store(projects_dir: Path) -> ProjectStore:
    return ProjectStore(projects_dir)


def make_video(
    path: Path,
    frames: int = 40,
    size: tuple[int, int] = (160, 120),
    fps: int = 10,
    blur_from: int | None = None,
) -> Path:
    """Write a small synthetic video with moving texture.

    Texture matters: a flat colour has a variance-of-Laplacian near zero and
    would make sharpness tests meaningless. ``blur_from`` softens every frame
    from that index onward, which is how the blur-rejection tests create a
    capture that genuinely degrades partway through.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    rng = np.random.default_rng(1234)
    texture = rng.integers(0, 255, size=(height, width * 2, 3), dtype=np.uint8)

    for i in range(frames):
        offset = (i * 3) % width
        frame = texture[:, offset : offset + width].copy()
        if blur_from is not None and i >= blur_from:
            frame = cv2.GaussianBlur(frame, (9, 9), 6)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture()
def project_with_video(store: ProjectStore, tmp_path: Path):
    """A project containing a valid minimal CapturePack."""
    from gausscapture.ingest.video import copy_video_into_pack, probe_video

    project = store.create("test capture", "object")
    source = make_video(tmp_path / "source.mp4")

    for name in manifest.REQUIRED_DIRS:
        (project.capturepack_dir / name).mkdir(parents=True, exist_ok=True)
    video_path = copy_video_into_pack(source, project.path)
    info = probe_video(video_path)
    manifest.write_manifest(
        project.capturepack_dir,
        manifest.create_minimal_manifest(
            video_relpath=f"video/{video_path.name}",
            info=info,
            session_name=project.name,
            target_type="object",
        ),
    )
    return project
