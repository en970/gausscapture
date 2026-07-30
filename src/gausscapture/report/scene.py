"""Exporting a COLMAP reconstruction for the browser viewer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from gausscapture.pose.model import read_model, read_points
from gausscapture.pose.orientation import rotation_to_y_up, world_up

#: Percentile used to frame the scene.
#:
#: Sparse reconstructions routinely contain a handful of points triangulated
#: kilometres away from everything else. Framing on the true extent puts the
#: actual scene in a few pixels at the centre, so the camera is fitted to the
#: bulk instead. The outliers are still rendered -- they are real output, and
#: hiding them would misrepresent the reconstruction -- just not framed on.
FRAMING_PERCENTILE = 2.0


def export_scene(model_dir: Path, out_dir: Path, name: str = "scene") -> dict[str, Any]:
    """Write ``<name>.bin`` and return the metadata the viewer needs.

    Positions are float32 and colours uint8, concatenated into one buffer so
    the page issues a single fetch. At a few tens of thousands of points that
    is a few hundred kilobytes -- small enough not to bother compressing, large
    enough that JSON would be wasteful.
    """
    model_dir = Path(model_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    positions, colors = read_points(model_dir)
    model = read_model(model_dir)

    if len(positions) == 0:
        return {"name": name, "points": 0, "cameras": len(model.images)}

    # Structure-from-motion's world frame is arbitrary, and COLMAP's convention
    # puts +y *down*. Leaving it alone hands the viewer a scene tipped by
    # whatever angle the first camera happened to sit at -- on real captures
    # here, more than 140 degrees -- which makes every height read wrong.
    up = world_up(model_dir)
    align = rotation_to_y_up(up) if up is not None else np.eye(3, dtype=np.float32)
    positions = (positions @ align.T).astype(np.float32)

    buffer = out_dir / f"{name}.bin"
    with buffer.open("wb") as handle:
        handle.write(positions.astype(np.float32).tobytes())
        handle.write(colors.astype(np.uint8).tobytes())

    low = np.percentile(positions, FRAMING_PERCENTILE, axis=0)
    high = np.percentile(positions, 100 - FRAMING_PERCENTILE, axis=0)
    centre = (low + high) / 2
    radius = float(np.linalg.norm(high - low)) / 2 or 1.0

    # Camera-to-world matrices in COLMAP's own convention: the viewer draws
    # frustums looking down +z, which is where this puts them.
    cameras = []
    for image in model.sorted_images():
        matrix = image.camera_to_world()
        # The frustums rotate with the points; moving only the cloud would
        # leave the cameras floating at the old orientation.
        position = align @ matrix[:3, 3]
        rotation = align @ matrix[:3, :3]
        cameras.append(
            {
                "name": Path(image.name).name,
                "position": [float(v) for v in position],
                "rotation": [float(v) for v in rotation.flatten()],
            }
        )

    intrinsics = None
    if model.cameras:
        camera = next(iter(model.cameras.values()))
        intrinsics = {
            "fx": camera.fx,
            "fy": camera.fy,
            "cx": camera.cx,
            "cy": camera.cy,
            "width": camera.width,
            "height": camera.height,
            "model": camera.model,
        }

    return {
        "name": name,
        "buffer": buffer.name,
        "points": int(len(positions)),
        "cameras": cameras,
        "centre": [float(v) for v in centre],
        "radius": radius,
        "intrinsics": intrinsics,
        "aligned_up": None if up is None else [float(v) for v in up],
    }


def export_splat(ply_path: Path, out_dir: Path, name: str) -> dict[str, Any]:
    """Write a trained splat into the viewer's buffer format.

    The viewer draws gaussian centres as points rather than as gaussians. That
    is a deliberate limit, not an oversight: proper splat rendering needs
    per-frame depth sorting and projected covariances, and SuperSplat and Spark
    already do it well (``docs/RESEARCH.md`` section 7.1). What this view is
    good for is comparison -- the same scene, sparse input beside dense output,
    in one camera.

    Near-transparent gaussians are dropped. Training leaves a long tail the
    renderer never shows, and keeping them would make the result look noisier
    than it is.
    """
    from gausscapture.export.splat_ply import read_splat_ply

    ply_path = Path(ply_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splat = read_splat_ply(ply_path).visible()
    if len(splat) == 0:
        return {"name": name, "points": 0}

    positions = splat.positions.astype(np.float32)
    buffer = out_dir / f"{name}.bin"
    with buffer.open("wb") as handle:
        handle.write(positions.tobytes())
        handle.write(splat.colors.astype(np.uint8).tobytes())

    low = np.percentile(positions, FRAMING_PERCENTILE, axis=0)
    high = np.percentile(positions, 100 - FRAMING_PERCENTILE, axis=0)
    centre = (low + high) / 2
    radius = float(np.linalg.norm(high - low)) / 2 or 1.0

    return {
        "name": name,
        "buffer": buffer.name,
        "points": int(len(splat)),
        "cameras": [],
        "centre": [float(v) for v in centre],
        "radius": radius,
        "kind": "splat",
        "source": ply_path.name,
        "median_scale": float(np.median(splat.scales)),
        "mean_opacity": float(splat.opacities.mean()),
    }


def write_manifest(scenes: list[dict[str, Any]], out_dir: Path) -> Path:
    path = Path(out_dir) / "scenes.json"
    path.write_text(json.dumps(scenes, indent=2), encoding="utf-8")
    return path
