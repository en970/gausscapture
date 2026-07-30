"""Packing a splat into the compact 32-byte-per-gaussian web format.

A trained ``.ply`` carries full spherical harmonics — 45 floats per gaussian
that describe how its colour changes with viewing angle. That is most of the
file size and none of what a first look needs, so the web format keeps only the
base colour. A 174 MB ply becomes about 22 MB, which a browser can fetch and
upload to the GPU without ceremony.

The layout is the one `antimatter15/splat
<https://github.com/antimatter15/splat>`_ established and other web viewers
adopted, so a file written here also opens in tools that have never heard of
this project:

===========  ======  ==========================================
field        bytes   encoding
===========  ======  ==========================================
position     12      float32 x, y, z
scale        12      float32, already exponentiated
colour       4       uint8 RGBA, alpha carries opacity
rotation     4       uint8, quaternion mapped from [-1,1]
===========  ======  ==========================================

Gaussians are written **largest-contribution first**. A viewer that has to give
up partway — because the file is still downloading, or the device cannot hold
all of it — then keeps the ones that matter, instead of an arbitrary prefix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from gausscapture.errors import CaptureFormatError
from gausscapture.export.splat_ply import GaussianSplat, read_splat_ply

#: Bytes per gaussian in the packed format.
STRIDE = 32


def align_splat(splat: GaussianSplat, up: np.ndarray) -> GaussianSplat:
    """Rotate a splat so ``up`` becomes +Y.

    Structure-from-motion picks its world frame arbitrarily, and COLMAP's is the
    vision convention with +y pointing *down*. Handing that to a viewer that
    assumes +y up puts the scene upside down -- which does not look like a bug,
    it looks like a bad reconstruction. Normalising the frame here means every
    consumer downstream can simply assume +y is up.

    Orientations are rotated along with positions. Moving only the centres would
    leave every ellipse pointing the way it did before, which reads as a smeared
    reconstruction rather than as an obvious error.
    """
    from gausscapture.pose.orientation import (
        matrix_to_quaternion,
        quaternion_multiply,
        rotation_to_y_up,
    )

    rotation = rotation_to_y_up(up)
    positions = splat.positions @ rotation.T

    rotations = splat.rotations
    if rotations is not None:
        rotations = quaternion_multiply(matrix_to_quaternion(rotation), rotations)

    return GaussianSplat(
        positions=positions.astype(np.float32),
        colors=splat.colors,
        opacities=splat.opacities,
        scales=splat.scales,
        rotations=rotations,
        sh_bands=splat.sh_bands,
    )


def pack_splat(splat: GaussianSplat) -> np.ndarray:
    """Pack a splat into the 32-byte-per-gaussian buffer."""
    if splat.rotations is None:
        raise CaptureFormatError(
            "This splat has no rotations, so its gaussians have no orientation "
            "and can only be drawn as points."
        )

    count = len(splat)
    if count == 0:
        return np.zeros(0, dtype=np.uint8)

    # Importance is roughly how much screen a gaussian covers times how opaque
    # it is. Ordering by it means a truncated read degrades gracefully.
    volume = np.prod(splat.scales, axis=1)
    importance = volume * splat.opacities
    order = np.argsort(-importance)

    positions = splat.positions[order].astype(np.float32)
    scales = splat.scales[order].astype(np.float32)
    colors = splat.colors[order].astype(np.uint8)
    opacities = splat.opacities[order]
    rotations = splat.rotations[order]

    buffer = np.zeros((count, STRIDE), dtype=np.uint8)
    buffer[:, 0:12] = positions.view(np.uint8).reshape(count, 12)
    buffer[:, 12:24] = scales.view(np.uint8).reshape(count, 12)
    buffer[:, 24:27] = colors
    buffer[:, 27] = np.clip(opacities * 255, 0, 255).astype(np.uint8)
    # Quaternions live in [-1, 1]; 128 is zero and the scale is 128 per unit.
    buffer[:, 28:32] = np.clip(rotations * 128 + 128, 0, 255).astype(np.uint8)
    return buffer.reshape(-1)


def write_splat_binary(
    ply_path: Path,
    out_path: Path,
    min_opacity: float = 0.02,
    max_gaussians: int | None = None,
    colmap_model: Path | None = None,
) -> dict[str, Any]:
    """Convert a trained ``.ply`` to the packed web format.

    ``max_gaussians`` truncates after the importance sort, so a cap keeps the
    most significant gaussians rather than whichever happened to come first.

    ``colmap_model`` points at the sparse reconstruction the splat was trained
    from. It is what lets the up direction be recovered -- from the capture's
    accelerometer where one exists, from the camera poses otherwise -- and the
    splat is rotated so that direction becomes +Y. Without it the scene keeps
    structure-from-motion's arbitrary frame, which on real captures here has
    been more than 140 degrees off vertical.
    """
    ply_path = Path(ply_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    splat = read_splat_ply(ply_path).visible(min_opacity)

    aligned_from = None
    if colmap_model is not None:
        from gausscapture.pose.orientation import world_up

        up = world_up(Path(colmap_model))
        if up is not None:
            splat = align_splat(splat, up)
            aligned_from = [float(v) for v in up]
    buffer = pack_splat(splat)
    count = len(buffer) // STRIDE

    if max_gaussians is not None and count > max_gaussians:
        buffer = buffer[: max_gaussians * STRIDE]
        count = max_gaussians

    out_path.write_bytes(buffer.tobytes())

    positions = splat.positions[:count] if count else splat.positions
    low = np.percentile(positions, 2, axis=0) if count else np.zeros(3)
    high = np.percentile(positions, 98, axis=0) if count else np.zeros(3)
    centre = (low + high) / 2

    return {
        "file": out_path.name,
        "count": int(count),
        "aligned_up": aligned_from,
        "bytes": int(out_path.stat().st_size),
        "centre": [float(v) for v in centre],
        "radius": float(np.linalg.norm(high - low)) / 2 or 1.0,
        "median_scale": float(np.median(splat.scales)) if len(splat) else 0.0,
        "mean_opacity": float(splat.opacities.mean()) if len(splat) else 0.0,
        "sh_bands": splat.sh_bands,
    }
