"""The dataset handoff between the laptop and whatever GPU is rented that week.

Preparation runs on an M2 Air; training does not. So the boundary between them
has to be a single file that a Colab cell can download and open with nothing
installed but the ``train4d`` extra -- no repository checkout on the GPU host, no
COLMAP, no ffmpeg. That file is ``dataset4d.zip``, and
:mod:`gausscapture.recon.deform.init4d` is the only thing that writes or reads
it.

This module is the *tensor* half of that boundary: it turns the archive's arrays
into a :class:`CanonicalParams`, a :class:`ScaffoldField` and a
:class:`FixedCameraClip`. It therefore imports torch at module scope, which is
exactly why the schema and the zip do not live here -- ``recon/prepare4d.py``
writes ``scene4d_init.npz`` on a machine that has no torch at all, and importing
this module to do it was a ``ModuleNotFoundError`` in the last step of a
five-minute preparation run.

``SceneInit``, ``pack_dataset``, ``unpack_dataset`` and ``write_meta`` are
re-exported here under their original names so that "the bundle module" remains
one import for callers that do have torch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from gausscapture.errors import CaptureFormatError, PipelineStateError
from gausscapture.recon.deform.field import (
    ExplicitTrajectoryField,
    KPlanesField,
    ScaffoldField,
    scene_aabb,
)
from gausscapture.recon.deform.init4d import (
    IMAGES_DIR,
    INIT_NAME,
    MASK_PATH,
    META_NAME,
    REQUIRED_ARRAYS,
    SceneInit,
    pack_dataset,
    unpack_dataset,
    write_meta,
)
from gausscapture.recon.deform.params import CanonicalParams
from gausscapture.recon.deform.train4d import FixedCameraClip, scene_scale_from_points

__all__ = [
    "IMAGES_DIR",
    "INIT_NAME",
    "MASK_PATH",
    "META_NAME",
    "REQUIRED_ARRAYS",
    "SceneInit",
    "build_from_init",
    "images_to_tensor",
    "load_clip",
    "pack_dataset",
    "unpack_dataset",
    "write_meta",
]


def build_from_init(
    init: SceneInit,
    field_kind: str = "explicit",
    device: torch.device | str = "cpu",
    field_kwargs: dict[str, Any] | None = None,
) -> tuple[CanonicalParams, ScaffoldField]:
    """Materialise the canonical splat and the field from a prepared init.

    ``field_kind`` selects the motion parameterisation: ``"explicit"`` gives
    free per-node, per-keyframe transforms (the default, and what the container
    stores either way), ``"kplanes"`` gives the multi-resolution feature field
    that predicts them.
    """
    points = torch.from_numpy(init.points).to(device)
    colors = torch.from_numpy(init.point_colors).to(device)
    dynamic = torch.from_numpy(init.point_dynamic).float().to(device)
    node_rest = torch.from_numpy(init.node_rest).to(device)
    times = torch.from_numpy(init.frame_times).to(device)

    params = CanonicalParams.from_points(
        points, colors, dynamic=dynamic, scene_scale=scene_scale_from_points(points)
    )
    kwargs = dict(field_kwargs or {})
    if field_kind == "explicit":
        fld: ScaffoldField = ExplicitTrajectoryField(node_rest, times, **kwargs)
    elif field_kind == "kplanes":
        kwargs.setdefault("aabb", scene_aabb(points))
        fld = KPlanesField(node_rest, **kwargs)
    else:
        raise ValueError(f"unknown field kind {field_kind!r}; use 'explicit' or 'kplanes'")
    return params, fld.to(device)


def load_clip(directory: Path, device: torch.device | str = "cpu") -> FixedCameraClip:
    """Read the hold-phase frames and the fixed camera into a clip.

    Images are decoded with OpenCV, which the core already depends on, and
    converted to linear-ish ``[0, 1]`` RGB without a colour transform: the
    capture app locks the photometric pipeline for exactly this reason, so the
    frames are already consistent with each other and any transform we applied
    here would only be undone at export.
    """
    import cv2

    directory = Path(directory)
    init = SceneInit.load(directory / INIT_NAME)
    files = sorted((directory / IMAGES_DIR).glob("*"))
    files = [f for f in files if f.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    if not files:
        raise PipelineStateError(f"no frames in {directory / IMAGES_DIR}")
    if len(files) != init.frame_times.shape[0]:
        raise CaptureFormatError(
            f"{len(files)} frames but {init.frame_times.shape[0]} frame times"
        )

    frames = []
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            raise CaptureFormatError(f"could not decode {f}")
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    images = torch.from_numpy(np.stack(frames))

    weights = None
    mask_file = directory / MASK_PATH
    if mask_file.exists():
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise CaptureFormatError(f"could not decode {mask_file}")
        # The background is not ignored, only down-weighted: it is what anchors
        # the fixed pose, and a loss that sees none of it lets the whole scene
        # drift along the optical axis for free.
        weights = torch.from_numpy((0.1 + 0.9 * (mask.astype(np.float32) / 255.0))[..., None])

    return FixedCameraClip(
        images=images,
        viewmat=torch.from_numpy(init.viewmat),
        intrinsics=torch.from_numpy(init.intrinsics),
        times=torch.from_numpy(init.frame_times),
        weights=weights,
    ).to(device)


def images_to_tensor(images: np.ndarray) -> Tensor:
    """Convert ``(T, H, W, 3)`` uint8 or float frames to a float tensor in [0, 1]."""
    arr = np.asarray(images)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
