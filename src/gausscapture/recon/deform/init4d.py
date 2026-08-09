"""The laptop's half of the dataset handoff: ``scene4d_init.npz`` and its zip.

Everything here is numpy, json and zipfile, and that is a requirement rather
than an observation. ``recon/prepare4d.py`` runs COLMAP, measures the dynamic
region, measures the cone and then *writes this file*, on a machine the whole
architecture assumes has no torch -- an M2 Air. When :class:`SceneInit` lived
beside the trainer it dragged ``import torch`` in with it through
``deform/bundle.py``, so preparation did all of its work and then died at the
write step with a bare ``ModuleNotFoundError``, which ``cli.main`` does not
catch and therefore reached the user as a traceback.

So the schema and the archive live here, torch-free, and
:mod:`gausscapture.recon.deform.bundle` -- which does need torch, because it
materialises tensors -- imports them from here and re-exports them under their
original names.

Layout of ``dataset4d.zip``::

    scene4d_init.npz   points, colours, dynamic labels, scaffold seeds,
                       the single fixed camera, and the frame times
    images/            the hold-phase frames, sorted, one per frame time
    masks/dynamic.png  optional; weights the photometric loss
    meta.json          provenance, and the cone measured during preparation
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gausscapture.errors import CaptureFormatError, PipelineStateError

INIT_NAME = "scene4d_init.npz"
IMAGES_DIR = "images"
MASK_PATH = "masks/dynamic.png"
META_NAME = "meta.json"

REQUIRED_ARRAYS = (
    "points",
    "point_colors",
    "point_dynamic",
    "node_rest",
    "viewmat",
    "intrinsics",
    "frame_times",
)


@dataclass
class SceneInit:
    """Everything the trainer needs that is not a pixel."""

    points: np.ndarray
    point_colors: np.ndarray
    point_dynamic: np.ndarray
    node_rest: np.ndarray
    viewmat: np.ndarray
    intrinsics: np.ndarray
    frame_times: np.ndarray
    meta: dict[str, Any]

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            points=self.points.astype(np.float32),
            point_colors=self.point_colors.astype(np.float32),
            point_dynamic=self.point_dynamic.astype(bool),
            node_rest=self.node_rest.astype(np.float32),
            viewmat=self.viewmat.astype(np.float32),
            intrinsics=self.intrinsics.astype(np.float32),
            frame_times=self.frame_times.astype(np.float32),
            meta=json.dumps(self.meta),
        )
        return path

    @classmethod
    def load(cls, path: Path) -> SceneInit:
        path = Path(path)
        if not path.exists():
            raise PipelineStateError(f"no scene initialisation at {path}; run prep4d first")
        with np.load(path, allow_pickle=False) as data:
            missing = [k for k in REQUIRED_ARRAYS if k not in data]
            if missing:
                raise CaptureFormatError(
                    f"{path.name} is missing required arrays: {', '.join(missing)}"
                )
            meta_raw = str(data["meta"]) if "meta" in data else "{}"
            return cls(
                points=np.asarray(data["points"], dtype=np.float32),
                point_colors=np.asarray(data["point_colors"], dtype=np.float32),
                point_dynamic=np.asarray(data["point_dynamic"], dtype=bool),
                node_rest=np.asarray(data["node_rest"], dtype=np.float32),
                viewmat=np.asarray(data["viewmat"], dtype=np.float32),
                intrinsics=np.asarray(data["intrinsics"], dtype=np.float32),
                frame_times=np.asarray(data["frame_times"], dtype=np.float32),
                meta=json.loads(meta_raw or "{}"),
            )


def pack_dataset(directory: Path, out_zip: Path) -> Path:
    """Zip a prepared dataset directory, refusing one that is incomplete."""
    directory = Path(directory)
    out_zip = Path(out_zip)
    if not (directory / INIT_NAME).exists():
        raise PipelineStateError(f"{directory} has no {INIT_NAME}; run prep4d first")
    if not any((directory / IMAGES_DIR).glob("*")):
        raise PipelineStateError(f"{directory / IMAGES_DIR} is empty")
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(directory).as_posix())
    return out_zip


def unpack_dataset(zip_path: Path, dest: Path) -> Path:
    """Extract a ``dataset4d.zip``, rejecting any path that escapes ``dest``.

    A zip is an untrusted input the moment it has been round-tripped through a
    cloud drive, and ``..`` in a member name is the oldest trick there is.
    """
    zip_path = Path(zip_path)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            target = (dest / member).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise CaptureFormatError(f"refusing zip member outside the destination: {member}")
        zf.extractall(dest)
    if not (dest / INIT_NAME).exists():
        raise CaptureFormatError(f"{zip_path.name} contains no {INIT_NAME}")
    return dest


def write_meta(directory: Path, meta: dict[str, Any]) -> Path:
    path = Path(directory) / META_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
