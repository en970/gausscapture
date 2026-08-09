"""Everything between "a recording" and "a thing a GPU can start fitting".

:mod:`gausscapture.recon.dataset4d` gets as far as a directory of frames, masks,
one solved camera and a measured cone. That is not yet an initialisation: a
trainer still has to be told which Gaussians are allowed to move, where the
scaffold nodes sit, and what time each frame is. Deciding those on the GPU host
would put three decisions that need the mask, the sparse model and the fixed
pose on the machine that has none of them, so they are made here and written
into one file.

The file is ``training/scene4d_init.npz`` and its schema is
:class:`gausscapture.recon.deform.init4d.SceneInit`. It is deliberately small --
points, colours, a dynamic label, node rest positions, one camera, and the frame
times -- because everything else the trainer needs is either recoverable from
those (the skinning is a k-nearest-neighbour query, recomputed at every refine
boundary anyway) or is a pixel and belongs in ``images/``.

Two decisions here are measurements rather than parameters, and both would be
guesses anywhere else in the pipeline:

**Which Gaussians may move.** A sparse point is dynamic when it reprojects,
through the one solved fixed pose, into the dynamic mask -- the region the
stationary camera *measured* as moving. No appearance model, no heuristic about
what a head looks like. Points that land on the background keep the label
``static`` and are passed through the deformation untouched (decision D11).

**Where the scaffold goes.** Farthest-point sampling over the dynamic points
only. Seeding over the whole cloud would spend most of a 256-node budget on a
wall that never moves, and a random subset leaves holes exactly where the point
density is lowest, which on a face is the cheek and the neck.

This module imports no torch, and neither does anything it imports. Preparation
runs on the laptop; the trainer's dependencies belong to the trainer. That is
enforced by ``tests/test_prep4d_torchless.py``, which blocks ``torch`` at the
import system and runs the whole preparation, because the claim was previously
false in exactly one place -- the write step -- and therefore failed after
COLMAP, the mask and the cone measurement had already been paid for.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from gausscapture.errors import CaptureFormatError, PipelineStateError
from gausscapture.export.scene4d import MAX_NODES, MAX_SKIN_K
from gausscapture.pose.model import quaternion_to_rotation, read_points
from gausscapture.progress import NullProgress, Progress
from gausscapture.project import MASKS_SUBDIR, SCENE4D_INIT_RELPATH, record_fourd_stage
from gausscapture.recon.dataset4d import (
    Dataset4DResult,
    Solver,
    frame_times,
    prepare_4d_dataset,
    read_scene4d,
)
from gausscapture.recon.deform.init4d import SceneInit
from gausscapture.recon.dynamic_mask import read_mask

#: Where preparation leaves the hold frames for the packager. ``export/colab4d``
#: looks here first; keeping a copy outside ``dataset4d/`` means a project can be
#: packaged again after the working directory has been cleaned up.
FRAMES4D_SUBDIR = Path("training") / "frames4d"

#: The union mask, under the name the trainer's bundle loader expects. It
#: weights the photometric loss; it does not gate it -- see ``bundle.load_clip``.
TRAINER_MASK_NAME = "dynamic.png"

#: Fewest dynamic points a scaffold can be seeded from. Below this the "4D"
#: scene is a static scene with a few floaters attached, and saying so now is
#: cheaper than saying it after a GPU has been rented.
MIN_DYNAMIC_POINTS = 8


def prepare_4d(
    project_path: Path,
    nodes: int = MAX_NODES,
    skin_k: int = MAX_SKIN_K,
    cap_max: int = 400_000,
    solver: Solver | None = None,
    allow_inference: bool = True,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Run the whole CPU preparation and write the canonical initialisation.

    Returns the dictionary ``gausscapture prep4d`` prints: where the init went,
    how many Gaussians and nodes it holds, the measured cone, and which tier the
    fixed pose came from. Nothing in it is a target; every number is something
    that was measured on this take.

    ``allow_inference`` is passed through to the phase splitter. Left on, a
    CapturePack with no ``protocol`` block has its boundaries measured off the
    footage; turned off, such a pack is refused, which is what a caller wants
    when the point is to check that the capture app wrote a protocol at all.
    """
    progress = progress or NullProgress()
    project_path = Path(project_path).resolve()

    dataset = prepare_4d_dataset(
        project_path,
        solver=solver,
        allow_inference=allow_inference,
        progress=_Sub(progress, 0, 82),
    )

    progress.update(84, "Labelling the dynamic points")
    scene = read_scene4d(dataset.path)
    points, colors = read_points(dataset.path / "sparse" / "0")
    if points.shape[0] == 0:
        raise CaptureFormatError(
            f"The sparse model in {dataset.path / 'sparse' / '0'} has no 3D points, so there "
            "is nothing to initialise a splat from. Structure-from-motion solved cameras but "
            "no structure, which on this protocol means the background carried no texture."
        )

    union = read_mask(_union_mask_path(dataset))
    dynamic = label_dynamic_points(points, scene, union)

    progress.update(88, "Seeding the motion scaffold")
    node_rest = seed_scaffold(points[dynamic], nodes)

    progress.update(92, "Writing the canonical initialisation")
    times = frame_times(dataset.path)
    init_path = _write_init(
        project_path=project_path,
        dataset=dataset,
        scene=scene,
        points=points,
        colors=colors,
        dynamic=dynamic,
        node_rest=node_rest,
        times=times,
        skin_k=skin_k,
        cap_max=cap_max,
    )

    progress.update(96, "Staging the training frames")
    images = _stage_frames(project_path, dataset)
    _stage_trainer_mask(project_path, dataset)

    result: dict[str, Any] = {
        "init_path": str(init_path),
        "dataset4d": str(dataset.path),
        "images": images,
        "frames": int(times.size),
        "gaussians": int(points.shape[0]),
        "dynamic_gaussians": int(dynamic.sum()),
        "nodes": int(node_rest.shape[0]),
        "skin_k": int(max(1, min(int(skin_k), MAX_SKIN_K))),
        "cap_max": int(cap_max),
        "cone": dataset.cone.to_dict(),
        "fixed_pose": dataset.fixed_pose.to_dict(),
        "split": dataset.split.to_dict(),
        "mask": dataset.mask.to_dict(),
        "warnings": list(dataset.warnings),
    }
    record_fourd_stage(
        project_path,
        "prepared",
        init=str(init_path),
        dataset4d=str(dataset.path),
        gaussians=result["gaussians"],
        dynamic_gaussians=result["dynamic_gaussians"],
        nodes=result["nodes"],
        frames=result["frames"],
        cone=result["cone"],
        fixed_pose=result["fixed_pose"],
    )
    progress.update(100, f"{result['gaussians']} gaussians, {result['nodes']} scaffold nodes")
    return result


# --------------------------------------------------------------------------
# the two measurements
# --------------------------------------------------------------------------


def label_dynamic_points(
    points: np.ndarray, scene: dict[str, Any], mask: np.ndarray
) -> np.ndarray:
    """Flag the sparse points that reproject into the measured dynamic region.

    The projection uses the *fixed* pose, because that is the only pose the
    dynamic region was ever measured from: the mask is an image-space fact about
    one camera, and reprojecting it through an arc camera would be asking where
    the subject was in a frame the mask does not describe.

    Distortion is ignored deliberately. The mask is dilated by two percent of
    the frame height precisely because its boundary is not a sharp fact, and no
    phone lens moves a point by anything like that far.
    """
    array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    fixed = scene.get("fixed_pose") or {}
    camera = scene.get("camera") or {}
    try:
        rotation = quaternion_to_rotation(np.asarray(fixed["qvec"], dtype=np.float64))
        tvec = np.asarray(fixed["tvec"], dtype=np.float64).reshape(3)
        fx, fy = float(camera["fx"]), float(camera["fy"])
        cx, cy = float(camera["cx"]), float(camera["cy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CaptureFormatError(
            f"scene4d.json is missing the fixed pose or the intrinsics needed to decide which "
            f"points move: {exc}. Re-run the preparation step."
        ) from exc

    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape[:2]
    camera_space = array @ rotation.T + tvec
    depth = camera_space[:, 2]
    in_front = depth > 1e-6
    safe = np.where(in_front, depth, 1.0)
    u = fx * camera_space[:, 0] / safe + cx
    v = fy * camera_space[:, 1] / safe + cy

    inside = in_front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    col = np.clip(np.round(u), 0, width - 1).astype(int)
    row = np.clip(np.round(v), 0, height - 1).astype(int)
    return inside & mask[row, col]


def seed_scaffold(dynamic_points: np.ndarray, nodes: int) -> np.ndarray:
    """Farthest-point sample the scaffold's rest positions.

    Deterministic: the first seed is the point furthest from the centroid, not a
    random one, so two preparations of the same capture produce the same
    scaffold and a difference downstream is never this function's fault.
    """
    array = np.asarray(dynamic_points, dtype=np.float32).reshape(-1, 3)
    if array.shape[0] < MIN_DYNAMIC_POINTS:
        raise PipelineStateError(
            f"Only {array.shape[0]} sparse points fall inside the measured dynamic region, "
            f"under the {MIN_DYNAMIC_POINTS} a motion scaffold can be seeded from. Either the "
            "subject barely moved during the hold, or the sweep never triangulated them. "
            "There is no 4D scene in this take; the static pipeline still applies to it."
        )
    count = int(max(1, min(int(nodes), MAX_NODES, array.shape[0])))

    centroid = array.mean(axis=0)
    chosen = np.empty(count, dtype=np.int64)
    chosen[0] = int(np.argmax(((array - centroid) ** 2).sum(axis=1)))
    distance = ((array - array[chosen[0]]) ** 2).sum(axis=1)
    for i in range(1, count):
        chosen[i] = int(np.argmax(distance))
        distance = np.minimum(distance, ((array - array[chosen[i]]) ** 2).sum(axis=1))
    return np.ascontiguousarray(array[chosen], dtype=np.float32)


# --------------------------------------------------------------------------
# writing the initialisation
# --------------------------------------------------------------------------


def _write_init(
    project_path: Path,
    dataset: Dataset4DResult,
    scene: dict[str, Any],
    points: np.ndarray,
    colors: np.ndarray,
    dynamic: np.ndarray,
    node_rest: np.ndarray,
    times: np.ndarray,
    skin_k: int,
    cap_max: int,
) -> Path:
    camera = scene.get("camera") or {}
    _check_camera_matches_frames(camera, scene)
    intrinsics = np.array(
        [
            [float(camera["fx"]), 0.0, float(camera["cx"])],
            [0.0, float(camera["fy"]), float(camera["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    init = SceneInit(
        points=np.asarray(points, dtype=np.float32),
        # The trainer works in linear [0, 1] RGB at sh_degree 0, and COLMAP
        # stores RGB bytes; the divide is the whole conversion.
        point_colors=(np.asarray(colors, dtype=np.float32).reshape(-1, 3) / 255.0),
        point_dynamic=np.asarray(dynamic, dtype=bool),
        node_rest=node_rest,
        viewmat=_viewmat(scene),
        intrinsics=intrinsics,
        frame_times=np.asarray(times, dtype=np.float32),
        meta=_init_meta(dataset, scene, skin_k, cap_max),
    )
    init_path = project_path / SCENE4D_INIT_RELPATH
    init.save(init_path)
    return init_path


def _check_camera_matches_frames(camera: dict[str, Any], scene: dict[str, Any]) -> None:
    """Refuse intrinsics that describe a differently sized image than the frames.

    ``fx``, ``cx`` and the rest are copied straight into the initialisation and
    every later stage -- the dynamic labelling, the trainer, the exported cone --
    projects through them into pixels that came from ``images/``. If COLMAP
    solved a 160x120 camera and the hold frames are 120x90, every one of those
    projections is out by the ratio, the principal point is 17% of the frame
    width off, and nothing downstream raises: points land in the wrong part of
    the mask, the scaffold seeds off the subject, and the fit converges on the
    wrong geometry. The two numbers sit next to each other in ``scene4d.json``,
    so the check costs nothing.

    ``dataset4d`` already records this as a warning on the mask report. A
    warning is right there, where the dataset can still be salvaged by
    re-extracting; here the numbers are about to be frozen into the file the GPU
    host trains from, so it is a refusal.
    """
    declared = (int(camera.get("width") or 0), int(camera.get("height") or 0))
    frames = (int(scene.get("width") or 0), int(scene.get("height") or 0))
    if not all(declared) or not all(frames):
        return
    if declared != frames:
        raise CaptureFormatError(
            f"The solved camera is {declared[0]}x{declared[1]} but the hold frames are "
            f"{frames[0]}x{frames[1]}. The intrinsics would be written into the training "
            "initialisation describing an image size the pixels do not have, so every "
            "reprojection -- the dynamic labelling, the scaffold seeding, the fit itself -- "
            "would be wrong by that ratio. Re-run the preparation so the reconstruction and "
            "the training views are extracted at one size."
        )


def _viewmat(scene: dict[str, Any]) -> np.ndarray:
    """The one world-to-camera matrix, OpenCV convention, as the trainer wants it."""
    fixed = scene.get("fixed_pose") or {}
    rotation = quaternion_to_rotation(np.asarray(fixed["qvec"], dtype=np.float64))
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = rotation
    viewmat[:3, 3] = np.asarray(fixed["tvec"], dtype=np.float64).reshape(3)
    return viewmat


def _init_meta(
    dataset: Dataset4DResult, scene: dict[str, Any], skin_k: int, cap_max: int
) -> dict[str, Any]:
    """Provenance the trainer must carry into ``scene4d.npz`` unchanged.

    The cone, the capture pose and the verification tier are measured here and
    are not recoverable on the GPU host, so they travel with the data. Anything
    that reads them and does not put them back into the checkpoint has broken
    the viewer's ability to say what the scene is evidence for.
    """
    fixed = dataset.fixed_pose
    cone = dataset.cone
    rotation = np.asarray(fixed.rotation(), dtype=np.float64)
    return {
        "capture_type": scene.get("capture_type"),
        "session_id": scene.get("session_id"),
        "fps": float(scene.get("fps") or 0.0),
        "width": int(scene.get("width") or 0),
        "height": int(scene.get("height") or 0),
        "skin_k": int(max(1, min(int(skin_k), MAX_SKIN_K))),
        "cap_max": int(cap_max),
        "cone": cone.to_dict(),
        "fixed_pose": fixed.to_dict(),
        "capture": {
            # The viewer orbits about the subject from the pose the take was
            # shot from, so it wants a camera centre and a frame, not a
            # world-to-camera matrix.
            "eye": [float(v) for v in fixed.centre],
            # COLMAP's camera axes are x right, y down, z forward, so the rows
            # of R are the camera's world-space axes and "up" is -y.
            "forward": [float(v) for v in rotation[2, :]],
            "up": [float(-v) for v in rotation[1, :]],
            "verified": bool(fixed.verified),
        },
        "claims": scene.get("claims", {}),
        "time_base": scene.get("time_base", "normalised_hold_phase"),
    }


def _union_mask_path(dataset: Dataset4DResult) -> Path:
    path = Path(dataset.mask.union_path or (dataset.path / "masks" / "union.png"))
    if not path.exists():
        raise PipelineStateError(
            f"The dynamic-region union mask is missing at {path}, so there is no measurement "
            "of what moved. Re-run the preparation step."
        )
    return path


def _stage_frames(project_path: Path, dataset: Dataset4DResult) -> int:
    """Put the hold frames where the packager looks for them.

    Hardlinked rather than copied: the frames are the largest thing in a project
    and the two directories hold the same bytes by construction.
    """
    source = dataset.path / "images"
    destination = project_path / FRAMES4D_SUBDIR
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    staged = 0
    for item in sorted(source.iterdir()):
        if not item.is_file():
            continue
        target = destination / item.name
        try:
            os.link(item, target)
        except OSError:
            shutil.copy2(item, target)
        staged += 1
    return staged


def _stage_trainer_mask(project_path: Path, dataset: Dataset4DResult) -> Path:
    """Copy the union mask to the single name the trainer's loader reads."""
    destination = project_path / MASKS_SUBDIR / TRAINER_MASK_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_union_mask_path(dataset), destination)
    return destination


class _Sub:
    """Rescales a child step's 0-100 progress into a slice of the parent's."""

    def __init__(self, parent: Progress, start: int, end: int):
        self.parent = parent
        self.start = start
        self.span = max(0, end - start)

    def update(self, percent: int, message: str | None = None) -> None:
        self.parent.update(self.start + int(self.span * max(0, min(100, percent)) / 100), message)

    def log(self, message: str) -> None:
        self.parent.log(message)


def read_init_meta(init_path: Path) -> dict[str, Any]:
    """The provenance block from a ``scene4d_init.npz``, without loading points.

    ``np.load`` on an ``.npz`` reads only the archive directory until a member is
    touched, so this stays cheap on a file holding half a million points.
    """
    init_path = Path(init_path)
    if not init_path.exists():
        raise PipelineStateError(f"No scene initialisation at {init_path}; run prep4d first")
    with np.load(init_path, allow_pickle=False) as data:
        if "meta" not in data.files:
            return {}
        raw = np.asarray(data["meta"]).reshape(-1)[0]
    try:
        meta = json.loads(raw.item() if hasattr(raw, "item") else str(raw))
    except (TypeError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}
