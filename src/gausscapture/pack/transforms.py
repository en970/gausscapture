"""Writing ``transforms.json``.

``transforms.json`` is the de-facto interchange format for posed image sets in
this ecosystem: nerfstudio, gsplat's examples, Brush, Postshot and most
research code will load one without conversion. Emitting it is what lets a
CapturePack be consumed by tools that have never heard of GaussCapture, which
matters more than any container we could invent (see ``docs/RESEARCH.md``
section 8).

The one subtlety is the axis convention. COLMAP stores world-to-camera poses in
a right-handed frame with **+x right, +y down, +z forward**. The NeRF/OpenGL
convention used by ``transforms.json`` is **+x right, +y up, +z back**. So the
camera-to-world matrix needs its second and third columns negated. Getting this
wrong produces a reconstruction that trains to a plausible loss and renders
upside down and inside out, which is why it is spelled out here rather than
left as a magic line.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from gausscapture.errors import PipelineStateError
from gausscapture.pose.model import Camera, SparseModel, read_model

#: Distortion parameter names that nerfstudio understands at the top level.
_DISTORTION_KEYS = ("k1", "k2", "k3", "k4", "p1", "p2")

#: Maps COLMAP camera models onto the three nerfstudio accepts.
_CAMERA_MODEL_MAP = {
    "SIMPLE_PINHOLE": "PINHOLE",
    "PINHOLE": "PINHOLE",
    "SIMPLE_RADIAL": "OPENCV",
    "RADIAL": "OPENCV",
    "OPENCV": "OPENCV",
    "FULL_OPENCV": "OPENCV",
    "OPENCV_FISHEYE": "OPENCV_FISHEYE",
    "SIMPLE_RADIAL_FISHEYE": "OPENCV_FISHEYE",
    "RADIAL_FISHEYE": "OPENCV_FISHEYE",
}

#: Negates the y and z axes of a camera-to-world matrix, converting COLMAP's
#: computer-vision convention to the OpenGL one ``transforms.json`` expects.
COLMAP_TO_OPENGL = np.diag([1.0, -1.0, -1.0, 1.0])


def colmap_to_opengl(camera_to_world: np.ndarray) -> np.ndarray:
    """Apply the axis flip to a 4x4 camera-to-world matrix."""
    return camera_to_world @ COLMAP_TO_OPENGL


def build_transforms(
    model: SparseModel,
    image_dir_name: str = "images",
    applies_to: set[str] | None = None,
) -> dict[str, Any]:
    """Build a ``transforms.json`` document from a COLMAP sparse model.

    ``applies_to`` optionally restricts output to a set of file names, which is
    how a pack whose frames were filtered after reconstruction stays
    self-consistent: a frame listed in ``transforms.json`` but absent from disk
    makes most loaders raise.
    """
    images = [
        image
        for image in model.sorted_images()
        if applies_to is None or Path(image.name).name in applies_to
    ]
    if not images:
        raise PipelineStateError(
            "No registered images remain after filtering; cannot write transforms.json"
        )

    cameras_used = {image.camera_id for image in images}
    document: dict[str, Any] = {}

    # With a single shared camera -- which is what `--ImageReader.single_camera`
    # gives us for phone video -- intrinsics live at the top level. Otherwise
    # they are repeated per frame, which every loader also accepts.
    shared_camera: Camera | None = None
    if len(cameras_used) == 1:
        shared_camera = model.cameras[next(iter(cameras_used))]
        document.update(_intrinsics(shared_camera))

    frames = []
    for image in images:
        camera = model.cameras.get(image.camera_id)
        frame: dict[str, Any] = {
            "file_path": f"{image_dir_name}/{Path(image.name).name}",
            "transform_matrix": colmap_to_opengl(image.camera_to_world()).tolist(),
            "colmap_im_id": image.id,
        }
        if shared_camera is None and camera is not None:
            frame.update(_intrinsics(camera))
        frames.append(frame)

    document["frames"] = frames
    document["ply_file_path"] = "sparse_points.ply" if model.points else None
    if document["ply_file_path"] is None:
        document.pop("ply_file_path")
    return document


def _intrinsics(camera: Camera) -> dict[str, Any]:
    values: dict[str, Any] = {
        "camera_model": _CAMERA_MODEL_MAP.get(camera.model, "OPENCV"),
        "fl_x": camera.fx,
        "fl_y": camera.fy,
        "cx": camera.cx,
        "cy": camera.cy,
        "w": camera.width,
        "h": camera.height,
    }
    distortion = camera.distortion()
    for key in _DISTORTION_KEYS:
        values[key] = float(distortion.get(key, 0.0))
    return values


def write_transforms(
    model_dir: Path,
    out_path: Path,
    image_dir_name: str = "images",
    applies_to: set[str] | None = None,
) -> Path:
    """Read a COLMAP model and write ``transforms.json`` beside the images."""
    document = build_transforms(
        read_model(model_dir), image_dir_name=image_dir_name, applies_to=applies_to
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return out_path


def validate_transforms(path: Path, image_root: Path | None = None) -> dict[str, Any]:
    """Check a ``transforms.json`` for the mistakes that break loaders quietly.

    Returns a result dict in the same shape as pack validation, so the CLI can
    present both the same way.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not path.exists():
        return {"valid": False, "errors": [f"{path.name} is missing"], "warnings": [], "frames": 0}

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"valid": False, "errors": [f"Not valid JSON: {exc}"], "warnings": [], "frames": 0}

    frames = document.get("frames") or []
    if not frames:
        errors.append("No frames listed")

    has_top_level_intrinsics = "fl_x" in document
    for i, frame in enumerate(frames):
        if "file_path" not in frame:
            errors.append(f"Frame {i} has no file_path")
        matrix = frame.get("transform_matrix")
        if not matrix or len(matrix) != 4 or any(len(row) != 4 for row in matrix):
            errors.append(f"Frame {i} has no valid 4x4 transform_matrix")
        elif matrix[3] != [0.0, 0.0, 0.0, 1.0]:
            warnings.append(f"Frame {i}'s transform_matrix has a non-standard bottom row")
        if not has_top_level_intrinsics and "fl_x" not in frame:
            errors.append(f"Frame {i} has no intrinsics and none are defined globally")

    if image_root is not None:
        missing = [
            frame["file_path"]
            for frame in frames
            if "file_path" in frame and not (image_root / frame["file_path"]).exists()
        ]
        if missing:
            errors.append(
                f"{len(missing)} referenced image(s) are missing, starting with {missing[0]}"
            )

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "frames": len(frames),
        "camera_model": document.get("camera_model"),
    }
