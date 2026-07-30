"""Working out which way is up, by measurement wherever possible.

Structure-from-motion has no idea what "up" means. COLMAP fixes the world frame
on whichever image it happened to start from, and its camera convention is the
computer-vision one — **+x right, +y down, +z forward** — so a scene handed
straight to an OpenGL-style viewer arrives upside down. Heights read wrong,
orbiting feels inverted, and nothing about it is obviously a bug: it just looks
like a badly behaved scene.

The right answer comes from the accelerometer, and :mod:`gausscapture.pose.gravity`
gets it. What lives here is the arithmetic for applying an up direction, plus
the fallback used when a capture carries no IMU: averaging the camera
orientations, on the assumption the phone was held upright.

That fallback is genuinely poor and is kept only because some captures leave no
alternative. Measured against gravity on this project's three captures it is off
by 41.3°, 3.6° and 9.1°, and it cannot tell which case it is in. Prefer
:func:`world_up`, which uses it only as a last resort.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def world_up(model_dir: Path, project_dir: Path | None = None) -> np.ndarray | None:
    """The world up direction for a reconstruction, best source first.

    Gravity is a measurement and everything else is an assumption, so the
    accelerometer wins whenever the capture recorded one and its per-frame
    estimates agree. ``project_dir`` is inferred from ``model_dir`` when not
    given, by walking up to the directory that holds the capture pack.
    """
    from gausscapture.pose.gravity import up_from_gravity

    model_dir = Path(model_dir)
    project_dir = project_dir or _find_project(model_dir)

    if project_dir is not None:
        measured = up_from_gravity(project_dir, model_dir)
        if measured is not None:
            return measured.up
    return up_from_cameras(model_dir)


def _find_project(model_dir: Path) -> Path | None:
    """Walk up from a sparse model to the project directory that contains it.

    A model normally sits at ``<project>/colmap/sparse/<n>``, but the search is
    written against what a project *has* rather than against that depth, so a
    caller with a differently arranged tree still gets an answer.
    """
    for parent in list(model_dir.resolve().parents)[:5]:
        if (parent / "capturepack").is_dir() and (parent / "frames").is_dir():
            return parent
    return None


def up_from_cameras(model_dir: Path) -> np.ndarray | None:
    """Estimate the world up direction from a COLMAP model's camera poses.

    Returns a unit vector, or ``None`` when the model cannot be read or the
    cameras disagree so much that no direction is meaningful.
    """
    from gausscapture.errors import CaptureFormatError
    from gausscapture.pose.model import read_model

    try:
        model = read_model(Path(model_dir))
    except CaptureFormatError:
        return None
    if not model.images:
        return None

    # A camera's up in world coordinates is the negative of its second
    # camera-to-world column, because +y points down in the vision convention.
    ups = np.array([-image.camera_to_world()[:3, 1] for image in model.images.values()])
    mean = ups.mean(axis=0)
    length = float(np.linalg.norm(mean))
    if length < 1e-6:
        return None

    direction = mean / length
    # `length` also measures agreement: near 1 when the cameras share an up,
    # near 0 when they point every which way. A capture that tumbled gives no
    # usable answer, and guessing one would be worse than declining.
    if length < 0.35:
        return None
    return direction.astype(np.float32)


def rotation_to_y_up(up: np.ndarray) -> np.ndarray:
    """Rotation taking ``up`` onto +Y, by the shortest arc.

    The shortest arc matters: any rotation that maps the vector would do for
    the vertical, but adding an arbitrary spin about it would leave the scene
    upright and facing a random direction.
    """
    up = np.asarray(up, dtype=np.float64)
    up = up / (np.linalg.norm(up) or 1.0)
    target = np.array([0.0, 1.0, 0.0])

    axis = np.cross(up, target)
    sin = float(np.linalg.norm(axis))
    cos = float(np.dot(up, target))

    if sin < 1e-8:
        # Already aligned, or exactly inverted -- in which case any axis
        # perpendicular to up will do for the half turn.
        if cos > 0:
            return np.eye(3, dtype=np.float32)
        perpendicular = np.array([1.0, 0.0, 0.0])
        if abs(up[0]) > 0.9:
            perpendicular = np.array([0.0, 0.0, 1.0])
        axis = np.cross(up, perpendicular)
        axis /= np.linalg.norm(axis)
        return _axis_angle(axis, np.pi).astype(np.float32)

    axis = axis / sin
    angle = float(np.arctan2(sin, cos))
    return _axis_angle(axis, angle).astype(np.float32)


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation formula."""
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a (w, x, y, z) quaternion."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q = np.array([0.25 / s, (m[2, 1] - m[1, 2]) * s,
                      (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                      (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                      0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                      (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return (q / np.linalg.norm(q)).astype(np.float32)


def quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, broadcasting a single quaternion over an array.

    Rotating a scene means rotating each gaussian's *orientation* as well as
    its position. Moving the centres alone leaves every ellipse pointing the
    way it did before, which reads as a subtly smeared reconstruction rather
    than as an obvious bug.
    """
    a = np.atleast_2d(np.asarray(a, dtype=np.float32))
    b = np.atleast_2d(np.asarray(b, dtype=np.float32))
    aw, ax, ay, az = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bw, bx, by, bz = b[:, 0:1], b[:, 1:2], b[:, 2:3], b[:, 3:4]
    return np.concatenate([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=1).astype(np.float32)
