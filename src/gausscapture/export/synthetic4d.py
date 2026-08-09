"""A synthetic 4D scene, so the container and the viewer can be finished first.

No trained ``.g4d`` exists yet and none will until a phone, a COLMAP run and a
rented GPU have all worked at once. Waiting for that before writing the viewer
would mean debugging the browser, the format and the trainer simultaneously,
on the day the first real capture arrives.

So this module builds a scene with the shape of the real deliverable -- a
dynamic head in front of a static background, skinned to a small scaffold --
and with a motion whose answer is known in closed form. The head turns about a
fixed pivot at the neck, with the rotation ramped from zero at the neck to full
at the crown, so node ``j`` at frame ``f`` is *exactly*

    ``p' = R(yaw_f * ramp_j) (p - pivot) + pivot``

and a test can assert against that rather than against a screenshot. The
translation each node needs to express that rotation about a shared pivot is
``(R_j - I)(n_j - pivot)``, which follows from the deformation contract's
"rotate about the node's own rest position" and is the one piece of arithmetic
here worth stating.

The motion is also *obvious to the eye*: a head that turns left and right
through 24 degrees while the wall behind it does not move at all. If the viewer
draws the wall moving, or the head shearing, the mistake is visible in the
first second without any reference to compare against.
"""

from __future__ import annotations

import numpy as np

from gausscapture.export.scene4d import MAX_NODES, MAX_SKIN_K, Scene4D

#: Where the head pivots. A real neck, not the scene origin.
PIVOT = np.array([0.0, -0.06, 0.0], dtype=np.float64)


def _rotation_y(angle: np.ndarray) -> np.ndarray:
    """Rotation about +Y for an array of angles, shaped ``(..., 3, 3)``."""
    c, s = np.cos(angle), np.sin(angle)
    zero, one = np.zeros_like(c), np.ones_like(c)
    return np.stack([
        np.stack([c, zero, s], axis=-1),
        np.stack([zero, one, zero], axis=-1),
        np.stack([-s, zero, c], axis=-1),
    ], axis=-2)


def _matrix_to_quaternion(matrices: np.ndarray) -> np.ndarray:
    """``(..., 3, 3)`` rotations to ``(..., 4)`` ``(w, x, y, z)`` quaternions.

    Only ever fed proper rotations built above, so the trace branch that
    general converters need for 180-degree cases cannot be reached here.
    """
    m = np.asarray(matrices, dtype=np.float64)
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    w = np.sqrt(np.maximum(trace + 1.0, 1e-12)) / 2.0
    return np.stack([
        w,
        (m[..., 2, 1] - m[..., 1, 2]) / (4 * w),
        (m[..., 0, 2] - m[..., 2, 0]) / (4 * w),
        (m[..., 1, 0] - m[..., 0, 1]) / (4 * w),
    ], axis=-1)


def _skin(points: np.ndarray, nodes: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Bind each point to its ``k`` nearest nodes with inverse-distance weights.

    Chunked because the real scene has 150,000 dynamic gaussians and 256 nodes,
    and the full distance matrix would be 300 MB for no reason.
    """
    count = len(points)
    indices = np.zeros((count, k), dtype=np.int32)
    weights = np.zeros((count, k), dtype=np.float32)

    for start in range(0, count, 32768):
        stop = min(start + 32768, count)
        block = points[start:stop]
        distance = np.linalg.norm(block[:, None, :] - nodes[None, :, :], axis=2)
        nearest = np.argsort(distance, axis=1)[:, :k]
        near = np.take_along_axis(distance, nearest, axis=1)
        # A softening term keeps a point sitting exactly on a node from taking
        # all the weight and producing a hard seam at every node.
        inverse = 1.0 / (near + 1e-3)
        indices[start:stop] = nearest
        weights[start:stop] = inverse / inverse.sum(axis=1, keepdims=True)

    return indices, weights


def synthetic_scene(
    gaussians: int = 6000,
    dynamic_fraction: float = 0.45,
    nodes: int = 12,
    frames: int = 30,
    fps: float = 30.0,
    yaw_degrees: float = 12.0,
    skin_k: int = MAX_SKIN_K,
    seed: int = 4231,
) -> Scene4D:
    """Build a head-turning scene with a known answer.

    ``yaw_degrees`` is the *amplitude*: the head sweeps from ``-yaw`` to
    ``+yaw``, so the visible motion is twice that. The default 12 degrees is
    about what a person actually does in three seconds when told to hold still
    and look at the phone.
    """
    if not 1 <= nodes <= MAX_NODES:
        raise ValueError(f"nodes must be 1..{MAX_NODES}")
    if not 1 <= skin_k <= MAX_SKIN_K:
        raise ValueError(f"skin_k must be 1..{MAX_SKIN_K}")

    rng = np.random.default_rng(seed)
    dynamic_count = max(1, int(round(gaussians * dynamic_fraction)))
    static_count = max(1, gaussians - dynamic_count)

    # --- the subject: an ellipsoidal head above a slab of shoulders ---------
    head_count = int(dynamic_count * 0.72)
    shoulder_count = dynamic_count - head_count

    direction = rng.normal(size=(head_count, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    radius = np.cbrt(rng.uniform(0.35, 1.0, size=(head_count, 1)))
    head = direction * radius * np.array([0.085, 0.105, 0.090]) + np.array([0.0, 0.10, 0.0])

    shoulders = np.column_stack([
        rng.uniform(-0.20, 0.20, shoulder_count),
        rng.uniform(-0.22, -0.05, shoulder_count),
        rng.uniform(-0.06, 0.06, shoulder_count),
    ])
    dynamic_means = np.vstack([head, shoulders])

    # --- the background: a wall behind, a floor below ----------------------
    wall_count = int(static_count * 0.7)
    floor_count = static_count - wall_count
    wall = np.column_stack([
        rng.uniform(-0.9, 0.9, wall_count),
        rng.uniform(-0.6, 0.8, wall_count),
        np.full(wall_count, -0.85) + rng.normal(0, 0.01, wall_count),
    ])
    floor = np.column_stack([
        rng.uniform(-0.9, 0.9, floor_count),
        np.full(floor_count, -0.55) + rng.normal(0, 0.01, floor_count),
        rng.uniform(-0.85, 0.35, floor_count),
    ])
    static_means = np.vstack([wall, floor])

    means = np.vstack([dynamic_means, static_means]).astype(np.float32)

    # --- appearance ---------------------------------------------------------
    scales = np.empty((len(means), 3), dtype=np.float32)
    scales[:dynamic_count] = rng.uniform(0.004, 0.009, (dynamic_count, 3))
    scales[dynamic_count:] = rng.uniform(0.008, 0.016, (static_count, 3))

    quats = rng.normal(size=(len(means), 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)

    colors = np.zeros((len(means), 4), dtype=np.uint8)
    # Skin tone shaded by height, so a turn is legible even in a still frame.
    shade = np.clip((dynamic_means[:, 1] + 0.25) / 0.55, 0.0, 1.0)
    colors[:dynamic_count, 0] = (150 + 90 * shade).astype(np.uint8)
    colors[:dynamic_count, 1] = (105 + 70 * shade).astype(np.uint8)
    colors[:dynamic_count, 2] = (90 + 55 * shade).astype(np.uint8)
    # A checkerboard wall: a uniform background hides a viewer that is drawing
    # the background in the wrong place.
    checker = ((np.floor(static_means[:, 0] * 8) + np.floor(static_means[:, 1] * 8)) % 2)
    colors[dynamic_count:, 0] = (40 + 40 * checker).astype(np.uint8)
    colors[dynamic_count:, 1] = (48 + 44 * checker).astype(np.uint8)
    colors[dynamic_count:, 2] = (62 + 52 * checker).astype(np.uint8)
    colors[:, 3] = 235

    # --- the scaffold -------------------------------------------------------
    # Nodes are spread up the subject so the ramp has something to ramp over.
    heights = np.linspace(-0.20, 0.20, nodes)
    angles = np.linspace(0.0, 4.7, nodes)
    node_rest = np.column_stack([
        0.05 * np.cos(angles),
        heights,
        0.05 * np.sin(angles),
    ]).astype(np.float32)

    # Rotation ramps from nothing at the neck to full at the crown. Clamped at
    # zero so the shoulders are not counter-rotated.
    ramp = np.clip((node_rest[:, 1] - PIVOT[1]) / 0.28, 0.0, 1.0)

    time = np.arange(frames, dtype=np.float64)
    yaw = np.radians(yaw_degrees) * np.sin(2 * np.pi * time / max(frames, 1))
    angle = yaw[:, None] * ramp[None, :]                   # (T, K)

    rotations = _rotation_y(angle)                         # (T, K, 3, 3)
    node_quats = _matrix_to_quaternion(rotations).astype(np.float32)

    offset = node_rest.astype(np.float64) - PIVOT          # (K, 3)
    node_trans = (np.einsum("tkij,kj->tki", rotations, offset) - offset).astype(np.float32)

    skin_indices, skin_weights = _skin(
        dynamic_means, node_rest.astype(np.float64), skin_k
    )

    return Scene4D(
        means=means, scales=scales, quats=quats.astype(np.float32), colors=colors,
        node_rest=node_rest, node_quats=node_quats, node_trans=node_trans,
        skin_indices=skin_indices, skin_weights=skin_weights,
        dynamic_count=dynamic_count, fps=fps,
        # A hand-held arc of this length reconstructs a narrow cone, and the
        # elevation is always the poor axis because the sweep is horizontal.
        cone_azimuth_deg=21.0, cone_elevation_deg=8.0,
        capture_eye=np.array([0.0, 0.05, 0.55], dtype=np.float32),
        capture_forward=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        capture_up=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        fixed_pose_verified=False,
        loop="ping-pong",
        meta={
            "synthetic": True,
            "label": "Synthetic head turn",
            "motion": "yaw about a neck pivot, ramped from neck to crown",
            "yaw_amplitude_deg": float(yaw_degrees),
            "pivot": [float(v) for v in PIVOT],
            "fixed_pose_source": "synthetic",
            "caveat": (
                "This is a generated scene, not a reconstruction. It exists so the "
                "container and the viewer can be tested before any capture exists."
            ),
        },
    )


def expected_position(scene: Scene4D, point: np.ndarray, frame: int, ramp: float) -> np.ndarray:
    """Where a point rigidly attached at ramp fraction ``ramp`` should be.

    The closed form the tests check the whole pipeline against: quantise,
    write, read, skin, and land within a grid step of this.
    """
    frames = scene.frame_count
    yaw_amplitude = np.radians(float(scene.meta["yaw_amplitude_deg"]))
    yaw = yaw_amplitude * np.sin(2 * np.pi * frame / max(frames, 1))
    rotation = _rotation_y(np.array(yaw * ramp))
    return (rotation @ (np.asarray(point, dtype=np.float64) - PIVOT) + PIVOT).astype(np.float32)


__all__ = ["PIVOT", "expected_position", "synthetic_scene"]
