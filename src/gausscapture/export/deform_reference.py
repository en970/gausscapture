"""The deformation contract, written once in NumPy so three sides can agree.

Three implementations evaluate the same motion: the PyTorch field the trainer
optimises, this module, and the GLSL in ``report/viewer_4d.js``. Nothing forces
them to agree, and every way they can disagree is silent -- a transposed
rotation looks like a subject who leans the wrong way, a quaternion blended
without hemisphere alignment looks like a head that snaps round between two
frames, and a scaffold applied about the origin instead of about each node's
rest position looks like a face that swells as it turns. None of those raise.

So the arithmetic is stated here, exactly once.
``tests/test_deform_reference.py`` checks this module against the PyTorch field
the trainer optimises -- position, orientation, keyframe interpolation, the
hemisphere alignment and the depth fold -- and against closed forms neither of
them computes. The GLSL is the one side that cannot be executed from Python; it
is kept honest by being written from these formulas and by the synthetic scene,
whose motion is obvious enough to the eye that a shader disagreeing with this
module is visible in the first second of playback.

The contract
------------

A gaussian at canonical position :math:`p` with canonical orientation
:math:`q` is bound to :math:`k` scaffold nodes with indices :math:`i` and
weights :math:`w_i` summing to one. Node :math:`j` has a rest position
:math:`n_j` and, at frame :math:`f`, a rotation :math:`r_j^f` and a translation
:math:`t_j^f`.

At continuous time :math:`\\tau` between frames :math:`f` and :math:`f+1`:

1. **Sample.** :math:`r_j(\\tau)` is the normalised linear blend of
   :math:`r_j^f` and :math:`r_j^{f+1}` after aligning their hemispheres;
   :math:`t_j(\\tau)` is their linear blend. Normalised linear blending rather
   than slerp: at 30 fps the two differ by well under the quantisation error,
   and nlerp is four multiplies where slerp is two trigonometric functions per
   node per frame.

2. **Skin the position** by linear blend skinning about each node's own rest
   position, which is what makes the motion *piecewise rigid* rather than an
   arbitrary field:

   .. math:: p' = \\sum_i w_i \\left( R(r_i)\\,(p - n_i) + n_i + t_i \\right)

3. **Skin the orientation** by blending the node rotations first and applying
   the result on the left, because a rotation of the world composes before the
   gaussian's own orientation, not after:

   .. math:: q' = \\mathrm{normalise}\\!\\left(\\sum_i w_i \\hat r_i\\right) \\otimes q

   where :math:`\\hat r_i` is :math:`r_i` negated when it lies in the opposite
   hemisphere from the highest-weighted influence. Blending the *rotations* and
   applying once, rather than rotating and averaging, is what keeps a gaussian
   spanning two nodes from collapsing.

Static gaussians -- everything from index ``dynamic_count`` on -- are not
skinned at all. They receive the identity, which is the whole reason the export
orders them last.
"""

from __future__ import annotations

import numpy as np

from gausscapture.export.scene4d import Scene4D


def quaternion_to_matrix(quats: np.ndarray) -> np.ndarray:
    """``(..., 4)`` ``(w, x, y, z)`` quaternions to ``(..., 3, 3)`` rotations."""
    q = np.asarray(quats, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
    ], axis=-2)


def quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a ⊗ b`` over broadcasting ``(..., 4)`` arrays."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def sample_nodes(
    node_quats: np.ndarray, node_trans: np.ndarray, tau: float
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate the scaffold to a continuous frame index ``tau``.

    ``tau`` is clamped to ``[0, frames - 1]``: extrapolating a fitted motion
    past the clip is inventing frames nobody recorded, and the viewer's loop
    modes never ask for it.
    """
    node_quats = np.asarray(node_quats, dtype=np.float64)
    node_trans = np.asarray(node_trans, dtype=np.float64)
    frames = len(node_quats)

    tau = float(np.clip(tau, 0.0, frames - 1))
    lower = int(np.floor(tau))
    upper = min(lower + 1, frames - 1)
    alpha = tau - lower

    a, b = node_quats[lower], node_quats[upper]
    # Hemisphere alignment per node, not per frame: two nodes in the same frame
    # can sit on opposite sides, and a whole-frame flip would fix one and break
    # the other.
    b = np.where((np.sum(a * b, axis=-1, keepdims=True) < 0.0), -b, b)
    blended = (1.0 - alpha) * a + alpha * b
    blended = blended / np.maximum(np.linalg.norm(blended, axis=-1, keepdims=True), 1e-12)

    trans = (1.0 - alpha) * node_trans[lower] + alpha * node_trans[upper]
    return blended, trans


def skin_positions(
    means: np.ndarray,
    node_rest: np.ndarray,
    node_quats: np.ndarray,
    node_trans: np.ndarray,
    skin_indices: np.ndarray,
    skin_weights: np.ndarray,
) -> np.ndarray:
    """Linear blend skinning about each node's rest position."""
    means = np.asarray(means, dtype=np.float64)
    rest = np.asarray(node_rest, dtype=np.float64)
    idx = np.asarray(skin_indices, dtype=np.int64)
    w = np.asarray(skin_weights, dtype=np.float64)

    rotations = quaternion_to_matrix(node_quats)          # (K, 3, 3)
    local = means[:, None, :] - rest[idx]                 # (N, k, 3)
    rotated = np.einsum("nkij,nkj->nki", rotations[idx], local)
    moved = rotated + rest[idx] + np.asarray(node_trans, dtype=np.float64)[idx]
    return np.sum(w[..., None] * moved, axis=1).astype(np.float32)


def skin_quaternions(
    quats: np.ndarray,
    node_quats: np.ndarray,
    skin_indices: np.ndarray,
    skin_weights: np.ndarray,
) -> np.ndarray:
    """Blend the influencing node rotations, then apply on the left."""
    q = np.asarray(quats, dtype=np.float64)
    idx = np.asarray(skin_indices, dtype=np.int64)
    w = np.asarray(skin_weights, dtype=np.float64)

    influencing = np.asarray(node_quats, dtype=np.float64)[idx]   # (N, k, 4)
    # Align every influence with the highest-weighted one, so the blend does
    # not cancel between two nodes describing the same rotation with opposite
    # signs -- which produces a near-zero quaternion and, after normalising,
    # an arbitrary orientation.
    reference = influencing[np.arange(len(idx)), np.argmax(w, axis=1)][:, None, :]
    aligned = np.where(np.sum(influencing * reference, axis=-1, keepdims=True) < 0.0,
                       -influencing, influencing)

    blended = np.sum(w[..., None] * aligned, axis=1)
    blended = blended / np.maximum(np.linalg.norm(blended, axis=-1, keepdims=True), 1e-12)

    out = quaternion_multiply(blended, q)
    return (out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-12)).astype(np.float32)


def deform_scene(scene: Scene4D, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a whole scene at continuous time ``tau``.

    Returns ``(means, quats)`` for every gaussian, dynamic and static, in the
    file's own order -- so the result indexes the same way ``colors`` and
    ``scales`` do.
    """
    quats_t, trans_t = sample_nodes(scene.node_quats, scene.node_trans, tau)

    means = np.array(scene.means, dtype=np.float32, copy=True)
    quats = np.array(scene.quats, dtype=np.float32, copy=True)

    dynamic = scene.dynamic_count
    if dynamic:
        means[:dynamic] = skin_positions(
            scene.means[:dynamic], scene.node_rest, quats_t, trans_t,
            scene.skin_indices, scene.skin_weights,
        )
        quats[:dynamic] = skin_quaternions(
            scene.quats[:dynamic], quats_t, scene.skin_indices, scene.skin_weights,
        )
    return means, quats


def fold_view_row(
    node_rest: np.ndarray,
    node_quats: np.ndarray,
    node_trans: np.ndarray,
    view_row: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce depth-under-a-view to a dot product per influence.

    Sorting for alpha blending needs only the depth of each deformed gaussian,
    never its deformed position. Substituting the skinning formula into
    :math:`d \\cdot p'` and collecting terms gives

    .. math:: d \\cdot p' = \\sum_i w_i \\left( g_i \\cdot p + h_i \\right),
              \\quad g_i = R(r_i)^{\\!\\top} d,
              \\quad h_i = d \\cdot (n_i + t_i) - g_i \\cdot n_i

    so the per-gaussian cost drops to four dot products and no 3-vector is ever
    materialised. The reduction is per node, of which there are at most 256, so
    it happens once per sort rather than once per gaussian. This is what makes
    a per-frame re-sort of several hundred thousand deformed gaussians cheap
    enough to do in a worker while the clip plays.
    """
    rest = np.asarray(node_rest, dtype=np.float64)
    d = np.asarray(view_row, dtype=np.float64).reshape(3)
    rotations = quaternion_to_matrix(node_quats)                   # (K, 3, 3)

    g = np.einsum("kji,j->ki", rotations, d)                       # Rᵀ d
    h = (rest + np.asarray(node_trans, dtype=np.float64)) @ d - np.sum(g * rest, axis=1)
    return g.astype(np.float32), h.astype(np.float32)


def deformed_depths(
    means: np.ndarray,
    skin_indices: np.ndarray,
    skin_weights: np.ndarray,
    g: np.ndarray,
    h: np.ndarray,
) -> np.ndarray:
    """Depths of skinned gaussians under a folded view row."""
    idx = np.asarray(skin_indices, dtype=np.int64)
    w = np.asarray(skin_weights, dtype=np.float64)
    p = np.asarray(means, dtype=np.float64)
    per_influence = np.einsum("nki,ni->nk", np.asarray(g, dtype=np.float64)[idx], p)
    per_influence = per_influence + np.asarray(h, dtype=np.float64)[idx]
    return np.sum(w * per_influence, axis=1)


__all__ = [
    "deform_scene",
    "deformed_depths",
    "fold_view_row",
    "quaternion_multiply",
    "quaternion_to_matrix",
    "sample_nodes",
    "skin_positions",
    "skin_quaternions",
]
