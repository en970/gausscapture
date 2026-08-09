"""Rotation primitives shared by the field, the rasterizer and the exporter.

These live in their own module because three otherwise unrelated pieces of the
4D path have to agree on them *bit for bit*: the deformation field composes
node rotations, the reference rasterizer builds 2D covariances from Gaussian
rotations, and ``export/deform_reference.py`` reproduces the same arithmetic in
NumPy so that ``viewer_4d.js`` can be validated against it. A second, subtly
different quaternion convention anywhere in that chain shows up as a scene that
renders correctly in Python and wrongly in the browser, which is the most
expensive class of bug this project can have.

Convention, stated once so nothing has to guess:

* quaternions are ``(w, x, y, z)``, right-handed, and rotate **column vectors**
  as ``p' = R(q) p``;
* ``quat_multiply(a, b)`` is the rotation "apply ``b`` first, then ``a``", i.e.
  ``R(a @ b) == R(a) @ R(b)``;
* nothing here normalises implicitly. Callers normalise where they mean to,
  because an unnoticed normalisation hides a drifting parameterisation.
"""

from __future__ import annotations

import torch
from torch import Tensor


def quat_normalise(q: Tensor, eps: float = 1e-12) -> Tensor:
    """Unit-normalise the last axis, safe for the exactly-zero quaternion."""
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def quat_conjugate(q: Tensor) -> Tensor:
    """Conjugate, which is the inverse for a unit quaternion."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_multiply(a: Tensor, b: Tensor) -> Tensor:
    """Hamilton product, broadcasting over every leading axis."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_to_matrix(q: Tensor) -> Tensor:
    """Rotation matrix ``(..., 3, 3)`` from a unit quaternion ``(..., 4)``."""
    w, x, y, z = quat_normalise(q).unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def quat_rotate(q: Tensor, p: Tensor) -> Tensor:
    """Rotate points ``(..., 3)`` by unit quaternions ``(..., 4)``.

    The leading axes of ``q`` and ``p`` are broadcast against each other
    explicitly, because ``torch.cross`` requires equal *rank* and not merely
    compatible shapes. Callers rely on that: the anchor gauge rotates the
    ``(K, 3)`` node rest positions by a whole batch of ``(B, K, 4)`` node
    rotations, and the ARAP term rotates one ``(1, E, 3)`` set of rest edges by
    ``(B, E, 4)``. Without the broadcast the first evaluation of any field with
    an anchor dies with "linalg.cross: inputs must have the same number of
    dimensions", which is a shape bug wearing a numerical error's clothes.
    """
    qn = quat_normalise(q)
    batch = torch.broadcast_shapes(qn.shape[:-1], p.shape[:-1])
    w = qn[..., :1].expand(*batch, 1)
    v = qn[..., 1:].expand(*batch, 3)
    p = p.expand(*batch, 3)
    t = 2.0 * torch.cross(v, p, dim=-1)
    return p + w * t + torch.cross(v, t, dim=-1)


def axis_angle_to_quat(aa: Tensor, eps: float = 1e-8) -> Tensor:
    """Exponential map from ``so(3)`` to a unit quaternion.

    The small-angle branch is not an optimisation: ``sin(theta/2)/theta`` is
    ``0/0`` at the identity, and the identity is exactly where every trajectory
    parameter starts, so the naive form produces NaN gradients on step one.
    """
    theta = aa.norm(dim=-1, keepdim=True)
    half = 0.5 * theta
    # Second-order Taylor of sin(x)/x, accurate to ~1e-13 over the branch range.
    small = theta < eps
    sinc_half = torch.where(small, 0.5 - theta * theta / 48.0, torch.sin(half) / theta.clamp_min(eps))
    return torch.cat([torch.cos(half), aa * sinc_half], dim=-1)


def quat_to_axis_angle(q: Tensor, eps: float = 1e-8) -> Tensor:
    """Logarithmic map, taking the shorter of the two equivalent rotations."""
    qn = quat_normalise(q)
    qn = torch.where(qn[..., :1] < 0, -qn, qn)
    v = qn[..., 1:]
    sin_half = v.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, qn[..., :1])
    scale = torch.where(sin_half < eps, torch.full_like(sin_half, 2.0), angle / sin_half.clamp_min(eps))
    return v * scale


def align_hemisphere(q: Tensor, reference: Tensor) -> Tensor:
    """Flip ``q`` onto the hemisphere of ``reference``.

    ``q`` and ``-q`` are the same rotation but average to zero, so any linear
    blend or temporal smoothness term over raw quaternions is meaningless until
    the signs agree. Every place this module blends or differences quaternions
    calls this first.
    """
    sign = torch.where((q * reference).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    return q * sign


def nlerp(a: Tensor, b: Tensor, weight: Tensor) -> Tensor:
    """Normalised linear interpolation, hemisphere-corrected.

    The viewer interpolates node rotations this way -- it cannot afford slerp's
    trigonometry per frame -- so training and export use it too. Over the
    per-frame angles a head produces in three seconds, nlerp and slerp differ
    by well under the quaternion quantisation the container applies anyway.
    """
    return quat_normalise(torch.lerp(a, align_hemisphere(b, a), weight))
