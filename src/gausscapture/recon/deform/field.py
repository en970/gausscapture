"""The deformation field: an SE(3) scaffold, and the 4D feature field that drives it.

**What the representation is.** Motion is carried by ``K <= 256`` scaffold
nodes, each holding a rigid transform per frame, and every dynamic Gaussian is
bound to its nearest few nodes with fixed weights. The deformed position of a
Gaussian is

    p'(t) = sum_s w_s [ R_{k_s}(t) (p - n_{k_s}) + n_{k_s} + d_{k_s}(t) ]

and its deformed rotation is ``normalise(sum_s w_s q_{k_s}(t)) * q``, with the
node quaternions hemisphere-aligned before the sum. That single pair of
equations is the whole writer/reader contract: ``export/deform_reference.py``
reproduces it in NumPy and ``report/viewer_4d.js`` reproduces it in GLSL, and
both are checked against this module.

**Why a scaffold rather than a HexPlane over the Gaussians themselves.** A
fixed camera supplies no parallax to contradict an over-expressive motion
model, so the prior matters more than the capacity. 256 nodes at 90 frames is
~138k motion parameters and the motion is *constrained to be piecewise rigid*,
which is the right prior for a head and shoulders over three seconds. It is
also the only representation of the three we considered that replays in a
browser for free: the viewer uploads K interpolated transforms per frame
instead of evaluating a network per Gaussian.

**Where the 4D feature field comes back in.** :class:`KPlanesField` predicts the
node transforms from ``(x, y, z, t)`` through a multi-resolution K-Planes /
HexPlane grid and a small head, instead of storing them as free parameters. The
grid's bilinear interpolation and its total-variation regularisers impose
spatial and temporal smoothness on the *scaffold*, where 256 sample points make
it cheap, and the result is baked back to explicit per-frame transforms at
export -- so the container format and the viewer never learn that a network was
involved. :class:`ExplicitTrajectoryField` is the free-parameter alternative and
the default, because on a three-second clip it is strictly easier to debug and
the smoothness prior can be supplied by the loss instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gausscapture.recon.deform.knn import knn_indices
from gausscapture.recon.deform.so3 import (
    align_hemisphere,
    axis_angle_to_quat,
    quat_conjugate,
    quat_multiply,
    quat_normalise,
    quat_rotate,
)

#: Plane coordinate pairs, in the order K-Planes names them. Indices 0..2 are
#: space, index 3 is time. The first three planes are purely spatial and the
#: last three are the space-time planes that carry all of the motion -- the
#: regularisers treat the two groups differently, so the split is structural.
PLANE_PAIRS: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3))
N_SPATIAL_PLANES = 3


def _node_affine(quat: Tensor, trans: Tensor, rest: Tensor) -> Tensor:
    """Translation of the equivalent global affine map ``p -> R p + c``."""
    return rest + trans - quat_rotate(quat, rest)


def _from_affine(quat: Tensor, affine_t: Tensor, rest: Tensor) -> Tensor:
    """Inverse of :func:`_node_affine`: recover the node-relative translation."""
    return affine_t + quat_rotate(quat, rest) - rest


@dataclass
class Skinning:
    """A fixed binding of Gaussians to scaffold nodes.

    ``weights`` sums to one along the last axis. It is a plain dataclass rather
    than module state because decision D10 recomputes it from canonical
    positions at every refine boundary; treating it as durable state is exactly
    the mistake that lets split Gaussians inherit a parent's binding.
    """

    idx: Tensor
    weights: Tensor

    def to(self, device: torch.device | str) -> Skinning:
        return Skinning(self.idx.to(device), self.weights.to(device))


def build_skinning(
    points: Tensor, node_rest: Tensor, k: int = 4, sigma: float | None = None
) -> Skinning:
    """Bind each point to its ``k`` nearest nodes with smooth distance weights.

    The kernel width defaults to the mean distance to the ``k``-th node over the
    whole cloud, which makes the binding scale-free: the same code produces
    sensible weights for a head measured in metres and for a synthetic unit-cube
    test scene, and neither needs a tuned constant.
    """
    idx, dist = knn_indices(points, node_rest, k)
    if sigma is None:
        sigma = float(dist[:, -1].mean().clamp_min(1e-6))
    w = torch.exp(-0.5 * (dist / sigma) ** 2)
    w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return Skinning(idx.long(), w)


def deform_gaussians(
    means: Tensor,
    quats: Tensor,
    node_quat: Tensor,
    node_trans: Tensor,
    node_rest: Tensor,
    skinning: Skinning,
    dynamic: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Apply one frame of scaffold motion to the canonical Gaussians.

    ``node_trans`` is each node's own displacement -- node ``k`` moves from
    ``n_k`` to ``n_k + d_k`` -- and the rotation is about that node's rest
    position, matching the formula in the module docstring, in
    ``export/deform_reference.py`` and in the viewer's vertex shader. A *global*
    rigid motion ``p -> R p + s`` is therefore the per-node translations
    ``d_k = R n_k + s - n_k``, not the constant ``d_k = s``: with a constant
    ``s`` the blend leaves the residual ``sum_s w_s (n_s - R n_s)``, which
    vanishes only when every influence happens to share a rest position. Given
    the pivot-absorbing form, every summand equals ``R p + s`` exactly, so the
    result is rigid for *any* weights that sum to one -- which is the property
    the reader implementations are checked against.

    ``dynamic`` is a per-Gaussian 0/1 mask. Static Gaussians are passed through
    untouched rather than being given zero skinning weight, because zero weights
    would blend to the origin, not to the identity -- one canonical splat with a
    label, not two models (decision D11).

    ``skinning.weights`` must be in descending order along the influence axis:
    influence 0 is the hemisphere reference here, in the NumPy reference and in
    the shader, and none of the three searches for the largest weight.
    """
    idx = skinning.idx
    w = skinning.weights.unsqueeze(-1)
    q_sel = node_quat[idx]
    rest_sel = node_rest[idx]
    local = means.unsqueeze(1) - rest_sel
    posed = quat_rotate(q_sel, local) + rest_sel + node_trans[idx]
    means_def = (w * posed).sum(dim=1)

    aligned = align_hemisphere(q_sel, q_sel[:, :1])
    q_blend = quat_normalise((w * aligned).sum(dim=1))
    quats_def = quat_multiply(q_blend, quats)

    if dynamic is not None:
        gate = dynamic.reshape(-1, 1).to(means.dtype)
        means_def = means + gate * (means_def - means)
        quats_def = torch.where(gate > 0.5, quats_def, quats)
    return means_def, quats_def


def interpolate_nodes(
    node_quat: Tensor, node_trans: Tensor, frame_times: Tensor, t: Tensor
) -> tuple[Tensor, Tensor]:
    """Sample baked ``(T, K, ...)`` trajectories at continuous times ``t``.

    This is the same nlerp the viewer runs on the CPU once per displayed frame.
    Keeping it here means the "continuous tau" playback the browser offers can
    be evaluated in a test, rather than being a JavaScript-only behaviour.
    """
    t = t.reshape(-1)
    hi = torch.searchsorted(frame_times, t.clamp(frame_times[0], frame_times[-1]))
    hi = hi.clamp(1, frame_times.shape[0] - 1)
    lo = hi - 1
    span = (frame_times[hi] - frame_times[lo]).clamp_min(1e-12)
    u = ((t - frame_times[lo]) / span).clamp(0.0, 1.0).reshape(-1, 1, 1)
    q_lo, q_hi = node_quat[lo], node_quat[hi]
    q = quat_normalise(torch.lerp(q_lo, align_hemisphere(q_hi, q_lo), u))
    d = torch.lerp(node_trans[lo], node_trans[hi], u)
    return q, d


class ScaffoldField(nn.Module):
    """Base class fixing the node-transform contract and the anchor gauge.

    Subclasses implement :meth:`raw_transforms`. The base class optionally
    composes every output with the inverse of the transform at ``anchor_time``,
    which makes the canonical Gaussians mean "the scene as it was at that
    instant" rather than "the scene at whatever pose the optimiser drifted the
    field to". Without a gauge fix the canonical geometry and the field can
    rotate against each other indefinitely at zero loss, and the exported
    ``frame0.splat`` -- which is what opens in SuperSplat -- then shows a pose
    nobody recorded.
    """

    def __init__(self, node_rest: Tensor, anchor_time: float | None = 0.0) -> None:
        super().__init__()
        if node_rest.ndim != 2 or node_rest.shape[-1] != 3:
            raise ValueError(f"node_rest must be (K, 3), got {tuple(node_rest.shape)}")
        self.register_buffer("node_rest", node_rest.detach().clone().float())
        self.anchor_time = anchor_time

    @property
    def n_nodes(self) -> int:
        return int(self.node_rest.shape[0])

    def raw_transforms(self, t: Tensor) -> tuple[Tensor, Tensor]:
        """Node quaternions ``(B, K, 4)`` and translations ``(B, K, 3)``."""
        raise NotImplementedError

    def forward(self, t: Tensor) -> tuple[Tensor, Tensor]:
        t = torch.as_tensor(t, dtype=self.node_rest.dtype, device=self.node_rest.device).reshape(-1)
        quat, trans = self.raw_transforms(t)
        if self.anchor_time is None:
            return quat, trans
        anchor = t.new_full((1,), float(self.anchor_time))
        q_a, d_a = self.raw_transforms(anchor)
        return self._compose_with_anchor_inverse(quat, trans, q_a[0], d_a[0])

    def _compose_with_anchor_inverse(
        self, quat: Tensor, trans: Tensor, q_a: Tensor, d_a: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return ``T(t) . T(anchor)^-1`` per node, in the node-relative form.

        A node transform is only a rigid motion about that node's own rest
        position, so it cannot be composed in the stored parameterisation:
        ``(q, d)`` is first expanded to the equivalent global affine ``p -> R p
        + c``, composed there, and folded back. Composing the translations
        directly is the mistake that leaves the anchor pose correct in rotation
        and drifting in position.

        Both quaternions are normalised *before* the product rather than only
        after it. At ``t == anchor`` the two operands are then identical, so the
        vector part of ``q * conj(q)`` cancels term by term and the composed
        transform is the identity to within the rounding of a single product
        -- around 1e-9 on the quaternion and 1e-11 on the translation, which is
        the float32 noise floor rather than a drift. Normalising only the anchor
        operand makes the two sides differ in their last bits before they are
        multiplied, and the residual grows by two orders of magnitude.
        """
        rest = self.node_rest
        c = _node_affine(quat, trans, rest)
        c_a = _node_affine(q_a, d_a, rest)
        q_f = quat_normalise(
            quat_multiply(quat_normalise(quat), quat_conjugate(quat_normalise(q_a)))
        )
        c_f = c - quat_rotate(q_f, c_a)
        return q_f, _from_affine(q_f, c_f, rest)

    @torch.no_grad()
    def bake(self, times: Tensor) -> tuple[Tensor, Tensor]:
        """Evaluate the field at every frame time, for export and for the viewer."""
        quat, trans = self.forward(times)
        return quat.contiguous(), trans.contiguous()


class ExplicitTrajectoryField(ScaffoldField):
    """Free per-node, per-keyframe ``se(3)`` parameters.

    Parameters are stored as axis-angle plus translation, both zero-initialised,
    so the field is exactly the identity before the first optimiser step -- which
    is what lets the coarse stage run with the field present but inert instead
    of needing a separate code path. Between keyframes the *parameters* are
    linearly interpolated and then exponentiated; at keyframes, which is where
    every training sample and every exported frame lands, the result is exact.
    """

    def __init__(
        self,
        node_rest: Tensor,
        frame_times: Tensor,
        anchor_time: float | None = 0.0,
    ) -> None:
        super().__init__(node_rest, anchor_time=anchor_time)
        times = frame_times.detach().clone().float().reshape(-1)
        if times.numel() < 2:
            raise ValueError("a trajectory needs at least two keyframes")
        if bool((times[1:] <= times[:-1]).any()):
            raise ValueError("frame_times must be strictly increasing")
        self.register_buffer("frame_times", times)
        shape = (times.numel(), self.n_nodes, 3)
        self.axis_angle = nn.Parameter(torch.zeros(shape))
        self.translation = nn.Parameter(torch.zeros(shape))

    def raw_transforms(self, t: Tensor) -> tuple[Tensor, Tensor]:
        times = self.frame_times
        hi = torch.searchsorted(times, t.clamp(times[0], times[-1])).clamp(1, times.numel() - 1)
        lo = hi - 1
        span = (times[hi] - times[lo]).clamp_min(1e-12)
        u = ((t - times[lo]) / span).clamp(0.0, 1.0).reshape(-1, 1, 1)
        aa = torch.lerp(self.axis_angle[lo], self.axis_angle[hi], u)
        tr = torch.lerp(self.translation[lo], self.translation[hi], u)
        return axis_angle_to_quat(aa), tr

    @property
    def spacetime_parameters(self) -> tuple[Tensor, Tensor]:
        """The raw ``(T, K, 3)`` trajectories, for the temporal regularisers."""
        return self.axis_angle, self.translation


class KPlanesField(ScaffoldField):
    """Multi-resolution K-Planes / HexPlane feature field over ``(x, y, z, t)``.

    Six planes per scale -- three spatial and three space-time -- are sampled
    bilinearly and multiplied together, the scales are concatenated, and a small
    MLP maps the result to a per-node ``se(3)`` residual. The final layer is
    zero-initialised so the field is the identity everywhere at step zero,
    matching :class:`ExplicitTrajectoryField` and letting the same coarse stage
    run in front of either.

    ``rotation_scale`` and ``translation_scale`` are hard bounds, not gains: the
    head is squashed through ``tanh`` before them, so no node can rotate by more
    than ``rotation_scale`` radians per axis or translate by more than
    ``translation_scale`` world units, however the optimiser drives it. They are
    in world units, so a scene whose subject genuinely moves further than the
    default must be given a larger bound rather than left to saturate.

    The time axis is deliberately *not* upsampled with the spatial scales. A
    fixed camera can drive an arbitrarily fine temporal grid straight into
    per-frame appearance memorisation, which looks like a falling loss and is
    not reconstruction; the spatial multiresolution is what the representation
    is for.
    """

    def __init__(
        self,
        node_rest: Tensor,
        aabb: Tensor,
        feature_dim: int = 16,
        base_resolution: int = 32,
        time_resolution: int = 25,
        multires: tuple[int, ...] = (1, 2),
        hidden_dim: int = 64,
        rotation_scale: float = 0.25,
        translation_scale: float = 0.25,
        anchor_time: float | None = 0.0,
    ) -> None:
        super().__init__(node_rest, anchor_time=anchor_time)
        if aabb.shape != (2, 3):
            raise ValueError(f"aabb must be (2, 3), got {tuple(aabb.shape)}")
        if not multires:
            raise ValueError("multires must name at least one scale")
        self.register_buffer("aabb", aabb.detach().clone().float())
        self.feature_dim = int(feature_dim)
        self.multires = tuple(int(m) for m in multires)
        self.rotation_scale = float(rotation_scale)
        self.translation_scale = float(translation_scale)

        self.scales = nn.ModuleList()
        for mult in self.multires:
            res = [base_resolution * mult] * 3 + [time_resolution]
            planes = nn.ParameterList()
            for i, j in PLANE_PAIRS:
                # (1, C, H, W) with W indexing coordinate i and H coordinate j,
                # matching the grid built in _sample_features.
                p = torch.empty(1, self.feature_dim, res[j], res[i])
                # Multiplicative composition means the neutral element is one,
                # not zero; initialising near one keeps the product of six
                # planes from starting at 1e-6 and stalling the head.
                nn.init.uniform_(p, 0.9, 1.1)
                planes.append(nn.Parameter(p))
            self.scales.append(planes)

        in_dim = self.feature_dim * len(self.multires)
        head_out = nn.Linear(hidden_dim, 6)
        nn.init.zeros_(head_out.weight)
        nn.init.zeros_(head_out.bias)
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            head_out,
        )

    def spatial_planes(self) -> list[Tensor]:
        """The three purely spatial planes of every scale."""
        return [p for planes in self.scales for p in list(planes)[:N_SPATIAL_PLANES]]

    def spacetime_planes(self) -> list[Tensor]:
        """The three space-time planes of every scale, whose last axis is time."""
        return [p for planes in self.scales for p in list(planes)[N_SPATIAL_PLANES:]]

    def _normalised_coords(self, t: Tensor) -> Tensor:
        """Build ``(B, K, 4)`` coordinates in ``[-1, 1]`` for grid_sample."""
        lo, hi = self.aabb[0], self.aabb[1]
        xyz = 2.0 * (self.node_rest - lo) / (hi - lo).clamp_min(1e-8) - 1.0
        b, k = t.shape[0], self.n_nodes
        xyz = xyz.unsqueeze(0).expand(b, k, 3)
        tt = (2.0 * t - 1.0).reshape(b, 1, 1).expand(b, k, 1)
        return torch.cat([xyz, tt], dim=-1).clamp(-1.0, 1.0)

    def _sample_features(self, coords: Tensor) -> Tensor:
        b, k, _ = coords.shape
        flat = coords.reshape(1, 1, b * k, 4)
        feats = []
        for planes in self.scales:
            acc: Tensor | None = None
            for plane, (i, j) in zip(list(planes), PLANE_PAIRS, strict=True):
                grid = torch.stack([flat[..., i], flat[..., j]], dim=-1)
                sampled = F.grid_sample(
                    plane, grid, mode="bilinear", align_corners=True, padding_mode="border"
                )
                sampled = sampled.reshape(self.feature_dim, b * k).T
                acc = sampled if acc is None else acc * sampled
            assert acc is not None
            feats.append(acc)
        return torch.cat(feats, dim=-1).reshape(b, k, -1)

    def raw_transforms(self, t: Tensor) -> tuple[Tensor, Tensor]:
        """Node transforms, with the head's output saturated onto the scales.

        ``tanh`` rather than a bare multiply, so ``rotation_scale`` and
        ``translation_scale`` are hard per-axis bounds on the residual instead
        of arbitrary gains. A fixed camera supplies no parallax to contradict a
        large motion, so an unbounded linear head can drive a node metres out of
        the scene and still lower the loss by painting the hole it leaves; the
        bound is the cheapest prior that makes that unreachable rather than
        merely unlikely. ``tanh(0) == 0`` exactly, so the zero-initialised final
        layer still makes the field the exact identity at step zero.
        """
        feats = self._sample_features(self._normalised_coords(t))
        out = torch.tanh(self.head(feats))
        aa = out[..., :3] * self.rotation_scale
        tr = out[..., 3:] * self.translation_scale
        return axis_angle_to_quat(aa), tr


def seed_nodes(points: Tensor, n_nodes: int, generator: torch.Generator | None = None) -> Tensor:
    """Farthest-point sample ``n_nodes`` scaffold seeds from a point cloud.

    Farthest-point rather than k-means or random: the scaffold has to *cover*
    the dynamic region, and coverage is what farthest-point sampling optimises
    directly. A random subset leaves gaps exactly where the point density is
    low, which on a face is the cheek and the neck -- large, smooth, and the
    first places a missing node shows as a smear.
    """
    n = points.shape[0]
    if n == 0:
        raise ValueError("cannot seed a scaffold from an empty point set")
    k = min(int(n_nodes), n)
    idx = torch.empty(k, dtype=torch.long, device=points.device)
    if generator is None:
        idx[0] = int(points.norm(dim=1).argmax())
    else:
        idx[0] = int(torch.randint(0, n, (1,), generator=generator, device=points.device))
    dist = (points - points[idx[0]]).square().sum(dim=1)
    for i in range(1, k):
        idx[i] = int(dist.argmax())
        dist = torch.minimum(dist, (points - points[idx[i]]).square().sum(dim=1))
    return points[idx].contiguous()


def scene_aabb(points: Tensor, margin: float = 0.1) -> Tensor:
    """Padded axis-aligned bounds, used to normalise K-Planes coordinates.

    The margin matters because MCMC relocation and the position noise move
    canonical means after the bounds are computed; a Gaussian that leaves the
    box would otherwise be clamped onto the border texel and share one feature
    with everything else outside.
    """
    lo = points.amin(dim=0)
    hi = points.amax(dim=0)
    pad = (hi - lo).clamp_min(1e-6) * margin
    return torch.stack([lo - pad, hi + pad])


def default_frame_times(n_frames: int) -> Tensor:
    """Uniform normalised times in ``[0, 1]`` for ``n_frames`` samples."""
    if n_frames < 2:
        raise ValueError("need at least two frames")
    return torch.linspace(0.0, 1.0, n_frames)


def node_edges(node_rest: Tensor, k: int = 6) -> Tensor:
    """Symmetric k-nearest-neighbour edge list ``(E, 2)`` for the ARAP term.

    ARAP needs a graph, and the scaffold has no mesh. k-nearest over the rest
    positions is the graph the nodes themselves imply; ``k = 6`` is the smallest
    that reliably connects a farthest-point sampling of a head into one
    component, and a disconnected scaffold is one whose halves can drift apart
    at zero regularisation cost.
    """
    idx, _ = knn_indices(node_rest, node_rest, min(k + 1, node_rest.shape[0]))
    src = torch.arange(node_rest.shape[0], device=node_rest.device).unsqueeze(1).expand_as(idx)
    pairs = torch.stack([src.reshape(-1), idx.reshape(-1)], dim=1)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    n = node_rest.shape[0]
    # Canonicalise (i, j) so that the two directed copies of an edge collapse to
    # one; ARAP counting an edge twice silently doubles its weight.
    key = torch.unique(pairs.min(dim=1).values * n + pairs.max(dim=1).values)
    return torch.stack([torch.div(key, n, rounding_mode="floor"), key % n], dim=1)
