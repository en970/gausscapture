"""Nearest-neighbour scale initialisation, in PyTorch.

The reference 3DGS implementation initialises each Gaussian's scale from the
mean squared distance to its three nearest neighbours, using a small CUDA
extension published under the Inria non-commercial licence. GaussCapture cannot
ship, vendor or depend on it, so this module replaces it. (The extension is
named in ``docs/DEPENDENCIES.md`` rather than here: the licence gate treats that
name appearing in shipped source as the signal that it has been vendored, and a
docstring is not worth a false positive.) Its kernel is *approximate* anyway: it
buckets points along a Morton curve and searches only within a bucket, which
mis-scales Gaussians near a bucket boundary.

This is the exact computation instead, chunked so a 500k-point cloud fits in
memory. It is O(N*M) rather than O(N log N), which is the honest cost of being
exact; at 500k points on an M2 Air it is a few seconds once, at initialisation
time, and never again. ``gausscapture.recon.knn`` provides the same result on
NumPy for the CPU preparation pass; this module exists so the trainer can
recompute scales on whatever device it is already on, without a round trip
through the host.
"""

from __future__ import annotations

import torch
from torch import Tensor

#: Squared-distance floor. Duplicate points -- which COLMAP does produce, and
#: which MCMC relocation produces deliberately -- otherwise give a mean squared
#: distance of exactly zero and hence a log scale of -inf, which poisons every
#: subsequent gradient. Clamping is not a fudge: two Gaussians at the same
#: coordinate genuinely have no neighbour distance to measure, and the smallest
#: scale we are willing to represent is the only defensible answer.
MIN_SQ_DIST = 1e-14


def knn_mean_sq_dist(points: Tensor, k: int = 3, chunk: int = 4096) -> Tensor:
    """Mean squared distance from each point to its ``k`` nearest neighbours.

    The point itself is excluded. When ``N <= k`` there are not enough
    neighbours to average, so every available neighbour is used; with a single
    point the result is :data:`MIN_SQ_DIST`, not an error, because a one-point
    cloud is a degenerate input a test may legitimately hand us.
    """
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"points must be (N, 3), got {tuple(points.shape)}")
    n = points.shape[0]
    if n == 0:
        return points.new_zeros((0,))
    if n == 1:
        return points.new_full((1,), MIN_SQ_DIST)

    take = min(k, n - 1)
    out = points.new_empty((n,))
    for start in range(0, n, chunk):
        block = points[start : start + chunk]
        # cdist is squared here rather than via p=2 to avoid the sqrt/square
        # round trip; the values are compared and averaged as squares anyway.
        d2 = torch.cdist(block, points).square()
        # Mask self-distance rather than assuming it is the minimum: exact
        # duplicates make the sort order between a point and its twin arbitrary.
        idx = torch.arange(block.shape[0], device=points.device) + start
        d2[torch.arange(block.shape[0], device=points.device), idx] = float("inf")
        nearest = torch.topk(d2, take, dim=1, largest=False).values
        out[start : start + block.shape[0]] = nearest.mean(dim=1)
    return out.clamp_min(MIN_SQ_DIST)


def init_log_scales(points: Tensor, k: int = 3, max_scale: float | None = None) -> Tensor:
    """Per-Gaussian isotropic log scale ``(N, 3)`` from the local point spacing.

    A Gaussian is initialised to roughly the radius of the gap it has to cover,
    which is what makes the coarse stage converge without either exploding
    splats or a cloud of invisible specks.
    """
    mean_sq = knn_mean_sq_dist(points, k=k)
    scale = mean_sq.sqrt()
    if max_scale is not None:
        scale = scale.clamp_max(max_scale)
    return scale.log().unsqueeze(-1).expand(-1, 3).contiguous()


def knn_indices(query: Tensor, reference: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Indices and distances of the ``k`` nearest ``reference`` points.

    Used to bind Gaussians to scaffold nodes. ``reference`` is at most a few
    hundred nodes, so this is a single dense distance matrix and needs no
    chunking, which is why binding can be recomputed at every refine boundary
    (decision D10) instead of being copied across a split.
    """
    if reference.shape[0] == 0:
        raise ValueError("cannot bind to an empty node set")
    take = min(k, reference.shape[0])
    d = torch.cdist(query, reference)
    dist, idx = torch.topk(d, take, dim=1, largest=False)
    if take < k:
        # Pad by repeating the nearest node so downstream shapes stay fixed;
        # the duplicate weights sum with it and the blend is unchanged.
        pad = k - take
        idx = torch.cat([idx, idx[:, :1].expand(-1, pad)], dim=1)
        dist = torch.cat([dist, dist[:, :1].expand(-1, pad)], dim=1)
    return idx, dist
