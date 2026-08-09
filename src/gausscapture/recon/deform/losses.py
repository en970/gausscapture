"""Losses and regularisers for the deformable stage.

A fixed camera gives one viewpoint, so the photometric term alone is massively
under-determined: infinitely many motions reproduce the observed pixels, and
almost all of them are nonsense the moment the viewer moves off the capture
axis. Every regulariser here exists to remove a specific family of those
solutions, and each docstring names which one.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from gausscapture.recon.deform.so3 import align_hemisphere

_SSIM_C1 = 0.01**2
_SSIM_C2 = 0.03**2


def _gaussian_window(size: int, sigma: float, channels: int, ref: Tensor) -> Tensor:
    coords = torch.arange(size, dtype=ref.dtype, device=ref.device) - (size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = torch.outer(g, g)
    return kernel.expand(channels, 1, size, size).contiguous()


def ssim(pred: Tensor, target: Tensor, window_size: int = 11, sigma: float = 1.5) -> Tensor:
    """Mean structural similarity between two ``(H, W, C)`` images.

    The window is shrunk to fit small images rather than raising: the CPU tests
    render at 32x32, and an 11-pixel window over a 32-pixel image would be
    dominated by padding.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    h, w, c = pred.shape
    window_size = min(window_size, h, w)
    if window_size % 2 == 0:
        window_size -= 1
    window_size = max(window_size, 3)
    pad = window_size // 2
    kernel = _gaussian_window(window_size, sigma, c, pred)
    x = pred.permute(2, 0, 1).unsqueeze(0)
    y = target.permute(2, 0, 1).unsqueeze(0)

    mu_x = F.conv2d(x, kernel, padding=pad, groups=c)
    mu_y = F.conv2d(y, kernel, padding=pad, groups=c)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x = F.conv2d(x * x, kernel, padding=pad, groups=c) - mu_x2
    sigma_y = F.conv2d(y * y, kernel, padding=pad, groups=c) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=pad, groups=c) - mu_xy

    num = (2 * mu_xy + _SSIM_C1) * (2 * sigma_xy + _SSIM_C2)
    den = (mu_x2 + mu_y2 + _SSIM_C1) * (sigma_x + sigma_y + _SSIM_C2)
    return (num / den).mean()


def photometric_loss(
    pred: Tensor,
    target: Tensor,
    lambda_ssim: float = 0.2,
    weights: Tensor | None = None,
) -> Tensor:
    """``(1 - w) * L1 + w * (1 - SSIM)``, optionally weighted per pixel.

    ``weights`` is how the dynamic mask enters training: the fixed camera sees
    the same background in every frame, so an unweighted loss spends most of its
    gradient re-fitting pixels that never changed. The SSIM term stays global,
    because structural similarity over a masked region is not a well-defined
    quantity once the window straddles the mask boundary.
    """
    if weights is None:
        l1 = (pred - target).abs().mean()
    else:
        w = weights.reshape(pred.shape[0], pred.shape[1], -1)
        l1 = ((pred - target).abs() * w).sum() / (w.sum() * pred.shape[-1]).clamp_min(1e-8)
    return (1.0 - lambda_ssim) * l1 + lambda_ssim * (1.0 - ssim(pred, target))


def opacity_regulariser(opacities: Tensor) -> Tensor:
    """L1 on opacity, as 3DGS-MCMC specifies.

    MCMC's relocation signal *is* opacity, so a term that pushes unneeded
    Gaussians towards transparent is what keeps the population turning over
    instead of accumulating invisible clutter against the ``cap_max`` budget.
    """
    return opacities.abs().mean()


def scale_regulariser(scales: Tensor) -> Tensor:
    """L1 on linear scale, the second half of the MCMC prior.

    Without it the cheapest way to explain a fixed camera's single view is one
    enormous flat Gaussian per region -- which reprojects perfectly and looks
    like a smear of coloured glass from three degrees off the capture axis.
    """
    return scales.abs().mean()


def arap_loss(node_rest: Tensor, node_quat: Tensor, node_trans: Tensor, edges: Tensor) -> Tensor:
    """As-rigid-as-possible energy over the scaffold graph, per frame.

    For each edge the deformed edge vector is compared against the rest vector
    carried by the node's own rotation. This is the term that stops the scaffold
    from shearing: a fixed camera cannot distinguish a node moving towards it
    from a node growing, so the neighbours' agreement is the only depth-like
    constraint available.

    Accepts either a single frame ``(K, ...)`` or a batch ``(B, K, ...)``.
    """
    if node_quat.ndim == 2:
        node_quat = node_quat.unsqueeze(0)
        node_trans = node_trans.unsqueeze(0)
    if edges.numel() == 0:
        return node_quat.new_zeros(())
    from gausscapture.recon.deform.so3 import quat_rotate

    i, j = edges[:, 0], edges[:, 1]
    posed = node_rest.unsqueeze(0) + node_trans
    rest_edge = (node_rest[j] - node_rest[i]).unsqueeze(0)
    posed_edge = posed[:, j] - posed[:, i]
    forward = posed_edge - quat_rotate(node_quat[:, i], rest_edge)
    backward = -posed_edge - quat_rotate(node_quat[:, j], -rest_edge)
    return 0.5 * (forward.square().sum(-1).mean() + backward.square().sum(-1).mean())


def trajectory_smoothness(node_quat: Tensor, node_trans: Tensor) -> Tensor:
    """Second-difference penalty along time on ``(T, K, ...)`` trajectories.

    Penalising acceleration rather than velocity is deliberate: a head that
    turns steadily through the clip should cost nothing, while per-frame jitter
    -- which is what a single view lets the optimiser hide arbitrary geometry
    errors in -- costs a lot.
    """
    if node_quat.shape[0] < 3:
        return node_quat.new_zeros(())
    q = align_hemisphere(node_quat, node_quat[:1])
    acc_q = q[2:] - 2 * q[1:-1] + q[:-2]
    acc_t = node_trans[2:] - 2 * node_trans[1:-1] + node_trans[:-2]
    return acc_q.square().sum(-1).mean() + acc_t.square().sum(-1).mean()


def plane_tv_loss(planes: list[Tensor]) -> Tensor:
    """Total variation on the spatial K-Planes grids.

    A grid has no smoothness of its own beyond one texel of bilinear support, so
    without this the field is free to encode a different motion in every cell --
    which is exactly the overfitting a zero-parallax capture invites.
    """
    if not planes:
        return torch.zeros(())
    total = planes[0].new_zeros(())
    for p in planes:
        dh = (p[..., 1:, :] - p[..., :-1, :]).square().mean()
        dw = (p[..., :, 1:] - p[..., :, :-1]).square().mean()
        total = total + dh + dw
    return total / len(planes)


def time_smoothness_loss(planes: list[Tensor]) -> Tensor:
    """Second-difference penalty along the time axis of the space-time planes.

    Time is stored on the height axis of those planes (see ``PLANE_PAIRS``), so
    this differences ``dim=-2``. Same argument as
    :func:`trajectory_smoothness`, applied to the field that generates the
    trajectories rather than to the trajectories themselves.
    """
    if not planes:
        return torch.zeros(())
    total = planes[0].new_zeros(())
    for p in planes:
        if p.shape[-2] < 3:
            continue
        acc = p[..., 2:, :] - 2 * p[..., 1:-1, :] + p[..., :-2, :]
        total = total + acc.square().mean()
    return total / len(planes)


def l1_time_planes(planes: list[Tensor]) -> Tensor:
    """Pull the space-time planes towards the multiplicative identity.

    The planes compose by multiplication, so one is the "no contribution"
    value. Penalising the deviation makes *no motion* the default that has to be
    paid for, which is the correct prior for a three-second clip in which most
    of the scene -- the entire background -- genuinely does not move.
    """
    if not planes:
        return torch.zeros(())
    total = planes[0].new_zeros(())
    for p in planes:
        total = total + (p - 1.0).abs().mean()
    return total / len(planes)


def psnr(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    """Peak signal-to-noise ratio, optionally over a mask.

    Reported by the trainer as a *temporal hold-out* figure and labelled as such
    everywhere it is written down. It measures motion interpolation on frames
    from the one camera that was there; it is not a novel-view number and must
    never be tabulated beside one (decision D9).
    """
    diff = (pred - target).square()
    mse = diff.mean() if mask is None else (diff * mask).sum() / (mask.sum() * 3).clamp_min(1e-8)
    return -10.0 * torch.log10(mse.clamp_min(1e-12))
