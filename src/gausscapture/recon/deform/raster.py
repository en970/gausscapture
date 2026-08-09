"""The rasterizer seam: one Protocol, a CUDA implementation and a CPU reference.

``recon/__init__.py`` states that GaussCapture implements no rasterizer and
drives external trainers across a subprocess boundary. There is no licence-clean
external 4D trainer to drive, so that policy cannot be honoured literally in the
4D path. It is honoured in substance here: the only rasterizer we would ever
ship is a *call* into ``gsplat`` (Apache-2.0), pip-installed and never vendored,
and it sits behind :class:`Rasterizer` so that nothing GPU-specific is a hard
import.

The second implementation is the reason the seam exists at all. The laptop this
project is developed on is an M2 Air with no CUDA, so if the training loop could
only run against a CUDA kernel then the loop, the field, the schedule, the
losses and the parameter surgery -- which is all of the new code except the
kernel -- would be untestable. :class:`ReferenceRasterizer` composites Gaussians
densely and differentiably in a few lines of PyTorch. It is far too slow for a
real scene and exactly fast enough for a 32x32 test image, which is the point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor

from gausscapture.errors import DependencyMissingError
from gausscapture.recon.deform.so3 import quat_to_matrix

#: Isotropic blur added to every projected 2D covariance, in pixels squared.
#: Without it a Gaussian that projects to less than a pixel has a near-singular
#: 2D covariance, whose inverse blows up and whose gradient is noise. 0.3 is the
#: value the 3DGS literature settled on and gsplat uses; matching it keeps the
#: reference and the CUDA path comparable.
DILATION = 0.3


@dataclass
class RenderOutput:
    """One rendered view plus whatever the backend wants to hand a densifier."""

    image: Tensor
    alpha: Tensor
    info: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Rasterizer(Protocol):
    """Renders one view of a Gaussian set.

    ``viewmat`` is world-to-camera, OpenCV convention (``+x`` right, ``+y``
    down, ``+z`` forward), matching what ``pose/colmap.py`` already produces.
    ``colors`` is linear RGB, not spherical harmonics: a fixed camera cannot
    constrain view-dependent colour, so the 4D path trains at ``sh_degree=0``
    and the decode happens before the call.
    """

    name: str

    def render(
        self,
        means: Tensor,
        quats: Tensor,
        scales: Tensor,
        opacities: Tensor,
        colors: Tensor,
        viewmat: Tensor,
        intrinsics: Tensor,
        width: int,
        height: int,
    ) -> RenderOutput:
        """Render a ``(height, width, 3)`` image and its alpha."""


class ReferenceRasterizer:
    """Dense, differentiable, CPU alpha compositor. For tests, not for scenes.

    Every Gaussian is evaluated at every pixel, so cost is ``O(N*H*W)`` with no
    tiling and no culling. At the 32x32 and a few hundred Gaussians the test
    suite uses that is microseconds; at 400k Gaussians and 1080p it would be
    absurd, and it is never asked to do that.
    """

    name = "reference"

    def __init__(self, background: float = 0.0, near: float = 0.01) -> None:
        self.background = float(background)
        self.near = float(near)

    def render(
        self,
        means: Tensor,
        quats: Tensor,
        scales: Tensor,
        opacities: Tensor,
        colors: Tensor,
        viewmat: Tensor,
        intrinsics: Tensor,
        width: int,
        height: int,
    ) -> RenderOutput:
        rot = viewmat[:3, :3]
        cam = means @ rot.T + viewmat[:3, 3]
        z = cam[:, 2].clamp_min(self.near)
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        uv = torch.stack([fx * cam[:, 0] / z + cx, fy * cam[:, 1] / z + cy], dim=-1)

        # 3D covariance, then the standard first-order projection: the Jacobian
        # of the pinhole map evaluated at each Gaussian centre.
        r = quat_to_matrix(quats)
        rs = r * scales.unsqueeze(-2)
        cov3d = rs @ rs.transpose(-1, -2)
        j = torch.zeros(means.shape[0], 2, 3, dtype=means.dtype, device=means.device)
        j[:, 0, 0] = fx / z
        j[:, 0, 2] = -fx * cam[:, 0] / z.square()
        j[:, 1, 1] = fy / z
        j[:, 1, 2] = -fy * cam[:, 1] / z.square()
        w = j @ rot
        cov2d = w @ cov3d @ w.transpose(-1, -2)
        eye = torch.eye(2, dtype=means.dtype, device=means.device) * DILATION
        cov2d = cov2d + eye

        det = (cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0]).clamp_min(1e-12)
        inv = torch.stack(
            [cov2d[:, 1, 1], -cov2d[:, 0, 1], -cov2d[:, 1, 0], cov2d[:, 0, 0]], dim=-1
        ).reshape(-1, 2, 2) / det.reshape(-1, 1, 1)

        ys, xs = torch.meshgrid(
            torch.arange(height, dtype=means.dtype, device=means.device),
            torch.arange(width, dtype=means.dtype, device=means.device),
            indexing="ij",
        )
        px = torch.stack([xs, ys], dim=-1).reshape(1, -1, 2)
        d = px - uv.reshape(-1, 1, 2)
        power = -0.5 * torch.einsum("npi,nij,npj->np", d, inv, d)
        alpha = (opacities.reshape(-1, 1) * torch.exp(power.clamp(max=0.0))).clamp(0.0, 0.999)

        # Front-to-back over-compositing. Sorting by depth is what makes this a
        # rasterizer rather than an unordered sum, and it is also where a
        # Gaussian behind the camera would corrupt the result -- hence the
        # explicit near-plane cull below.
        order = torch.argsort(cam[:, 2])
        alpha = alpha[order]
        rgb = colors[order]
        visible = (cam[order, 2] > self.near).to(alpha.dtype).reshape(-1, 1)
        alpha = alpha * visible

        transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=0)
        transmittance = torch.cat(
            [torch.ones_like(transmittance[:1]), transmittance[:-1]], dim=0
        )
        weight = alpha * transmittance
        image = torch.einsum("np,nc->pc", weight, rgb).reshape(height, width, 3)
        acc = weight.sum(dim=0).reshape(height, width, 1)
        image = image + (1.0 - acc) * self.background
        return RenderOutput(image=image, alpha=acc, info={"means2d": uv, "depths": cam[:, 2]})


class GsplatRasterizer:
    """Thin wrapper over ``gsplat.rasterization``.

    ``gsplat`` is imported inside :meth:`render`, not at module scope, so that
    importing ``gausscapture.recon.deform`` on a machine with no CUDA and no
    gsplat still works -- which is the machine this pipeline is prepared on.
    """

    name = "gsplat"

    def __init__(self, background: float = 0.0, packed: bool = True) -> None:
        self.background = float(background)
        self.packed = bool(packed)

    @staticmethod
    def available() -> bool:
        try:
            import gsplat  # noqa: F401
        except ImportError:
            return False
        return True

    def render(
        self,
        means: Tensor,
        quats: Tensor,
        scales: Tensor,
        opacities: Tensor,
        colors: Tensor,
        viewmat: Tensor,
        intrinsics: Tensor,
        width: int,
        height: int,
    ) -> RenderOutput:
        try:
            from gsplat import rasterization
        except ImportError as exc:  # pragma: no cover - exercised only with the extra
            raise DependencyMissingError(
                "gsplat is required for GPU rasterisation. "
                "Install the training extra: pip install 'gausscapture[train4d]'"
            ) from exc

        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat.unsqueeze(0),
            Ks=intrinsics.unsqueeze(0),
            width=width,
            height=height,
            packed=self.packed,
            backgrounds=means.new_full((1, 3), self.background),
        )
        return RenderOutput(
            image=render_colors[0], alpha=render_alphas[0], info=dict(info)
        )
