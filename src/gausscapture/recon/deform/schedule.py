"""The two-stage schedule, and the densification it turns on and off.

Stage one is *static*: the deformation field is held at the identity and MCMC
densifies the canonical splat up to a hard cap. Stage two is *deformable*: the
scaffold is released, densification stops, and the MCMC position noise is driven
to zero.

Both halves of stage two matter for a reason that is easy to get wrong. The
noise term mutates canonical means in place every step, and the field is indexed
by canonical position -- so leaving the noise on while the field is learning
means the field is chasing a target that is being randomly displaced underneath
it. And ``DefaultStrategy``-style gradient densification is worse than useless
here: it accumulates ``means2d.grad`` on *deformed* geometry and applies it to
*canonical* parameters, so a Gaussian on a moving mouth gets split because it is
moving, not because its canonical geometry is inadequate. That is runaway
densification concentrated in exactly the region a fixed camera has least
information about. MCMC's signal is opacity, which is deformation-agnostic, and
its ``cap_max`` gives a hard Gaussian budget and therefore a predictable
``.g4d`` payload (decision D3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from gausscapture.errors import GaussCaptureError
from gausscapture.recon.deform.params import CanonicalParams
from gausscapture.recon.deform.so3 import quat_to_matrix

COARSE = "coarse"
FINE = "fine"


@dataclass(frozen=True)
class ScheduleConfig:
    """Iteration budgets and the boundaries derived from them."""

    coarse_iters: int = 3_000
    fine_iters: int = 9_000
    refine_start: int = 500
    refine_every: int = 100
    #: Densification also stops at this fraction of the *total* budget, which is
    #: the DyGauBench finding. With the default split the coarse boundary is the
    #: binding one; the fraction matters when someone shortens the coarse stage.
    refine_stop_fraction: float = 0.6
    cap_max: int = 400_000
    #: Steps over which the position noise ramps from full to exactly zero once
    #: the field is released. A hard cut also works but leaves a visible step in
    #: the loss that is easy to misread as the field diverging.
    noise_decay_iters: int = 500
    #: Final position learning rate as a fraction of the initial one.
    means_lr_final_scale: float = 0.01

    def __post_init__(self) -> None:
        if self.coarse_iters < 0 or self.fine_iters < 0:
            raise GaussCaptureError("iteration counts must be non-negative")
        if self.coarse_iters + self.fine_iters == 0:
            raise GaussCaptureError("schedule has no iterations")
        if self.refine_every < 1:
            raise GaussCaptureError("refine_every must be at least 1")
        if not 0.0 < self.refine_stop_fraction <= 1.0:
            raise GaussCaptureError("refine_stop_fraction must be in (0, 1]")
        if self.cap_max < 1:
            raise GaussCaptureError("cap_max must be positive")


class TrainingSchedule:
    """Pure timing logic: what is on at step ``n``, and nothing else.

    Deliberately holds no tensors and no optimiser, so every boundary in the
    training run is a two-line unit test rather than something you have to run a
    trainer to observe.
    """

    def __init__(self, config: ScheduleConfig | None = None) -> None:
        self.config = config or ScheduleConfig()

    @property
    def total_iters(self) -> int:
        return self.config.coarse_iters + self.config.fine_iters

    @property
    def refine_stop(self) -> int:
        c = self.config
        return min(c.coarse_iters, int(c.refine_stop_fraction * self.total_iters))

    def stage(self, step: int) -> str:
        return COARSE if step < self.config.coarse_iters else FINE

    def deformation_enabled(self, step: int) -> bool:
        """The field is inert until the fine stage begins."""
        return step >= self.config.coarse_iters

    def should_densify(self, step: int) -> bool:
        c = self.config
        if step < c.refine_start or step >= self.refine_stop:
            return False
        return step % c.refine_every == 0

    def noise_scale(self, step: int) -> float:
        """Multiplier on the MCMC position noise; exactly zero once decayed."""
        c = self.config
        if step < c.coarse_iters:
            return 1.0
        if c.noise_decay_iters <= 0:
            return 0.0
        elapsed = step - c.coarse_iters
        return max(0.0, 1.0 - elapsed / c.noise_decay_iters)

    def means_lr_scale(self, step: int) -> float:
        """Exponential decay of the position learning rate over the whole run."""
        final = self.config.means_lr_final_scale
        t = min(max(step, 0), self.total_iters) / max(self.total_iters, 1)
        return math.exp(math.log(final) * t)

    def is_stage_boundary(self, step: int) -> bool:
        return step == self.config.coarse_iters and self.config.coarse_iters > 0


# --------------------------------------------------------------------------
# Densification
# --------------------------------------------------------------------------


@runtime_checkable
class DensificationStrategy(Protocol):
    """What the training loop needs from a densifier, and no more."""

    #: Recorded in the run summary, so a report says which densifier produced it.
    name: str

    def refine(self, params: CanonicalParams, cap_max: int) -> dict[str, object]:
        """Relocate dead Gaussians and grow towards ``cap_max``.

        Must report ``parent``: a ``(N_after,)`` long tensor giving, for every
        Gaussian that now exists, the index it held in the population *before*
        this call, or the index of the Gaussian it was copied from. The training
        loop needs it to build the carried-forward skinning that the rebind is
        checked against; without it a bad rebind is undetectable.
        """

    def add_noise(self, params: CanonicalParams, lr: float, scale: float) -> None:
        """Perturb canonical means, scaled by ``scale`` (zero means do nothing)."""


@dataclass
class ReferenceMcmcStrategy:
    """3DGS-MCMC relocation and growth, in plain PyTorch.

    This is the densifier the CPU tests run, and it is a faithful-enough
    reimplementation of the published rules to be worth using on a GPU too:
    relocate Gaussians whose opacity has collapsed onto ones that survived, grow
    the population by sampling proportionally to opacity, and inject
    covariance-shaped noise into positions so the sampler explores.

    "Reference" names its role in the rasterizer seam, not a limitation: every
    operation here is an ordinary torch op on ``params``' own tensors, so this is
    the densifier a CUDA run uses too. Unlike the rasterizer there is no CUDA
    kernel to defer to that would change an answer -- see
    :func:`gsplat_strategy_available` for the seam if that ever stops being
    true.
    """

    name = "reference-mcmc"
    min_opacity: float = 5e-3
    growth_factor: float = 0.05
    noise_lr: float = 5e5
    #: Steepness of the opacity gate on the noise. Sharp on purpose: noise is
    #: meant to move Gaussians that are about to be recycled anyway, not to
    #: shake ones that are currently explaining pixels.
    opacity_gate_k: float = 100.0
    generator: torch.Generator | None = None

    def refine(self, params: CanonicalParams, cap_max: int) -> dict[str, object]:
        with torch.no_grad():
            parent = torch.arange(params.n, device=params.device)
            opacities = params.opacities()
            dead = torch.nonzero(opacities < self.min_opacity, as_tuple=False).reshape(-1)
            alive = torch.nonzero(opacities >= self.min_opacity, as_tuple=False).reshape(-1)
            relocated = 0
            if dead.numel() and alive.numel():
                pick = alive[self._sample(opacities[alive], dead.numel())]
                params.relocate(dead, pick)
                parent[dead] = pick
                relocated = int(dead.numel())

            budget = max(0, cap_max - params.n)
            grown = min(budget, int(params.n * self.growth_factor))
            if grown > 0:
                idx = self._sample(params.opacities(), grown)
                params.duplicate(idx)
                self._halve_opacity(params, idx, grown)
                parent = torch.cat([parent, parent[idx]])
        return {"relocated": relocated, "grown": int(grown), "parent": parent}

    def _sample(self, probs: Tensor, count: int) -> Tensor:
        weights = probs.clamp_min(1e-12)
        return torch.multinomial(
            weights, count, replacement=True, generator=self.generator
        )

    @staticmethod
    def _halve_opacity(params: CanonicalParams, idx: Tensor, grown: int) -> None:
        """Split opacity between parent and child so the pair composites the same."""
        logit = params.params["logit_opacities"].data
        n_new = logit.shape[0]
        child = torch.arange(n_new - grown, n_new, device=logit.device)
        o = torch.sigmoid(logit[idx])
        new_o = (1.0 - torch.sqrt((1.0 - o).clamp_min(1e-8))).clamp(1e-4, 1.0 - 1e-4)
        new_logit = torch.log(new_o / (1.0 - new_o))
        logit[idx] = new_logit
        logit[child] = new_logit

    def add_noise(self, params: CanonicalParams, lr: float, scale: float) -> None:
        if scale <= 0.0:
            return
        with torch.no_grad():
            opacities = params.opacities()
            gate = torch.sigmoid(-self.opacity_gate_k * (opacities - self.min_opacity))
            rot = quat_to_matrix(params.normalised_quats())
            cov_sqrt = rot * params.scales().unsqueeze(-2)
            eps = torch.randn(
                params.n, 3, device=params.device, generator=self.generator
            )
            noise = torch.einsum("nij,nj->ni", cov_sqrt, eps)
            params.params["means"].data += noise * gate.unsqueeze(-1) * (
                self.noise_lr * lr * scale
            )


def gsplat_strategy_available() -> bool:
    """Whether ``gsplat.strategy.MCMCStrategy`` could be constructed here.

    The CUDA densifier is deliberately *not* wrapped in this module. Wrapping it
    would mean shipping an adapter whose only correctness evidence is that it
    imports, on a laptop with no GPU to run it against; the honest place for that
    glue is beside a machine that can execute it. :class:`CanonicalParams`
    already satisfies ``Strategy.check_sanity`` -- one optimiser per key, one
    parameter per group -- so ``MCMCStrategy`` can be driven directly against
    ``params.params`` and ``params.optimizers`` when that step happens, and
    :class:`DensificationStrategy` is the interface it has to present.
    """
    try:
        from gsplat.strategy import MCMCStrategy  # noqa: F401
    except ImportError:
        return False
    return True
