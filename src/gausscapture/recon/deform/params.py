"""Canonical Gaussian parameters, their optimisers, and parameter surgery.

Two structural rules govern this module, and both come from how ``gsplat``'s
densification strategies work rather than from taste.

*One optimiser per key.* ``Strategy.check_sanity`` asserts that the parameter
dict and the optimiser dict have identical key sets and that each optimiser owns
exactly one parameter group holding exactly one tensor. A single Adam over all
five tensors -- the obvious thing to write -- fails that check, and fails it at
the first refine step rather than at construction.

*The deformation parameters are not in this dict.* Parameter surgery iterates
every key and concatenates along dim 0. A ``[64, 128]`` MLP weight put in here
becomes ``[65, 128]`` after one duplication, silently, and the failure surfaces
thousands of steps later as a shape error in an unrelated place. The field owns
its own parameters and its own optimiser; see :mod:`gausscapture.recon.deform.field`.

What *does* ride in the dict is the per-Gaussian ``dynamic`` label, as a
non-trainable float tensor with a zero-learning-rate optimiser so it satisfies
the sanity check. It has to be carried because it is not recoverable: it was
derived from a mask reprojection during preparation, and a Gaussian created by
MCMC relocation has no mask to consult. Skinning is the opposite case -- it *is*
recoverable, from canonical position alone -- so it is recomputed by
:meth:`CanonicalParams.rebind` at every refine boundary rather than copied
across a split, which is decision D10.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from gausscapture.errors import PipelineStateError
from gausscapture.recon.deform.field import Skinning, build_skinning

#: Zeroth-order spherical-harmonic normalisation. The 4D path trains at
#: ``sh_degree = 0`` throughout, so this single constant is the whole colour
#: decode: a fixed camera cannot constrain view-dependent colour, and higher
#: bands would become a channel for memorising per-frame appearance as fake
#: view-dependence.
SH_C0 = 0.28209479177387814

#: Trainable per-Gaussian tensors, in the order the exporter expects them.
CANONICAL_KEYS: tuple[str, ...] = ("means", "log_scales", "quats", "logit_opacities", "sh0")

#: Carried through surgery but never optimised. See the module docstring.
LABEL_KEYS: tuple[str, ...] = ("dynamic",)

#: Base learning rates, following the published gsplat defaults. ``means`` is
#: additionally multiplied by the scene scale, because a position step that is
#: right for a scene measured in metres is 1000x too large for one in
#: millimetres and the optimiser has no other way to learn the unit.
DEFAULT_LRS: dict[str, float] = {
    "means": 1.6e-4,
    "log_scales": 5e-3,
    "quats": 1e-3,
    "logit_opacities": 5e-2,
    "sh0": 2.5e-3,
}


class CanonicalParams:
    """The canonical splat: five trainable tensors, one label, one Adam each."""

    def __init__(
        self,
        means: Tensor,
        log_scales: Tensor,
        quats: Tensor,
        logit_opacities: Tensor,
        sh0: Tensor,
        dynamic: Tensor | None = None,
        lrs: dict[str, float] | None = None,
        scene_scale: float = 1.0,
    ) -> None:
        n = means.shape[0]
        for name, t, shape in (
            ("means", means, (n, 3)),
            ("log_scales", log_scales, (n, 3)),
            ("quats", quats, (n, 4)),
            ("logit_opacities", logit_opacities, (n,)),
            ("sh0", sh0, (n, 3)),
        ):
            if tuple(t.shape) != shape:
                raise ValueError(f"{name} must be {shape}, got {tuple(t.shape)}")
        if dynamic is None:
            dynamic = torch.ones(n, dtype=means.dtype, device=means.device)
        if tuple(dynamic.shape) != (n,):
            raise ValueError(f"dynamic must be ({n},), got {tuple(dynamic.shape)}")

        self.scene_scale = float(scene_scale)
        self.lrs = {**DEFAULT_LRS, **(lrs or {})}
        self.params = nn.ParameterDict(
            {
                "means": nn.Parameter(means.float().clone()),
                "log_scales": nn.Parameter(log_scales.float().clone()),
                "quats": nn.Parameter(quats.float().clone()),
                "logit_opacities": nn.Parameter(logit_opacities.float().clone()),
                "sh0": nn.Parameter(sh0.float().clone()),
                "dynamic": nn.Parameter(dynamic.float().clone(), requires_grad=False),
            }
        )
        self.optimizers: dict[str, torch.optim.Adam] = {}
        self._build_optimizers()

    # -- construction ----------------------------------------------------

    def _lr(self, name: str) -> float:
        if name in LABEL_KEYS:
            return 0.0
        lr = self.lrs[name]
        return lr * self.scene_scale if name == "means" else lr

    def _build_optimizers(self) -> None:
        self.optimizers = {
            name: torch.optim.Adam([{"params": [p], "lr": self._lr(name), "name": name}], eps=1e-15)
            for name, p in self.params.items()
        }

    @classmethod
    def from_points(
        cls,
        points: Tensor,
        colors: Tensor,
        dynamic: Tensor | None = None,
        init_opacity: float = 0.1,
        knn_k: int = 3,
        scene_scale: float | None = None,
        lrs: dict[str, float] | None = None,
    ) -> CanonicalParams:
        """Initialise from a sparse point cloud, exactly as preparation hands it over."""
        from gausscapture.recon.deform.knn import init_log_scales

        n = points.shape[0]
        if n == 0:
            raise PipelineStateError("cannot initialise a splat from zero points")
        quats = torch.zeros(n, 4, dtype=points.dtype, device=points.device)
        quats[:, 0] = 1.0
        opacity = torch.full((n,), float(init_opacity), device=points.device)
        logit = torch.log(opacity / (1.0 - opacity))
        if scene_scale is None:
            scene_scale = float((points - points.mean(dim=0)).norm(dim=1).median().clamp_min(1e-6))
        return cls(
            means=points.float(),
            log_scales=init_log_scales(points.float(), k=knn_k),
            quats=quats,
            logit_opacities=logit,
            sh0=(colors.float() - 0.5) / SH_C0,
            dynamic=dynamic,
            lrs=lrs,
            scene_scale=scene_scale,
        )

    # -- accessors -------------------------------------------------------

    @property
    def n(self) -> int:
        return int(self.params["means"].shape[0])

    @property
    def device(self) -> torch.device:
        return self.params["means"].device

    def scales(self) -> Tensor:
        return torch.exp(self.params["log_scales"])

    def opacities(self) -> Tensor:
        return torch.sigmoid(self.params["logit_opacities"])

    def colors(self) -> Tensor:
        return SH_C0 * self.params["sh0"] + 0.5

    def normalised_quats(self) -> Tensor:
        q = self.params["quats"]
        return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def dynamic_mask(self) -> Tensor:
        return self.params["dynamic"] > 0.5

    def to(self, device: torch.device | str) -> CanonicalParams:
        self.params.to(device)
        self._build_optimizers()
        return self

    # -- optimisation ----------------------------------------------------

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for opt in self.optimizers.values():
            opt.step()

    def set_means_lr_scale(self, scale: float) -> None:
        """Rescale only the position learning rate.

        Positions are the one parameter whose right step size changes over
        training: early on they have to migrate across the scene, and late on a
        step of the same size undoes the fit. Everything else keeps its rate.
        """
        for group in self.optimizers["means"].param_groups:
            group["lr"] = self._lr("means") * float(scale)

    # -- surgery ---------------------------------------------------------

    def _replace(self, name: str, tensor: Tensor, state_fn) -> None:
        """Swap one parameter and map its Adam moments through the same index op."""
        old = self.params[name]
        opt = self.optimizers[name]
        state = opt.state.get(old, None)
        new = nn.Parameter(tensor.detach().clone(), requires_grad=old.requires_grad)
        if state is not None:
            new_state: dict[str, Any] = {}
            for key, value in state.items():
                new_state[key] = state_fn(value) if isinstance(value, Tensor) and value.ndim else value
            del opt.state[old]
            opt.state[new] = new_state
        opt.param_groups[0]["params"] = [new]
        self.params[name] = new

    def duplicate(self, idx: Tensor) -> None:
        """Append copies of the Gaussians at ``idx``.

        Adam moments for the copies start at zero rather than being inherited.
        An inherited moment is a claim about a history the new Gaussian did not
        have, and it makes a fresh split move in lockstep with its parent for
        the first few dozen steps -- the opposite of why it was split.
        """
        idx = idx.to(self.device).long()
        for name in list(self.params.keys()):
            p = self.params[name].detach()
            grown = torch.cat([p, p[idx]], dim=0)
            self._replace(name, grown, lambda s, idx=idx: torch.cat([s, torch.zeros_like(s[idx])]))

    def prune(self, keep: Tensor) -> None:
        """Keep only the Gaussians selected by the boolean mask ``keep``."""
        keep = keep.to(self.device)
        if keep.dtype != torch.bool:
            raise ValueError("prune expects a boolean mask")
        if int(keep.sum()) == 0:
            raise PipelineStateError("pruning would remove every Gaussian")
        for name in list(self.params.keys()):
            self._replace(name, self.params[name].detach()[keep], lambda s, keep=keep: s[keep])

    def relocate(self, dead: Tensor, alive: Tensor) -> None:
        """Move the Gaussians at ``dead`` onto the ones at ``alive`` (MCMC style).

        Opacity is corrected by the two-way split rule ``o' = 1 - sqrt(1 - o)``
        so that the pair composites to the same alpha the single Gaussian did.
        Without the correction, relocation brightens the scene every time it
        fires and the photometric loss spends the next hundred steps undoing it.
        """
        dead = dead.to(self.device).long()
        alive = alive.to(self.device).long()
        if dead.numel() == 0:
            return
        with torch.no_grad():
            o = torch.sigmoid(self.params["logit_opacities"][alive])
            new_o = (1.0 - torch.sqrt((1.0 - o).clamp_min(1e-8))).clamp(1e-4, 1.0 - 1e-4)
            new_logit = torch.log(new_o / (1.0 - new_o))
            for name in ("means", "log_scales", "quats", "sh0", "dynamic"):
                self.params[name].data[dead] = self.params[name].data[alive]
            self.params["logit_opacities"].data[alive] = new_logit
            self.params["logit_opacities"].data[dead] = new_logit
        # Reset the moments of both endpoints: the relocated Gaussian is a new
        # object, and the survivor's opacity just jumped discontinuously.
        touched = torch.cat([dead, alive])
        for name, opt in self.optimizers.items():
            state = opt.state.get(self.params[name], None)
            if not state:
                continue
            for value in state.values():
                if isinstance(value, Tensor) and value.shape[:1] == (self.n,):
                    value[touched] = 0.0

    def rebind(self, node_rest: Tensor, k: int = 4, sigma: float | None = None) -> Skinning:
        """Recompute the scaffold binding from current canonical positions."""
        with torch.no_grad():
            return build_skinning(self.params["means"].detach(), node_rest, k=k, sigma=sigma)

    # -- serialisation ---------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "params": {k: v.detach().cpu() for k, v in self.params.items()},
            "optimizers": {k: v.state_dict() for k, v in self.optimizers.items()},
            "scene_scale": self.scene_scale,
            "lrs": dict(self.lrs),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.scene_scale = float(state["scene_scale"])
        self.lrs = dict(state["lrs"])
        device = self.device
        self.params = nn.ParameterDict(
            {
                k: nn.Parameter(v.to(device), requires_grad=k not in LABEL_KEYS)
                for k, v in state["params"].items()
            }
        )
        self._build_optimizers()
        for name, opt_state in state["optimizers"].items():
            self.optimizers[name].load_state_dict(opt_state)

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Detached NumPy view in the form ``export/scene4d.py`` consumes."""
        with torch.no_grad():
            return {
                "means": self.params["means"].cpu().numpy().astype(np.float32),
                "scales": self.scales().cpu().numpy().astype(np.float32),
                "quats": self.normalised_quats().cpu().numpy().astype(np.float32),
                "opacities": self.opacities().cpu().numpy().astype(np.float32),
                "colors": self.colors().clamp(0.0, 1.0).cpu().numpy().astype(np.float32),
                "dynamic": self.dynamic_mask().cpu().numpy(),
            }
