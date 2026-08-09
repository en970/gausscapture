"""The 4D training loop, its checkpoints, and the scene it exports.

Everything the loop needs is injected: the rasterizer, the densification
strategy, the field and the schedule. That is not abstraction for its own sake.
The only part of this file that cannot run on a laptop is the CUDA kernel, so
making the kernel an argument is what turns "the trainer is untestable until
someone rents a GPU" into "the trainer is a normal unit test that finishes in
two seconds".

One number produced here needs its label attached wherever it travels. There is
no held-out *view* in a fixed-camera capture -- the training views are the only
views. ``holdout_psnr`` is measured on frames withheld from the time axis, so it
measures **motion interpolation** and nothing else. It is not comparable to a
masked mPSNR from DyCheck or N3DV, and printing it in the same table as one
would be a claim this project cannot support (decision D9).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from gausscapture.errors import PipelineStateError
from gausscapture.progress import NullProgress, Progress
from gausscapture.recon.deform import losses as L
from gausscapture.recon.deform.field import (
    KPlanesField,
    ScaffoldField,
    Skinning,
    deform_gaussians,
    node_edges,
)
from gausscapture.recon.deform.params import CanonicalParams
from gausscapture.recon.deform.raster import Rasterizer, ReferenceRasterizer
from gausscapture.recon.deform.schedule import (
    DensificationStrategy,
    ReferenceMcmcStrategy,
    ScheduleConfig,
    TrainingSchedule,
)

#: Format tag written into every checkpoint and every exported scene, so a file
#: from an older run fails loudly instead of loading with silently different
#: semantics.
SCENE4D_VERSION = 1


@dataclass
class FixedCameraClip:
    """The observations: one camera, many times.

    ``viewmat`` and ``intrinsics`` are singular on purpose. Per-frame solved
    poses on a stationary camera are estimation noise, and feeding that to a
    deformation trainer converts camera jitter into subject motion that never
    happened -- so the preparation stage assigns every hold frame the same pose
    and this class has nowhere to put a second one.
    """

    images: Tensor
    viewmat: Tensor
    intrinsics: Tensor
    times: Tensor
    weights: Tensor | None = None

    def __post_init__(self) -> None:
        if self.images.ndim != 4 or self.images.shape[-1] != 3:
            raise ValueError(f"images must be (T, H, W, 3), got {tuple(self.images.shape)}")
        if self.images.shape[0] != self.times.shape[0]:
            raise ValueError("images and times disagree on the frame count")
        if tuple(self.viewmat.shape) != (4, 4):
            raise ValueError("viewmat must be (4, 4)")
        if tuple(self.intrinsics.shape) != (3, 3):
            raise ValueError("intrinsics must be (3, 3)")

    @property
    def n_frames(self) -> int:
        return int(self.images.shape[0])

    @property
    def height(self) -> int:
        return int(self.images.shape[1])

    @property
    def width(self) -> int:
        return int(self.images.shape[2])

    def to(self, device: torch.device | str) -> FixedCameraClip:
        return FixedCameraClip(
            images=self.images.to(device),
            viewmat=self.viewmat.to(device),
            intrinsics=self.intrinsics.to(device),
            times=self.times.to(device),
            weights=None if self.weights is None else self.weights.to(device),
        )


@dataclass
class Train4DConfig:
    """Loss weights and the few knobs that are not timing."""

    schedule: ScheduleConfig = dataclass_field(default_factory=ScheduleConfig)
    lambda_ssim: float = 0.2
    lambda_opacity: float = 1e-2
    lambda_scale: float = 1e-2
    lambda_arap: float = 1e-1
    lambda_temporal: float = 5e-2
    lambda_plane_tv: float = 1e-4
    lambda_time_smooth: float = 1e-3
    lambda_time_l1: float = 1e-4
    field_lr: float = 1e-3
    skin_k: int = 4
    #: Frames withheld from training. Stride 8 keeps 87.5% of a short clip in
    #: the fit while still leaving enough held-out frames to average over.
    holdout_stride: int = 8
    #: A rebind is rejected when it moves deformed positions by more than this
    #: multiple of the median Gaussian scale. Recomputing the binding from
    #: canonical position is right in principle (D10) but untested on real
    #: motion, and a rebind that jumps is worse than a stale one.
    rebind_max_rms_ratio: float = 0.5
    #: Write a resumable checkpoint every this many steps; 0 disables it. The
    #: shipped run is 15,000 iterations on a Colab runtime that is recycled when
    #: it feels like it, so "the run is lost" is the ordinary outcome rather than
    #: the exceptional one -- and the checkpoint is the whole answer to it.
    checkpoint_every: int = 1_000
    seed: int = 0


@dataclass
class TrainMetrics:
    """What the run measured, with the labels that keep it honest."""

    steps: int = 0
    final_loss: float = float("nan")
    train_psnr: float = float("nan")
    #: Motion interpolation on withheld *frames*, from the one camera that was
    #: there. Never a novel-view number.
    holdout_psnr_temporal: float = float("nan")
    holdout_frames: int = 0
    gaussians: int = 0
    dynamic_gaussians: int = 0
    relocated: int = 0
    grown: int = 0
    rebinds: int = 0
    rebinds_rejected: int = 0
    max_rebind_rms: float = 0.0
    loss_history: list[float] = dataclass_field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainResult:
    params: CanonicalParams
    field: ScaffoldField
    skinning: Skinning
    metrics: TrainMetrics


def _regularise_field(
    fld: ScaffoldField,
    times: Tensor,
    edges: Tensor,
    cfg: Train4DConfig,
) -> Tensor:
    """ARAP, temporal smoothness, and the K-Planes grid priors if present."""
    quat, trans = fld(times)
    total = cfg.lambda_arap * L.arap_loss(fld.node_rest, quat, trans, edges)
    total = total + cfg.lambda_temporal * L.trajectory_smoothness(quat, trans)
    if isinstance(fld, KPlanesField):
        total = total + cfg.lambda_plane_tv * L.plane_tv_loss(fld.spatial_planes())
        total = total + cfg.lambda_time_smooth * L.time_smoothness_loss(fld.spacetime_planes())
        total = total + cfg.lambda_time_l1 * L.l1_time_planes(fld.spacetime_planes())
    return total


def _render(
    params: CanonicalParams,
    fld: ScaffoldField,
    skinning: Skinning,
    clip: FixedCameraClip,
    t: Tensor,
    rasterizer: Rasterizer,
    deform: bool,
):
    if deform:
        quat, trans = fld(t)
        means, quats = deform_gaussians(
            params.params["means"],
            params.normalised_quats(),
            quat[0],
            trans[0],
            fld.node_rest,
            skinning,
            dynamic=params.params["dynamic"],
        )
    else:
        means, quats = params.params["means"], params.normalised_quats()
    return rasterizer.render(
        means=means,
        quats=quats,
        scales=params.scales(),
        opacities=params.opacities(),
        colors=params.colors(),
        viewmat=clip.viewmat,
        intrinsics=clip.intrinsics,
        width=clip.width,
        height=clip.height,
    )


def split_holdout(n_frames: int, stride: int) -> tuple[list[int], list[int]]:
    """Every ``stride``-th frame is withheld, except the endpoints.

    The endpoints stay in training because the field is anchored at ``t = 0``
    and extrapolating past the last keyframe is not interpolation; withholding
    either would measure something the clip never claimed to do.
    """
    if stride <= 1:
        return list(range(n_frames)), []
    holdout = [i for i in range(1, n_frames - 1) if i % stride == 0]
    train = [i for i in range(n_frames) if i not in set(holdout)]
    return train, holdout


def train(
    clip: FixedCameraClip,
    params: CanonicalParams,
    fld: ScaffoldField,
    rasterizer: Rasterizer | None = None,
    strategy: DensificationStrategy | None = None,
    config: Train4DConfig | None = None,
    progress: Progress | None = None,
    checkpoint: Path | None = None,
    resume: bool = False,
) -> TrainResult:
    """Fit the canonical splat and the scaffold to one fixed-camera clip.

    ``checkpoint`` names a file written every ``config.checkpoint_every`` steps
    and once more at the end. With ``resume`` it is also *read*: an existing
    checkpoint sets the starting step, and the loop picks up from there. That is
    not a nicety on a 15,000-iteration run whose host is a Colab runtime with an
    idle timer.
    """
    cfg = config or Train4DConfig()
    progress = progress or NullProgress()
    rasterizer = rasterizer or ReferenceRasterizer()
    strategy = strategy or ReferenceMcmcStrategy()
    schedule = TrainingSchedule(cfg.schedule)
    metrics = TrainMetrics()

    if clip.n_frames < 2:
        raise PipelineStateError("a 4D fit needs at least two frames")

    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
    first_step = 0
    if resume and checkpoint is not None and Path(checkpoint).exists():
        first_step = load_checkpoint(checkpoint, params, fld)
        progress.log(f"resuming from {checkpoint} at step {first_step}")
    skinning = params.rebind(fld.node_rest, k=cfg.skin_k)
    edges = node_edges(fld.node_rest)
    field_opt = torch.optim.Adam(fld.parameters(), lr=cfg.field_lr)
    train_idx, holdout_idx = split_holdout(clip.n_frames, cfg.holdout_stride)
    if not train_idx:
        raise PipelineStateError("holdout stride left no training frames")

    total = schedule.total_iters
    if first_step >= total:
        raise PipelineStateError(
            f"the checkpoint at {checkpoint} is already at step {first_step} of a "
            f"{total}-step schedule; there is nothing left to resume"
        )
    for step in range(first_step, total):
        deform = schedule.deformation_enabled(step)
        lr_scale = schedule.means_lr_scale(step)
        params.set_means_lr_scale(lr_scale)

        pick = train_idx[int(torch.randint(len(train_idx), (1,), generator=generator))]
        out = _render(params, fld, skinning, clip, clip.times[pick : pick + 1], rasterizer, deform)
        loss = L.photometric_loss(
            out.image, clip.images[pick], lambda_ssim=cfg.lambda_ssim, weights=clip.weights
        )
        loss = loss + cfg.lambda_opacity * L.opacity_regulariser(params.opacities())
        loss = loss + cfg.lambda_scale * L.scale_regulariser(params.scales())
        if deform:
            loss = loss + _regularise_field(fld, clip.times, edges, cfg)

        params.zero_grad()
        field_opt.zero_grad(set_to_none=True)
        loss.backward()
        params.step()
        if deform:
            field_opt.step()

        if schedule.should_densify(step):
            report = strategy.refine(params, cfg.schedule.cap_max)
            metrics.relocated += int(report.get("relocated", 0))
            metrics.grown += int(report.get("grown", 0))
            skinning = _rebind(params, fld, skinning, report, clip, cfg, metrics)

        strategy.add_noise(
            params,
            lr=params.lrs["means"] * params.scene_scale * lr_scale,
            scale=schedule.noise_scale(step),
        )

        metrics.loss_history.append(float(loss.detach()))
        periodic = cfg.checkpoint_every > 0 and (step + 1) % cfg.checkpoint_every == 0
        if checkpoint is not None and periodic:
            save_checkpoint(checkpoint, params, fld, step + 1, cfg)
        if total >= 10 and step % max(total // 10, 1) == 0:
            progress.update(int(100 * step / total), f"{schedule.stage(step)} step {step}")

    if checkpoint is not None:
        save_checkpoint(checkpoint, params, fld, total, cfg)
    metrics.steps = total
    metrics.final_loss = metrics.loss_history[-1] if metrics.loss_history else float("nan")
    metrics.gaussians = params.n
    metrics.dynamic_gaussians = int(params.dynamic_mask().sum())
    _evaluate(clip, params, fld, skinning, rasterizer, train_idx, holdout_idx, metrics)
    progress.update(100, "training complete")
    return TrainResult(params=params, field=fld, skinning=skinning, metrics=metrics)


def _rebind(
    params: CanonicalParams,
    fld: ScaffoldField,
    previous: Skinning,
    report: dict[str, object],
    clip: FixedCameraClip,
    cfg: Train4DConfig,
    metrics: TrainMetrics,
) -> Skinning:
    """Recompute the binding, but only accept it if it does not jump.

    Decision D10 rebinds from canonical positions rather than copying a parent's
    binding across a split, which is right: a copied binding makes both children
    move identically, and a relocated Gaussian lands somewhere its old binding
    knows nothing about. But rebinding is untested on real motion, so the change
    it causes is *measured* against the carried-forward alternative and rejected
    when it exceeds a fraction of the median Gaussian scale.
    """
    metrics.rebinds += 1
    rebound = params.rebind(fld.node_rest, k=cfg.skin_k)
    parent = report.get("parent")
    if not isinstance(parent, Tensor) or parent.shape[0] != params.n:
        return rebound

    carried = Skinning(previous.idx[parent], previous.weights[parent])
    with torch.no_grad():
        mid = clip.times[clip.n_frames // 2].reshape(1)
        quat, trans = fld(mid)
        args = (fld.node_rest, params.params["dynamic"])
        old_pos, _ = deform_gaussians(
            params.params["means"], params.normalised_quats(), quat[0], trans[0], args[0], carried,
            dynamic=args[1],
        )
        new_pos, _ = deform_gaussians(
            params.params["means"], params.normalised_quats(), quat[0], trans[0], args[0], rebound,
            dynamic=args[1],
        )
        rms = float((new_pos - old_pos).square().sum(-1).mean().sqrt())
        median_scale = float(params.scales().median())
    metrics.max_rebind_rms = max(metrics.max_rebind_rms, rms)
    if rms > cfg.rebind_max_rms_ratio * max(median_scale, 1e-8):
        metrics.rebinds_rejected += 1
        return carried
    return rebound


@torch.no_grad()
def _evaluate(
    clip: FixedCameraClip,
    params: CanonicalParams,
    fld: ScaffoldField,
    skinning: Skinning,
    rasterizer: Rasterizer,
    train_idx: list[int],
    holdout_idx: list[int],
    metrics: TrainMetrics,
) -> None:
    def mean_psnr(indices: list[int]) -> float:
        if not indices:
            return float("nan")
        acc = 0.0
        for i in indices:
            out = _render(params, fld, skinning, clip, clip.times[i : i + 1], rasterizer, True)
            acc += float(L.psnr(out.image, clip.images[i]))
        return acc / len(indices)

    metrics.train_psnr = mean_psnr(train_idx)
    metrics.holdout_psnr_temporal = mean_psnr(holdout_idx)
    metrics.holdout_frames = len(holdout_idx)


# --------------------------------------------------------------------------
# Checkpoints and export
# --------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    params: CanonicalParams,
    fld: ScaffoldField,
    step: int,
    config: Train4DConfig | None = None,
) -> Path:
    """Write a resumable checkpoint.

    The skinning is deliberately absent: it is derived from canonical positions
    and the node rest positions, both of which are in here, so storing it would
    create a second source of truth that a resumed run could silently disagree
    with.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": SCENE4D_VERSION,
        "step": int(step),
        "params": params.state_dict(),
        "field": {k: v.cpu() for k, v in fld.state_dict().items()},
        "field_class": type(fld).__name__,
        "config": json.dumps(asdict(config)) if config is not None else "",
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: Path, params: CanonicalParams, fld: ScaffoldField) -> int:
    """Restore a checkpoint in place, returning the step it was written at."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if int(payload.get("version", -1)) != SCENE4D_VERSION:
        raise PipelineStateError(
            f"checkpoint version {payload.get('version')} does not match {SCENE4D_VERSION}"
        )
    if payload.get("field_class") != type(fld).__name__:
        raise PipelineStateError(
            f"checkpoint holds a {payload.get('field_class')}, not a {type(fld).__name__}"
        )
    params.load_state_dict(payload["params"])
    fld.load_state_dict(payload["field"])
    return int(payload["step"])


def export_scene4d(
    path: Path,
    params: CanonicalParams,
    fld: ScaffoldField,
    skinning: Skinning,
    times: Tensor,
    metrics: TrainMetrics | None = None,
    cone: dict[str, float] | None = None,
) -> Path:
    """Write ``scene4d.npz``, the single input to ``export/scene4d.py``.

    Dynamic Gaussians are written first so the container needs one integer
    rather than a per-Gaussian flag, and the skinning arrays cover exactly that
    leading block -- static Gaussians receive identity deformation and have no
    binding to store (decision D11).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = params.to_arrays()
    dynamic = torch.from_numpy(arrays["dynamic"])
    order = torch.cat([torch.nonzero(dynamic).reshape(-1), torch.nonzero(~dynamic).reshape(-1)])
    n_dynamic = int(dynamic.sum())

    quat, trans = fld.bake(times)
    idx = order.numpy()
    payload = {
        "version": np.int32(SCENE4D_VERSION),
        "means": arrays["means"][idx],
        "scales": arrays["scales"][idx],
        "quats": arrays["quats"][idx],
        "opacities": arrays["opacities"][idx],
        "colors": arrays["colors"][idx],
        "dynamic_count": np.int32(n_dynamic),
        "node_rest": fld.node_rest.cpu().numpy().astype(np.float32),
        "node_quat": quat.cpu().numpy().astype(np.float32),
        "node_trans": trans.cpu().numpy().astype(np.float32),
        "skin_idx": skinning.idx[order[:n_dynamic]].cpu().numpy().astype(np.int32),
        "skin_weights": skinning.weights[order[:n_dynamic]].cpu().numpy().astype(np.float32),
        "frame_times": times.detach().cpu().numpy().astype(np.float32),
        "meta": json.dumps(
            {
                "version": SCENE4D_VERSION,
                "metrics": metrics.to_dict() if metrics else {},
                "cone": cone or {},
                "holdout_label": "temporal interpolation, single camera; not novel view",
                "sh_degree": 0,
            }
        ),
    }
    np.savez_compressed(path, **payload)
    return path


def scene_scale_from_points(points: Tensor) -> float:
    """Median radius about the centroid, the unit the position LR is expressed in."""
    if points.shape[0] == 0:
        return 1.0
    radius = (points - points.mean(dim=0)).norm(dim=1)
    value = float(radius.median())
    return value if math.isfinite(value) and value > 0 else 1.0
