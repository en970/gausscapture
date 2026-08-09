"""One command's worth of training: a ``dataset4d.zip`` in, a ``scene4d.npz`` out.

:mod:`gausscapture.recon.deform.train4d` holds the loop, and it takes tensors --
a clip, a parameter set, a field, a rasterizer. That is the right shape for the
thing being tested, and the wrong shape for the thing being run, because nothing
in it knows how to open an archive, and nothing in it knows what the capture was
evidence for.

This module is the join. It unpacks the archive, materialises the clip and the
field from the initialisation preparation wrote, runs the loop, and then does the
one thing the loop cannot: it stamps the checkpoint with the *measurements* --
the cone the arc actually swept, the pose the take was shot from, and whether
that pose was ever verified. Those were measured on the laptop during
preparation and cannot be recovered on the GPU host, so if they did not travel
through the checkpoint the exported ``.g4d`` would have to guess them, and the
viewer would end up claiming a cone nobody measured.

The training loop is imported inside the function rather than at module scope.
Preparation, packaging and export all run on a machine that has no reason to
install torch, and an import at the top of this file would make ``gausscapture
export4d`` fail on it.
"""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from gausscapture.errors import (
    CaptureFormatError,
    DependencyMissingError,
    PipelineStateError,
)
from gausscapture.progress import NullProgress, Progress

#: Arrays the exporter reads out of a checkpoint that the training loop does not
#: know how to produce. Written down here because this is the only module that
#: writes them, and :mod:`gausscapture.export.checkpoint4d` is the only one that
#: reads them.
CAPTURE_PROVENANCE_KEYS = (
    "fps",
    "cone_azimuth_deg",
    "cone_elevation_deg",
    "capture_eye",
    "capture_forward",
    "capture_up",
    "fixed_pose_verified",
)


def train_4d(
    dataset_zip: Path,
    out: Path,
    coarse_iters: int = 3_000,
    iters: int = 12_000,
    cap_max: int = 400_000,
    device: str = "cpu",
    field_kind: str = "explicit",
    checkpoint_every: int = 1_000,
    resume: bool = False,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Fit one packaged 4D dataset and write the checkpoint the exporter reads.

    The returned summary reports the temporal hold-out PSNR under a name that
    says what it is. There is no held-out *view* anywhere in a fixed-camera
    capture -- the training views are the only views -- so this number measures
    motion interpolation between withheld frames and must never be printed as
    novel-view quality (decision D9).
    """
    progress = progress or NullProgress()
    dataset_zip = Path(dataset_zip)
    out = Path(out)
    if not dataset_zip.exists():
        raise PipelineStateError(
            f"No packaged dataset at {dataset_zip}. Package one first with "
            "`gausscapture colab4d <project>`."
        )

    _require_torch()
    from gausscapture.recon.deform.bundle import (
        INIT_NAME,
        SceneInit,
        build_from_init,
        load_clip,
        unpack_dataset,
    )
    from gausscapture.recon.deform.schedule import ScheduleConfig
    from gausscapture.recon.deform.train4d import Train4DConfig
    from gausscapture.recon.deform.train4d import export_scene4d as write_checkpoint
    from gausscapture.recon.deform.train4d import train as run_training

    with tempfile.TemporaryDirectory(prefix="gausscapture-train4d-") as workspace:
        work = Path(workspace)
        progress.update(2, f"Unpacking {dataset_zip.name}")
        try:
            unpack_dataset(dataset_zip, work)
        except zipfile.BadZipFile as exc:
            # The likeliest cause by far is a cloud-drive download that produced
            # an HTML error page under a .zip name, which is indistinguishable
            # from a real archive until it is opened.
            raise CaptureFormatError(
                f"{dataset_zip.name} is not a readable zip archive: {exc}. If it came from a "
                "browser or a cloud drive, check that it is not an error page saved under "
                "that name; repackage it with `gausscapture colab4d <project>` if it is."
            ) from exc

        init = SceneInit.load(work / INIT_NAME)
        meta = init.meta
        clip = load_clip(work, device=device)
        params, field = build_from_init(init, field_kind=field_kind, device=device)

        schedule = ScheduleConfig(
            coarse_iters=int(coarse_iters), fine_iters=int(iters), cap_max=int(cap_max)
        )
        config = Train4DConfig(
            schedule=schedule,
            skin_k=int(meta.get("skin_k", config_default_skin_k())),
            holdout_stride=_holdout_stride(clip.n_frames),
            checkpoint_every=int(checkpoint_every),
        )
        # Beside the output rather than in the temporary workspace: the whole
        # point is that it survives the process, and on Colab that means it has
        # to land in Drive next to `scene4d.npz`.
        checkpoint = out.with_suffix(".ckpt") if checkpoint_every > 0 or resume else None

        rasterizer, strategy = select_backend(device)
        progress.update(
            6,
            f"Training {clip.n_frames} frames on {device} "
            f"({rasterizer.name} rasterizer, {strategy.name} densifier)",
        )
        result = run_training(
            clip,
            params,
            field,
            rasterizer=rasterizer,
            strategy=strategy,
            config=config,
            progress=_Sub(progress, 6, 94),
            checkpoint=checkpoint,
            resume=resume,
        )

        progress.update(95, "Writing scene4d.npz")
        out.parent.mkdir(parents=True, exist_ok=True)
        write_checkpoint(
            out,
            result.params,
            result.field,
            result.skinning,
            clip.times.detach().cpu(),
            metrics=result.metrics,
            cone=dict(meta.get("cone") or {}),
        )
        stamp_capture_provenance(out, meta)

        metrics = result.metrics
        summary = {
            "scene": str(out),
            "size_bytes": int(out.stat().st_size),
            "device": str(device),
            # Which kernel actually ran. Without this the summary of a CUDA run
            # and the summary of a CPU-reference run are indistinguishable, and
            # the reference one is a correctness check, not a trainer.
            "rasterizer": rasterizer.name,
            "densifier": strategy.name,
            "field": str(field_kind),
            "checkpoint": str(checkpoint) if checkpoint else None,
            "gaussians": int(metrics.gaussians),
            "dynamic_gaussians": int(metrics.dynamic_gaussians),
            "nodes": int(result.field.n_nodes),
            "frames": int(clip.n_frames),
            "steps": int(metrics.steps),
            "final_loss": _finite(metrics.final_loss),
            "train_psnr_db": _finite(metrics.train_psnr),
            "holdout_psnr_db": _finite(metrics.holdout_psnr_temporal),
            "holdout_frames": int(metrics.holdout_frames),
            # Said here as well as at every print site, because a summary
            # dictionary is exactly the thing that gets pasted into a table.
            "holdout_label": (
                "temporal interpolation from one fixed camera; not novel-view synthesis"
            ),
            "grown": int(metrics.grown),
            "relocated": int(metrics.relocated),
            "rebinds": int(metrics.rebinds),
            "rebinds_rejected": int(metrics.rebinds_rejected),
        }
    progress.update(100, "training complete")
    return summary


def select_backend(device: str) -> tuple[Any, Any]:
    """Pick the rasterizer and the densifier the requested device deserves.

    This function is the reason ``--device cuda`` means anything. The training
    loop defaults both arguments, and defaulting them selects
    :class:`ReferenceRasterizer`, which evaluates every Gaussian at every pixel
    through a materialised ``(N, H*W, 2)`` tensor. At the shipped
    ``--cap-max 400000`` and 1080p that single intermediate is 7 TB, per
    iteration, for 15,000 iterations -- so a run that was told to use a GPU did
    not merely run slowly, it could not complete a forward pass. The kernel was
    already written (:class:`GsplatRasterizer`); nothing called it.

    The densifier is deliberately *not* swapped. ``ReferenceMcmcStrategy`` is
    plain PyTorch -- ``multinomial``, indexing, ``einsum`` -- and runs on CUDA
    tensors as happily as on CPU ones, so there is nothing to gain by wrapping
    gsplat's ``MCMCStrategy`` beyond an adapter whose only evidence would be
    that it imports. See :func:`gsplat_strategy_available` for the seam if that
    ever changes.
    """
    from gausscapture.recon.deform.raster import GsplatRasterizer, ReferenceRasterizer
    from gausscapture.recon.deform.schedule import ReferenceMcmcStrategy

    strategy = ReferenceMcmcStrategy()
    if str(device).startswith("cuda"):
        if not GsplatRasterizer.available():
            raise DependencyMissingError(
                "`--device cuda` was requested but gsplat is not installed, and the CPU "
                "reference rasterizer is a correctness check rather than a trainer -- at "
                "the default cap it cannot complete one forward pass. Install "
                "`gausscapture[train4d]` on the GPU host, or package the scene with "
                "`gausscapture colab4d` and train it there."
            )
        return GsplatRasterizer(), strategy
    return ReferenceRasterizer(), strategy


def stamp_capture_provenance(scene_npz: Path, meta: dict[str, Any]) -> Path:
    """Add the preparation's measurements to a freshly written checkpoint.

    The training loop writes a file that describes a *fit*: geometry, colour,
    node trajectories. It has no way to describe the *capture*, because nothing
    it was handed measured one. Rather than teach the loop about cones, the
    numbers preparation measured are appended here, in the array names
    :mod:`gausscapture.export.checkpoint4d` reads.

    The file is rewritten rather than appended to: ``.npz`` is a zip of
    individual arrays with no in-place add, and a checkpoint that is written
    twice is cheaper by orders of magnitude than one that reaches the exporter
    unable to say what cone it was shot in.
    """
    scene_npz = Path(scene_npz)
    with np.load(scene_npz, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}

    cone = dict(meta.get("cone") or {})
    capture = dict(meta.get("capture") or {})
    arrays["fps"] = np.float32(meta.get("fps") or 30.0)
    arrays["cone_azimuth_deg"] = np.float32(cone.get("azimuth_half_deg") or 0.0)
    arrays["cone_elevation_deg"] = np.float32(cone.get("elevation_half_deg") or 0.0)
    arrays["capture_eye"] = _vector(capture.get("eye"), (0.0, 0.0, 0.0))
    arrays["capture_forward"] = _vector(capture.get("forward"), (0.0, 0.0, -1.0))
    arrays["capture_up"] = _vector(capture.get("up"), (0.0, 1.0, 0.0))
    arrays["fixed_pose_verified"] = np.float32(1.0 if capture.get("verified") else 0.0)

    # The exporter's META chunk is what the viewer prints, so the preparation's
    # own record travels whole rather than as the handful of scalars above.
    existing = _decode_meta(arrays.get("meta"))
    existing["capture"] = capture
    existing["cone"] = cone
    existing["fixed_pose"] = meta.get("fixed_pose", {})
    existing["claims"] = meta.get("claims", {})
    existing.setdefault("sh_degree", 0)
    arrays["meta"] = np.asarray(json.dumps(existing, default=str))

    missing = [key for key in CAPTURE_PROVENANCE_KEYS if key not in arrays]
    if missing:
        raise PipelineStateError(
            f"the checkpoint at {scene_npz} would be written without "
            f"{', '.join(missing)}, so the exported scene could not state what cone it was "
            "shot in"
        )
    np.savez_compressed(scene_npz, **arrays)
    return scene_npz


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def config_default_skin_k() -> int:
    """The default binding width, read from the trainer rather than repeated."""
    from gausscapture.recon.deform.train4d import Train4DConfig

    return int(Train4DConfig().skin_k)


def _require_torch() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise DependencyMissingError(
            "torch is not installed, so `train4d` cannot run on this machine. That is the "
            "expected state on an Apple Silicon laptop: package the scene with "
            "`gausscapture colab4d <project>` and train it on a cloud GPU, then bring the "
            "result back with `--collect`."
        ) from exc


def _holdout_stride(n_frames: int) -> int:
    """Withhold frames only when there are enough left to fit anything.

    The default stride of 8 leaves nothing to measure on a six-frame clip and
    nothing to fit on a two-frame one, so a short clip trains on all of it and
    reports no hold-out rather than reporting a meaningless one.
    """
    default = 8
    return default if n_frames >= 2 * default else 0


def _finite(value: float) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _vector(value: Any, default: tuple[float, float, float]) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32).reshape(3)
    except (TypeError, ValueError):
        return np.asarray(default, dtype=np.float32)
    return array


def _decode_meta(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    try:
        item = np.asarray(raw).reshape(-1)[0]
        decoded = json.loads(item.item() if hasattr(item, "item") else str(item))
    except (TypeError, ValueError, IndexError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


class _Sub:
    """Rescales a child step's 0-100 progress into a slice of the parent's."""

    def __init__(self, parent: Progress, start: int, end: int):
        self.parent = parent
        self.start = start
        self.span = max(0, end - start)

    def update(self, percent: int, message: str | None = None) -> None:
        self.parent.update(self.start + int(self.span * max(0, min(100, percent)) / 100), message)

    def log(self, message: str) -> None:
        self.parent.log(message)
