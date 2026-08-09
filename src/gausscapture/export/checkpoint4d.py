"""From the trainer's checkpoint to the browser's container.

``scene4d.npz`` is what a training run produces and what comes back from a
rented GPU. ``.g4d`` is what a browser streams. The two disagree on purpose,
and this module is where they are reconciled:

* the trainer holds **log** scales and **logit** opacities, because those are
  the unconstrained quantities an optimiser can move; the container holds
  linear scales and RGBA8, because those are what a shader can use without
  decoding anything;
* the trainer holds a per-gaussian dynamic flag, because densification adds
  and removes gaussians in arbitrary order; the container holds a single
  ``dynamic_count`` and requires the dynamic group first, because that lets
  the shader branch on ``gl_InstanceID`` with no per-gaussian flag in the file;
* the trainer keeps every gaussian it fitted, including ones too faint to see;
  the container is a download, and a gaussian at 1 % opacity costs the same
  20 bytes as one that carries the image.

Doing this conversion in the exporter rather than in the trainer is deliberate:
the checkpoint stays the thing the training run can be resumed from, and the
lossy decisions -- what to prune, how many nodes to skin to, what to truncate
to -- are made where the size budget is known.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from gausscapture.errors import CaptureFormatError
from gausscapture.export.scene4d import MAX_SKIN_K, Scene4D, write_scene4d
from gausscapture.progress import NullProgress, Progress

#: gsplat's zeroth-order spherical-harmonic coefficient. ``sh_degree`` is 0
#: throughout the 4D path -- a fixed camera cannot constrain view-dependent
#: colour -- so this single constant is the whole colour decode.
SH0 = 0.28209479177387814


def export_scene4d(
    scene_npz: Path,
    out: Path,
    skin_k: int = 4,
    min_opacity: float = 0.02,
    max_gaussians: int | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Write a ``.g4d`` from a trained checkpoint and report what it cost.

    The returned dictionary is what the viewer's ``scene.json`` is built from.
    It states the pruning and the truncation explicitly, because a scene that
    was quietly reduced is not the scene that was trained and the page has to
    be able to say so.
    """
    progress = progress or NullProgress()
    scene_npz = Path(scene_npz)
    out = Path(out)

    progress.update(5, f"Reading {scene_npz.name}")
    scene, pruned = load_trained_scene(scene_npz, min_opacity=min_opacity, skin_k=skin_k)

    progress.update(45, "Packing the container")
    result = write_scene4d(scene, out, max_gaussians=max_gaussians)
    result.update(
        {
            "path": str(out),
            "size_bytes": result["bytes"],
            "source": str(scene_npz),
            "min_opacity": float(min_opacity),
            "pruned_faint": int(pruned),
        }
    )
    progress.update(100, "4D scene exported")
    return result


def load_trained_scene(
    path: Path,
    min_opacity: float = 0.0,
    skin_k: int | None = None,
) -> tuple[Scene4D, int]:
    """Build a :class:`Scene4D` from a trainer checkpoint.

    Returns the scene and how many gaussians the opacity threshold removed.
    """
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
    except (OSError, ValueError) as error:
        raise CaptureFormatError(f"{path.name} is not a readable .npz: {error}") from None
    if not arrays:
        raise CaptureFormatError(f"{path.name} contains no arrays")

    means = _required(arrays, path, "means", "positions", "xyz").astype(np.float32)
    count = len(means)
    quats = _unit(_required(arrays, path, "quats", "rotations", "quaternions"))
    scales = _linear_scales(arrays, path)
    colors = _rgba(arrays, path, count)

    node_rest = _required(arrays, path, "node_rest", "node_positions", "nodes").astype(np.float32)
    # ``node_quat`` (singular) is what the training loop writes; the plural names
    # are what a checkpoint hand-assembled from a gsplat-style export uses. All
    # three mean the same ``(T, K, 4)`` array and reading only one of them is how
    # a trained scene becomes unexportable.
    node_quats = _required(
        arrays, path, "node_quats", "node_quat", "node_rotations"
    ).astype(np.float32)
    node_trans = _required(
        arrays, path, "node_trans", "node_translations"
    ).astype(np.float32)

    dynamic = _dynamic_mask(arrays, count)
    # The container requires the dynamic group first; a checkpoint written after
    # densification has them in whatever order the strategy left behind.
    order = np.concatenate([np.flatnonzero(dynamic), np.flatnonzero(~dynamic)])
    dynamic_count = int(dynamic.sum())

    skin_indices = _required(arrays, path, "skin_indices", "skin_idx").astype(np.int64)
    skin_weights = _required(arrays, path, "skin_weights", "skin_w").astype(np.float32)
    if len(skin_indices) == count:
        # Per-gaussian skinning: take the dynamic rows in the new order, so a
        # static gaussian never carries a binding into the file.
        skin_indices = skin_indices[order][:dynamic_count]
        skin_weights = skin_weights[order][:dynamic_count]
    elif len(skin_indices) != dynamic_count:
        raise CaptureFormatError(
            f"{path.name}: skin_indices has {len(skin_indices)} rows, expected "
            f"{dynamic_count} (dynamic) or {count} (all gaussians)"
        )

    means, scales, quats, colors = (a[order] for a in (means, scales, quats, colors))

    pruned = 0
    if min_opacity > 0:
        keep = colors[:, 3] >= int(round(min_opacity * 255))
        pruned = int((~keep).sum())
        if not keep.any():
            raise CaptureFormatError(
                f"--min-opacity {min_opacity} removes every gaussian; the trained scene's "
                f"strongest opacity is {colors[:, 3].max() / 255:.3f}"
            )
        skin_indices = skin_indices[keep[:dynamic_count]]
        skin_weights = skin_weights[keep[:dynamic_count]]
        dynamic_count = int(keep[:dynamic_count].sum())
        means, scales, quats, colors = (a[keep] for a in (means, scales, quats, colors))

    if skin_k is not None and dynamic_count:
        skin_indices, skin_weights = _reduce_skin(skin_indices, skin_weights, skin_k)

    scene = Scene4D(
        means=means,
        scales=scales,
        quats=quats,
        colors=colors,
        node_rest=node_rest,
        node_quats=node_quats,
        node_trans=node_trans,
        skin_indices=skin_indices,
        skin_weights=skin_weights,
        dynamic_count=dynamic_count,
        fps=float(_scalar(arrays, "fps", 30.0)),
        cone_azimuth_deg=float(_scalar(arrays, "cone_azimuth_deg", 0.0)),
        cone_elevation_deg=float(_scalar(arrays, "cone_elevation_deg", 0.0)),
        capture_eye=_vector(arrays, "capture_eye", (0.0, 0.0, 0.0)),
        capture_forward=_vector(arrays, "capture_forward", (0.0, 0.0, -1.0)),
        capture_up=_vector(arrays, "capture_up", (0.0, 1.0, 0.0)),
        fixed_pose_verified=bool(_scalar(arrays, "fixed_pose_verified", 0.0)),
        meta=_meta(arrays),
    )
    scene.validate()
    return scene, pruned


# --------------------------------------------------------------------------
# checkpoint decoding
# --------------------------------------------------------------------------


def _required(arrays: dict[str, np.ndarray], path: Path, *names: str) -> np.ndarray:
    for name in names:
        if name in arrays:
            return arrays[name]
    raise CaptureFormatError(
        f"{path.name} has no {names[0]} array (looked for {', '.join(names)}). "
        "Was it written by `gausscapture train4d`?"
    )


def _unit(quats: np.ndarray) -> np.ndarray:
    quats = np.asarray(quats, dtype=np.float32).reshape(-1, 4)
    norm = np.linalg.norm(quats, axis=1, keepdims=True)
    # A zero quaternion is not a rotation; it is a gaussian the optimiser
    # drove to nothing, and identity is the only harmless substitution.
    safe = np.where(norm > 1e-12, norm, 1.0)
    out = quats / safe
    out[norm[:, 0] <= 1e-12] = (1.0, 0.0, 0.0, 0.0)
    return out.astype(np.float32)


def _linear_scales(arrays: dict[str, np.ndarray], path: Path) -> np.ndarray:
    if "scales" in arrays:
        return np.asarray(arrays["scales"], dtype=np.float32)
    log_scales = _required(arrays, path, "log_scales", "scaling")
    return np.exp(np.asarray(log_scales, dtype=np.float32)).astype(np.float32)


def _rgba(arrays: dict[str, np.ndarray], path: Path, count: int) -> np.ndarray:
    if "colors" in arrays and arrays["colors"].shape == (count, 4):
        return np.asarray(arrays["colors"], dtype=np.uint8)

    if "colors" in arrays and arrays["colors"].shape == (count, 3):
        # The training loop decodes sh0 to linear RGB before it writes, because
        # that is what its rasterizer took. Re-deriving it here from a band that
        # is not in the file would fail; exponentiating it as if it were still a
        # coefficient would be worse, because it would succeed and be wrong.
        rgb = np.asarray(arrays["colors"], dtype=np.float32)
    elif "rgb" in arrays:
        rgb = np.asarray(arrays["rgb"], dtype=np.float32)
    else:
        sh0 = _required(arrays, path, "sh0", "features_dc").astype(np.float32).reshape(count, -1)
        rgb = SH0 * sh0[:, :3] + 0.5

    if "opacities" in arrays:
        alpha = np.asarray(arrays["opacities"], dtype=np.float32).reshape(-1)
    else:
        logits = _required(arrays, path, "logit_opacities", "opacity").astype(np.float32)
        alpha = 1.0 / (1.0 + np.exp(-logits.reshape(-1)))

    rgba = np.empty((count, 4), dtype=np.uint8)
    rgba[:, :3] = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    rgba[:, 3] = np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return rgba


def _dynamic_mask(arrays: dict[str, np.ndarray], count: int) -> np.ndarray:
    for name in ("dynamic", "is_dynamic", "dynamic_mask"):
        if name in arrays:
            return np.asarray(arrays[name]).reshape(-1).astype(bool)
    if "dynamic_count" in arrays:
        # Already grouped: the checkpoint states the split rather than the flag.
        mask = np.zeros(count, dtype=bool)
        mask[: int(np.asarray(arrays["dynamic_count"]).reshape(-1)[0])] = True
        return mask
    return np.zeros(count, dtype=bool)


def _reduce_skin(
    indices: np.ndarray, weights: np.ndarray, skin_k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the ``skin_k`` strongest bindings per gaussian and renormalise.

    Dropping the weakest binding is close to free -- the weights fall off with
    distance -- and each one removed takes two bytes per dynamic gaussian out
    of ``SKIN``, so the budget is worth spending here rather than on geometry.
    """
    skin_k = max(1, min(int(skin_k), MAX_SKIN_K))
    if indices.shape[1] <= skin_k:
        return indices, weights
    keep = np.argsort(-weights, axis=1)[:, :skin_k]
    rows = np.arange(len(indices))[:, None]
    kept_weights = weights[rows, keep]
    total = kept_weights.sum(axis=1, keepdims=True)
    kept_weights = kept_weights / np.where(total > 0, total, 1.0)
    return indices[rows, keep].astype(np.int64), kept_weights.astype(np.float32)


def _scalar(arrays: dict[str, np.ndarray], name: str, default: float) -> float:
    if name not in arrays:
        return default
    return float(np.asarray(arrays[name]).reshape(-1)[0])


def _vector(arrays: dict[str, np.ndarray], name: str, default: tuple[float, float, float]):
    if name not in arrays:
        return np.array(default, dtype=np.float32)
    return np.asarray(arrays[name], dtype=np.float32).reshape(3)


def _meta(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """The trainer's own record, carried into the file's META chunk.

    Stored as a JSON string in the checkpoint because ``.npz`` holds arrays and
    nothing else; a malformed one is dropped rather than allowed to stop an
    export that is otherwise complete.
    """
    name = next((key for key in ("meta_json", "meta") if key in arrays), None)
    if name is None:
        return {}
    try:
        raw = np.asarray(arrays[name]).reshape(-1)[0]
        meta = json.loads(raw.item() if hasattr(raw, "item") else str(raw))
    except (ValueError, TypeError, AttributeError, IndexError):
        return {}
    return meta if isinstance(meta, dict) else {}
