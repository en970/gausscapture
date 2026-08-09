"""Measuring which pixels move, for a camera that does not.

Every other part of this pipeline has to *estimate* what is foreground. During
the hold phase we do not: the camera is bolted in place, so a pixel that changes
changed because something in the world moved. The dynamic region is a
measurement, and this module measures it.

The method is a temporal median background plus a per-pixel maximum deviation,
with the threshold read off the capture's own rest window -- the frames after
the phone is back in its support and before the subject is cued. Those frames
contain nothing but sensor and codec noise, so the robust deviation across them
is exactly the noise floor the mask has to clear. A constant threshold cannot be
right here: the same number is deaf on a bright ISO-50 take and hallucinating on
a dim ISO-1600 one.

**Why not a segmentation network.** SAM 2.1 is genuinely Apache-2.0 (model code
and weights alike), so licence is not the objection -- the objection is that it
would make PyTorch and a 150 MB checkpoint hard dependencies of a package whose
core is NumPy and OpenCV, to answer a question that a fixed camera answers
exactly. A learned segmenter guesses at object boundaries from appearance; this
measures them from evidence. Where the two disagree on a fixed-camera clip, the
measurement is right. A learned segmenter stays available as an optional
refinement for the cases where it wins -- a subject who holds part of themselves
perfectly still for the whole clip -- and it is not needed for v1.

**Mask polarity is a footgun, so it is written down.** The masks this module
writes for the trainer are *subject* masks: 255 where the subject moved. The
masks it writes for COLMAP follow COLMAP's own convention, which is the
opposite: a mask pixel of 0 means "ignore this pixel", so the dynamic region is
black and the static background is white. The two directories are never the same
directory.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from gausscapture.errors import CaptureFormatError
from gausscapture.progress import NullProgress, Progress

#: Multiples of the measured noise sigma a pixel must move to count as dynamic.
#: Six is the usual "this is not noise" bar; at four, JPEG ringing along strong
#: background edges leaks into the mask on every take.
NOISE_SIGMA_MULTIPLE = 6.0

#: Absolute floor on the threshold, in grey levels. A capture whose rest window
#: happens to be unusually clean would otherwise set a threshold below the codec
#: quantisation step and mask the entire frame.
MIN_THRESHOLD_LEVELS = 6.0

#: Dilation radius as a fraction of frame height. The mask has to cover the
#: subject's motion blur and the halo the codec smears around a moving edge,
#: neither of which crosses the deviation threshold everywhere.
DILATE_FRACTION = 0.02

#: Connected components smaller than this fraction of the frame are noise, not
#: subject. Deliberately generous: a hand at the edge of frame is far bigger.
MIN_COMPONENT_FRACTION = 5e-4

#: Frames sampled for the temporal median. The median of 32 evenly spaced frames
#: is indistinguishable from the median of 90 for this purpose and costs a third
#: of the memory on a 1080p clip.
MEDIAN_MAX_FRAMES = 32

#: Above this dynamic fraction there is not enough static background left for
#: COLMAP to register the masked keyframes against.
CROWDED_FRACTION = 0.60


@dataclass
class DynamicMaskReport:
    """What the mask measured, in the units it measured them in."""

    frames: int = 0
    median_frames: int = 0
    width: int = 0
    height: int = 0
    #: Robust standard deviation of the rest window, in grey levels.
    noise_sigma: float = 0.0
    noise_source: str = "rest_window"
    threshold_levels: float = 0.0
    dilate_px: int = 0
    dynamic_fraction_mean: float = 0.0
    dynamic_fraction_max: float = 0.0
    union_fraction: float = 0.0
    masks_dir: str = ""
    union_path: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "median_frames": self.median_frames,
            "width": self.width,
            "height": self.height,
            "noise_sigma": self.noise_sigma,
            "noise_source": self.noise_source,
            "threshold_levels": self.threshold_levels,
            "dilate_px": self.dilate_px,
            "dynamic_fraction_mean": self.dynamic_fraction_mean,
            "dynamic_fraction_max": self.dynamic_fraction_max,
            "union_fraction": self.union_fraction,
            "masks_dir": self.masks_dir,
            "union_path": self.union_path,
            "warnings": list(self.warnings),
        }


def build_dynamic_masks(
    frame_paths: Sequence[Path],
    out_dir: Path,
    rest_paths: Sequence[Path] = (),
    noise_multiple: float = NOISE_SIGMA_MULTIPLE,
    min_threshold: float = MIN_THRESHOLD_LEVELS,
    dilate_fraction: float = DILATE_FRACTION,
    min_component_fraction: float = MIN_COMPONENT_FRACTION,
    median_max_frames: int = MEDIAN_MAX_FRAMES,
    progress: Progress | None = None,
) -> tuple[DynamicMaskReport, np.ndarray]:
    """Write one subject mask per hold frame plus their union.

    Returns the report and the union mask, because every caller downstream --
    the COLMAP masks, the dynamic/static labelling of Gaussians, the subject
    centroid the view cone is measured about -- wants the union and would
    otherwise re-read it from disk immediately.
    """
    progress = progress or NullProgress()
    frame_paths = [Path(p) for p in frame_paths]
    if len(frame_paths) < 3:
        raise CaptureFormatError(
            f"The dynamic region is measured across the hold phase and needs at least 3 "
            f"frames; {len(frame_paths)} were given. Record a longer hold, or check that "
            "the phase split found the hold at all."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    background, median_frames = _temporal_median(frame_paths, median_max_frames, progress)
    height, width = background.shape[:2]

    noise_sigma, noise_source, warnings = _noise_floor(
        background, [Path(p) for p in rest_paths], frame_paths
    )
    threshold = max(noise_multiple * noise_sigma, min_threshold)
    dilate_px = max(1, int(round(dilate_fraction * height)))
    min_area = max(4, int(round(min_component_fraction * height * width)))

    union = np.zeros((height, width), dtype=bool)
    fractions: list[float] = []
    for i, path in enumerate(frame_paths):
        gray = _read_gray(path, expect=(height, width))
        deviation = np.abs(gray - background)
        mask = _clean(deviation > threshold, min_area, dilate_px)
        union |= mask
        fractions.append(float(mask.mean()))
        cv2.imwrite(str(out_dir / f"{path.stem}.png"), (mask.astype(np.uint8) * 255))
        if i % 8 == 0:
            progress.update(
                40 + int(55 * (i + 1) / len(frame_paths)),
                f"Masked {i + 1}/{len(frame_paths)} hold frames",
            )

    union_path = out_dir / "union.png"
    cv2.imwrite(str(union_path), (union.astype(np.uint8) * 255))

    union_fraction = float(union.mean())
    if union_fraction == 0.0:
        warnings.append(
            f"Nothing in the hold phase moved by more than {threshold:.1f} grey levels, so the "
            "dynamic region is empty. Either the subject held still or the take caught the "
            "wrong frames; a 4D scene trained on this is a static scene."
        )
    elif union_fraction > CROWDED_FRACTION:
        warnings.append(
            f"The moving region covers {union_fraction:.0%} of the frame, leaving little static "
            "background for structure-from-motion to register the masked hold keyframes "
            "against. Sit further from the phone, or include more of the room behind you."
        )

    report = DynamicMaskReport(
        frames=len(frame_paths),
        median_frames=median_frames,
        width=width,
        height=height,
        noise_sigma=noise_sigma,
        noise_source=noise_source,
        threshold_levels=float(threshold),
        dilate_px=dilate_px,
        dynamic_fraction_mean=float(np.mean(fractions)),
        dynamic_fraction_max=float(np.max(fractions)),
        union_fraction=union_fraction,
        masks_dir=str(out_dir),
        union_path=str(union_path),
        warnings=warnings,
    )
    progress.update(100, f"Dynamic region covers {union_fraction:.1%} of the frame")
    return report, union


def write_colmap_masks(
    mask: np.ndarray, image_names: Sequence[str], out_dir: Path
) -> list[Path]:
    """Write COLMAP ``--ImageReader.mask_path`` masks for the given images.

    COLMAP looks for ``<mask_path>/<image name>.png`` -- the *full* file name
    including its extension, not the stem -- and ignores every pixel that is
    zero there. So the mask is inverted relative to the subject mask: black
    where the subject moved, white everywhere the geometry is rigid. Images with
    no file in this directory are used in full, which is exactly what the static
    phases want.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = np.where(np.asarray(mask, dtype=bool), 0, 255).astype(np.uint8)
    written: list[Path] = []
    for name in image_names:
        path = out_dir / f"{name}.png"
        cv2.imwrite(str(path), keep)
        written.append(path)
    return written


def read_mask(path: Path) -> np.ndarray:
    """Read a subject mask back as a boolean array."""
    path = Path(path)
    data = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if data is None:
        raise CaptureFormatError(f"Mask could not be read: {path}")
    return data > 127


def _temporal_median(
    paths: Sequence[Path], max_frames: int, progress: Progress
) -> tuple[np.ndarray, int]:
    """Per-pixel median over evenly spaced frames.

    The median rather than the mean because the subject occupies each pixel for
    only part of the clip; a mean carries a ghost of them into the background
    and then subtracts that ghost from every frame.
    """
    count = min(max(3, max_frames), len(paths))
    indices = np.unique(np.linspace(0, len(paths) - 1, count).round().astype(int))
    first = _read_gray(paths[indices[0]])
    stack = np.empty((indices.size, *first.shape), dtype=np.float32)
    stack[0] = first
    for slot, index in enumerate(indices[1:], start=1):
        stack[slot] = _read_gray(paths[index], expect=first.shape)
        progress.update(5 + int(30 * slot / indices.size), "Building the static background")
    return np.median(stack, axis=0), int(indices.size)


def _noise_floor(
    background: np.ndarray, rest_paths: Sequence[Path], fallback_paths: Sequence[Path]
) -> tuple[float, str, list[str]]:
    """Robust sigma of the residual, measured on frames where nothing moved."""
    warnings: list[str] = []
    if rest_paths:
        residuals = [
            (_read_gray(path, expect=background.shape) - background).ravel()
            for path in rest_paths
        ]
        return _robust_sigma(np.concatenate(residuals)), "rest_window", warnings

    warnings.append(
        "No rest-window frames were available, so the noise floor is estimated from the hold "
        "frames themselves. The MAD ignores the moving minority of pixels, so this is close "
        "to right, but it is an estimate over frames that contain the subject rather than a "
        "measurement over frames that do not. Record with preset F to measure it properly."
    )
    sample = [fallback_paths[i] for i in _spread(len(fallback_paths), 5)]
    residuals = [
        (_read_gray(path, expect=background.shape) - background).ravel() for path in sample
    ]
    return _robust_sigma(np.concatenate(residuals)), "hold_frames", warnings


def _spread(total: int, count: int) -> list[int]:
    if total <= 0:
        return []
    return sorted({int(round(v)) for v in np.linspace(0, total - 1, min(count, total))})


def _robust_sigma(residual: np.ndarray) -> float:
    """Median-absolute-deviation sigma, which a moving subject cannot inflate.

    A plain standard deviation over a frame containing a subject measures the
    subject, not the sensor. The MAD ignores anything past the middle half of
    the distribution, so a foreground occupying up to half the frame moves it
    not at all.
    """
    if residual.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(residual - np.median(residual))))


def _clean(mask: np.ndarray, min_area: int, dilate_px: int) -> np.ndarray:
    """Open, drop specks, then dilate."""
    binary = mask.astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count > 1:
        keep = np.zeros(count, dtype=bool)
        for label in range(1, count):
            keep[label] = stats[label, cv2.CC_STAT_AREA] >= min_area
        binary = keep[labels].astype(np.uint8)
    if dilate_px > 0 and binary.any():
        size = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        binary = cv2.dilate(binary, kernel)
    return binary.astype(bool)


def _read_gray(path: Path, expect: tuple[int, int] | None = None) -> np.ndarray:
    data = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if data is None:
        raise CaptureFormatError(
            f"Hold-phase frame could not be read while measuring the dynamic region: {path}"
        )
    if expect is not None and data.shape[:2] != tuple(expect):
        raise CaptureFormatError(
            f"Frame {path.name} is {data.shape[1]}x{data.shape[0]} but the rest of the hold "
            f"phase is {expect[1]}x{expect[0]}. Every frame of a fixed-camera capture comes "
            "from one camera at one resolution; re-extract the frames without resizing."
        )
    return data.astype(np.float32)
