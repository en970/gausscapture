"""Per-frame capture signals.

Every function here operates on a single frame or a short window, takes plain
NumPy arrays, and has no knowledge of projects or files. That is what makes
them testable on synthetic images and reusable by an on-device implementation
later.
"""

from __future__ import annotations

import statistics
from collections import deque

import cv2
import numpy as np

#: Pixel values at or above this count as clipped highlights.
OVEREXPOSED_LEVEL = 245
#: Pixel values at or below this count as crushed shadows.
UNDEREXPOSED_LEVEL = 20

#: Frames whose sharpness falls below this fraction of the clip's rolling
#: median are treated as blurred. Chosen as a defensible starting point, not an
#: empirically validated value -- calibrating it is sprint S7's job.
DEFAULT_BLUR_RELATIVE_THRESHOLD = 0.60

#: How many recent frames define "normal sharpness for this clip".
BLUR_WINDOW = 31
#: Below this many samples the window is not yet representative, so nothing is
#: rejected. Without this, the first frames of every clip get discarded.
BLUR_WARMUP = 9


def variance_of_laplacian(gray: np.ndarray) -> float:
    """Sharpness proxy: the variance of the image's Laplacian.

    High variance means strong high-frequency content, which usually means a
    sharp frame. The measure is not absolute -- it scales with resolution,
    contrast, and how much texture the scene happens to contain -- so callers
    should compare it against other frames from the same clip rather than
    against a fixed number. See :class:`RollingBlurNormaliser`.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def exposure_ratios(gray: np.ndarray) -> tuple[float, float]:
    """Fraction of pixels that are clipped white and crushed black.

    Auto-exposure drift across a capture breaks the brightness-constancy
    assumption that Gaussian splatting relies on, and shows up as floaters.
    Correcting it after the fact is worth several dB, so detecting it during
    capture is worth a prompt.
    """
    over = float((gray >= OVEREXPOSED_LEVEL).mean())
    under = float((gray <= UNDEREXPOSED_LEVEL).mean())
    return over, under


def normalised_histogram(gray: np.ndarray, bins: int = 32) -> np.ndarray:
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
    cv2.normalize(hist, hist)
    return hist


class RollingBlurNormaliser:
    """Turns raw sharpness into a clip-relative score.

    Keeps a sliding window of recent variance-of-Laplacian values and divides
    the current one by their median. A result near 1.0 means "as sharp as this
    clip normally is"; 0.3 means "markedly softer than its neighbours".

    This replaces comparing against a fixed constant, which is resolution- and
    content-dependent and therefore not portable between captures.
    """

    def __init__(self, window: int = BLUR_WINDOW, warmup: int = BLUR_WARMUP):
        self._values: deque[float] = deque(maxlen=window)
        self._warmup = warmup

    def push(self, value: float) -> float:
        """Record a sharpness value and return its clip-relative form."""
        self._values.append(value)
        return self.relative(value)

    def relative(self, value: float) -> float:
        if len(self._values) < self._warmup:
            return 1.0  # Not enough context yet; assume typical.
        median = statistics.median(self._values)
        if median <= 0:
            return 1.0
        return float(value / median)

    @property
    def ready(self) -> bool:
        return len(self._values) >= self._warmup


def frame_signals(
    frame: np.ndarray,
    previous_gray: np.ndarray | None = None,
    previous_hist: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute every cheap per-frame signal in one pass.

    Returns raw values only. Clip-relative normalisation and thresholding are
    the caller's job, because they need the whole sequence.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    over, under = exposure_ratios(gray)
    hist = normalised_histogram(gray)

    result: dict[str, float] = {
        "blur_vol": variance_of_laplacian(gray),
        "brightness": float(gray.mean()),
        "overexposed_ratio": over,
        "underexposed_ratio": under,
    }
    if previous_gray is not None and previous_gray.shape == gray.shape:
        result["motion_diff"] = float(cv2.absdiff(gray, previous_gray).mean())
    if previous_hist is not None:
        result["hist_correlation"] = float(cv2.compareHist(previous_hist, hist, cv2.HISTCMP_CORREL))

    result["_gray"] = gray  # type: ignore[assignment]
    result["_hist"] = hist  # type: ignore[assignment]
    return result


def percentile(values: list[float], q: float) -> float:
    """Percentile with an empty-input guard, so reports never raise."""
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))
