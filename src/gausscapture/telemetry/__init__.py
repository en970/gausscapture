"""Capture telemetry: signals measurable at capture time.

This is the module the project's research question lives in. Everything else in
the package exists to get pixels to it and to turn its output into a
reconstruction that can be scored against it.

The central claim under investigation is that these signals predict downstream
reconstruction quality *before* training. That claim is not yet established --
see ``docs/RESEARCH.md`` section 7 -- so this module is careful to report each
signal separately and unweighted, and to label the composite ``score`` as the
heuristic it currently is.

Design notes that matter for the science:

* Blur is reported both raw (``blur_vol``) and relative to a rolling median of
  the same clip (``blur_relative``). Only the relative form is comparable
  across scenes and resolutions; an absolute variance-of-Laplacian threshold
  rejects sharp frames of textureless walls and accepts blurred frames of
  bookshelves.
* ``vol_p10`` is exposed as a first-class aggregate because it is the
  pre-registered candidate predictor.

**Known limitation: relative blur is blind to a uniformly soft capture.**
Because ``blur_relative`` compares each frame against a rolling median of its
neighbours, a clip that is blurred from beginning to end contains no frame that
is *relatively* soft, and ``blurry_frame_ratio`` stays near zero. Measured on
synthetic captures, a deliberately end-to-end-blurred clip scored *higher* than
a sharp one. Reverting to an absolute threshold is not the fix -- that
reintroduces the content and resolution dependence the relative measure exists
to remove -- so the composite score is currently wrong in this regime and is
labelled a heuristic for exactly this kind of reason. Candidate resolutions
(a resolution-normalised absolute term, or gyroscope-derived angular velocity,
since blur scales with angular rate times exposure) are open work, and
distinguishing them is one of the things the benchmark is for.
"""

from __future__ import annotations

from gausscapture.telemetry.report import analyze_capture, score_report
from gausscapture.telemetry.signals import (
    RollingBlurNormaliser,
    exposure_ratios,
    frame_signals,
    variance_of_laplacian,
)

__all__ = [
    "RollingBlurNormaliser",
    "analyze_capture",
    "exposure_ratios",
    "frame_signals",
    "score_report",
    "variance_of_laplacian",
]
