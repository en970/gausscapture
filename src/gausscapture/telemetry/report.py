from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from gausscapture.errors import CaptureFormatError
from gausscapture.pack.manifest import find_main_video, read_manifest
from gausscapture.progress import NullProgress, Progress
from gausscapture.telemetry.signals import (
    DEFAULT_BLUR_RELATIVE_THRESHOLD,
    RollingBlurNormaliser,
    frame_signals,
    percentile,
)
from gausscapture.types import FrameSignals, TelemetryReport, VideoInfo

#: Frame-to-frame mean absolute difference above which motion is "too fast".
#: Like the blur threshold, a starting point rather than a validated constant.
FAST_MOTION_DIFF = 35.0
#: Histogram correlation above which two frames are near-duplicates.
DUPLICATE_CORRELATION = 0.992
#: ...provided they also barely differ in pixels.
DUPLICATE_MAX_DIFF = 2.0

MAX_THUMBNAILS = 8
THUMBNAIL_MAX_SIDE = 360


def analyze_capture(
    project_path: Path,
    sample_count: int = 80,
    blur_relative_threshold: float = DEFAULT_BLUR_RELATIVE_THRESHOLD,
    write: bool = True,
    progress: Progress | None = None,
) -> TelemetryReport:
    """Sample a capture's video and compute its telemetry report.

    Samples evenly across the clip rather than reading every frame: the goal is
    a fast pre-flight verdict, not an exhaustive index. Frame extraction does
    the exhaustive pass later.
    """
    progress = progress or NullProgress()
    pack_dir = project_path / "capturepack"
    video_path = find_main_video(pack_dir)
    manifest = read_manifest(pack_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise CaptureFormatError(f"Video could not be opened for analysis: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or (manifest.get("video", {}).get("fps") or 0.0)
        indices = _sample_indices(total_frames, sample_count)

        signals: list[FrameSignals] = []
        thumbnails: list[str] = []
        normaliser = RollingBlurNormaliser()
        thumb_dir = project_path / "quality" / "thumbnails"
        if write:
            thumb_dir.mkdir(parents=True, exist_ok=True)

        prev_gray = None
        prev_hist = None
        for position, frame_idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue

            raw = frame_signals(frame, prev_gray, prev_hist)
            prev_gray = raw.pop("_gray")
            prev_hist = raw.pop("_hist")

            signals.append(
                FrameSignals(
                    index=frame_idx,
                    timestamp_sec=(frame_idx / fps) if fps else None,
                    blur_vol=raw["blur_vol"],
                    blur_relative=normaliser.push(raw["blur_vol"]),
                    brightness=raw["brightness"],
                    overexposed_ratio=raw["overexposed_ratio"],
                    underexposed_ratio=raw["underexposed_ratio"],
                    motion_diff=raw.get("motion_diff"),
                    hist_correlation=raw.get("hist_correlation"),
                )
            )

            if write and len(thumbnails) < MAX_THUMBNAILS:
                rel = Path("quality") / "thumbnails" / f"thumb_{len(thumbnails) + 1:02d}.jpg"
                cv2.imwrite(str(project_path / rel), _thumbnail(frame))
                thumbnails.append(str(rel))

            progress.update(
                5 + int(80 * ((position + 1) / max(1, len(indices)))),
                f"Analysed frame {position + 1}/{len(indices)}",
            )
    finally:
        cap.release()

    if not signals:
        raise CaptureFormatError("No frames could be sampled from the video")

    report = _aggregate(signals, manifest, fps, blur_relative_threshold)
    report.thumbnails = thumbnails
    score_report(report)

    if write:
        out = project_path / "quality" / "quality_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    progress.update(100, "Analysis complete")
    return report


def _aggregate(
    signals: list[FrameSignals],
    manifest: dict,
    fps: float,
    blur_relative_threshold: float,
) -> TelemetryReport:
    vols = [s.blur_vol for s in signals]
    relatives = [s.blur_relative for s in signals]
    brightness = [s.brightness for s in signals]
    diffs = [s.motion_diff for s in signals if s.motion_diff is not None]

    duplicates = sum(
        1
        for s in signals
        if s.hist_correlation is not None
        and s.hist_correlation > DUPLICATE_CORRELATION
        and (s.motion_diff or 0.0) < DUPLICATE_MAX_DIFF
    )

    metadata = manifest.get("metadata_files") or {}
    capture_settings = manifest.get("capture_settings") or {}
    video_meta = manifest.get("video") or {}

    return TelemetryReport(
        frame_count=len(signals),
        vol_mean=float(np.mean(vols)),
        vol_p10=percentile(vols, 10),
        vol_median=percentile(vols, 50),
        blurry_frame_ratio=float(np.mean([r < blur_relative_threshold for r in relatives])),
        brightness_mean=float(np.mean(brightness)),
        overexposed_ratio=float(np.mean([s.overexposed_ratio for s in signals])),
        underexposed_ratio=float(np.mean([s.underexposed_ratio for s in signals])),
        too_dark_ratio=float(np.mean([b < 45 for b in brightness])),
        too_bright_ratio=float(
            np.mean(
                [
                    b > 220 or s.overexposed_ratio > 0.12
                    for b, s in zip(brightness, signals, strict=True)
                ]
            )
        ),
        motion_mean=float(np.mean(diffs)) if diffs else 0.0,
        too_fast_ratio=float(np.mean([d > FAST_MOTION_DIFF for d in diffs])) if diffs else 0.0,
        duplicate_ratio=duplicates / max(1, len(signals) - 1),
        has_intrinsics=bool(metadata.get("intrinsics")),
        has_poses=bool(metadata.get("poses")),
        has_imu=bool(metadata.get("imu")),
        exposure_locked=capture_settings.get("exposure_locked"),
        white_balance_locked=capture_settings.get("white_balance_locked"),
        video=VideoInfo(
            duration_sec=video_meta.get("duration_sec"),
            width=video_meta.get("width"),
            height=video_meta.get("height"),
            fps=fps or video_meta.get("fps"),
            codec=video_meta.get("codec", "unknown"),
            bitrate=video_meta.get("bitrate"),
            has_audio=bool(video_meta.get("has_audio")),
        ),
        signals=signals,
    )


def score_report(report: TelemetryReport) -> TelemetryReport:
    """Attach warnings, recommendations, and a heuristic 0-100 score.

    The weights below are **not** empirically validated. They encode the
    literature's rank ordering of failure modes -- blur and exposure drift cost
    the most measured PSNR -- but the mapping from these ratios to a single
    number is an open question, and calibrating it against measured
    reconstruction quality is the point of the benchmark work.
    """
    warnings: list[str] = []
    recommendations: list[str] = []

    if not report.has_poses:
        warnings.append("No camera pose metadata found. COLMAP will be required.")
    if not report.has_intrinsics:
        warnings.append("No camera intrinsics metadata found.")
    if report.blurry_frame_ratio > 0.15:
        warnings.append(
            f"{report.blurry_frame_ratio:.0%} of sampled frames are markedly softer than this clip's median."
        )
        recommendations.append("Move more slowly and avoid sudden turns.")
    if report.too_dark_ratio > 0.10:
        warnings.append("A significant portion of sampled frames is underexposed.")
        recommendations.append("Add brighter, more even lighting.")
    if report.too_bright_ratio > 0.10:
        warnings.append("Highlights are clipping in a significant portion of frames.")
        recommendations.append("Reduce exposure so bright surfaces retain detail.")
    if report.too_fast_ratio > 0.12:
        warnings.append("Camera movement may be too fast for stable reconstruction.")
        recommendations.append("Capture with a slow, continuous orbit.")
    if report.duplicate_ratio > 0.25:
        warnings.append("Many sampled frames are near-duplicates.")
        recommendations.append("Keep moving; near-identical frames add no parallax.")
    if report.exposure_locked is not True:
        recommendations.append("Lock exposure before starting the capture.")
    if report.white_balance_locked is not True:
        recommendations.append("Lock white balance before starting the capture.")

    score = 100
    score -= int(report.blurry_frame_ratio * 35)
    score -= int(report.too_dark_ratio * 30)
    score -= int(report.too_fast_ratio * 25)
    score -= int(report.too_bright_ratio * 20)
    score -= int(report.duplicate_ratio * 20)
    score -= 0 if report.has_poses else 10
    score = max(0, min(100, score))

    report.score = score
    report.status = "good" if score >= 75 else "warning" if score >= 45 else "bad"
    report.warnings = warnings
    report.recommendations = sorted(set(recommendations))
    return report


def _sample_indices(total_frames: int, sample_count: int) -> list[int]:
    if total_frames <= 0:
        return list(range(sample_count))
    count = min(sample_count, total_frames)
    return sorted({int(v) for v in np.linspace(0, total_frames - 1, count)})


def _thumbnail(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = THUMBNAIL_MAX_SIDE / max(width, height)
    if scale >= 1.0:
        return frame
    return cv2.resize(
        frame, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA
    )
