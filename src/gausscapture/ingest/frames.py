from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import cv2

from gausscapture.errors import CaptureFormatError
from gausscapture.pack.manifest import find_main_video
from gausscapture.progress import NullProgress, Progress
from gausscapture.telemetry.signals import (
    DEFAULT_BLUR_RELATIVE_THRESHOLD,
    RollingBlurNormaliser,
    frame_signals,
)
from gausscapture.types import FrameIndex, FrameSignals

#: Histogram correlation above which a frame is a duplicate *candidate*.
DUPLICATE_CORRELATION = 0.996
#: ...but only if it also barely differs pixel-wise from the last kept frame.
#: Histograms are invariant to spatial rearrangement: a camera translating
#: through a static scene produces near-identical histograms frame after frame
#: while every pixel moves. Judged on correlation alone, the filter discards
#: exactly the frames that carry parallax -- measured on a synthetic scene, it
#: kept 3 of 12 well-spaced views. The pixel-difference gate is what actually
#: distinguishes "camera did not move" from "scene has stable lighting".
DUPLICATE_MAX_PIXEL_DIFF = 2.0

FRAME_PRESETS: dict[str, dict[str, Any]] = {
    "fast": {"target_fps": 1, "max_frames": 200, "resize_max_side": 1280},
    "balanced": {"target_fps": 2, "max_frames": 600, "resize_max_side": 1920},
    "dense": {"target_fps": 4, "max_frames": 1200, "resize_max_side": None},
}


def extract_frames(
    project_path: Path,
    settings: dict[str, Any] | None = None,
    progress: Progress | None = None,
) -> FrameIndex:
    """Decode the capture's video into training images.

    Two filters run while decoding:

    * **Blur** -- a frame is rejected when its sharpness falls below
      ``blur_relative_threshold`` times the rolling median of nearby frames.
      This is deliberately relative: an absolute variance-of-Laplacian cutoff
      throws away sharp frames of plain walls and keeps blurred frames of
      cluttered shelves, because the measure scales with scene texture and
      resolution.
    * **Duplicates** -- consecutive frames whose histograms are nearly
      identical add cost without adding parallax.

    Every sampled frame is recorded in the index, kept or not, with the signals
    that decided it. Rejections are data for the capture-quality study, so they
    are never silently dropped.
    """
    progress = progress or NullProgress()
    settings = {**FRAME_PRESETS["balanced"], **(settings or {})}

    video_path = find_main_video(project_path / "capturepack")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise CaptureFormatError(f"Video could not be opened for extraction: {video_path}")

    try:
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        target_fps = settings.get("target_fps", 2)
        take_every = 1 if target_fps in ("all", None) else max(1, round(source_fps / float(target_fps)))

        max_frames = settings.get("max_frames", 600)
        frame_budget = math.inf if max_frames in (None, "unlimited", 0) else int(max_frames)

        resize_max_side = settings.get("resize_max_side")
        if resize_max_side in ("original", None, 0):
            resize_max_side = None

        blur_filter = bool(settings.get("blur_filter", True))
        duplicate_filter = bool(settings.get("duplicate_filter", True))
        blur_threshold = float(
            settings.get("blur_relative_threshold", DEFAULT_BLUR_RELATIVE_THRESHOLD)
        )

        out_dir = project_path / "frames" / "images"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        records: list[FrameSignals] = []
        normaliser = RollingBlurNormaliser()
        prev_hist = None
        prev_kept_gray = None
        written = 0
        skipped_blur = 0
        skipped_duplicate = 0
        frame_no = -1

        while written < frame_budget:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            if frame_no % take_every != 0:
                continue

            # Motion is measured against the last *kept* frame, not the last
            # sampled one: after a rejected duplicate, the next frame must be
            # compared with what is actually in the output set.
            raw = frame_signals(frame, prev_kept_gray, prev_hist)
            gray = raw.pop("_gray")
            hist = raw.pop("_hist")
            relative = normaliser.push(raw["blur_vol"])

            keep = True
            if blur_filter and normaliser.ready and relative < blur_threshold:
                keep = False
                skipped_blur += 1
            if keep and duplicate_filter and prev_hist is not None:
                correlation = raw.get("hist_correlation")
                motion = raw.get("motion_diff")
                if (
                    correlation is not None
                    and correlation > DUPLICATE_CORRELATION
                    and motion is not None
                    and motion < DUPLICATE_MAX_PIXEL_DIFF
                ):
                    keep = False
                    skipped_duplicate += 1

            relpath = ""
            if keep:
                image = _resize(frame, resize_max_side) if resize_max_side else frame
                written += 1
                relpath = str(Path("frames") / "images" / f"frame_{written:06d}.jpg")
                cv2.imwrite(str(project_path / relpath), image, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
                prev_hist = hist
                prev_kept_gray = gray

            records.append(
                FrameSignals(
                    index=frame_no,
                    timestamp_sec=frame_no / source_fps if source_fps else None,
                    blur_vol=raw["blur_vol"],
                    blur_relative=relative,
                    brightness=raw["brightness"],
                    overexposed_ratio=raw["overexposed_ratio"],
                    underexposed_ratio=raw["underexposed_ratio"],
                    motion_diff=raw.get("motion_diff"),
                    hist_correlation=raw.get("hist_correlation"),
                    used=keep,
                    file=relpath,
                )
            )

            if total:
                progress.update(
                    5 + int(90 * min(1.0, frame_no / total)), f"Kept {written} of {len(records)} frames"
                )
    finally:
        cap.release()

    index = FrameIndex(
        source_video=video_path.name,
        source_fps=source_fps,
        frames_total_sampled=len(records),
        frames_used=written,
        frames_skipped_blur=skipped_blur,
        frames_skipped_duplicate=skipped_duplicate,
        extraction_settings=settings,
        frames=records,
    )

    out = project_path / "frames" / "frame_index.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index.to_dict(), indent=2), encoding="utf-8")
    progress.update(100, f"Extracted {written} usable frames")
    return index


def _resize(frame, max_side: int):
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return frame
    scale = max_side / longest
    return cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
