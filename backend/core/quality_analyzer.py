from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.core.capturepack import find_main_video, read_manifest
from backend.core.job_manager import Job


def analyze_quality(job: Job, project_path: Path, sample_count: int = 80) -> dict[str, Any]:
    video_path = find_main_video(project_path)
    manifest = read_manifest(project_path / "capturepack")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Video could not be opened for quality analysis")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or manifest.get("video", {}).get("fps") or 0)
    indices = _sample_indices(total_frames, sample_count)
    brightness: list[float] = []
    over_ratios: list[float] = []
    under_ratios: list[float] = []
    blur_scores: list[float] = []
    diffs: list[float] = []
    duplicates = 0
    prev_gray = None
    prev_hist = None
    thumbs: list[str] = []
    thumb_dir = project_path / "quality" / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    for pos, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness.append(float(gray.mean()))
        over_ratios.append(float((gray > 245).mean()))
        under_ratios.append(float((gray < 20).mean()))
        blur_scores.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

        if prev_gray is not None:
            diffs.append(float(cv2.absdiff(gray, prev_gray).mean()))
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        cv2.normalize(hist, hist)
        if prev_hist is not None:
            similarity = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if similarity > 0.992 and diffs and diffs[-1] < 2.0:
                duplicates += 1
        prev_gray = gray
        prev_hist = hist

        if len(thumbs) < 8:
            thumb = cv2.resize(frame, _thumb_size(frame), interpolation=cv2.INTER_AREA)
            rel = Path("quality") / "thumbnails" / f"thumb_{len(thumbs)+1:02d}.jpg"
            cv2.imwrite(str(project_path / rel), thumb)
            thumbs.append(str(rel))
        job.set_progress(5 + int(80 * ((pos + 1) / max(1, len(indices)))), f"Analyzed frame {pos + 1}/{len(indices)}")

    cap.release()
    if not brightness:
        raise ValueError("No frames could be sampled from video")

    mean_brightness = float(np.mean(brightness))
    too_dark_ratio = float(np.mean([v < 45 for v in brightness]))
    too_bright_ratio = float(np.mean([v > 220 or over > 0.12 for v, over in zip(brightness, over_ratios)]))
    mean_blur = float(np.mean(blur_scores))
    blurry_ratio = float(np.mean([v < 80 for v in blur_scores]))
    mean_motion = float(np.mean(diffs)) if diffs else 0.0
    too_fast_ratio = float(np.mean([v > 35 for v in diffs])) if diffs else 0.0
    duplicate_ratio = duplicates / max(1, len(indices) - 1)
    metadata = manifest.get("metadata_files", {})
    has_audio = bool(manifest.get("video", {}).get("has_audio"))

    warnings: list[str] = []
    recommendations: list[str] = []
    if not metadata.get("poses"):
        warnings.append("No camera pose metadata found. COLMAP will be required.")
    if not metadata.get("intrinsics"):
        warnings.append("No camera intrinsics metadata found.")
    if blurry_ratio > 0.15:
        warnings.append("Some frames may be blurry.")
        recommendations.append("Use slower camera movement and avoid sudden turns.")
    if too_dark_ratio > 0.1:
        warnings.append("A significant portion of sampled frames is dark.")
        recommendations.append("Use brighter, more even lighting.")
    if too_fast_ratio > 0.12:
        warnings.append("Camera movement may be too fast for stable reconstruction.")
        recommendations.append("Capture with slow orbital or walking motion.")
    if duplicate_ratio > 0.25:
        warnings.append("Many sampled frames are very similar.")
        recommendations.append("Move the camera continuously to increase baseline.")
    if manifest.get("capture_settings", {}).get("exposure_locked") is not True:
        recommendations.append("Lock exposure in the capture app when possible.")
    if manifest.get("capture_settings", {}).get("white_balance_locked") is not True:
        recommendations.append("Lock white balance in the capture app when possible.")

    score = 100
    score -= int(too_dark_ratio * 30)
    score -= int(too_bright_ratio * 20)
    score -= int(blurry_ratio * 35)
    score -= int(too_fast_ratio * 25)
    score -= int(duplicate_ratio * 20)
    score -= 10 if not metadata.get("poses") else 0
    score = max(0, min(100, score))
    status = "good" if score >= 75 else "warning" if score >= 45 else "bad"

    report = {
        "overall_score": score,
        "status": status,
        "video": {
            "duration_sec": manifest.get("video", {}).get("duration_sec"),
            "resolution": f"{manifest.get('video', {}).get('width')}x{manifest.get('video', {}).get('height')}",
            "fps": fps,
        },
        "brightness": {
            "mean": mean_brightness,
            "too_dark_ratio": too_dark_ratio,
            "too_bright_ratio": too_bright_ratio,
            "underexposed_pixel_ratio": float(np.mean(under_ratios)),
            "overexposed_pixel_ratio": float(np.mean(over_ratios)),
            "status": "good" if too_dark_ratio < 0.1 and too_bright_ratio < 0.1 else "warning",
        },
        "blur": {
            "mean_laplacian_var": mean_blur,
            "blurry_frame_ratio": blurry_ratio,
            "status": "good" if blurry_ratio < 0.15 else "warning",
        },
        "motion": {
            "mean_frame_diff": mean_motion,
            "too_fast_motion_ratio": too_fast_ratio,
            "duplicate_frame_ratio": duplicate_ratio,
            "status": "good" if too_fast_ratio < 0.12 else "warning",
        },
        "metadata": {
            "has_intrinsics": bool(metadata.get("intrinsics")),
            "has_poses": bool(metadata.get("poses")),
            "has_imu": bool(metadata.get("imu")),
            "has_audio": has_audio,
            "exposure_locked": manifest.get("capture_settings", {}).get("exposure_locked"),
            "white_balance_locked": manifest.get("capture_settings", {}).get("white_balance_locked"),
        },
        "warnings": warnings,
        "recommendations": sorted(set(recommendations)),
        "thumbnails": thumbs,
    }
    out = project_path / "quality" / "quality_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"quality_report": str(out), "report": report}


def _sample_indices(total_frames: int, sample_count: int) -> list[int]:
    if total_frames <= 0:
        return list(range(sample_count))
    count = min(sample_count, total_frames)
    return sorted(set(int(v) for v in np.linspace(0, total_frames - 1, count)))


def _thumb_size(frame: np.ndarray) -> tuple[int, int]:
    h, w = frame.shape[:2]
    max_side = 360
    scale = max_side / max(w, h)
    return max(1, int(w * scale)), max(1, int(h * scale))

