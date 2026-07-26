from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import cv2

from backend.core.capturepack import find_main_video
from backend.core.job_manager import Job


def extract_frames(job: Job, project_path: Path, settings: dict[str, Any]) -> dict[str, Any]:
    video_path = find_main_video(project_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Video could not be opened for frame extraction")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target_fps = settings.get("target_fps", 2)
    all_frames = target_fps == "all"
    interval = 1 if all_frames else max(1, int(round(source_fps / float(target_fps))))
    max_frames = settings.get("max_frames", 600)
    max_frames_int = math.inf if max_frames in (None, "unlimited", 0) else int(max_frames)
    resize_max_side = settings.get("resize_max_side")
    blur_filter = bool(settings.get("blur_filter", True))
    duplicate_filter = bool(settings.get("duplicate_filter", True))

    out_dir = project_path / "frames" / "images"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_index: list[dict[str, Any]] = []
    used_count = 0
    skipped_blur = 0
    skipped_duplicate = 0
    frame_no = 0
    prev_hist = None
    written_id = 1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_no % interval != 0:
            frame_no += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        cv2.normalize(hist, hist)
        used = True
        if blur_filter and blur_score < 60:
            used = False
            skipped_blur += 1
        if used and duplicate_filter and prev_hist is not None:
            similarity = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if similarity > 0.996:
                used = False
                skipped_duplicate += 1
        rel = ""
        if used:
            if resize_max_side and resize_max_side != "original":
                frame = _resize_max_side(frame, int(resize_max_side))
            rel_path = Path("frames") / "images" / f"frame_{written_id:06d}.jpg"
            cv2.imwrite(str(project_path / rel_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
            rel = str(rel_path)
            prev_hist = hist
            used_count += 1
            written_id += 1
        frame_index.append(
            {
                "id": len(frame_index),
                "file": rel,
                "timestamp_sec": frame_no / source_fps if source_fps else None,
                "blur_score": blur_score,
                "brightness": brightness,
                "used": used,
            }
        )
        if used_count >= max_frames_int:
            break
        if total:
            job.set_progress(5 + int(90 * (frame_no / total)), f"Extracted {used_count} usable frames")
        frame_no += 1

    cap.release()
    index = {
        "source_video": str(video_path.relative_to(project_path / "capturepack")),
        "extraction_settings": settings,
        "source_fps": source_fps,
        "frames_total_sampled": len(frame_index),
        "frames_used": used_count,
        "frames_skipped_blur": skipped_blur,
        "frames_skipped_duplicate": skipped_duplicate,
        "frames": frame_index,
    }
    out = project_path / "frames" / "frame_index.json"
    out.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return {"frame_index": str(out), "frames_used": used_count}


def _resize_max_side(frame, max_side: int):
    h, w = frame.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return frame
    scale = max_side / side
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

