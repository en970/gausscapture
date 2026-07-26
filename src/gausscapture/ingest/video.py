from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from gausscapture.config import Settings, get_settings
from gausscapture.errors import CaptureFormatError
from gausscapture.types import VideoInfo

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def probe_video(video_path: Path, settings: Settings | None = None) -> VideoInfo:
    """Read container metadata, preferring ffprobe and falling back to OpenCV.

    ffprobe knows the codec, bitrate, and audio streams; OpenCV knows none of
    those but is always installed alongside us. Falling back rather than
    failing means a user without ffmpeg still gets a working import, just with
    less metadata in their manifest.
    """
    settings = settings or get_settings()
    cmd = [
        settings.ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=duration,bit_rate:stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
        data = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return _probe_with_opencv(video_path)

    streams = data.get("streams", [])
    video_stream: dict[str, Any] = next(
        (s for s in streams if s.get("codec_type") == "video"), {}
    )
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = data.get("format", {})

    return VideoInfo(
        duration_sec=_as_float(fmt.get("duration")),
        width=video_stream.get("width"),
        height=video_stream.get("height"),
        fps=_parse_rate(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")),
        codec=video_stream.get("codec_name") or "unknown",
        bitrate=_as_int(fmt.get("bit_rate")),
        has_audio=audio_stream is not None,
    )


def _probe_with_opencv(video_path: Path) -> VideoInfo:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise CaptureFormatError(f"Video cannot be opened: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        return VideoInfo(
            duration_sec=(frame_count / fps) if fps else None,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None,
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None,
            fps=fps or None,
            codec="unknown",
            bitrate=None,
            has_audio=False,
        )
    finally:
        cap.release()


def copy_video_into_pack(source: Path, project_path: Path) -> Path:
    """Copy a source video to ``capturepack/video/main_video.<ext>``."""
    if not source.exists():
        raise CaptureFormatError(f"Source video does not exist: {source}")
    video_dir = project_path / "capturepack" / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    target = video_dir / f"main_video{source.suffix.lower()}"
    shutil.copy2(source, target)
    return target


def _parse_rate(rate: str | None) -> float | None:
    """Parse ffprobe's rational frame rates such as ``30000/1001``."""
    if not rate or rate == "0/0":
        return None
    try:
        if "/" in rate:
            num, den = rate.split("/", 1)
            denominator = float(den)
            return float(num) / denominator if denominator else None
        return float(rate)
    except ValueError:
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
