"""Getting pixels and container metadata out of a source video."""

from __future__ import annotations

from gausscapture.ingest.frames import FRAME_PRESETS, extract_frames
from gausscapture.ingest.video import (
    VIDEO_EXTENSIONS,
    copy_video_into_pack,
    is_video_file,
    probe_video,
)

__all__ = [
    "FRAME_PRESETS",
    "VIDEO_EXTENSIONS",
    "copy_video_into_pack",
    "extract_frames",
    "is_video_file",
    "probe_video",
]
