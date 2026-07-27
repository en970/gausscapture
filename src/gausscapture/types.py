"""Data types passed between pipeline stages.

These are plain dataclasses rather than Pydantic models: the core has no
business depending on a web framework's validation library, and the server can
wrap them for its own schemas. Every one of them round-trips through
``to_dict``/``from_dict`` so intermediate results can be written to disk and
re-read by a later stage or a benchmark run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class VideoInfo:
    """Container-level facts about a source video.

    ``width`` and ``height`` are **display** dimensions: what a decoder actually
    hands you, after any rotation the container asks for. A phone recording in
    portrait stores a landscape frame plus a 90-degree rotation flag, so the
    coded size and the decoded size differ. OpenCV applies the flag, so
    reporting the coded size here would describe frames nobody ever sees, and
    any camera intrinsics matched against it would be transposed.
    ``rotation`` keeps the raw value for anyone who needs it.
    """

    duration_sec: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    codec: str = "unknown"
    bitrate: int | None = None
    has_audio: bool = False
    rotation: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoInfo:
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class FrameSignals:
    """Per-frame capture telemetry.

    ``blur_vol`` is the raw variance of the Laplacian. It is **not** comparable
    across scenes -- a sharp photograph of a blank wall scores lower than a
    blurred photograph of a bookshelf, and it scales with resolution. Use
    ``blur_relative`` for any decision; it divides by a rolling median of the
    same clip, which cancels both effects.
    """

    index: int
    timestamp_sec: float | None = None
    blur_vol: float = 0.0
    blur_relative: float = 1.0
    brightness: float = 0.0
    overexposed_ratio: float = 0.0
    underexposed_ratio: float = 0.0
    motion_diff: float | None = None
    hist_correlation: float | None = None
    used: bool = True
    file: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TelemetryReport:
    """Aggregated capture telemetry plus a heuristic quality score.

    The aggregate fields are the candidate predictors named in the project's
    pre-registered hypothesis (see ``docs/RESEARCH.md`` section 9). They are
    reported individually and unweighted so a later study can fit them against
    measured reconstruction quality, rather than being collapsed into the score
    and lost.

    ``score`` is explicitly a *heuristic*, not a validated prediction. Its
    weights are not empirically grounded yet; that is the open research
    question, not a solved one.
    """

    frame_count: int = 0
    vol_mean: float = 0.0
    vol_p10: float = 0.0
    vol_median: float = 0.0
    blurry_frame_ratio: float = 0.0
    brightness_mean: float = 0.0
    overexposed_ratio: float = 0.0
    underexposed_ratio: float = 0.0
    too_dark_ratio: float = 0.0
    too_bright_ratio: float = 0.0
    motion_mean: float = 0.0
    too_fast_ratio: float = 0.0
    duplicate_ratio: float = 0.0
    has_intrinsics: bool = False
    has_poses: bool = False
    has_imu: bool = False
    exposure_locked: bool | None = None
    white_balance_locked: bool | None = None
    video: VideoInfo = field(default_factory=VideoInfo)
    score: int = 0
    status: str = "unknown"
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    thumbnails: list[str] = field(default_factory=list)
    signals: list[FrameSignals] = field(default_factory=list)

    def to_dict(self, include_signals: bool = True) -> dict[str, Any]:
        data = asdict(self)
        if not include_signals:
            data.pop("signals", None)
        return data

    def summary(self) -> str:
        """One-line human summary, used by the CLI."""
        return (
            f"score {self.score}/100 ({self.status})  "
            f"VoL p10 {self.vol_p10:.1f}  blurry {self.blurry_frame_ratio:.0%}  "
            f"dup {self.duplicate_ratio:.0%}  frames {self.frame_count}"
        )


@dataclass
class FrameIndex:
    """Result of frame extraction."""

    source_video: str = ""
    source_fps: float = 0.0
    frames_total_sampled: int = 0
    frames_used: int = 0
    frames_skipped_blur: int = 0
    frames_skipped_duplicate: int = 0
    extraction_settings: dict[str, Any] = field(default_factory=dict)
    frames: list[FrameSignals] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PoseReport:
    """Result of a pose-estimation backend."""

    method: str = "unknown"
    images_total: int = 0
    images_registered: int = 0
    registered_ratio: float = 0.0
    sparse_points: int = 0
    status: str = "unknown"
    model_dir: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
