"""GaussCapture: local-first phone capture for 3D Gaussian splatting.

The public surface of this package is deliberately small and free of any web
framework. Everything here is importable and testable without starting a
server, which is what lets the FastAPI and React layers be thin clients over
it -- and what makes the capture-telemetry work independently citable.

Typical use::

    from gausscapture import analyze_capture, extract_frames, Project

    report = analyze_capture(project.path)
    print(report.vol_p10, report.score)
"""

from __future__ import annotations

__version__ = "0.3.0"

from gausscapture.errors import (
    CaptureFormatError,
    DependencyMissingError,
    GaussCaptureError,
    PipelineStateError,
)
from gausscapture.progress import ConsoleProgress, NullProgress, Progress
from gausscapture.project import Project, ProjectStore
from gausscapture.types import (
    FrameIndex,
    FrameSignals,
    PoseReport,
    TelemetryReport,
    VideoInfo,
)

__all__ = [
    "__version__",
    "CaptureFormatError",
    "ConsoleProgress",
    "DependencyMissingError",
    "FrameIndex",
    "FrameSignals",
    "GaussCaptureError",
    "NullProgress",
    "PipelineStateError",
    "PoseReport",
    "Progress",
    "Project",
    "ProjectStore",
    "TelemetryReport",
    "VideoInfo",
]
