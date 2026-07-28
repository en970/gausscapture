from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from gausscapture.progress import Progress
from gausscapture.types import PoseReport


@runtime_checkable
class PoseBackend(Protocol):
    """Solves camera poses for a project's extracted frames.

    Implementations write their output under the project in whatever native
    format they use, and return a :class:`~gausscapture.types.PoseReport`
    naming the directory a trainer should read.
    """

    name: str

    def available(self) -> bool:
        """Whether this backend can run on this machine right now."""

    def solve(self, project_path: Path, progress: Progress | None = None) -> PoseReport:
        """Estimate poses, returning a report with the model directory."""
