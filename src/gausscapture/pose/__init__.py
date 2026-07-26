"""Camera pose estimation.

One protocol, swappable backends. COLMAP is the only implementation today; the
protocol exists because the evaluation plan requires showing that the
capture-telemetry relationship survives a second, independent pose front-end
(``docs/RESEARCH.md`` section 7.4).
"""

from __future__ import annotations

from gausscapture.pose.base import PoseBackend
from gausscapture.pose.colmap import ColmapBackend, colmap_available, run_colmap

__all__ = ["ColmapBackend", "PoseBackend", "colmap_available", "run_colmap"]
