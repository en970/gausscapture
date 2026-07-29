"""Building a local, offline report from a benchmark run.

Two things are worth looking at after a run, and neither is a number: the
reconstruction itself, and how the captures compare. This package writes both
into a folder of static files that opens from disk with no server, no network,
and no build step -- which matters because the interesting reconstructions live
on the machine that produced them and are far too large to upload.
"""

from __future__ import annotations

from gausscapture.report.scene import export_scene
from gausscapture.report.site import build_report

__all__ = ["build_report", "export_scene"]
