"""Exception hierarchy.

Callers should be able to distinguish "you have not run the previous step yet"
from "your machine is missing COLMAP" from "this archive is malformed", because
those three need different messages in a UI and different exit codes in a CLI.
"""

from __future__ import annotations


class GaussCaptureError(Exception):
    """Base class for every error this package raises deliberately."""


class DependencyMissingError(GaussCaptureError):
    """An external binary or optional Python package is not available.

    Raised for ffmpeg, ffprobe, COLMAP and Open3D. The message should name the
    dependency and how to install it, because this is the most common failure
    for a first-time user.
    """


class CaptureFormatError(GaussCaptureError):
    """A CapturePack or media file is missing, malformed, or unreadable."""


class PipelineStateError(GaussCaptureError):
    """A pipeline step was invoked before its inputs existed.

    For example, running COLMAP before extracting frames.
    """
