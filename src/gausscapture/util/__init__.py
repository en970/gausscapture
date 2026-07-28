"""Small shared helpers with no dependencies on other GaussCapture modules."""

from __future__ import annotations

from gausscapture.util.hash import sha256_file
from gausscapture.util.log import append_log, read_log, utc_now

__all__ = ["append_log", "read_log", "sha256_file", "utc_now"]
