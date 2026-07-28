"""Progress reporting.

The core must not know about the job manager, threads, or HTTP polling. It
reports progress through this protocol; the server adapts its ``Job`` to it and
the CLI prints to stderr. Long-running functions accept ``progress`` and default
to :class:`NullProgress`, so every one of them is callable from a test with no
setup.
"""

from __future__ import annotations

import sys
from typing import Protocol, runtime_checkable


@runtime_checkable
class Progress(Protocol):
    """Sink for progress updates from a long-running operation."""

    def update(self, percent: int, message: str | None = None) -> None:
        """Report completion between 0 and 100, optionally naming the step."""

    def log(self, message: str) -> None:
        """Report a line of detail, such as a subprocess's stdout."""


class NullProgress:
    """Discards everything. The default, so the core is usable from a REPL."""

    def update(self, percent: int, message: str | None = None) -> None:
        return None

    def log(self, message: str) -> None:
        return None


class ConsoleProgress:
    """Writes to stderr so that stdout stays clean for machine-readable output.

    This matters for the CLI: ``gausscapture telemetry ... --json`` must be
    pipeable into ``jq`` while still showing progress on a terminal.
    """

    def __init__(self, stream=None, quiet: bool = False):
        self.stream = stream or sys.stderr
        self.quiet = quiet
        self._last = -1

    def update(self, percent: int, message: str | None = None) -> None:
        if self.quiet:
            return
        percent = max(0, min(100, int(percent)))
        # Avoid flooding a terminal with a line per frame.
        if percent == self._last and message is None:
            return
        self._last = percent
        suffix = f"  {message}" if message else ""
        print(f"[{percent:3d}%]{suffix}", file=self.stream, flush=True)

    def log(self, message: str) -> None:
        if self.quiet:
            return
        print(f"       {message}", file=self.stream, flush=True)


class CollectingProgress:
    """Keeps everything in memory. Used by tests to assert on reported steps."""

    def __init__(self) -> None:
        self.updates: list[tuple[int, str | None]] = []
        self.lines: list[str] = []

    def update(self, percent: int, message: str | None = None) -> None:
        self.updates.append((int(percent), message))

    def log(self, message: str) -> None:
        self.lines.append(message)
