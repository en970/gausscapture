"""Taking captures off a phone over adb.

The operator's routine is to record often and plug the phone in afterwards, so
retrieval has to be one command that knows what it has already seen. It cannot
be a file-manager drag: the app writes under ``Android/data``, which Samsung's
One UI has hidden from both the on-device file manager and from MTP since
Android 11 (``docs/ANDROID_AUDIT.md`` §3.1). adb still reaches it.

Two properties matter more than speed.

**Nothing is pulled twice.** Every capture carries a ``session_id`` in its
manifest, and a session already present locally is skipped. The id rather than
the directory name is the key, because directory names are derived from a
one-second clock and can collide.

**A capture is never half-imported.** Each session lands in a temporary
directory, is checked for the files that make it usable, and only then is moved
into place. An interrupted pull leaves the phone untouched and nothing partial
on disk, so running the command again simply resumes.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from gausscapture.errors import DependencyMissingError, GaussCaptureError
from gausscapture.progress import NullProgress, Progress

#: Where the capture app writes on the device.
DEVICE_ROOT = "/sdcard/Android/data/com.gausscapture.capture/files"

#: Written back to the phone once a capture is verifiably on this computer. The app reads it to
#: show which takes are still the only copy.
OFFLOAD_SENTINEL = "offloaded.json"

#: A session is only worth importing if it has video to reconstruct from and a
#: manifest to describe it. The rest is recorded as missing rather than fatal,
#: so a capture cut short by a flat battery can still be looked at.
REQUIRED = ("video.mp4", "manifest.json")
EXPECTED = ("imu.jsonl", "frames.jsonl", "intrinsics.json")


@dataclass
class RemoteCapture:
    """A capture sitting on the phone."""

    name: str                       # directory name on the device
    session_id: str | None = None
    offloaded: bool = False
    preset: str | None = None
    created_at: str | None = None
    bytes: int = 0
    files: dict[str, int] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return all(name in self.files for name in REQUIRED)

    @property
    def missing(self) -> list[str]:
        return [n for n in (*REQUIRED, *EXPECTED) if n not in self.files]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "session_id": self.session_id,
            "preset": self.preset,
            "created_at": self.created_at,
            "bytes": self.bytes,
            "offloaded": self.offloaded,
            "files": self.files,
            "usable": self.usable,
            "missing": self.missing,
        }


def adb_available(adb: str = "adb") -> bool:
    return bool(shutil.which(adb))


def _adb(args: list[str], adb: str = "adb", serial: str | None = None,
         timeout: int = 600) -> subprocess.CompletedProcess:
    command = [adb]
    if serial:
        command += ["-s", serial]
    command += args
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout)


def devices(adb: str = "adb") -> list[str]:
    """Serial numbers of attached, authorised devices.

    A phone that is plugged in but has not had USB debugging accepted shows up
    as ``unauthorized``; reporting it as absent would send the operator looking
    for a cable fault instead of at the dialog on their screen.
    """
    if not adb_available(adb):
        raise DependencyMissingError(
            "adb", "Install Android platform-tools, then enable USB debugging on the phone."
        )
    result = _adb(["devices"], adb, timeout=30)
    attached, unauthorised = [], []
    for line in result.stdout.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        if parts[1] == "device":
            attached.append(parts[0])
        elif parts[1] in ("unauthorized", "offline"):
            unauthorised.append(parts[0])

    if not attached and unauthorised:
        raise GaussCaptureError(
            "The phone is connected but has not authorised this computer. "
            "Unlock it and accept the 'Allow USB debugging' prompt."
        )
    return attached


def list_captures(adb: str = "adb", serial: str | None = None,
                  root: str = DEVICE_ROOT) -> list[RemoteCapture]:
    """Everything the app has recorded on the device."""
    # One shell invocation for the whole listing: a round trip per session adds
    # up quickly once there are dozens, and this is run on every plug-in.
    result = _adb(["shell", f"ls -1 {root} 2>/dev/null"], adb, serial, timeout=60)
    names = [n.strip() for n in result.stdout.splitlines() if n.strip()]
    if not names:
        return []

    captures: list[RemoteCapture] = []
    for name in sorted(names):
        listing = _adb(
            ["shell", f"ls -l {root}/{name} 2>/dev/null"], adb, serial, timeout=60
        )
        files: dict[str, int] = {}
        for line in listing.stdout.splitlines():
            parts = line.split()
            if len(parts) < 8 or line.startswith("total"):
                continue
            try:
                files[parts[-1]] = int(parts[4])
            except ValueError:
                continue
        if not files:
            continue

        capture = RemoteCapture(
            name=name,
            files=files,
            bytes=sum(files.values()),
            offloaded=OFFLOAD_SENTINEL in files,
        )
        _read_manifest(capture, root, adb, serial)
        captures.append(capture)
    return captures


def _read_manifest(capture: RemoteCapture, root: str, adb: str, serial: str | None) -> None:
    """Fill in what the manifest says, without failing if it cannot be read."""
    if "manifest.json" not in capture.files:
        return
    result = _adb(
        ["shell", f"cat {root}/{capture.name}/manifest.json"], adb, serial, timeout=60
    )
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return
    capture.session_id = data.get("session_id")
    capture.preset = data.get("preset")
    capture.created_at = data.get("created_at")


def known_session_ids(store) -> set[str]:
    """Session ids already imported, read from the projects on disk."""
    known: set[str] = set()
    for project in store.list():
        manifest = Path(project.path) / "capturepack" / "manifest.json"
        if not manifest.exists():
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        session = data.get("session_id")
        if session:
            known.add(str(session))
    return known


def pull_capture(
    capture: RemoteCapture,
    destination: Path,
    adb: str = "adb",
    serial: str | None = None,
    root: str = DEVICE_ROOT,
    progress: Progress | None = None,
) -> Path:
    """Copy one capture off the phone, arriving complete or not at all."""
    progress = progress or NullProgress()
    destination = Path(destination)
    if destination.exists():
        raise GaussCaptureError(f"{destination} already exists")

    # Staged alongside the destination rather than in the system temp
    # directory, so the final move is a rename within one filesystem instead of
    # a second copy of a file that is routinely hundreds of megabytes.
    created_parent = not destination.parent.exists()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="gausscapture-pull-", dir=destination.parent))
    try:
        result = _adb(["pull", f"{root}/{capture.name}", str(staging / capture.name)],
                      adb, serial, timeout=3600)
        if result.returncode != 0:
            raise GaussCaptureError(
                f"adb could not read {capture.name}: {result.stderr.strip() or 'no detail given'}"
            )

        pulled = staging / capture.name
        missing = [name for name in REQUIRED if not (pulled / name).exists()]
        if missing:
            raise GaussCaptureError(
                f"{capture.name} arrived without {', '.join(missing)}, so it was not imported"
            )

        shutil.move(str(pulled), str(destination))
        progress.log(f"  {capture.name}  {_size(capture.bytes)}")
        created_parent = False
        return destination
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        if created_parent:
            # A failed pull should not leave a directory that later reads as an
            # imported capture.
            try:
                destination.parent.rmdir()
            except OSError:
                pass


def mark_offloaded(
    capture: RemoteCapture,
    host: str,
    when: str,
    adb: str = "adb",
    serial: str | None = None,
    root: str = DEVICE_ROOT,
) -> bool:
    """Tell the phone this capture is safely on a computer.

    Recording all day produces dozens of takes, and the only way to know which are still the sole
    copy is to write the answer where the phone can read it. The app shows a filled dot per take
    from exactly this file, so without it the interface promises a distinction it cannot make.

    Written after the pull has been verified, never before: a sentinel that arrives first would
    mark a capture safe that is not.
    """
    payload = json.dumps(
        {"host": host, "copied_at": when, "tool": "gausscapture pull"}, indent=2
    )
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        handle.write(payload)
        local = Path(handle.name)
    try:
        result = _adb(
            ["push", str(local), f"{root}/{capture.name}/{OFFLOAD_SENTINEL}"],
            adb, serial, timeout=120,
        )
        return result.returncode == 0
    finally:
        local.unlink(missing_ok=True)


def _size(count: int) -> str:
    if count >= 1 << 30:
        return f"{count / (1 << 30):.1f} GB"
    if count >= 1 << 20:
        return f"{count / (1 << 20):.0f} MB"
    return f"{count / 1024:.0f} KB"
