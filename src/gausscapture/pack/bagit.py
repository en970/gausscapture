"""BagIt packaging for distributable CapturePacks.

`BagIt (RFC 8493) <https://datatracker.ietf.org/doc/html/rfc8493>`_ is an
existing, boring, widely implemented answer to "how do I ship a directory of
files with checksums and a little metadata". Adopting it means a CapturePack
can be verified by ``bagit-python``, by an institutional repository, or by
Zenodo's own tooling -- none of which will ever learn a bespoke zip dialect.
This is the concrete form of the decision in ``docs/RESEARCH.md`` section 8 not
to invent a format.

**Where the profile applies.** A bag is a distribution artifact, not a working
directory. Projects on disk keep a flat ``capturepack/`` because every pipeline
stage reads and rewrites it, and a checksum manifest that is stale the moment
you extract a frame is worse than no manifest. The bag is produced on export
and unwrapped on import, so the guarantees land where files actually travel
between machines.

Layout produced::

    <session>.capturepack           (a zip)
      bagit.txt                     declares version and encoding
      bag-info.txt                  metadata, including Payload-Oxum
      manifest-sha256.txt           one line per payload file
      tagmanifest-sha256.txt        checksums of the tag files themselves
      data/                         the pack: manifest.json, video/, camera/, ...
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

from gausscapture.errors import CaptureFormatError
from gausscapture.progress import NullProgress, Progress
from gausscapture.util.hash import sha256_file

BAGIT_VERSION = "1.0"
PAYLOAD_DIR = "data"

_MANIFEST_LINE = re.compile(r"^([0-9a-fA-F]{64})\s+(.+)$")


def is_bag(directory: Path) -> bool:
    """Whether a directory looks like a BagIt bag."""
    return (directory / "bagit.txt").exists() and (directory / PAYLOAD_DIR).is_dir()


def write_bag(
    payload_dir: Path,
    bag_dir: Path,
    info: dict[str, str] | None = None,
    progress: Progress | None = None,
) -> Path:
    """Copy ``payload_dir`` into ``bag_dir/data`` and write the tag files.

    The payload is hardlinked where the filesystem allows it, so bagging a
    four-gigabyte capture does not cost four gigabytes.
    """
    import os
    import shutil

    progress = progress or NullProgress()
    payload_dir = Path(payload_dir)
    bag_dir = Path(bag_dir)
    if not payload_dir.is_dir():
        raise CaptureFormatError(f"Payload directory does not exist: {payload_dir}")

    data_dir = bag_dir / PAYLOAD_DIR
    if bag_dir.exists():
        shutil.rmtree(bag_dir)
    data_dir.mkdir(parents=True)

    sources = [p for p in sorted(payload_dir.rglob("*")) if p.is_file()]
    if not sources:
        raise CaptureFormatError(f"Nothing to bag: {payload_dir} contains no files")

    total_bytes = 0
    manifest_lines: list[str] = []
    for i, source in enumerate(sources):
        relative = source.relative_to(payload_dir)
        destination = data_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        total_bytes += destination.stat().st_size
        manifest_lines.append(f"{sha256_file(destination)}  {PAYLOAD_DIR}/{relative.as_posix()}")
        progress.update(
            int(85 * (i + 1) / len(sources)), f"Bagged {i + 1}/{len(sources)} files"
        )

    (bag_dir / "bagit.txt").write_text(
        f"BagIt-Version: {BAGIT_VERSION}\nTag-File-Character-Encoding: UTF-8\n",
        encoding="utf-8",
    )
    (bag_dir / "manifest-sha256.txt").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )
    _write_bag_info(bag_dir, total_bytes, len(sources), info or {})
    _write_tagmanifest(bag_dir)

    progress.update(100, "Bag written")
    return bag_dir


def validate_bag(bag_dir: Path, complete_only: bool = False) -> dict[str, Any]:
    """Verify a bag's structure and, unless ``complete_only``, its checksums.

    ``complete_only`` skips re-hashing, which matters for multi-gigabyte
    captures where the caller only wants to know whether files are present.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not (bag_dir / "bagit.txt").exists():
        return {"valid": False, "errors": ["bagit.txt is missing"], "warnings": [], "checked": 0}
    data_dir = bag_dir / PAYLOAD_DIR
    if not data_dir.is_dir():
        return {"valid": False, "errors": ["data/ payload directory is missing"], "warnings": [], "checked": 0}

    manifest_path = bag_dir / "manifest-sha256.txt"
    if not manifest_path.exists():
        return {
            "valid": False,
            "errors": ["manifest-sha256.txt is missing"],
            "warnings": [],
            "checked": 0,
        }

    recorded: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = _MANIFEST_LINE.match(line.strip())
        if not match:
            warnings.append(f"Unparseable manifest line: {line[:60]}")
            continue
        recorded[match.group(2)] = match.group(1).lower()

    checked = 0
    for relpath, digest in recorded.items():
        target = bag_dir / relpath
        if not target.exists():
            errors.append(f"Manifest lists a missing file: {relpath}")
            continue
        if not complete_only:
            checked += 1
            if sha256_file(target) != digest:
                errors.append(f"Checksum mismatch: {relpath}")

    # A file present in the payload but absent from the manifest is a bag
    # completeness failure under RFC 8493, not a warning.
    on_disk = {
        f"{PAYLOAD_DIR}/{p.relative_to(data_dir).as_posix()}"
        for p in data_dir.rglob("*")
        if p.is_file()
    }
    for extra in sorted(on_disk - set(recorded)):
        errors.append(f"Payload file is not listed in the manifest: {extra}")

    oxum = _read_bag_info(bag_dir).get("Payload-Oxum")
    if oxum:
        expected_bytes = sum((bag_dir / rel).stat().st_size for rel in recorded if (bag_dir / rel).exists())
        actual = f"{expected_bytes}.{len(on_disk)}"
        if oxum != actual:
            warnings.append(f"Payload-Oxum says {oxum} but the payload measures {actual}")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "checked": checked,
        "files": len(recorded),
    }


def payload_dir(bag_dir: Path) -> Path:
    """The directory holding a bag's actual content."""
    return bag_dir / PAYLOAD_DIR


def _write_bag_info(bag_dir: Path, total_bytes: int, file_count: int, extra: dict[str, str]) -> None:
    from gausscapture import __version__

    fields = {
        "Bag-Software-Agent": f"GaussCapture {__version__} (https://github.com/en970/gausscapture)",
        "Bagging-Date": date.today().isoformat(),
        # Payload-Oxum is BagIt's cheap integrity check: total octets and file
        # count, verifiable without hashing anything.
        "Payload-Oxum": f"{total_bytes}.{file_count}",
    }
    fields.update({k: v for k, v in extra.items() if v})
    lines = [f"{key}: {_fold(value)}" for key, value in fields.items()]
    (bag_dir / "bag-info.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_bag_info(bag_dir: Path) -> dict[str, str]:
    path = bag_dir / "bag-info.txt"
    if not path.exists():
        return {}
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line and not line.startswith((" ", "\t")):
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
    return fields


def _write_tagmanifest(bag_dir: Path) -> None:
    """Checksum the tag files, so the metadata is verifiable too."""
    lines = []
    for name in ("bagit.txt", "bag-info.txt", "manifest-sha256.txt"):
        path = bag_dir / name
        if path.exists():
            lines.append(f"{sha256_file(path)}  {name}")
    (bag_dir / "tagmanifest-sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fold(value: str) -> str:
    """Collapse newlines; bag-info values are single logical lines."""
    return " ".join(str(value).split())
