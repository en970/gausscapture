from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from gausscapture.errors import CaptureFormatError
from gausscapture.pack.manifest import read_manifest, validate
from gausscapture.progress import NullProgress, Progress
from gausscapture.util.hash import sha256_file


def write_checksums(pack_dir: Path, progress: Progress | None = None) -> Path:
    """Hash every file in the pack except the checksum file itself.

    On a multi-gigabyte video this is I/O bound and can take a while, hence the
    progress reporting.
    """
    progress = progress or NullProgress()
    files = [
        f
        for f in sorted(pack_dir.rglob("*"))
        if f.is_file() and "checksums" not in f.relative_to(pack_dir).parts
    ]
    checksums: dict[str, str] = {}
    for i, file in enumerate(files):
        checksums[str(file.relative_to(pack_dir))] = sha256_file(file)
        progress.update(int(100 * (i + 1) / max(1, len(files))), f"Hashed {file.name}")

    checksums_dir = pack_dir / "checksums"
    checksums_dir.mkdir(parents=True, exist_ok=True)
    out = checksums_dir / "sha256.json"
    out.write_text(json.dumps(checksums, indent=2), encoding="utf-8")
    return out


def verify_checksums(pack_dir: Path) -> dict[str, Any]:
    """Re-hash and compare. Used by the CLI's ``pack validate --verify``."""
    path = pack_dir / "checksums" / "sha256.json"
    if not path.exists():
        return {"verified": False, "reason": "No checksums/sha256.json in pack", "mismatches": []}
    recorded: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    missing: list[str] = []
    for relpath, digest in recorded.items():
        target = pack_dir / relpath
        if not target.exists():
            missing.append(relpath)
        elif sha256_file(target) != digest:
            mismatches.append(relpath)
    return {
        "verified": not mismatches and not missing,
        "checked": len(recorded),
        "mismatches": mismatches,
        "missing": missing,
    }


def import_archive(project_path: Path, archive_path: Path) -> dict[str, Any]:
    """Extract a ``.capturepack`` into a project, flattening a single wrapper dir.

    Archives produced by zipping a folder (rather than its contents) nest
    everything one level down; users hit this constantly, so we correct it
    rather than rejecting the file.
    """
    if not zipfile.is_zipfile(archive_path):
        raise CaptureFormatError(f"Not a zip archive: {archive_path.name}")

    target = project_path / "capturepack"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as archive:
        _safe_extract(archive, target)

    if not (target / "manifest.json").exists():
        nested = next(target.glob("*/manifest.json"), None)
        if nested:
            _flatten(nested.parent, target, project_path)

    result = validate(target)
    if result["valid"]:
        write_checksums(target)
    return result


def export_archive(project_path: Path, out_path: Path | None = None) -> Path:
    """Zip a project's pack back into a distributable ``.capturepack``."""
    pack_dir = project_path / "capturepack"
    manifest = read_manifest(pack_dir)
    if out_path is None:
        out_path = project_path / f"{manifest.get('session_id', 'session')}.capturepack"
    if out_path.exists():
        out_path.unlink()
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(pack_dir.rglob("*")):
            if file.is_file():
                archive.write(file, file.relative_to(pack_dir))
    return out_path


def _safe_extract(archive: zipfile.ZipFile, target: Path) -> None:
    """Extract, refusing entries that would escape the target directory.

    ``ZipFile.extractall`` does sanitise absolute paths and ``..`` on modern
    Python, but a pack is untrusted input that users download from each other,
    so the check is made explicit rather than assumed.
    """
    target = target.resolve()
    for member in archive.infolist():
        destination = (target / member.filename).resolve()
        if not destination.is_relative_to(target):
            raise CaptureFormatError(f"Archive entry escapes the pack directory: {member.filename}")
    archive.extractall(target)


def _flatten(nested_root: Path, target: Path, project_path: Path) -> None:
    temp = project_path / "_capturepack_nested"
    if temp.exists():
        shutil.rmtree(temp)
    nested_root.rename(temp)
    shutil.rmtree(target)
    temp.rename(target)
