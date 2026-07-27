from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from gausscapture.errors import CaptureFormatError
from gausscapture.pack import bagit
from gausscapture.pack.manifest import read_manifest, validate
from gausscapture.progress import NullProgress, Progress
from gausscapture.util.hash import sha256_file

#: Large, immutable payload worth hardlinking rather than copying on import.
_LINKABLE_MEDIA = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def write_checksums(pack_dir: Path, progress: Progress | None = None) -> Path:
    """Hash every file in a working pack directory.

    This is the in-place, working-directory manifest. The stronger guarantee --
    a BagIt manifest covering a frozen payload -- is applied at export, because
    a checksum file inside a directory that pipeline stages keep rewriting goes
    stale immediately.
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
    """Re-hash a working pack and compare against ``checksums/sha256.json``."""
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


def export_archive(
    project_path: Path,
    out_path: Path | None = None,
    include_dataset: bool = False,
    progress: Progress | None = None,
) -> Path:
    """Write a distributable ``.capturepack`` as a BagIt bag.

    With ``include_dataset``, the extracted frames, the COLMAP model and a
    ``transforms.json`` are added alongside the capture, which makes the
    archive directly trainable: unzip it, point a trainer at ``data/``, and no
    conversion step is involved. Without it the archive carries only the
    capture itself, which is what you want for sharing raw footage.
    """
    progress = progress or NullProgress()
    pack_dir = project_path / "capturepack"
    manifest = read_manifest(pack_dir)

    if out_path is None:
        out_path = project_path / f"{manifest.get('session_id', 'session')}.capturepack"
    out_path = Path(out_path)

    with tempfile.TemporaryDirectory(prefix="gausscapture-bag-") as tmp:
        staging = Path(tmp) / "payload"
        shutil.copytree(pack_dir, staging)
        if include_dataset:
            _stage_dataset(project_path, staging, progress)

        bag_dir = Path(tmp) / "bag"
        bagit.write_bag(
            staging,
            bag_dir,
            info={
                "External-Identifier": str(manifest.get("session_id", "")),
                "Internal-Sender-Identifier": str(manifest.get("session_name", "")),
                "Internal-Sender-Description": (
                    f"GaussCapture CapturePack {manifest.get('capturepack_version', '?')}, "
                    f"capture type {manifest.get('capture_type', 'unknown')}"
                ),
            },
            progress=progress,
        )

        if out_path.exists():
            out_path.unlink()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(bag_dir.rglob("*")):
                if file.is_file():
                    archive.write(file, file.relative_to(bag_dir))

    progress.update(100, f"Archive written to {out_path.name}")
    return out_path


def import_archive(project_path: Path, archive_path: Path) -> dict[str, Any]:
    """Extract a ``.capturepack`` into a project.

    Handles three shapes, because archives arrive from three eras and three
    tools: a BagIt bag, a flat pack, and a pack accidentally zipped with its
    parent folder (which is what happens when you right-click a directory and
    choose Compress).
    """
    if not zipfile.is_zipfile(archive_path):
        raise CaptureFormatError(f"Not a zip archive: {archive_path.name}")

    target = project_path / "capturepack"

    with tempfile.TemporaryDirectory(prefix="gausscapture-import-") as tmp:
        extracted = Path(tmp) / "extracted"
        extracted.mkdir(parents=True)
        with zipfile.ZipFile(archive_path, "r") as archive:
            _safe_extract(archive, extracted)

        source = _locate_pack_root(extracted)
        if source is None:
            raise CaptureFormatError(
                "No manifest.json found in the archive; this does not look like a CapturePack"
            )

        bag_report = None
        if bagit.is_bag(extracted):
            bag_report = bagit.validate_bag(extracted)

        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)

    result = validate(target)
    if bag_report is not None:
        result["bag"] = bag_report
        # A bag whose checksums do not match is a corrupt transfer, not a
        # merely incomplete capture, so it must not pass validation.
        if not bag_report["valid"]:
            result["valid"] = False
            result["errors"] = [*result["errors"], *bag_report["errors"][:5]]
    if result["valid"]:
        write_checksums(target)
    return result


def import_directory(project_path: Path, source_dir: Path) -> dict[str, Any]:
    """Import an unpacked capture directory, as the native capture app writes one.

    The Android app records straight to a directory rather than zipping on the
    phone: a session is hundreds of megabytes, and compressing already-compressed
    H.264 on a battery to produce a file that is immediately unzipped again is
    pure waste. So the directory is the transfer format, and this is how it
    enters a project.
    """
    source_dir = Path(source_dir)
    if not (source_dir / "manifest.json").exists():
        raise CaptureFormatError(f"No manifest.json in {source_dir}; not a capture directory")

    target = project_path / "capturepack"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    for source in sorted(source_dir.rglob("*")):
        if not source.is_file():
            continue
        destination = target / source.relative_to(source_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Media is hardlinked, metadata is copied. A benchmark sweep imports the
        # same capture once per preset, and a phone take is hundreds of
        # megabytes -- copying the video each time turns a 40-scene dataset into
        # tens of gigabytes of identical bytes. Metadata is copied rather than
        # linked because a later stage rewriting a manifest in a project must
        # never reach back and alter the original capture.
        if source.suffix.lower() in _LINKABLE_MEDIA:
            try:
                os.link(source, destination)
                continue
            except OSError:
                pass  # Cross-device or unsupported; fall through to a copy.
        shutil.copy2(source, destination)

    result = validate(target)
    if result["valid"]:
        write_checksums(target)
    return result


def _stage_dataset(project_path: Path, staging: Path, progress: Progress) -> None:
    """Add images, the COLMAP model, and transforms.json to a bag payload."""
    from gausscapture.pack.transforms import build_transforms
    from gausscapture.pose.model import read_model
    from gausscapture.recon.dataset import _find_sparse_model

    images_src = project_path / "frames" / "images"
    if not images_src.exists() or not any(images_src.glob("*.jpg")):
        raise CaptureFormatError(
            "Cannot include a dataset: no extracted frames. Run frame extraction first."
        )
    model_dir = _find_sparse_model(project_path)
    if model_dir is None:
        raise CaptureFormatError(
            "Cannot include a dataset: no COLMAP model. Run pose estimation first."
        )

    images_dst = staging / "images"
    images_dst.mkdir(parents=True, exist_ok=True)
    names = set()
    for frame in sorted(images_src.glob("*.jpg")):
        shutil.copy2(frame, images_dst / frame.name)
        names.add(frame.name)

    sparse_dst = staging / "sparse" / "0"
    sparse_dst.mkdir(parents=True, exist_ok=True)
    for file in sorted(model_dir.iterdir()):
        if file.is_file():
            shutil.copy2(file, sparse_dst / file.name)

    document = build_transforms(read_model(model_dir), applies_to=names)
    (staging / "transforms.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
    progress.log(f"Staged {len(names)} images and transforms.json with {len(document['frames'])} poses")


def _locate_pack_root(extracted: Path) -> Path | None:
    """Find the directory containing ``manifest.json``.

    Searched in order of likelihood: a bag's payload, the root itself, then a
    single wrapper directory.
    """
    if bagit.is_bag(extracted) and (bagit.payload_dir(extracted) / "manifest.json").exists():
        return bagit.payload_dir(extracted)
    if (extracted / "manifest.json").exists():
        return extracted
    nested = sorted(extracted.glob("*/manifest.json"))
    if nested:
        return nested[0].parent
    deeper = sorted(extracted.glob("*/*/manifest.json"))
    return deeper[0].parent if deeper else None


def _safe_extract(archive: zipfile.ZipFile, target: Path) -> None:
    """Extract, refusing entries that would escape the target directory.

    ``extractall`` does sanitise absolute paths and ``..`` on modern Python,
    but a pack is untrusted input that users hand each other, so the check is
    explicit rather than assumed.
    """
    target = target.resolve()
    for member in archive.infolist():
        destination = (target / member.filename).resolve()
        if not destination.is_relative_to(target):
            raise CaptureFormatError(f"Archive entry escapes the pack directory: {member.filename}")
    archive.extractall(target)
