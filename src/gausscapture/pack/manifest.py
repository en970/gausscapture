from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from gausscapture.errors import CaptureFormatError
from gausscapture.types import VideoInfo
from gausscapture.util.log import utc_now

CAPTUREPACK_VERSION = "0.1"

#: Directories a well-formed pack contains. Empty ones are allowed; the pack is
#: a superset envelope so a richer capture app can fill in more without
#: changing the import path.
REQUIRED_DIRS = ("video", "frames", "camera", "motion", "environment", "quality", "checksums")

#: Metadata files whose absence downgrades reconstruction quality but does not
#: make the pack invalid.
OPTIONAL_METADATA = ("intrinsics", "poses", "imu", "light")


def create_minimal_manifest(
    video_relpath: str,
    info: VideoInfo,
    session_name: str,
    target_type: str = "unknown",
    device: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the manifest for a pack created from a bare video file.

    Everything a phone could have told us -- intrinsics, ARKit poses, IMU,
    exposure lock state -- is null here, which is exactly why such a pack needs
    COLMAP downstream.
    """
    return {
        "capturepack_version": CAPTUREPACK_VERSION,
        "session_id": str(uuid.uuid4()),
        "session_name": session_name,
        "capture_type": "static_scene",
        "target_type": target_type,
        "created_at": utc_now(),
        "device": device
        or {
            "manufacturer": "unknown",
            "model": "phone_video_import",
            "os": "unknown",
            "app_version": CAPTUREPACK_VERSION,
        },
        "video": {
            "main_file": video_relpath,
            "duration_sec": info.duration_sec,
            "width": info.width,
            "height": info.height,
            "fps": info.fps,
            "codec": info.codec,
            "bitrate": info.bitrate,
            "has_audio": info.has_audio,
        },
        "capture_settings": {
            "exposure_locked": None,
            "white_balance_locked": None,
            "focus_locked": None,
            "storage_mode": "unknown",
        },
        "metadata_files": {key: None for key in (*OPTIONAL_METADATA, "audio")},
    }


def write_manifest(pack_dir: Path, manifest: dict[str, Any]) -> None:
    pack_dir.mkdir(parents=True, exist_ok=True)
    (pack_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def read_manifest(pack_dir: Path) -> dict[str, Any]:
    path = pack_dir / "manifest.json"
    if not path.exists():
        raise CaptureFormatError(f"CapturePack manifest.json is missing in {pack_dir}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CaptureFormatError(f"CapturePack manifest.json is not valid JSON: {exc}") from exc


def validate(pack_dir: Path) -> dict[str, Any]:
    """Check a pack.

    Returns a result dict rather than raising, because the UI shows warnings and
    errors side by side and a pack with every optional field missing is still
    perfectly usable.
    """
    warnings: list[str] = []
    errors: list[str] = []
    try:
        manifest = read_manifest(pack_dir)
    except CaptureFormatError as exc:
        return {"valid": False, "manifest": None, "warnings": [], "errors": [str(exc)]}

    video_file = manifest.get("video", {}).get("main_file")
    if not video_file or not (pack_dir / video_file).exists():
        errors.append("Main video file referenced by the manifest is missing")

    metadata = manifest.get("metadata_files") or {}
    for label in OPTIONAL_METADATA:
        value = metadata.get(label)
        if not value or not (pack_dir / value).exists():
            warnings.append(f"Metadata file missing: {label}")

    settings = manifest.get("capture_settings") or {}
    if settings.get("exposure_locked") is not True:
        warnings.append("Exposure lock was not recorded as enabled")
    if settings.get("white_balance_locked") is not True:
        warnings.append("White balance lock was not recorded as enabled")

    return {"valid": not errors, "manifest": manifest, "warnings": warnings, "errors": errors}


def find_main_video(pack_dir: Path) -> Path:
    """Resolve the manifest's main video to an existing path."""
    manifest = read_manifest(pack_dir)
    main_file = manifest.get("video", {}).get("main_file")
    if not main_file:
        raise CaptureFormatError("CapturePack manifest declares no main video")
    video_path = pack_dir / main_file
    if not video_path.exists():
        raise CaptureFormatError(f"Main video referenced by the manifest not found: {main_file}")
    return video_path
