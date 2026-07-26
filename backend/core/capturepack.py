from __future__ import annotations

import hashlib
import json
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Any

from backend.core.logging_utils import utc_now
from backend.core.video_importer import copy_video_to_project, ffprobe_video_info


REQUIRED_DIRS = ["video", "frames", "camera", "motion", "environment", "quality", "checksums"]


def create_minimal_capturepack(project_path: Path, video_source: Path, session_name: str, target_type: str = "unknown") -> dict[str, Any]:
    cp_dir = project_path / "capturepack"
    for dirname in REQUIRED_DIRS:
        (cp_dir / dirname).mkdir(parents=True, exist_ok=True)
    video_path = copy_video_to_project(video_source, project_path)
    info = ffprobe_video_info(video_path)
    manifest = {
        "capturepack_version": "0.1",
        "session_id": str(uuid.uuid4()),
        "session_name": session_name,
        "capture_type": "static_scene",
        "target_type": target_type,
        "created_at": utc_now(),
        "device": {
            "manufacturer": "unknown",
            "model": "phone_video_import",
            "os": "unknown",
            "app_version": "0.1",
        },
        "video": {
            "main_file": f"video/{video_path.name}",
            "duration_sec": info.get("duration_sec"),
            "width": info.get("width"),
            "height": info.get("height"),
            "fps": info.get("fps"),
            "codec": info.get("codec") or "unknown",
            "bitrate": info.get("bitrate"),
            "has_audio": info.get("has_audio", False),
        },
        "capture_settings": {
            "exposure_locked": None,
            "white_balance_locked": None,
            "focus_locked": None,
            "storage_mode": "unknown",
        },
        "metadata_files": {
            "intrinsics": None,
            "poses": None,
            "imu": None,
            "light": None,
            "audio": None,
        },
    }
    write_manifest(cp_dir, manifest)
    write_checksums(cp_dir)
    return manifest


def write_manifest(capturepack_dir: Path, manifest: dict[str, Any]) -> None:
    capturepack_dir.mkdir(parents=True, exist_ok=True)
    (capturepack_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def read_manifest(capturepack_dir: Path) -> dict[str, Any]:
    path = capturepack_dir / "manifest.json"
    if not path.exists():
        raise ValueError("CapturePack manifest.json is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_capturepack(capturepack_dir: Path) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []
    try:
        manifest = read_manifest(capturepack_dir)
    except ValueError as exc:
        return {"valid": False, "manifest": None, "warnings": [], "errors": [str(exc)]}

    video_file = manifest.get("video", {}).get("main_file")
    if not video_file or not (capturepack_dir / video_file).exists():
        errors.append("Main video file referenced by manifest is missing")

    metadata = manifest.get("metadata_files", {})
    for label in ["intrinsics", "poses", "imu", "light"]:
        value = metadata.get(label)
        if not value or not (capturepack_dir / value).exists():
            warnings.append(f"Metadata file missing: {label}")

    if not manifest.get("capture_settings", {}).get("exposure_locked"):
        warnings.append("Exposure lock metadata is missing or false")
    if not manifest.get("capture_settings", {}).get("white_balance_locked"):
        warnings.append("White balance lock metadata is missing or false")

    return {"valid": not errors, "manifest": manifest, "warnings": warnings, "errors": errors}


def import_capturepack_archive(project_path: Path, archive_path: Path) -> dict[str, Any]:
    target = project_path / "capturepack"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(target)
    nested_manifest = next(target.glob("*/manifest.json"), None)
    if nested_manifest and not (target / "manifest.json").exists():
        nested_root = nested_manifest.parent
        temp = project_path / "_capturepack_nested"
        if temp.exists():
            shutil.rmtree(temp)
        nested_root.rename(temp)
        shutil.rmtree(target)
        temp.rename(target)
    result = validate_capturepack(target)
    if result["valid"]:
        write_checksums(target)
    return result


def export_capturepack_archive(project_path: Path) -> Path:
    cp_dir = project_path / "capturepack"
    manifest = read_manifest(cp_dir)
    out_path = project_path / f"{manifest.get('session_id', 'session')}.capturepack"
    if out_path.exists():
        out_path.unlink()
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in cp_dir.rglob("*"):
            if file.is_file():
                archive.write(file, file.relative_to(cp_dir))
    return out_path


def write_checksums(capturepack_dir: Path) -> None:
    checksums: dict[str, str] = {}
    for file in capturepack_dir.rglob("*"):
        if file.is_file() and "checksums" not in file.relative_to(capturepack_dir).parts:
            checksums[str(file.relative_to(capturepack_dir))] = sha256(file)
    checksums_dir = capturepack_dir / "checksums"
    checksums_dir.mkdir(parents=True, exist_ok=True)
    (checksums_dir / "sha256.json").write_text(json.dumps(checksums, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_main_video(project_path: Path) -> Path:
    manifest = read_manifest(project_path / "capturepack")
    main_file = manifest.get("video", {}).get("main_file")
    if not main_file:
        raise FileNotFoundError("CapturePack has no main video")
    video_path = project_path / "capturepack" / main_file
    if not video_path.exists():
        raise FileNotFoundError(f"Main video not found: {main_file}")
    return video_path

