from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.deps import get_project, save_upload, store, to_http
from gausscapture.errors import GaussCaptureError
from gausscapture.export.preview import MODEL_SUFFIXES, import_training_result
from gausscapture.ingest.video import VIDEO_EXTENSIONS, copy_video_into_pack, probe_video
from gausscapture.pack import archive, manifest
from gausscapture.project import STATUS_IMPORTED, STATUS_PREVIEW

router = APIRouter(prefix="/api/projects/{project_id}/import", tags=["import"])


@router.post("/video")
def import_video(project_id: str, file: UploadFile = File(...)):
    project = get_project(project_id)
    temp = save_upload(file, VIDEO_EXTENSIONS)
    try:
        for name in manifest.REQUIRED_DIRS:
            (project.capturepack_dir / name).mkdir(parents=True, exist_ok=True)
        video_path = copy_video_into_pack(temp, project.path)
        info = probe_video(video_path)
        pack_manifest = manifest.create_minimal_manifest(
            video_relpath=f"video/{video_path.name}",
            info=info,
            session_name=project.name,
            target_type=project.target_type,
        )
        manifest.write_manifest(project.capturepack_dir, pack_manifest)
        archive.write_checksums(project.capturepack_dir)
        updated = store.update(
            project_id, status=STATUS_IMPORTED, last_step="Video imported as minimal CapturePack"
        )
        return {
            "project": updated.to_dict(),
            "manifest": pack_manifest,
            "warnings": [
                "Minimal CapturePack created from a video. No camera intrinsics or poses are "
                "available, so COLMAP will be required."
            ],
            "errors": [],
        }
    except GaussCaptureError as exc:
        raise to_http(exc) from exc
    finally:
        temp.unlink(missing_ok=True)


@router.post("/capturepack")
def import_capturepack(project_id: str, file: UploadFile = File(...)):
    project = get_project(project_id)
    temp = save_upload(file, {".capturepack", ".zip"})
    try:
        result = archive.import_archive(project.path, temp)
        if not result["valid"]:
            raise HTTPException(
                status_code=400,
                detail={"errors": result["errors"], "warnings": result["warnings"]},
            )
        updated = store.update(project_id, status=STATUS_IMPORTED, last_step="CapturePack imported")
        result.pop("manifest", None)
        return {"project": updated.to_dict(), **result}
    except GaussCaptureError as exc:
        raise to_http(exc) from exc
    finally:
        temp.unlink(missing_ok=True)


@router.post("/training-result")
def import_training(project_id: str, file: UploadFile = File(...)):
    project = get_project(project_id)
    temp = save_upload(file, {".zip", *MODEL_SUFFIXES})
    try:
        result = import_training_result(project.path, temp)
        updated = store.update(
            project_id, status=STATUS_PREVIEW, last_step="Training result imported"
        )
        return {"project": updated.to_dict(), **result}
    except GaussCaptureError as exc:
        raise to_http(exc) from exc
    finally:
        temp.unlink(missing_ok=True)


@router.get("/validate")
def validate_current_capturepack(project_id: str):
    return manifest.validate(get_project(project_id).capturepack_dir)
