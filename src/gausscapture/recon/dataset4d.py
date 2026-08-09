"""Turning a two-phase capture into something a 4D trainer can open.

This module is the bridge. On one side is a recording: one video containing a
hand-held sweep and then a stationary hold, plus whatever the capture app
managed to write down about where one ended and the other began. On the other
side is a directory with images, masks, a pose per frame, normalised times, a
sparse point cloud, and a JSON that says what the whole thing is and is not
evidence for. Everything in between happens here, in one pass, on a laptop.

The order is forced by what depends on what:

1. **Split the recording into phases.** Declared boundaries when the app wrote
   them, measured ones when it did not (``ingest.phases``).
2. **Extract the frames, once.** Every hold frame, because those are the
   training views; the sweep thinned to ten a second, the perch and reseat
   likewise, and eight evenly spaced hold keyframes, because those are what
   structure-from-motion gets. Phone video is long-GOP, so this is a single
   linear decode rather than a seek per frame.
3. **Measure the dynamic region** from the hold frames, using the rest window
   for the noise floor (``recon.dynamic_mask``).
4. **Run one reconstruction** over the static frames unmasked and the hold
   keyframes masked, so a single model holds one camera, the background, and
   the head geometry the sweep triangulated.
5. **Derive the fixed pose** from the rest-window cameras and, where possible,
   verify it against the masked keyframes (``pose.fixed``).
6. **Measure the view cone** from the solved sweep cameras about the subject
   (``pose.cone``).
7. **Write the dataset** and validate it.

Two decisions in here are worth stating plainly because they look like bugs
otherwise. Every hold frame gets *the same* pose -- see ``pose.fixed`` for why a
per-frame pose on a stationary camera is noise dressed as data. And the hold
frames are written at native resolution with no downscaling, because the
intrinsics in ``scene4d.json`` are the ones the solver fitted to these exact
pixels, and a resize that forgets to rescale them is silent and fatal.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from gausscapture.errors import CaptureFormatError, PipelineStateError
from gausscapture.ingest.phases import (
    FIXED_CAMERA_CAPTURE_TYPE,
    PHASE_ORDER,
    PhaseSplit,
    iter_video_frames,
    phase_split_for_pack,
)
from gausscapture.pack.manifest import find_main_video, read_manifest
from gausscapture.pose.colmap import run_colmap
from gausscapture.pose.cone import ConeReport, measure_cone, subject_centroid
from gausscapture.pose.fixed import FixedPoseReport, derive_fixed_pose
from gausscapture.pose.model import read_model, read_points
from gausscapture.progress import NullProgress, Progress
from gausscapture.project import MASKS_SUBDIR, record_fourd_stage
from gausscapture.recon.dynamic_mask import (
    DynamicMaskReport,
    build_dynamic_masks,
    write_colmap_masks,
)
from gausscapture.types import PoseReport
from gausscapture.util.log import utc_now

#: Where a project keeps its 4D training set.
DATASET4D_DIRNAME = "dataset4d"

#: Bumped when the layout changes in a way a reader must notice.
DATASET4D_VERSION = "0.1"

#: Hold frames handed to structure-from-motion, masked. Eight is enough to
#: verify the fixed pose and few enough that a bad mask cannot dominate the
#: reconstruction.
HOLD_KEYFRAMES = 8

#: Rate the static phases are thinned to before matching. The phone is at rest
#: during the perch and the reseat, so consecutive frames there are the same
#: image; the sweep is thinned for cost, not redundancy.
STATIC_SAMPLE_FPS = 10.0
ARC_SAMPLE_FPS = 10.0

#: Frame-name prefixes inside the reconstruction workspace. The prefix is how
#: every later stage recovers which phase a solved camera came from, without a
#: sidecar that could go stale.
PHASE_PREFIX = {"perch": "p", "arc": "a", "reseat": "r", "hold": "h"}

#: A hold shorter than this cannot carry motion worth fitting.
MIN_HOLD_FRAMES = 8

Solver = Callable[[Path, Path, Progress], PoseReport]


@dataclass
class Dataset4DResult:
    """Everything the preparation measured, and where it put it."""

    path: Path
    split: PhaseSplit
    mask: DynamicMaskReport
    fixed_pose: FixedPoseReport
    cone: ConeReport
    pose: PoseReport
    frames: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "frames": self.frames,
            "split": self.split.to_dict(),
            "mask": self.mask.to_dict(),
            "fixed_pose": self.fixed_pose.to_dict(),
            "cone": self.cone.to_dict(),
            "pose": self.pose.to_dict(),
            "warnings": list(self.warnings),
        }

    def summary(self) -> str:
        """One line for the CLI, naming the two numbers that decide the take."""
        return (
            f"{self.frames} hold frames  "
            f"cone ±{self.cone.azimuth_half_deg:.0f}° × ±{self.cone.elevation_half_deg:.0f}°  "
            f"fixed pose {self.fixed_pose.source}"
            + (
                f" ({self.fixed_pose.residual_px:.1f} px)"
                if self.fixed_pose.residual_px is not None
                else ""
            )
        )


def prepare_4d_dataset(
    project_path: Path,
    out_dir: Path | None = None,
    solver: Solver | None = None,
    hold_keyframes: int = HOLD_KEYFRAMES,
    arc_fps: float = ARC_SAMPLE_FPS,
    static_fps: float = STATIC_SAMPLE_FPS,
    allow_inference: bool = True,
    progress: Progress | None = None,
) -> Dataset4DResult:
    """Prepare one capture for 4D training, from CapturePack to dataset."""
    progress = progress or NullProgress()
    project_path = Path(project_path).resolve()
    pack_dir = project_path / "capturepack"
    manifest = read_manifest(pack_dir)
    video_path = find_main_video(pack_dir)

    progress.update(2, "Reading capture phases")
    split = phase_split_for_pack(pack_dir, video_path, allow_inference=allow_inference)
    _require_phases(split)

    out_dir = Path(out_dir) if out_dir else project_path / DATASET4D_DIRNAME
    if out_dir.exists():
        shutil.rmtree(out_dir)
    images_dir = out_dir / "images"
    dataset_masks = out_dir / "masks"
    workspace = out_dir / "sfm"
    sfm_images = workspace / "frames" / "images"
    sfm_masks = workspace / "masks"
    mask_home = project_path / MASKS_SUBDIR
    for directory in (images_dir, dataset_masks, sfm_images, sfm_masks, mask_home):
        directory.mkdir(parents=True, exist_ok=True)

    progress.update(6, "Extracting phase frames")
    plan = _plan_frames(split, hold_keyframes, arc_fps, static_fps)
    written = _extract(video_path, plan, images_dir, sfm_images, progress)

    progress.update(30, "Measuring the dynamic region")
    hold_images = [images_dir / plan.hold_files[n] for n in plan.hold_frames]
    rest_images = [sfm_images / plan.sfm_files[n] for n in plan.rest_frames]
    mask_report, union = build_dynamic_masks(
        hold_images, dataset_masks, rest_paths=rest_images, progress=_Sub(progress, 30, 55)
    )
    _mirror(dataset_masks, mask_home)
    keyframe_names = [plan.sfm_files[n] for n in plan.hold_keyframes]
    write_colmap_masks(union, keyframe_names, sfm_masks)

    progress.update(58, "Solving the reconstruction")
    pose_report = (solver or colmap_solver)(workspace, sfm_masks, progress)
    model_dir = _model_dir(pose_report, workspace)
    model = read_model(model_dir)
    points, _colors = read_points(model_dir)

    progress.update(84, "Deriving the fixed camera pose")
    fixed = derive_fixed_pose(
        model,
        rest_names=[plan.sfm_files[n] for n in plan.rest_frames],
        hold_names=keyframe_names,
        points=points,
    )

    progress.update(88, "Measuring the view cone")
    camera = _single_camera(model, mask_report)
    centroid, centroid_source = subject_centroid(points, fixed, camera, union)
    cone = measure_cone(
        model,
        arc_names=[plan.sfm_files[n] for n in plan.arc_frames],
        fixed=fixed,
        centroid=centroid,
        centroid_source=centroid_source,
        imu_implied_half_deg=_imu_implied_half_deg(manifest),
    )

    progress.update(92, "Writing the dataset")
    result = _write_dataset(
        out_dir=out_dir,
        manifest=manifest,
        split=split,
        plan=plan,
        written=written,
        model_dir=model_dir,
        model=model,
        camera=camera,
        mask_report=mask_report,
        fixed=fixed,
        cone=cone,
        pose_report=pose_report,
    )
    validate_dataset4d(out_dir)
    record_fourd_stage(
        project_path,
        "masked",
        dataset4d=str(out_dir),
        cone_azimuth_half_deg=cone.azimuth_half_deg,
        cone_elevation_half_deg=cone.elevation_half_deg,
        fixed_pose_source=fixed.source,
        fixed_pose_residual_px=fixed.residual_px,
        hold_frames=result.frames,
    )
    progress.update(100, result.summary())
    return result


def colmap_solver(workspace: Path, mask_path: Path, progress: Progress) -> PoseReport:
    """The default reconstruction: one COLMAP run, sequential, masked keyframes."""
    return run_colmap(
        workspace, matcher="sequential", mask_path=mask_path, progress=_Sub(progress, 58, 84)
    )


# --------------------------------------------------------------------------
# choosing frames
# --------------------------------------------------------------------------


@dataclass
class FramePlan:
    """Which source frames go where, decided before the video is opened."""

    hold_frames: list[int] = field(default_factory=list)
    hold_keyframes: list[int] = field(default_factory=list)
    arc_frames: list[int] = field(default_factory=list)
    rest_frames: list[int] = field(default_factory=list)
    #: Source frame number to file name inside ``images/``.
    hold_files: dict[int, str] = field(default_factory=dict)
    #: Source frame number to file name inside the reconstruction workspace.
    sfm_files: dict[int, str] = field(default_factory=dict)
    phase_of: dict[int, str] = field(default_factory=dict)

    @property
    def wanted(self) -> list[int]:
        return sorted(set(self.hold_files) | set(self.sfm_files))


def _plan_frames(
    split: PhaseSplit, hold_keyframes: int, arc_fps: float, static_fps: float
) -> FramePlan:
    plan = FramePlan()
    fps = split.fps or 30.0
    hold = split.require("hold")

    plan.hold_frames = hold.frame_numbers()
    for position, frame_no in enumerate(plan.hold_frames):
        plan.hold_files[frame_no] = f"frame_{position:06d}.jpg"
        plan.phase_of[frame_no] = "hold"

    for name in PHASE_ORDER:
        phase = split.phases.get(name)
        if phase is None:
            continue
        if name == "hold":
            chosen = phase.evenly_spaced(hold_keyframes)
            plan.hold_keyframes = chosen
        elif name == "arc":
            chosen = phase.sample(fps, arc_fps)
            plan.arc_frames = chosen
        else:
            chosen = phase.sample(fps, static_fps)
        for frame_no in chosen:
            plan.sfm_files[frame_no] = f"{PHASE_PREFIX[name]}_{frame_no:06d}.jpg"
            plan.phase_of.setdefault(frame_no, name)

    # The rest window enters the reconstruction whole and unmasked: it is the
    # only evidence for the fixed pose, and thinning it to save a few seconds of
    # matching would throw away the measurement the whole tier ladder rests on.
    for frame_no in split.rest_frames():
        name = split.phase_of(frame_no) or "reseat"
        plan.rest_frames.append(frame_no)
        plan.sfm_files.setdefault(frame_no, f"{PHASE_PREFIX[name]}_{frame_no:06d}.jpg")
        plan.phase_of.setdefault(frame_no, name)
    return plan


def _require_phases(split: PhaseSplit):
    hold = split.phases.get("hold")
    arc = split.phases.get("arc")
    if hold is None:
        raise CaptureFormatError(
            "This capture has no hold phase -- no stretch of footage with the camera at rest "
            "and the subject moving -- so there is nothing 4D about it. Record with preset F: "
            "perch, sweep, set the phone down, then move."
        )
    if arc is None:
        raise CaptureFormatError(
            "This capture has no arc phase. A stationary camera produces zero parallax, so a "
            "clip shot entirely from one position is undefined for structure-from-motion and "
            "there is no way to recover either the scene or the subject's geometry from it. "
            "The hand-held sweep before the phone is set down is what makes the rest possible."
        )
    if hold.frame_count < MIN_HOLD_FRAMES:
        raise CaptureFormatError(
            f"The hold phase is {hold.frame_count} frames long, under the {MIN_HOLD_FRAMES} "
            "needed to fit any motion at all. Let the take run its full length and auto-stop."
        )
    return hold, arc


def _extract(
    video_path: Path,
    plan: FramePlan,
    images_dir: Path,
    sfm_images: Path,
    progress: Progress,
) -> dict[int, tuple[int, int]]:
    """One decode pass, writing each wanted frame to every place it belongs."""
    wanted = plan.wanted
    sizes: dict[int, tuple[int, int]] = {}
    for position, (frame_no, frame) in enumerate(iter_video_frames(video_path, wanted)):
        height, width = frame.shape[:2]
        sizes[frame_no] = (width, height)
        encode = [int(cv2.IMWRITE_JPEG_QUALITY), 94]
        if frame_no in plan.hold_files:
            cv2.imwrite(str(images_dir / plan.hold_files[frame_no]), frame, encode)
        if frame_no in plan.sfm_files:
            cv2.imwrite(str(sfm_images / plan.sfm_files[frame_no]), frame, encode)
        seen = position + 1
        if seen % 20 == 0:
            progress.update(6 + int(22 * seen / max(1, len(wanted))), f"Extracted {seen} frames")

    missing = [n for n in wanted if n not in sizes]
    if missing:
        raise CaptureFormatError(
            f"The phase split asks for frame {missing[0]} but the video decoded only "
            f"{len(sizes)} of the {len(wanted)} frames requested. The manifest's phase bounds "
            "do not match the video it ships with; re-pull the capture."
        )
    return sizes


# --------------------------------------------------------------------------
# writing and validating the layout
# --------------------------------------------------------------------------


def _write_dataset(
    out_dir: Path,
    manifest: dict[str, Any],
    split: PhaseSplit,
    plan: FramePlan,
    written: dict[int, tuple[int, int]],
    model_dir: Path,
    model: Any,
    camera: Any,
    mask_report: DynamicMaskReport,
    fixed: FixedPoseReport,
    cone: ConeReport,
    pose_report: PoseReport,
) -> Dataset4DResult:
    sparse_dst = out_dir / "sparse" / "0"
    sparse_dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for stem in ("cameras", "images", "points3D"):
        for suffix in (".bin", ".txt"):
            candidate = Path(model_dir) / f"{stem}{suffix}"
            if candidate.exists():
                shutil.copy2(candidate, sparse_dst / candidate.name)
                copied += 1
    if copied == 0:
        raise CaptureFormatError(
            f"The sparse model at {model_dir} holds no cameras/images/points3D files, so there "
            "is nothing to initialise a splat from."
        )

    frames = _frame_records(plan, written)
    (out_dir / "frames.json").write_text(json.dumps(frames, indent=2), encoding="utf-8")

    warnings = [
        *split.warnings,
        *mask_report.warnings,
        *fixed.warnings,
        *cone.warnings,
        *pose_report.warnings,
    ]
    width, height = written[plan.hold_frames[0]]
    scene = {
        "dataset4d_version": DATASET4D_VERSION,
        "created_at": utc_now(),
        "session_id": manifest.get("session_id"),
        "session_name": manifest.get("session_name"),
        "capture_type": manifest.get("capture_type", FIXED_CAMERA_CAPTURE_TYPE),
        "frames": len(frames),
        "width": width,
        "height": height,
        "fps": split.fps,
        "duration_sec": (len(frames) / split.fps) if split.fps else None,
        # Times are the hold phase mapped onto [0, 1] rather than seconds,
        # because the deformation field is parameterised on tau and a trainer
        # that had to divide by a duration would need the duration to be right
        # in two files at once.
        "time_base": "normalised_hold_phase",
        "split": split.to_dict(),
        "camera": _camera_record(camera, width, height),
        "fixed_pose": fixed.to_dict(),
        "cone": cone.to_dict(),
        "mask": mask_report.to_dict(),
        "sparse": {"model_dir": "sparse/0", "points": model.points, "images": len(model.images)},
        "pose": pose_report.to_dict(),
        "warnings": warnings,
        # Said once, here, so that nothing downstream has to reconstruct what
        # these numbers are evidence for from their names alone.
        "claims": {
            "pose_verified": fixed.verified,
            "pose_source": fixed.source,
            "cone_measured_from": "solved_arc_cameras",
            "cone_is_a_quality_claim": False,
            "novel_view_holdout": False,
            "evaluation": (
                "Every training view is the same view. Any PSNR computed from this dataset "
                "measures motion interpolation between held-out frames, never novel-view "
                "synthesis, and is not comparable to masked mPSNR from DyCheck or N3DV."
            ),
        },
    }
    (out_dir / "scene4d.json").write_text(json.dumps(scene, indent=2), encoding="utf-8")

    return Dataset4DResult(
        path=out_dir,
        split=split,
        mask=mask_report,
        fixed_pose=fixed,
        cone=cone,
        pose=pose_report,
        frames=len(frames),
        warnings=warnings,
    )


def _frame_records(plan: FramePlan, written: dict[int, tuple[int, int]]) -> list[dict[str, Any]]:
    numbers = plan.hold_frames
    first, last = numbers[0], numbers[-1]
    span = float(last - first) or 1.0
    records: list[dict[str, Any]] = []
    for position, frame_no in enumerate(numbers):
        name = plan.hold_files[frame_no]
        records.append(
            {
                "index": position,
                "source_frame": frame_no,
                "file": f"images/{name}",
                "mask": f"masks/{Path(name).stem}.png",
                # Normalised on the source frame numbers rather than on the
                # position in the list: if extraction ever drops a frame, the
                # remaining ones must keep their true place in time instead of
                # closing the gap and speeding the subject up.
                "time": (frame_no - first) / span,
                "phase": "hold",
                "width": written[frame_no][0],
                "height": written[frame_no][1],
            }
        )
    return records


def _camera_record(camera: Any, width: int, height: int) -> dict[str, Any]:
    return {
        "model": camera.model,
        "width": camera.width or width,
        "height": camera.height or height,
        "fx": camera.fx,
        "fy": camera.fy,
        "cx": camera.cx,
        "cy": camera.cy,
        "distortion": camera.distortion(),
        "shared": True,
    }


def validate_dataset4d(path: Path) -> dict[str, Any]:
    """Refuse a dataset that a trainer would fail on later and less clearly.

    Raises :class:`CaptureFormatError` naming the file at fault and what to do;
    returns the non-fatal observations so a caller can print them.
    """
    path = Path(path)
    scene_path = path / "scene4d.json"
    frames_path = path / "frames.json"
    if not scene_path.exists():
        raise CaptureFormatError(
            f"No scene4d.json in {path}. This is not a 4D dataset directory; run the 4D "
            "preparation step on the project first."
        )
    try:
        scene = json.loads(scene_path.read_text(encoding="utf-8"))
        frames = json.loads(frames_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptureFormatError(f"4D dataset metadata is unreadable in {path}: {exc}") from exc

    if not isinstance(frames, list) or not frames:
        raise CaptureFormatError(
            f"{frames_path} lists no frames, so the dataset has no training views."
        )

    for record in frames:
        for key in ("file", "mask"):
            target = path / record[key]
            if not target.exists():
                raise CaptureFormatError(
                    f"Frame {record['index']} references {record[key]}, which does not exist. "
                    "The dataset directory is incomplete; re-run the preparation step rather "
                    "than training on a partial set."
                )

    times = [float(record["time"]) for record in frames]
    if times[0] != 0.0 or abs(times[-1] - 1.0) > 1e-9:
        raise CaptureFormatError(
            f"Frame times run from {times[0]} to {times[-1]}; they must span exactly 0 to 1. "
            "The deformation field is indexed on that range and would extrapolate off the end."
        )
    if any(later <= earlier for earlier, later in zip(times, times[1:], strict=False)):
        raise CaptureFormatError(
            "Frame times are not strictly increasing. Frames must be listed in the order they "
            "were recorded, because the trainer reads their order as the direction of time."
        )

    sparse = path / "sparse" / "0"
    if not any(sparse.glob("cameras.*")):
        raise CaptureFormatError(
            f"No COLMAP camera file in {sparse}. Without intrinsics the images cannot be "
            "projected and no splat can be initialised."
        )

    qvec = np.asarray(scene.get("fixed_pose", {}).get("qvec", []), dtype=float)
    if qvec.size != 4 or abs(float(np.linalg.norm(qvec)) - 1.0) > 1e-6:
        raise CaptureFormatError(
            "The fixed camera pose in scene4d.json is not a unit quaternion, so it does not "
            "describe a rotation. The dataset is corrupt; re-run the preparation step."
        )

    cone = scene.get("cone", {})
    if float(cone.get("azimuth_half_deg", 0.0)) <= 0.0:
        raise CaptureFormatError(
            "The measured view cone has zero width: no arc camera sits off the fixed pose, so "
            "the reconstruction contains no parallax. There is nothing here that a bullet-time "
            "viewer could move through. Reshoot with a wider, slower sweep."
        )

    notes = list(scene.get("warnings", []))
    if not scene.get("claims", {}).get("pose_verified", False):
        notes.append(
            "The fixed camera pose was never confirmed from inside the dynamic phase "
            f"(source: {scene.get('fixed_pose', {}).get('source')}). Do not present this "
            "scene's geometry as verified."
        )
    return {"valid": True, "frames": len(frames), "warnings": notes}


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _model_dir(report: PoseReport, workspace: Path) -> Path:
    if not report.model_dir:
        raise PipelineStateError(
            "Structure-from-motion produced no model from this capture. "
            + (report.warnings[0] if report.warnings else "")
            + " A fixed-camera 4D take depends entirely on the hand-held sweep registering; "
            "there is no fallback, because a zero-parallax clip is undefined for SfM."
        )
    model_dir = Path(report.model_dir)
    if not model_dir.is_absolute():
        model_dir = workspace / model_dir
    if not model_dir.exists():
        raise PipelineStateError(f"The reported sparse model directory does not exist: {model_dir}")
    return model_dir


def _single_camera(model: Any, mask_report: DynamicMaskReport):
    if not model.cameras:
        raise CaptureFormatError(
            "The sparse model declares no camera, so the images have no intrinsics."
        )
    camera = next(iter(model.cameras.values()))
    if mask_report.width and camera.width and camera.width != mask_report.width:
        mask_report.warnings.append(
            f"The solved camera is {camera.width}x{camera.height} but the hold frames are "
            f"{mask_report.width}x{mask_report.height}. Intrinsics and images disagree; the "
            "reconstruction and the training views were not extracted at the same size."
        )
    return camera


def _imu_implied_half_deg(manifest: dict[str, Any]) -> float | None:
    """The phone's own estimate of the sweep, used only to narrow the cone."""
    arc = manifest.get("arc")
    if not isinstance(arc, dict):
        return None
    value = arc.get("implied_cone_deg")
    try:
        implied = float(value)
    except (TypeError, ValueError):
        return None
    return implied if implied > 0 else None


def _mirror(source: Path, destination: Path) -> None:
    """Hardlink the masks into the project's canonical mask directory."""
    destination.mkdir(parents=True, exist_ok=True)
    for item in sorted(source.glob("*.png")):
        target = destination / item.name
        if target.exists():
            target.unlink()
        try:
            os.link(item, target)
        except OSError:
            shutil.copy2(item, target)


class _Sub:
    """Rescales a child step's 0-100 progress into a slice of the parent's."""

    def __init__(self, parent: Progress, start: int, end: int):
        self.parent = parent
        self.start = start
        self.span = max(0, end - start)

    def update(self, percent: int, message: str | None = None) -> None:
        self.parent.update(self.start + int(self.span * max(0, min(100, percent)) / 100), message)

    def log(self, message: str) -> None:
        self.parent.log(message)


def hold_frame_files(dataset_dir: Path) -> list[Path]:
    """Training views in recorded order, for anything that reads a prepared set."""
    dataset_dir = Path(dataset_dir)
    records = json.loads((dataset_dir / "frames.json").read_text(encoding="utf-8"))
    return [dataset_dir / record["file"] for record in records]


def frame_times(dataset_dir: Path) -> np.ndarray:
    """Normalised timestamps in recorded order."""
    dataset_dir = Path(dataset_dir)
    records = json.loads((dataset_dir / "frames.json").read_text(encoding="utf-8"))
    return np.asarray([record["time"] for record in records], dtype=np.float32)


def read_scene4d(dataset_dir: Path) -> dict[str, Any]:
    path = Path(dataset_dir) / "scene4d.json"
    if not path.exists():
        raise CaptureFormatError(f"No scene4d.json in {dataset_dir}")
    return json.loads(path.read_text(encoding="utf-8"))
