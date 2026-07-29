from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from gausscapture.config import Settings, get_settings
from gausscapture.errors import DependencyMissingError, PipelineStateError
from gausscapture.progress import NullProgress, Progress
from gausscapture.types import PoseReport

#: Fraction of images that must register for the result to be trustworthy.
GOOD_REGISTRATION_RATIO = 0.75
POOR_REGISTRATION_RATIO = 0.35


def colmap_available(settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    configured = Path(settings.colmap_path)
    return configured.exists() or shutil.which(settings.colmap_path) is not None


class ColmapBackend:
    """Structure-from-motion via the COLMAP CLI.

    COLMAP is invoked as a subprocess and never linked. Its own BSD-3 licence
    is permissive, but its documentation notes that the licence is independent
    of its third-party dependencies, and prebuilt binaries can embed GPL
    components -- so we require the user's system installation rather than
    shipping one. See ``docs/DEPENDENCIES.md``.
    """

    name = "colmap"

    def __init__(
        self,
        settings: Settings | None = None,
        matcher: str = "sequential",
        seed_intrinsics: bool = False,
        refine_intrinsics: bool = True,
    ):
        self.settings = settings or get_settings()
        self.matcher = matcher
        self.seed_intrinsics = seed_intrinsics
        self.refine_intrinsics = refine_intrinsics

    def available(self) -> bool:
        return colmap_available(self.settings)

    def solve(self, project_path: Path, progress: Progress | None = None) -> PoseReport:
        return run_colmap(
            project_path,
            matcher=self.matcher,
            settings=self.settings,
            progress=progress,
            seed_intrinsics=self.seed_intrinsics,
            refine_intrinsics=self.refine_intrinsics,
        )


def run_colmap(
    project_path: Path,
    matcher: str = "sequential",
    settings: Settings | None = None,
    progress: Progress | None = None,
    camera_model: str = "OPENCV",
    seed_intrinsics: bool = False,
    refine_intrinsics: bool = True,
) -> PoseReport:
    """Run feature extraction, matching, and mapping.

    Two flags matter and were previously missing:

    ``--ImageReader.single_camera 1``
        Every frame comes from one phone camera with fixed intrinsics. Without
        this, COLMAP solves an independent intrinsic set per image, which is
        slower and materially less stable on the low-parallax sequences that
        handheld phone capture produces.

    ``--ImageReader.camera_model OPENCV``
        Models radial and tangential distortion, which phone lenses have. The
        default SIMPLE_RADIAL underfits them.

    ``sequential`` matching is the default because frames come from video, where
    temporal neighbours are spatial neighbours; exhaustive matching costs
    quadratic time for no benefit until loop closure matters.

    ``seed_intrinsics`` hands COLMAP the calibration the phone reports instead
    of making it solve for focal length and distortion from scratch.

    **It is off by default, because measurement said so.** Across three real
    captures it changed no registration ratio at all (93/36/4 percent either
    way), moved sparse point counts by under one percent, and cost 17 to 61
    percent more wall-clock time -- the opposite of the roadmap's hoped-for
    speed-up. On the one capture that reconstructed well, seeded and unseeded
    runs converged to the same focal length from different starting points,
    which says the images already determine it and the prior adds nothing.

    Note this tests *intrinsic* seeding only. Seeding *poses* from ARCore is a
    far stronger prior and remains untested, because the capture app does not
    record them yet.
    """
    settings = settings or get_settings()
    progress = progress or NullProgress()

    if not colmap_available(settings):
        raise DependencyMissingError(
            "COLMAP was not found. Install it (macOS: `brew install colmap`) or set "
            "`colmap_path` in settings."
        )

    # Absolute from here down. COLMAP runs as a subprocess with its own working
    # directory, so every path it is given must stand on its own.
    project_path = Path(project_path).resolve()
    images = project_path / "frames" / "images"
    if not images.exists() or not any(images.glob("*.jpg")):
        raise PipelineStateError("No extracted frames found. Run frame extraction first.")

    colmap_dir = project_path / "colmap"
    sparse_dir = colmap_dir / "sparse"
    database = colmap_dir / "database.db"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    intrinsics = None
    if seed_intrinsics:
        from gausscapture.pose import intrinsics as intrinsics_module

        intrinsics = intrinsics_module.for_project(project_path)

    binary = settings.colmap_path
    extractor = [
        binary,
        "feature_extractor",
        "--database_path",
        str(database),
        "--image_path",
        str(images),
        # All frames share one physical camera.
        "--ImageReader.single_camera",
        "1",
        "--ImageReader.camera_model",
        camera_model,
    ]
    if intrinsics is not None:
        extractor += ["--ImageReader.camera_params", intrinsics.colmap_camera_params()]
        progress.log(
            f"Seeding intrinsics from the device: fx={intrinsics.fx:.1f} "
            f"fy={intrinsics.fy:.1f} cx={intrinsics.cx:.1f} cy={intrinsics.cy:.1f}"
        )

    mapper = [
        binary,
        "mapper",
        "--database_path",
        str(database),
        "--image_path",
        str(images),
        "--output_path",
        str(sparse_dir),
    ]
    if intrinsics is not None and not refine_intrinsics:
        mapper += [
            "--Mapper.ba_refine_focal_length", "0",
            "--Mapper.ba_refine_principal_point", "0",
            "--Mapper.ba_refine_extra_params", "0",
        ]

    steps = [
        ("Extracting features", extractor),
        ("Matching features", [binary, f"{matcher}_matcher", "--database_path", str(database)]),
        ("Mapping sparse reconstruction", mapper),
    ]

    for i, (label, cmd) in enumerate(steps):
        progress.update(5 + i * 30, label)
        try:
            _run(cmd, cwd=project_path, progress=progress)
        except RuntimeError:
            # The mapper exits non-zero when it cannot initialise a
            # reconstruction -- one image, no parallax, no matches. For this
            # project that is a *measured outcome*, not an environment error:
            # "SfM could not produce a model from this capture" is precisely
            # the endpoint the study predicts. Feature extraction or matching
            # failing, by contrast, means something is wrong with the install
            # or the images, so those still raise.
            if cmd[1] == "mapper":
                progress.log("COLMAP mapper could not initialise a reconstruction")
                return _build_report(sparse_dir, images, matcher)  # noqa: TRY300
            raise

    report = _build_report(sparse_dir, images, matcher)
    if intrinsics is not None:
        report.seeded_intrinsics = intrinsics.to_dict()
    return report


def _build_report(sparse_dir: Path, images: Path, matcher: str) -> PoseReport:
    model_dirs = sorted(p for p in sparse_dir.iterdir() if p.is_dir())
    images_total = len(list(images.glob("*.jpg")))

    if not model_dirs:
        return PoseReport(
            method=f"colmap/{matcher}",
            images_total=images_total,
            status="bad",
            warnings=[
                "COLMAP produced no reconstruction. The capture likely lacks parallax, "
                "is too blurred, or shows too little texture."
            ],
        )

    # COLMAP splits a capture it cannot connect into several models, numbered
    # in the order they were reconstructed -- NOT by size. Taking sparse/0
    # therefore reports whichever fragment happened to be built first, which on
    # a real capture meant announcing a 3-image fragment as the result while a
    # usable 63-image reconstruction sat in sparse/1. Pick the largest.
    #
    # Counting goes through the shared model reader because the mapper writes
    # *binary* models by default, and a text-only count silently reports a
    # successful reconstruction as zero registered images.
    measured = [(*_count_model(d), d) for d in model_dirs]
    registered, points, model_dir = max(measured, key=lambda item: item[0])
    ratio = registered / images_total if images_total else 0.0

    if ratio >= GOOD_REGISTRATION_RATIO:
        status = "good"
    elif ratio >= POOR_REGISTRATION_RATIO:
        status = "warning"
    else:
        status = "bad"

    warnings: list[str] = []
    if status != "good":
        warnings.append(
            f"COLMAP registered {registered}/{images_total} images ({ratio:.0%}); "
            "training quality will suffer."
        )
    if len(model_dirs) > 1:
        other = sorted((count for count, _, d in measured if d != model_dir), reverse=True)
        warnings.append(
            f"COLMAP produced {len(model_dirs)} disconnected models and the largest "
            f"({registered} images) was kept; the rest hold {other} images. The capture broke "
            "into segments -- re-shoot with continuous motion and more overlap."
        )

    report = PoseReport(
        method=f"colmap/{matcher}",
        images_total=images_total,
        images_registered=registered,
        registered_ratio=ratio,
        sparse_points=points,
        status=status,
        model_dir=str(model_dir),
        warnings=warnings,
    )
    report.solved_fx, report.solved_fy = _solved_focal(model_dir)
    (sparse_dir.parent / "pose_report.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )
    return report


def _run(cmd: list[str], cwd: Path, progress: Progress) -> None:
    progress.log("$ " + " ".join(cmd))
    process = subprocess.Popen(
        cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    assert process.stdout is not None
    for line in process.stdout:
        progress.log(line.rstrip())
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"COLMAP step failed with exit code {code}: {' '.join(cmd[:2])}")


def _solved_focal(model_dir: Path) -> tuple[float | None, float | None]:
    """Focal length the solver settled on, for comparison against the device's."""
    from gausscapture.errors import CaptureFormatError
    from gausscapture.pose.model import read_model

    try:
        model = read_model(model_dir)
    except CaptureFormatError:
        return None, None
    if not model.cameras:
        return None, None
    camera = next(iter(model.cameras.values()))
    return camera.fx, camera.fy


def _count_model(model_dir: Path) -> tuple[int, int]:
    """Registered-image and sparse-point counts, whichever dialect is on disk."""
    from gausscapture.errors import CaptureFormatError
    from gausscapture.pose.model import read_model

    try:
        model = read_model(model_dir)
    except CaptureFormatError:
        return 0, 0
    return len(model.images), model.points
