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

    def __init__(self, settings: Settings | None = None, matcher: str = "sequential"):
        self.settings = settings or get_settings()
        self.matcher = matcher

    def available(self) -> bool:
        return colmap_available(self.settings)

    def solve(self, project_path: Path, progress: Progress | None = None) -> PoseReport:
        return run_colmap(
            project_path, matcher=self.matcher, settings=self.settings, progress=progress
        )


def run_colmap(
    project_path: Path,
    matcher: str = "sequential",
    settings: Settings | None = None,
    progress: Progress | None = None,
    camera_model: str = "OPENCV",
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
    """
    settings = settings or get_settings()
    progress = progress or NullProgress()

    if not colmap_available(settings):
        raise DependencyMissingError(
            "COLMAP was not found. Install it (macOS: `brew install colmap`) or set "
            "`colmap_path` in settings."
        )

    images = project_path / "frames" / "images"
    if not images.exists() or not any(images.glob("*.jpg")):
        raise PipelineStateError("No extracted frames found. Run frame extraction first.")

    colmap_dir = project_path / "colmap"
    sparse_dir = colmap_dir / "sparse"
    database = colmap_dir / "database.db"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    binary = settings.colmap_path
    steps = [
        (
            "Extracting features",
            [
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
            ],
        ),
        (
            "Matching features",
            [binary, f"{matcher}_matcher", "--database_path", str(database)],
        ),
        (
            "Mapping sparse reconstruction",
            [
                binary,
                "mapper",
                "--database_path",
                str(database),
                "--image_path",
                str(images),
                "--output_path",
                str(sparse_dir),
            ],
        ),
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
                return _build_report(sparse_dir, images, matcher)
            raise

    return _build_report(sparse_dir, images, matcher)


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

    # COLMAP can split a capture into several disconnected models; the first is
    # the largest and the one a trainer should use. Counting must go through
    # the shared model reader: the mapper writes *binary* models by default,
    # and a text-only count silently reports a successful reconstruction as
    # zero registered images -- which is exactly what happened on the first
    # real run of the evaluation harness.
    model_dir = model_dirs[0]
    registered, points = _count_model(model_dir)
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
        warnings.append(
            f"COLMAP produced {len(model_dirs)} disconnected models. The capture probably "
            "broke into segments; re-shoot with continuous motion and more overlap."
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


def _count_model(model_dir: Path) -> tuple[int, int]:
    """Registered-image and sparse-point counts, whichever dialect is on disk."""
    from gausscapture.errors import CaptureFormatError
    from gausscapture.pose.model import read_model

    try:
        model = read_model(model_dir)
    except CaptureFormatError:
        return 0, 0
    return len(model.images), model.points
