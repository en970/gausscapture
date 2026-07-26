"""Command-line interface.

Every pipeline stage is reachable from here without starting a server. That is
what makes unattended batch runs possible -- turning 180 capture packs into 180
reconstructions is the whole point of the evaluation harness -- and it is what
lets the package qualify as a library with a thin GUI on top rather than a web
application.

Machine-readable output goes to stdout; progress goes to stderr, so

    gausscapture telemetry <project> --json | jq .vol_p10

works while still showing progress on a terminal.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from gausscapture import __version__
from gausscapture.config import get_settings
from gausscapture.errors import GaussCaptureError
from gausscapture.progress import ConsoleProgress


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "handler", None):
        parser.print_help()
        return 0
    progress = ConsoleProgress(quiet=getattr(args, "quiet", False) or getattr(args, "json", False))
    try:
        return args.handler(args, progress)
    except GaussCaptureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gausscapture",
        description="Local-first phone capture for 3D Gaussian splatting.",
    )
    parser.add_argument("--version", action="version", version=f"gausscapture {__version__}")
    sub = parser.add_subparsers(dest="command")

    def add(name: str, help_text: str, handler) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.add_argument("--quiet", action="store_true", help="Suppress progress output")
        p.set_defaults(handler=handler)
        return p

    # doctor -----------------------------------------------------------------
    p = add("doctor", "Check that external dependencies are available.", _cmd_doctor)
    p.add_argument("--json", action="store_true", help="Emit JSON")

    # project ----------------------------------------------------------------
    p = add("project", "Create, list, or delete projects.", _cmd_project)
    p.add_argument("action", choices=["list", "create", "show", "delete"])
    p.add_argument("value", nargs="?", help="Project name (create) or id (show/delete)")
    p.add_argument("--target-type", default="unknown", help="object, room, outdoor, ...")
    p.add_argument("--json", action="store_true", help="Emit JSON")

    # import -----------------------------------------------------------------
    p = add("import", "Import a video or .capturepack into a project.", _cmd_import)
    p.add_argument("source", type=Path, help="Path to a video file or .capturepack archive")
    p.add_argument("--project", help="Existing project id or path; a new one is created if omitted")
    p.add_argument("--name", help="Name for a newly created project")
    p.add_argument("--target-type", default="unknown")
    p.add_argument("--json", action="store_true")

    # pack -------------------------------------------------------------------
    p = add("pack", "Validate or export a CapturePack.", _cmd_pack)
    p.add_argument("action", choices=["validate", "export"])
    p.add_argument("target", help="Project id, project path, or a pack directory")
    p.add_argument("--verify", action="store_true", help="Re-hash files against the checksums")
    p.add_argument("--json", action="store_true")

    # telemetry --------------------------------------------------------------
    p = add("telemetry", "Analyse capture quality and write a report.", _cmd_telemetry)
    p.add_argument("project", help="Project id or path")
    p.add_argument("--samples", type=int, default=80, help="How many frames to sample")
    p.add_argument("--json", action="store_true", help="Emit the full report as JSON")
    p.add_argument("--signals", action="store_true", help="Include per-frame signals in JSON")

    # frames -----------------------------------------------------------------
    p = add("frames", "Extract training images from the capture video.", _cmd_frames)
    p.add_argument("project")
    p.add_argument("--preset", default="balanced", choices=["fast", "balanced", "dense"])
    p.add_argument("--fps", type=float, help="Override the preset's target frame rate")
    p.add_argument("--max-frames", type=int, help="Override the preset's frame budget")
    p.add_argument("--no-blur-filter", action="store_true")
    p.add_argument("--no-duplicate-filter", action="store_true")
    p.add_argument("--json", action="store_true")

    # pose -------------------------------------------------------------------
    p = add("pose", "Estimate camera poses with COLMAP.", _cmd_pose)
    p.add_argument("project")
    p.add_argument("--matcher", default="sequential", choices=["sequential", "exhaustive", "vocab_tree"])
    p.add_argument("--json", action="store_true")

    # dataset ----------------------------------------------------------------
    p = add("dataset", "Assemble a trainer-ready dataset directory.", _cmd_dataset)
    p.add_argument("project")
    p.add_argument("--out", type=Path, help="Destination (defaults to <project>/dataset)")
    p.add_argument("--copy", action="store_true", help="Copy images instead of hardlinking")
    p.add_argument("--json", action="store_true")

    # train ------------------------------------------------------------------
    p = add("train", "Run the configured external Gaussian trainer.", _cmd_train)
    p.add_argument("project")
    p.add_argument("--preset", default="draft", choices=["draft", "balanced", "high"])
    p.add_argument("--json", action="store_true")

    # colab ------------------------------------------------------------------
    p = add("colab", "Package a dataset.zip for a cloud GPU run.", _cmd_colab)
    p.add_argument("project")
    p.add_argument("--json", action="store_true")

    # export -----------------------------------------------------------------
    p = add("export", "Build an export bundle from the trained model.", _cmd_export)
    p.add_argument("project")
    p.add_argument("--type", default="web", choices=["raw", "web", "blender", "unity", "proxy_mesh"])
    p.add_argument("--json", action="store_true")

    # run --------------------------------------------------------------------
    p = add("run", "Run telemetry, frames, pose, and dataset in one pass.", _cmd_run)
    p.add_argument("project")
    p.add_argument("--preset", default="balanced", choices=["fast", "balanced", "dense"])
    p.add_argument("--skip-pose", action="store_true", help="Stop before COLMAP")
    p.add_argument("--json", action="store_true")

    return parser


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _resolve_project(reference: str) -> Path:
    """Accept a project id, an absolute path, or a relative path."""
    from gausscapture.project import ProjectStore

    candidate = Path(reference).expanduser()
    if (candidate / "project.json").exists():
        return candidate
    store = ProjectStore()
    try:
        return store.get(reference).path
    except FileNotFoundError:
        raise GaussCaptureError(
            f"No project matching '{reference}'. List projects with `gausscapture project list`."
        ) from None


def _emit(data: Any, as_json: bool, human: str | None = None) -> int:
    if as_json:
        print(json.dumps(data, indent=2, default=str))
    elif human is not None:
        print(human)
    return 0


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def _cmd_doctor(args, progress) -> int:
    from gausscapture.pose.colmap import colmap_available
    from gausscapture.recon.external import trainer_status

    settings = get_settings()
    checks = {
        "ffmpeg": bool(shutil.which(settings.ffmpeg_path)),
        "ffprobe": bool(shutil.which(settings.ffprobe_path)),
        "colmap": colmap_available(settings),
        "opencv": _importable("cv2"),
        "numpy": _importable("numpy"),
        "open3d (optional)": _importable("open3d"),
    }
    trainer = trainer_status(settings)
    report = {
        "version": __version__,
        "projects_dir": settings.projects_dir,
        "checks": checks,
        "trainer": trainer,
    }
    if args.json:
        return _emit(report, True)

    print(f"gausscapture {__version__}")
    print(f"projects: {settings.projects_dir}\n")
    for name, ok in checks.items():
        print(f"  {'ok  ' if ok else 'MISS'}  {name}")
    print()
    if trainer["usable"]:
        print(f"  ok    trainer: {trainer['entrypoint']}")
    elif trainer["configured"]:
        print(f"  MISS  trainer configured but unusable: {trainer['path']}")
    else:
        print("  --    no trainer configured (use `gausscapture colab` for cloud training)")

    if not checks["ffmpeg"]:
        print("\nInstall ffmpeg:  brew install ffmpeg")
    if not checks["colmap"]:
        print("Install COLMAP:  brew install colmap")
    return 0


def _cmd_project(args, progress) -> int:
    from gausscapture.project import ProjectStore

    store = ProjectStore()
    if args.action == "list":
        projects = store.list()
        if args.json:
            return _emit([p.to_dict() for p in projects], True)
        if not projects:
            print("No projects yet. Create one with `gausscapture project create <name>`.")
            return 0
        for p in projects:
            print(f"{p.id}  {p.status:<18}  {p.name}")
        return 0

    if args.action == "create":
        if not args.value:
            raise GaussCaptureError("A project name is required: `project create <name>`")
        project = store.create(args.value, args.target_type)
        return _emit(project.to_dict(), args.json, f"{project.id}  {project.path}")

    if not args.value:
        raise GaussCaptureError(f"A project id is required: `project {args.action} <id>`")

    if args.action == "show":
        return _emit(store.get(args.value).to_dict(), True)

    store.delete(args.value)
    return _emit({"deleted": args.value}, args.json, f"deleted {args.value}")


def _cmd_import(args, progress) -> int:
    from gausscapture.ingest.video import copy_video_into_pack, is_video_file, probe_video
    from gausscapture.pack import archive, manifest
    from gausscapture.project import STATUS_IMPORTED, ProjectStore

    store = ProjectStore()
    if args.project:
        project_path = _resolve_project(args.project)
        project = store.get(project_path.name)
    else:
        project = store.create(args.name or args.source.stem, args.target_type)
        project_path = project.path

    source = args.source.expanduser()
    if source.suffix.lower() == ".capturepack" or source.suffix.lower() == ".zip":
        result = archive.import_archive(project_path, source)
        payload = {
            "project": project.to_dict(),
            "valid": result["valid"],
            "warnings": result["warnings"],
            "errors": result["errors"],
        }
    elif is_video_file(source):
        for name in manifest.REQUIRED_DIRS:
            (project_path / "capturepack" / name).mkdir(parents=True, exist_ok=True)
        video_path = copy_video_into_pack(source, project_path)
        info = probe_video(video_path)
        pack_manifest = manifest.create_minimal_manifest(
            video_relpath=f"video/{video_path.name}",
            info=info,
            session_name=project.name,
            target_type=project.target_type,
        )
        manifest.write_manifest(project_path / "capturepack", pack_manifest)
        archive.write_checksums(project_path / "capturepack", progress)
        payload = {"project": project.to_dict(), "video": info.to_dict()}
    else:
        raise GaussCaptureError(f"Unsupported input: {source.name}")

    store.update(project.id, status=STATUS_IMPORTED, last_step="Imported capture")
    return _emit(payload, args.json, f"imported into {project.id}")


def _cmd_pack(args, progress) -> int:
    from gausscapture.pack import archive, manifest

    candidate = Path(args.target).expanduser()
    if (candidate / "manifest.json").exists():
        pack_dir = candidate
        project_path = candidate.parent
    else:
        project_path = _resolve_project(args.target)
        pack_dir = project_path / "capturepack"

    if args.action == "export":
        out = archive.export_archive(project_path)
        return _emit({"archive": str(out)}, args.json, str(out))

    result = manifest.validate(pack_dir)
    if args.verify:
        result["checksums"] = archive.verify_checksums(pack_dir)
    if args.json:
        result.pop("manifest", None)
        return _emit(result, True)

    print("valid" if result["valid"] else "INVALID")
    for error in result["errors"]:
        print(f"  error:   {error}")
    for warning in result["warnings"]:
        print(f"  warning: {warning}")
    if args.verify:
        checks = result["checksums"]
        print(f"  checksums: {'ok' if checks['verified'] else 'MISMATCH'} ({checks.get('checked', 0)} files)")
    return 0 if result["valid"] else 1


def _cmd_telemetry(args, progress) -> int:
    from gausscapture.telemetry import analyze_capture

    project_path = _resolve_project(args.project)
    report = analyze_capture(project_path, sample_count=args.samples, progress=progress)

    if args.json:
        return _emit(report.to_dict(include_signals=args.signals), True)

    print(report.summary())
    for warning in report.warnings:
        print(f"  warning: {warning}")
    for recommendation in report.recommendations:
        print(f"  hint:    {recommendation}")
    return 0


def _cmd_frames(args, progress) -> int:
    from gausscapture.ingest.frames import FRAME_PRESETS, extract_frames
    from gausscapture.project import STATUS_PREPROCESSED, ProjectStore

    project_path = _resolve_project(args.project)
    settings = dict(FRAME_PRESETS[args.preset])
    if args.fps:
        settings["target_fps"] = args.fps
    if args.max_frames:
        settings["max_frames"] = args.max_frames
    settings["blur_filter"] = not args.no_blur_filter
    settings["duplicate_filter"] = not args.no_duplicate_filter

    index = extract_frames(project_path, settings, progress=progress)
    ProjectStore().update(
        project_path.name, status=STATUS_PREPROCESSED, last_step="Frames extracted"
    )
    summary = (
        f"kept {index.frames_used} of {index.frames_total_sampled} sampled "
        f"(blur {index.frames_skipped_blur}, duplicate {index.frames_skipped_duplicate})"
    )
    return _emit(
        {k: v for k, v in index.to_dict().items() if k != "frames"}, args.json, summary
    )


def _cmd_pose(args, progress) -> int:
    from gausscapture.pose.colmap import run_colmap
    from gausscapture.project import STATUS_READY, ProjectStore

    project_path = _resolve_project(args.project)
    report = run_colmap(project_path, matcher=args.matcher, progress=progress)
    ProjectStore().update(project_path.name, status=STATUS_READY, last_step="Poses estimated")
    summary = (
        f"{report.status}: registered {report.images_registered}/{report.images_total} "
        f"({report.registered_ratio:.0%}), {report.sparse_points} points"
    )
    return _emit(report.to_dict(), args.json, summary)


def _cmd_dataset(args, progress) -> int:
    from gausscapture.recon.dataset import build_dataset

    project_path = _resolve_project(args.project)
    out = build_dataset(project_path, args.out, link=not args.copy, progress=progress)
    images = len(list((out / "images").glob("*.jpg")))
    return _emit({"dataset": str(out), "images": images}, args.json, f"{out}  ({images} images)")


def _cmd_train(args, progress) -> int:
    from gausscapture.project import STATUS_TRAINED, ProjectStore
    from gausscapture.recon.dataset import build_dataset
    from gausscapture.recon.external import ExternalTrainer

    project_path = _resolve_project(args.project)
    dataset_dir = build_dataset(project_path, progress=progress)
    run_dir = _next_run_dir(project_path / "training")
    summary = ExternalTrainer().fit(
        dataset_dir, run_dir / "output", preset=args.preset, progress=progress
    )
    ProjectStore().update(project_path.name, status=STATUS_TRAINED, last_step="Training complete")
    return _emit(summary, args.json, f"trained: {summary['models'][0]}")


def _cmd_colab(args, progress) -> int:
    from gausscapture.export.colab import create_colab_package

    project_path = _resolve_project(args.project)
    result = create_colab_package(project_path, progress=progress)
    human = f"{result['dataset_zip']}  ({result['size_bytes'] / 1e6:.1f} MB)"
    return _emit(result, args.json, human)


def _cmd_export(args, progress) -> int:
    from gausscapture.export.bundles import create_export

    project_path = _resolve_project(args.project)
    result = create_export(project_path, args.type)
    human = f"{result['path']}  ({result['size_bytes'] / 1e6:.1f} MB)"
    return _emit(result, args.json, human)


def _cmd_run(args, progress) -> int:
    """Chain the deterministic stages, which is what batch evaluation needs."""
    from gausscapture.ingest.frames import FRAME_PRESETS, extract_frames
    from gausscapture.pose.colmap import colmap_available, run_colmap
    from gausscapture.recon.dataset import build_dataset
    from gausscapture.telemetry import analyze_capture

    project_path = _resolve_project(args.project)
    results: dict[str, Any] = {"project": str(project_path)}

    progress.update(0, "Analysing capture quality")
    telemetry = analyze_capture(project_path, progress=progress)
    results["telemetry"] = telemetry.to_dict(include_signals=False)

    progress.update(0, "Extracting frames")
    index = extract_frames(project_path, dict(FRAME_PRESETS[args.preset]), progress=progress)
    results["frames"] = {k: v for k, v in index.to_dict().items() if k != "frames"}

    if args.skip_pose:
        results["pose"] = {"skipped": True}
    elif not colmap_available():
        results["pose"] = {"skipped": True, "reason": "COLMAP not installed"}
    else:
        progress.update(0, "Estimating poses")
        report = run_colmap(project_path, progress=progress)
        results["pose"] = report.to_dict()
        if report.status != "bad":
            results["dataset"] = str(build_dataset(project_path, progress=progress))

    if args.json:
        return _emit(results, True)

    print(telemetry.summary())
    print(f"frames: kept {index.frames_used}/{index.frames_total_sampled}")
    pose = results.get("pose", {})
    if pose.get("skipped"):
        print(f"pose:   skipped ({pose.get('reason', 'requested')})")
    else:
        print(f"pose:   {pose['status']} ({pose['registered_ratio']:.0%} registered)")
    if "dataset" in results:
        print(f"dataset: {results['dataset']}")
    return 0


def _next_run_dir(training_dir: Path) -> Path:
    training_dir.mkdir(parents=True, exist_ok=True)
    index = len([p for p in training_dir.glob("run_*") if p.is_dir()]) + 1
    run_dir = training_dir / f"run_{index:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _importable(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


if __name__ == "__main__":
    raise SystemExit(main())
