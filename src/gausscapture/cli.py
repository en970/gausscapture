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

    # pull -------------------------------------------------------------------
    p = add("pull", "Take new captures off a connected phone.", _cmd_pull)
    p.add_argument("--list", action="store_true",
                   help="Show what is on the phone without copying anything")
    p.add_argument("--all", action="store_true",
                   help="Re-import captures already present locally")
    p.add_argument("--serial", default=None, help="Target one device when several are attached")
    p.add_argument("--adb", default="adb", help="Path to adb")
    p.add_argument("--json", action="store_true", help="Emit JSON")

    # pack -------------------------------------------------------------------
    p = add("pack", "Validate or export a CapturePack.", _cmd_pack)
    p.add_argument("action", choices=["validate", "export"])
    p.add_argument("target", help="Project id, project path, or a pack directory")
    p.add_argument("--verify", action="store_true", help="Re-hash files against the checksums")
    p.add_argument("--out", type=Path, help="Destination archive path (export)")
    p.add_argument(
        "--with-dataset",
        action="store_true",
        help="Include images, the COLMAP model and transforms.json, making the archive trainable",
    )
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
    p.add_argument(
        "--seed-intrinsics",
        action="store_true",
        help="Hand COLMAP the device's calibration (measured: no benefit, 17-61%% slower)",
    )
    p.add_argument(
        "--fix-intrinsics",
        action="store_true",
        help="Seed device intrinsics and forbid bundle adjustment from refining them",
    )
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

    # bench ------------------------------------------------------------------
    p = add("bench", "Batch-evaluate captures and analyse the results.", _cmd_bench)
    p.add_argument("action", choices=["run", "analyze", "analyse"])
    p.add_argument(
        "target",
        type=Path,
        help="Directory of videos/packs (run), or a results.jsonl (analyze)",
    )
    p.add_argument("--out", type=Path, help="Output directory for a run")
    p.add_argument("--preset", default="balanced", choices=["fast", "balanced", "dense"])
    p.add_argument("--matcher", default="sequential", choices=["sequential", "exhaustive"])
    p.add_argument("--skip-pose", action="store_true", help="Telemetry and frames only")
    p.add_argument(
        "--seed-intrinsics",
        action="store_true",
        help="Hand COLMAP the device's calibration (measured: no benefit, 17-61%% slower)",
    )
    p.add_argument(
        "--fix-intrinsics",
        action="store_true",
        help="Seed device intrinsics and forbid bundle adjustment from refining them",
    )
    p.add_argument(
        "--outcome",
        default="registered_ratio",
        help="Column to correlate signals against (analyze)",
    )
    p.add_argument(
        "--permutations", type=int, default=10_000, help="Permutation count for p-values"
    )
    p.add_argument("--json", action="store_true")

    # report -----------------------------------------------------------------
    p = add("report", "Build a local, offline report from a benchmark run.", _cmd_report)
    p.add_argument("run", type=Path, help="A benchmark run directory")
    p.add_argument("--out", type=Path, help="Where to write it (defaults to <run>/report)")
    p.add_argument("--splat", type=Path, action="append", default=None,
                   help="A trained .ply to include; repeatable")
    p.add_argument("--serve", action="store_true", help="Serve it and print the URL")
    p.add_argument("--port", type=int, default=8800)
    p.add_argument("--json", action="store_true")

    # viewer -----------------------------------------------------------------
    p = add("viewer", "Build a browser viewer for trained splats.", _cmd_viewer)
    p.add_argument("ply", type=Path, nargs="+", help="One or more trained .ply files")
    p.add_argument("--out", type=Path, default=Path("viewer"), help="Output directory")
    p.add_argument("--title", default="GaussCapture")
    p.add_argument("--min-opacity", type=float, default=0.02,
                   help="Drop gaussians fainter than this")
    p.add_argument("--max-gaussians", type=int, default=None,
                   help="Cap the count, keeping the most significant")
    p.add_argument("--colmap", type=Path, default=None,
                   help="The sparse model the splat was trained from; its cameras "
                        "fix which way is up")
    p.add_argument("--serve", action="store_true")
    p.add_argument("--port", type=int, default=8900)
    p.add_argument("--json", action="store_true")

    # ---- the fixed-camera 4D path -------------------------------------------
    # A stationary camera produces no parallax, so these commands never claim a
    # free-viewpoint scene. They produce a bullet-time scene bounded by the cone
    # the hand-held arc actually swept, and every one of them reports what was
    # measured rather than what was hoped for.

    # mask ---------------------------------------------------------------------
    p = add("mask", "Measure the dynamic region of a fixed-camera capture.", _cmd_mask)
    p.add_argument("project")
    p.add_argument("--json", action="store_true")

    # prep4d -------------------------------------------------------------------
    p = add("prep4d", "Prepare a 4D scene for training: mask, poses, cone, canonical init.",
            _cmd_prep4d)
    p.add_argument("project")
    p.add_argument("--nodes", type=int, default=256,
                   help="SE(3) scaffold nodes seeded over the dynamic region")
    p.add_argument("--skin-k", type=int, default=4,
                   help="Scaffold nodes each dynamic gaussian is bound to")
    p.add_argument("--cap-max", type=int, default=400_000,
                   help="Gaussian budget the trainer will be capped at")
    p.add_argument("--json", action="store_true")

    # colab4d ------------------------------------------------------------------
    p = add("colab4d", "Package dataset4d.zip for a cloud GPU, or collect the trained scene.",
            _cmd_colab4d)
    p.add_argument("project")
    p.add_argument("--out", type=Path, help="Destination archive (defaults to the project)")
    p.add_argument("--collect", type=Path, metavar="SCENE4D_NPZ",
                   help="Adopt a scene4d.npz trained elsewhere instead of packaging")
    p.add_argument("--coarse-iters", type=int, default=3000,
                   help="Recorded in the package so the GPU host needs no flags")
    p.add_argument("--iters", type=int, default=12_000)
    p.add_argument("--cap-max", type=int, default=400_000)
    p.add_argument("--json", action="store_true")

    # train4d ------------------------------------------------------------------
    p = add("train4d", "Train the deformable 4D scene. Needs torch; a CUDA GPU to be useful.",
            _cmd_train4d)
    p.add_argument("target", help="Project id or path, or a dataset4d.zip")
    p.add_argument("--out", type=Path, help="Where to write scene4d.npz")
    p.add_argument("--coarse-iters", type=int, default=3000,
                   help="Static stage: deformation forced to identity, MCMC densifies")
    p.add_argument("--iters", type=int, default=12_000,
                   help="Deformable stage: densification off, position noise driven to zero")
    p.add_argument("--cap-max", type=int, default=400_000)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--field", default="explicit", choices=["explicit", "kplanes"],
                   help="Motion parameterisation: free per-node keyframes, or the "
                        "multi-resolution K-Planes field that predicts them")
    p.add_argument("--checkpoint-every", type=int, default=1000, metavar="STEPS",
                   help="Write a resumable checkpoint beside the output; 0 disables it")
    p.add_argument("--resume", action="store_true",
                   help="Continue from that checkpoint instead of starting over")
    p.add_argument("--json", action="store_true")

    # export4d -----------------------------------------------------------------
    p = add("export4d", "Write a .g4d bullet-time scene from a trained 4D model.", _cmd_export4d)
    p.add_argument("target", help="Project id or path, or a scene4d.npz")
    p.add_argument("--out", type=Path, help="Destination .g4d")
    p.add_argument("--skin-k", type=int, default=4)
    p.add_argument("--min-opacity", type=float, default=0.02,
                   help="Drop gaussians fainter than this")
    p.add_argument("--max-gaussians", type=int, default=None,
                   help="Cap the count, keeping the most significant of each group")
    p.add_argument("--json", action="store_true")

    # viewer4d -----------------------------------------------------------------
    p = add("viewer4d", "Build the browser bullet-time viewer for a .g4d scene.", _cmd_viewer4d)
    p.add_argument("target", help="Project id or path, or a scene.g4d")
    p.add_argument("--out", type=Path, help="Output directory")
    p.add_argument("--single-file", action="store_true",
                   help="Inline everything into one .html")
    p.add_argument("--cone-azimuth", type=float, default=None,
                   help="Narrow the published azimuth half-angle; it can never widen "
                        "the cone that was measured")
    p.add_argument("--cone-elevation", type=float, default=None,
                   help="Narrow the published elevation half-angle")
    p.add_argument("--serve", action="store_true")
    p.add_argument("--port", type=int, default=8900)
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


def _project_or_file(reference: str, suffixes: tuple[str, ...]) -> tuple[Path | None, Path | None]:
    """Resolve an argument that may name a project or a bare artefact.

    The 4D commands accept both because training happens on someone else's
    machine: what comes back is a file, and being made to import it into a
    project before it can be looked at would be a step that exists only to
    satisfy the tool.
    """
    candidate = Path(reference).expanduser()
    if candidate.suffix.lower() in suffixes:
        if not candidate.exists():
            raise GaussCaptureError(f"No such file: {candidate}")
        return None, candidate
    return _resolve_project(reference), None


def _require_frames(project_path: Path) -> None:
    """Refuse a step whose pixels do not exist yet."""
    from gausscapture.errors import PipelineStateError

    images = project_path / "frames" / "images"
    if not images.is_dir() or not any(images.iterdir()):
        raise PipelineStateError(
            f"No extracted frames in {images}. Run `gausscapture frames <project>` first."
        )


def _set_status(project_path: Path, status: str, step: str) -> None:
    """Record a stage on a project, tolerating one addressed by bare path.

    ``_resolve_project`` accepts a directory anywhere on disk, and such a
    directory has no entry in the store, so a status write is a courtesy rather
    than a contract. The artefacts on disk are what the pipeline reads back.
    """
    import contextlib

    from gausscapture.project import ProjectStore

    with contextlib.suppress(FileNotFoundError):
        ProjectStore().update(project_path.name, status=status, last_step=step)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def _cmd_doctor(args, progress) -> int:
    from gausscapture.pose.colmap import colmap_available
    from gausscapture.recon.external import deform_trainer_status, trainer_status

    settings = get_settings()
    checks = {
        "ffmpeg": bool(shutil.which(settings.ffmpeg_path)),
        "ffprobe": bool(shutil.which(settings.ffprobe_path)),
        "colmap": colmap_available(settings),
        "opencv": _importable("cv2"),
        "numpy": _importable("numpy"),
        "open3d (optional)": _importable("open3d"),
        # Exact k-nearest-neighbour scale initialisation uses cKDTree above
        # 20k points; below that a NumPy brute force covers it, which is why
        # this is optional rather than a dependency.
        "scipy (optional, 4D)": _importable("scipy"),
    }
    trainer = trainer_status(settings)
    fourd = deform_trainer_status(probe_cuda=True)
    report = {
        "version": __version__,
        "projects_dir": settings.projects_dir,
        "checks": checks,
        "trainer": trainer,
        "fourd_trainer": fourd,
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

    # The 4D row says what this machine can do, not what the pipeline needs, so
    # a laptop with no CUDA reads as "prepare and export here, train elsewhere"
    # rather than as a broken install.
    mark = "ok  " if fourd["gpu_usable"] else ("--  " if fourd["usable"] else "--  ")
    versions = ", ".join(
        name
        for name, present in (
            (f"torch {fourd['torch_version']}".strip(), fourd["torch"]),
            ("gsplat", fourd["gsplat"]),
            ("cuda", bool(fourd["cuda"])),
        )
        if present
    )
    print(f"  {mark}  4D training: {versions or 'not available here'}")
    print(f"        {fourd['note']}")

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
        project = store.get(args.value)
        payload = project.to_dict()
        # Only for projects that have started down the 4D path, so a static
        # project's output does not grow a block of nulls it will never fill.
        state = project.fourd_state()
        if state.is_fourd:
            payload["fourd"] = state.to_dict()
        return _emit(payload, True)

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


def _cmd_pull(args, progress) -> int:
    """Import every capture the phone has that this machine does not."""
    import socket
    from datetime import datetime, timezone

    from gausscapture.ingest.pull import (
        devices,
        known_session_ids,
        list_captures,
        mark_offloaded,
        pull_capture,
    )
    from gausscapture.pack import archive
    from gausscapture.project import STATUS_IMPORTED, ProjectStore

    attached = devices(args.adb)
    if not attached:
        raise GaussCaptureError(
            "No phone found. Connect it by USB, unlock it, and allow USB debugging."
        )
    serial = args.serial or attached[0]
    if len(attached) > 1 and not args.serial:
        progress.log(f"{len(attached)} devices attached; using {serial}")

    captures = list_captures(args.adb, serial)
    if not captures:
        return _emit({"device": serial, "captures": []}, args.json,
                     "The phone has no captures on it")

    store = ProjectStore()
    known = set() if args.all else known_session_ids(store)

    # A capture with no session id cannot be recognised on a later run, so it is
    # matched by name instead -- less reliable, but better than importing it
    # again on every plug-in.
    existing_names = {p.name for p in store.list()}
    fresh = [
        c for c in captures
        if c.usable
        and (c.session_id not in known if c.session_id else c.name not in existing_names)
    ]
    skipped = len(captures) - len(fresh)

    if args.list or not fresh:
        for capture in captures:
            state = "new" if capture in fresh else ("unusable" if not capture.usable else "have")
            progress.log(f"  [{state:>8}] {capture.name}  {capture.bytes / (1 << 20):.0f} MB"
                         + (f"  missing {', '.join(capture.missing)}" if capture.missing else ""))
        summary = f"{len(fresh)} new, {skipped} already here"
        return _emit(
            {"device": serial, "captures": [c.to_dict() for c in captures],
             "new": [c.name for c in fresh]},
            args.json, summary,
        )

    imported = []
    for index, capture in enumerate(fresh, start=1):
        progress.update(
            int(100 * (index - 1) / len(fresh)), f"pulling {capture.name} ({index}/{len(fresh)})"
        )
        project = store.create(capture.name, "unknown")
        try:
            pull_capture(capture, project.path / "capturepack",
                         args.adb, serial, progress=progress)
        except GaussCaptureError as error:
            # The project is empty, so removing it keeps a failed pull from
            # leaving a shell that later looks like a real capture.
            store.delete(project.id)
            progress.log(f"  {capture.name}: {error}")
            continue

        archive.write_checksums(project.path / "capturepack", progress)
        store.update(project.id, status=STATUS_IMPORTED,
                     last_step=f"Pulled from {serial}")

        # Only now, with the files verified and checksummed, is it true that this capture exists
        # somewhere other than the phone. The app reads this to show which takes are still the
        # only copy, so writing it any earlier would mark something safe that is not.
        marked = mark_offloaded(
            capture,
            host=socket.gethostname(),
            when=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            adb=args.adb,
            serial=serial,
        )
        if not marked:
            progress.log(f"  {capture.name}: copied, but the phone could not be marked")
        imported.append({"capture": capture.name, "project": project.id,
                         "marked_on_phone": marked})

    progress.update(100, "done")
    return _emit(
        {"device": serial, "imported": imported, "skipped": skipped},
        args.json,
        f"imported {len(imported)} capture(s); {skipped} already here",
    )


def _cmd_pack(args, progress) -> int:
    from gausscapture.pack import archive, manifest, transforms

    candidate = Path(args.target).expanduser()
    if (candidate / "manifest.json").exists():
        pack_dir = candidate
        project_path = candidate.parent
    else:
        project_path = _resolve_project(args.target)
        pack_dir = project_path / "capturepack"

    if args.action == "export":
        out = archive.export_archive(
            project_path,
            out_path=args.out,
            include_dataset=args.with_dataset,
            progress=progress,
        )
        size_mb = out.stat().st_size / 1e6
        return _emit(
            {"archive": str(out), "size_bytes": out.stat().st_size, "bagit": True},
            args.json,
            f"{out}  ({size_mb:.1f} MB, BagIt)",
        )

    result = manifest.validate(pack_dir)
    if args.verify:
        result["checksums"] = archive.verify_checksums(pack_dir)

    transforms_path = pack_dir / "transforms.json"
    if transforms_path.exists():
        result["transforms"] = transforms.validate_transforms(transforms_path, pack_dir)

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
        state = "ok" if checks["verified"] else "MISMATCH"
        print(f"  checksums: {state} ({checks.get('checked', 0)} files)")
    if "transforms" in result:
        t = result["transforms"]
        state = "ok" if t["valid"] else "INVALID"
        print(f"  transforms.json: {state} ({t['frames']} poses, {t.get('camera_model')})")
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
    report = run_colmap(
        project_path,
        matcher=args.matcher,
        progress=progress,
        seed_intrinsics=args.seed_intrinsics,
        refine_intrinsics=not args.fix_intrinsics,
    )
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


def _cmd_bench(args, progress) -> int:
    from gausscapture.eval.analysis import analyse
    from gausscapture.eval.harness import load_results, run_batch, write_csv
    from gausscapture.ingest.video import VIDEO_EXTENSIONS

    if args.action == "run":
        target = args.target.expanduser()
        if (target / "manifest.json").exists():
            sources = [target]  # a single unpacked capture directory
        elif target.is_dir():
            sources = sorted(
                p
                for p in target.iterdir()
                # Capture directories written by the native app, alongside plain
                # videos and packed archives.
                if (p.is_dir() and (p / "manifest.json").exists())
                or p.suffix.lower() in VIDEO_EXTENSIONS
                or p.suffix.lower() == ".capturepack"
            )
        else:
            sources = [target]
        if not sources:
            raise GaussCaptureError(f"No videos or capture packs found in {target}")

        out_dir = args.out or (Path("benchmarks") / "runs" / target.name)
        records = run_batch(
            sources,
            out_dir,
            frame_preset=args.preset,
            matcher=args.matcher,
            skip_pose=args.skip_pose,
            seed_intrinsics=args.seed_intrinsics,
            refine_intrinsics=not args.fix_intrinsics,
            progress=progress,
        )
        write_csv(records, out_dir / "results.csv")

        summary = {
            "out_dir": str(out_dir),
            "captures": len(records),
            "registered": sum(1 for r in records if r.registered),
            "errors": sum(1 for r in records if r.error),
            "results": str(out_dir / "results.jsonl"),
        }
        if args.json:
            return _emit(summary, True)

        print(f"\n{'capture':<22} {'frames':>7} {'reg':>6} {'ratio':>7} {'points':>8}  status")
        for record in records:
            print(
                f"{record.name[:22]:<22} {record.frames_used:>7} "
                f"{'yes' if record.registered else 'no':>6} "
                f"{record.registered_ratio:>6.0%} {record.sparse_points:>8}  "
                f"{record.error or record.pose_status}"
            )
        print(f"\nresults: {out_dir}/results.jsonl  (+ .csv)")
        return 0

    # analyze ---------------------------------------------------------------
    path = args.target.expanduser()
    if path.is_dir():
        path = path / "results.jsonl"
    report = analyse(
        load_results(path), outcome=args.outcome, permutations=args.permutations
    )
    if args.json:
        return _emit(report.to_dict(), True)
    print(report.render())
    return 0


def _cmd_report(args, progress) -> int:
    from gausscapture.report import build_report

    run_dir = args.run.expanduser()
    if not (run_dir / "results.jsonl").exists():
        raise GaussCaptureError(f"No results.jsonl in {run_dir}; is that a benchmark run?")

    out_dir = args.out or (run_dir / "report")
    index = build_report(run_dir, out_dir, splats=args.splat)
    payload = {"report": str(index), "directory": str(out_dir)}

    if not args.serve:
        return _emit(payload, args.json, f"{index}\n  open it directly, or add --serve")
    return _serve_directory(out_dir, args.port, [f"  serving {out_dir}"])


def _cmd_viewer(args, progress) -> int:
    from gausscapture.report.splat_site import build_splat_site

    index = build_splat_site(
        [p.expanduser() for p in args.ply],
        args.out.expanduser(),
        title=args.title,
        min_opacity=args.min_opacity,
        max_gaussians=args.max_gaussians,
        colmap_model=args.colmap.expanduser() if args.colmap else None,
    )
    out_dir = index.parent
    scenes = json.loads((out_dir / "scenes.json").read_text(encoding="utf-8"))
    payload = {
        "viewer": str(index),
        "scenes": [{"label": s["label"], "gaussians": s["count"],
                    "bytes": s["bytes"]} for s in scenes],
    }

    lines = [
        f"  {s['label']}: {s['count']:,} gaussians, {s['bytes'] / 1e6:.1f} MB" for s in scenes
    ]
    if not args.serve:
        return _emit(payload, args.json, "\n".join([str(index), *lines, "  add --serve to open it"]))
    return _serve_directory(out_dir, args.port, lines)


def _cmd_mask(args, progress) -> int:
    from gausscapture.project import MASKS_SUBDIR, STATUS_MASKED, record_fourd_stage

    project_path = _resolve_project(args.project)
    _require_frames(project_path)

    from gausscapture.ingest.phases import phase_split_for_pack
    from gausscapture.recon.dynamic_mask import build_dynamic_masks

    split = phase_split_for_pack(project_path / "capturepack")
    hold = split.require("hold")
    hold_paths = _frames_in_range(project_path, hold.start_frame, hold.end_frame)
    rest_window = split.rest_window
    rest_paths = (
        _frames_in_range(project_path, rest_window[0], rest_window[1]) if rest_window else []
    )

    report, _union = build_dynamic_masks(
        hold_paths,
        project_path / MASKS_SUBDIR,
        rest_paths=rest_paths,
        progress=progress,
    )
    payload = report.to_dict()
    payload["phases"] = split.to_dict()

    record_fourd_stage(
        project_path,
        "masked",
        dynamic_fraction=report.union_fraction,
        mask_threshold_levels=report.threshold_levels,
        noise_source=report.noise_source,
    )
    _set_status(project_path, STATUS_MASKED, "Dynamic region measured")

    if args.json:
        return _emit(payload, True)

    print(f"{report.frames} masks in {report.masks_dir}")
    print(
        f"  {report.union_fraction:.1%} of the frame moves at some point; "
        f"threshold {report.threshold_levels:.1f} grey levels "
        f"from the {report.noise_source} noise floor"
    )
    for warning in [*split.warnings, *report.warnings]:
        print(f"  warning: {warning}")
    return 0


def _frames_in_range(project_path: Path, first: int, last: int) -> list[Path]:
    """Extracted images whose source frame number falls inside a phase.

    The frame index records the source frame number of every image it kept, so
    a phase measured on the video maps onto files on disk without re-decoding
    anything.
    """
    from gausscapture.errors import PipelineStateError

    index_path = project_path / "frames" / "frame_index.json"
    if not index_path.exists():
        raise PipelineStateError(
            f"No frame index at {index_path}. Run `gausscapture frames <project>` first."
        )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    # ``file`` is recorded relative to the project root, not to frames/.
    paths = [
        project_path / record["file"]
        for record in index.get("frames", [])
        if record.get("used") and first <= record["index"] <= last and record.get("file")
    ]
    return [p for p in paths if p.exists()]


def _cmd_prep4d(args, progress) -> int:
    from gausscapture.project import STATUS_PREPARED_4D, record_fourd_stage

    project_path = _resolve_project(args.project)
    _require_frames(project_path)

    from gausscapture.recon.prepare4d import prepare_4d

    result = prepare_4d(
        project_path,
        nodes=args.nodes,
        skin_k=args.skin_k,
        cap_max=args.cap_max,
        progress=progress,
    )
    cone = result.get("cone", {}) or {}
    fixed = result.get("fixed_pose", {}) or {}
    record_fourd_stage(
        project_path,
        "prepared",
        init=result.get("init_path"),
        cone=cone,
        fixed_pose=fixed,
        gaussians=result.get("gaussians"),
        dynamic_gaussians=result.get("dynamic_gaussians"),
        nodes=result.get("nodes"),
    )
    _set_status(project_path, STATUS_PREPARED_4D, "4D scene prepared")

    if args.json:
        return _emit(result, True)

    print(
        f"init:  {result.get('init_path')}\n"
        f"  {result.get('gaussians', 0):,} gaussians "
        f"({result.get('dynamic_gaussians', 0):,} dynamic), "
        f"{result.get('nodes', 0)} scaffold nodes"
    )
    # The keys are ConeReport's own: half-angles, because a cone that reported a
    # full angle under the same name as a half one would be wrong by a factor of
    # two in the one number the viewer clamps to.
    print(
        f"  cone: +/-{cone.get('azimuth_half_deg', 0.0):.1f} deg azimuth "
        f"x +/-{cone.get('elevation_half_deg', 0.0):.1f} deg elevation, "
        f"measured from {cone.get('cameras', 0)} arc cameras"
    )
    source = fixed.get("source", "unknown")
    residual = fixed.get("residual_px")
    if source == "asserted":
        # Tier 3 is the quiet failure mode: the pipeline completes, and nothing
        # about the result says the camera pose was never checked against the
        # scene. It is said here, and again in the viewer.
        print(
            "  pose: ASSERTED, not verified -- the masked hold keyframes did not register, "
            "so the fixed pose is inherited and unchecked"
        )
    elif residual is None:
        # No residual means nothing was compared, not that nothing disagreed.
        print(f"  pose: {source}, no pixel-space comparison was possible")
    else:
        print(f"  pose: {source}, residual {float(residual):.1f} px")
    for warning in result.get("warnings", []):
        print(f"  warning: {warning}")
    return 0


def _cmd_colab4d(args, progress) -> int:
    from gausscapture.export.colab4d import collect_trained_scene, create_colab4d_package
    from gausscapture.project import STATUS_PACKAGED_4D, STATUS_TRAINED_4D

    project_path = _resolve_project(args.project)

    if args.collect is not None:
        result = collect_trained_scene(project_path, args.collect)
        _set_status(project_path, STATUS_TRAINED_4D, "Collected a trained 4D scene")
        return _emit(
            result,
            args.json,
            f"{result['scene']}  ({result['size_bytes'] / 1e6:.1f} MB)\n"
            f"  export it with `gausscapture export4d {args.project}`",
        )

    result = create_colab4d_package(
        project_path,
        out=args.out,
        training_config={
            "coarse_iters": args.coarse_iters,
            "iters": args.iters,
            "cap_max": args.cap_max,
        },
        progress=progress,
    )
    _set_status(project_path, STATUS_PACKAGED_4D, "Packaged for GPU training")
    human = (
        f"{result['dataset_zip']}  ({result['size_bytes'] / 1e6:.1f} MB, "
        f"{result['images']} frames)\n"
        "  train it with notebooks/GaussCapture_Colab_4D.ipynb, then bring it back:\n"
        f"  gausscapture colab4d {args.project} --collect scene4d.npz"
    )
    return _emit(result, args.json, human)


def _cmd_train4d(args, progress) -> int:
    from gausscapture.errors import DependencyMissingError
    from gausscapture.project import (
        SCENE4D_RELPATH,
        STATUS_TRAINED_4D,
        FourDState,
        record_fourd_stage,
    )
    from gausscapture.recon.external import deform_trainer_status

    project_path, dataset_zip = _project_or_file(args.target, (".zip",))
    if project_path is not None:
        state = FourDState.observe(project_path)
        state.require("packaged", "Package it first: `gausscapture colab4d <project>`.")
        dataset_zip = state.dataset_zip
    assert dataset_zip is not None

    out = args.out or (
        project_path / SCENE4D_RELPATH if project_path else dataset_zip.with_name("scene4d.npz")
    )

    # Checked before the trainer is imported, so a machine without the extra
    # gets the sentence that tells it what to do instead of an ImportError.
    status = deform_trainer_status(probe_cuda=False)
    if not status["usable"]:
        raise DependencyMissingError(status["note"])
    if args.device == "cuda" and not status["gsplat"]:
        raise DependencyMissingError(
            "gsplat is not installed, and the CPU reference rasterizer is a correctness "
            "check rather than a trainer. Install `gausscapture[train4d]`, or package the "
            "scene with `gausscapture colab4d` and train on a cloud GPU."
        )

    from gausscapture.recon.fit4d import train_4d

    summary = train_4d(
        dataset_zip,
        Path(out),
        coarse_iters=args.coarse_iters,
        iters=args.iters,
        cap_max=args.cap_max,
        device=args.device,
        field_kind=args.field,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        progress=progress,
    )
    if project_path is not None:
        record_fourd_stage(
            project_path,
            "trained",
            scene4d=str(out),
            iterations=args.coarse_iters + args.iters,
            device=args.device,
            holdout_psnr_db=summary.get("holdout_psnr_db"),
        )
        _set_status(project_path, STATUS_TRAINED_4D, "4D training complete")

    if args.json:
        return _emit(summary, True)

    print(f"{out}  ({Path(out).stat().st_size / 1e6:.1f} MB)")
    print(
        f"  {summary.get('gaussians', 0):,} gaussians, {summary.get('nodes', 0)} nodes, "
        f"{summary.get('frames', 0)} frames"
    )
    if summary.get("holdout_psnr_db") is not None:
        # D9: there is no held-out viewpoint anywhere in this design, so this
        # number must never be read as novel-view synthesis quality.
        print(
            f"  temporal hold-out PSNR {summary['holdout_psnr_db']:.1f} dB "
            "-- motion interpolation only, not novel view"
        )
    return 0


def _cmd_export4d(args, progress) -> int:
    from gausscapture.project import (
        SCENE4D_G4D_RELPATH,
        STATUS_EXPORTED_4D,
        FourDState,
        record_fourd_stage,
    )

    project_path, scene_npz = _project_or_file(args.target, (".npz",))
    if project_path is not None:
        state = FourDState.observe(project_path)
        state.require(
            "trained",
            "Train it first, or adopt a scene trained elsewhere with "
            "`gausscapture colab4d <project> --collect scene4d.npz`.",
        )
        scene_npz = state.scene_path
    assert scene_npz is not None

    out = args.out or (
        project_path / SCENE4D_G4D_RELPATH if project_path else scene_npz.with_suffix(".g4d")
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    from gausscapture.export.checkpoint4d import export_scene4d

    result = export_scene4d(
        scene_npz,
        Path(out),
        skin_k=args.skin_k,
        min_opacity=args.min_opacity,
        max_gaussians=args.max_gaussians,
        progress=progress,
    )
    if project_path is not None:
        record_fourd_stage(project_path, "exported", g4d=str(out),
                           g4d_bytes=result["size_bytes"])
        _set_status(project_path, STATUS_EXPORTED_4D, "4D scene exported")

    if args.json:
        return _emit(result, True)

    print(f"{result['path']}  ({result['size_bytes'] / 1e6:.1f} MB, "
          f"{result['bytes_per_gaussian']:.1f} bytes/gaussian)")
    print(
        f"  {result['gaussian_count']:,} gaussians ({result['dynamic_count']:,} dynamic), "
        f"{result['node_count']} nodes x {result['frame_count']} frames "
        f"at {result['fps']:g} fps"
    )
    if result["pruned_faint"]:
        print(f"  dropped {result['pruned_faint']:,} gaussians below {args.min_opacity} opacity")
    if result["truncated"]:
        # A truncated scene is not the scene that was trained, so it is said
        # here and again in the viewer rather than only in scene.json.
        dropped = result["trained_gaussian_count"] - result["gaussian_count"]
        print(f"  TRUNCATED to the budget: {dropped:,} more dropped in importance order")
    return 0


def _cmd_viewer4d(args, progress) -> int:
    from gausscapture.project import VIEWER4D_RELPATH, FourDState

    project_path, g4d_path = _project_or_file(args.target, (".g4d",))
    if project_path is not None:
        state = FourDState.observe(project_path)
        state.require("exported", "Export it first: `gausscapture export4d <project>`.")
        g4d_path = state.g4d_path
    assert g4d_path is not None

    out_dir = Path(
        args.out
        or (project_path / VIEWER4D_RELPATH if project_path else g4d_path.parent / "viewer4d")
    )

    from gausscapture.report.scene4d_site import build_scene4d_site

    index = build_scene4d_site(
        g4d_path,
        out_dir,
        single_file=args.single_file,
        cone_azimuth_deg=args.cone_azimuth,
        cone_elevation_deg=args.cone_elevation,
    )
    scene: dict[str, Any] = {}
    scene_path = out_dir / "scene.json"
    if scene_path.exists():
        scene = json.loads(scene_path.read_text(encoding="utf-8"))

    cone = scene.get("cone", {}) or {}
    lines = [
        f"  trained view: +/-{cone.get('azimuth_deg', 0.0):.1f} deg "
        f"x +/-{cone.get('elevation_deg', 0.0):.1f} deg -- the viewer clamps to it",
    ]
    if scene.get("fixed_pose_source") == "asserted":
        lines.append("  the fixed camera pose was asserted, not verified")
    payload = {"viewer": str(index), "directory": str(out_dir), "scene": scene}

    if not args.serve:
        return _emit(payload, args.json, "\n".join([str(index), *lines, "  add --serve to open it"]))
    return _serve_directory(out_dir, args.port, lines)


def _serve_directory(out_dir: Path, port: int, lines: list[str]) -> int:
    """Serve a built site over HTTP.

    A module page fetches its own data, and fetch() is blocked on file:// URLs,
    so even a wholly static viewer needs a server.
    """
    import functools
    import http.server
    import socketserver

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out_dir))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), handler) as server:
        print(f"http://127.0.0.1:{port}/")
        for line in lines:
            print(line)
        print("  ctrl-c to stop")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("stopped")
    return 0


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
