from __future__ import annotations

import contextlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from gausscapture.config import Settings, get_settings
from gausscapture.errors import DependencyMissingError, PipelineStateError
from gausscapture.progress import NullProgress, Progress
from gausscapture.recon.base import TRAINING_PRESETS

MODEL_SUFFIXES = (".ply", ".splat", ".spz", ".sog")

#: Matches "Iteration 3000/30000" and "step 3000" style progress lines. Trainers
#: differ, so this is best-effort: a missed match costs a progress bar, not a run.
_ITER_PATTERN = re.compile(r"(?:iter(?:ation)?|step)\D{0,4}(\d{2,7})", re.IGNORECASE)


def trainer_status(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    configured = bool(settings.gaussian_trainer_path)
    path = Path(settings.gaussian_trainer_path).expanduser() if configured else None
    entry = _find_entrypoint(path) if path and path.exists() else None
    return {
        "configured": configured,
        "exists": bool(path and path.exists()),
        "path": str(path) if path else "",
        "entrypoint": str(entry) if entry else "",
        "usable": bool(entry),
    }


def deform_trainer_status(probe_cuda: bool = True) -> dict[str, Any]:
    """Report whether the in-tree 4D trainer can run on this machine.

    The 4D trainer is not an external process -- it lives in
    :mod:`gausscapture.recon.deform` -- but ``doctor`` should ask one module
    "what is the training situation here", so it is reported alongside the
    external one.

    ``cuda`` is ``None`` rather than ``False`` when torch is absent, because
    "no CUDA" and "we could not look" are different answers and a laptop that
    reports the first about a machine it never inspected is lying. Probing it
    costs a torch import (seconds), so it is skippable.
    """
    import importlib.util

    def installed(module: str) -> bool:
        return importlib.util.find_spec(module) is not None

    torch_ok = installed("torch")
    gsplat_ok = installed("gsplat")
    scipy_ok = installed("scipy")

    cuda: bool | None = None
    torch_version = ""
    if torch_ok and probe_cuda:
        with contextlib.suppress(Exception):
            import torch

            torch_version = str(torch.__version__)
            cuda = bool(torch.cuda.is_available())

    # Whether gsplat's own MCMC densifier could be constructed here. Asked only
    # where gsplat is installed, because asking costs the import. The 4D trainer
    # does not use it -- see `schedule.gsplat_strategy_available` for why the
    # plain-PyTorch densifier is the one that runs on CUDA too -- so this is
    # reported rather than acted on, and `doctor` is where a fact about the
    # machine belongs.
    gsplat_mcmc: bool | None = None
    if gsplat_ok and torch_ok:
        with contextlib.suppress(Exception):
            from gausscapture.recon.deform.schedule import gsplat_strategy_available

            gsplat_mcmc = gsplat_strategy_available()

    if not torch_ok:
        note = (
            "torch is not installed, so `train4d` cannot run here. That is the expected "
            "state on an Apple Silicon laptop: package with `gausscapture colab4d` and "
            "train on a cloud GPU."
        )
    elif not gsplat_ok:
        note = (
            "torch is present but gsplat is not, so only the CPU reference rasterizer is "
            "available -- correct, and far too slow for a real scene. `pip install "
            "'gausscapture[train4d]'`."
        )
    elif cuda is False:
        note = (
            "gsplat is installed but no CUDA device is visible; its kernels are CUDA-only, "
            "so training here would fall back to the CPU reference rasterizer."
        )
    elif cuda is None:
        note = "torch and gsplat are installed; CUDA was not probed."
    else:
        note = "torch, gsplat and CUDA are all present; `train4d` can run here."

    return {
        "torch": torch_ok,
        "torch_version": torch_version,
        "gsplat": gsplat_ok,
        # None means "not asked", which on a machine with no gsplat is the only
        # honest answer -- the same distinction `cuda` makes.
        "gsplat_mcmc": gsplat_mcmc,
        "scipy": scipy_ok,
        "cuda": cuda,
        # Runnable at all: the reference rasterizer needs nothing but torch.
        "usable": torch_ok,
        # Runnable at a useful speed on a real scene.
        "gpu_usable": bool(torch_ok and gsplat_ok and cuda),
        "note": note,
    }


class ExternalTrainer:
    """Drives a third-party trainer's ``train.py`` as a subprocess.

    The dataset handed over is always in the conventional layout produced by
    :func:`gausscapture.recon.dataset.build_dataset`, not the project root.
    """

    name = "external"

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def available(self) -> bool:
        return trainer_status(self.settings)["usable"]

    def fit(
        self,
        dataset_dir: Path,
        output_dir: Path,
        preset: str = "draft",
        progress: Progress | None = None,
    ) -> dict[str, Any]:
        progress = progress or NullProgress()
        status = trainer_status(self.settings)

        if not status["configured"]:
            raise DependencyMissingError(
                "No Gaussian trainer configured. Set `gaussian_trainer_path` to a checkout of "
                "gsplat (https://github.com/nerfstudio-project/gsplat), or export a Colab "
                "package instead."
            )
        if not status["exists"]:
            raise DependencyMissingError(f"Trainer path does not exist: {status['path']}")
        if not status["usable"]:
            raise DependencyMissingError(
                f"No train.py or simple_trainer.py found under {status['path']}"
            )
        if not (dataset_dir / "images").exists() or not (dataset_dir / "sparse").exists():
            raise PipelineStateError(
                f"{dataset_dir} is not a trainer-ready dataset. Build it with `build_dataset` first."
            )

        params = TRAINING_PRESETS.get(preset, TRAINING_PRESETS["draft"])
        # The trainer runs from its own checkout, so relative paths would
        # resolve against that directory rather than ours.
        dataset_dir = Path(dataset_dir).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.settings.python_path,
            status["entrypoint"],
            "-s",
            str(dataset_dir),
            "-m",
            str(output_dir),
            "--iterations",
            str(params["iterations"]),
        ]
        if params.get("resolution"):
            cmd += ["--resolution", str(params["resolution"])]

        progress.update(5, f"Starting external trainer ({preset})")
        progress.log("$ " + " ".join(cmd))

        log_path = output_dir.parent / "logs.txt"
        process = subprocess.Popen(
            cmd,
            cwd=str(Path(status["entrypoint"]).parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        with log_path.open("a", encoding="utf-8") as log_file:
            for line in process.stdout:
                stripped = line.rstrip()
                log_file.write(stripped + "\n")
                progress.log(stripped)
                percent = _parse_progress(stripped, params["iterations"])
                if percent is not None:
                    progress.update(percent)

        code = process.wait()
        if code != 0:
            raise RuntimeError(f"Training failed with exit code {code}. See {log_path}")

        models = sorted(
            p for p in output_dir.rglob("*") if p.suffix.lower() in MODEL_SUFFIXES
        )
        if not models:
            raise RuntimeError(
                "Training finished but produced no .ply/.splat/.spz/.sog output. "
                f"Check {log_path}."
            )

        summary = {
            "status": "success",
            "preset": preset,
            "parameters": params,
            "output_dir": str(output_dir),
            "models": [str(p) for p in models],
        }
        (output_dir.parent / "training_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        progress.update(100, "Training complete")
        return summary


def _find_entrypoint(trainer_path: Path) -> Path | None:
    """Locate a trainer script, preferring gsplat's example over Inria's."""
    candidates = [
        trainer_path / "examples" / "simple_trainer.py",
        trainer_path / "simple_trainer.py",
        trainer_path / "train.py",
    ]
    return next((c for c in candidates if c.exists()), None)


def _parse_progress(line: str, total_iterations: int) -> int | None:
    """Best-effort iteration count from a trainer's stdout.

    The previous implementation treated *any* digit on a line as an iteration
    count, so a loss value or a timestamp would drive the progress bar
    backwards. This requires the number to be adjacent to an iteration keyword.
    """
    match = _ITER_PATTERN.search(line)
    if not match:
        return None
    value = int(match.group(1))
    if not 0 < value <= total_iterations:
        return None
    return min(95, int(value / total_iterations * 95))
