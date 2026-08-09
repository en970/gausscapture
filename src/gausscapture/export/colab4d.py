"""Package a prepared 4D scene for a cloud GPU, and collect the result back.

The laptop this project is developed on is an Apple Silicon M2 Air, so the one
step of the 4D path that cannot run locally is the one that needs CUDA kernels.
Everything either side of it -- phase splitting, the dynamic mask, COLMAP, the
fixed pose, the measured cone, the canonical initialisation, the `.g4d` export
and the viewer -- runs here. This module is the seam: it hands a GPU host a
single file that contains everything it needs and nothing about this repository
beyond the `train4d` extra, and it takes back a single file that the local
export path can ingest.

That round trip is why ``--collect`` lives here rather than in the exporter.
A ``scene4d.npz`` downloaded from Drive into the wrong directory is the most
likely way for a training run to be lost after it succeeded, so putting it in
the project is a command rather than an instruction.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from gausscapture.errors import CaptureFormatError, PipelineStateError
from gausscapture.progress import NullProgress, Progress
from gausscapture.project import (
    DATASET4D_RELPATH,
    SCENE4D_RELPATH,
    FourDState,
    record_fourd_stage,
)

COLAB4D_README = """# GaussCapture 4D package

`dataset4d.zip` is the complete input to the deformable trainer:

    scene4d_init.npz    canonical gaussians, scaffold seed, skinning,
                        the single fixed camera pose and the measured cone
    images/             the hold-phase frames, one per timestep
    masks/              the dynamic region measured from the stationary camera
    gausscapture/       manifest, 4D notes and the run configuration

Train it with `notebooks/GaussCapture_Colab_4D.ipynb`, which installs a
permissive stack (PyTorch plus gsplat, Apache-2.0) and runs *this project's*
trainer -- `gausscapture train4d` -- not a third-party one. No component of
the Inria/MPII non-commercial licence is involved at any point.

The run produces `scene4d.npz`. Bring it home with

    gausscapture colab4d <project> --collect scene4d.npz

then `export4d` and `viewer4d` on your own machine.

A note on what this can be: the camera is stationary for the whole of the
`hold` phase, so there is no parallax during the motion. The trained scene is
honest only inside the cone the hand-held arc actually swept, which is measured
and written into the file. The viewer clamps to it.
"""

#: Where the hold-phase frames may live, most specific first. Preparation
#: writes ``training/frames4d``; a project prepared by an earlier revision, or
#: by hand, may only have the general frame directory. Guessing between them is
#: better than failing on a layout difference that costs a Colab session.
_IMAGE_SOURCES = (
    Path("training") / "frames4d",
    Path("frames") / "hold",
    Path("frames") / "images",
)

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def create_colab4d_package(
    project_path: Path,
    out: Path | None = None,
    training_config: dict[str, Any] | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Zip a prepared 4D scene for upload to Colab or a rented GPU.

    This module owns the layout. The reader is
    :func:`gausscapture.recon.deform.init4d.unpack_dataset` together with
    ``bundle.load_clip``, which look for ``scene4d_init.npz``, ``images/`` and
    ``masks/dynamic.png`` by exactly those names -- so those names, not this
    function, are the contract. The returned ``packer`` field records who wrote
    the archive, because a zip whose contents are ambiguous is worse than one
    that names its author.
    """
    progress = progress or NullProgress()
    project_path = Path(project_path)

    state = FourDState.observe(project_path)
    state.require(
        "prepared",
        "Prepare the scene first: `gausscapture prep4d <project>`.",
    )

    dataset_zip = Path(out) if out else project_path / DATASET4D_RELPATH
    dataset_zip.parent.mkdir(parents=True, exist_ok=True)
    if dataset_zip.exists():
        dataset_zip.unlink()

    config_path = dataset_zip.parent / "run_config_4d.json"
    config_path.write_text(json.dumps(training_config or {}, indent=2), encoding="utf-8")
    readme_path = dataset_zip.parent / "README_COLAB_4D.md"
    readme_path.write_text(COLAB4D_README, encoding="utf-8")

    members = _members(project_path, state, config_path, readme_path)
    packer_name = "colab4d"
    with zipfile.ZipFile(dataset_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, (source, arcname) in enumerate(members):
            archive.write(source, arcname)
            if index % 25 == 0:
                progress.update(
                    5 + int(85 * ((index + 1) / len(members))),
                    f"Packed {index + 1}/{len(members)} files",
                )

    # Written by either packer, so the human-facing context is present whichever
    # one ran. Appending is cheap: zip is a trailing-directory format.
    with zipfile.ZipFile(dataset_zip, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        present = set(archive.namelist())
        for extra, arcname in (
            (readme_path, "README_COLAB_4D.md"),
            (config_path, "gausscapture/run_config_4d.json"),
        ):
            if arcname not in present:
                archive.write(extra, arcname)
        names = archive.namelist()

    images = sum(1 for name in names if name.startswith("images/") and not name.endswith("/"))
    result = {
        "dataset_zip": str(dataset_zip),
        "size_bytes": dataset_zip.stat().st_size,
        "file_count": len(names),
        "images": images,
        "packer": packer_name,
        "readme": str(readme_path),
        "run_config": str(config_path),
    }
    record_fourd_stage(
        project_path,
        "packaged",
        dataset4d_zip=str(dataset_zip),
        dataset4d_bytes=result["size_bytes"],
        dataset4d_packer=packer_name,
    )
    progress.update(100, "4D package ready")
    return result


def collect_trained_scene(project_path: Path, scene_npz: Path) -> dict[str, Any]:
    """Adopt a ``scene4d.npz`` trained elsewhere into the project.

    Validated before it is copied, because the failure this guards against is
    a browser download that produced an HTML error page with a ``.npz`` name --
    which then sits in the project looking exactly like a successful run.
    """
    project_path = Path(project_path)
    scene_npz = Path(scene_npz).expanduser()
    if not scene_npz.exists():
        raise PipelineStateError(f"No such trained scene: {scene_npz}")

    keys = _npz_keys(scene_npz)
    if not keys:
        raise CaptureFormatError(
            f"{scene_npz.name} is not a readable .npz archive. If it came from a browser "
            "download, check that it is not an error page saved under that name."
        )

    destination = project_path / SCENE4D_RELPATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    if scene_npz.resolve() != destination.resolve():
        shutil.copy2(scene_npz, destination)

    record_fourd_stage(
        project_path,
        "trained",
        scene4d=str(destination),
        scene4d_bytes=destination.stat().st_size,
        scene4d_arrays=sorted(keys),
    )
    return {
        "scene": str(destination),
        "size_bytes": destination.stat().st_size,
        "arrays": sorted(keys),
    }


# --------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------


def _members(
    project_path: Path,
    state: FourDState,
    config_path: Path,
    readme_path: Path,
) -> list[tuple[Path, str]]:
    members: list[tuple[Path, str]] = [(state.init_path, "scene4d_init.npz")]

    image_dir = next(
        (project_path / rel for rel in _IMAGE_SOURCES if (project_path / rel).is_dir()), None
    )
    images = (
        sorted(p for p in image_dir.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
        if image_dir
        else []
    )
    if not images and not _init_carries_images(state.init_path):
        raise PipelineStateError(
            "Nothing to train on: no hold-phase frames were found under "
            f"{', '.join(str(rel) for rel in _IMAGE_SOURCES)}, and scene4d_init.npz does not "
            "carry them either. Re-run `gausscapture prep4d <project>`."
        )
    members += [(p, f"images/{p.name}") for p in images]

    if state.masks_dir.is_dir():
        members += [(p, f"masks/{p.name}") for p in sorted(state.masks_dir.glob("*.png"))]

    for extra, arcname in (
        (project_path / "capturepack" / "manifest.json", "gausscapture/manifest.json"),
        (project_path / "training" / "fourd.json", "gausscapture/fourd.json"),
        (config_path, "gausscapture/run_config_4d.json"),
        (readme_path, "README_COLAB_4D.md"),
    ):
        if extra.exists():
            members.append((extra, arcname))
    return members


def _npz_keys(path: Path) -> list[str]:
    import numpy as np

    try:
        with np.load(path, allow_pickle=False) as data:
            return list(data.files)
    except (OSError, ValueError, zipfile.BadZipFile):
        return []


def _init_carries_images(init_path: Path) -> bool:
    """Whether the init file already holds the pixels the trainer fits.

    ``np.load`` on an ``.npz`` reads the archive directory only, so this costs
    a few kilobytes rather than the whole file.
    """
    return any(key.startswith(("images", "frames")) for key in _npz_keys(init_path))
