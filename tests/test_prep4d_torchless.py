"""``prep4d`` on the machine the architecture is designed for: no torch at all.

Preparation is the CPU half of the pipeline. It runs COLMAP, measures the noise
floor, measures the dynamic region, measures the cone, seeds the scaffold, and
writes one file -- on an M2 Air with no CUDA and therefore no reason to install
torch. ``recon/prepare4d.py`` says so in its module docstring.

It was not true. ``_write_init`` imported ``SceneInit`` from
``recon/deform/bundle.py``, whose line 32 is a module-scope ``import torch``, so
a torch-less machine did every expensive thing in the run and then died at the
write step with a ``ModuleNotFoundError`` -- which ``cli.main`` does not catch,
so it reached the user as a traceback rather than as a sentence.

A test that merely imports ``prepare4d`` would not have caught it: the import
was inside the function. So this runs the whole preparation with ``torch``
blocked at the import system, in a subprocess, and then asserts that torch was
never imported. The subprocess is not squeamishness -- torch is already loaded
in this interpreter by the trainer tests, and the only honest way to test its
absence is an interpreter that never had it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests import fourd_fixtures as fx
from tests.test_pipeline4d import TINY

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Runs inside a fresh interpreter. Everything torch-adjacent must fail loudly.
_SCRIPT = textwrap.dedent(
    """
    import importlib.abc, importlib.machinery, json, sys
    from pathlib import Path


    class NoTorch(importlib.abc.MetaPathFinder):
        \"\"\"Make this interpreter one on which torch is simply not installed.\"\"\"

        def find_spec(self, fullname, path=None, target=None):
            if fullname == "torch" or fullname.startswith("torch."):
                raise ModuleNotFoundError("No module named " + repr(fullname))
            return None


    sys.meta_path.insert(0, NoTorch())
    assert "torch" not in sys.modules

    sys.path.insert(0, __REPO__)
    from tests import fourd_fixtures as fx
    from gausscapture.recon.prepare4d import prepare_4d
    from gausscapture.types import PoseReport

    project = Path(__PROJECT__)
    cloud = __import__("numpy").random.default_rng(99).uniform(-0.6, 0.6, (200, 3))


    def solve(workspace, mask_path, progress):
        images = Path(workspace) / "frames" / "images"
        names = sorted(p.name for p in images.glob("*.jpg"))
        arc = [n for n in names if n.startswith("a_")]
        poses = dict(zip(arc, fx.arc_poses(len(arc)), strict=True))
        for name in names:
            if not name.startswith("a_"):
                poses[name] = fx.fixed_pose()
        model_dir = Path(workspace) / "sparse" / "0"
        fx.write_colmap_model(model_dir, poses, cloud)
        return PoseReport(method="stub", images_total=len(names),
                          images_registered=len(poses), registered_ratio=1.0,
                          sparse_points=len(cloud), status="ok", model_dir=str(model_dir))


    result = prepare_4d(project, nodes=8, solver=solve)
    assert "torch" not in sys.modules, sorted(m for m in sys.modules if "torch" in m)
    print(json.dumps({"init": result["init_path"], "nodes": result["nodes"],
                       "gaussians": result["gaussians"]}))
    """
)


@pytest.fixture()
def capture_pack(store, tmp_path: Path) -> Path:
    project = store.create("torchless", "person")
    video = fx.make_phase_video(tmp_path / "capture.mp4", **TINY)
    fx.write_pack(project.path, video)
    return project.path


def test_preparation_completes_with_torch_blocked_from_the_import_system(capture_pack: Path):
    script = _SCRIPT.replace("__REPO__", repr(str(REPO_ROOT))).replace(
        "__PROJECT__", repr(str(capture_pack))
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert Path(payload["init"]).exists()
    assert payload["nodes"] == 8
    assert payload["gaussians"] == 200


def test_the_initialisation_schema_module_imports_without_torch():
    """The schema is numpy and json; nothing about it needs a tensor library."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, 'src');"
            " import gausscapture.recon.deform.init4d as m;"
            " assert 'torch' not in sys.modules;"
            " print(m.SceneInit.__name__, m.INIT_NAME)",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "SceneInit scene4d_init.npz"
