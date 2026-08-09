"""The 3D half of the CLI: `train`, `export`, `viewer`. Executed, not assumed.

``gausscapture viewer``, ``train``, ``export`` and ``report`` all ship handlers
whose engines nothing ever ran. ``report/splat_site.py`` was 0 % covered,
``export/splat_binary.py`` 0 %, and ``ExternalTrainer.fit`` -- the entire body of
the ``train`` command, including its progress parser and its "produced no model"
refusal -- 0 %. None of that was broken; all of it was unguarded, which is a
different problem with the same eventual cost.

The two things worth pinning are the ones a viewer cannot recover from: the
packed file is exactly 32 bytes per gaussian (a stride off by one is a scene of
noise), and the draw order is back to front (getting it backwards paints the
background over the subject and reads as haze).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest

from gausscapture.config import Settings
from gausscapture.errors import DependencyMissingError, PipelineStateError
from gausscapture.export.splat_binary import STRIDE, write_splat_binary
from gausscapture.recon.external import ExternalTrainer, _parse_progress
from gausscapture.report.splat_site import build_splat_site
from tests.test_splat_ply import one, write_splat_ply

SPLAT_VIEWER = (
    Path(__file__).resolve().parents[1]
    / "src" / "gausscapture" / "report" / "splat_viewer.js"
)


def a_splat(path: Path, count: int = 200, seed: int = 7) -> Path:
    """A trained-looking PLY, written by the reader's own test helper."""
    rng = np.random.default_rng(seed)
    gaussians = []
    for _ in range(count):
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        gaussians.append(
            one(
                xyz=tuple(rng.uniform(-1.0, 1.0, 3)),
                dc=tuple(rng.uniform(-1.0, 1.0, 3)),
                # Comfortably above the 0.02 visibility floor.
                opacity=float(rng.uniform(1.0, 4.0)),
                log_scales=tuple(rng.uniform(-4.0, -2.0, 3)),
                quat=tuple(quat),
            )
        )
    return write_splat_ply(path, gaussians)


class TestThePackedFormat:
    """32 bytes per gaussian, and the numbers a viewer frames the scene with."""

    def test_the_file_is_exactly_thirty_two_bytes_per_gaussian(self, tmp_path: Path):
        packed = write_splat_binary(a_splat(tmp_path / "s.ply"), tmp_path / "s.splat")

        assert packed["count"] == 200
        assert packed["bytes"] == 200 * STRIDE == 6_400
        assert (tmp_path / "s.splat").stat().st_size == packed["bytes"]

    def test_a_cap_keeps_the_most_important_gaussians_rather_than_a_prefix(
        self, tmp_path: Path
    ):
        """Truncation has to degrade gracefully; that is why the sort exists."""
        packed = write_splat_binary(
            a_splat(tmp_path / "s.ply"), tmp_path / "s.splat", max_gaussians=50
        )
        assert packed["count"] == 50
        assert packed["bytes"] == 50 * STRIDE

        data = np.frombuffer((tmp_path / "s.splat").read_bytes(), dtype=np.uint8)
        scales = data.reshape(50, STRIDE)[:, 12:24].copy().view(np.float32).reshape(50, 3)
        opacity = data.reshape(50, STRIDE)[:, 27].astype(np.float64) / 255.0
        importance = np.prod(scales, axis=1) * opacity
        assert np.all(np.diff(importance) <= 1e-12), "gaussians are not importance-ordered"

    def test_an_opacity_floor_drops_gaussians_before_they_are_packed(self, tmp_path: Path):
        rng = np.random.default_rng(3)
        gaussians = []
        for i in range(20):
            quat = rng.normal(size=4)
            quat /= np.linalg.norm(quat)
            gaussians.append(
                one(opacity=-6.0 if i % 2 else 4.0, log_scales=(-3.0,) * 3, quat=tuple(quat))
            )
        write_splat_ply(tmp_path / "s.ply", gaussians)

        packed = write_splat_binary(tmp_path / "s.ply", tmp_path / "s.splat", min_opacity=0.1)
        assert packed["count"] == 10


class TestTheSplatSite:
    """A directory you can open with a file:// URL and nothing else."""

    @pytest.fixture()
    def site(self, tmp_path: Path):
        plys = [a_splat(tmp_path / "A_good_7000.ply", count=120, seed=1),
                a_splat(tmp_path / "A_good_30000.ply", count=200, seed=2)]
        out = tmp_path / "site"
        index = build_splat_site(plys, out, title="Test scene")
        scenes = json.loads((out / "scenes.json").read_text(encoding="utf-8"))
        return out, index, scenes

    def test_it_writes_a_page_a_viewer_and_one_binary_per_splat(self, site):
        out, index, scenes = site

        assert index == out / "index.html"
        assert (out / "splat_viewer.js").exists()
        assert (out / "scenes.json").exists()
        assert len(scenes) == 2
        for scene in scenes:
            payload = out / scene["file"]
            assert payload.exists()
            assert payload.stat().st_size == scene["count"] * STRIDE

    def test_every_scene_carries_what_the_camera_needs_to_frame_it(self, site):
        """Without a centre and a radius the viewer opens looking at nothing."""
        _out, _index, scenes = site
        for scene in scenes:
            assert scene["count"] > 0
            assert len(scene["centre"]) == 3
            assert all(np.isfinite(scene["centre"]))
            assert scene["radius"] > 0.0
            assert scene["label"]
            assert scene["source"].endswith(".ply")

    def test_the_label_reads_as_a_person_would_say_it(self, site):
        _out, _index, scenes = site
        labels = {scene["label"] for scene in scenes}
        assert "A good · 7,000 steps" in labels
        assert "A good · 30,000 steps" in labels

    def test_the_page_fetches_nothing_from_the_network(self, site):
        out, index, _scenes = site
        html = index.read_text(encoding="utf-8")
        for attribute in ('src="http', "src='http", 'href="http', "href='http"):
            assert attribute not in html
        assert "cdn." not in html and "@import" not in html
        assert "./splat_viewer.js" in html
        # The viewer is copied, not linked, so the directory is movable.
        assert (out / "splat_viewer.js").read_bytes() == SPLAT_VIEWER.read_bytes()

    def test_a_missing_splat_is_skipped_and_an_empty_set_is_refused(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="No usable splats"):
            build_splat_site([tmp_path / "nope.ply"], tmp_path / "empty")


#: Drives the shipped static viewer's sort worker under Node. Same reasoning as
#: the 4D one in ``test_viewer_contract``: the direction of the draw order is
#: invisible in every count and catastrophic in the image.
_SORT_DRIVER = """
import fs from 'node:fs';
import { WORKER_SOURCE, lookAt } from './splat_viewer.mjs';

const input = JSON.parse(fs.readFileSync('./sort_input.json', 'utf8'));
let result = null;
const shim = { onmessage: null, postMessage: (message) => { result = message; } };
new Function('self', WORKER_SOURCE)(shim);

const view = lookAt(input.eye, input.target, input.up);
shim.onmessage({ data: { positions: Float32Array.from(input.positions) } });
shim.onmessage({ data: { view } });

const distance = [];
for (let i = 0; i < input.positions.length / 3; i++) {
  const p = input.positions.slice(i * 3, i * 3 + 3);
  distance.push(-(view[2] * p[0] + view[6] * p[1] + view[10] * p[2] + view[14]));
}
fs.writeFileSync('./sort_output.json', JSON.stringify({ order: [...result.order], distance }));
"""


class TestTheStaticViewerDrawsBackToFront:
    """``splat_viewer.js`` carries the same sort idiom as the 4D viewer.

    It had the same inversion, for the same reason: the key comes from row 2 of
    the VIEW matrix, which ``lookAt`` writes as ``-forward``, so it decreases
    with distance -- and the counting sort emitted in reverse, putting the
    nearest gaussian first. With ``blendFuncSeparate(ONE, ONE_MINUS_SRC_ALPHA)``
    over a premultiplied fragment, first-drawn must be farthest.
    """

    def test_the_first_gaussian_drawn_is_the_farthest_one(self, tmp_path: Path):
        node = shutil.which("node")
        if node is None:
            pytest.skip("node is not installed; the browser half cannot be executed")

        (tmp_path / "splat_viewer.mjs").write_bytes(SPLAT_VIEWER.read_bytes())
        (tmp_path / "driver.mjs").write_text(_SORT_DRIVER, encoding="utf-8")
        (tmp_path / "sort_input.json").write_text(
            json.dumps({
                "eye": [0.0, 0.0, 0.0],
                "target": [0.0, 0.0, 1.0],
                "up": [0.0, 1.0, 0.0],
                "positions": [0.0, 0.0, 5.0, 0.0, 0.0, 1.0, 0.0, 0.0, 12.0],
            }),
            encoding="utf-8",
        )
        result = subprocess.run(
            [node, "driver.mjs"], cwd=tmp_path, capture_output=True, text=True,
            timeout=120, check=False,
        )
        assert result.returncode == 0, result.stderr
        out = json.loads((tmp_path / "sort_output.json").read_text(encoding="utf-8"))

        drawn = [out["distance"][i] for i in out["order"]]
        assert drawn == sorted(drawn, reverse=True), drawn
        assert out["order"][0] == 2


def fake_trainer(root: Path, script: str) -> Path:
    """A checkout with a ``train.py`` that behaves like a real trainer's."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "train.py").write_text(textwrap.dedent(script), encoding="utf-8")
    return root


def a_dataset(root: Path) -> Path:
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    return root


class TestTheExternalTrainer:
    """`gausscapture train` is a subprocess driver, and its body was never run."""

    def _settings(self, trainer: Path) -> Settings:
        import sys

        settings = Settings()
        settings.gaussian_trainer_path = str(trainer)
        settings.python_path = sys.executable
        return settings

    def test_it_runs_the_trainer_collects_the_model_and_writes_a_summary(
        self, tmp_path: Path
    ):
        trainer = fake_trainer(
            tmp_path / "gsplat",
            """
            import sys
            from pathlib import Path

            out = Path(sys.argv[sys.argv.index('-m') + 1])
            print('Iteration 500/5000')
            print('loss 0.0123 at 1700000000')
            print('Iteration 5000/5000')
            (out / 'point_cloud').mkdir(parents=True, exist_ok=True)
            (out / 'point_cloud' / 'scene.ply').write_bytes(b'ply\\n')
            """,
        )
        dataset = a_dataset(tmp_path / "dataset")
        output = tmp_path / "run" / "model"

        summary = ExternalTrainer(self._settings(trainer)).fit(dataset, output)

        assert summary["status"] == "success"
        assert summary["preset"] == "draft"
        assert summary["models"] == [str(output / "point_cloud" / "scene.ply")]
        written = json.loads(
            (output.parent / "training_summary.json").read_text(encoding="utf-8")
        )
        assert written == summary
        assert "Iteration 500/5000" in (output.parent / "logs.txt").read_text(encoding="utf-8")

    def test_a_run_that_produced_no_model_is_a_failure_and_says_where_to_look(
        self, tmp_path: Path
    ):
        trainer = fake_trainer(
            tmp_path / "gsplat", "print('Iteration 5000/5000')\nprint('done')\n"
        )
        dataset = a_dataset(tmp_path / "dataset")

        with pytest.raises(RuntimeError, match="produced no"):
            ExternalTrainer(self._settings(trainer)).fit(dataset, tmp_path / "run" / "model")
        assert (tmp_path / "run" / "logs.txt").exists()

    def test_a_trainer_that_exits_non_zero_names_its_log(self, tmp_path: Path):
        trainer = fake_trainer(tmp_path / "gsplat", "import sys\nsys.exit(3)\n")
        dataset = a_dataset(tmp_path / "dataset")

        with pytest.raises(RuntimeError, match="exit code 3"):
            ExternalTrainer(self._settings(trainer)).fit(dataset, tmp_path / "run" / "model")

    def test_an_unconfigured_trainer_is_named_rather_than_attempted(self, tmp_path: Path):
        settings = Settings()
        settings.gaussian_trainer_path = ""
        with pytest.raises(DependencyMissingError, match="No Gaussian trainer configured"):
            ExternalTrainer(settings).fit(a_dataset(tmp_path / "d"), tmp_path / "out")

    def test_a_directory_that_is_not_a_dataset_is_named(self, tmp_path: Path):
        trainer = fake_trainer(tmp_path / "gsplat", "pass\n")
        with pytest.raises(PipelineStateError, match="not a trainer-ready dataset"):
            ExternalTrainer(self._settings(trainer)).fit(tmp_path / "empty", tmp_path / "out")

    def test_a_checkout_with_no_entrypoint_is_named(self, tmp_path: Path):
        (tmp_path / "gsplat").mkdir()
        settings = Settings()
        settings.gaussian_trainer_path = str(tmp_path / "gsplat")
        with pytest.raises(DependencyMissingError, match="No train.py"):
            ExternalTrainer(settings).fit(a_dataset(tmp_path / "d"), tmp_path / "out")


class TestTheProgressParser:
    """A loss value is not an iteration count, and neither is a unix timestamp."""

    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            ("Iteration 500/5000", 9),
            ("iter 2500", 47),
            ("step: 5000", 95),
        ],
    )
    def test_it_reads_an_iteration_count(self, line: str, expected: int):
        assert _parse_progress(line, 5_000) == expected

    @pytest.mark.parametrize(
        "line",
        [
            "loss 0.0123",
            "elapsed 1700000000",
            "Iteration 999999/5000",
            "nothing numeric here",
            # No iteration keyword adjacent to the number. Deliberate: a missed
            # match costs a progress bar, a false match drives it backwards.
            "Training [ 1000/5000 ]",
        ],
    )
    def test_it_declines_anything_that_is_not_one(self, line: str):
        assert _parse_progress(line, 5_000) is None
