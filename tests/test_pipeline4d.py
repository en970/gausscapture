"""The fixed-camera 4D path, driven end to end on synthesised input.

``test_cli_4d`` covers flag parsing and the state guards; this module covers the
work. Everything it needs is generated in-process: a four-phase video whose
phase boundaries and moving region are known to the frame, a CapturePack with
the ``protocol`` block a capturepack 0.3 app would write, and a stand-in solver
that places cameras on an arc of a known angle instead of running COLMAP.

The solver is a stub for one reason only: COLMAP is a several-hundred-megabyte
binary that is not on a CI runner and takes minutes on footage this synthetic.
Everything it feeds -- the fixed pose derivation, the cone measurement, the
dynamic labelling, the scaffold seeding, the container and the page -- is the
real code, reading a real COLMAP text model off a real disk.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest

from gausscapture.cli import main
from gausscapture.errors import CaptureFormatError, PipelineStateError
from gausscapture.export.checkpoint4d import load_trained_scene
from gausscapture.export.scene4d import read_scene4d, read_scene4d_header
from gausscapture.project import SCENE4D_G4D_RELPATH, FourDState, ProjectStore
from gausscapture.recon.prepare4d import (
    label_dynamic_points,
    prepare_4d,
    read_init_meta,
    seed_scaffold,
)
from gausscapture.types import PoseReport
from tests import fourd_fixtures as fx

#: A capture small enough that the whole chain, including four steps of CPU
#: training through the reference rasterizer, runs in a couple of seconds.
TINY = {"size": (120, 90), "fps": 20, "perch": 12, "arc": 24, "reseat": 12, "hold": 24,
        "blob": 24}

#: The half-angle the stub solver places its arc cameras on. The cone the
#: pipeline reports is measured back off those cameras, so this is the ground
#: truth for the one number the viewer clamps to.
ARC_HALF_DEG = 26.0
SPARSE_POINTS = 200


def make_stub_solver(points: np.ndarray | None = None, register_hold: bool = True):
    """A solver that writes the COLMAP model the fixture's geometry implies.

    Frame names carry their phase as a prefix (``a_`` for the arc), which is how
    every later stage recovers which phase a solved camera came from -- so the
    stub can honour that contract exactly by reading the same prefixes.
    """
    cloud = _default_points() if points is None else np.asarray(points, dtype=np.float64)

    def solve(workspace: Path, mask_path: Path, progress) -> PoseReport:
        images_dir = Path(workspace) / "frames" / "images"
        names = sorted(p.name for p in images_dir.glob("*.jpg"))
        arc = [n for n in names if n.startswith("a_")]
        still = [n for n in names if not n.startswith("a_")]
        if not register_hold:
            still = [n for n in still if not n.startswith("h_")]

        poses = dict(zip(arc, fx.arc_poses(len(arc), azimuth_half_deg=ARC_HALF_DEG),
                         strict=True))
        for name in still:
            poses[name] = fx.fixed_pose()

        model_dir = Path(workspace) / "sparse" / "0"
        fx.write_colmap_model(model_dir, poses, cloud)
        return PoseReport(
            method="stub-solver",
            images_total=len(names),
            images_registered=len(poses),
            registered_ratio=len(poses) / max(len(names), 1),
            sparse_points=len(cloud),
            status="ok",
            model_dir=str(model_dir),
        )

    return solve


def _default_points(seed: int = 99) -> np.ndarray:
    """A cloud that straddles the moving region: some on it, some behind it."""
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.uniform(-0.9, 0.9, SPARSE_POINTS),
        rng.uniform(-0.7, 0.7, SPARSE_POINTS),
        rng.uniform(-0.4, 0.4, SPARSE_POINTS),
    ])


@pytest.fixture()
def capture(store: ProjectStore, tmp_path: Path):
    """A project holding a synthetic four-phase CapturePack."""
    project = store.create("bullet time", "person")
    video = fx.make_phase_video(tmp_path / "capture.mp4", **TINY)
    fx.write_pack(project.path, video)
    return project, video


@pytest.fixture()
def prepared(capture):
    """The same project, taken as far as ``prep4d`` takes it."""
    project, video = capture
    result = prepare_4d(project.path, nodes=12, solver=make_stub_solver())
    return project, video, result


# --------------------------------------------------------------------------
# preparation
# --------------------------------------------------------------------------


class TestPrepare4D:
    def test_it_writes_an_initialisation_the_trainer_can_open(self, prepared):
        project, _video, result = prepared
        from gausscapture.recon.deform.init4d import SceneInit

        init = SceneInit.load(Path(result["init_path"]))
        assert init.points.shape == (SPARSE_POINTS, 3)
        assert init.point_colors.shape == (SPARSE_POINTS, 3)
        assert init.point_colors.min() >= 0.0 and init.point_colors.max() <= 1.0
        assert init.point_dynamic.shape == (SPARSE_POINTS,)
        assert init.node_rest.shape == (result["nodes"], 3)
        assert init.viewmat.shape == (4, 4) and init.intrinsics.shape == (3, 3)
        # The frame times are the clip's parameterisation; the field is indexed
        # on exactly [0, 1] and would extrapolate off either end.
        assert init.frame_times[0] == pytest.approx(0.0)
        assert init.frame_times[-1] == pytest.approx(1.0)
        assert np.all(np.diff(init.frame_times) > 0)
        assert FourDState.observe(project.path).stage == "prepared"

    def test_the_camera_it_writes_is_the_one_the_solver_placed(self, prepared):
        _project, _video, result = prepared
        from gausscapture.recon.deform.init4d import SceneInit

        init = SceneInit.load(Path(result["init_path"]))
        qvec, tvec = fx.fixed_pose()
        from gausscapture.pose.model import quaternion_to_rotation

        assert init.viewmat[:3, :3] == pytest.approx(quaternion_to_rotation(qvec), abs=1e-5)
        assert init.viewmat[:3, 3] == pytest.approx(tvec, abs=1e-5)
        assert init.viewmat[3] == pytest.approx([0, 0, 0, 1])

    def test_the_cone_is_measured_back_off_the_arc_it_was_given(self, prepared):
        _project, _video, result = prepared
        cone = result["cone"]
        # The stub swept +/-26 degrees about the origin, and the cone is
        # measured about the *subject* centroid, which sits a little off it --
        # so the answer is a few degrees narrower. Narrower is the only
        # direction that is allowed to be wrong: a cone wider than the sweep
        # would authorise the viewer to render angles nobody recorded.
        assert cone["azimuth_half_deg"] <= ARC_HALF_DEG + 1e-6
        assert cone["azimuth_half_deg"] == pytest.approx(ARC_HALF_DEG, abs=4.0)
        # A horizontal sweep contributes almost no vertical baseline. This is
        # geometry, not a defect, and the pipeline must not flatter it.
        assert cone["elevation_half_deg"] < cone["azimuth_half_deg"] / 3.0
        assert cone["centroid_source"] == "dynamic_mask"
        assert cone["source"] == "measured"

    def test_the_fixed_pose_is_verified_when_the_hold_keyframes_register(self, prepared):
        _project, _video, result = prepared
        fixed = result["fixed_pose"]
        assert fixed["source"] == "verified"
        assert fixed["verified"] is True
        assert fixed["rest_cameras"] > 0 and fixed["hold_cameras"] > 0

    def test_an_unregistered_hold_leaves_the_pose_asserted_and_says_so(self, capture):
        project, _video = capture
        result = prepare_4d(
            project.path, nodes=8, solver=make_stub_solver(register_hold=False)
        )
        fixed = result["fixed_pose"]
        assert fixed["source"] == "asserted"
        assert fixed["verified"] is False
        assert any("never confirmed" in w for w in fixed["warnings"])

    def test_the_provenance_travels_in_the_init_rather_than_being_re_derived(self, prepared):
        _project, _video, result = prepared
        meta = read_init_meta(Path(result["init_path"]))
        assert meta["cone"]["azimuth_half_deg"] == pytest.approx(
            result["cone"]["azimuth_half_deg"]
        )
        assert meta["capture"]["verified"] is True
        assert meta["fps"] == pytest.approx(TINY["fps"])
        assert meta["skin_k"] == 4
        # The capture frame is what the viewer orbits in; COLMAP's y points down
        # the image, so "up" has to be its negation or the page renders upside
        # down and nothing else in the pipeline would notice.
        assert np.dot(meta["capture"]["up"], [0.0, 1.0, 0.0]) > 0.9
        assert np.dot(meta["capture"]["forward"], [0.0, 0.0, 1.0]) > 0.9

    def test_the_hold_frames_are_staged_where_the_packager_looks(self, prepared):
        project, _video, result = prepared
        staged = sorted((project.path / "training" / "frames4d").glob("*.jpg"))
        assert len(staged) == result["frames"] == result["images"]
        assert (project.path / "masks" / "dynamic.png").exists()


class TestDynamicLabelling:
    """Which Gaussians may move is a measurement, and this is the measurement."""

    def _scene(self):
        return {
            "fixed_pose": {"qvec": [1.0, 0.0, 0.0, 0.0], "tvec": [0.0, 0.0, 2.0]},
            "camera": {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 40.0},
        }

    def test_only_points_landing_inside_the_mask_are_dynamic(self):
        mask = np.zeros((80, 100), dtype=bool)
        mask[30:50, 40:60] = True
        # Identity rotation, camera two units back: u = 100 * x / (z + 2) + 50.
        inside = np.array([0.0, 0.0, 0.0])            # -> (50, 40), inside
        outside = np.array([0.6, 0.0, 0.0])           # -> (80, 40), outside
        behind = np.array([0.0, 0.0, -5.0])           # behind the camera
        points = np.stack([inside, outside, behind])
        assert list(label_dynamic_points(points, self._scene(), mask)) == [True, False, False]

    def test_an_empty_mask_labels_nothing_dynamic(self):
        points = np.zeros((5, 3))
        flags = label_dynamic_points(points, self._scene(), np.zeros((80, 100), dtype=bool))
        assert not flags.any()

    def test_a_scene_without_intrinsics_is_refused_by_name(self):
        with pytest.raises(CaptureFormatError, match="fixed pose or the intrinsics"):
            label_dynamic_points(np.zeros((2, 3)), {"fixed_pose": {}}, np.zeros((4, 4), bool))


class TestScaffoldSeeding:
    def test_it_is_deterministic_and_covers_the_cloud(self):
        rng = np.random.default_rng(3)
        points = rng.normal(size=(500, 3))
        first = seed_scaffold(points, 32)
        second = seed_scaffold(points, 32)
        assert first.shape == (32, 3)
        assert np.array_equal(first, second)
        # Farthest-point sampling exists to cover; a random subset of the same
        # size would leave a much larger worst-case gap.
        gap = np.linalg.norm(points[:, None, :] - first[None, :, :], axis=2).min(axis=1).max()
        random_subset = points[rng.choice(len(points), 32, replace=False)]
        random_gap = np.linalg.norm(
            points[:, None, :] - random_subset[None, :, :], axis=2
        ).min(axis=1).max()
        assert gap < random_gap

    def test_it_never_asks_for_more_nodes_than_there_are_points(self):
        points = np.random.default_rng(1).normal(size=(11, 3))
        assert seed_scaffold(points, 256).shape == (11, 3)

    def test_a_subject_that_never_moved_is_refused_before_a_gpu_is_rented(self):
        with pytest.raises(PipelineStateError, match="no 4D scene in this take"):
            seed_scaffold(np.zeros((3, 3)), 16)


class TestPreparationRefusals:
    """A take that cannot become a 4D scene is refused by name, not by traceback."""

    def _pack_without(self, project_path: Path, video, dropped: str) -> None:
        block = video.protocol_block()
        block["phases"] = [p for p in block["phases"] if p["name"] != dropped]
        fx.write_pack(project_path, video, declare_protocol=False)
        manifest_path = project_path / "capturepack" / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["protocol"] = block
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    def test_a_capture_that_declares_no_hold_phase(self, store: ProjectStore, tmp_path: Path):
        project = store.create("no hold", "person")
        video = fx.make_phase_video(tmp_path / "c.mp4", **TINY)
        self._pack_without(project.path, video, "hold")
        with pytest.raises(CaptureFormatError, match="hold"):
            prepare_4d(project.path, solver=make_stub_solver(), allow_inference=False)

    def test_a_capture_that_declares_no_arc_phase(self, store: ProjectStore, tmp_path: Path):
        project = store.create("no arc", "person")
        video = fx.make_phase_video(tmp_path / "c.mp4", **TINY)
        self._pack_without(project.path, video, "arc")
        with pytest.raises(CaptureFormatError, match="zero parallax"):
            prepare_4d(project.path, solver=make_stub_solver(), allow_inference=False)

    def test_a_pack_with_no_protocol_block_is_measured_rather_than_refused(
        self, store: ProjectStore, tmp_path: Path
    ):
        """Footage from any other camera app still works; it just says so."""
        project = store.create("no protocol", "person")
        video = fx.make_phase_video(tmp_path / "c.mp4", **TINY)
        fx.write_pack(project.path, video, declare_protocol=False)
        result = prepare_4d(project.path, nodes=8, solver=make_stub_solver())
        assert result["split"]["source"] == "inferred"
        assert any("inferred from image motion" in w for w in result["split"]["warnings"])

    def test_a_pack_with_no_protocol_block_when_inference_is_off(
        self, store: ProjectStore, tmp_path: Path
    ):
        project = store.create("strict", "person")
        video = fx.make_phase_video(tmp_path / "c.mp4", **TINY)
        fx.write_pack(project.path, video, declare_protocol=False)
        with pytest.raises(CaptureFormatError, match="declares no `protocol` block"):
            prepare_4d(project.path, solver=make_stub_solver(), allow_inference=False)

    def test_a_solver_that_produced_no_model(self, store: ProjectStore, tmp_path: Path):
        project = store.create("no model", "person")
        video = fx.make_phase_video(tmp_path / "c.mp4", **TINY)
        fx.write_pack(project.path, video)

        def empty_solver(workspace, mask_path, progress):
            return PoseReport(method="empty", status="failed", model_dir="")

        with pytest.raises(PipelineStateError, match="no model from this capture"):
            prepare_4d(project.path, solver=empty_solver)


# --------------------------------------------------------------------------
# training, export and the page
# --------------------------------------------------------------------------


@pytest.fixture()
def packaged(prepared):
    project, video, result = prepared
    assert main(["colab4d", str(project.path), "--json"]) == 0
    return project, video, result


@pytest.fixture()
def trained(packaged, capsys):
    # Everything above this line -- preparation, the cone, the labelling, the
    # packaging -- is numpy and OpenCV and runs anywhere. From here down the
    # trainer is PyTorch, which is the `train4d` extra and not a core
    # dependency, so this half skips rather than erroring out the module.
    pytest.importorskip("torch", reason="training needs the train4d extra")
    project, video, result = packaged
    capsys.readouterr()
    assert main([
        "train4d", str(project.path), "--device", "cpu",
        "--coarse-iters", "2", "--iters", "2", "--json",
    ]) == 0
    summary = json.loads(capsys.readouterr().out)
    return project, video, result, summary


class TestPackaging:
    def test_the_archive_holds_everything_the_gpu_host_needs(self, packaged):
        project, _video, result = packaged
        archive = project.path / "colab_export" / "dataset4d.zip"
        with zipfile.ZipFile(archive) as zf:
            names = set(zf.namelist())
        assert "scene4d_init.npz" in names
        assert "masks/dynamic.png" in names
        assert sum(1 for n in names if n.startswith("images/")) == result["frames"]
        assert "gausscapture/run_config_4d.json" in names
        assert FourDState.observe(project.path).stage == "packaged"


class TestTraining:
    def test_a_cpu_run_produces_a_checkpoint_and_labels_its_own_psnr(self, trained):
        project, _video, result, summary = trained
        assert Path(summary["scene"]).exists()
        assert summary["gaussians"] >= SPARSE_POINTS
        assert summary["nodes"] == result["nodes"]
        assert summary["frames"] == result["frames"]
        assert summary["device"] == "cpu"
        # Which kernel actually ran. `train_4d` used to pass no rasterizer at
        # all, so the reference one was selected by omission -- including under
        # `--device cuda`, where it cannot complete a forward pass at any real
        # scene size. A summary that does not name the backend cannot tell a
        # CUDA run from a CPU correctness check.
        assert summary["rasterizer"] == "reference"
        assert summary["densifier"] == "reference-mcmc"
        assert summary["field"] == "explicit"
        # D9: there is no held-out viewpoint anywhere in this design, and the
        # summary is exactly the thing that gets pasted into a table.
        assert "not novel-view synthesis" in summary["holdout_label"]
        assert FourDState.observe(project.path).stage == "trained"

    def test_the_default_device_refuses_rather_than_pretending(self, packaged, capsys):
        """No gsplat means no GPU trainer, and the CPU one is not a substitute."""
        pytest.importorskip("torch", reason="the message under test is the one after torch")
        project, _video, _result = packaged
        capsys.readouterr()
        assert main(["train4d", str(project.path)]) == 2
        message = capsys.readouterr().err
        assert "gsplat is not installed" in message
        assert "colab4d" in message
        assert not (project.path / "training" / "scene4d.npz").exists()

    def test_the_checkpoint_carries_the_cone_the_capture_measured(self, trained):
        _project, _video, result, summary = trained
        with np.load(summary["scene"], allow_pickle=False) as data:
            azimuth = float(data["cone_azimuth_deg"])
            verified = float(data["fixed_pose_verified"])
        assert azimuth == pytest.approx(result["cone"]["azimuth_half_deg"], abs=1e-3)
        assert verified == 1.0


class TestExportContract:
    """The trainer writes one file and the exporter reads it. No renaming step."""

    def test_the_checkpoint_loads_without_any_translation(self, trained):
        _project, _video, result, summary = trained
        scene, pruned = load_trained_scene(Path(summary["scene"]))
        scene.validate()
        assert scene.gaussian_count == summary["gaussians"]
        assert scene.node_count == result["nodes"]
        assert scene.frame_count == result["frames"]
        assert pruned == 0
        assert scene.cone_azimuth_deg == pytest.approx(
            result["cone"]["azimuth_half_deg"], abs=1e-3
        )
        # Colour is already linear RGB in the checkpoint: sh_degree is 0
        # throughout the 4D path, and decoding it twice would be silent.
        assert scene.colors.dtype == np.uint8 and scene.colors.shape[1] == 4

    def test_export4d_writes_a_container_that_reads_back(self, trained, capsys):
        project, _video, _result, _summary = trained
        capsys.readouterr()
        assert main(["export4d", str(project.path), "--json"]) == 0
        report = json.loads(capsys.readouterr().out)

        g4d = Path(report["path"])
        header = read_scene4d_header(g4d)
        assert header.gaussian_count == report["gaussian_count"]
        assert header.node_count == report["node_count"]
        assert header.frame_count == report["frame_count"]

        scene = read_scene4d(g4d)
        assert scene.gaussian_count == header.gaussian_count
        assert scene.skin_weights.shape[0] == scene.dynamic_count
        assert np.allclose(scene.skin_weights.sum(axis=1), 1.0, atol=2e-2)
        assert FourDState.observe(project.path).stage == "exported"

    def test_a_checkpoint_missing_an_array_is_refused_by_name(self, tmp_path: Path):
        path = tmp_path / "half.npz"
        np.savez(path, means=np.zeros((4, 3), np.float32))
        with pytest.raises(CaptureFormatError, match="quats"):
            load_trained_scene(path)


def write_checkpoint(
    path: Path,
    count: int = 64,
    dynamic_count: int = 32,
    nodes: int = 6,
    frames: int = 5,
    seed: int = 11,
) -> Path:
    """A ``scene4d.npz`` in exactly the names the training loop writes.

    Hand-built rather than trained, because the properties under test -- what a
    budget truncates, what an opacity floor prunes -- need a controlled spread
    of opacities, and four steps of a real fit produce none.
    """
    rng = np.random.default_rng(seed)
    quats = rng.normal(size=(count, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    node_quats = np.zeros((frames, nodes, 4), dtype=np.float32)
    node_quats[..., 0] = 1.0
    np.savez_compressed(
        path,
        means=rng.normal(scale=0.3, size=(count, 3)).astype(np.float32),
        scales=np.full((count, 3), 0.01, dtype=np.float32),
        quats=quats.astype(np.float32),
        # A ramp, so any threshold keeps a known fraction.
        opacities=np.linspace(0.02, 0.98, count).astype(np.float32),
        colors=rng.uniform(0.0, 1.0, size=(count, 3)).astype(np.float32),
        dynamic_count=np.int32(dynamic_count),
        node_rest=rng.normal(scale=0.3, size=(nodes, 3)).astype(np.float32),
        # The singular name is what the loop writes; the exporter has to read it.
        node_quat=node_quats,
        node_trans=rng.normal(scale=0.01, size=(frames, nodes, 3)).astype(np.float32),
        skin_idx=rng.integers(0, nodes, size=(dynamic_count, 4)).astype(np.int32),
        skin_weights=np.full((dynamic_count, 4), 0.25, dtype=np.float32),
        frame_times=np.linspace(0.0, 1.0, frames).astype(np.float32),
        fps=np.float32(30.0),
        cone_azimuth_deg=np.float32(21.0),
        cone_elevation_deg=np.float32(6.0),
        capture_eye=np.array([0.0, 0.0, 1.0], np.float32),
        capture_forward=np.array([0.0, 0.0, -1.0], np.float32),
        capture_up=np.array([0.0, 1.0, 0.0], np.float32),
        fixed_pose_verified=np.float32(0.0),
        meta=np.asarray(json.dumps({"label": "hand-built"})),
    )
    return path


class TestExportBudgets:
    """Pruning and truncation are reported, because they change what is shown."""

    def test_the_loops_own_array_names_export_without_translation(self, tmp_path: Path):
        scene, pruned = load_trained_scene(write_checkpoint(tmp_path / "scene4d.npz"))
        scene.validate()
        assert pruned == 0
        assert scene.gaussian_count == 64
        assert scene.dynamic_count == 32
        assert scene.node_count == 6 and scene.frame_count == 5
        assert scene.cone_azimuth_deg == pytest.approx(21.0)
        assert scene.fixed_pose_verified is False
        assert scene.meta["label"] == "hand-built"

    def test_an_opacity_floor_drops_gaussians_and_says_how_many(
        self, tmp_path: Path, capsys
    ):
        source = write_checkpoint(tmp_path / "scene4d.npz")
        capsys.readouterr()
        assert main(["export4d", str(source), "--min-opacity", "0.5", "--json"]) == 0
        report = json.loads(capsys.readouterr().out)
        # Opacities ramp from 0.02 to 0.98 over 64 gaussians, so half go.
        assert report["pruned_faint"] == pytest.approx(32, abs=1)
        assert report["gaussian_count"] + report["pruned_faint"] == 64
        read_scene4d(Path(report["path"])).validate()

    def test_a_floor_above_every_gaussian_is_refused_rather_than_writing_nothing(
        self, tmp_path: Path, capsys
    ):
        source = write_checkpoint(tmp_path / "scene4d.npz")
        capsys.readouterr()
        assert main(["export4d", str(source), "--min-opacity", "0.999"]) == 2
        assert "removes every gaussian" in capsys.readouterr().err

    def test_a_budget_truncates_and_the_report_says_so(self, tmp_path: Path, capsys):
        source = write_checkpoint(tmp_path / "scene4d.npz")
        capsys.readouterr()
        assert main(["export4d", str(source), "--max-gaussians", "20", "--json"]) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["truncated"] is True
        assert report["gaussian_count"] == 20
        assert report["trained_gaussian_count"] == 64
        scene = read_scene4d(Path(report["path"]))
        assert scene.gaussian_count == 20
        assert scene.skin_weights.shape[0] == scene.dynamic_count

    def test_the_page_built_from_a_bare_container_still_states_the_cone(
        self, tmp_path: Path, capsys
    ):
        source = write_checkpoint(tmp_path / "scene4d.npz")
        assert main(["export4d", str(source), "--json"]) == 0
        capsys.readouterr()
        assert main(["viewer4d", str(tmp_path / "scene4d.g4d"), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        sidecar = json.loads(
            (Path(payload["directory"]) / "scene.json").read_text(encoding="utf-8")
        )
        assert sidecar["cone"]["azimuth_deg"] == pytest.approx(21.0, abs=1e-2)
        assert sidecar["fixed_pose"]["source"] == "asserted"


class TestCollectingATrainedScene:
    def test_a_scene_trained_elsewhere_is_adopted_and_then_exports(
        self, trained, tmp_path: Path, store: ProjectStore, capsys
    ):
        _project, video, _result, summary = trained
        # A second project standing in for "the same capture, different laptop".
        elsewhere = store.create("collected", "person")
        fx.write_pack(elsewhere.path, video)
        capsys.readouterr()
        assert main([
            "colab4d", str(elsewhere.path), "--collect", summary["scene"], "--json"
        ]) == 0
        adopted = json.loads(capsys.readouterr().out)
        assert Path(adopted["scene"]).exists()
        assert FourDState.observe(elsewhere.path).stage == "trained"
        assert main(["export4d", str(elsewhere.path), "--json"]) == 0

    def test_a_download_that_is_really_an_error_page_is_refused(
        self, capture, tmp_path: Path, capsys
    ):
        project, _video = capture
        fake = tmp_path / "scene4d.npz"
        fake.write_bytes(b"<html><body>404 Not Found</body></html>")
        capsys.readouterr()
        assert main(["colab4d", str(project.path), "--collect", str(fake)]) == 2
        assert "not a readable .npz" in capsys.readouterr().err


class TestBrokenInputsAreNamed:
    """No 4D command may reach the user as an AttributeError or a stack trace."""

    def test_a_dataset_zip_that_is_not_a_zip(self, tmp_path: Path, capsys):
        # Without torch the command refuses earlier, and says so -- also a
        # named error, but not the one this test is about.
        pytest.importorskip("torch", reason="train4d refuses before the zip is opened")
        broken = tmp_path / "dataset4d.zip"
        broken.write_bytes(b"<html>404</html>")
        capsys.readouterr()
        assert main(["train4d", str(broken), "--device", "cpu"]) == 2
        assert "not a readable zip archive" in capsys.readouterr().err

    def test_a_checkpoint_that_is_not_an_npz(self, tmp_path: Path, capsys):
        broken = tmp_path / "scene4d.npz"
        broken.write_bytes(b"<html>404</html>")
        capsys.readouterr()
        assert main(["export4d", str(broken)]) == 2
        assert "not a readable .npz" in capsys.readouterr().err

    def test_a_container_that_is_not_a_g4d(self, tmp_path: Path, capsys):
        broken = tmp_path / "scene.g4d"
        broken.write_bytes(b"nope")
        capsys.readouterr()
        assert main(["viewer4d", str(broken)]) == 2
        assert "g4d" in capsys.readouterr().err


class TestViewerPage:
    @pytest.fixture()
    def exported(self, trained):
        project = trained[0]
        assert main(["export4d", str(project.path), "--json"]) == 0
        return trained

    def test_the_page_is_self_contained_and_states_what_it_shows(self, exported, capsys):
        project, _video, result, _summary = exported
        capsys.readouterr()
        assert main(["viewer4d", str(project.path), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)

        directory = Path(payload["directory"])
        assert (directory / "index.html").exists()
        assert (directory / "viewer_4d.js").exists()
        assert (directory / "scene.g4d").exists()
        html = (directory / "index.html").read_text(encoding="utf-8")
        # No CDN, no build step, no network at all: every URL the page resolves
        # is relative or a data: URI. (An inline SVG favicon carries the SVG
        # namespace, which is a name and not a fetch.)
        for attribute in ('src="http', "src='http", 'href="http', "href='http"):
            assert attribute not in html
        assert "@import" not in html and "cdn." not in html

        sidecar = json.loads((directory / "scene.json").read_text(encoding="utf-8"))
        assert sidecar["cone"]["azimuth_deg"] == pytest.approx(
            result["cone"]["azimuth_half_deg"], abs=1e-2
        )
        assert sidecar["fixed_pose"]["source"] == "verified"
        assert "no evidence" in sidecar["caveat"]

    def test_the_shipped_javascript_reads_the_container_this_run_produced(
        self, exported, tmp_path: Path
    ):
        """The browser half, fed a real fit rather than a hand-built scene.

        ``test_viewer_contract`` only ever decoded ``synthetic_scene()``, whose
        every value was chosen by the fixture. The container written from an
        actual checkpoint -- sh0-decoded colours, a measured cone, a real
        dynamic split, whatever opacity spread the fit produced -- had never
        been through the reader that decides what a user sees.
        """
        node = shutil.which("node")
        if node is None:
            pytest.skip("node is not installed; the browser half cannot be executed")
        from tests.test_viewer_contract import decode_with_node

        project = exported[0]
        g4d = project.path / SCENE4D_G4D_RELPATH
        decoded = decode_with_node(node, tmp_path / "js", g4d)

        read = decoded["read"]
        np.testing.assert_allclose(decoded["positions"], read.means, atol=1e-5)
        np.testing.assert_array_equal(decoded["colors"], read.colors)
        np.testing.assert_allclose(decoded["node_trans"], read.node_trans, atol=1e-6)
        assert decoded["header"]["coneAzimuthDeg"] == pytest.approx(
            read.cone_azimuth_deg, abs=1e-4
        )
        assert decoded["header"]["fixedPoseVerified"] is True

    def test_a_single_file_page_needs_no_sidecar(self, exported, tmp_path: Path):
        project = exported[0]
        out = tmp_path / "one"
        assert main([
            "viewer4d", str(project.path), "--out", str(out), "--single-file", "--json"
        ]) == 0
        html = (out / "index.html").read_text(encoding="utf-8")
        assert "SCENE_BASE64" in html
        assert not (out / "scene.g4d").exists()

    def test_the_published_cone_can_be_narrowed(self, exported, tmp_path: Path):
        project, _video, result, _summary = exported
        out = tmp_path / "narrow"
        assert main([
            "viewer4d", str(project.path), "--out", str(out),
            "--cone-azimuth", "5", "--json",
        ]) == 0
        cone = json.loads((out / "scene.json").read_text(encoding="utf-8"))["cone"]
        assert cone["azimuth_deg"] == pytest.approx(5.0)
        assert cone["narrowed"] is True
        assert cone["measured_azimuth_deg"] == pytest.approx(
            result["cone"]["azimuth_half_deg"], abs=1e-2
        )

    def test_the_published_cone_can_never_be_widened(self, exported, tmp_path: Path):
        project, _video, result, _summary = exported
        out = tmp_path / "wide"
        assert main([
            "viewer4d", str(project.path), "--out", str(out),
            "--cone-azimuth", "170", "--json",
        ]) == 0
        cone = json.loads((out / "scene.json").read_text(encoding="utf-8"))["cone"]
        assert cone["azimuth_deg"] == pytest.approx(
            result["cone"]["azimuth_half_deg"], abs=1e-2
        )
        assert cone["narrowed"] is False
        assert any("only ever be narrowed" in note for note in cone["notes"])


class TestMaskCommand:
    def test_it_measures_the_region_the_fixture_moved(self, capture, capsys):
        project, video = capture
        from gausscapture.ingest.frames import FRAME_PRESETS, extract_frames

        extract_frames(project.path, FRAME_PRESETS["dense"])
        capsys.readouterr()
        assert main(["mask", str(project.path), "--json"]) == 0
        report = json.loads(capsys.readouterr().out)

        assert report["frames"] >= 3
        assert 0.0 < report["union_fraction"] < 0.6
        assert report["phases"]["source"] == "declared"
        assert Path(report["masks_dir"]).is_dir()
        assert FourDState.observe(project.path).stage == "masked"

        # The measured region has to land on the rectangle the fixture actually
        # swept and essentially nowhere else. It does not fill the swept box:
        # `mask` reads the project's thinned frame extraction, so it sees the
        # blob at a subset of its positions, which is a fact about the input
        # rather than about the measurement.
        import cv2

        union = cv2.imread(str(Path(report["union_path"])), cv2.IMREAD_GRAYSCALE) > 127
        x0, y0, x1, y1 = video.blob_bbox
        assert union[y0:y1, x0:x1].mean() > 0.5

        pad = report["dilate_px"] + 2
        outside = union.copy()
        outside[max(0, y0 - pad) : y1 + pad, max(0, x0 - pad) : x1 + pad] = False
        assert outside.mean() < 0.02
