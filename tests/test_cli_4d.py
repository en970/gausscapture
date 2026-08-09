"""CLI tests for the fixed-camera 4D path.

The 4D commands are the interface a user drives across two machines -- prepare
here, train on a rented GPU, export and view here again -- so the contracts
that matter are the ones that survive the gap: exit codes, JSON on stdout, and
a refusal that names the command which produces the missing input. Those are
what is tested here.

The heavy stages (mask, prep4d, train4d, export4d, viewer4d) call into modules
that need COLMAP, torch or a GPU. Their *state* handling is checked -- every one
of them validates before it imports anything -- and the work itself is left to
each module's own tests.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from gausscapture.cli import _build_parser, main
from gausscapture.project import FourDState, ProjectStore, record_fourd_stage


def _write_init(project_path: Path, frames: int = 3) -> Path:
    """A scene4d_init.npz with the right shape and no meaning.

    Enough for the packaging and state code to be exercised without a capture,
    a COLMAP model, or a GPU anywhere in the test suite.
    """
    path = project_path / "training" / "scene4d_init.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    np.savez(
        path,
        means=rng.normal(size=(64, 3)).astype(np.float32),
        log_scales=np.full((64, 3), -3.0, dtype=np.float32),
        quats=np.tile(np.array([1, 0, 0, 0], np.float32), (64, 1)),
        dynamic=np.zeros(64, dtype=bool),
        node_positions=rng.normal(size=(8, 3)).astype(np.float32),
        cone_deg=np.array([21.4, 6.8], dtype=np.float32),
        frame_count=np.array(frames, dtype=np.int32),
    )
    return path


def _write_frames(project_path: Path, count: int = 4) -> None:
    images = project_path / "frames" / "images"
    images.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (images / f"hold_{index:04d}.jpg").write_bytes(b"\xff\xd8\xff\xd9")


@pytest.fixture()
def prepared(store: ProjectStore, tmp_path: Path):
    """A project that has reached the 'prepared' 4D stage."""
    project = store.create("bullet time", "person")
    _write_frames(project.path)
    _write_init(project.path)
    record_fourd_stage(project.path, "prepared", cone={"azimuth_deg": 21.4, "elevation_deg": 6.8})
    return project


class TestParsing:
    """Every flag the plan promises is accepted, with the promised default."""

    @pytest.mark.parametrize(
        ("argv", "expected"),
        [
            (["mask", "P"], {"project": "P"}),
            (["prep4d", "P"], {"nodes": 256, "skin_k": 4, "cap_max": 400_000}),
            (["prep4d", "P", "--nodes", "64", "--skin-k", "2"], {"nodes": 64, "skin_k": 2}),
            (["colab4d", "P"], {"collect": None, "iters": 12_000}),
            (["train4d", "d.zip"], {"device": "cuda", "coarse_iters": 3000, "iters": 12_000}),
            (["train4d", "d.zip", "--device", "cpu"], {"device": "cpu"}),
            # The motion parameterisation and the resume path. Both existed as
            # function parameters no flag ever set, which made the K-Planes
            # field and its three regularisers unreachable from any shipped
            # command, and made `save_checkpoint`/`load_checkpoint` a resume
            # feature that looked implemented and was not wired to anything.
            (["train4d", "d.zip"],
             {"field": "explicit", "resume": False, "checkpoint_every": 1000}),
            (["train4d", "d.zip", "--field", "kplanes", "--resume",
              "--checkpoint-every", "250"],
             {"field": "kplanes", "resume": True, "checkpoint_every": 250}),
            (["export4d", "s.npz"], {"min_opacity": 0.02, "max_gaussians": None, "skin_k": 4}),
            (["viewer4d", "s.g4d"], {"port": 8900, "single_file": False, "cone_azimuth": None}),
            (["viewer4d", "s.g4d", "--single-file", "--cone-azimuth", "12"],
             {"single_file": True, "cone_azimuth": 12.0}),
        ],
    )
    def test_defaults(self, argv, expected):
        args = _build_parser().parse_args(argv)
        assert args.handler is not None
        for key, value in expected.items():
            assert getattr(args, key) == value

    def test_every_4d_command_supports_json(self):
        parser = _build_parser()
        for argv in (["mask", "P"], ["prep4d", "P"], ["colab4d", "P"],
                     ["train4d", "d.zip"], ["export4d", "s.npz"], ["viewer4d", "s.g4d"]):
            assert parser.parse_args([*argv, "--json"]).json is True


class TestDoctor:
    def test_reports_the_4d_stack_honestly(self, projects_dir, capsys):
        assert main(["doctor", "--json"]) == 0
        report = json.loads(capsys.readouterr().out)
        fourd = report["fourd_trainer"]
        assert set(fourd) >= {"torch", "gsplat", "cuda", "usable", "gpu_usable", "note"}
        # "no CUDA" and "we could not look" are different answers, and only the
        # second is honest on a machine with no torch to ask.
        if not fourd["torch"]:
            assert fourd["cuda"] is None
            assert fourd["usable"] is False
        assert fourd["gpu_usable"] is (fourd["torch"] and fourd["gsplat"] and bool(fourd["cuda"]))
        assert fourd["note"]

    def test_scipy_row_is_present(self, projects_dir, capsys):
        main(["doctor", "--json"])
        assert "scipy (optional, 4D)" in json.loads(capsys.readouterr().out)["checks"]


class TestState:
    def test_a_fresh_project_is_not_4d(self, store):
        project = store.create("static only")
        state = project.fourd_state()
        assert state.stage == "none"
        assert state.is_fourd is False

    def test_stage_follows_the_artefacts(self, store):
        project = store.create("staged")
        _write_init(project.path)
        assert project.fourd_state().stage == "prepared"

        project.dataset4d_zip.parent.mkdir(parents=True, exist_ok=True)
        project.dataset4d_zip.write_bytes(b"PK\x05\x06" + b"\0" * 18)
        assert project.fourd_state().stage == "packaged"

        # Deleting the artefact takes the stage with it: the recorded note says
        # 'packaged', and the observed stage must not believe it.
        project.dataset4d_zip.unlink()
        assert project.fourd_state().stage == "prepared"

    def test_require_names_the_command_that_produces_the_input(self, store):
        from gausscapture.errors import PipelineStateError

        project = store.create("empty")
        with pytest.raises(PipelineStateError) as error:
            project.fourd_state().require("trained", "Train it first: `gausscapture train4d`.")
        assert "train4d" in str(error.value)
        assert "'none'" in str(error.value)

    def test_notes_survive_a_corrupt_state_file(self, store):
        project = store.create("corrupt")
        _write_init(project.path)
        (project.path / "training" / "fourd.json").write_text("{not json", encoding="utf-8")
        assert FourDState.observe(project.path).notes == {}

    def test_project_show_reports_the_4d_block(self, prepared, capsys):
        assert main(["project", "show", prepared.id]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["fourd"]["stage"] == "prepared"
        assert payload["fourd"]["next_command"] == "colab4d"
        assert payload["fourd"]["init"].endswith("scene4d_init.npz")
        assert payload["fourd"]["scene"] is None

    def test_project_show_omits_it_for_a_static_project(self, store, capsys):
        project = store.create("static")
        main(["project", "show", project.id])
        assert "fourd" not in json.loads(capsys.readouterr().out)


class TestRefusals:
    """Each command validates before it imports, so a wrong state is exit 2."""

    def test_mask_without_frames(self, store, capsys):
        project = store.create("no frames")
        assert main(["mask", str(project.path)]) == 2
        assert "gausscapture frames" in capsys.readouterr().err

    def test_prep4d_without_frames(self, store):
        project = store.create("no frames")
        assert main(["prep4d", str(project.path)]) == 2

    def test_colab4d_without_preparation(self, store, capsys):
        project = store.create("unprepared")
        assert main(["colab4d", str(project.path)]) == 2
        assert "prep4d" in capsys.readouterr().err

    def test_train4d_without_a_package(self, store, capsys):
        project = store.create("unpackaged")
        assert main(["train4d", str(project.path)]) == 2
        assert "colab4d" in capsys.readouterr().err

    def test_train4d_on_a_missing_zip(self, projects_dir, tmp_path):
        assert main(["train4d", str(tmp_path / "absent.zip")]) == 2

    def test_train4d_says_what_to_do_without_torch(self, prepared, tmp_path, monkeypatch, capsys):
        import gausscapture.recon.external as external

        prepared.dataset4d_zip.parent.mkdir(parents=True, exist_ok=True)
        prepared.dataset4d_zip.write_bytes(b"PK\x05\x06" + b"\0" * 18)
        monkeypatch.setattr(
            external,
            "deform_trainer_status",
            lambda probe_cuda=True: {
                "torch": False, "gsplat": False, "cuda": None, "usable": False,
                "gpu_usable": False, "note": "torch is not installed ... use `colab4d`.",
            },
        )
        assert main(["train4d", str(prepared.path)]) == 2
        assert "colab4d" in capsys.readouterr().err

    def test_export4d_without_training(self, store, capsys):
        project = store.create("untrained")
        _write_init(project.path)
        assert main(["export4d", str(project.path)]) == 2
        assert "--collect" in capsys.readouterr().err

    def test_viewer4d_without_an_export(self, store, capsys):
        project = store.create("unexported")
        _write_init(project.path)
        assert main(["viewer4d", str(project.path)]) == 2
        assert "export4d" in capsys.readouterr().err

    def test_unknown_project_is_exit_two(self, projects_dir):
        assert main(["prep4d", "does-not-exist"]) == 2


class TestColabPackage:
    def test_packages_the_prepared_scene(self, prepared, capsys):
        assert main(["colab4d", str(prepared.path), "--json"]) == 0
        result = json.loads(capsys.readouterr().out)

        assert result["images"] == 4
        assert result["packer"] == "colab4d"
        assert Path(result["dataset_zip"]).exists()

        with zipfile.ZipFile(result["dataset_zip"]) as archive:
            names = set(archive.namelist())
        assert "scene4d_init.npz" in names
        assert "README_COLAB_4D.md" in names
        assert "gausscapture/run_config_4d.json" in names
        assert sum(1 for n in names if n.startswith("images/")) == 4

    def test_records_the_stage_and_the_run_configuration(self, prepared, capsys):
        main(["colab4d", str(prepared.path), "--iters", "500", "--cap-max", "1000", "--json"])
        capsys.readouterr()

        assert prepared.fourd_state().stage == "packaged"
        assert ProjectStore().get(prepared.id).status == "Packaged for GPU (4D)"

        with zipfile.ZipFile(prepared.dataset4d_zip) as archive:
            config = json.loads(archive.read("gausscapture/run_config_4d.json"))
        assert config == {"coarse_iters": 3000, "iters": 500, "cap_max": 1000}

    def test_honours_an_explicit_destination(self, prepared, tmp_path, capsys):
        out = tmp_path / "elsewhere" / "dataset4d.zip"
        assert main(["colab4d", str(prepared.path), "--out", str(out), "--json"]) == 0
        assert out.exists()
        assert json.loads(capsys.readouterr().out)["dataset_zip"] == str(out)

    def test_refuses_a_package_with_no_pixels(self, store, capsys):
        project = store.create("no frames")
        _write_init(project.path)  # prepared, but nothing to fit
        assert main(["colab4d", str(project.path)]) == 2
        assert "prep4d" in capsys.readouterr().err

    def test_human_output_names_the_next_step(self, prepared, capsys):
        assert main(["colab4d", str(prepared.path)]) == 0
        out = capsys.readouterr().out
        assert "dataset4d.zip" in out
        assert "--collect" in out


class TestCollect:
    def test_adopts_a_scene_trained_elsewhere(self, prepared, tmp_path, capsys):
        downloaded = tmp_path / "scene4d.npz"
        np.savez(downloaded, means=np.zeros((4, 3), np.float32),
                 node_trajectories=np.zeros((3, 8, 7), np.float32))

        assert main(["colab4d", str(prepared.path), "--collect", str(downloaded), "--json"]) == 0
        result = json.loads(capsys.readouterr().out)

        assert prepared.scene4d_path.exists()
        assert result["arrays"] == ["means", "node_trajectories"]
        assert prepared.fourd_state().stage == "trained"
        assert ProjectStore().get(prepared.id).status == "Trained (4D)"

    def test_refuses_a_downloaded_error_page(self, prepared, tmp_path, capsys):
        # The failure this guards against: a browser saved an HTML error under
        # the .npz name, and it then sits in the project looking like a run.
        fake = tmp_path / "scene4d.npz"
        fake.write_text("<!doctype html><title>404</title>", encoding="utf-8")
        assert main(["colab4d", str(prepared.path), "--collect", str(fake)]) == 2
        assert not prepared.scene4d_path.exists()

    def test_refuses_a_missing_file(self, prepared, tmp_path):
        assert main(["colab4d", str(prepared.path), "--collect", str(tmp_path / "gone.npz")]) == 2


def _declare_phases(project, hold: tuple[int, int] = (25, 39), with_hold: bool = True) -> None:
    """Add a capturepack 0.3 protocol block to an already-valid pack."""
    path = project.capturepack_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    phases = [
        {"name": "perch", "start_frame": 0, "end_frame": 9},
        {"name": "arc", "start_frame": 10, "end_frame": 19},
        {"name": "reseat", "start_frame": 20, "end_frame": hold[0] - 1},
    ]
    if with_hold:
        phases.append({"name": "hold", "start_frame": hold[0], "end_frame": hold[1]})
    manifest["capture_type"] = "fixed_camera_4d"
    manifest["protocol"] = {
        "phases": phases,
        "rest_window": {"start_frame": hold[0] - 3, "end_frame": hold[0] - 1},
        "fps": 10,
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


class TestMask:
    def test_measures_the_hold_phase(self, project_with_video, capsys):
        assert main(["frames", str(project_with_video.path), "--preset", "dense", "--json"]) == 0
        capsys.readouterr()
        _declare_phases(project_with_video)

        assert main(["mask", str(project_with_video.path), "--json"]) == 0
        report = json.loads(capsys.readouterr().out)

        assert report["frames"] >= 3
        assert Path(report["masks_dir"]).is_dir()
        assert Path(report["union_path"]).exists()
        # The threshold is measured, never assumed: it comes from the rest
        # window's own noise, and the report says which span supplied it.
        assert report["noise_source"]
        assert report["threshold_levels"] > 0
        assert report["phases"]["source"] == "declared"
        assert project_with_video.fourd_state().stage == "masked"

    def test_refuses_a_fixed_camera_pack_with_no_hold(self, project_with_video, capsys):
        main(["frames", str(project_with_video.path), "--preset", "dense", "--json"])
        capsys.readouterr()
        _declare_phases(project_with_video, with_hold=False)

        assert main(["mask", str(project_with_video.path)]) == 2
        assert "hold" in capsys.readouterr().err

    def test_refuses_before_frames_are_extracted(self, project_with_video, capsys):
        _declare_phases(project_with_video)
        assert main(["mask", str(project_with_video.path)]) == 2
        assert "gausscapture frames" in capsys.readouterr().err


def _write_checkpoint(
    path: Path, count: int = 12, dynamic: int = 5, nodes: int = 3, frames: int = 4
) -> Path:
    """A trainer checkpoint in the form the trainer actually holds it.

    Log scales, logit opacities, an SH0 colour and a scattered per-gaussian
    dynamic flag -- the three conversions and the regrouping the exporter
    exists to perform.
    """
    rng = np.random.default_rng(11)
    is_dynamic = np.zeros(count, dtype=bool)
    is_dynamic[rng.choice(count, size=dynamic, replace=False)] = True

    np.savez(
        path,
        means=rng.normal(scale=0.5, size=(count, 3)).astype(np.float32),
        log_scales=np.full((count, 3), np.log(0.05), dtype=np.float32),
        quats=rng.normal(size=(count, 4)).astype(np.float32),
        sh0=rng.normal(scale=0.3, size=(count, 3)).astype(np.float32),
        logit_opacities=np.full(count, 2.0, dtype=np.float32),
        dynamic=is_dynamic,
        node_rest=rng.normal(scale=0.3, size=(nodes, 3)).astype(np.float32),
        node_quats=np.tile(np.array([1, 0, 0, 0], np.float32), (frames, nodes, 1)),
        node_trans=rng.normal(scale=0.02, size=(frames, nodes, 3)).astype(np.float32),
        skin_indices=rng.integers(0, nodes, size=(count, 4)).astype(np.int64),
        skin_weights=np.full((count, 4), 0.25, dtype=np.float32),
        fps=np.float32(30.0),
        cone_azimuth_deg=np.float32(21.4),
        cone_elevation_deg=np.float32(6.8),
    )
    return path


class TestExport:
    def test_writes_a_readable_container(self, projects_dir, tmp_path, capsys):
        from gausscapture.export.scene4d import read_scene4d_header

        checkpoint = _write_checkpoint(tmp_path / "scene4d.npz")
        out = tmp_path / "scene.g4d"
        assert main(["export4d", str(checkpoint), "--out", str(out), "--json"]) == 0
        result = json.loads(capsys.readouterr().out)

        assert result["gaussian_count"] == 12
        assert result["dynamic_count"] == 5
        assert result["pruned_faint"] == 0
        assert result["truncated"] is False

        header = read_scene4d_header(out)
        assert header.gaussian_count == 12
        assert header.dynamic_count == 5
        assert header.frame_count == 4
        # The cone travels from the capture to the container untouched: it is
        # the one number the viewer is not allowed to invent.
        assert header.cone_azimuth_deg == pytest.approx(21.4, abs=1e-3)
        assert header.cone_elevation_deg == pytest.approx(6.8, abs=1e-3)

    def test_prunes_faint_gaussians(self, projects_dir, tmp_path, capsys):
        checkpoint = _write_checkpoint(tmp_path / "scene4d.npz")
        with np.load(checkpoint) as data:
            arrays = dict(data)
        # Four gaussians the optimiser drove to nothing.
        arrays["logit_opacities"][:4] = -12.0
        np.savez(checkpoint, **arrays)

        out = tmp_path / "scene.g4d"
        assert main(["export4d", str(checkpoint), "--out", str(out),
                     "--min-opacity", "0.05", "--json"]) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["pruned_faint"] == 4
        assert result["gaussian_count"] == 8

    def test_truncates_to_the_budget_and_says_so(self, projects_dir, tmp_path, capsys):
        checkpoint = _write_checkpoint(tmp_path / "scene4d.npz")
        out = tmp_path / "scene.g4d"
        assert main(["export4d", str(checkpoint), "--out", str(out),
                     "--max-gaussians", "6", "--json"]) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["truncated"] is True
        assert result["gaussian_count"] == 6
        assert result["trained_gaussian_count"] == 12

    def test_reduces_the_skinning_width(self, projects_dir, tmp_path, capsys):
        checkpoint = _write_checkpoint(tmp_path / "scene4d.npz")
        out = tmp_path / "scene.g4d"
        assert main(["export4d", str(checkpoint), "--out", str(out),
                     "--skin-k", "2", "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["skin_k"] == 2

    def test_refuses_a_checkpoint_that_is_not_one(self, projects_dir, tmp_path, capsys):
        broken = tmp_path / "scene4d.npz"
        np.savez(broken, means=np.zeros((4, 3), np.float32))
        assert main(["export4d", str(broken), "--out", str(tmp_path / "s.g4d")]) == 2
        assert "quats" in capsys.readouterr().err

    def test_writes_into_the_project_and_records_the_stage(self, prepared, tmp_path, capsys):
        _write_checkpoint(prepared.scene4d_path)
        assert main(["export4d", str(prepared.path), "--json"]) == 0
        capsys.readouterr()

        assert prepared.scene4d_g4d_path.exists()
        assert prepared.fourd_state().stage == "exported"
        assert ProjectStore().get(prepared.id).status == "Exported (4D)"
