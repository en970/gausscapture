"""End-to-end tests of the stages that do not need COLMAP or a GPU.

This is the CPU smoke test the CI gate runs: import a synthetic capture,
analyse it, extract frames, and assemble a dataset, all in a few seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gausscapture.errors import CaptureFormatError, PipelineStateError
from gausscapture.ingest.frames import extract_frames
from gausscapture.pack import archive, manifest
from gausscapture.progress import CollectingProgress
from gausscapture.recon.dataset import build_dataset
from gausscapture.telemetry import analyze_capture
from tests.conftest import make_video


class TestProjectStore:
    def test_create_makes_the_expected_layout(self, store):
        project = store.create("living room", "room")
        assert (project.path / "project.json").exists()
        for child in ("capturepack", "frames", "colmap", "dataset", "export"):
            assert (project.path / child).is_dir()

    def test_round_trips_through_disk(self, store):
        created = store.create("kitchen")
        loaded = store.get(created.id)
        assert loaded.name == "kitchen"
        assert loaded.path == created.path

    def test_listing_skips_unreadable_projects(self, store):
        good = store.create("good")
        broken = store.projects_dir / "broken"
        broken.mkdir()
        (broken / "project.json").write_text("{not json", encoding="utf-8")
        assert [p.id for p in store.list()] == [good.id]

    def test_missing_project_raises(self, store):
        with pytest.raises(FileNotFoundError):
            store.get("nope")


class TestPack:
    def test_minimal_pack_is_valid_with_warnings(self, project_with_video):
        result = manifest.validate(project_with_video.capturepack_dir)
        assert result["valid"]
        # Every optional sensor file is absent, which is expected for a bare
        # video import and is exactly why COLMAP is needed downstream.
        assert any("poses" in w for w in result["warnings"])

    def test_missing_manifest_is_an_error_not_a_crash(self, tmp_path):
        result = manifest.validate(tmp_path)
        assert not result["valid"]
        assert result["errors"]

    def test_missing_video_invalidates_the_pack(self, project_with_video):
        video = manifest.find_main_video(project_with_video.capturepack_dir)
        video.unlink()
        assert not manifest.validate(project_with_video.capturepack_dir)["valid"]

    def test_checksums_round_trip(self, project_with_video):
        pack = project_with_video.capturepack_dir
        archive.write_checksums(pack)
        assert archive.verify_checksums(pack)["verified"]

    def test_checksums_detect_tampering(self, project_with_video):
        pack = project_with_video.capturepack_dir
        archive.write_checksums(pack)
        (pack / "manifest.json").write_text("{}", encoding="utf-8")
        result = archive.verify_checksums(pack)
        assert not result["verified"]
        assert "manifest.json" in result["mismatches"]

    def test_export_and_reimport(self, project_with_video, store):
        exported = archive.export_archive(project_with_video.path)
        assert exported.exists()
        other = store.create("reimported")
        result = archive.import_archive(other.path, exported)
        assert result["valid"]

    def test_rejects_a_non_zip(self, project_with_video, tmp_path):
        junk = tmp_path / "not.capturepack"
        junk.write_text("definitely not a zip", encoding="utf-8")
        with pytest.raises(CaptureFormatError):
            archive.import_archive(project_with_video.path, junk)


class TestVideoProbe:
    """A portrait phone recording stores a landscape frame plus a rotation flag.

    OpenCV applies that flag, so probe metadata must describe the decoded frame
    rather than the coded one -- otherwise every consumer sees dimensions
    transposed relative to what it actually reads, and camera intrinsics
    matched against them are silently wrong.
    """

    def _rotated(self, source: Path, degrees: int, out: Path) -> Path:
        import shutil
        import subprocess

        if not shutil.which("ffmpeg"):
            pytest.skip("ffmpeg not installed")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-display_rotation", str(degrees),
             "-i", str(source), "-c", "copy", str(out)],
            check=True,
        )
        return out

    def test_unrotated_dimensions(self, tmp_path):
        from gausscapture.ingest.video import probe_video

        source = make_video(tmp_path / "plain.mp4", frames=10, size=(160, 120))
        info = probe_video(source)
        assert (info.width, info.height) == (160, 120)
        assert info.rotation == 0

    def test_quarter_turn_transposes_reported_dimensions(self, tmp_path):
        import cv2

        from gausscapture.ingest.video import probe_video

        source = make_video(tmp_path / "plain.mp4", frames=10, size=(160, 120))
        rotated = self._rotated(source, 90, tmp_path / "rot90.mp4")

        info = probe_video(rotated)
        assert info.rotation == 90
        assert (info.width, info.height) == (120, 160)

        # The contract that matters: probe agrees with what a decoder returns.
        capture = cv2.VideoCapture(str(rotated))
        ok, frame = capture.read()
        capture.release()
        assert ok
        assert (frame.shape[1], frame.shape[0]) == (info.width, info.height)

    def test_half_turn_leaves_dimensions_alone(self, tmp_path):
        from gausscapture.ingest.video import probe_video

        source = make_video(tmp_path / "plain.mp4", frames=10, size=(160, 120))
        rotated = self._rotated(source, 180, tmp_path / "rot180.mp4")
        info = probe_video(rotated)
        assert info.rotation == 180
        assert (info.width, info.height) == (160, 120)


class TestTelemetry:
    def test_produces_a_report(self, project_with_video):
        report = analyze_capture(project_with_video.path, sample_count=20)
        assert report.frame_count > 0
        assert 0 <= report.score <= 100
        assert report.status in {"good", "warning", "bad"}
        assert report.vol_p10 <= report.vol_median

    def test_writes_json_to_disk(self, project_with_video):
        analyze_capture(project_with_video.path, sample_count=15)
        path = project_with_video.quality_report_path
        assert path.exists()
        assert "vol_p10" in json.loads(path.read_text(encoding="utf-8"))

    def test_warns_about_missing_poses(self, project_with_video):
        report = analyze_capture(project_with_video.path, sample_count=10)
        assert any("COLMAP" in w for w in report.warnings)

    def test_reports_progress(self, project_with_video):
        progress = CollectingProgress()
        analyze_capture(project_with_video.path, sample_count=10, progress=progress)
        assert progress.updates
        assert progress.updates[-1][0] == 100

    def test_write_false_leaves_no_artifacts(self, project_with_video):
        analyze_capture(project_with_video.path, sample_count=10, write=False)
        assert not project_with_video.quality_report_path.exists()


class TestFrameExtraction:
    def test_extracts_and_indexes(self, project_with_video):
        index = extract_frames(project_with_video.path, {"target_fps": 5, "max_frames": 20})
        assert index.frames_used > 0
        assert len(list(project_with_video.frames_dir.glob("*.jpg"))) == index.frames_used
        assert project_with_video.frame_index_path.exists()

    def test_records_rejected_frames_too(self, project_with_video):
        """Rejections are study data, so they must survive in the index."""
        index = extract_frames(project_with_video.path, {"target_fps": "all"})
        assert index.frames_total_sampled >= index.frames_used
        assert any(not f.used for f in index.frames) or index.frames_skipped_blur == 0

    def test_rejects_frames_that_go_soft(self, store, tmp_path):
        """A clip that degrades halfway through should lose its soft half."""
        from gausscapture.ingest.video import copy_video_into_pack, probe_video

        project = store.create("degrading")
        source = make_video(tmp_path / "degrading.mp4", frames=60, blur_from=30)
        for name in manifest.REQUIRED_DIRS:
            (project.capturepack_dir / name).mkdir(parents=True, exist_ok=True)
        video = copy_video_into_pack(source, project.path)
        manifest.write_manifest(
            project.capturepack_dir,
            manifest.create_minimal_manifest(
                f"video/{video.name}", probe_video(video), "degrading"
            ),
        )
        index = extract_frames(
            project.path, {"target_fps": "all", "duplicate_filter": False, "max_frames": 0}
        )
        assert index.frames_skipped_blur > 0

    def test_filters_can_be_disabled(self, project_with_video):
        index = extract_frames(
            project_with_video.path,
            {"target_fps": "all", "blur_filter": False, "duplicate_filter": False, "max_frames": 0},
        )
        assert index.frames_skipped_blur == 0
        assert index.frames_skipped_duplicate == 0

    def test_respects_the_frame_budget(self, project_with_video):
        index = extract_frames(
            project_with_video.path, {"target_fps": "all", "max_frames": 3, "blur_filter": False}
        )
        assert index.frames_used <= 3

    def test_moving_camera_is_not_mistaken_for_duplicates(self, project_with_video):
        """Histograms are invariant to spatial rearrangement.

        The fixture video slides a window across a fixed texture: every frame
        moves everywhere, yet the grayscale histogram barely changes. Judged on
        histogram correlation alone the filter discarded most of these frames
        -- which is exactly the parallax a reconstruction needs. Regression
        test for the pixel-difference gate.
        """
        index = extract_frames(
            project_with_video.path,
            {"target_fps": "all", "blur_filter": False, "max_frames": 0},
        )
        assert index.frames_skipped_duplicate == 0
        assert index.frames_used == index.frames_total_sampled

    def test_a_genuinely_static_video_is_deduplicated(self, store, tmp_path):
        """The inverse case: identical frames must still be caught."""
        import cv2
        import numpy as np

        from gausscapture.ingest.video import copy_video_into_pack, probe_video

        source = tmp_path / "static.mp4"
        writer = cv2.VideoWriter(
            str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120)
        )
        frame = np.random.default_rng(5).integers(0, 255, (120, 160, 3), dtype=np.uint8)
        for _ in range(30):
            writer.write(frame)
        writer.release()

        project = store.create("static")
        for name in manifest.REQUIRED_DIRS:
            (project.capturepack_dir / name).mkdir(parents=True, exist_ok=True)
        video = copy_video_into_pack(source, project.path)
        manifest.write_manifest(
            project.capturepack_dir,
            manifest.create_minimal_manifest(f"video/{video.name}", probe_video(video), "static"),
        )
        index = extract_frames(
            project.path, {"target_fps": "all", "blur_filter": False, "max_frames": 0}
        )
        assert index.frames_skipped_duplicate > 0
        # Lossy H.264-ish encoding adds noise, so a few frames may squeak past
        # the pixel gate; what matters is that the bulk are caught.
        assert index.frames_used < index.frames_total_sampled / 2


class TestDataset:
    def test_requires_frames(self, project_with_video):
        with pytest.raises(PipelineStateError, match="frame extraction"):
            build_dataset(project_with_video.path)

    def test_requires_a_sparse_model(self, project_with_video):
        extract_frames(project_with_video.path, {"target_fps": 5, "max_frames": 5})
        with pytest.raises(PipelineStateError, match="COLMAP"):
            build_dataset(project_with_video.path)

    def test_builds_the_conventional_layout(self, project_with_video):
        """The layout every 3DGS trainer expects: images/ and sparse/0/.

        Regression test for the bug where the project root was handed to the
        trainer with `-s`, so nothing could ever load.
        """
        extract_frames(project_with_video.path, {"target_fps": 5, "max_frames": 6})
        _fake_colmap_model(project_with_video.path / "colmap" / "sparse" / "0")

        dataset = build_dataset(project_with_video.path)
        assert (dataset / "images").is_dir()
        assert (dataset / "sparse" / "0" / "cameras.txt").exists()
        assert (dataset / "sparse" / "0" / "points3D.txt").exists()
        assert list((dataset / "images").glob("*.jpg"))

    def test_finds_a_model_written_without_a_numbered_subdir(self, project_with_video):
        extract_frames(project_with_video.path, {"target_fps": 5, "max_frames": 4})
        _fake_colmap_model(project_with_video.path / "colmap" / "sparse")
        assert (build_dataset(project_with_video.path) / "sparse" / "0" / "images.txt").exists()

    def test_copy_mode_works_when_hardlinks_are_unavailable(self, project_with_video):
        extract_frames(project_with_video.path, {"target_fps": 5, "max_frames": 4})
        _fake_colmap_model(project_with_video.path / "colmap" / "sparse" / "0")
        dataset = build_dataset(project_with_video.path, link=False)
        assert list((dataset / "images").glob("*.jpg"))


def _fake_colmap_model(model_dir: Path) -> None:
    """Minimal text model, enough for layout assembly to be exercised."""
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "cameras.txt").write_text("# Camera list\n1 PINHOLE 160 120 100 100 80 60\n", encoding="utf-8")
    (model_dir / "images.txt").write_text("# Image list\n1 1 0 0 0 0 0 0 1 frame_000001.jpg\n\n", encoding="utf-8")
    (model_dir / "points3D.txt").write_text("# 3D points\n1 0 0 0 255 255 255 0.5\n", encoding="utf-8")
