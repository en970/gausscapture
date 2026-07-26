"""CLI tests.

The CLI is the interface the evaluation harness drives, so its contracts --
exit codes, JSON on stdout, progress on stderr -- are part of the API.
"""

from __future__ import annotations

import json

import pytest

from gausscapture.cli import main


class TestBasics:
    def test_version(self, capsys):
        with pytest.raises(SystemExit) as exit_info:
            main(["--version"])
        assert exit_info.value.code == 0
        assert "gausscapture" in capsys.readouterr().out

    def test_no_arguments_prints_help(self, capsys):
        assert main([]) == 0
        assert "usage" in capsys.readouterr().out.lower()

    def test_doctor_reports_environment(self, projects_dir, capsys):
        assert main(["doctor", "--json"]) == 0
        report = json.loads(capsys.readouterr().out)
        assert "checks" in report
        assert report["checks"]["opencv"] is True


class TestProjectCommands:
    def test_create_then_list(self, projects_dir, capsys):
        assert main(["project", "create", "hallway", "--json"]) == 0
        created = json.loads(capsys.readouterr().out)

        assert main(["project", "list", "--json"]) == 0
        listed = json.loads(capsys.readouterr().out)
        assert [p["id"] for p in listed] == [created["id"]]

    def test_create_requires_a_name(self, projects_dir):
        assert main(["project", "create"]) == 2

    def test_delete(self, projects_dir, capsys):
        main(["project", "create", "temporary", "--json"])
        project_id = json.loads(capsys.readouterr().out)["id"]

        assert main(["project", "delete", project_id, "--json"]) == 0
        capsys.readouterr()  # discard the delete confirmation

        main(["project", "list", "--json"])
        assert json.loads(capsys.readouterr().out) == []

    def test_unknown_project_exits_with_two(self, projects_dir):
        assert main(["telemetry", "does-not-exist"]) == 2


class TestPipelineCommands:
    def test_import_creates_a_project(self, projects_dir, tmp_path, capsys):
        from tests.conftest import make_video

        source = make_video(tmp_path / "clip.mp4")
        assert main(["import", str(source), "--name", "imported", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["project"]["name"] == "imported"
        assert payload["video"]["width"] == 160

    def test_pack_validate_exits_zero(self, project_with_video, capsys):
        assert main(["pack", "validate", str(project_with_video.path), "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["valid"] is True

    def test_pack_validate_fails_on_a_broken_pack(self, project_with_video, capsys):
        (project_with_video.capturepack_dir / "manifest.json").unlink()
        assert main(["pack", "validate", str(project_with_video.path)]) == 1

    def test_telemetry_json_is_parseable(self, project_with_video, capsys):
        assert main(["telemetry", str(project_with_video.path), "--samples", "10", "--json"]) == 0
        report = json.loads(capsys.readouterr().out)
        assert "vol_p10" in report
        # Per-frame signals are omitted unless asked for, so the default output
        # stays small enough to log for every run in a batch.
        assert "signals" not in report

    def test_telemetry_signals_flag(self, project_with_video, capsys):
        main(["telemetry", str(project_with_video.path), "--samples", "10", "--json", "--signals"])
        assert json.loads(capsys.readouterr().out)["signals"]

    def test_frames(self, project_with_video, capsys):
        assert main(["frames", str(project_with_video.path), "--preset", "fast", "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["frames_used"] > 0

    def test_run_skips_pose_when_asked(self, project_with_video, capsys):
        assert main(["run", str(project_with_video.path), "--skip-pose", "--json"]) == 0
        results = json.loads(capsys.readouterr().out)
        assert results["pose"]["skipped"] is True
        assert results["telemetry"]["frame_count"] > 0
        assert results["frames"]["frames_used"] > 0
