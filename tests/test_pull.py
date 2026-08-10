"""Taking captures off a phone.

adb is stubbed, so these exercise the parsing and the decisions rather than the
transport. What matters is that a capture is never imported twice and never
imported halfway.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from gausscapture.errors import GaussCaptureError
from gausscapture.ingest import pull

ROOT = pull.DEVICE_ROOT


def _completed(stdout: str = "", returncode: int = 0, stderr: str = ""):
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


def _listing(sizes: dict[str, int]) -> str:
    rows = ["total 123456"]
    for name, size in sizes.items():
        rows.append(f"-rw-rw---- 1 u0_a1 ext_data {size} 2026-07-30 19:00 {name}")
    return "\n".join(rows)


FULL = {"video.mp4": 180_000_000, "manifest.json": 2_100,
        "imu.jsonl": 21_000_000, "frames.jsonl": 240_000, "intrinsics.json": 800}


class FakePhone:
    """Answers the handful of adb calls the module makes."""

    def __init__(self, captures: dict[str, dict], manifests: dict[str, dict] | None = None,
                 devices_output: str | None = None):
        self.captures = captures
        self.manifests = manifests or {}
        self.devices_output = devices_output or "List of devices attached\nR3CT90ABCDE\tdevice\n"
        self.pulled: list[str] = []
        self.fail_pull: set[str] = set()

    def __call__(self, args, adb="adb", serial=None, timeout=600):
        if args[0] == "devices":
            return _completed(self.devices_output)
        if args[0] == "shell":
            command = args[1]
            if command.startswith("ls -1"):
                return _completed("\n".join(self.captures) + "\n")
            if command.startswith("ls -l"):
                name = command.split(f"{ROOT}/")[1].split()[0]
                return _completed(_listing(self.captures.get(name, {})))
            if command.startswith("cat"):
                name = command.split(f"{ROOT}/")[1].split("/")[0]
                if name not in self.manifests:
                    return _completed("")
                return _completed(json.dumps(self.manifests[name]))
        if args[0] == "pull":
            source, target = args[1], Path(args[2])
            name = source.rsplit("/", 1)[-1]
            self.pulled.append(name)
            if name in self.fail_pull:
                return _completed(returncode=1, stderr="remote object does not exist")
            target.mkdir(parents=True, exist_ok=True)
            for file in self.captures.get(name, {}):
                (target / file).write_text("x")
            return _completed("1 file pulled")
        return _completed()


@pytest.fixture
def phone(monkeypatch):
    def install(fake):
        monkeypatch.setattr(pull, "_adb", fake)
        monkeypatch.setattr(pull, "adb_available", lambda adb="adb": True)
        return fake
    return install


class TestDevices:
    def test_lists_authorised_devices(self, phone):
        phone(FakePhone({}))
        assert pull.devices() == ["R3CT90ABCDE"]

    def test_says_so_when_debugging_was_not_accepted(self, phone):
        # Reporting this as "no device" would send someone hunting for a cable
        # fault instead of at the dialog on their screen.
        phone(FakePhone({}, devices_output="List of devices attached\nR3CT9\tunauthorized\n"))
        with pytest.raises(GaussCaptureError, match="authorise"):
            pull.devices()

    def test_reports_nothing_when_nothing_is_attached(self, phone):
        phone(FakePhone({}, devices_output="List of devices attached\n"))
        assert pull.devices() == []

    def test_requires_adb(self, monkeypatch):
        from gausscapture.errors import DependencyMissingError

        monkeypatch.setattr(pull, "adb_available", lambda adb="adb": False)
        with pytest.raises(DependencyMissingError):
            pull.devices()


class TestListCaptures:
    def test_reads_sizes_and_manifest_fields(self, phone):
        phone(FakePhone(
            {"A_good_20260730_190000": FULL},
            {"A_good_20260730_190000": {"session_id": "abc-123", "preset": "A_good",
                                        "created_at": "2026-07-30T19:00:00Z"}},
        ))
        captures = pull.list_captures()

        assert len(captures) == 1
        capture = captures[0]
        assert capture.session_id == "abc-123"
        assert capture.preset == "A_good"
        assert capture.bytes == sum(FULL.values())
        assert capture.usable
        assert capture.missing == []

    def test_a_capture_without_video_is_not_usable(self, phone):
        phone(FakePhone({"broken": {"manifest.json": 100, "imu.jsonl": 50}}))
        capture = pull.list_captures()[0]

        assert not capture.usable
        assert "video.mp4" in capture.missing

    def test_survives_an_unreadable_manifest(self, phone):
        # A capture cut short still has a video worth looking at.
        fake = FakePhone({"partial": FULL})
        fake.manifests = {}
        phone(fake)
        capture = pull.list_captures()[0]

        assert capture.session_id is None
        assert capture.usable

    def test_empty_phone(self, phone):
        phone(FakePhone({}))
        assert pull.list_captures() == []


class TestPullCapture:
    def test_moves_into_place_only_when_complete(self, phone, tmp_path):
        fake = phone(FakePhone({"take": FULL}))
        capture = pull.list_captures()[0]

        out = pull.pull_capture(capture, tmp_path / "project" / "capturepack")

        assert (out / "video.mp4").exists()
        assert (out / "manifest.json").exists()
        assert fake.pulled == ["take"]

    def test_leaves_nothing_behind_when_the_pull_fails(self, phone, tmp_path):
        fake = phone(FakePhone({"take": FULL}))
        fake.fail_pull = {"take"}
        capture = pull.list_captures()[0]

        destination = tmp_path / "project" / "capturepack"
        with pytest.raises(GaussCaptureError, match="adb could not read"):
            pull.pull_capture(capture, destination)

        assert not destination.exists()
        # The staging directory must not survive either.
        assert list(tmp_path.iterdir()) == []

    def test_refuses_to_overwrite_a_destination_that_holds_anything(self, phone, tmp_path):
        phone(FakePhone({"take": FULL}))
        capture = pull.list_captures()[0]
        destination = tmp_path / "capturepack"
        destination.mkdir()
        (destination / "manifest.json").write_text("{}", encoding="utf-8")

        with pytest.raises(GaussCaptureError, match="already exists"):
            pull.pull_capture(capture, destination)
        # And it is still there: a refusal that deleted the thing it refused to
        # overwrite would be worse than the overwrite.
        assert (destination / "manifest.json").read_text(encoding="utf-8") == "{}"

    def test_accepts_an_empty_destination(self, phone, tmp_path):
        """`gausscapture pull` creates the project first, and that lays down `capturepack`.

        Treating the empty directory as a conflict made the command impossible to run: every
        capture raised, and the caller counted it as "already here" for a phone whose captures
        had never been copied.
        """
        phone(FakePhone({"take": FULL}))
        capture = pull.list_captures()[0]
        destination = tmp_path / "capturepack"
        destination.mkdir()

        pull.pull_capture(capture, destination)

        # Landed at the destination itself, not nested one level inside it.
        assert (destination / "manifest.json").exists()
        assert not (destination / "take").exists()

    def test_rejects_an_arrival_missing_required_files(self, phone, tmp_path, monkeypatch):
        fake = phone(FakePhone({"take": FULL}))
        capture = pull.list_captures()[0]

        # The listing promised a video; the transfer did not deliver one.
        original = fake.__call__

        def truncated(args, *rest, **kwargs):
            result = original(args, *rest, **kwargs)
            if args[0] == "pull":
                (Path(args[2]) / "video.mp4").unlink()
            return result

        monkeypatch.setattr(pull, "_adb", truncated)
        with pytest.raises(GaussCaptureError, match="video.mp4"):
            pull.pull_capture(capture, tmp_path / "capturepack")


class TestMarkOffloaded:
    def test_writes_the_sentinel_onto_the_phone(self, phone, tmp_path):
        fake = phone(FakePhone({"take": FULL}))
        pushed: list[tuple[str, str]] = []
        original = fake.__call__

        def record(args, *rest, **kwargs):
            if args[0] == "push":
                pushed.append((args[1], args[2]))
                return _completed("1 file pushed")
            return original(args, *rest, **kwargs)

        import gausscapture.ingest.pull as module

        module._adb = record
        capture = pull.list_captures()[0]
        assert pull.mark_offloaded(capture, host="mac", when="2026-08-04T09:00:00+00:00")

        assert len(pushed) == 1
        local, remote = pushed[0]
        assert remote.endswith(f"take/{pull.OFFLOAD_SENTINEL}")
        # The temporary file must not survive the call.
        assert not Path(local).exists()

    def test_reports_failure_rather_than_claiming_success(self, phone):
        fake = phone(FakePhone({"take": FULL}))
        original = fake.__call__

        def refuse(args, *rest, **kwargs):
            if args[0] == "push":
                return _completed(returncode=1, stderr="read-only file system")
            return original(args, *rest, **kwargs)

        import gausscapture.ingest.pull as module

        module._adb = refuse
        capture = pull.list_captures()[0]
        assert pull.mark_offloaded(capture, host="mac", when="now") is False

    def test_a_marked_capture_is_recognised_on_the_next_listing(self, phone):
        phone(FakePhone({"take": {**FULL, pull.OFFLOAD_SENTINEL: 120}}))
        assert pull.list_captures()[0].offloaded is True

    def test_an_unmarked_capture_is_not(self, phone):
        phone(FakePhone({"take": FULL}))
        assert pull.list_captures()[0].offloaded is False


class TestKnownSessionIds:
    def test_reads_ids_out_of_imported_projects(self, tmp_path):
        class Store:
            def list(self):
                return [type("P", (), {"path": tmp_path / "one"})()]

        pack = tmp_path / "one" / "capturepack"
        pack.mkdir(parents=True)
        (pack / "manifest.json").write_text(json.dumps({"session_id": "abc-123"}))

        assert pull.known_session_ids(Store()) == {"abc-123"}

    def test_ignores_a_project_with_no_manifest(self, tmp_path):
        class Store:
            def list(self):
                return [type("P", (), {"path": tmp_path / "empty"})()]

        (tmp_path / "empty").mkdir()
        assert pull.known_session_ids(Store()) == set()
