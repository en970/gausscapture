"""The install lines the code prints, and the schema the phone writes against.

Two contracts that live outside Python and were enforced by nobody.

**Extras.** Three separate error messages tell a user to run
``pip install 'gausscapture[train4d]'``. There was no ``train4d`` extra, so pip
printed "does not provide the extra", installed the bare package, and exited
zero -- the instruction reads as followed when nothing happened. An extra named
in a string literal is a promise, so every such string is checked against what
the package actually declares.

**The CapturePack schema.** ``schemas/capturepack.schema.json`` is the published
phone-to-desktop contract and nothing in the repository had ever validated a
manifest against it: no jsonschema dependency, no validator call, no test. It
was enforced by code reading, which is exactly how the ``capture_settings`` key
divergence between ``CaptureEngine.java`` and ``pack/manifest.py`` survived.
"""

from __future__ import annotations

import importlib.metadata as metadata
import json
import re
import sys
from pathlib import Path

import pytest

# tomllib arrived in 3.11 and this project supports 3.10, where reading
# pyproject.toml directly is simply not available without a dependency. Rather
# than add one for a single test, the 3.10 path falls back to the installed
# distribution's own metadata -- a weaker source, because it describes the last
# install rather than the file on disk, but one that still catches an extra a
# message promises and the package does not provide.
if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on the 3.10 leg of the CI matrix
    tomllib = None

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "gausscapture"
SCHEMA_PATH = SRC / "schemas" / "capturepack.schema.json"

#: Matches ``gausscapture[train4d]`` however it is quoted in a message.
_EXTRA = re.compile(r"gausscapture\[([a-z0-9,_\-]+)\]")


def declared_extras() -> set[str]:
    """What ``pyproject.toml`` declares -- the source a wheel is built from."""
    if tomllib is None:
        return set(metadata.metadata("gausscapture").get_all("Provides-Extra") or [])
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    return set(data["project"].get("optional-dependencies", {}))


def advertised_extras() -> dict[str, list[str]]:
    """Every extra any source file tells a user to install, and where."""
    found: dict[str, list[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        for group in _EXTRA.findall(path.read_text(encoding="utf-8")):
            for name in group.split(","):
                found.setdefault(name.strip(), []).append(str(path.relative_to(REPO)))
    return found


class TestExtras:
    def test_the_sources_do_advertise_at_least_one(self):
        """A vacuous pass here would hide everything below it."""
        assert advertised_extras(), "no 'gausscapture[...]' install line found in src/"

    def test_every_extra_a_message_names_is_one_pip_can_resolve(self):
        declared = declared_extras()
        for name, sites in advertised_extras().items():
            assert name in declared, (
                f"{', '.join(sorted(set(sites)))} tells the user to install "
                f"'gausscapture[{name}]', which pyproject.toml does not declare; "
                f"pip warns and installs the bare package. Declared: {sorted(declared)}"
            )

    @pytest.mark.skipif(tomllib is None, reason="reading pyproject.toml needs tomllib (3.11+)")
    def test_the_training_extra_carries_the_things_the_trainer_imports(self):
        """torch runs the loop, gsplat is the only rasterizer we may ship, scipy is reported."""
        data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
        packages = " ".join(data["project"]["optional-dependencies"]["train4d"])
        assert "torch" in packages
        assert "gsplat" in packages
        assert "scipy" in packages

    def test_the_installed_package_agrees_with_the_declaration(self):
        """Otherwise the declaration is right and the thing on disk is not."""
        installed = set(metadata.metadata("gausscapture").get_all("Provides-Extra") or [])
        assert declared_extras() <= installed, (
            "reinstall the package (`pip install -e .`) -- its metadata predates "
            f"pyproject.toml. Declared {sorted(declared_extras())}, installed {sorted(installed)}"
        )


class TestTheCapturePackSchema:
    """The phone writes it, the desktop reads it; the schema is the only referee."""

    def test_the_schema_itself_is_a_valid_schema(self):
        jsonschema = pytest.importorskip("jsonschema", reason="jsonschema is in the dev extra")
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        jsonschema.validators.validator_for(schema).check_schema(schema)

    def test_the_manifest_this_package_writes_validates(self, tmp_path: Path):
        """`create_minimal_manifest` is what non-phone footage gets; it has to conform."""
        jsonschema = pytest.importorskip("jsonschema")
        from gausscapture.pack import manifest
        from gausscapture.types import VideoInfo

        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        payload = manifest.create_minimal_manifest(
            video_relpath="video/main_video.mp4",
            info=VideoInfo(
                duration_sec=4.0, width=1920, height=1080, fps=30.0, codec="h264"
            ),
            session_name="synthetic",
            target_type="person",
        )
        jsonschema.validate(payload, schema)

    def test_a_reconstructed_f_bullet_manifest_validates(self):
        """The four-phase manifest the capture app writes, as the schema sees it."""
        jsonschema = pytest.importorskip("jsonschema")
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        payload = {
            "capturepack_version": "0.3",
            "session_id": "9f1c1b1e-0000-4000-8000-000000000000",
            "session_name": "bullet time",
            "capture_type": "fixed_camera_4d",
            "target_type": "person",
            "created_at": "2026-08-07T00:00:00+00:00",
            "device": {
                "manufacturer": "Google", "model": "Pixel 8",
                "os": "Android 15", "app_version": "0.3",
            },
            "video": {
                "main_file": "video/main_video.mp4",
                "duration_sec": 6.4, "width": 1920, "height": 1080,
                "fps": 30.0, "codec": "h264", "has_audio": False,
            },
            "capture_settings": {
                "exposure_locked": True,
                "white_balance_locked": True,
                "focus_locked": True,
                "stabilisation_disabled": True,
                "storage_mode": "internal",
            },
            "protocol": {
                "name": "F_bullet",
                "fps": 30.0,
                "phases": [
                    {"name": "perch", "start_frame": 0, "end_frame": 29},
                    {"name": "arc", "start_frame": 30, "end_frame": 119},
                    {"name": "reseat", "start_frame": 120, "end_frame": 149},
                    {"name": "hold", "start_frame": 150, "end_frame": 191},
                ],
                "rest_window": {"start_frame": 140, "end_frame": 149},
                "auto_stopped": True,
            },
        }
        jsonschema.validate(payload, schema)

    def test_the_lock_keys_the_schema_publishes_are_the_ones_the_readers_use(self):
        """The divergence this file exists for.

        ``CaptureEngine.writeManifest()`` wrote ``exposure_lock_requested`` and
        friends; the schema names, and both Python readers consume,
        ``exposure_locked``. ``additionalProperties: true`` means the manifest
        validated anyway and the values were silently absent, so every take from
        the app was reported as having no locks at all.
        """
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        settings = schema["properties"]["capture_settings"]["properties"]
        for key in (
            "exposure_locked",
            "white_balance_locked",
            "focus_locked",
            "stabilisation_disabled",
        ):
            assert key in settings, key

        # The readers name the same keys, in source, not by coincidence.
        manifest_source = (SRC / "pack" / "manifest.py").read_text(encoding="utf-8")
        report_source = (SRC / "telemetry" / "report.py").read_text(encoding="utf-8")
        for key in ("exposure_locked", "white_balance_locked"):
            assert key in manifest_source, key
            assert key in report_source, key

    def test_the_android_app_writes_the_keys_the_schema_publishes(self):
        """The app is the only writer of this file, and it is not tested from Python.

        Reading its source is a weak check and a great deal better than none:
        the two sides had drifted apart for the whole life of the format, and
        nothing anywhere would have said so.
        """
        capture = (
            REPO / "app" / "android" / "app" / "src" / "main" / "java"
            / "com" / "gausscapture" / "capture"
        )
        if not capture.is_dir():
            pytest.skip("the Android app is not present in this checkout")
        source = "\n".join(p.read_text(encoding="utf-8") for p in sorted(capture.glob("*.java")))
        for key in (
            "exposure_locked",
            "white_balance_locked",
            "focus_locked",
            "stabilisation_disabled",
        ):
            assert f'"{key}"' in source, (
                f"the capture app writes no {key}; the desktop reads that key and will "
                "report the take as having no locks."
            )
        # And the manifest writer really does merge them in, rather than the
        # producer existing beside a caller that ignores it.
        engine = (capture / "CaptureEngine.java").read_text(encoding="utf-8")
        assert "achievedLocks()" in engine
