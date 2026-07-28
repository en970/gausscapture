"""Settings resolution.

Settings live in a per-user config file so the same installed package serves
the GUI, the CLI, and a batch benchmark run without any of them hard-coding
paths. Environment variables win over the file, which is what makes CI runs and
containerised benchmarks reproducible without editing anything on disk.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

APP_NAME = "GaussCapture"

#: Repository root when running from a source checkout; used only for defaults.
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"


@dataclass
class Settings:
    projects_dir: str = str(DEFAULT_DATA_DIR / "projects")
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    colmap_path: str = "colmap"
    python_path: str = "python"
    gaussian_trainer_path: str = ""
    default_preview_port: int = 7860
    default_frame_preset: str = "balanced"
    default_training_preset: str = "draft"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def settings_file() -> Path:
    """Location of the settings JSON, per platform convention."""
    override = os.environ.get("GAUSSCAPTURE_SETTINGS_PATH")
    if override:
        return Path(override).expanduser()
    system = platform.system().lower()
    if system == "windows":
        base = Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
        return base / APP_NAME / "settings.json"
    if system == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME / "settings.json"
    xdg = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    return Path(xdg) / "gausscapture" / "settings.json"


_ENV_OVERRIDES = {
    "projects_dir": "GAUSSCAPTURE_PROJECTS_DIR",
    "ffmpeg_path": "GAUSSCAPTURE_FFMPEG",
    "ffprobe_path": "GAUSSCAPTURE_FFPROBE",
    "colmap_path": "GAUSSCAPTURE_COLMAP",
    "python_path": "GAUSSCAPTURE_PYTHON",
    "gaussian_trainer_path": "GAUSSCAPTURE_TRAINER",
}


def load_settings(create: bool = True) -> Settings:
    """Read settings, applying environment overrides.

    ``create`` is False in tests and read-only contexts so that merely importing
    the package never writes to a user's home directory.
    """
    path = settings_file()
    data: dict[str, Any] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # A corrupt settings file should not make the tool unusable.
            data = {}

    known = {f.name for f in fields(Settings)}
    settings = Settings(**{k: v for k, v in data.items() if k in known})

    for attr, env in _ENV_OVERRIDES.items():
        value = os.environ.get(env)
        if value:
            setattr(settings, attr, value)

    if create and not path.exists():
        # A read-only home directory is survivable; the defaults still work.
        with contextlib.suppress(OSError):
            save_settings(settings)
    return settings


def save_settings(settings: Settings) -> None:
    path = settings_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings.to_dict(), indent=2), encoding="utf-8")


def get_settings() -> Settings:
    """Convenience accessor; reads fresh so env changes take effect in tests."""
    return load_settings(create=False)
