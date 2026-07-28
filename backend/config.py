"""Server-side configuration.

Settings themselves live in :mod:`gausscapture.config` so the CLI, the library,
and this server all read the same file. This module only adds the paths the web
layer needs.
"""

from __future__ import annotations

from pathlib import Path

from gausscapture.config import (
    APP_NAME,
    Settings,
    get_settings,
    load_settings,
    save_settings,
    settings_file,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"

#: Uploads and job history live beside the projects directory so that pointing
#: `projects_dir` at an external drive moves the scratch space with it.
DATA_DIR = Path(get_settings().projects_dir).parent

#: Compatibility alias for existing imports.
AppSettings = Settings

__all__ = [
    "APP_NAME",
    "AppSettings",
    "BACKEND_DIR",
    "DATA_DIR",
    "ROOT_DIR",
    "Settings",
    "get_settings",
    "load_settings",
    "save_settings",
    "settings_file",
]
