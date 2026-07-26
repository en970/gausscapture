from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


APP_NAME = "GaussCapture"
ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
DATA_DIR = BACKEND_DIR / "data"


class AppSettings(BaseModel):
    projects_dir: str = Field(default=str(DATA_DIR / "projects"))
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    colmap_path: str = "colmap"
    python_path: str = "python"
    gaussian_trainer_path: str = ""
    default_preview_port: int = 7860
    default_frame_preset: str = "balanced"
    default_training_preset: str = "draft"


def settings_file() -> Path:
    override = os.environ.get("GAUSSCAPTURE_SETTINGS_PATH")
    if override:
        return Path(override).expanduser()
    system = platform.system().lower()
    if system == "windows":
        base = Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
        return base / APP_NAME / "settings.json"
    if system == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME / "settings.json"
    return Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "gausscapture" / "settings.json"


def _dump_model(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def load_settings() -> AppSettings:
    path = settings_file()
    if not path.exists():
        settings = AppSettings()
        try:
            save_settings(settings)
        except PermissionError:
            fallback = DATA_DIR / "settings.json"
            fallback.write_text(json.dumps(_dump_model(settings), indent=2), encoding="utf-8")
        Path(settings.projects_dir).mkdir(parents=True, exist_ok=True)
        return settings
    data = json.loads(path.read_text(encoding="utf-8"))
    settings = AppSettings(**data)
    Path(settings.projects_dir).mkdir(parents=True, exist_ok=True)
    return settings


def save_settings(settings: AppSettings) -> None:
    path = settings_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_dump_model(settings), indent=2), encoding="utf-8")


settings = load_settings()
