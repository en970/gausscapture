# GaussCapture MVP

GaussCapture converts phone videos or `.capturepack` archives into a local 3D Gaussian Splatting preparation workflow. It runs on localhost with a FastAPI backend and React/Three.js frontend.

## MVP Scope

- Static rooms, static interiors, static objects, and phone video captures.
- Minimal CapturePack creation from `.mp4` / `.mov`.
- CapturePack validation/import with metadata warnings.
- OpenCV quality analysis and frame extraction.
- COLMAP subprocess wrapper for camera pose generation.
- External 3DGS trainer wrapper via configurable path.
- Colab dataset package export.
- Trained `.ply`, `.splat`, `.ksplat`, or result `.zip` import.
- Web preview and export zip generation.

This MVP does not support 4D or dynamic-scene reconstruction.

## Requirements

- Python 3.10 or 3.11
- Node.js 18+
- ffmpeg and ffprobe on PATH
- COLMAP for local SfM pose generation
- CUDA GPU and a compatible external Gaussian Splatting trainer for local training
- Open3D is optional for experimental proxy mesh export: `python -m pip install -r requirements-optional.txt`

## macOS Install

```bash
cd gausscapture
chmod +x install_macos.sh start_macos.sh
./install_macos.sh
./start_macos.sh
```

The backend runs at `http://localhost:7860`. The Vite frontend runs at `http://localhost:3000` and proxies API calls to the backend.

## Simple Phone Capture App

The backend also serves a dependency-free mobile PWA:

```text
http://localhost:7860/mobile/
```

For a phone on the same Wi-Fi, start the backend on your LAN interface:

```bash
GAUSSCAPTURE_HOST=0.0.0.0 ./start_macos.sh
```

Then open `http://<desktop-lan-ip>:7860/mobile/` on the phone. The app can:

- record a simple environment-facing camera video,
- collect basic `DeviceMotionEvent` and orientation logs when the browser allows it,
- download a `.capturepack` directly on the phone,
- upload the recorded video to an existing or newly created desktop project.

This first phone app is intentionally minimal. Browser APIs do not expose calibrated intrinsics or ARKit/ARCore camera poses, so COLMAP is still required unless a future native app adds those logs.

Install ffmpeg and COLMAP with Homebrew:

```bash
brew install ffmpeg colmap
```

## Windows Install

Run these from Command Prompt:

```bat
cd gausscapture
install_windows.bat
start_windows.bat
```

Install ffmpeg and COLMAP separately and make sure `ffmpeg`, `ffprobe`, and `colmap` are available on PATH.

## Settings

Settings are created on first run:

- Windows: `%APPDATA%/GaussCapture/settings.json`
- macOS: `~/Library/Application Support/GaussCapture/settings.json`
- Linux: `~/.config/gausscapture/settings.json`

Important fields:

```json
{
  "projects_dir": "...",
  "ffmpeg_path": "ffmpeg",
  "ffprobe_path": "ffprobe",
  "colmap_path": "colmap",
  "python_path": "python",
  "gaussian_trainer_path": "",
  "default_preview_port": 7860
}
```

Set `gaussian_trainer_path` to an external 3D Gaussian Splatting implementation folder that contains a compatible `train.py`. If it is blank, local training fails clearly but import, quality, preprocessing, Colab package, preview, and export continue to work.

## Demo Flow

1. Open `http://localhost:3000`.
2. Create a project.
3. Import a `.mp4` or `.mov`; GaussCapture creates a minimal CapturePack.
4. Run Quality Analyze.
5. Extract frames in Preprocess.
6. Run COLMAP if installed, or continue with Colab packaging.
7. Create a Colab package and download `dataset.zip`.
8. Import a trained `.ply`, `.splat`, `.ksplat`, or result `.zip`.
9. Open Preview and inspect the model.
10. Create a Web Viewer Bundle export.

## Colab Training

Use `notebooks/GaussCapture_Colab_Trainer.ipynb`. Upload `dataset.zip`, select a CUDA GPU runtime, configure or clone your 3DGS trainer, run training, then zip and download the output. TPU is not targeted.

## Export Options

- Raw model zip: model plus metadata.
- Web viewer bundle: static `index.html`, model, and metadata.
- Blender export: model plus import notes.
- Unity package: model folder and loader placeholder. A Unity splat renderer integration is required.
- Proxy mesh: experimental Open3D Poisson reconstruction from `.ply`.

## Known Limitations

- Static scenes only; 4D/dynamic scenes are not supported.
- Local Gaussian training requires a configured external trainer and CUDA GPU.
- COLMAP must be installed for local SfM when pose metadata is missing.
- `.splat` preview uses a minimal point fallback for common binary splat layout.
- Mesh export is experimental and may fail on sparse/noisy point clouds.
- Splat-to-clean-mesh conversion is not guaranteed.
