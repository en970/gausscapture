# GaussCapture

**Turn a phone video into a 3D Gaussian splat, entirely on your own machine — and know whether the capture was good enough *before* you spend an hour training it.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Project site](https://img.shields.io/badge/site-en970.github.io%2Fgausscapture-8a4f2d)](https://en970.github.io/gausscapture/)

**[en970.github.io/gausscapture](https://en970.github.io/gausscapture/)** — project page, with the argument and the numbers.

---

## Status

**Early, and honest about it.** The pipeline below runs; the research programme it is being rebuilt around is documented and just beginning. This repository is public from the start so its development history is visible — including the parts that do not work yet.

| | |
|---|---|
| Works today | `pip`-installable core library and CLI, video import, CapturePack archive with checksum verification, capture telemetry, frame extraction, COLMAP wrapper, trainer-ready dataset assembly, Colab export, trained-model import, web preview, export bundles, mobile capture PWA |
| Still unvalidated | The quality score's weights are a documented heuristic, not a fitted predictor. Establishing whether these signals actually predict reconstruction quality is [the research programme](docs/RESEARCH.md#7-the-contribution), not a solved problem. |
| Not supported | Dynamic scenes. See [why](docs/RESEARCH.md#5-the-4d-question-answered-honestly). |

Recently fixed, and worth naming because they were silent: the trainer was handed a dataset layout no trainer accepts, so local training had never worked; COLMAP ran without `--ImageReader.single_camera` on single-camera video; blur rejection used an absolute Laplacian threshold on a resolution- and content-dependent measure; and the capture PWA stamped IMU samples and video start with different clock epochs, making the sensor logs unalignable.

---

## Why this exists

Every phone-capture app tells you your scan failed *after* it fails. Coverage grids and "move slower" prompts are qualitative; none of them emit a number that predicts how good your reconstruction will be.

The individual signals are well studied. Motion-blur compensation is worth **+1.68 PSNR** on real phone captures. Correcting auto-exposure drift is worth up to **+5.3 dB**. But nobody has composed those signals into a predictor, and no public dataset pairs capture-time telemetry with reconstruction outcomes.

That is the open problem this project is aimed at:

> **Can signals measured on your phone during capture predict the quality of a reconstruction that has not been trained yet?**

Everything else here — the pipeline, the container format, the viewer — exists to make that question answerable and the answer reproducible.

---

## What makes this different

- **Local-first.** Your footage stays on your machine. Most competitors moved reconstruction to the cloud; one of them states that mapping the world for machines is the point.
- **Licence-clean.** Almost every Gaussian-splatting research repository is non-commercially encumbered, often while declaring a permissive licence. GaussCapture uses a fully permissive stack so you can ship what you make. The cost is about 0.6 dB. See [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md).
- **Honest about limits.** Where the physics says no, this project says no and shows the numbers.

---

## The library and CLI

Everything the GUI does is reachable from Python and from the command line, with
no server running. That is what makes unattended batch evaluation possible — and
it is the part under active development.

```bash
pip install -e .

gausscapture doctor                      # what's installed, what's missing
gausscapture import capture.mp4 --name "living room"
gausscapture telemetry <project>         # capture-quality report
gausscapture frames <project> --preset balanced
gausscapture pose <project>              # COLMAP
gausscapture dataset <project>           # images/ + sparse/0/, ready for a trainer
gausscapture run <project>               # all of the above in one pass
```

Progress goes to stderr and data to stdout, so reports pipe cleanly:

```bash
gausscapture telemetry <project> --json | jq '.vol_p10, .score'
```

```python
from gausscapture import ProjectStore
from gausscapture.telemetry import analyze_capture

report = analyze_capture(ProjectStore().list()[0].path)
print(report.vol_p10, report.blurry_frame_ratio, report.score)
```

## Quick start

Requirements: Python 3.10–3.12, Node.js 18+, `ffmpeg` and `ffprobe` on `PATH`, and COLMAP for pose estimation.

```bash
brew install ffmpeg colmap      # macOS
```

```bash
git clone https://github.com/en970/gausscapture
cd gausscapture
./install_macos.sh
./start_macos.sh
```

Backend on `http://localhost:7860`, frontend on `http://localhost:3000`. Windows: `install_windows.bat` then `start_windows.bat`.

### Capturing from a phone

The backend serves a dependency-free PWA at `http://localhost:7860/mobile/`. To reach it from a phone on the same network:

```bash
GAUSSCAPTURE_HOST=0.0.0.0 ./start_macos.sh
```

This serves **HTTPS with a self-signed certificate**, generated on first run. That is not optional: browsers grant camera access only in a [secure context](https://developer.mozilla.org/en-US/docs/Web/Security/Secure_Contexts) — HTTPS or `localhost` — so a plain-HTTP LAN address cannot open the camera at all, no matter what the page does. The phone will warn once about the certificate; accept it (iOS Safari: *Show Details → visit this website*; Android Chrome: *Advanced → Proceed*).

Browsers cannot expose calibrated intrinsics or ARKit poses, so COLMAP is still required. A native capture app is on the roadmap and is a prerequisite for the research programme.

If you only need footage, the phone's own camera app plus AirDrop is simpler and gives better video — the PWA exists to capture *sensor data alongside* the video, not to replace the camera app.

### Training

Local training needs a CUDA GPU and an external trainer configured via `gaussian_trainer_path` in settings. **This path is currently broken** (wrong dataset layout — being fixed in S5). Until then, use the Colab export: create a Colab package, download `dataset.zip`, and run [`notebooks/GaussCapture_Colab_Trainer.ipynb`](notebooks/GaussCapture_Colab_Trainer.ipynb).

On Apple Silicon there is no CUDA path. [Brush](https://github.com/ArthurBrussee/brush) trains on Metal at roughly 2 hours for a 30k-step scene; renting an RTX 4090 costs about $0.34/hour, which makes a 30-minute run cost $0.17. We recommend renting.

---

## Documentation

| Document | Contents |
|---|---|
| [**RESEARCH.md**](docs/RESEARCH.md) | The technical report: competitive landscape, what is licensed, why stationary-phone 4D is physically bounded, the proposed contribution, evaluation protocol |
| [**ROADMAP.md**](docs/ROADMAP.md) | Twelve sprints with go/no-go criteria, and what is explicitly out of scope |
| [**EXPERIMENTS.md**](docs/EXPERIMENTS.md) | What was actually run and what it showed, negative results included |
| [**DEPENDENCIES.md**](docs/DEPENDENCIES.md) | Bundled vs. invoked classification; the repositories that declare permissive licences while shipping encumbered code |
| [CAPTUREPACK_SPEC.md](docs/CAPTUREPACK_SPEC.md) | Container format (being migrated to a BagIt profile) |
| [TRAINING_PIPELINE.md](docs/TRAINING_PIPELINE.md) · [EXPORT_FORMATS.md](docs/EXPORT_FORMATS.md) | Pipeline stages and export targets |
| [SUPPORT.md](SUPPORT.md) | Maintenance status and how to get help |

---

## On dynamic scenes

The long-term goal is a stationary phone reconstructing a moving subject. The research says this is bounded by physics rather than engineering: with zero camera parallax, the occluded side of the subject is **not in the recording**. Published results reach roughly 19–22 dB masked PSNR inside a ±30° view cone; beyond that, methods hallucinate, and every technique that fills the gap depends on a multi-view video diffusion prior that is unreleased, non-commercial, or needs 32 GB200s to train.

So we will ship a **bounded bullet-time card** — static background plus dynamic foreground, with the viewer hard-clamped to the trained ±25° cone and the cone labelled in the interface — and pursue a **sparse two-to-four-phone mode** as the tractable route to genuine novel views.

Full argument with numbers: [`docs/RESEARCH.md §5`](docs/RESEARCH.md#5-the-4d-question-answered-honestly).

---

## Contributing

Issues and pull requests are welcome. Contributions are accepted under the [DCO](https://developercertificate.org/) — sign off commits with `git commit -s`.

The most useful contributions right now are captures: if you record a scene and the reconstruction fails, that failure is data.

---

## Citing

See [CITATION.cff](CITATION.cff). A DOI will be minted at the first tagged release.

## Licence

MIT — see [LICENSE](LICENSE). Copyright © 2026 Enes Öz.

Third-party components are catalogued in [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md) with their licences and whether they are bundled or invoked.

## Acknowledgement of AI assistance

The research survey and much of the documentation in this repository were produced with substantial generative-AI assistance, disclosed in [`docs/RESEARCH.md §12`](docs/RESEARCH.md#12-disclosure) in accordance with JOSS's 2026 policy.
