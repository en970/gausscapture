# GaussCapture

**Turn a phone video into a 3D Gaussian splat, entirely on your own machine — and know whether the capture was good enough *before* you spend an hour training it.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Status

**Early, and honest about it.** The pipeline below runs; the research programme it is being rebuilt around is documented and just beginning. This repository is public from the start so its development history is visible — including the parts that do not work yet.

| | |
|---|---|
| Works today | Video import, CapturePack archive, OpenCV quality analysis, frame extraction, COLMAP wrapper, Colab dataset export, trained-model import, web preview, export bundles, mobile capture PWA |
| Known broken | Local training invocation writes the wrong dataset layout — it cannot succeed against a standard trainer. COLMAP runs without `--ImageReader.single_camera`. Quality thresholds are unvalidated constants. The PWA stamps IMU and video with different clock epochs, so sensor logs cannot be aligned to frames. |
| Not supported | Dynamic scenes. See [why](docs/RESEARCH.md#5-the-4d-question-answered-honestly). |

Every item in "known broken" is diagnosed in [`docs/RESEARCH.md §2`](docs/RESEARCH.md) and scheduled in [`docs/ROADMAP.md`](docs/ROADMAP.md).

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

Browsers cannot expose calibrated intrinsics or ARKit poses, so COLMAP is still required. A native capture app is on the roadmap and is a prerequisite for the research programme.

### Training

Local training needs a CUDA GPU and an external trainer configured via `gaussian_trainer_path` in settings. **This path is currently broken** (wrong dataset layout — being fixed in S5). Until then, use the Colab export: create a Colab package, download `dataset.zip`, and run [`notebooks/GaussCapture_Colab_Trainer.ipynb`](notebooks/GaussCapture_Colab_Trainer.ipynb).

On Apple Silicon there is no CUDA path. [Brush](https://github.com/ArthurBrussee/brush) trains on Metal at roughly 2 hours for a 30k-step scene; renting an RTX 4090 costs about $0.34/hour, which makes a 30-minute run cost $0.17. We recommend renting.

---

## Documentation

| Document | Contents |
|---|---|
| [**RESEARCH.md**](docs/RESEARCH.md) | The technical report: competitive landscape, what is licensed, why stationary-phone 4D is physically bounded, the proposed contribution, evaluation protocol |
| [**ROADMAP.md**](docs/ROADMAP.md) | Twelve sprints with go/no-go criteria, and what is explicitly out of scope |
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
