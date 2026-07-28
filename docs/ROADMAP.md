# Roadmap

Six months, twelve two-week sprints, calibrated to one MacBook Air M2 and a $0–35/month compute budget.
Derived from [RESEARCH.md](RESEARCH.md). Every sprint has a go/no-go criterion — a sprint that misses it changes the plan rather than sliding.

**Status:** S1–S3 complete, S4 next. Last updated 2026-07-27.

---

## Sequencing rationale

The original plan was v1 static polish → v2 capture intelligence → v3 static-camera 4D. That order is right in dependency terms and wrong in framing:

- **v1's job is not polish.** It is to become a headless, deterministic batch harness — a CLI that turns 180 capture packs into 180 reconstructions unattended, with pinned dependencies. Polishing the UI produces nothing citable; building the harness turns v2 into a data-collection run rather than a rewrite.
- **Two items move to v0, immediately.** JOSS requires more than six months of public repository history spanning real development; that clock is the binding constraint on the publication path and it started on 2026-07-26. JOSS also desk-rejects web tools lacking a core library, so extracting a pip-installable core is a v0 refactor, not a v3 cleanup.
- **v2 needs a native iOS capture app.** The dataset cannot be built from a PWA — Safari exposes no calibrated intrinsics, no ARKit pose, and no synchronised timestamps. This is budgeted inside v2, not deferred past it.
- **v3 is re-scoped.** Monocular static-camera 4D ships as a bounded demonstration; the *claimed* dynamic capability becomes a sparse two-to-four-phone mode, which converts an unwinnable hallucination problem into a solvable sparse-multiview one.

---

## v0 — Foundation (S1–S3)

### S1 · Repository as a research artifact
Public repository, MIT with REUSE-compliant SPDX headers, `CITATION.cff`, DCO, Zenodo concept DOI, `SUPPORT.md` declaring a bus factor of one.

**Go/no-go:** `reuse lint` clean; JOSS six-month clock started.

### S2 · Core library extraction
`gausscapture` becomes a pip-installable core with a CLI. FastAPI and React become thin clients over it. CPU smoke test in CI.

**Go/no-go:** `pip install gausscapture && gausscapture --help` works on macOS, Linux, and Windows; CI completes in under 10 minutes.

### S3 · `.capturepack` as a profile — **done**
BagIt layout with `manifest-sha256.txt` and `tagmanifest-sha256.txt`, a nerfstudio-convention `transforms.json`, a published JSON Schema, and a COLMAP model reader covering both the text and binary dialects. PWA timestamp epoch bug fixed; the PWA now writes bags too.

**Go/no-go — met.** `pack export --with-dataset` produces `data/images/`, `data/sparse/0/` and `data/transforms.json`; a trainer pointed at `data/` needs no conversion. Interop is asserted in CI against `bagit-python`, the reference implementation, which knows nothing about this project.

---

## v1 — Evaluation harness (S4–S5)

### S4 · Pose backends
`PoseBackend` protocol with COLMAP global mapper (adding the missing `--ImageReader.single_camera`) seeded by ARKit priors; MapAnything-apache as a second backend.

**Go/no-go:** ARKit seeding cuts COLMAP time by ≥30% across 10 packs with no registration regressions.

### S5 · Static reconstruction, correctly wired
gsplat MCMC with bilateral grid, replacing the currently broken external-trainer invocation. SPZ and SOG export via splat-transform. Spark viewer replaces the deprecated Three.js splat loaders. Drop `.ksplat`.

**Go/no-go:** Mip-NeRF 360 within 0.3 dB of gsplat's published 29.15 dB.

---

## v2 — Capture intelligence — *the paper* (S6–S8)

### S6 · On-device coach v1
Variance-of-Laplacian with a rolling adaptive threshold (replacing the current absolute `< 60`), gyroscope gate, highlight-clipping percentage, view-sphere bin fill. Brush wired as the documented Mac fallback.

**Go/no-go:** coach sustains ≥25 fps on an iPhone 13; a 30k-step Mac run completes in under 2.5 hours.

### S7 · Static capture campaign and correlation study
24 static scenes: 8 subjects × 3 deliberate quality tiers. Eight capture-time signals against final PSNR by Spearman ρ, plus logistic regression on registration failure.

**Pre-registered hypothesis:** VoL-p10 achieves |ρ| ≥ 0.40 and the logistic model achieves AUC ≥ 0.80.

**Go/no-go:** hypothesis holds → the paper is the capture-quality predictor. Hypothesis fails → the paper pivots to systems-only, and we publish the negative result.

### S8 · Prior bundle
Video-Depth-Anything-Small, TAPNext++, SAM 2.1, SEA-RAFT — all permissive tiers — cached to disk.

**Go/no-go:** full prior pass over 900 frames in under 8 minutes on a rented 4090.

---

## v3 — Dynamic, bounded (S9–S12)

### S9 · Background/foreground decomposition
Pre-roll SfM, frozen static background, SAM 2.1 foreground masks.

**Go/no-go:** background-only PSNR ≥ 26 dB across 5 tripod scenes.

### S10 · Motion scaffold and joint optimisation
SE(3) node trajectories, dual-quaternion skinning, scale-invariant Pearson depth loss, ARAP regularisation, density control disabled after 60% of iterations.

**Go/no-go:** N3DV under the MoDGS single-camera protocol ≥ 21.5 dB (MoDGS reports 22.64) → otherwise ship 3D-only and defer 4D.

### S11 · Dynamic capture campaign and dataset release
16 dynamic scenes with witness phones at ±25° and ±45° supplying genuine held-out novel views. Zenodo release with DOI; Codabench leaderboard.

**Go/no-go:** DyCheck masked mPSNR ≥ 18.5; dataset DOI resolves.

### S12 · Submission
JOSS paper (750–1750 words, with the mandatory AI-usage disclosure). arXiv endorsement secured.

**Go/no-go:** endorser confirmed; tagged `v0.5.0` with a Zenodo version DOI.

---

## Explicitly out of scope

| Item | Reason |
|---|---|
| Apple Silicon native training as a *goal* | 8–10× slower than a $0.34/h rental; bus factor 1. Brush stays a fallback. |
| Depending on Nerfstudio | Dormant ~12 months, 856 open issues. |
| Inria 3DGS or any repo vendoring its rasterizer | Non-commercial licence. |
| `.capturepack` as a proposed standard | Prior art covers it; demoted to a profile. |
| Splat editing and measurement | SuperSplat owns this at 9,716 stars. |
| Free-viewpoint orbit from a stationary phone | Physically impossible without a permissive generative prior. Cone is clamped and labelled instead. |

---

## Publication calendar

| When | Action |
|---|---|
| Now | Repository public — JOSS clock running |
| ~Sep 2026 | Secure an arXiv cs.CV endorser (unaffiliated authors need one since 2026-01-21) |
| ~Jan 2027 | MMSys'27 Open Dataset & Software track |
| ~Feb 2027 | JOSS submission (six-month history satisfied) |
| ~May 2027 | ACM MM'27 Open Source Software Competition |
