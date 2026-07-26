# Score Before You Scan

**A technical report on phone-based Gaussian splatting: what is achievable, what is licensed, and where the open problem actually is**

Enes Öz · `enesozile@gmail.com` · [github.com/en970/gausscapture](https://github.com/en970/gausscapture)
Report version 1.0 · 2026-07-26

---

## Abstract

Phone-based 3D Gaussian splatting fails for well-understood reasons — motion blur, rolling shutter, auto-exposure drift, insufficient parallax, textureless geometry — yet every shipping capture application communicates these qualitatively, if at all. This report surveys the field as of July 2026 and reaches three conclusions that redirect the GaussCapture project.

First, the capture side of the pipeline is scientifically underexplored while the reconstruction side is crowded. No public dataset pairs capture-time telemetry with downstream reconstruction outcomes, and no shipping tool emits a number predictive of final quality before training begins. This is the project's defensible contribution.

Second, the stated long-term goal — a stationary phone reconstructing a moving subject in orbitable 4D — is bounded by a physical limit rather than an engineering one. With zero camera parallax, the occluded side of the subject is not present in the recording. The published ceiling for this regime is approximately 19–22 dB masked PSNR within a ±20–30° view cone; beyond that, methods hallucinate. Every technique that pushes past the cone depends on a multi-view video diffusion prior, and no permissively licensed such prior exists. We therefore scope 4D as a bounded "bullet-time card" demonstration rather than a headline claim.

Third, the monocular-dynamic literature is almost entirely unusable in a commercially permissive project. Every released method is encumbered through at least one of three chokepoints — the Inria rasterizer, CoTracker3, or UniDepth — and no author has removed all three. Producing the first fully permissive substitution is itself a citable artifact.

We propose a revised architecture, a pre-registered evaluation protocol, and a six-month milestone plan calibrated to a single Apple Silicon laptop and a $0–35 monthly compute budget.

---

## 1. Introduction

### 1.1 Starting point

GaussCapture began as a localhost pipeline: import a phone video, wrap it in a `.capturepack` archive, analyse quality with OpenCV, extract frames, run COLMAP, hand off to an external Gaussian splatting trainer, import the trained model, preview it in Three.js, export it. Roughly 2,500 lines across a FastAPI backend, a React frontend, and a dependency-free mobile PWA. Static scenes only.

The stated direction is twofold: near term, walk around a room or object with a phone and obtain a good splat; next step, capture a *moving* subject — including the case where the phone is stationary on a table and the subject moves.

This report asks what that costs, what it is worth, and what is already taken.

### 1.2 Method

Twelve research agents conducted 437 distinct web queries across nine axes: the competitive landscape, open-source training stacks, pose estimation, dynamic/4D methods, monocular prior models, capture-quality science, free compute and hosting, interchange formats, and open-source research norms. Three further agents performed adversarial synthesis: a dependency licensing audit that re-verified every claim against live `LICENSE` files, a novelty analysis written from a hostile reviewer's perspective, and an architecture and evaluation design.

Every factual claim below carries a source. Claims that could not be verified are marked. Repository health figures (last commit, open issues, stars) were fetched live from the GitHub API on 2026-07-26 and will age.

### 1.3 An honest note on what changed

The research overturned two assumptions that were embedded in the project's original direction, and confirmed the value of one that was not yet articulated:

- **Overturned:** that local training on Apple Silicon is a viable target. It is not, and pursuing it would consume the project.
- **Overturned:** that `.capturepack` is a novel contribution. It is a useful engineering convenience and a poor research claim.
- **Confirmed and elevated:** that capture-time quality analysis — already present in the codebase as `quality_analyzer.py` — sits on the one genuinely open problem in this space.

---

## 2. Audit of the current implementation

Before surveying the field, we state what the existing code does and where it is wrong. These are not stylistic objections; each affects correctness.

| Location | Finding | Consequence |
|---|---|---|
| `backend/core/gaussian_runner.py:60` | Invokes `train.py -s <project_path>`, but the Inria/gsplat convention expects `<source>/images` and `<source>/sparse/0`. The project writes frames to `frames/images` and COLMAP output to `colmap/sparse`. | The local training path cannot succeed against any standard trainer. It has almost certainly never run end to end. |
| `backend/core/colmap_runner.py:38` | Feature extraction omits `--ImageReader.single_camera 1` and does not set a camera model. All frames come from one phone camera with fixed intrinsics. | COLMAP solves an independent intrinsic set per image. Slower, less stable, and materially worse on low-parallax sequences. |
| `backend/core/quality_analyzer.py:88-110` | Thresholds are hardcoded magic numbers: blur `< 80`, brightness `< 45`, frame difference `> 35`, duplicate correlation `> 0.992`. The composite score subtracts fixed weights. | No empirical basis. The score is presentable but not predictive — which is precisely the gap this report proposes to close. |
| `backend/core/frame_extractor.py:59` | Rejects frames on absolute variance-of-Laplacian `< 60`. VoL scales with resolution, contrast, and content. | A sharp frame of a textureless wall scores lower than a blurred frame of a bookshelf. Thresholds must be relative to a rolling window. |
| `mobile/app.js:169,113` | IMU samples are stamped with `performance.now()`; recording start is stamped with `Date.now()`. Different epochs, no synchronisation anchor. | Recorded motion data cannot be aligned to video frames. The sensor logs are currently unusable for anything but coarse statistics. |
| repository-wide | No tests, no CI, no dependency pinning. | Blocks the reproducibility claims in §8 and the publication path in §10. |

The mobile PWA's hand-rolled ZIP writer (`mobile/app.js:270-350`) is, by contrast, correct and dependency-free — worth keeping.

---

## 3. Competitive landscape

### 3.1 Consumer capture applications

| Product | Method | Processing | 4D? | Open? |
|---|---|---|---|---|
| Polycam | Photogrammetry + LiDAR + 3DGS | Cloud for splats | No | No |
| Scaniverse (Niantic Spatial) | 3DGS + mesh | Classic on-device; 2026 platform is cloud | No | No (SPZ codec is open) |
| KIRI Engine | Photogrammetry/NeRF/3DGS | Cloud | Capture no; Blender 4DGS sequence export yes | Plugin only |
| RealityScan Mobile 1.8 (Epic) | Photogrammetry | Cloud | No | No |
| Luma 3D Capture | NeRF + 3DGS | Cloud | No | No |
| Postshot (Jawset) | 3DGS/NeRF | Fully local, Windows only | No | No (free-as-in-beer) |
| Gracia / 4DV.ai | 4DGS | Cloud | Yes — but 60-camera rigs | No |

Two 2026 iOS newcomers occupy territory adjacent to GaussCapture. **SplatKing** exports native COLMAP text models, a `splatpack.json`, dual-lens capture, RAW, and manual ISO/shutter/WB control ([App Store](https://apps.apple.com/us/app/gaussian-splatking/id6759175085)). **SplatCam** exports full intrinsics and extrinsics plus a tracking-quality indicator with motion warnings, loop-closure guidance, and blur rejection ([App Store](https://apps.apple.com/us/app/splatcam-lidar-capture/id6759800588)). Both are hobby-scale — 12 and 4 ratings respectively — closed source, and neither does anything dynamic.

### 3.2 Adversarial reading of the incumbents

- **Luma AI is in maintenance.** Genie was sunset 1 January 2026; the capture app last shipped 14 January 2026. Company revenue has moved to video generation. Do not build against their API.
- **Abound (formerly Metascan) appears abandoned** — last release 7 April 2025, with user reports of jobs stuck in queue and splat processing broken since the Gaussian update.
- **Scaniverse is drifting from on-device to cloud.** The historical privacy claim was that data never leaves the device; the April 2026 platform relaunch routes reconstruction through Niantic's cloud, and Niantic Spatial's stated purpose is mapping the world for machines ([Niantic Spatial](https://www.nianticspatial.com/blog/scaniverse)). Captures become training data for their visual positioning system.
- **Polycam's splat work appears stalled** — release notes for February–March 2026 mention HDRI relighting, floor plans, and sync; nothing about Gaussian splatting, video, or 4D.

### 3.3 The capture-guidance gap

Polycam shows a blue coverage grid and warns when motion is too fast. RealityScan 1.8 ships AR guidance with colour-coded coverage. SplatCam shows a tracking-quality meter. Scaniverse warns above 180 seconds.

**Nobody publishes a quantitative coverage or parallax score, and nobody emits a number that predicts final reconstruction quality.** Guidance is qualitative everywhere. This is the exploitable gap, and GaussCapture already has the scaffolding for it.

---

## 4. The reconstruction stack: what is usable

### 4.1 Trainers

| Project | License | Stars | Last commit | Verdict |
|---|---|---|---|---|
| Inria 3DGS | **Non-commercial research** | 22,785 | 2024-10-30 | License-incompatible. Never vendor. |
| [gsplat](https://github.com/nerfstudio-project/gsplat) | Apache-2.0 | 5,452 | 2026-07-24 | **The trainer to depend on.** |
| Nerfstudio | Apache-2.0 | 11,848 | 2025-07-29 | **Dormant ~12 months**, 856 open issues. Do not build on it. |
| [Brush](https://github.com/ArthurBrussee/brush) | Apache-2.0 | 4,865 | 2026-07-01 | Active, bus factor 1. Mac fallback only. |
| OpenSplat | AGPL-3.0 | 2,113 | 2026-05-31 | Copyleft; subprocess only. |
| LichtFeld Studio | GPL-3.0 | 3,442 | 2026-07-26 | Fastest 2026 native trainer; GPL blocks linking. |

gsplat vendors the MCMC densification strategy and PNG compression internally, which means the entire family of algorithmic variants (Mip-Splatting, 2DGS, Scaffold-GS, 3DGS-MCMC, Taming-3DGS) — all research-licensed and mostly frozen — should be treated as papers to read, not dependencies to install.

Reference numbers: gsplat on Mip-NeRF 360, 30k iterations, PSNR **29.15**, 6.31 GB peak, 872 s ([ROCm benchmark](https://rocm.docs.amd.com/projects/gsplat/en/latest/reference/benchmark-evaluation.html)). PNG compression takes 1M Gaussians from 236 MB to 16.5 MB for −0.53 dB.

**The critical implication of §4.1 for this project's original direction:** GaussCapture's Colab export currently targets a generic `train.py`. It must target `gsplat/examples/simple_trainer.py` specifically, and must never install the Inria reference implementation.

### 4.2 Apple Silicon: a negative result

This matters because the project has exactly one machine, a MacBook Air M2.

- **gsplat has no Metal or MPS support.** Issue [#163](https://github.com/nerfstudio-project/gsplat/issues/163) has been open since April 2024. Pull request [#832](https://github.com/nerfstudio-project/gsplat/pull/832), "Add Apple Metal (MPS) support", was **closed 42 seconds after it was opened**.
- **Brush is the only credible end-to-end Mac trainer.** A documented real-world run: 30,000 steps at 4M splats took **2 hours 5 minutes** on Apple Silicon, with COLMAP adding 18:30 for 63 frames ([rodrigopolo, 2026-03-15](https://rodrigopolo.com/2026/03/15/gaussian-splatting-in-macos-100-free/)).
- `msplat` claims 30k iterations in 700 s on an M4 Max, but the repository has 46 stars, one contributor, and has been idle since March 2026. Unreproduced.
- `splat-apple` (MLX) has **no license file** — legally unusable.

**Conclusion: Apple Silicon native training is not a target.** It is roughly 8–10× slower than a CUDA run of the same job, and the maintained path has a bus factor of one. A RunPod Community RTX 4090 costs **$0.34/hour**; a 30-minute training run costs **$0.17**. Spending six months optimising a Metal rasterizer to save seventeen cents is not a defensible allocation of effort. Brush ships as a documented local sanity-check fallback, never as a claim.

### 4.3 Viewers and formats

[Spark](https://github.com/sparkjsdev/spark) (MIT, World Labs, 3,442 stars, active) is the de-facto three.js path — three.js core has zero built-in splat support — and is **the only maintained viewer with a real 4D path**, rendering dynamic sequences by interpolating between scanned frames. This directly de-risks the dynamic goal on the display side. The project's current viewer dependencies, `mkkellogg/GaussianSplats3D` and `antimatter15/splat`, are both effectively archived and both now point users to Spark.

Formats: `.spz` (Niantic, MIT) is ~10× smaller than PLY; `.sog` (PlayCanvas) is 15–20× smaller, taking 4M Gaussians from 1 GB to 42 MB. `.ksplat` is tied to a deprecated viewer and should be dropped from GaussCapture's export list. The converter of record is [splat-transform](https://github.com/playcanvas/splat-transform) (MIT, active).

---

## 5. The 4D question, answered honestly

This is the section that matters most for the project's stated direction.

### 5.1 The regime

A phone on a tripod with a moving subject is a monocular, **zero-parallax**, dynamic reconstruction problem. Structure-from-motion is not merely difficult here; it is undefined. Every COLMAP-dependent dynamic method fails at step zero. The only viable family is monocular depth → 2D point tracks → deformation field.

### 5.2 What the literature achieves

| Method | Regime | Cost | Masked mPSNR (DyCheck) |
|---|---|---|---|
| Shape of Motion (ICCV'25, MIT) | Monocular casual | hours | 17.32 |
| MoSca (CVPR'25, MIT core) | Monocular casual | — | 19.32 |
| **World from Motion (Jul'26, SOTA)** | Monocular | **48 h on 32× GB200** | **19.96** |
| MoDGS (ICLR'25) | **Static camera — the only method naming this setting** | 3.5 h A6000, 14 GB | 22.64 on DyNeRF cam0-train |

Two years of work moved the state of the art on this benchmark by about 2.6 dB, to roughly 20 dB — which is **visibly blurry**.

A caution that must be carried into any future comparison: Instant4D reports 24.52 dB on DyCheck, but this is *not* the covisibility-masked protocol, and its comparison table omits Shape of Motion and MoSca entirely. Masked and unmasked numbers must never appear in the same table.

### 5.3 The physical ceiling

MoDGS's reported 22.64 dB is measured against test cameras only **tens of centimetres** off the training axis. ExpanDyNeRF, the only work that quantifies off-axis extrapolation, evaluates at ±30° azimuth and reports **FID 142.7** on synthetic data — a value that reads as unambiguously synthetic.

Beyond roughly ±30°, these methods are not reconstructing. They are hallucinating, because the subject's back, sides, and the floor they occlude are information **physically absent from a zero-parallax recording**.

### 5.4 The blocker

The only things that fill disoccluded regions are multi-view video diffusion priors. Their licensing status as of July 2026:

- **CAT4D** (Google, CVPR'25) — no public code.
- **Stable Virtual Camera** — non-commercial licence.
- **TrajectoryCrafter** — academic use only.
- **Lift4D** — unreleased.
- **World from Motion** — academic licence, 32× GB200 to train.

ViewCrafter and DimensionX are Apache-2.0 but are *static-scene* view synthesisers, not 4D.

**There is no permissively licensed multi-view video diffusion prior. This is the single blocker, and it is not one a solo project can remove.**

### 5.5 There is also no benchmark

| Dataset | Truly static-camera monocular? |
|---|---|
| DyCheck iPhone | No — handheld moving, merely low-parallax |
| N3DV / DyNeRF | Only under MoDGS's train-on-cam0 protocol |
| NVIDIA Dynamic Scenes | Same trick, train cam4 |
| HyperNeRF / NeRF-DS / D-NeRF | No |

**No public benchmark exists for "phone on a table, subject moves."** Quality claims in this regime currently cannot be validated by anyone.

### 5.6 What we will actually ship

A **4D bullet-time card**: static-background plus dynamic-foreground decomposition, 30–60 second clips, a **±25° view cone that the viewer hard-clamps the camera to**, approximately 8 GB VRAM and 12–25 minutes of training. Expected quality 18.5–20.0 dB masked PSNR, LPIPS 0.26–0.30.

We will label the cone in the interface rather than letting a user orbit into hallucinated geometry and conclude the tool is broken. This is a demonstration and a limitations section — not the paper's claim.

**The tractable alternative.** 4C4D (CVPR'26) demonstrates that four portable cameras suffice for this class of scene. A "two to four phones on tripods" mode reuses the existing PWA and the existing container, and converts an unwinnable hallucination problem into a solvable sparse-multiview one. Note that 4C4D's own repository, despite declaring MIT, ships the Inria rasterizer and builds on MASt3R (CC-BY-NC-SA) — the *approach* is adoptable, the *code* is not.

---

## 6. Licensing: the field is more encumbered than it appears

This section is the most actionable output of the research, because several widely cited repositories declare a permissive licence while shipping encumbered code.

### 6.1 Verified corrections

| Common belief | Verified reality |
|---|---|
| MegaSaM is Apache-2.0 and safe | Apache-2.0 code, but **vendors a `UniDepth` subdirectory (CC-BY-NC-4.0)** |
| hustvl/4DGaussians is Apache-2.0 | Repository is Apache-2.0; the **rasterizer is an Inria fork** |
| 4C4D is MIT | Declares MIT; **ships `diff-gaussian-rasterization` and builds on MASt3R (CC-BY-NC-SA)** |
| MoSca is MIT | MIT covers `lib_moca`/`lib_mosca` only; **`lib_prior` carries CoTracker3, UniDepth, DepthCrafter** |
| Depth Anything 3: only Giant tiers are NC | **DA3-Large is also CC-BY-NC-4.0** |

### 6.2 The three chokepoints

Every released monocular-4D method is encumbered through at least one of:

1. **The Inria rasterizer** (`diff-gaussian-rasterization`) — non-commercial. Inherited by MoDGS, 4C4D, hustvl/4DGaussians, Dynamic3DGaussians, SpacetimeGaussians, Ex4DGS, InstantSplat.
2. **CoTracker3** — CC-BY-NC. Inherited transitively via Shape-of-Motion and MoSca prior bundles.
3. **UniDepth** — CC-BY-NC. The most-missed transitive taint, arriving inside MegaSaM.

**No author has removed all three.** Producing the substitution is the project's second defensible contribution.

### 6.3 The permissive-only reference stack

| Stage | Permissive choice | Quality cost vs. best available |
|---|---|---|
| Pose | COLMAP 3.13 (BSD-3), MapAnything-apache seed | Latency, not accuracy |
| Static 3DGS | **gsplat** (Apache-2.0) | −0.2…0.5 dB vs LichtFeld (GPL) |
| Mono depth | DA3-Metric-Large / VDA-Small (Apache-2.0) | ~0.3–0.6 dB |
| Point tracking | TAPNext++ (Apache-2.0) / AllTracker (MIT) | **≈0 — the cheapest swap in the stack** |
| Segmentation | SAM 2.1 (Apache-2.0), EdgeTAM on-device | Capability, not fidelity |
| Dynamic/4D | Shape-of-Motion core (MIT) re-hosted on gsplat + TAPNext + DA3 | ~0.6 dB below SOTA |
| Viewer / export | Spark, SPZ, SOG, splat-transform (all MIT) | Zero |

The aggregate cost of full licence cleanliness is roughly **0.6 dB**. That is a price worth paying for a tool whose users can legally ship its output.

### 6.4 Enforcement

Five traps and their mitigations:

1. Any repository vendoring `diff-gaussian-rasterization` → CI gate failing the build if `diff_gaussian_rasterization`, `simple-knn`, or `LICENSE_gaussian_splatting*` appears in the resolved tree.
2. CoTracker3 weights arriving transitively → vendor only `lib_moca`/`lib_mosca`; wire TAPNext++.
3. MegaSaM's bundled UniDepth → excise the subdirectory, substitute DA3-Metric-Large, pin the fork.
4. Checkpoint-tier substitution (`vitl` and Giant weights are NC while the code is Apache) → **pin SHA-256 per checkpoint** with a CI assertion mapping each hash to an allowlisted SPDX identifier.
5. Bundling a GPL FFmpeg or prebuilt COLMAP inside a release → invoke via subprocess, document as *invoked* not *bundled*, require the user's system binaries.

FFmpeg is LGPL-2.1+ by default and GPL-2.0+ when built with `--enable-gpl`. Calling the CLI over a pipe is separate-program invocation under the [GPL FAQ's mere-aggregation reading](https://www.gnu.org/licenses/gpl-faq.html#MereAggregation); statically linking libav* is not. GaussCapture already invokes rather than links — this is correct and must be preserved.

---

## 7. The contribution

### 7.1 Candidates, scored

Eleven candidate differentiators were scored on novelty, solo feasibility, user value, and defensibility. The top four:

| Candidate | Σ/20 |
|---|---|
| Open benchmark linking capture quality to reconstruction quality | **17** |
| Capture-time quantitative quality prediction and coaching | **16** |
| Static-camera monocular 4D | 15 |
| Fully permissive pipeline with CI-enforced licence assertion | 13 |

Rejected outright: Apple Silicon native training (7), PWA-first capture as a differentiator (9), reproducibility harness as a contribution rather than a floor (9), `.capturepack` as a novel format (10), splat editing (10 — SuperSplat owns this with 9,716 stars).

### 7.2 The claim

> **Capture telemetry measurable on-device at capture time predicts downstream Gaussian-splatting reconstruction quality, and we publish the first benchmark that demonstrates it.**

Supporting claim 1: a composed on-device coverage, parallax, blur, and exposure score that measurably raises final quality in a paired recapture study — the first quantitative alternative to qualitative coaching.

Supporting claim 2: a fully permissive, local-first reference pipeline with CI-enforced licence assertion over checkpoint tiers — the artifact that makes the benchmark reproducible without non-commercial contamination.

4D is future work with a demonstration attached.

### 7.3 Why this is not obvious

The individual signals are separately validated in the literature — blur compensation is worth **+1.68 PSNR on real phone captures** ([3dgs-deblur](https://spectacularai.github.io/3dgs-deblur/)), exposure-drift correction up to **+5.3 dB** ([Splatfacto-W](https://arxiv.org/pdf/2407.12306)) — but they have **never been composed and validated as a predictor**. The widely circulated community rules (120–300 images, 60–80% overlap, 1/500 s shutter) are blog consensus, not peer-reviewed, and make an excellent baseline to beat.

### 7.4 Anticipated objections

**"This is variance-of-Laplacian plus folk wisdom dressed as science."** Answered by a held-out predictive evaluation: Spearman ρ and AUC of the composed score against final colour-corrected PSNR and against binary COLMAP registration success, with per-signal ablations and the community heuristic as an explicit baseline. Prediction, not description, is the claim.

**"n=1 developer, n=6 scenes, unfalsifiable."** Answered by a pre-registered protocol of ≥60 scenes × ≥3 capture policies on a pinned gsplat commit. At $0.34/hour that is roughly **$31 of compute** — the sample-size objection is refutable for the price of lunch.

**"Confounded by COLMAP, and the metric is wrong."** Answered by dual pose front-ends (COLMAP global mapper and MapAnything-apache), showing the relationship survives both; and by reporting colour-corrected *and* raw PSNR, since the bilateral grid raises one while lowering the other ([gsplat #476](https://github.com/nerfstudio-project/gsplat/issues/476)). Registration failure is a first-class binary outcome, not an excluded sample.

---

## 8. Proposed architecture

One principle: **a single pip-installable core (`gausscapture`), with every third-party component invoked across a subprocess or HTTP boundary**, so that GPL, AGPL, and non-commercial components never link into the MIT distribution.

```
PHONE (native iOS/Android; PWA = degraded)
  capture.session      4K30 HEVC, AE/AF/WB locked, stabilisation OFF
  capture.sensors      ARKit/ARCore pose + intrinsics @60 Hz, raw IMU
  coach.signals        VoL blur · gyro gate · clip% · view-sphere bins · flow parallax
  pack.writer          .capturepack = BagIt + manifest-sha256 + transforms.json
        |
        v
DESKTOP (gausscapture core)
  ingest.validate  ->  pose.PoseBackend  ->  priors.PriorBundle
                                                   |
                          static <----------- scene type -----------> dynamic
                    recon.StaticTrainer                        recon.DynamicTrainer
                    (gsplat MCMC + bilagrid)                   (static BG + deformable FG)
                                       \        /
                                    export.writer  (PLY / SPZ / SOG)
                                            |
                                    serve.FastAPI + Spark viewer
        |
        v
CLOUD (optional, job-shaped)
  Kaggle 30 GPU-h/wk · Colab 12 h · Modal $30/mo · RunPod 4090 $0.34/h
```

Stable interfaces, all `typing.Protocol`:

```python
PoseBackend.solve(pack: CapturePack, priors: PosePrior | None) -> PoseGraph
PriorBackend.run(frames: FrameSeq, kinds: set[Prior])          -> PriorBundle
Trainer.fit(PoseGraph, PriorBundle, cfg)                       -> SplatModel
Exporter.write(SplatModel, fmt)                                -> Path
Metric.score(pred, gt, mask | None)                            -> dict[str, float]
```

`.capturepack` is demoted from a proposed standard to a **profile**: BagIt (RFC 8493) layout, `manifest-sha256.txt`, an unmodified nerfstudio-valid `transforms.json` inside, sidecar JSONL for IMU and pose, and a published JSON Schema. The success criterion is that unzipping any pack trains directly in gsplat with zero conversion. Prior art — Stray Scanner, Record3D, SplatKing's `splatpack.json`, ARCore's Recording API — already covers most of this ground; inventing a rival dialect with one implementer is how `.r3d` and `.ksplat` ended up where they are.

### 8.1 The 4D path, concretely

The key architectural move: **require a 20-second handheld background pre-roll before the tripod segment.** This converts an unsolvable zero-parallax problem into a solved static problem plus a bounded dynamic one.

1. Pre-roll SfM — 20 s orbit of the empty scene, COLMAP global mapper seeded with ARKit poses. The fixed tripod frame is registered into the same coordinate system by matching against the pre-roll.
2. Static background — gsplat MCMC, 30k iterations, bilateral grid. A genuine multi-view splat at 26–29 dB.
3. Foreground masks — SAM 2.1 with a single click prompt on frame 0; EdgeTAM for the on-phone preview.
4. Video depth — Video-Depth-Anything-Small, scale and shift aligned per clip against pre-roll sparse points on static pixels.
5. Tracks — TAPNext++ on 4,096 query points inside the frame-0 mask, AllTracker for occlusion flags, SEA-RAFT for a flow-consistency residual.
6. Canonical foreground init — unproject masked depth at t=0 into ~150k Gaussians.
7. Motion representation — a scaffold of 64–256 SE(3) node trajectories, each Gaussian skinned to its 8 nearest nodes. Parameters scale with K×T, not N×T.
8. Joint optimisation — 8k–15k iterations, ~8 GB VRAM, 12–25 minutes on a 4090, with a **scale-invariant Pearson correlation depth loss** (MoDGS's key finding: absolute depth loss fails), 3D track reprojection, as-rigid-as-possible regularisation on scaffold edges, and mask BCE. Adaptive density control is **disabled after 60% of iterations** — DyGauBench identifies it as the dominant instability source.

Ranked failure modes: disocclusion (unfixable, §5.4); depth flicker on textureless clothing; mask leakage on hair and hands; fast motion above ~30 px/frame breaking track re-detection; speculars; per-scene hyperparameter sensitivity.

---

## 9. Evaluation protocol

**Static.** Mip-NeRF 360 (nine scenes, every eighth image held out), Tanks & Temples, Deep Blending. Baselines: gsplat default vs MCMC vs bilateral grid, Brush, Postshot, Scaniverse. Target: within **0.3 dB of gsplat's published 29.15 dB**.

**Dynamic.** DyCheck iPhone with covisibility-masked mPSNR/mSSIM/mLPIPS at half resolution — masked numbers never reported beside unmasked ones. N3DV under the MoDGS single-camera protocol. NVIDIA Dynamic Scenes, train cam4 only.

**Systems metrics, all mandatory:** model size across PLY/SPZ/SOG, wall-clock on three fixed rigs, peak VRAM, render FPS at 1920×1080 on desktop, iPhone, and Quest 3 browser.

**The capture-quality correlation study — the novel contribution.** N=60 captures × 8 capture-time signals (VoL mean and 10th percentile, gyro |ω| 90th percentile, highlight-clip percentage, view-sphere bin fill, median parallax-to-rotation ratio, FAST track count, track lifetime) against final PSNR by Spearman ρ, plus a logistic regression predicting COLMAP registration failure.

**Pre-registered hypothesis: VoL-p10 achieves |ρ| ≥ 0.40 and the logistic model achieves AUC ≥ 0.80.** If it fails, the paper pivots to systems-only and we say so.

**New dataset — `GaussCapture-Phone-40`.** Forty scenes. Twenty-four static: eight subjects across object, room, and outdoor categories × three deliberate quality tiers (locked-AE and slow, AE-unlocked, fast and blurry) — deliberate variation is what makes the study causal rather than observational. Sixteen dynamic: phone A on a tripod plus **two witness phones at ±25° and ±45° providing genuine held-out ground-truth novel views** — the missing benchmark identified in §5.5. Approximately 45 GB, one Zenodo record (50 GB quota), CC-BY-4.0, minted DOI, Hugging Face mirror, Codabench leaderboard.

**Reproducibility.** Pinned CUDA image, `uv.lock`, checkpoint SHA-256 allowlist with a CI job asserting no non-commercial checkpoint hash is reachable, seeds threaded through NumPy and Torch, mean ± standard deviation over three seeds, and a CPU smoke test (8 frames, 200 iterations) finishing in under 10 minutes on standard GitHub runners.

---

## 10. Infrastructure and publication

### 10.1 Compute, at $0–35/month

| Purpose | Choice | Terms |
|---|---|---|
| Primary training | Kaggle | ~30 GPU-h/week, P100 or 2× T4 |
| Burst | Colab Free | 12 h ceiling, T4 in practice, quota unpublished |
| Overflow | Modal | $30/month free credits ≈ 37 h L4 |
| Escape hatch | RunPod Community 4090 | **$0.34/h** — a 30-min run is $0.17 |
| Local sanity check | Brush on the M2 | ~2 h for a 30k-step scene |

Programmes that do **not** work: Google Cloud's $300 trial excludes GPUs entirely; NVIDIA Inception requires an incorporated company. TRUBA (TÜBİTAK ULAKBİM, 312 GPUs) is affiliation-gated — *unverified whether an independent researcher can hold an account; worth one email to `trubadestek@tubitak.gov.tr`.* EuroHPC Development Access is plausible since Türkiye is Horizon Europe-associated, but *unverified whether an unaffiliated individual can be PI.*

### 10.2 Hosting

| Need | Choice | Why |
|---|---|---|
| Documentation and viewer | **GitHub Pages** | Free HTTPS, custom domains; 100 GB/month soft bandwidth |
| Splat assets (50–200 MB) | **GitHub Releases** | Explicitly **no bandwidth limit**, 2 GiB per file |
| Scaling path | Cloudflare R2 | Zero egress; 500 scenes × 150 MB ≈ $1.13/month |
| Dataset + DOI | Zenodo | 50 GB per record, free DOI |
| CI | GitHub Actions | Free on public repositories, standard runners |

Disqualified: Cloudflare Pages caps files at **25 MiB**. Netlify's free tier is a hard 300-credit cap with no overage — sites pause. Hugging Face free Docker Spaces were removed in 2026; a free hosted FastAPI backend there no longer exists. Oracle's "Always Free" tier was silently halved in June 2026 and should not be trusted.

Note the Cloudflare terms trap: serving large files through the free CDN from a non-Cloudflare origin violates their service-specific terms; serving from R2 is explicitly permitted.

### 10.3 Publication path

**JOSS is the best fit and has a binding constraint:** the repository must be **public for more than six months with development spanning that period**. A repository made public just before submission is desk-rejected. That clock started on 2026-07-26 with this repository going public. JOSS also excludes web tools that lack a core library — which is why §8's pip-installable core refactor is a v0 task, not a cleanup.

JOSS requires an **AI usage disclosure**. See §12.

arXiv tightened endorsement on 21 January 2026: unaffiliated authors require a personal endorser. That has months of latency and should be started by September 2026.

Secondary venues: MMSys Open Dataset & Software track (~January deadline) and ACM MM Open Source Software Competition (~May deadline), targeting 2027 cycles. Papers with Code is dead — retired by Meta in July 2025; Codabench (Apache-2.0, self-hostable) is the replacement substrate for a leaderboard.

Name check: `gausscapture` is free on PyPI, npm, and GitHub. *No trademark clearance has been performed.*

---

## 11. Risks

| Risk | P | Mitigation |
|---|---|---|
| Disocclusion unfixable — no permissive 4D generative prior | **90%** | Ship the bounded ±25° cone, hard-clamp the viewer, frame as bullet-time. Hedge with a 2–4 phone mode. |
| Mac-only development has no CUDA path | **80%** | Kaggle primary, Colab burst, RunPod escape at $0.17/run, Brush for local sanity |
| Per-scene hyperparameter brittleness (DyGauBench) | **70%** | One frozen config across all scenes, reported even where worse; three seeds ± σ |
| Checkpoint licence contamination | **60%** | CI hash allowlist; ban NC checkpoints from extras |
| gsplat API churn (no tag since v1.5.3, 352 open issues) | **50%** | Pin exact commit SHA; thin adapter layer; nightly CI against `main` |
| Pre-roll ↔ tripod registration fails | **40%** | Require 5 s overlap; fall back to metric depth pose; warn at capture time |
| JOSS scope rejection | **35%** | Core-library refactor early; genuine commit spread; full repo signals before submitting |
| A competitor ships an open pack format or dynamic mode | **25%** | Differentiate on the two things that cannot be copied quickly: a published ρ, and a DOI'd benchmark |

---

## 12. Disclosure

This report was produced with substantial generative-AI assistance: a multi-agent literature and market survey (12 agents, 437 web queries) followed by adversarial synthesis, drafted collaboratively with Claude Opus 5. All factual claims carry primary sources; claims that resisted verification are marked as such in this document. Problem framing, scope decisions, the choice to demote 4D from claim to demonstration, and the architectural direction are the author's. This disclosure is maintained in accordance with [JOSS's 2026 AI usage policy](https://github.com/openjournals/joss/blob/main/docs/submitting.md) and will accompany any submission derived from this work.

---

## 13. Summary of decisions

1. **Do not** pursue Apple Silicon native training. Rent a 4090 for $0.34/hour.
2. **Do not** depend on Nerfstudio, the Inria reference implementation, or any repository vendoring its rasterizer. Depend on gsplat.
3. **Do not** claim `.capturepack` as a standard. Make it a BagIt profile wrapping a valid `transforms.json`.
4. **Do not** claim orbitable 4D from a stationary phone. Ship a ±25° bullet-time card with the cone labelled, and pursue a sparse multi-phone mode as the tractable path to genuine novel views.
5. **Do** make the capture-quality predictor the primary research contribution, with a pre-registered hypothesis and a public benchmark.
6. **Do** refactor to a pip-installable core immediately — it gates the publication venue.
7. **Do** enforce licence cleanliness in CI. The ~0.6 dB it costs buys users the right to ship what they make.

---

## References

Primary sources are linked inline throughout. Key repositories and papers:

- gsplat — https://github.com/nerfstudio-project/gsplat (Apache-2.0)
- Brush — https://github.com/ArthurBrussee/brush (Apache-2.0)
- Spark — https://github.com/sparkjsdev/spark (MIT)
- splat-transform — https://github.com/playcanvas/splat-transform (MIT)
- COLMAP — https://github.com/colmap/colmap (BSD-3)
- MoDGS (ICLR'25) — https://arxiv.org/html/2406.00434v2
- Shape of Motion (ICCV'25) — https://github.com/vye16/shape-of-motion
- MoSca (CVPR'25) — https://github.com/JiahuiLei/MoSca
- DyGauBench (TMLR'25) — https://arxiv.org/abs/2412.04457
- Splatfacto-W — https://arxiv.org/pdf/2407.12306
- 3DGS motion-blur and rolling-shutter compensation — https://spectacularai.github.io/3dgs-deblur/
- ExpanDyNeRF — https://arxiv.org/html/2512.14406v1
- JOSS review criteria — https://github.com/openjournals/joss/blob/main/docs/review_criteria.md
