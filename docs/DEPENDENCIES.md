# Dependencies and licensing

GaussCapture is MIT licensed, and its users must be able to ship what they produce with it commercially. That constraint is stricter than it first appears: **most of the Gaussian-splatting research ecosystem is non-commercially licensed, and several repositories declare a permissive licence while shipping encumbered code.**

This document classifies every component as **bundled** (distributed inside our artifacts — its licence binds us) or **invoked** (called as a separate program over a subprocess or network boundary — its licence does not propagate under the [GPL FAQ's mere-aggregation reading](https://www.gnu.org/licenses/gpl-faq.html#MereAggregation)).

Audit date: 2026-07-26. Verdicts were verified against live `LICENSE` files, not secondary sources. See [RESEARCH.md §6](RESEARCH.md) for the full analysis.

---

## Safe — permissive, may be bundled

| Component | License | Role |
|---|---|---|
| [gsplat](https://github.com/nerfstudio-project/gsplat) | Apache-2.0 | Rasterizer and trainer. **The** dependency. |
| [Brush](https://github.com/ArthurBrussee/brush) | Apache-2.0 | Local Apple Silicon fallback trainer |
| [Spark](https://github.com/sparkjsdev/spark) | MIT | three.js viewer; only maintained 4D-capable path |
| [splat-transform](https://github.com/playcanvas/splat-transform) | MIT | PLY / SOG / SPZ / GLB conversion |
| [spz](https://github.com/nianticlabs/spz) | MIT | Compressed splat container (~10×) |
| [SAM 2.1](https://github.com/facebookresearch/sam2) / [EdgeTAM](https://github.com/facebookresearch/EdgeTAM) | Apache-2.0 | Foreground segmentation, server and on-device |
| [tapnet](https://github.com/google-deepmind/tapnet) (TAPNext++) | Apache-2.0 | Point tracking — replaces CoTracker3 |
| [AllTracker](https://github.com/aharley/alltracker) | MIT | Dense tracking |
| [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT) | BSD-3 | Optical flow |
| [Depth Anything 3](https://github.com/ByteDance-Seed/depth-anything-3) | Apache-2.0 — **Small, Base, Metric-Large, Mono-Large tiers only** | Monocular depth |
| [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything) | Apache-2.0 — **Small tier only** | Temporally consistent depth |
| [MapAnything](https://github.com/facebookresearch/map-anything) | Apache-2.0 — **`map-anything-apache` checkpoint only** | Pose prior |
| FastAPI, Pydantic, NumPy, Pillow, OpenCV | MIT / BSD / Apache-2.0 | Application stack |

---

## Invoked, never bundled

| Component | License | Boundary |
|---|---|---|
| [COLMAP](https://github.com/colmap/colmap) | BSD-3 core | `subprocess`. COLMAP's own docs note its licence is independent of its third-party dependencies; **prebuilt binaries may embed GPL components**, so we require the user's system installation and never vendor one. |
| [FFmpeg](https://github.com/FFmpeg/FFmpeg/blob/master/LICENSE.md) | LGPL-2.1+ default; **GPL-2.0+ with `--enable-gpl`** | `subprocess` over argv. We must never statically link `libav*` (LGPL relinking obligation) nor ship an FFmpeg binary inside a release artifact. |
| [OpenSplat](https://github.com/pierotofy/OpenSplat) | AGPL-3.0 | Optional user-installed binary only. Linking `libopensplat`, **or exposing it through a network service we host**, triggers AGPL §13 source disclosure of the whole combined work. |
| [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio) | GPL-3.0 | Optional, subprocess only. Any Python C-extension binding or bundled binary relicenses GaussCapture to GPL. |

---

## Excluded — do not install, vendor, or depend on

### Non-commercial licences

**Inria 3DGS** ([LICENSE](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md)) states plainly that you *cannot use, exploit or distribute* the work for commercial purposes. Everything that vendors its `diff-gaussian-rasterization` inherits this: MoDGS, 4C4D, hustvl/4DGaussians, Dynamic3DGaussians, SpacetimeGaussians, Ex4DGS, InstantSplat, Mip-Splatting, 2DGS, Scaffold-GS, Octree-GS, 3DGS-MCMC.

CC-BY-NC: **CoTracker3**, **UniDepth**, DUSt3R, MASt3R, Splatt3R, π³, DA3-Large and Giant tiers, VDA Base and Large tiers, `facebook/map-anything` default weights, VGGT-1B base.

Research-only: Fast3R, DepthCrafter, Stable Virtual Camera, TrajectoryCrafter, Metric3D v2.

### No licence file — all rights reserved

Instant4D, splat-apple, Grid4D, SplatFields, ActiveNeRF, FisherRF (`NOASSERTION`). FisherRF's Fisher-information criterion must be **reimplemented from the paper**, never vendored.

### Repositories that declare permissive but ship encumbered code

These are the traps. Each was verified directly:

| Repository | Declares | Actually ships |
|---|---|---|
| [MegaSaM](https://github.com/mega-sam/mega-sam) | Apache-2.0 | A vendored `UniDepth` subdirectory (CC-BY-NC-4.0) |
| [hustvl/4DGaussians](https://github.com/hustvl/4DGaussians) | Apache-2.0 | An Inria rasterizer fork |
| [4C4D](https://github.com/yangzf-1023/4C4D) | MIT | `diff-gaussian-rasterization` + builds on MASt3R (CC-BY-NC-SA) |
| [MoSca](https://github.com/JiahuiLei/MoSca) | MIT | MIT covers `lib_moca`/`lib_mosca` only; `lib_prior` carries CoTracker3, UniDepth, DepthCrafter |
| [MoDGS](https://github.com/Mobiuslqm/MoDGS) | MIT | Vendors Inria 3DGS |

---

## Enforcement

Planned CI gates (see [ROADMAP.md](ROADMAP.md) S1–S2):

1. **Rasterizer gate** — fail the build if `diff_gaussian_rasterization`, `simple-knn`, or `LICENSE_gaussian_splatting*` appears anywhere in the resolved dependency tree.
2. **Checkpoint hash allowlist** — pin SHA-256 per model checkpoint and assert each hash maps to an allowlisted SPDX identifier. This catches the tier-substitution trap, where `vitl` and Giant weights are non-commercial while the code is Apache-2.0.
3. **`reuse lint`** — every file carries `SPDX-FileCopyrightText` and `SPDX-License-Identifier`.
4. **SBOM** — `reuse spdx` output published with each release.

## Cost of licence cleanliness

Roughly **0.6 dB** of reconstruction quality versus using the best available (encumbered) component at every stage. The largest single gap is monocular depth (~0.3–0.6 dB); point tracking is at parity, making it the cheapest swap in the stack.

That is the price of letting users ship what they make.

---

## Reporting

If you believe a component here is misclassified, please open an issue with a link to the licence text. Licence status changes; this document is a snapshot and will drift.
