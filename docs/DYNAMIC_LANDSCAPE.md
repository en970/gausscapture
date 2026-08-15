# Who else is doing dynamic capture, and how

A survey of the fixed-camera / monocular-dynamic corner of the field, current to
**August 2026**. `RESEARCH.md` §3 surveys consumer capture apps and §5 argues the physics of
the fixed-camera case; this document is narrower and more recent. It asks one question:

> Somebody wants to record a person with a phone and watch the result move in a browser.
> Who can already do that, how, and what does it cost them?

The short answer is that everyone who does it well uses many cameras, everyone who does it
from one camera compensates in software with learned priors, and the one phone app that
now claims it needs hardware GaussCapture cannot assume.

---

## 1. The product that overlaps us most

**SplatCam** (iOS, free) records what it calls "holographic videos — 4D Gaussian Splats, or
Volumetric Video" on a phone and plays them back at splatcam.com
([App Store](https://apps.apple.com/us/app/splatcam/id6758737104)). It is the closest thing
to this project that exists as a shipping product, and it is worth being precise about how
it differs, because two of the three differences are the reason this project exists.

| | SplatCam | GaussCapture |
|---|---|---|
| Sensor | **Requires LiDAR** ("the LiDAR Depth sensor on iPhone Pro") | RGB + IMU only |
| Processing | Cloud — uploaded to splatcam.com | Local; the laptop does preparation, a rented GPU trains |
| Source | Closed | MIT |
| Camera | "hand-held or with your phone mounted in a tripod" | Two-phase: hand-held arc, then fixed |

The LiDAR requirement is the substantive one. A depth sensor supplies, directly, the thing a
fixed camera cannot infer: metric geometry without parallax. It is also why their approach
does not transfer here — **no Android phone in this project's reach has one**, and the
handset this was built against (Galaxy S22) certainly does not. An approach that needs LiDAR
is not a cheaper version of ours; it is a different problem with a hardware answer.

Cloud processing is the second difference and it is a positioning one rather than a
technical one. It is the same drift `RESEARCH.md` §3.2 records for Scaniverse.

Note the App Store page carries no rating summary — "this app hasn't received enough ratings
or reviews to display an overview". It is early, not established.

## 2. What serious 4D actually costs right now

Two rigs demonstrated at NAB 2026 set the reference for quality dynamic capture, and both
answer the parallax problem the expensive way — with more cameras:

- **Radiant Images Meridian 4D Volume Stage**: 24 iPhones plus Gaussian splatting, letting a
  virtual camera fly through a captured moment. It won CineD's Best-of-Show for camera
  control at NAB 2026
  ([CineD](https://www.cined.com/radiant-images-combines-24-iphones-with-gaussian-splatting-for-next-gen-bullet-time/)).
- **4DV.ai with OBSBOT**: a 60-camera hologram rig for 4D Gaussian splatting
  ([CineD](https://www.cined.com/step-inside-the-video-4dv-ai-and-obsbot-build-a-60-camera-hologram-rig-for-4d-gaussian-splatting/)).

This is the honest frame for what one phone can deliver. The industry's answer to "bullet
time of a person" is 24 to 60 synchronised viewpoints. Anything claiming the same result
from a single fixed camera is either using a depth sensor, hallucinating the unseen side, or
quietly restricting how far the viewer may orbit — which is exactly what this project does,
and says so in the interface.

## 3. The monocular-dynamic literature has become crowded

Between mid-2024 and mid-2026 this went from a handful of papers to a dense field. A
non-exhaustive list of what a re-implementer should know about:

| Work | Contribution |
|---|---|
| [Shape of Motion](https://arxiv.org/html/2407.13764) | Joint long-range 3D tracking and NVS from one video; motion bases |
| [MoDGS](https://arxiv.org/pdf/2510.12768) (ICLR 2025) | Casually-captured monocular video; the scale-invariant depth finding cited in `RESEARCH.md` §8 |
| [RoDyGS](https://arxiv.org/pdf/2412.03077) | Robustness for casual video |
| 4D-Fly (CVPR 2025) | Fast monocular 4D |
| MEGA (ICCV 2025) | Memory-efficient 4DGS |
| [ProDyG](https://arxiv.org/html/2509.17864v1) | Progressive dynamic reconstruction |
| [4D3R](https://arxiv.org/html/2511.05229) | Pose-free; two stages — foundation-model geometry, then motion-aware refinement |
| [MonoFusion](https://arxiv.org/html/2507.23782) | Sparse-view 4D; monocular geometry init aligned to a static multi-view reference |
| [Uncertainty Matters](https://arxiv.org/pdf/2510.12768) | Where monocular 4D is least trustworthy, quantified |
| [Prior-Enhanced GS](https://arxiv.org/html/2512.11356) | Better 2D priors rather than a new representation; CC BY 4.0 |
| [World from Motion](https://arxiv.org/pdf/2607.01202) | Generative dynamic reconstruction |
| [RiGS](https://arxiv.org/pdf/2605.23672) | Rigid-aware 4DGS from one monocular video |
| [GP-4DGS](https://arxiv.org/pdf/2604.02915) | Probabilistic 4DGS via Gaussian processes |

**The shared pattern is more informative than any single paper.** Almost all of them spend
their novelty budget on *priors that substitute for missing parallax*: monocular depth
(MoGe, MegaSAM), dense 2D point tracks (SpatialTracker), segmentation (SAM 2), optical flow
(RAFT), and increasingly video diffusion. Prior-Enhanced GS is explicit that it changes the
priors rather than the representation, and its dependency list — RAFT, SAM 2, MegaSAM, MoGe,
SpatialTracker, MoSca — is a fair picture of what a modern monocular-dynamic pipeline drags
in.

That has a licensing consequence this project must weigh before adopting any of it. Each of
those components carries its own terms, and `RESEARCH.md` §6 has already documented how
often a permissive-looking repository in this field is encumbered one level down. **A prior
stack is a licence surface**, and a large one.

## 4. The literature agrees with our physics, in its own words

The searches surfaced statements of the constraint this project is built around, from
authors with no stake in our argument:

- Monocular reconstruction is "particularly challenging due to limited parallax, occlusion,
  and motion"; classical geometric constraints are "highly effective under sufficient
  parallax", which monocular capture lacks.
- In monocular scenes with objects moving in every frame, "the SfM algorithm struggles to
  obtain accurate camera poses and point clouds, often either removing point clouds of
  dynamic objects or failing to find camera poses for each frame."
- On small-baseline capture: gains are "modest due to the small-baseline setup, indicating
  that fixed or minimally-moving camera configurations provide limited baseline geometry."
- Prior-Enhanced GS designs a virtual-view depth loss precisely because "small camera
  baseline videos let Gaussians overfit training views, producing floaters visible only from
  novel viewpoints."

The last one deserves emphasis because it describes a failure that *looks like success*: a
model that renders the training view beautifully and falls apart the moment the viewer
moves. A project shipping a browser viewer with a free camera would show exactly that. It is
an independent argument for clamping the viewer to the measured cone.

Finally, and most directly: facial reconstruction from fixed monocular cameras is where
these parallax and geometric problems appear in their most acute form. That is our exact
case.

## 5. Where this leaves GaussCapture

The field splits into three answers to one problem — a single viewpoint does not contain the
scene:

1. **Add cameras.** 24 to 60 of them. Correct, and out of reach.
2. **Add a depth sensor.** SplatCam. Correct where the hardware exists; unavailable here.
3. **Add learned priors.** The academic mainstream. Powerful, and it imports a large licence
   surface plus a dependence on models whose failure modes are hard to bound.

GaussCapture takes a fourth, which the survey did not find anyone else shipping: **create
the parallax at capture time.** The `arc` phase is a hand-held sweep with the subject
holding still, so their head is briefly part of the rigid scene and ordinary
structure-from-motion triangulates it. Then the phone is set down and only the subject
moves. The geometry problem and the motion problem are separated in time rather than
disentangled by a prior.

What is genuinely ours in that: it needs no depth sensor, no diffusion model, no learned
prior in the critical path, and therefore no new licence surface — and the reconstruction it
starts from is measured rather than inferred. What it costs: a protocol the operator has to
perform correctly, which is not free. The first three attempts at it failed, and the app now
refuses a take whose sweep never happened rather than letting that discovery wait for the
desktop.

Two ideas from the survey are worth taking, and neither compromises the above:

- **A virtual-view or hold-out penalty during training.** Prior-Enhanced GS added one
  because small-baseline captures overfit their training views. Ours is a small-baseline
  capture by construction. We already compute a temporal hold-out PSNR; a view-space
  equivalent would catch the floater failure before a scene is published.
- **Monocular depth as an optional initialiser, never a dependency.** MonoFusion reports
  better results when sparse-view reconstruction is initialised from a monocular geometry
  estimator. Behind the existing rasterizer-style seam this would be an experiment rather
  than a commitment, and it must stay optional: the moment it becomes required, the licence
  and failure-mode arguments above change.

What the survey does **not** support is loosening the cone. Nothing found here reconstructs
the unobserved side of a subject from one fixed viewpoint without generating it.

---

## Method

Searched August 2026 via web search and direct page reads: monocular/fixed-camera 4D
Gaussian splatting methods, phone apps claiming 4D or volumetric capture, and professional
bullet-time rigs. Claims about SplatCam come from its App Store listing; rig details from
CineD's NAB 2026 coverage; method claims from the linked papers. Where a paper's licence or
dependencies are stated above, they were read from the paper itself rather than inferred.

This is a landscape survey, not a benchmark. Nothing here was reproduced, and no number in
the linked works has been independently verified.
