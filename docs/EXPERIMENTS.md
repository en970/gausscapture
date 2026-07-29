# Experiments

A log of what was actually run and what it showed, including the results that
did not go the way the roadmap expected. Negative results are recorded here at
the same weight as positive ones; a roadmap item that turns out not to work is
information, and quietly dropping it would leave the next person to rediscover
it.

Every entry states its sample size. At the sizes reached so far these are
directions, not results — the pre-registered study in [RESEARCH.md](RESEARCH.md)
needs n ≥ 30.

---

## E1 · Does the pipeline work on real phone footage?

**2026-07-27 · n=3, one subject, Galaxy S22**

Three captures of the same subject, deliberately varied: `A_good` (locked
settings, slow orbit, 65 s), `B_normal` (ordinary pace, 30 s), `C_fast`
(deliberately shaky, 27 s). Extracted at 2 fps and again at 4 fps, then through
COLMAP.

| capture | 2 fps | 4 fps |
|---|---|---|
| A_good | 93/100 registered (93%), 34k points | 221/234 (94%), 105k points |
| B_normal | 17/47 (36%), 2k points | 63/91 (69%), 13k points |
| C_fast | 2/48 (4%), 43 points | 15/89 (17%), 0.8k points |

**Yes.** `A_good` produced a genuine reconstruction.

The second row exists because frame count was a confound: A had 100 frames and
B and C had under 50, below the 120–300 the community treats as a working
range. At matched counts — A at 100, B at 91, C at 89 — the ordering still
holds at 93 / 69 / 17%, so capture quality has an effect that frame count does
not explain. Frame count matters a great deal on its own, though: doubling it
moved B from 36% to 69%.

**Mechanism**, from the COLMAP database:

| capture | features/frame | verified pairs | median inliers |
|---|---|---|---|
| A_good | 10,031 | 1,220 | 449 |
| B_normal | 7,757 | 266 | 352 |
| C_fast | 4,051 | 145 | 73 |

Blur suppresses corner detection, which thins the match graph, which leaves the
mapper unable to connect a reconstruction.

**Caveat that limits everything above:** all three captures are of one subject.
"Capture quality matters" and "this subject happens to work at slow speed" are
not yet distinguishable.

---

## E2 · Do the telemetry signals order the captures correctly?

**2026-07-27 · same n=3**

| signal | A_good | B_normal | C_fast | ordered? |
|---|---|---|---|---|
| `vol_p10` | 107 | 73 | 36 | yes |
| `vol_median` | 267 | 213 | 112 | yes |
| `motion_mean` | 48 | 53 | 56 | yes |
| `too_fast_ratio` | 92% | 96% | 97% | saturated |
| `blurry_frame_ratio` | 20% | 29% | 24% | **no** |
| composite `score` | 60 | 56 | 58 | **no** |

**The raw statistics order correctly; the thresholded derivatives do not.**

`too_fast_ratio` saturates because its fixed cutoff of 35 sits far below what
real 1080p footage produces. `blurry_frame_ratio` puts the worst capture in the
middle, because relative blur compares each frame to its neighbours and
`C_fast` is uniformly shaky — there is no locally-soft frame to flag. The
composite score inherits both failures.

The cutoffs are **not** being retuned against these three captures. Fitting a
threshold to n=3 is the error this project exists to avoid, and the pull to do
it is itself an argument for collecting the benchmark.

---

## E3 · Does seeding COLMAP with the device's intrinsics help?

**2026-07-29 · n=3 · negative result**

Roadmap S4 assumed device metadata would speed up or stabilise
structure-from-motion. Testing the intrinsic half of that:

| capture | registered (unseeded → seeded) | points | COLMAP seconds |
|---|---|---|---|
| A_good | 93% → 93% | 34,047 → 34,140 | 188 → **219** |
| B_normal | 36% → 36% | 2,058 → 2,073 | 52 → **66** |
| C_fast | 4% → 4% | 43 → 34 | 18 → **29** |

**No benefit, and a consistent 17–61% slowdown.** The roadmap's criterion was a
≥30% speed-up; the measured effect is the opposite sign.

The solved focal lengths explain why. On `A_good`, seeded and unseeded runs
converge to 1351.3 and 1352.7 from entirely different starting points — the
images already determine the focal length, so the prior contributes nothing and
only adds bundle-adjustment work. On `C_fast`, which does not reconstruct, the
solved value is meaningless either way (1717 unseeded, 3042 seeded).

Seeding is therefore **off by default**, behind `--seed-intrinsics`.

### A more interesting side finding

On `A_good`, COLMAP converges to **fx ≈ 1352** while the device reports
**fx ≈ 1298** after rescaling — a systematic 4.2% gap that seeding does not
close, reached independently from two starting points.

A plausible explanation is that the video path does not use the sensor's full
width: if the recording covers about 96% of the 4080-pixel active array rather
than all of it, the true focal length in recorded pixels is 1298 / 0.96 ≈ 1352.
That is a testable hypothesis, not a conclusion — and it would mean the
device-reported intrinsics need a per-device correction factor for the video
path before they can be used as ground truth.

**What this experiment does not test:** seeding *poses* from ARCore, which is a
far stronger prior than intrinsics and is what roadmap S4 was really about. The
capture app does not record ARCore poses yet, so that remains open.

---

## Reproducing

```bash
gausscapture bench run <captures/> --out runs/e1 --preset balanced
gausscapture bench run <captures/> --out runs/e3-seeded --preset balanced --seed-intrinsics
gausscapture bench analyze runs/e1
```

Raw COLMAP databases are not committed — they are large and regenerable. The
capture footage is not committed either; see [ROADMAP.md](ROADMAP.md) for the
dataset release plan.
