# CapturePack

A `.capturepack` is a **[BagIt](https://datatracker.ietf.org/doc/html/rfc8493) bag, distributed as a zip**, whose payload holds a capture: a manifest, the source video, and whatever sensor metadata the recording device could provide.

JSON Schema for the manifest: [`src/gausscapture/schemas/capturepack.schema.json`](../src/gausscapture/schemas/capturepack.schema.json)

---

## Why a profile and not a format

This is deliberately **not** a new standard. Stray Scanner, Record3D, ARCore's Recording API and SplatKing's `splatpack.json` already cover most of this ground, and a zip dialect with a single implementer is not a format — it is a liability, as `.r3d` and `.ksplat` both demonstrate.

What GaussCapture contributes is a *profile*: an existing container (BagIt) wrapping an existing interchange format (`transforms.json`). The practical test is that a pack is readable by tools that have never heard of this project. It passes — an exported pack validates under `bagit-python`, the reference implementation, with no knowledge of GaussCapture involved.

The full argument is in [RESEARCH.md §8](RESEARCH.md).

---

## Layout

```text
session.capturepack                  (a zip)
  bagit.txt                          BagIt version and encoding
  bag-info.txt                       metadata, including Payload-Oxum
  manifest-sha256.txt                one checksum line per payload file
  tagmanifest-sha256.txt             checksums of the tag files themselves
  data/                              the payload
    manifest.json                    required
    video/main_video.mp4             required
    camera/intrinsics.json           optional
    camera/lens_profile.json         optional
    motion/arkit_arcore_poses.json   optional
    motion/imu_gyro.json             optional
    environment/light_estimation.json optional
    quality/capture_warnings.json    optional
    checksums/sha256.json            working-directory checksums
```

`manifest.json` and the main video are required. Everything else is optional and produces a warning when absent — a pack made from a bare phone video has none of it, which is exactly why such a capture needs structure-from-motion downstream.

### With `--with-dataset`

`gausscapture pack export <project> --with-dataset` adds a directly trainable dataset to the payload:

```text
  data/
    images/frame_000001.jpg …        extracted frames
    sparse/0/{cameras,images,points3D}.bin
    transforms.json                  nerfstudio-convention poses
```

Unzip it, point a trainer at `data/`, and there is no conversion step. gsplat's examples read the COLMAP layout; nerfstudio, Brush and most research code read `transforms.json`. Both are present because writing both costs a few kilobytes and removes a conversion either way.

---

## Working directories are not bags

A project on disk keeps a **flat** `capturepack/` directory. Bagging applies at export and is unwrapped at import.

This is intentional. Every pipeline stage reads and rewrites the working directory, so a checksum manifest inside it would be stale the moment you extract a frame — worse than no manifest, because it looks authoritative. The guarantees belong where files actually travel between machines.

Import accepts three shapes, because archives arrive from three eras and three tools:

1. a BagIt bag — checksums are verified, and a mismatch fails the import as a corrupt transfer rather than an incomplete capture;
2. a flat pack — packs written before this profile existed;
3. a pack zipped together with its parent folder — what right-click → Compress produces.

---

## Timestamps

Sensor sidecars written by GaussCapture express time as **milliseconds since recording start**, and the manifest declares this in `time_base`.

This is stated explicitly because getting it wrong is silent and fatal: the capture PWA previously stamped IMU samples with `performance.now()` and the recording start with `Date.now()` — two different epochs with no anchor between them, which made the sensor logs impossible to align to video frames at all. Any producer using a different time base must say so in `time_base`.

---

## Versioning

`capturepack_version` follows the project's pre-1.0 discipline: the minor version bumps on breaking changes. Readers should accept older versions. The current writer emits `0.1` manifests inside a BagIt container; the manifest schema itself is unchanged from the pre-profile layout, so old manifests remain valid.

---

## Validating

```bash
gausscapture pack validate <project> --verify
```

Reports manifest validity, missing optional metadata, checksum verification, and — when present — whether `transforms.json` is well formed and its referenced images exist.

Third-party validation works too, and is a better test of the claim:

```python
import bagit
bagit.Bag("extracted-pack/").validate()
```
