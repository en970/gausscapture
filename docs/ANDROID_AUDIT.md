# Audit of the Android capture app, v0.2

**Status:** input to a rewrite. Read with `docs/RESEARCH.md` §5 (capture-side
quality) and `docs/ROADMAP.md` S4 (pose seeding).

The capture app is the only part of this project that produces original data,
so its defects propagate into every downstream claim. This records what it gets
right, what it gets wrong, and what a replacement must preserve. Line references
are to the v0.2 sources as of commit `b252793`.

---

## 1. What works, and the evidence for it

One thing in this app matters more than everything else: **`imu.jsonl` and
`frames.jsonl` share a hardware clock.** That is what lets a frame's camera pose
be paired with the accelerometer reading taken at that instant, and it is the
whole reason a native app exists rather than a screen recording.

It has now been verified end to end. Pairing each registered COLMAP pose with
its interpolated accelerometer sample and rotating into the world frame gives a
per-frame estimate of "up"; those estimates agree at **0.998 or better** across
all three captures (`src/gausscapture/pose/gravity.py`). Agreement approaches 1
only if the poses, the sensor axis mapping and the clock alignment are all
correct simultaneously, so a single number validates the entire chain.

The measurement replaced two heuristics that had been wrong by up to 41° and 81°
respectively. **The app was already recording the right data; the pipeline was
ignoring it.**

Everything in the rewrite is subordinate to keeping that property.

---

## 2. The frame-index defect (D03)

`frames.jsonl` carries a field named `frame`. It is *not* an index into
`video.mp4` — it is an ordinal over `CaptureResult`s that arrived while
`recording == true` (`CameraController.java:422-443`). The Python side maps
video frame *N* to `frames.jsonl` frame *N*. Nothing enforces that these agree.

Measured against the encoded files:

| capture | video frames | `frames.jsonl` rows | drift |
|---|---|---|---|
| `0ea05a00` | 1941 | 1939 | video ahead by 2 |
| `3027613c` |  825 |  827 | log ahead by 2 |
| `c92faaaa` |  905 |  907 | log ahead by 2 |

The drift is small but its **sign varies between captures**, so it cannot be
corrected by a constant. At 30 fps, two frames is 67 ms.

How much does that matter? For the up direction, not much — a deliberate
sensitivity sweep, shifting the mapping by ±1, ±2 and ±4 frames, moves the
estimate by at most **1.58°** and never drops agreement below **0.9978**.
Gravity is a slowly varying signal and averages over the whole capture, so it
tolerates the error. That is why the 0.998 result stands.

It will not survive the next use. ROADMAP S4 seeds reconstruction with per-frame
orientation, where a 67 ms error is a real angular error at walking speed, and
its sign is unknown per capture.

**The fix is available and cheap.** For a camera feeding a `MediaRecorder`
surface, `SENSOR_TIMESTAMP` *is* the presentation timestamp the muxer writes.
Recording the `SENSOR_TIMESTAMP` of the first encoded frame, and matching on
timestamp rather than on ordinal, makes the mapping exact. The value is in hand
at `CameraController.java:404` and is discarded.

---

## 3. Defects

34 were found. Grouped by what they cost.

### 3.1 The stated workflow does not work

**D14 — captures cannot be retrieved by plugging the phone in.** Everything is
written under `getExternalFilesDir(null)`, i.e.
`/sdcard/Android/data/com.gausscapture.capture/files`. Since Android 11, Samsung
One UI hides `Android/data` from both the on-device file manager and from MTP
enumeration. Retrieval requires `adb`. Nothing in the app exposes takes through
MediaStore, the Storage Access Framework, or a share intent.

This is the single defect that blocks the operator's actual routine, and it is
a design choice rather than a bug.

### 3.2 Data loss

| id | trigger | what is lost |
|---|---|---|
| D02 | camera disconnect or HAL error mid-take | MP4 never gets its `moov` atom → the whole video is unplayable; IMU tail unflushed; UI keeps counting as if fine |
| D04 | screen timeout (30 s default), call, notification tap | no `keepScreenOn`, no wake lock, no foreground service → `onPause` truncates the take. Fatal for a "record all the time" routine |
| D05 | storage fills, or MP4 passes ~4 GB (~22 min at 24 Mbit/s) | `MediaRecorder` error and info listeners are never registered → encoding stops silently while the UI counts on |
| D07 | double-tap on record | second session dir; second writer opened over `imu.jsonl` with the first dropped unflushed; two capture sessions race. Window is the 100–500 ms session configuration |
| D08 | stop and restart within the same wall-clock second | session directory name has 1 s granularity and an existing directory is treated as success → the previous take is overwritten without warning |
| D13 | `recorder.stop()` throws at the end of a long take | the entire session directory is deleted, video and IMU, and the operator is told "Take was too short" |
| D24 | process death | both writers use the default 8 KB buffer and are never flushed mid-take; no crash-recovery scan on next launch |

### 3.3 Crashes

- **D01 (critical).** `new MediaRecorder(Context)` requires API 31; `minSdk` and
  `d8 --min-api` are both 28. Latent on the S22, fatal on Android 9/10/11 — and
  the Gradle-free build has no lint step, so it cannot catch this class of error
  at all (D33).
- **D09.** `onConfigured` catches only `CameraAccessException`;
  `setRepeatingRequest` on a closed session throws `IllegalStateException`,
  uncaught on the background thread. Triggered by backgrounding while recording.
- **D18.** `getSurfaceTexture()` dereferenced without a null check after
  `onSurfaceTextureDestroyed` released it.
- **D29.** `getExternalFilesDir(null)` may return null; `takes()` guards for it,
  `startRecording` does not.

### 3.4 The IMU stream is being starved (D10)

Camera `CaptureResult` callbacks and all five sensors' `SensorEventListener`
callbacks are delivered on **one** background `Handler`, and each does
synchronous `BufferedWriter` I/O plus JSON allocation on it. That is roughly
2000 sensor events per second on an S22 plus 30 capture results per second,
serialised behind a single thread doing blocking writes to external storage.
A flush stall both delays camera metadata (widening D03) and back-pressures the
sensor queue, which the framework resolves by **dropping samples**.

The most valuable stream in the project is the one being starved.

### 3.5 Metadata asserted rather than measured

- **D11.** `capture_settings.{exposure_locked, white_balance_locked,
  focus_locked, stabilisation_disabled, lens_switching_disabled}` are hardcoded
  `true` literals. Samsung HALs routinely honour these only partially. The
  `CaptureResult` needed to report the truth is already in hand.
- **D11 (cont).** `summary()` prints "imu offset recorded" when the timestamp
  source is *not* `REALTIME` — but no offset is ever computed or stored. The
  string tells the operator an unusable capture is fine.
- **D12.** `SensorEvent.timestamp` is assumed to be on `CLOCK_BOOTTIME` without
  a check. Several OEMs have shipped sensors on `CLOCK_MONOTONIC` instead, and
  the error is the accumulated deep-sleep time since boot — potentially hours.
  One comparison against `SystemClock.elapsedRealtimeNanos()` would both detect
  it and produce the offset the app already claims to record. Given that clock
  alignment is the entire reason this app exists, this is the top structural gap.
- **D20.** `video.fps` is the literal 30 regardless of what the encoder
  achieved; `width`/`height` are pre-rotation while `orientation_hint` says the
  file plays rotated.
- **D21/D23.** Camera 0 on an S22 is a logical multi-camera and no
  `SCALER_CROP_REGION` or zoom ratio is pinned, so the HAL may switch physical
  lenses in low light while `intrinsics.json` describes one lens.
  `focal_pixels_recorded` rescales by the width ratio only, assuming a purely
  vertical crop; the authoritative per-frame `SCALER_CROP_REGION` is never read.
- **D26.** The chosen preset is not persisted; every process start resets to
  `A_good`. `Preset.java`'s own documentation argues a mislabelled capture is a
  wrong answer — this is that failure mode, built in.

### 3.6 Smaller

D15 (encoder instances leak until force-stop on repeated start failures),
D16 (`RECORD_AUDIO` declared, never used — a privacy-dashboard liability for a
public research artifact), D17 (camera opened while the permission dialog is
still up, so first launch always shows a spurious error; no recovery on denial),
D19 (`Surface.release()` is never called anywhere), D22 (3A locked without
waiting for convergence, and the locked values are never recorded), D25 (blocking
I/O on the UI thread), D27 (cross-thread fields without `volatile`; the first
samples of a take can be timed against the *previous* take's start), D28 (orphan
session directories are invisible to the UI and undeletable), D30 (no window
insets — the record button sits under the gesture pill; no launcher icon), D31
(manifest says 0.1, pack says 0.2, so a pack cannot be traced to its APK),
D32 (dead code), D34 (an orphan `.idsig` committed for an ignored APK).

---

## 4. What the rewrite must preserve

These were got right the hard way and must survive verbatim or in substance.

| what | why |
|---|---|
| `previewRotation()`, `(sensorOrientation - displayDegrees + 360) % 360` | Reproduces the canonical Camera2 orientation table exactly, and is correct for *both* `setOrientationHint` and the preview transform |
| `applyTransform()` **and its comment** | Cancel `TextureView`'s stretch using the *true* buffer dimensions, then rotate, then fit. Using post-rotation dimensions in step 1 is unrecoverable, and the comment is the only record of why |
| `SENSOR_INFO_TIMESTAMP_SOURCE` probe → `manifest.clocks` | The one check the entire gravity pipeline depends on |
| `imu.jsonl` row schema | Absolute `t_ns` as primary key *alongside* a relative one is precisely what makes 0.998 alignment possible. `Sensor.getStringType()` gives the literal the reader matches on |
| registering the *uncalibrated* gyro and accelerometer | Raw signal plus the bias being subtracted. Costs nothing, cannot be recovered later |
| `blurPixels()` = ω · t_exp · f_px | Exact small-angle geometry, not a fitted heuristic, and the doc is honest about the translation-blur limitation. The only defensible on-device quality signal the project has |
| `intrinsics()` | Correct `LENS_INTRINSIC_CALIBRATION` ordering and distortion model label, and it emits `pre_correction_active_array` so the rescale can be redone offline |
| the non-lock half of `applyCaptureSettings` | Stabilisation off, scene mode off, effects off, distortion correction off — the correct complete set for keeping one intrinsic model valid |
| `HIGH_SAMPLING_RATE_SENSORS` | Required from API 31 for anything above 200 Hz. Without it the S22's gyro is silently capped and nothing says so |
| `build.sh`: `-bootclasspath android.jar` at `-source 8`; keystore kept outside `$BUILD`; zipalign *then* apksigner | Compiling against Android's API surface stops nonexistent `java.*` calls reaching the device. A regenerated key forces an uninstall, which destroys every recorded session. Aligning after signing invalidates the signature |
| `Preset` and its `expectedToFail` flag | Two of five presets must be expected failures so a predictor is not trained only on successes — the scientific core of ROADMAP S7 |
| confirming deletion | The target is minutes of walking |
| blur hysteresis with a minimum visible time | The right shape for a warning that must not flicker at the boundary |

---

## 5. Requirements for the replacement

1. **Retrieval must work by plugging the phone in** (D14), or the routine the
   app exists to serve does not happen.
2. **Anchor `frames.jsonl` to the video by timestamp, not by ordinal** (D03).
   Record the first encoded frame's `SENSOR_TIMESTAMP`.
3. **Verify the sensor clock** against `elapsedRealtimeNanos()` and record the
   measured offset instead of asserting one (D12).
4. **Give the IMU its own thread** (D10). It is the most valuable stream.
5. **Survive interruption**: keep the screen on, hold the take through
   `onPause`, register the recorder's error and info listeners, flush
   periodically, and recover orphaned sessions on launch (D02, D04, D05, D24).
6. **Measure metadata, never assert it** (D11, D20, D21, D22). Where a value
   cannot be verified, record that it could not.
7. **Make the build able to fail** on API-level violations (D01, D33).
