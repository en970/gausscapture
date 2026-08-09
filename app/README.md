# GaussCapture — capture app

Flutter interface, native capture engine. The split is the architecture, not an accident of
history.

## Why the recording is not in Dart

Flutter's camera plugin does not expose a per-frame `SENSOR_TIMESTAMP`, and its sensor plugins do
not expose `SensorEvent.timestamp` at all. Those two values, on one hardware clock, are what let a
frame's camera pose be paired with the accelerometer reading taken at that instant — which is how
the world's up direction is recovered, measured at **0.998 agreement** across a capture against
heuristics that were wrong by as much as 41 degrees (`src/gausscapture/pose/gravity.py`).

A pure-Dart rewrite would have produced a better-looking app that gathered unusable data.

So Flutter owns every pixel except the camera preview, which is a platform view because the
transform that makes it upright and correctly proportioned is written against a real `TextureView`
and is the most error-prone code in the project.

## Six presets, and why one of them is different

A–E are one undivided stretch of recording, stopped by hand. **F is a script**: four phases inside
a single file.

```
perch    2 s      phone in its support, face framed, hands off
arc     12 s      lift it and sweep around your own face — you hold still
reseat   ~3 s     put it back and let go; ends when the phone is measurably at rest
hold     3 s      now move. the phone stays put. auto-stops.
```

F exists because a fixed camera pointed at a moving face produces zero parallax, and zero parallax
is undefined for structure from motion — the geometry simply is not in the recording. The arc puts
it there: while the subject holds still their head is part of a rigid scene and triangulates like
the wall behind them. One file rather than four, because the phase that carries the geometry and
the phase that carries the motion have to share an exposure, a white balance and a clock, and
because stopping the recorder to start it again moves the phone.

Two of its rules are load-bearing rather than cosmetic:

* **The reseat ends on a measurement, never on a clock.** Every frame of the hold is given the same
  camera pose downstream. If the hold began while the phone was still settling, that is false for
  the first half second and nothing later would ever say so. `Stillness` decides, from the
  gyroscope and the accelerometer together, and the countdown restarts the moment the phone is
  touched.
* **The hold auto-stops.** Reaching for the phone to press stop moves it, and the frames that would
  corrupt are the last ones — the ones closest to the motion the take exists to record.

The hold raises no warnings at all. The subject is being asked to move and look natural while a
camera records their face; a large amber word in front of them is both useless and destructive of
the thing being recorded. If something goes wrong there the take ends and the manifest says why.

## What the interface does with that

Five layers, ordered by how little attention each demands — the operator of an F take is also its
subject, facing the front camera with their hands busy.

| Layer | Carries |
|---|---|
| Haptics | one buzz per phase change, a tick a second through the hands-off countdown |
| Perimeter | the primary signal. Amber and pulsing when something is wrong; otherwise white, brightening as rest accumulates, with one edge lit during the sweep to say which way still owes its share |
| The one word | 44 sp amber pill: the warning if there is one, otherwise the phase cue |
| Countdown | the seconds between letting go and being asked to move |
| Phase rail + ring | which of the four, and how far through it |

Silence is still the success state for A–E. F is the deliberate exception, and the reason is in
`lib/screens/capture_screen.dart`: an instruction that changes every few seconds is not a
persistent quality signal, it is the content of the take.

Reduced motion is honoured throughout — every animated element reports the same state standing
still.

## Layout

```
lib/                       Dart: interface only
  design.dart              palette, type scale, spacing — derived in docs/APP_SPEC.md §2
  capture_engine.dart      the channel wrapper and the data model
  main.dart                the app shell
  screens/
    capture_screen.dart    the only screen used while moving; the guidance priority queue
    takes_screen.dart      what is on disk, its health, and whether it has been copied off
  widgets/
    coach.dart             the one 44 sp word, warning or cue, on an opaque amber pill
    perimeter.dart         the peripheral border: warning, rest confidence, sweep direction
    phase_rail.dart        the four-segment script strip, and the hands-off countdown
    record_button.dart     start/stop and the progress ring
    shot_picker.dart       the six protocols
tool/
  make_launcher_icon.py    generates every launcher asset; run it, commit the PNGs
android/app/src/main/java/com/gausscapture/capture/
  MainActivity.java        FlutterActivity; three channels and nothing else
  PreviewFactory.java      the preview as a platform view
  CaptureEngine.java       what a capture is, with no interface attached; drives the script,
                           writes the manifest
  CameraController.java    Camera2, per-frame metadata, intrinsics, phase spans, the 3A locks
  PhaseMachine.java        the script: which phase, what to say, when to stop. No Android imports
  Stillness.java           whether the phone is actually at rest, and by how much it was not
  Arc.java                 how far the sweep went, and whether it was an orbit or a pivot
  SensorLogger.java        five sensors on a thread of their own
  JsonlWriter.java         double-buffered writer; never blocks a producer
  Clocks.java              measures which clock the sensors are on
  Storage.java             where captures live and whether a Mac can see them
  Preset.java              the six capture protocols and F's four phases
android/app/src/test/java/com/gausscapture/capture/
  ProtocolLogicTest.java   66 checks over PhaseMachine, Stillness and Arc, in a plain JVM
```

`PhaseMachine`, `Stillness` and `Arc` import nothing from Android on purpose. Each is a place where
a wrong answer produces a take that looks fine and is not reconstructable, so "it worked when I
tried it on the phone" is not good enough evidence. They run on a laptop:

```sh
cd android/app/src
javac -d /tmp/gc-test \
  main/java/com/gausscapture/capture/{Preset,PhaseMachine,Stillness,Arc}.java \
  test/java/com/gausscapture/capture/ProtocolLogicTest.java
java -cp /tmp/gc-test com.gausscapture.capture.ProtocolLogicTest
```

## Building

```sh
export JAVA_HOME=/opt/homebrew/opt/openjdk@17
flutter analyze
flutter build apk --release          # or: flutter run
```

## The launcher icon

Generated, not drawn. A Gaussian splat is a soft anisotropic ellipse, and a 4D capture is that
ellipse at several instants — so the mark is one elliptical Gaussian shown three times, the leading
instance opaque and the two behind it fading, offset across their own long axis. Amber on
near-black, from `lib/design.dart` and nowhere else.

```sh
python3 tool/make_launcher_icon.py     # needs only Pillow
```

It writes the five legacy densities (48/72/96/144/192, square and round), the adaptive foreground,
background and monochrome layers at all five densities (108 → 432), and
`res/mipmap-anydpi-v26/ic_launcher.xml`. The PNGs are committed: a build must not depend on the
script having been run.

## Getting captures onto a computer

Captures go to `Documents/GaussCapture`, which appears over MTP. That needs all-files access; the
app asks once and falls back to its private directory if refused. Either way:

```sh
gausscapture pull            # over adb, works regardless of where they landed
```

## What is recorded

Per take: `video.mp4`, `frames.jsonl` (one row per capture result, each tagged with its phase),
`imu.jsonl` (five sensors), `events.jsonl` (anything that invalidated an assertion),
`intrinsics.json`, `manifest.json`. A scripted take adds `stillness.jsonl` — what the rest detector
actually saw, so its thresholds can be revised from data rather than from opinion.

The manifest is capturepack **0.3**. Beyond what 0.1 carried it declares:

| Block | What it says |
|---|---|
| `protocol` | the frame bounds of each phase, the rest window, whether the take stopped itself, and why it stopped early if it did |
| `arc` | what the phone's own sensors made of the sweep — extent, median rate, implied cone, orbit versus pivot |
| `fixed_camera` | whether rest was confirmed, and the rotation residual integrated across the hold |
| `photometric` | the 3A state actually achieved and the fraction of frames that verified it, as against `capture_settings`, which records only what was asked for |
| `camera` | which lens, because the sensor-to-camera mapping differs between front and back and no downstream check can catch the substitution |

Two honesty rules hold in there. `arc` is a cross-check and never the answer — the published view
cone is measured offline from the cameras structure-from-motion actually solved, and an on-device
figure may only ever narrow it. And a take that aborted before the hold produced any frames is
**not** labelled `fixed_camera_4d`, because it contains none of the footage that label means; it is
recorded as the static scene it actually is, with `capture_type_intended` saying what was asked
for.

The schema and the reasoning behind each field are in `docs/APP_SPEC.md` and
`src/gausscapture/schemas/capturepack.schema.json`; what the previous version got wrong, and why,
is in `docs/ANDROID_AUDIT.md`.
