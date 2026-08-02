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

```
lib/                       Dart: interface only
  design.dart              palette, type scale, spacing — derived in docs/APP_SPEC.md §2
  capture_engine.dart      the channel wrapper and the data model
  screens/                 capture, takes
  widgets/                 coach, perimeter, record button, shot picker
android/app/src/main/java/com/gausscapture/capture/
  MainActivity.java        FlutterActivity; three channels and nothing else
  PreviewFactory.java      the preview as a platform view
  CaptureEngine.java       what a capture is, with no interface attached
  CameraController.java    Camera2, per-frame metadata, intrinsics
  SensorLogger.java        five sensors on a thread of their own
  JsonlWriter.java         double-buffered writer; never blocks a producer
  Clocks.java              measures which clock the sensors are on
  Storage.java             where captures live and whether a Mac can see them
  Preset.java              the five capture protocols
```

## Building

```sh
export JAVA_HOME=/opt/homebrew/opt/openjdk@17
flutter build apk --debug          # or: flutter run
```

## Getting captures onto a computer

Captures go to `Documents/GaussCapture`, which appears over MTP. That needs all-files access; the
app asks once and falls back to its private directory if refused. Either way:

```sh
gausscapture pull            # over adb, works regardless of where they landed
```

## What is recorded

Per take: `video.mp4`, `frames.jsonl` (one row per capture result), `imu.jsonl` (five sensors),
`intrinsics.json`, `manifest.json`. The schema and the reasoning behind each field are in
`docs/APP_SPEC.md`; what the previous version got wrong, and why, is in `docs/ANDROID_AUDIT.md`.
