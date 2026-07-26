# CapturePack 0.1

`.capturepack` is a zip archive with a custom extension. `manifest.json` is required. The main video is required. Metadata files are optional and produce warnings when absent.

```text
session.capturepack
  manifest.json
  video/main_video.mp4
  frames/
  camera/intrinsics.json
  camera/lens_profile.json
  motion/arkit_arcore_poses.json
  motion/imu_gyro.json
  environment/light_estimation.json
  quality/capture_warnings.json
  checksums/sha256.json
```

Minimal phone video imports create this structure automatically. Future mobile apps can add intrinsics, pose logs, IMU logs, exposure, white balance, light estimation, audio sync, storage mode, and capture warnings without changing the import flow.

