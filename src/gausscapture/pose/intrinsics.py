"""Turning device-reported intrinsics into the frame's own pixel coordinates.

Android reports camera intrinsics against the sensor's full pre-correction
active array. What we record is a cropped, scaled and often rotated view of
that array, so the reported numbers describe a coordinate system no frame is
ever in. Three transforms stand between them, and each is easy to get subtly
wrong in a way nothing complains about:

1. **Aspect crop.** Widescreen video takes a 16:9 window out of a 4:3 sensor by
   discarding rows, top and bottom. The principal point moves; the focal length
   does not.
2. **Scale.** The cropped window is resampled to the recording size. Focal
   length and principal point both scale; distortion coefficients, being
   defined on normalised coordinates, do not.
3. **Rotation.** A portrait recording stores a landscape frame plus a rotation
   flag, and the decoder applies it. A quarter turn swaps fx with fy and
   rearranges the principal point.

Getting any of these wrong produces intrinsics that look plausible -- the right
order of magnitude, near the middle of the frame -- and quietly bias every
reconstruction that uses them. The check that catches it is cheap and worth
running: a correctly transformed principal point lands close to the frame
centre, because that is where lens centres nearly are.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CameraIntrinsics:
    """Intrinsics in the pixel coordinates of a decoded frame."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0

    def colmap_params(self) -> list[float]:
        """Parameters for COLMAP's ``OPENCV`` model, in its expected order."""
        return [self.fx, self.fy, self.cx, self.cy, self.k1, self.k2, self.p1, self.p2]

    def colmap_camera_params(self) -> str:
        """The comma-separated form ``--ImageReader.camera_params`` expects."""
        return ",".join(f"{v:.6f}" for v in self.colmap_params())

    def principal_point_offset(self) -> tuple[float, float]:
        """How far the principal point sits from the frame centre, in pixels.

        A sanity check rather than a correction. Real lens centres are within a
        few percent of the frame centre, so a large offset means the transform
        chain above went wrong, not that the lens is unusual.
        """
        return (self.cx - self.width / 2, self.cy - self.height / 2)

    def looks_plausible(self, tolerance: float = 0.08) -> bool:
        """Whether the principal point is near enough the centre to be believed."""
        dx, dy = self.principal_point_offset()
        return (
            abs(dx) <= tolerance * self.width
            and abs(dy) <= tolerance * self.height
            and self.fx > 0
            and self.fy > 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "k1": self.k1,
            "k2": self.k2,
            "p1": self.p1,
            "p2": self.p2,
            "width": self.width,
            "height": self.height,
        }


def rotate(intrinsics: CameraIntrinsics, degrees: int) -> CameraIntrinsics:
    """Rotate intrinsics clockwise to match a rotated frame.

    Android's ``setOrientationHint(90)`` means "display this rotated 90 degrees
    clockwise", and decoders comply, so the frames we analyse are the sensor
    frame turned that far. Under a clockwise quarter turn of a ``w x h`` image,
    source pixel ``(x, y)`` lands at ``(h - y, x)``, which is where the
    principal-point arithmetic below comes from.
    """
    degrees %= 360
    if degrees == 0:
        return intrinsics

    w, h = intrinsics.width, intrinsics.height
    common = {
        "k1": intrinsics.k1,
        "k2": intrinsics.k2,
        # Tangential terms follow the axes, so a quarter turn exchanges them.
        # They are almost always zero on phone cameras, but exchanging them
        # costs nothing and being silently wrong when they are not is expensive.
        "p1": intrinsics.p2 if degrees in (90, 270) else intrinsics.p1,
        "p2": intrinsics.p1 if degrees in (90, 270) else intrinsics.p2,
    }

    if degrees == 90:
        return CameraIntrinsics(
            fx=intrinsics.fy, fy=intrinsics.fx,
            cx=h - intrinsics.cy, cy=intrinsics.cx,
            width=h, height=w, **common,
        )
    if degrees == 180:
        return CameraIntrinsics(
            fx=intrinsics.fx, fy=intrinsics.fy,
            cx=w - intrinsics.cx, cy=h - intrinsics.cy,
            width=w, height=h, **common,
        )
    if degrees == 270:
        return CameraIntrinsics(
            fx=intrinsics.fy, fy=intrinsics.fx,
            cx=intrinsics.cy, cy=w - intrinsics.cx,
            width=h, height=w, **common,
        )
    raise ValueError(f"Rotation must be a multiple of 90, got {degrees}")


def from_device(
    intrinsics_json: dict[str, Any],
    frame_width: int,
    frame_height: int,
    rotation: int = 0,
) -> CameraIntrinsics | None:
    """Convert an ``intrinsics.json`` written by the capture app to frame pixels.

    ``frame_width`` and ``frame_height`` are the dimensions of the frames as
    decoded -- already rotated. Returns ``None`` when the device did not report
    a calibration, which is common enough that it is a normal outcome rather
    than an error.
    """
    calibration = intrinsics_json.get("intrinsic_calibration")
    active = intrinsics_json.get("pre_correction_active_array")
    if not calibration or not active:
        return None
    if not active.get("width") or not active.get("height"):
        return None

    # Work in the recording's own orientation, undoing the decoder's rotation.
    recorded_width, recorded_height = (
        (frame_height, frame_width) if rotation % 180 == 90 else (frame_width, frame_height)
    )
    if not recorded_width or not recorded_height:
        return None

    active_width = float(active["width"])
    active_height = float(active["height"])

    # Video keeps the sensor's full width and discards rows to reach its aspect
    # ratio, so the scale factor comes from the width and the crop from the
    # height. Deriving the scale from the height instead would fold the crop
    # into it and shrink the focal length by the aspect difference.
    scale = recorded_width / active_width
    cropped_height = recorded_height / scale
    crop_top = max(0.0, (active_height - cropped_height) / 2)

    distortion = intrinsics_json.get("distortion_kappa") or []
    # Android orders these [k1, k2, k3, p1, p2]; COLMAP's OPENCV model has no
    # k3, so it is dropped rather than mistaken for a tangential term.
    k1 = float(distortion[0]) if len(distortion) > 0 else 0.0
    k2 = float(distortion[1]) if len(distortion) > 1 else 0.0
    p1 = float(distortion[3]) if len(distortion) > 3 else 0.0
    p2 = float(distortion[4]) if len(distortion) > 4 else 0.0

    unrotated = CameraIntrinsics(
        fx=float(calibration["fx"]) * scale,
        fy=float(calibration["fy"]) * scale,
        cx=float(calibration["cx"]) * scale,
        cy=(float(calibration["cy"]) - crop_top) * scale,
        width=int(recorded_width),
        height=int(recorded_height),
        # Distortion is defined on normalised coordinates, so cropping and
        # scaling leave it alone.
        k1=k1, k2=k2, p1=p1, p2=p2,
    )
    return rotate(unrotated, rotation)


def from_pack(pack_dir, frame_width: int, frame_height: int, rotation: int = 0):
    """Read a capture pack's ``intrinsics.json`` and convert it, if present."""
    import json
    from pathlib import Path

    path = Path(pack_dir) / "intrinsics.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return from_device(data, frame_width, frame_height, rotation)


def for_project(project_path) -> CameraIntrinsics | None:
    """Device intrinsics matching a project's *extracted frames*.

    Frame dimensions are measured from an extracted image rather than taken
    from the manifest, and the rotation is probed from the video rather than
    trusted, because both are places this project has already been bitten by
    metadata that disagreed with the file.

    Returns ``None`` whenever anything is missing or implausible -- a capture
    without device intrinsics is the normal case, not a failure, and seeding
    structure-from-motion with numbers we cannot vouch for is worse than
    letting it solve them.
    """
    from pathlib import Path

    import cv2

    project_path = Path(project_path)
    frames = sorted((project_path / "frames" / "images").glob("*.jpg"))
    if not frames:
        return None
    sample = cv2.imread(str(frames[0]))
    if sample is None:
        return None
    height, width = sample.shape[:2]

    pack_dir = project_path / "capturepack"
    rotation = 0
    try:
        from gausscapture.ingest.video import probe_video
        from gausscapture.pack.manifest import find_main_video

        rotation = probe_video(find_main_video(pack_dir)).rotation
    except Exception:  # noqa: BLE001 - absent or unreadable video is not fatal here
        rotation = 0

    intrinsics = from_pack(pack_dir, width, height, rotation)
    if intrinsics is None or not intrinsics.looks_plausible():
        return None
    return intrinsics
