"""Synthetic inputs for the fixed-camera 4D tests.

Everything here is generated: a four-phase video whose motion is known to the
frame, and a COLMAP text model whose cameras are placed on an arc of a known
angle. No fixture binaries, no phone, no COLMAP, no GPU -- which is the only way
this part of the pipeline can be tested at all, since its whole job is to turn a
recording nobody has made yet into a dataset nobody has trained on yet.

The video is built to exercise the three motion regimes the phase splitter
distinguishes, and nothing else: a still opening, a pan that moves every pixel,
a still stretch, and finally a still frame with one bright rectangle swinging
across it. That last part is the important one -- it is a subject moving in
front of a camera that is not, which is the entire premise of the capture.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class PhaseVideo:
    """A synthetic four-phase capture and the ground truth about it."""

    path: Path
    width: int
    height: int
    fps: float
    bounds: dict[str, tuple[int, int]] = field(default_factory=dict)
    #: Pixel bounding box the moving rectangle sweeps out, as (x0, y0, x1, y1).
    blob_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)

    @property
    def total_frames(self) -> int:
        return max(end for _, end in self.bounds.values()) + 1

    def protocol_block(self) -> dict:
        """The manifest block a capturepack 0.3 app would have written."""
        reseat_start, reseat_end = self.bounds["reseat"]
        return {
            "name": "F_bullet",
            "fps": self.fps,
            "phases": [
                {"name": name, "start_frame": start, "end_frame": end}
                for name, (start, end) in self.bounds.items()
            ],
            "rest_window": {
                "start_frame": reseat_end - 9,
                "end_frame": reseat_end,
            },
            "auto_stopped": True,
        }


def make_phase_video(
    path: Path,
    size: tuple[int, int] = (160, 120),
    fps: int = 20,
    perch: int = 20,
    arc: int = 40,
    reseat: int = 20,
    hold: int = 40,
    blob: int = 36,
    seed: int = 20260807,
) -> PhaseVideo:
    """Write a four-phase capture whose phase boundaries are known exactly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    background = _texture(height, width * 2, seed)

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    pan_step = max(1, width // arc)

    # Perch: the phone in its support, nothing moving.
    for _ in range(perch):
        writer.write(_crop(background, 0, width))

    # Arc: the operator sweeping. Every pixel moves, which is what separates
    # camera motion from subject motion without any model of either.
    offset = 0
    for i in range(arc):
        offset = min(width, (i + 1) * pan_step)
        writer.write(_crop(background, offset, width))

    # Reseat: the phone back down, hands off, subject not yet cued. These frames
    # are the rest window -- the only ones that measure the sensor rather than
    # the scene.
    for _ in range(reseat):
        writer.write(_crop(background, offset, width))

    # Hold: camera at rest, one bright rectangle swinging across the frame.
    centre_x, centre_y = width // 2, height // 2
    amplitude = 35
    x0s, x1s = [], []
    for i in range(hold):
        frame = _crop(background, offset, width)
        shift = int(round(amplitude * np.sin(2.0 * np.pi * i / 20.0)))
        x = centre_x + shift - blob // 2
        y = centre_y - blob // 2
        cv2.rectangle(frame, (x, y), (x + blob, y + blob), (250, 250, 250), thickness=-1)
        x0s.append(x)
        x1s.append(x + blob)
        writer.write(frame)
    writer.release()

    bounds = {
        "perch": (0, perch - 1),
        "arc": (perch, perch + arc - 1),
        "reseat": (perch + arc, perch + arc + reseat - 1),
        "hold": (perch + arc + reseat, perch + arc + reseat + hold - 1),
    }
    return PhaseVideo(
        path=path,
        width=width,
        height=height,
        fps=float(fps),
        bounds=bounds,
        blob_bbox=(min(x0s), centre_y - blob // 2, max(x1s), centre_y - blob // 2 + blob),
    )


def _texture(height: int, width: int, seed: int) -> np.ndarray:
    """A blocky, well-compressing pattern with real gradients.

    Per-pixel noise would be shredded by the codec and every static frame would
    then differ from the last, which is precisely the thing the phase splitter
    measures. Big blocks survive compression, so a still frame stays still.
    """
    rng = np.random.default_rng(seed)
    coarse = rng.integers(20, 180, size=(height // 8 + 1, width // 8 + 1, 3), dtype=np.uint8)
    return cv2.resize(coarse, (width, height), interpolation=cv2.INTER_NEAREST)


def _crop(texture: np.ndarray, offset: int, width: int) -> np.ndarray:
    return texture[:, offset : offset + width].copy()


def write_pack(
    project_path: Path,
    video: PhaseVideo,
    declare_protocol: bool = True,
    capture_type: str = "fixed_camera_4d",
    extra: dict | None = None,
) -> Path:
    """Lay out a CapturePack around a synthetic video."""
    pack_dir = Path(project_path) / "capturepack"
    (pack_dir / "video").mkdir(parents=True, exist_ok=True)
    destination = pack_dir / "video" / "main_video.mp4"
    destination.write_bytes(video.path.read_bytes())

    payload = {
        "capturepack_version": "0.3",
        "session_id": "synthetic-4d-session",
        "session_name": "synthetic",
        "capture_type": capture_type,
        "created_at": "2026-08-07T00:00:00+00:00",
        "video": {
            "main_file": "video/main_video.mp4",
            "width": video.width,
            "height": video.height,
            "fps": video.fps,
            "duration_sec": video.total_frames / video.fps,
        },
        "camera": {"lens_facing": "front"},
    }
    if declare_protocol:
        payload["protocol"] = video.protocol_block()
    if extra:
        payload.update(extra)
    (pack_dir / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return pack_dir


# --------------------------------------------------------------------------
# synthetic reconstructions
# --------------------------------------------------------------------------


@dataclass
class SyntheticCamera:
    """A pinhole camera in the units the COLMAP text dialect wants.

    The defaults match the frame size the 4D tests render (``TINY`` in
    ``test_pipeline4d``: 120x90) and put the principal point at its centre.
    They used to say 160x120 with the centre at (80, 60), which is a camera
    describing an image 1.333x larger than the pixels it was used against, with
    the principal point 20 px and 15 px off in a 120-pixel-wide frame. Nothing
    downstream raised on it, which is exactly why the fixture has to be
    self-consistent: a test that feeds the pipeline mismatched intrinsics is
    measuring the pipeline's tolerance for its own fixture's bug.
    """

    fx: float = 150.0
    fy: float = 150.0
    cx: float = 60.0
    cy: float = 45.0
    width: int = 120
    height: int = 90

    def params_line(self) -> str:
        # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2. Zero distortion, because a
        # synthetic capture has none and the tests are about geometry.
        return f"{self.fx} {self.fy} {self.cx} {self.cy} 0 0 0 0"

    def unproject(self, u: float, v: float, depth: float) -> np.ndarray:
        return np.array(
            [(u - self.cx) / self.fx * depth, (v - self.cy) / self.fy * depth, depth],
            dtype=np.float64,
        )


def look_at(
    eye: np.ndarray, target: np.ndarray, up: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """World-to-camera (qvec, tvec) for a camera at ``eye`` looking at ``target``.

    Camera axes follow COLMAP and OpenCV: x right, y **down**, z forward.
    """
    up = np.array([0.0, 1.0, 0.0]) if up is None else np.asarray(up, dtype=np.float64)
    eye = np.asarray(eye, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - eye
    forward /= np.linalg.norm(forward)
    down = -up + float(up @ forward) * forward
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    rotation = np.stack([right, down, forward], axis=0)
    return rotation_to_quaternion(rotation), -rotation @ eye


def rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """(w, x, y, z) from a rotation matrix, via the largest-component branch."""
    m = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    quaternion = np.asarray(q, dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


def write_colmap_model(
    model_dir: Path,
    images: dict[str, tuple[np.ndarray, np.ndarray]],
    points: np.ndarray,
    camera: SyntheticCamera | None = None,
) -> Path:
    """Write a COLMAP text model: one shared camera, given poses, given points."""
    camera = camera or SyntheticCamera()
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    (model_dir / "cameras.txt").write_text(
        "# camera list\n"
        f"1 OPENCV {camera.width} {camera.height} {camera.params_line()}\n",
        encoding="utf-8",
    )

    lines = ["# image list"]
    for image_id, (name, (qvec, tvec)) in enumerate(sorted(images.items()), start=1):
        values = " ".join(f"{v:.12f}" for v in [*qvec, *tvec])
        lines.append(f"{image_id} {values} 1 {name}")
        lines.append("")
    (model_dir / "images.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    point_lines = ["# point list"]
    for point_id, position in enumerate(np.asarray(points, dtype=np.float64).reshape(-1, 3), 1):
        point_lines.append(
            f"{point_id} {position[0]:.9f} {position[1]:.9f} {position[2]:.9f} 200 200 200 0.5"
        )
    (model_dir / "points3D.txt").write_text("\n".join(point_lines) + "\n", encoding="utf-8")
    return model_dir


def arc_poses(
    count: int,
    radius: float = 2.0,
    azimuth_half_deg: float = 26.0,
    elevation_half_deg: float = 3.0,
    target: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cameras on a horizontal arc of a known angular extent about ``target``."""
    target = np.zeros(3) if target is None else np.asarray(target, dtype=np.float64)
    azimuths = np.linspace(-azimuth_half_deg, azimuth_half_deg, count)
    elevations = np.linspace(-elevation_half_deg, elevation_half_deg, count)
    poses = []
    for azimuth, elevation in zip(azimuths, elevations, strict=True):
        a = np.radians(azimuth)
        e = np.radians(elevation)
        eye = target + radius * np.array(
            [np.sin(a) * np.cos(e), np.sin(e), -np.cos(a) * np.cos(e)]
        )
        poses.append(look_at(eye, target))
    return poses


def fixed_pose(radius: float = 2.0, target: np.ndarray | None = None):
    """The pose the phone sat in: dead centre of the arc."""
    target = np.zeros(3) if target is None else np.asarray(target, dtype=np.float64)
    return look_at(target + np.array([0.0, 0.0, -radius]), target)


def jitter_pose(
    pose: tuple[np.ndarray, np.ndarray], degrees: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Perturb a pose by a small rotation, the way a solver would on noise."""
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.radians(degrees)
    delta = np.array(
        [
            np.cos(angle / 2),
            *(np.sin(angle / 2) * axis),
        ]
    )
    qvec, tvec = pose
    return _quaternion_multiply(delta, qvec), np.asarray(tvec, dtype=np.float64)


def _quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )
