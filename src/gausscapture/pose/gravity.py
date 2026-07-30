"""Recovering which way is up from the accelerometer, not from a guess.

Structure-from-motion has no notion of vertical. COLMAP fixes the world frame on
whichever image it started from, so heights, orbiting and "the floor" all come
out at whatever angle the first frame happened to sit at. Every viewer then
needs to know the true up direction, and there are three ways to get it:

1. **Average the camera orientations.** Assumes the phone was held upright.
2. **Fit a plane to the camera path.** Assumes the operator walked a level loop.
3. **Read the accelerometer.** Assumes nothing; gravity is a measurement.

Measured on this project's three captures, the first is off by 41.3°, 3.6° and
9.1°, and the second by 8.7°, 81.3° and 15.4°. Each is occasionally excellent
and occasionally catastrophic, and neither reports which case it is in. Gravity
is right every time and says how sure it is.

The one unknown is the fixed rotation between the sensor axes and the camera
axes. Android specifies it — sensor +y is up the screen where camera +y is down
the image, and sensor +z faces the user where camera +z faces the scene, giving
``diag(1, -1, -1)`` — so it is applied as a known constant rather than fitted.

It is nonetheless *checked*, because the check is nearly free and catches a
mis-set device orientation. Whatever the true rotation is, world gravity must
come out the same for every frame, so agreement across frames — the mean
resultant length of the per-frame estimates — scores a candidate. That number
doubles as a confidence score: it approaches 1 only if the camera poses, the
sensor mapping and the video-to-IMU clock alignment are all right at once. On
this project's captures it is 0.998 or better.

Agreement alone cannot *choose* the mapping, though, and it is worth saying why,
because the failure it hides is the ugly one. ``diag(1, -1, -1)`` is a half turn
about the camera's x axis. When a phone is held level and walked around a
subject, that axis is horizontal and square to gravity, so the half turn maps
the true up exactly onto its own negative — consistently, in every frame. The
identity therefore scores just as well as the truth and returns the scene upside
down. On a perfectly circular synthetic capture the two tie to machine
precision; on the real captures here the correct mapping led by only 0.0013 to
0.0052, which is noise. So the constant is trusted, and an alternative has to
beat it by :data:`DECISIVE_MARGIN` before it is believed.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Sensor type string Android reports for the gravity-inclusive accelerometer.
ACCELEROMETER = "android.sensor.accelerometer"

#: Android's sensor axes expressed in the camera's: +x agrees, +y and +z flip.
#: Trusted as a constant rather than fitted, for the reason in the module
#: docstring -- fitting it cannot tell up from down.
SENSOR_TO_CAMERA = np.diag([1.0, -1.0, -1.0])

#: How far another candidate must beat the known mapping before it is believed.
#: The spurious runner-up leads by under 0.006 on real captures, so this sits an
#: order of magnitude above the noise and only a genuinely different device
#: orientation clears it.
DECISIVE_MARGIN = 0.05

#: Earth gravity, and how far a capture's median reading may sit from it. A
#: stream that fails this is not an accelerometer reporting metres per second
#: squared, so nothing about it should be read as gravity.
STANDARD_GRAVITY = 9.81
GRAVITY_TOLERANCE = 1.0

#: How far a sample's magnitude may sit from the capture's median before it is
#: dropped. A phone being carried is never perfectly still, and a reading that
#: contains real acceleration is not a reading of gravity.
STATIONARY_TOLERANCE = 1.5

#: Below this agreement the estimate is not trustworthy: something in the pose
#: chain or the clock alignment is wrong, and aligning to it would be worse than
#: declining and letting a caller fall back.
MIN_CONSISTENCY = 0.90

#: Fewer frames than this and the average is dominated by whichever handful
#: happened to survive the stationarity filter.
MIN_FRAMES = 5


@dataclass(frozen=True)
class GravityUp:
    """A measured up direction, with the evidence for it."""

    up: np.ndarray               # unit vector in the reconstruction's world frame
    consistency: float           # mean resultant length in [0, 1]; 1 = total agreement
    frames: int                  # frames that contributed
    sensor_to_camera: np.ndarray  # the (3, 3) sensor-to-camera rotation used
    margin: float = 0.0           # how far it beat the best alternative

    def describe(self) -> str:
        axes = " ".join(
            ("-" if row[int(np.argmax(np.abs(row)))] < 0 else "+") + "xyz"[int(np.argmax(np.abs(row)))]
            for row in self.sensor_to_camera
        )
        return (
            f"up {np.round(self.up, 3).tolist()} from {self.frames} frames, "
            f"agreement {self.consistency:.3f}, sensor axes {axes}, "
            f"margin {self.margin:+.4f}"
        )


def up_from_gravity(project_dir: Path, model_dir: Path) -> GravityUp | None:
    """Measure the world up direction of a reconstruction from the capture's IMU.

    Returns ``None`` when the capture carries no usable accelerometer data, when
    too few frames survive the stationarity filter, or when the per-frame
    estimates disagree -- all cases where a caller is better off with a fallback
    than with a confident wrong answer.
    """
    from gausscapture.errors import CaptureFormatError
    from gausscapture.pose.model import read_model

    project_dir, model_dir = Path(project_dir), Path(model_dir)

    samples = _read_accelerometer(project_dir / "capturepack" / "imu.jsonl")
    if samples is None:
        return None
    timestamps = _frame_timestamps(project_dir)
    if not timestamps:
        return None

    try:
        model = read_model(model_dir)
    except CaptureFormatError:
        return None

    times, vectors = samples
    # A sample interpolated outside the IMU's own span is an extrapolation
    # numpy performs silently by clamping, so those frames are excluded.
    span = (times[0], times[-1])
    magnitudes = np.linalg.norm(vectors, axis=1)
    nominal = float(np.median(magnitudes))
    # The median absorbs a small sensor bias, but it must still land on gravity;
    # a capture that was accelerating throughout, or a stream in the wrong
    # units, has nothing in it to read as vertical.
    if abs(nominal - STANDARD_GRAVITY) > GRAVITY_TOLERANCE:
        return None

    rotations: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    for image in model.images.values():
        stamp = timestamps.get(image.name)
        if stamp is None or not span[0] <= stamp <= span[1]:
            continue
        reading = np.array([np.interp(stamp, times, vectors[:, i]) for i in range(3)])
        magnitude = float(np.linalg.norm(reading))
        if abs(magnitude - nominal) > STATIONARY_TOLERANCE or magnitude == 0:
            continue
        rotations.append(image.camera_to_world()[:3, :3])
        directions.append(reading / magnitude)

    if len(rotations) < MIN_FRAMES:
        return None

    def score(candidate: np.ndarray) -> tuple[float, np.ndarray]:
        # The accelerometer measures specific force, so a resting phone reads
        # *upward* -- the vector is already up, with no sign to flip.
        world = np.array([r @ (candidate @ d) for r, d in zip(rotations, directions)])
        mean = world.mean(axis=0)
        length = float(np.linalg.norm(mean))
        return length, (mean / length if length else mean)

    known, up = score(SENSOR_TO_CAMERA)
    rival = max(
        (score(c)[0] for c in _axis_rotations() if not np.allclose(c, SENSOR_TO_CAMERA)),
        default=0.0,
    )

    chosen, consistency, margin = SENSOR_TO_CAMERA, known, known - rival
    if rival > known + DECISIVE_MARGIN:
        # Something about this device or its recording orientation differs from
        # what Android documents, by more than noise could explain.
        chosen = max(
            (c for c in _axis_rotations()), key=lambda c: score(c)[0]
        )
        consistency, up = score(chosen)
        margin = consistency - known

    if consistency < MIN_CONSISTENCY:
        return None
    return GravityUp(
        up=up.astype(np.float32),
        consistency=consistency,
        frames=len(rotations),
        sensor_to_camera=chosen,
        margin=margin,
    )


def _axis_rotations() -> list[np.ndarray]:
    """The 24 rotations that map axes onto axes.

    A camera is bolted to a phone in some fixed orientation, so the sensor-to-
    camera transform is one of these by construction. Searching the discrete set
    rather than solving for a free rotation keeps the answer from absorbing
    sensor noise and pose error into a plausible-looking tilt.
    """
    out = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            matrix = np.zeros((3, 3))
            for row, (column, sign) in enumerate(zip(permutation, signs)):
                matrix[row, column] = sign
            if np.isclose(np.linalg.det(matrix), 1.0):  # reflections are not poses
                out.append(matrix)
    return out


def _read_accelerometer(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Read accelerometer samples as (timestamps, vectors), sorted by time."""
    if not path.exists():
        return None

    times: list[int] = []
    vectors: list[list[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if ACCELEROMETER not in line:  # cheaper than parsing every record
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") != ACCELEROMETER:
                continue
            value = record.get("v")
            stamp = record.get("t_ns")
            if stamp is None or not isinstance(value, list) or len(value) != 3:
                continue
            times.append(int(stamp))
            vectors.append([float(v) for v in value])

    if len(times) < 2:
        return None
    order = np.argsort(np.asarray(times))
    return np.asarray(times, dtype=np.float64)[order], np.asarray(vectors)[order]


def _frame_timestamps(project_dir: Path) -> dict[str, int]:
    """Map each extracted image's filename to its hardware capture timestamp.

    The frame index records where in the video each image came from, and the
    capture pack records a hardware timestamp per video frame. Going through the
    video frame number rather than scaling seconds keeps the two in step: both
    sides then refer to the same frame, so a wrong frame rate cannot quietly
    shear the alignment across the capture.
    """
    index_path = project_dir / "frames" / "frame_index.json"
    frames_path = project_dir / "capturepack" / "frames.jsonl"
    if not index_path.exists() or not frames_path.exists():
        return {}

    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}

    fps = float(index.get("source_fps") or 0)
    if fps <= 0:
        return {}

    by_frame: dict[int, int] = {}
    with frames_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            number, stamp = record.get("frame"), record.get("t_ns")
            if number is not None and stamp is not None:
                by_frame[int(number)] = int(stamp)

    out: dict[str, int] = {}
    for entry in index.get("frames", []):
        file = entry.get("file")
        seconds = entry.get("timestamp_sec")
        if file is None or seconds is None:
            continue
        stamp = by_frame.get(int(round(float(seconds) * fps)))
        if stamp is not None:
            out[Path(file).name] = stamp
    return out
