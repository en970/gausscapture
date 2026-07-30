"""Recovering the up direction from a capture's accelerometer.

These build a synthetic capture with a *known* up direction, so the tests check
recovery against ground truth rather than against whatever the code happens to
produce.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from gausscapture.pose.gravity import (
    MIN_CONSISTENCY,
    SENSOR_TO_CAMERA,
    _axis_rotations,
    up_from_gravity,
)

FPS = 30.0
GRAVITY = 9.81


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """A camera-to-world rotation for a camera at ``eye`` looking at ``target``.

    Built in the vision convention -- +z forward, +y down -- because that is
    what COLMAP stores and what the estimator has to cope with.
    """
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.column_stack([right, down, forward])


def _roll(angle: float) -> np.ndarray:
    """Rotation about the camera's own forward axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _write_capture(
    root: Path,
    up: np.ndarray,
    frames: int = 24,
    noise: float = 0.0,
    stationary: bool = True,
    roll: float = 0.0,
    seed: int = 0,
) -> None:
    """Write a project whose true world up direction is ``up``.

    ``roll`` tilts the phone about its own optical axis by that many radians on
    average. A real operator holds a phone with some habitual tilt, and it is
    exactly that bias which survives averaging and defeats the camera-orientation
    heuristic while leaving the accelerometer unaffected.
    """
    rng = np.random.default_rng(seed)
    up = np.asarray(up, dtype=float)
    up = up / np.linalg.norm(up)

    # Two directions spanning the plane the operator walks in.
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, up)) > 0.9:
        a = np.array([0.0, 0.0, 1.0])
    a = a - np.dot(a, up) * up
    a = a / np.linalg.norm(a)
    b = np.cross(up, a)

    pack = root / "capturepack"
    pack.mkdir(parents=True, exist_ok=True)
    (root / "frames").mkdir(parents=True, exist_ok=True)
    sparse = root / "colmap" / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)

    images: list[tuple[str, np.ndarray, np.ndarray]] = []
    imu_lines: list[str] = []
    frame_lines: list[str] = []
    index_frames: list[dict] = []

    for i in range(frames):
        angle = 2 * np.pi * i / frames
        eye = 4.0 * (np.cos(angle) * a + np.sin(angle) * b) + 1.2 * up
        rotation = _look_at(eye, np.zeros(3), up)
        if roll:
            rotation = rotation @ _roll(roll + rng.normal(0, 0.05))

        name = f"frame_{i:06d}.jpg"
        images.append((name, rotation, eye))

        video_frame = i * 5
        stamp = 1_000_000_000 + video_frame * int(1e9 / FPS)
        frame_lines.append(json.dumps({"frame": video_frame, "t_ns": stamp}))
        index_frames.append(
            {"file": f"frames/images/{name}", "timestamp_sec": video_frame / FPS}
        )

        # The accelerometer reads specific force, so at rest it points *up*,
        # expressed in the phone's own axes.
        reading = SENSOR_TO_CAMERA.T @ (rotation.T @ up) * GRAVITY
        if noise:
            reading = reading + rng.normal(0, noise, 3)
        if not stationary:
            reading = reading * 2.5
        for offset in (-2_000_000, 0, 2_000_000):  # bracket the frame in time
            imu_lines.append(
                json.dumps(
                    {
                        "type": "android.sensor.accelerometer",
                        "t_ns": stamp + offset,
                        "v": [float(v) for v in reading],
                    }
                )
            )

    (pack / "imu.jsonl").write_text("\n".join(imu_lines), encoding="utf-8")
    (pack / "frames.jsonl").write_text("\n".join(frame_lines), encoding="utf-8")
    (root / "frames" / "frame_index.json").write_text(
        json.dumps({"source_fps": FPS, "frames": index_frames}), encoding="utf-8"
    )
    _write_colmap(sparse, images)


def _write_colmap(sparse: Path, images: list[tuple[str, np.ndarray, np.ndarray]]) -> None:
    """Write the minimal binary model the reader needs."""
    with (sparse / "cameras.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<iiQQ", 1, 1, 640, 480))       # PINHOLE
        handle.write(struct.pack("<dddd", 500.0, 500.0, 320.0, 240.0))

    with (sparse / "images.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", len(images)))
        for i, (name, rotation, centre) in enumerate(images, start=1):
            # COLMAP stores world-to-camera; the fixtures are built the other way.
            world_to_camera = rotation.T
            translation = -world_to_camera @ centre
            handle.write(struct.pack("<I", i))
            handle.write(struct.pack("<dddd", *_quaternion(world_to_camera)))
            handle.write(struct.pack("<ddd", *translation))
            handle.write(struct.pack("<I", 1))
            handle.write(name.encode() + b"\x00")
            handle.write(struct.pack("<Q", 0))

    with (sparse / "points3D.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 0))


def _quaternion(m: np.ndarray) -> tuple[float, float, float, float]:
    trace = np.trace(m)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return (0.25 / s, (m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s,
                (m[1, 0] - m[0, 1]) * s)
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k])
    q = [0.0, 0.0, 0.0, 0.0]
    q[0] = (m[k, j] - m[j, k]) / s
    q[i + 1] = 0.25 * s
    q[j + 1] = (m[j, i] + m[i, j]) / s
    q[k + 1] = (m[k, i] + m[i, k]) / s
    return tuple(q)


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(a, b)), -1, 1))))


class TestAxisRotations:
    def test_there_are_twenty_four(self):
        assert len(_axis_rotations()) == 24

    def test_all_are_proper_rotations(self):
        # A reflection would map the sensor onto a mirrored camera, which no
        # physical mounting can do.
        for matrix in _axis_rotations():
            assert np.isclose(np.linalg.det(matrix), 1.0)
            assert np.allclose(matrix @ matrix.T, np.eye(3))

    def test_the_android_mapping_is_among_them(self):
        assert any(np.allclose(m, SENSOR_TO_CAMERA) for m in _axis_rotations())


class TestUpFromGravity:
    @pytest.mark.parametrize(
        "up",
        [
            (0.0, -1.0, 0.0),         # COLMAP's own convention
            (0.0, 1.0, 0.0),
            (-0.012, -0.670, -0.742),  # the direction measured on a real capture
            (0.577, 0.577, 0.577),
        ],
    )
    def test_recovers_a_known_direction(self, tmp_path, up):
        _write_capture(tmp_path, np.array(up))
        result = up_from_gravity(tmp_path, tmp_path / "colmap" / "sparse" / "0")

        assert result is not None
        assert _angle(result.up, np.array(up) / np.linalg.norm(up)) < 1.0
        assert result.consistency > 0.99

    def test_recovers_the_sensor_mapping(self, tmp_path):
        _write_capture(tmp_path, np.array([0.0, -1.0, 0.0]))
        result = up_from_gravity(tmp_path, tmp_path / "colmap" / "sparse" / "0")

        assert result is not None
        assert np.allclose(result.sensor_to_camera, SENSOR_TO_CAMERA)

    def test_survives_realistic_sensor_noise(self, tmp_path):
        _write_capture(tmp_path, np.array([0.1, -0.9, 0.3]), noise=0.25)
        result = up_from_gravity(tmp_path, tmp_path / "colmap" / "sparse" / "0")

        assert result is not None
        assert _angle(result.up, np.array([0.1, -0.9, 0.3]) / np.linalg.norm([0.1, -0.9, 0.3])) < 3.0

    def test_declines_when_the_stream_is_not_gravity(self, tmp_path):
        # A capture whose readings never settle near 9.81 was either
        # accelerating throughout or is not in metres per second squared;
        # either way there is no vertical in it.
        _write_capture(tmp_path, np.array([0.0, -1.0, 0.0]), stationary=False)
        assert up_from_gravity(tmp_path, tmp_path / "colmap" / "sparse" / "0") is None

    def test_declines_without_an_imu(self, tmp_path):
        _write_capture(tmp_path, np.array([0.0, -1.0, 0.0]))
        (tmp_path / "capturepack" / "imu.jsonl").unlink()
        assert up_from_gravity(tmp_path, tmp_path / "colmap" / "sparse" / "0") is None

    def test_declines_when_frames_cannot_be_timed(self, tmp_path):
        _write_capture(tmp_path, np.array([0.0, -1.0, 0.0]))
        (tmp_path / "frames" / "frame_index.json").unlink()
        assert up_from_gravity(tmp_path, tmp_path / "colmap" / "sparse" / "0") is None

    def test_declines_when_the_clock_does_not_overlap(self, tmp_path):
        # A capture whose IMU and video are on different clocks would otherwise
        # interpolate to a clamped edge sample and look confident about it.
        _write_capture(tmp_path, np.array([0.0, -1.0, 0.0]))
        imu = tmp_path / "capturepack" / "imu.jsonl"
        shifted = [
            json.dumps({**json.loads(line), "t_ns": json.loads(line)["t_ns"] + 10**15})
            for line in imu.read_text().splitlines()
        ]
        imu.write_text("\n".join(shifted), encoding="utf-8")
        assert up_from_gravity(tmp_path, tmp_path / "colmap" / "sparse" / "0") is None

    def test_declines_when_frames_disagree(self, tmp_path):
        _write_capture(tmp_path, np.array([0.0, -1.0, 0.0]))
        imu = tmp_path / "capturepack" / "imu.jsonl"
        rng = np.random.default_rng(3)
        scrambled = []
        for line in imu.read_text().splitlines():
            record = json.loads(line)
            direction = rng.normal(size=3)
            record["v"] = list(direction / np.linalg.norm(direction) * GRAVITY)
            scrambled.append(json.dumps(record))
        imu.write_text("\n".join(scrambled), encoding="utf-8")

        result = up_from_gravity(tmp_path, tmp_path / "colmap" / "sparse" / "0")
        assert result is None or result.consistency >= MIN_CONSISTENCY


class TestWorldUp:
    def test_prefers_gravity_over_the_camera_heuristic(self, tmp_path):
        from gausscapture.pose.orientation import up_from_cameras, world_up

        # A capture filmed with the phone pitched down: the cameras' own up
        # direction is tilted away from the world's, which is exactly the case
        # the heuristic gets wrong and gravity gets right.
        truth = np.array([0.0, -0.866, -0.5])
        _write_capture(tmp_path, truth, roll=0.45)
        model = tmp_path / "colmap" / "sparse" / "0"

        measured = world_up(model)
        heuristic = up_from_cameras(model)

        assert measured is not None
        assert _angle(measured, truth) < 1.0
        assert _angle(measured, truth) < _angle(heuristic, truth)

    def test_falls_back_when_the_imu_is_missing(self, tmp_path):
        from gausscapture.pose.orientation import world_up

        _write_capture(tmp_path, np.array([0.0, -1.0, 0.0]))
        (tmp_path / "capturepack" / "imu.jsonl").unlink()

        assert world_up(tmp_path / "colmap" / "sparse" / "0") is not None

    def test_infers_the_project_from_the_model_path(self, tmp_path):
        from gausscapture.pose.gravity import up_from_gravity
        from gausscapture.pose.orientation import world_up

        _write_capture(tmp_path, np.array([0.2, -0.9, 0.1]))
        model = tmp_path / "colmap" / "sparse" / "0"

        # No project directory passed, so it has to be found from the model.
        assert np.allclose(world_up(model), up_from_gravity(tmp_path, model).up)
