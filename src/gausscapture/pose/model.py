"""Reading COLMAP sparse reconstructions.

COLMAP writes its models in either a text or a binary dialect, and the mapper
produces binary by default. Both are parsed here so that downstream code -- in
practice, ``transforms.json`` generation -- never has to care which one is on
disk, and so that GaussCapture does not need pycolmap (which drags in a build
toolchain) to read three small files.

The binary layout follows COLMAP's ``src/colmap/scene/reconstruction.cc``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gausscapture.errors import CaptureFormatError

#: COLMAP's model ids and their parameter names, in file order.
#: From ``src/colmap/sensor/models.h``.
CAMERA_MODELS: dict[int, tuple[str, tuple[str, ...]]] = {
    0: ("SIMPLE_PINHOLE", ("f", "cx", "cy")),
    1: ("PINHOLE", ("fx", "fy", "cx", "cy")),
    2: ("SIMPLE_RADIAL", ("f", "cx", "cy", "k1")),
    3: ("RADIAL", ("f", "cx", "cy", "k1", "k2")),
    4: ("OPENCV", ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2")),
    5: ("OPENCV_FISHEYE", ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4")),
    6: (
        "FULL_OPENCV",
        ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"),
    ),
    7: ("FOV", ("fx", "fy", "cx", "cy", "omega")),
    8: ("SIMPLE_RADIAL_FISHEYE", ("f", "cx", "cy", "k1")),
    9: ("RADIAL_FISHEYE", ("f", "cx", "cy", "k1", "k2")),
    10: (
        "THIN_PRISM_FISHEYE",
        ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1"),
    ),
}

_MODEL_IDS = {name: model_id for model_id, (name, _) in CAMERA_MODELS.items()}


@dataclass
class Camera:
    """An intrinsic calibration shared by one or more images."""

    id: int
    model: str
    width: int
    height: int
    params: dict[str, float]

    @property
    def fx(self) -> float:
        # The SIMPLE_* and *_FISHEYE models with a single focal length store it
        # as `f`; everything else stores fx and fy separately.
        return self.params.get("fx", self.params.get("f", 0.0))

    @property
    def fy(self) -> float:
        return self.params.get("fy", self.params.get("f", 0.0))

    @property
    def cx(self) -> float:
        return self.params.get("cx", self.width / 2)

    @property
    def cy(self) -> float:
        return self.params.get("cy", self.height / 2)

    def distortion(self) -> dict[str, float]:
        """Radial and tangential terms, omitting the intrinsics themselves."""
        return {
            key: value
            for key, value in self.params.items()
            if key not in {"f", "fx", "fy", "cx", "cy"}
        }


@dataclass
class Image:
    """One registered view: its pose, its camera, and its file name.

    ``qvec`` and ``tvec`` are COLMAP's world-to-camera rotation quaternion
    (w, x, y, z) and translation.
    """

    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str

    def rotation_matrix(self) -> np.ndarray:
        return quaternion_to_rotation(self.qvec)

    def world_to_camera(self) -> np.ndarray:
        """The 4x4 extrinsic matrix as COLMAP stores it."""
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation_matrix()
        matrix[:3, 3] = self.tvec
        return matrix

    def camera_to_world(self) -> np.ndarray:
        """Inverse of :meth:`world_to_camera`, still in COLMAP's axis convention.

        Computed by transposing the rotation rather than inverting the whole
        matrix: the rotation block is orthonormal, so the transpose is both
        exact and cheaper than a general inverse.
        """
        rotation = self.rotation_matrix()
        matrix = np.eye(4)
        matrix[:3, :3] = rotation.T
        matrix[:3, 3] = -rotation.T @ self.tvec
        return matrix


@dataclass
class SparseModel:
    cameras: dict[int, Camera]
    images: dict[int, Image]
    points: int = 0

    def sorted_images(self) -> list[Image]:
        """Images ordered by file name, so output is deterministic."""
        return sorted(self.images.values(), key=lambda image: image.name)


def quaternion_to_rotation(qvec: np.ndarray) -> np.ndarray:
    """Convert COLMAP's (w, x, y, z) quaternion to a rotation matrix."""
    w, x, y, z = (float(v) for v in qvec)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        raise CaptureFormatError("Encountered a zero-length rotation quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def read_points(model_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the sparse point cloud's positions and colours.

    Separate from :func:`read_model`, which only counts points: pose export
    never needs the coordinates, and parsing a million tracks to answer "how
    many" would be waste. Visualisation does need them.

    Returns ``(positions, colors)`` as ``(N, 3)`` float32 and uint8 arrays.
    """
    model_dir = Path(model_dir)
    binary = model_dir / "points3D.bin"
    text = model_dir / "points3D.txt"
    if binary.exists():
        return _read_points_binary(binary)
    if text.exists():
        return _read_points_text(text)
    return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)


def _read_points_binary(path: Path) -> tuple[np.ndarray, np.ndarray]:
    positions: list[tuple[float, float, float]] = []
    colors: list[tuple[int, int, int]] = []
    with path.open("rb") as handle:
        (count,) = _read(handle, "<Q")
        for _ in range(count):
            _id, x, y, z, r, g, b, _error = _read(handle, "<QdddBBBd")
            (track_length,) = _read(handle, "<Q")
            # Each track entry is (image_id, point2D_idx); we need none of them.
            handle.seek(track_length * struct.calcsize("<ii"), 1)
            positions.append((x, y, z))
            colors.append((r, g, b))
    return (
        np.asarray(positions, dtype=np.float32).reshape(-1, 3),
        np.asarray(colors, dtype=np.uint8).reshape(-1, 3),
    )


def _read_points_text(path: Path) -> tuple[np.ndarray, np.ndarray]:
    positions: list[tuple[float, float, float]] = []
    colors: list[tuple[int, int, int]] = []
    for line in _iter_data_lines(path):
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            positions.append((float(parts[1]), float(parts[2]), float(parts[3])))
            colors.append((int(parts[4]), int(parts[5]), int(parts[6])))
        except ValueError:
            continue
    return (
        np.asarray(positions, dtype=np.float32).reshape(-1, 3),
        np.asarray(colors, dtype=np.uint8).reshape(-1, 3),
    )


def read_model(model_dir: Path) -> SparseModel:
    """Read a sparse model, preferring binary and falling back to text."""
    model_dir = Path(model_dir)
    if (model_dir / "cameras.bin").exists() and (model_dir / "images.bin").exists():
        cameras = _read_cameras_binary(model_dir / "cameras.bin")
        images = _read_images_binary(model_dir / "images.bin")
        points = _count_points_binary(model_dir / "points3D.bin")
    elif (model_dir / "cameras.txt").exists() and (model_dir / "images.txt").exists():
        cameras = _read_cameras_text(model_dir / "cameras.txt")
        images = _read_images_text(model_dir / "images.txt")
        points = _count_points_text(model_dir / "points3D.txt")
    else:
        raise CaptureFormatError(
            f"No COLMAP model found in {model_dir}: expected cameras/images in .bin or .txt"
        )
    if not images:
        raise CaptureFormatError(f"COLMAP model at {model_dir} contains no registered images")
    return SparseModel(cameras=cameras, images=images, points=points)


# --------------------------------------------------------------------------
# text dialect
# --------------------------------------------------------------------------


def _iter_data_lines(path: Path):
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            yield stripped


def _read_cameras_text(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    for line in _iter_data_lines(path):
        parts = line.split()
        camera_id = int(parts[0])
        model = parts[1]
        width, height = int(parts[2]), int(parts[3])
        values = [float(v) for v in parts[4:]]
        cameras[camera_id] = Camera(
            id=camera_id,
            model=model,
            width=width,
            height=height,
            params=_name_params(model, values),
        )
    return cameras


def _read_images_text(path: Path) -> dict[int, Image]:
    images: dict[int, Image] = {}
    lines = list(_iter_data_lines(path))
    # Each image occupies two entries: a pose line and a 2D-point line. The
    # point line can be empty, in which case the blank was already dropped, so
    # entries are identified by shape rather than by position.
    for line in lines:
        parts = line.split()
        if len(parts) < 10:
            continue
        try:
            image_id = int(parts[0])
            qvec = np.array([float(v) for v in parts[1:5]])
            tvec = np.array([float(v) for v in parts[5:8]])
            camera_id = int(parts[8])
        except ValueError:
            continue  # A points2D line; skip it.
        images[image_id] = Image(
            id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=" ".join(parts[9:]),
        )
    return images


def _count_points_text(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in _iter_data_lines(path))


# --------------------------------------------------------------------------
# binary dialect
# --------------------------------------------------------------------------


def _read(handle, fmt: str):
    size = struct.calcsize(fmt)
    data = handle.read(size)
    if len(data) != size:
        raise CaptureFormatError("COLMAP binary model ended unexpectedly")
    return struct.unpack(fmt, data)


def _read_cameras_binary(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    with path.open("rb") as handle:
        (count,) = _read(handle, "<Q")
        for _ in range(count):
            camera_id, model_id, width, height = _read(handle, "<iiQQ")
            if model_id not in CAMERA_MODELS:
                raise CaptureFormatError(f"Unknown COLMAP camera model id {model_id}")
            model, names = CAMERA_MODELS[model_id]
            values = _read(handle, f"<{len(names)}d")
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model,
                width=width,
                height=height,
                params=dict(zip(names, values, strict=True)),
            )
    return cameras


def _read_images_binary(path: Path) -> dict[int, Image]:
    images: dict[int, Image] = {}
    with path.open("rb") as handle:
        (count,) = _read(handle, "<Q")
        for _ in range(count):
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = _read(handle, "<idddddddi")
            name_bytes = bytearray()
            while True:
                char = handle.read(1)
                if not char or char == b"\x00":
                    break
                name_bytes += char
            (num_points2d,) = _read(handle, "<Q")
            # The 2D observations are not needed for pose export; skip them in
            # one seek rather than parsing millions of triples.
            handle.seek(num_points2d * struct.calcsize("<ddq"), 1)
            images[image_id] = Image(
                id=image_id,
                qvec=np.array([qw, qx, qy, qz]),
                tvec=np.array([tx, ty, tz]),
                camera_id=camera_id,
                name=name_bytes.decode("utf-8", errors="replace"),
            )
    return images


def _count_points_binary(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        try:
            (count,) = _read(handle, "<Q")
        except CaptureFormatError:
            return 0
    return int(count)


def _name_params(model: str, values: list[float]) -> dict[str, float]:
    names = CAMERA_MODELS.get(_MODEL_IDS.get(model, -1), (model, ()))[1]
    if not names:
        # Unknown model: keep the values positionally so nothing is silently lost.
        return {f"p{i}": v for i, v in enumerate(values)}
    return dict(zip(names, values, strict=False))
