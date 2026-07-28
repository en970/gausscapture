"""Tests for COLMAP model parsing and transforms.json conversion.

The axis-convention conversion gets its own careful coverage because it fails
silently: a wrong flip still trains to a plausible loss and only reveals itself
as a reconstruction rendered upside down and inside out.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from gausscapture.errors import CaptureFormatError
from gausscapture.pack.transforms import build_transforms, colmap_to_opengl, validate_transforms
from gausscapture.pose.model import quaternion_to_rotation, read_model

# --------------------------------------------------------------------------
# fixtures: minimal COLMAP models in both dialects
# --------------------------------------------------------------------------


def write_text_model(model_dir: Path, images: list[tuple[int, list[float], list[float], str]]) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "cameras.txt").write_text(
        "# Camera list\n1 OPENCV 1920 1080 1400.0 1401.0 960.0 540.0 0.01 -0.002 0.0001 0.0002\n",
        encoding="utf-8",
    )
    lines = ["# Image list"]
    for image_id, qvec, tvec, name in images:
        lines.append(
            f"{image_id} {' '.join(str(v) for v in qvec)} {' '.join(str(v) for v in tvec)} 1 {name}"
        )
        lines.append("")  # the points2D line
    (model_dir / "images.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (model_dir / "points3D.txt").write_text(
        "# 3D points\n1 0 0 0 255 255 255 0.5\n2 1 1 1 128 128 128 0.4\n", encoding="utf-8"
    )
    return model_dir


def write_binary_model(
    model_dir: Path, images: list[tuple[int, list[float], list[float], str]]
) -> Path:
    """Write a model in COLMAP's binary dialect, which the mapper emits by default."""
    model_dir.mkdir(parents=True, exist_ok=True)

    with (model_dir / "cameras.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        # camera_id, model_id (4 = OPENCV), width, height, then 8 doubles
        handle.write(struct.pack("<iiQQ", 1, 4, 1920, 1080))
        handle.write(struct.pack("<8d", 1400.0, 1401.0, 960.0, 540.0, 0.01, -0.002, 0.0001, 0.0002))

    with (model_dir / "images.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", len(images)))
        for image_id, qvec, tvec, name in images:
            handle.write(struct.pack("<idddddddi", image_id, *qvec, *tvec, 1))
            handle.write(name.encode("utf-8") + b"\x00")
            # Two 2D observations, which the reader must skip rather than parse.
            handle.write(struct.pack("<Q", 2))
            handle.write(struct.pack("<ddq", 1.0, 2.0, 1))
            handle.write(struct.pack("<ddq", 3.0, 4.0, -1))

    with (model_dir / "points3D.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 7))

    return model_dir


IDENTITY_POSE = (1, [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], "frame_000001.jpg")
SHIFTED_POSE = (2, [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -5.0], "frame_000002.jpg")


# --------------------------------------------------------------------------


class TestQuaternion:
    def test_identity(self):
        assert np.allclose(quaternion_to_rotation(np.array([1.0, 0, 0, 0])), np.eye(3))

    def test_is_orthonormal(self):
        rotation = quaternion_to_rotation(np.array([0.5, 0.5, 0.5, 0.5]))
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(rotation), 1.0)

    def test_unnormalised_input_is_normalised(self):
        scaled = quaternion_to_rotation(np.array([2.0, 0, 0, 0]))
        assert np.allclose(scaled, np.eye(3))

    def test_zero_quaternion_raises(self):
        with pytest.raises(CaptureFormatError):
            quaternion_to_rotation(np.array([0.0, 0, 0, 0]))


class TestModelReading:
    @pytest.mark.parametrize("writer", [write_text_model, write_binary_model])
    def test_reads_both_dialects_identically(self, tmp_path, writer):
        model = read_model(writer(tmp_path / "m", [IDENTITY_POSE, SHIFTED_POSE]))
        assert len(model.images) == 2
        camera = model.cameras[1]
        assert camera.model == "OPENCV"
        assert camera.width == 1920 and camera.height == 1080
        assert camera.fx == 1400.0 and camera.fy == 1401.0
        assert camera.distortion()["k1"] == pytest.approx(0.01)

    def test_binary_is_preferred_over_text(self, tmp_path):
        """Real COLMAP output contains .bin; a stale .txt must not win."""
        model_dir = tmp_path / "m"
        write_text_model(model_dir, [IDENTITY_POSE])
        write_binary_model(model_dir, [IDENTITY_POSE, SHIFTED_POSE])
        assert len(read_model(model_dir).images) == 2

    def test_images_sort_by_name(self, tmp_path):
        model = read_model(
            write_text_model(tmp_path / "m", [SHIFTED_POSE, IDENTITY_POSE])
        )
        assert [image.name for image in model.sorted_images()] == [
            "frame_000001.jpg",
            "frame_000002.jpg",
        ]

    def test_missing_model_raises(self, tmp_path):
        with pytest.raises(CaptureFormatError, match="No COLMAP model"):
            read_model(tmp_path)

    def test_camera_centre_is_recovered(self, tmp_path):
        """COLMAP stores world-to-camera; the camera position is -R^T t."""
        model = read_model(write_text_model(tmp_path / "m", [SHIFTED_POSE]))
        image = next(iter(model.images.values()))
        assert np.allclose(image.camera_to_world()[:3, 3], [0.0, 0.0, 5.0])


class TestAxisConversion:
    def test_translation_is_preserved(self):
        """The flip changes orientation only; the camera does not move."""
        c2w = np.eye(4)
        c2w[:3, 3] = [1.0, 2.0, 3.0]
        assert np.allclose(colmap_to_opengl(c2w)[:3, 3], [1.0, 2.0, 3.0])

    def test_y_and_z_axes_are_negated(self):
        converted = colmap_to_opengl(np.eye(4))
        assert np.allclose(converted[:3, 0], [1, 0, 0])
        assert np.allclose(converted[:3, 1], [0, -1, 0])
        assert np.allclose(converted[:3, 2], [0, 0, -1])

    def test_result_stays_a_proper_rotation(self):
        """Negating two columns preserves the determinant, so no mirroring."""
        rotation = quaternion_to_rotation(np.array([0.5, 0.5, 0.5, 0.5]))
        c2w = np.eye(4)
        c2w[:3, :3] = rotation
        block = colmap_to_opengl(c2w)[:3, :3]
        assert np.isclose(np.linalg.det(block), 1.0)
        assert np.allclose(block @ block.T, np.eye(3), atol=1e-9)

    def test_applying_it_twice_is_the_identity(self):
        c2w = np.eye(4)
        c2w[:3, :3] = quaternion_to_rotation(np.array([0.2, 0.4, 0.5, 0.7]))
        c2w[:3, 3] = [4.0, -1.0, 2.0]
        assert np.allclose(colmap_to_opengl(colmap_to_opengl(c2w)), c2w)


class TestTransformsDocument:
    def test_shared_intrinsics_are_hoisted(self, tmp_path):
        """One camera means top-level intrinsics, which is the phone case."""
        model = read_model(write_binary_model(tmp_path / "m", [IDENTITY_POSE, SHIFTED_POSE]))
        document = build_transforms(model)
        assert document["camera_model"] == "OPENCV"
        assert document["fl_x"] == 1400.0
        assert document["w"] == 1920
        assert document["k1"] == pytest.approx(0.01)
        assert len(document["frames"]) == 2
        assert "fl_x" not in document["frames"][0]

    def test_frame_paths_are_relative_to_the_image_dir(self, tmp_path):
        model = read_model(write_text_model(tmp_path / "m", [IDENTITY_POSE]))
        document = build_transforms(model)
        assert document["frames"][0]["file_path"] == "images/frame_000001.jpg"

    def test_matrices_are_four_by_four_with_a_standard_bottom_row(self, tmp_path):
        model = read_model(write_text_model(tmp_path / "m", [IDENTITY_POSE, SHIFTED_POSE]))
        for frame in build_transforms(model)["frames"]:
            matrix = frame["transform_matrix"]
            assert len(matrix) == 4 and all(len(row) == 4 for row in matrix)
            assert matrix[3] == [0.0, 0.0, 0.0, 1.0]

    def test_filtering_drops_unlisted_frames(self, tmp_path):
        """Frames removed by blur filtering must not remain in transforms.json.

        A pose referencing an image that is not on disk makes most loaders
        raise, so the two have to stay consistent.
        """
        model = read_model(write_text_model(tmp_path / "m", [IDENTITY_POSE, SHIFTED_POSE]))
        document = build_transforms(model, applies_to={"frame_000002.jpg"})
        assert [f["file_path"] for f in document["frames"]] == ["images/frame_000002.jpg"]

    def test_filtering_everything_out_raises(self, tmp_path):
        from gausscapture.errors import PipelineStateError

        model = read_model(write_text_model(tmp_path / "m", [IDENTITY_POSE]))
        with pytest.raises(PipelineStateError):
            build_transforms(model, applies_to={"nothing.jpg"})


class TestTransformsValidation:
    def test_accepts_a_well_formed_document(self, tmp_path):
        from gausscapture.pack.transforms import write_transforms

        model_dir = write_text_model(tmp_path / "m", [IDENTITY_POSE, SHIFTED_POSE])
        out = write_transforms(model_dir, tmp_path / "transforms.json")
        result = validate_transforms(out)
        assert result["valid"]
        assert result["frames"] == 2

    def test_reports_missing_file(self, tmp_path):
        assert not validate_transforms(tmp_path / "absent.json")["valid"]

    def test_reports_malformed_json(self, tmp_path):
        path = tmp_path / "transforms.json"
        path.write_text("{oh no", encoding="utf-8")
        assert not validate_transforms(path)["valid"]

    def test_reports_missing_images(self, tmp_path):
        from gausscapture.pack.transforms import write_transforms

        model_dir = write_text_model(tmp_path / "m", [IDENTITY_POSE])
        out = write_transforms(model_dir, tmp_path / "transforms.json")
        result = validate_transforms(out, image_root=tmp_path)
        assert not result["valid"]
        assert "missing" in result["errors"][0]
