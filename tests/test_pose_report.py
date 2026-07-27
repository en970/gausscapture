"""Tests for the pose report against both COLMAP model dialects.

Regression coverage for a bug the first real harness run exposed: the mapper
writes binary models by default, and a text-only count reported a *successful*
reconstruction as zero registered images -- the pipeline called a good capture
"bad" while the model sat on disk. The counting now goes through the shared
reader, and these tests pin it against both dialects.
"""

from __future__ import annotations

import pytest

from gausscapture.pose.colmap import _build_report
from tests.test_colmap_model import (
    IDENTITY_POSE,
    SHIFTED_POSE,
    write_binary_model,
    write_text_model,
)


@pytest.fixture()
def images_dir(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for name in ("frame_000001.jpg", "frame_000002.jpg"):
        (images / name).write_bytes(b"\xff\xd8\xff\xd9")  # smallest possible JPEG-ish stub
    return images


class TestBuildReport:
    @pytest.mark.parametrize("writer", [write_binary_model, write_text_model])
    def test_counts_both_dialects(self, tmp_path, images_dir, writer):
        """The binary case is the regression: it used to count zero."""
        sparse = tmp_path / "sparse"
        writer(sparse / "0", [IDENTITY_POSE, SHIFTED_POSE])

        report = _build_report(sparse, images_dir, "sequential")
        assert report.images_registered == 2
        assert report.registered_ratio == 1.0
        assert report.sparse_points > 0
        assert report.status == "good"

    def test_no_model_directory_is_bad(self, tmp_path, images_dir):
        sparse = tmp_path / "sparse"
        sparse.mkdir()
        report = _build_report(sparse, images_dir, "sequential")
        assert report.status == "bad"
        assert report.images_registered == 0

    def test_unreadable_model_is_bad_not_a_crash(self, tmp_path, images_dir):
        sparse = tmp_path / "sparse"
        (sparse / "0").mkdir(parents=True)
        (sparse / "0" / "project.ini").write_text("", encoding="utf-8")
        report = _build_report(sparse, images_dir, "sequential")
        assert report.status == "bad"

    def test_partial_registration_warns(self, tmp_path, images_dir):
        """One of two images registered crosses the warning threshold."""
        sparse = tmp_path / "sparse"
        write_binary_model(sparse / "0", [IDENTITY_POSE])
        report = _build_report(sparse, images_dir, "sequential")
        assert report.images_registered == 1
        assert report.registered_ratio == 0.5
        assert report.status == "warning"
        assert report.warnings

    def test_disconnected_models_warn(self, tmp_path, images_dir):
        sparse = tmp_path / "sparse"
        write_binary_model(sparse / "0", [IDENTITY_POSE, SHIFTED_POSE])
        write_binary_model(sparse / "1", [IDENTITY_POSE])
        report = _build_report(sparse, images_dir, "sequential")
        assert any("disconnected" in warning for warning in report.warnings)
