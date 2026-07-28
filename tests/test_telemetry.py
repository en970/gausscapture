"""Tests for the capture-telemetry signals.

These matter more than the plumbing tests: the project's research claim rests
on these numbers meaning what they say they mean. In particular, the blur
measure must be comparable across scenes, which the previous absolute threshold
was not.
"""

from __future__ import annotations

import numpy as np
import pytest

from gausscapture.telemetry.signals import (
    RollingBlurNormaliser,
    exposure_ratios,
    percentile,
    variance_of_laplacian,
)


def _textured(size=(120, 160), seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=size, dtype=np.uint8)


def _flat(value: int, size=(120, 160)) -> np.ndarray:
    return np.full(size, value, dtype=np.uint8)


class TestVarianceOfLaplacian:
    def test_texture_scores_higher_than_flat(self):
        assert variance_of_laplacian(_textured()) > variance_of_laplacian(_flat(128))

    def test_blurring_reduces_the_score(self):
        import cv2

        sharp = _textured()
        blurred = cv2.GaussianBlur(sharp, (9, 9), 6)
        assert variance_of_laplacian(blurred) < variance_of_laplacian(sharp)

    def test_is_content_dependent(self):
        """The reason an absolute threshold cannot work.

        A sharp image of a low-texture scene scores lower than a sharp image of
        a high-texture scene, so any fixed cutoff encodes an assumption about
        what the user is filming.
        """
        import cv2

        busy = _textured()
        calm = cv2.GaussianBlur(_textured(seed=3), (3, 3), 0.6)
        assert variance_of_laplacian(calm) < variance_of_laplacian(busy)


class TestRollingBlurNormaliser:
    def test_returns_one_during_warmup(self):
        normaliser = RollingBlurNormaliser()
        assert normaliser.push(500.0) == 1.0
        assert not normaliser.ready

    def test_typical_frame_scores_near_one(self):
        normaliser = RollingBlurNormaliser()
        for _ in range(20):
            normaliser.push(100.0)
        assert normaliser.push(100.0) == pytest.approx(1.0, abs=0.01)

    def test_soft_frame_scores_below_one(self):
        normaliser = RollingBlurNormaliser()
        for _ in range(20):
            normaliser.push(100.0)
        assert normaliser.push(30.0) == pytest.approx(0.3, abs=0.05)

    def test_scale_invariance(self):
        """A 4K clip and a 720p clip of the same motion must score the same.

        This is the property that makes the signal portable between captures,
        and the one an absolute threshold lacks.
        """
        low = RollingBlurNormaliser()
        high = RollingBlurNormaliser()
        for _ in range(20):
            low.push(50.0)
            high.push(5000.0)
        assert low.push(25.0) == pytest.approx(high.push(2500.0), abs=1e-6)

    def test_window_forgets_old_values(self):
        normaliser = RollingBlurNormaliser(window=10, warmup=3)
        for _ in range(10):
            normaliser.push(1000.0)
        for _ in range(10):
            normaliser.push(10.0)
        # The window now holds only the recent, softer values, so a frame at
        # that level is "normal" rather than an outlier.
        assert normaliser.push(10.0) == pytest.approx(1.0, abs=0.01)


class TestExposure:
    def test_detects_clipped_highlights(self):
        over, under = exposure_ratios(_flat(255))
        assert over == 1.0
        assert under == 0.0

    def test_detects_crushed_shadows(self):
        over, under = exposure_ratios(_flat(0))
        assert under == 1.0
        assert over == 0.0

    def test_midtones_are_clean(self):
        over, under = exposure_ratios(_flat(128))
        assert over == 0.0
        assert under == 0.0


class TestPercentile:
    def test_empty_input_does_not_raise(self):
        assert percentile([], 10) == 0.0

    def test_p10_is_low_end(self):
        assert percentile([float(i) for i in range(101)], 10) == pytest.approx(10.0)
