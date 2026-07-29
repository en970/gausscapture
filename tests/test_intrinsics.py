"""Tests for device-intrinsics conversion.

Every one of the three transforms -- crop, scale, rotate -- produces plausible
numbers when done wrong: right order of magnitude, near the middle of the
frame. So these tests check the arithmetic against values derived by hand from
a real Galaxy S22 calibration, and against invariants (four quarter turns are
the identity, the principal point stays near centre) that a wrong transform
cannot satisfy by accident.
"""

from __future__ import annotations

import pytest

from gausscapture.pose.intrinsics import CameraIntrinsics, from_device, rotate

#: Verbatim from a Galaxy S22 (SM-S901E), main back camera.
S22 = {
    "intrinsic_calibration": {
        "fx": 2760.6787109375,
        "fy": 2758.15234375,
        "cx": 2030.89404296875,
        "cy": 1550.8883056640625,
        "skew": 0,
    },
    "pre_correction_active_array": {"width": 4080, "height": 3060},
    "distortion_kappa": [0.07962691783905029, -0.11437220126390457, 0.04509209096431732, 0, 0],
}


class TestFromDevice:
    def test_landscape_matches_hand_computation(self):
        """4080 -> 1920 is a scale of 0.470588; the 16:9 crop discards 382.5 rows."""
        k = from_device(S22, 1920, 1080, rotation=0)
        assert k is not None
        assert k.fx == pytest.approx(1299.14, abs=0.1)
        assert k.fy == pytest.approx(1297.95, abs=0.1)
        assert k.cx == pytest.approx(955.71, abs=0.1)
        assert k.cy == pytest.approx(549.83, abs=0.1)
        assert (k.width, k.height) == (1920, 1080)

    def test_principal_point_lands_near_the_frame_centre(self):
        """The check that catches a wrong crop or scale.

        Lens centres are within a few percent of the frame centre; a transform
        chain that has gone wrong puts them somewhere else entirely.
        """
        k = from_device(S22, 1920, 1080, rotation=0)
        dx, dy = k.principal_point_offset()
        assert abs(dx) < 0.05 * 1920
        assert abs(dy) < 0.05 * 1080
        assert k.looks_plausible()

    def test_portrait_swaps_axes_and_stays_centred(self):
        """A portrait recording is the same sensor turned a quarter."""
        k = from_device(S22, 1080, 1920, rotation=90)
        assert (k.width, k.height) == (1080, 1920)
        # fx and fy exchange under a quarter turn.
        assert k.fx == pytest.approx(1297.95, abs=0.1)
        assert k.fy == pytest.approx(1299.14, abs=0.1)
        assert k.looks_plausible()

    def test_distortion_survives_crop_and_scale_unchanged(self):
        """Distortion is defined on normalised coordinates."""
        landscape = from_device(S22, 1920, 1080, rotation=0)
        half = from_device(S22, 960, 540, rotation=0)
        assert landscape.k1 == pytest.approx(half.k1)
        assert landscape.k2 == pytest.approx(half.k2)
        assert landscape.k1 == pytest.approx(0.0796, abs=1e-4)

    def test_android_k3_is_dropped_not_mistaken_for_tangential(self):
        """Android orders [k1,k2,k3,p1,p2]; COLMAP's OPENCV model has no k3."""
        k = from_device(S22, 1920, 1080)
        assert k.p1 == 0.0 and k.p2 == 0.0  # positions 3 and 4, not k3 at position 2

    def test_focal_length_scales_with_resolution(self):
        full = from_device(S22, 1920, 1080)
        half = from_device(S22, 960, 540)
        assert half.fx == pytest.approx(full.fx / 2, rel=1e-6)

    def test_returns_none_without_a_calibration(self):
        assert from_device({"pre_correction_active_array": {"width": 1, "height": 1}}, 100, 100) is None
        assert from_device({"intrinsic_calibration": S22["intrinsic_calibration"]}, 100, 100) is None
        assert from_device({}, 100, 100) is None

    def test_colmap_parameter_order(self):
        k = from_device(S22, 1920, 1080)
        params = k.colmap_params()
        assert params[:4] == [k.fx, k.fy, k.cx, k.cy]
        assert params[4:] == [k.k1, k.k2, k.p1, k.p2]
        assert len(k.colmap_camera_params().split(",")) == 8


class TestRotate:
    def _sample(self) -> CameraIntrinsics:
        return CameraIntrinsics(fx=1000, fy=1010, cx=940, cy=520, width=1920, height=1080)

    def test_zero_is_identity(self):
        k = self._sample()
        assert rotate(k, 0) is k

    def test_four_quarter_turns_return_the_original(self):
        """The strongest available check on the arithmetic."""
        k = self._sample()
        turned = k
        for _ in range(4):
            turned = rotate(turned, 90)
        assert turned.fx == pytest.approx(k.fx)
        assert turned.fy == pytest.approx(k.fy)
        assert turned.cx == pytest.approx(k.cx)
        assert turned.cy == pytest.approx(k.cy)
        assert (turned.width, turned.height) == (k.width, k.height)

    def test_ninety_and_two_seventy_are_inverses(self):
        k = self._sample()
        assert rotate(rotate(k, 90), 270).cx == pytest.approx(k.cx)
        assert rotate(rotate(k, 90), 270).cy == pytest.approx(k.cy)

    def test_quarter_turn_swaps_dimensions_and_focals(self):
        k = rotate(self._sample(), 90)
        assert (k.width, k.height) == (1080, 1920)
        assert k.fx == 1010 and k.fy == 1000

    def test_half_turn_keeps_dimensions_and_mirrors_the_centre(self):
        k = rotate(self._sample(), 180)
        assert (k.width, k.height) == (1920, 1080)
        assert k.cx == pytest.approx(1920 - 940)
        assert k.cy == pytest.approx(1080 - 520)

    def test_tangential_terms_follow_the_axes(self):
        k = CameraIntrinsics(fx=1, fy=1, cx=1, cy=1, width=4, height=2, p1=0.3, p2=-0.7)
        turned = rotate(k, 90)
        assert turned.p1 == pytest.approx(-0.7)
        assert turned.p2 == pytest.approx(0.3)

    def test_rejects_a_non_quarter_turn(self):
        with pytest.raises(ValueError, match="multiple of 90"):
            rotate(self._sample(), 45)
