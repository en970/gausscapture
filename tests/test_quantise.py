"""Tests for the fixed-point encodings the ``.g4d`` writer and viewer share.

Every one of these checks a number rather than a code path, because the failure
mode here is not an exception -- it is geometry that is quietly the wrong shape
in a browser, on a machine that is not this one.
"""

from __future__ import annotations

import numpy as np
import pytest

from gausscapture.errors import CaptureFormatError
from gausscapture.export.quantise import (
    INV_SQRT2,
    SMALLEST_THREE_HALF,
    angular_error_degrees,
    dequantise_grid,
    dequantise_weights,
    grid_step,
    pack_quaternion_smallest_three,
    pack_quaternions_snorm16,
    quantise_grid,
    quantise_weights,
    unpack_quaternion_smallest_three,
    unpack_quaternions_snorm16,
)


def random_quaternions(count: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(count, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


#: These tests all measure sub-degree rotation errors, so they must use the
#: ``atan2`` form. The obvious ``2 * arccos(|dot|)`` reported 0.0335 degrees
#: for the 16-bit encoding whose true error is 0.0032, purely because
#: ``arccos`` has an infinite derivative at 1 -- see the note in
#: :mod:`gausscapture.export.quantise`. Importing the production helper rather
#: than restating it here means a test can never disagree with the encoder
#: about what an angle is.
angular_error = angular_error_degrees


class TestSmallestThree:
    def test_round_trip_over_ten_thousand_quaternions(self):
        quats = random_quaternions(10_000)
        back = unpack_quaternion_smallest_three(pack_quaternion_smallest_three(quats))
        error = angular_error(quats, back)

        # 30 bits of mantissa across three components cannot do better than
        # this: the derived largest component absorbs all three errors. The
        # architecture note's 0.1 degree figure is unreachable at 32 bits, and
        # the measured bound is recorded here rather than in a comment so that
        # a change to the encoding has to move a number a test can see.
        # Measured: 0.209 max, 0.0843 mean here; 0.240 max over 200,000.
        assert error.max() < 0.30
        assert error.mean() < 0.10

    def test_still_beats_the_eight_bit_encoding_the_static_format_uses(self):
        """The 32-byte ``.splat`` stores each component in 8 bits."""
        quats = random_quaternions(5_000, seed=11)
        coarse = np.clip(np.rint(quats * 128 + 128), 0, 255)
        coarse = (coarse - 128) / 128
        coarse /= np.linalg.norm(coarse, axis=1, keepdims=True)

        fine = unpack_quaternion_smallest_three(pack_quaternion_smallest_three(quats))
        assert angular_error(quats, fine).mean() < angular_error(quats, coarse).mean() / 4

    def test_identity_survives_exactly_enough_to_be_identity(self):
        identity = np.array([[1.0, 0.0, 0.0, 0.0]])
        back = unpack_quaternion_smallest_three(pack_quaternion_smallest_three(identity))
        assert angular_error(identity, back)[0] < 1e-3

    def test_zero_is_a_code_rather_than_a_point_between_two(self):
        """The reason the identity survives, stated as the property itself.

        An encoding that spreads 1024 codes evenly across the closed interval
        puts zero at code 511.5. Nothing at rest is then at rest: the identity
        comes back 0.137 degrees out, and a rest scaffold jitters.
        """
        packed = pack_quaternion_smallest_three(np.array([[1.0, 0.0, 0.0, 0.0]]))
        stored = [(int(packed[0]) >> shift) & 1023 for shift in (20, 10, 0)]
        assert stored == [SMALLEST_THREE_HALF] * 3

    def test_the_axis_aligned_rotations_are_exact(self):
        """Every component is 0, +-1 or +-1/sqrt(2), and each lands on a code."""
        exact = np.array([
            [1.0, 0.0, 0.0, 0.0],            # identity
            [0.0, 1.0, 0.0, 0.0],            # 180 degrees about x
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [INV_SQRT2, INV_SQRT2, 0.0, 0.0],  # 90 degrees about x
            [INV_SQRT2, 0.0, -INV_SQRT2, 0.0],
            [0.0, 0.0, INV_SQRT2, -INV_SQRT2],
        ])
        back = unpack_quaternion_smallest_three(pack_quaternion_smallest_three(exact))
        assert angular_error(exact, back).max() < 1e-3

    def test_the_largest_component_index_survives_the_round_trip(self):
        """The two index bits and the ascending order of the other three.

        If either were dropped or permuted the error would be structural
        rather than a fraction of a step, so this is checked directly instead
        of being inferred from an aggregate.
        """
        quats = random_quaternions(2_000, seed=41)
        rows = np.arange(len(quats))
        largest = np.argmax(np.abs(quats), axis=1)
        quats = quats * np.where(quats[rows, largest] < 0, -1.0, 1.0)[:, None]

        back = unpack_quaternion_smallest_three(pack_quaternion_smallest_three(quats))

        # One code step; a component nearer zero than half of one legitimately
        # decodes to zero, and a near-tie for largest may legitimately swap.
        step = INV_SQRT2 / SMALLEST_THREE_HALF
        magnitudes = np.sort(np.abs(quats), axis=1)
        clear = (magnitudes[:, 3] - magnitudes[:, 2]) > step
        assert clear.sum() > 1_900
        assert np.array_equal(
            np.argmax(np.abs(back[clear]), axis=1), largest[clear]
        )

        resolved = np.abs(quats) > step
        assert np.all(np.sign(back[resolved]) == np.sign(quats[resolved]))

    def test_sign_is_irrelevant_because_q_and_minus_q_are_one_rotation(self):
        quats = random_quaternions(500, seed=3)
        a = pack_quaternion_smallest_three(quats)
        b = pack_quaternion_smallest_three(-quats)
        assert np.array_equal(a, b)

    def test_rejects_a_non_quaternion(self):
        with pytest.raises(CaptureFormatError):
            pack_quaternion_smallest_three(np.zeros((4, 3)))


class TestSnorm16:
    def test_trajectory_quaternions_round_trip_far_below_a_hundredth_degree(self):
        """16 bits per component genuinely reaches the budget; here is the sum.

        A component is stored to within half a step, 0.5 / 32767 = 1.53e-5.
        The four-component error vector is therefore at most 2 * 1.53e-5 =
        3.05e-5 long, and a quaternion perturbation of that size is a rotation
        of at most twice it, 6.1e-5 rad = 0.0035 degrees. Measured worst case
        over this sample is 0.0032. No widening of the field is warranted.
        """
        quats = random_quaternions(5_000, seed=19)
        back = unpack_quaternions_snorm16(pack_quaternions_snorm16(quats))
        assert angular_error(quats, back).max() < 0.01

    def test_the_budget_is_bounded_by_the_step_and_not_by_the_measurement(self):
        """The component error really is under half a step.

        This is the check that would have caught the earlier 0.0335-degree
        reading for what it was. Component error is measured by subtraction,
        which is exact here, so it cannot be inflated by an ill-conditioned
        inverse trigonometric function the way the angle was.
        """
        quats = random_quaternions(5_000, seed=19)
        codes = pack_quaternions_snorm16(quats)
        back = np.asarray(codes, dtype=np.float64) / 32767.0
        half_step = 0.5 / 32767.0
        assert np.abs(back - quats).max() <= half_step * 1.0001

    def test_the_arccos_metric_would_have_lied_about_this_encoding(self):
        """Pin the reason the tests use ``atan2``, so nobody reverts it.

        ``2 * arccos(|dot|)`` on the same data reports an error ten times the
        truth, and reports exactly zero for a large share of the sample
        because the dot product rounds to 1. It is the measurement that was
        broken, not the encoding, and the tolerance was right all along.
        """
        quats = random_quaternions(5_000, seed=19)
        back = unpack_quaternions_snorm16(pack_quaternions_snorm16(quats))

        dot = np.abs(np.sum(quats * np.asarray(back, dtype=np.float64), axis=1))
        naive = np.degrees(2 * np.arccos(np.clip(dot, -1.0, 1.0)))

        assert naive.max() > 5 * angular_error(quats, back).max()
        assert (naive == 0.0).sum() > len(quats) // 4
        assert not np.any(angular_error(quats, back) == 0.0)

    def test_sign_is_preserved_because_trajectories_depend_on_it(self):
        """Unlike smallest-three, this encoding must not canonicalise the sign.

        Trajectories are stored hemisphere-continuous so that blending between
        neighbouring frames takes the short arc; canonicalising every frame
        would throw that away and reintroduce the long-way-round flip.
        """
        quats = np.array([[-0.9, 0.1, 0.2, 0.3]])
        quats = quats / np.linalg.norm(quats)
        back = unpack_quaternions_snorm16(pack_quaternions_snorm16(quats))
        assert back[0, 0] < 0


class TestGrid:
    def test_round_trip_within_one_step(self):
        rng = np.random.default_rng(5)
        points = rng.uniform(-3, 7, size=(2_000, 3))
        codes, origin, span = quantise_grid(points)
        back = dequantise_grid(codes, origin, span)
        assert np.abs(back - points).max() <= grid_step(span).max()

    def test_endpoints_land_on_the_grid_ends(self):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        codes, origin, span = quantise_grid(points)
        assert codes.min() == 0
        assert codes.max() == 65535

    def test_a_planar_axis_does_not_divide_by_zero(self):
        points = np.column_stack([
            np.linspace(0, 1, 50), np.linspace(0, 1, 50), np.zeros(50),
        ])
        codes, origin, span = quantise_grid(points)
        back = dequantise_grid(codes, origin, span)
        assert np.isfinite(back).all()
        assert np.abs(back[:, 2]).max() < 1e-6


class TestWeights:
    def test_every_row_sums_to_exactly_two_hundred_and_fifty_five(self):
        rng = np.random.default_rng(23)
        weights = rng.random((5_000, 4))
        weights /= weights.sum(axis=1, keepdims=True)
        codes = quantise_weights(weights)
        assert codes.dtype == np.uint8
        assert np.all(codes.astype(np.int64).sum(axis=1) == 255)

    def test_the_hard_case_of_four_equal_weights(self):
        """255 does not divide by four, so naive rounding gives 256."""
        codes = quantise_weights(np.full((1, 4), 0.25))
        assert codes.sum() == 255
        assert sorted(codes[0].tolist()) == [63, 64, 64, 64]

    def test_leftover_units_go_to_the_largest_fractions(self):
        codes = quantise_weights(np.array([[0.5, 0.3, 0.1, 0.1]]))
        assert codes.sum() == 255
        # 0.5*255 = 127.5, 0.3*255 = 76.5, 0.1*255 = 25.5 twice: four halves,
        # three units to give away, and the last column must lose out.
        assert codes[0, 3] == 25

    def test_unnormalised_rows_are_normalised_rather_than_refused(self):
        codes = quantise_weights(np.array([[2.0, 2.0]]))
        assert codes.sum() == 255

    def test_a_row_bound_to_nothing_falls_back_to_its_first_influence(self):
        codes = quantise_weights(np.zeros((1, 4)))
        assert codes[0, 0] == 255
        assert codes.sum() == 255

    def test_round_trip_is_within_a_single_unit(self):
        rng = np.random.default_rng(29)
        weights = rng.random((1_000, 4))
        weights /= weights.sum(axis=1, keepdims=True)
        back = dequantise_weights(quantise_weights(weights))
        assert np.abs(back - weights).max() < 1.5 / 255
        assert np.abs(back.sum(axis=1) - 1.0).max() < 1e-6

    def test_rejects_a_shape_that_is_not_per_gaussian_influences(self):
        with pytest.raises(CaptureFormatError):
            quantise_weights(np.zeros(4))
        with pytest.raises(CaptureFormatError):
            quantise_weights(np.zeros((4, 0)))
