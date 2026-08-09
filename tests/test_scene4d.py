"""Round-trip tests for the ``.g4d`` container.

The failure this file exists to catch is a chunk that is written correctly and
read back as something else -- a stride off by a byte, a quantisation grid the
reader reconstructs from the wrong header field, a skinning row that sums to
254. None of those raise. They produce a file that opens, draws, and is subtly
the wrong scene, on a machine that is not this one.

So every test here writes a real file to a real path and reads it back through
the public reader, and asserts on the numbers that came out rather than on the
bytes that went in.
"""

from __future__ import annotations

import numpy as np
import pytest

from gausscapture.errors import CaptureFormatError
from gausscapture.export.quantise import angular_error_degrees, grid_step
from gausscapture.export.scene4d import (
    ALIGNMENT,
    COLR_STRIDE,
    DIRECTORY_ENTRY_SIZE,
    GEOM_STRIDE,
    HEADER_SIZE,
    MAGIC,
    TRAJ_STRIDE,
    read_scene4d,
    read_scene4d_header,
    write_scene4d,
)
from gausscapture.export.synthetic4d import synthetic_scene

#: Chunks a drawable scene always carries, in the order the writer emits them.
CHUNK_TAGS = ("GEOM", "COLR", "NODE", "TRAJ", "SKIN", "META")


def small_scene(**overrides):
    """A scene big enough to exercise every chunk and small enough to be fast."""
    settings = dict(gaussians=3_000, nodes=12, frames=16, dynamic_fraction=0.4)
    settings.update(overrides)
    scene = synthetic_scene(**settings)
    scene.fixed_pose_verified = True
    scene.cone_azimuth_deg = 21.0
    scene.cone_elevation_deg = 8.0
    scene.meta = {"capture": "test", "note": "round trip"}
    return scene


class TestRoundTrip:
    """Write a scene, read it back, and account for every chunk."""

    def test_the_header_survives(self, tmp_path):
        scene = small_scene()
        write_scene4d(scene, tmp_path / "s.g4d", order_by_importance=False)
        header = read_scene4d_header(tmp_path / "s.g4d")

        assert header.gaussian_count == scene.gaussian_count
        assert header.dynamic_count == scene.dynamic_count
        assert header.node_count == scene.node_count
        assert header.frame_count == scene.frame_count
        assert header.skin_k == scene.skin_k
        assert header.fps == pytest.approx(scene.fps)
        assert header.clip_seconds == pytest.approx(scene.clip_seconds)
        assert header.cone_azimuth_deg == pytest.approx(21.0)
        assert header.cone_elevation_deg == pytest.approx(8.0)
        assert header.fixed_pose_verified is True
        assert header.loop == "ping-pong"
        assert set(header.chunks) == set(CHUNK_TAGS)

    def test_every_chunk_is_present_aligned_and_inside_the_file(self, tmp_path):
        """Alignment is load-bearing: the viewer takes typed views, not copies."""
        write_scene4d(small_scene(), tmp_path / "s.g4d")
        data = (tmp_path / "s.g4d").read_bytes()
        header = read_scene4d_header(data)

        assert data[: len(MAGIC)] == MAGIC
        for tag in CHUNK_TAGS:
            offset, length = header.chunks[tag]
            assert offset % ALIGNMENT == 0, f"{tag} is not {ALIGNMENT}-byte aligned"
            assert offset + length <= len(data)

    def test_geometry_survives_within_a_fraction_of_a_gaussian(self, tmp_path):
        """GEOM: positions, scales and orientations, each against a budget of the scene.

        Both bounds come from the *source scene*. The old assertion was against
        the writer's own reported ``position_step``, which is computed from the
        same span and the same level count the writer just encoded with -- so it
        could not disagree with the encoder, and any self-consistent one passed.
        Redirecting the quantiser to 12 bits, a silent four-fold precision loss,
        left it true because the reported step grew with the error.

        ``LEVELS`` is written as a literal here on purpose: it is the number the
        format documents, the number the shader decodes with, and the thing a
        regression would quietly change.
        """
        scene = small_scene()
        report = write_scene4d(scene, tmp_path / "s.g4d", order_by_importance=False)
        back = read_scene4d(tmp_path / "s.g4d")

        assert back.means.shape == scene.means.shape
        error = float(np.abs(back.means - scene.means).max())

        # 16 bits over the scene's own extent.
        levels = 65535
        step = float((np.ptp(np.asarray(scene.means, dtype=np.float64), axis=0) / levels).max())
        assert error <= step, f"{error:.6g} exceeds one 16-bit step ({step:.6g})"

        # And that step is small enough to be invisible: a tenth of a gaussian.
        budget = float(np.median(scene.scales)) / 10.0
        assert error < budget, f"{error:.6g} exceeds a tenth of a gaussian ({budget:.6g})"
        assert error <= report["position_step"]

        # Scales are quantised in log space, so the error is relative.
        assert np.abs(back.scales / scene.scales - 1.0).max() < 1e-3

        # 0.30 degrees is the smallest-three worst case; see test_quantise.
        assert angular_error_degrees(scene.quats, back.quats).max() < 0.30

    def test_colour_survives_exactly_because_it_is_already_bytes(self, tmp_path):
        """COLR: uint8 in, uint8 out, no encoding in between and so no excuse."""
        scene = small_scene()
        write_scene4d(scene, tmp_path / "s.g4d", order_by_importance=False)
        back = read_scene4d(tmp_path / "s.g4d")

        assert back.colors.dtype == np.uint8
        assert np.array_equal(back.colors, scene.colors)

    def test_the_scaffold_rest_pose_survives_exactly(self, tmp_path):
        """NODE: stored as raw float32, so anything but equality is a bug."""
        scene = small_scene()
        write_scene4d(scene, tmp_path / "s.g4d", order_by_importance=False)
        back = read_scene4d(tmp_path / "s.g4d")

        assert back.node_rest.shape == scene.node_rest.shape
        assert np.array_equal(back.node_rest, scene.node_rest.astype(np.float32))

    def test_trajectories_survive_rotation_and_translation_alike(self, tmp_path):
        """TRAJ: 8 bytes of quaternion and 6 of translation, per node per frame."""
        scene = small_scene()
        write_scene4d(scene, tmp_path / "s.g4d", order_by_importance=False)
        back = read_scene4d(tmp_path / "s.g4d")

        assert back.node_quats.shape == scene.node_quats.shape
        assert back.node_trans.shape == scene.node_trans.shape

        error = angular_error_degrees(
            np.asarray(scene.node_quats, dtype=np.float64).reshape(-1, 4),
            np.asarray(back.node_quats, dtype=np.float64).reshape(-1, 4),
        )
        assert error.max() < 0.01

        # Same two bounds as the positions, recomputed from the source scene.
        span = np.ptp(np.asarray(scene.node_trans, dtype=np.float64).reshape(-1, 3), axis=0)
        error = float(np.abs(back.node_trans - scene.node_trans).max())
        assert error <= float((span / 65535).max())
        assert error <= grid_step(span).max()

        # A node translation carries a whole group of gaussians, so an error in
        # it is at least as visible as an error in a position.
        budget = float(np.median(scene.scales)) / 10.0
        assert error < budget, f"{error:.6g} exceeds a tenth of a gaussian ({budget:.6g})"

    def test_skinning_survives_and_every_row_still_sums_to_255(self, tmp_path):
        """SKIN: the sum is the whole point.

        A row that sums to anything but 255 is a gaussian uniformly brighter
        or darker than its neighbours after the blend, which reads as a seam
        along whatever line the scaffold happens to divide the subject on.
        """
        scene = small_scene()
        write_scene4d(scene, tmp_path / "s.g4d", order_by_importance=False)
        back = read_scene4d(tmp_path / "s.g4d")

        assert back.skin_indices.shape == scene.skin_indices.shape
        assert back.skin_weights.shape == scene.skin_weights.shape

        codes = np.rint(back.skin_weights * 255.0).astype(np.int64)
        assert np.all(codes.sum(axis=1) == 255)
        assert np.abs(back.skin_weights.sum(axis=1) - 1.0).max() < 1e-6
        assert back.skin_indices.min() >= 0
        assert back.skin_indices.max() < back.node_count

    def test_skinning_names_the_same_nodes_with_the_same_weights(self, tmp_path):
        """The writer re-sorts each row into descending weight order.

        That is a promise the shader relies on -- influence 0 is the
        hemisphere reference for the rotation blend -- so the row comes back
        permuted on purpose, and it is the *set* of (node, weight) pairs that
        has to match, not the order.
        """
        scene = small_scene()
        write_scene4d(scene, tmp_path / "s.g4d", order_by_importance=False)
        back = read_scene4d(tmp_path / "s.g4d")

        before = np.rint(np.asarray(scene.skin_weights, dtype=np.float64) * 255.0)
        after = np.rint(np.asarray(back.skin_weights, dtype=np.float64) * 255.0)
        for row in range(scene.dynamic_count):
            sent = dict(zip(scene.skin_indices[row].tolist(), before[row], strict=True))
            got = dict(zip(back.skin_indices[row].tolist(), after[row], strict=True))
            assert set(sent) == set(got)
            # Largest-remainder moves a code by at most one unit from the
            # naive rounding this comparison uses.
            for node in sent:
                assert abs(sent[node] - got[node]) <= 1

        descending = np.diff(back.skin_weights, axis=1)
        assert descending.max() <= 1e-6

    def test_metadata_survives_and_the_loop_mode_with_it(self, tmp_path):
        """META: JSON, so that adding a field is never a format change."""
        scene = small_scene()
        write_scene4d(scene, tmp_path / "s.g4d", order_by_importance=False)
        back = read_scene4d(tmp_path / "s.g4d")

        assert back.meta["capture"] == "test"
        assert back.meta["note"] == "round trip"
        assert back.loop == "ping-pong"

    def test_a_scene_with_no_dynamic_gaussians_still_round_trips(self, tmp_path):
        """A capture where the subject never moved is a valid, boring scene."""
        scene = small_scene()
        scene.dynamic_count = 0
        scene.skin_indices = np.zeros((0, 4), dtype=np.int32)
        scene.skin_weights = np.zeros((0, 4), dtype=np.float32)

        write_scene4d(scene, tmp_path / "s.g4d", order_by_importance=False)
        back = read_scene4d(tmp_path / "s.g4d")

        assert back.dynamic_count == 0
        assert back.skin_indices.shape[0] == 0
        assert back.gaussian_count == scene.gaussian_count

    def test_importance_ordering_keeps_the_skinning_attached_to_its_gaussians(
        self, tmp_path
    ):
        """Reordering is the easiest way to silently detach SKIN from GEOM.

        The dynamic group is sorted, and its skinning rows have to be carried
        along by the same permutation. If they are not, every dynamic gaussian
        is driven by some other gaussian's nodes -- which still draws.
        """
        scene = small_scene()
        write_scene4d(scene, tmp_path / "s.g4d", order_by_importance=True)
        back = read_scene4d(tmp_path / "s.g4d")

        ordered = scene.ordered_by_importance()
        assert np.array_equal(back.colors, ordered.colors)
        assert np.array_equal(
            back.skin_indices, np.asarray(ordered.skin_indices, dtype=np.int32)
        )


class TestTheClock:
    """What the page prints beside the scrubber has to be reachable by the scrubber."""

    def test_the_duration_is_the_span_the_file_can_be_sampled_over(self):
        """``frame_count - 1`` intervals, not ``frame_count``.

        The viewer's time cursor is a frame index clamped to
        ``[0, frame_count - 1]``, so the largest instant it can display is
        ``(frame_count - 1) / fps``. Reporting ``frame_count / fps`` put the
        clock one frame period short of the duration printed next to it, for
        every scene, forever -- while the scrubber, which divides by
        ``frame_count - 1``, did reach the end.
        """
        scene = synthetic_scene(gaussians=100, nodes=4, frames=30, fps=30.0)
        assert scene.frame_count == 30
        assert scene.clip_seconds == pytest.approx(29 / 30.0)
        # The last frame the viewer can seek to is exactly the stated duration.
        assert (scene.frame_count - 1) / scene.fps == pytest.approx(scene.clip_seconds)

    def test_a_single_frame_clip_lasts_no_time_rather_than_one_frame(self):
        scene = synthetic_scene(gaussians=50, nodes=4, frames=1, fps=30.0)
        assert scene.clip_seconds == 0.0

    def test_a_scene_with_no_frame_rate_reports_no_duration(self):
        scene = synthetic_scene(gaussians=50, nodes=4, frames=4, fps=30.0)
        scene.fps = 0.0
        assert scene.clip_seconds == 0.0


class TestSize:
    """The format's whole argument is a number, so the number is tested."""

    @staticmethod
    def predicted_bytes(
        gaussians: int, dynamic: int, nodes: int, frames: int, skin_k: int, meta: int
    ) -> int:
        """Rebuild the file size from the strides the format documents."""
        payloads = [
            gaussians * GEOM_STRIDE,
            gaussians * COLR_STRIDE,
            nodes * 12,
            frames * nodes * TRAJ_STRIDE,
            dynamic * 2 * skin_k,
            meta,
        ]
        cursor = HEADER_SIZE + DIRECTORY_ENTRY_SIZE * len(payloads)
        for length in payloads:
            cursor = (cursor + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT + length
        return cursor

    def test_the_size_model_matches_a_real_file_byte_for_byte(self, tmp_path):
        """Validate the arithmetic against a file before trusting it at scale."""
        scene = small_scene()
        report = write_scene4d(scene, tmp_path / "s.g4d", order_by_importance=False)

        predicted = self.predicted_bytes(
            scene.gaussian_count, scene.dynamic_count, scene.node_count,
            scene.frame_count, scene.skin_k, report["chunk_bytes"]["META"],
        )
        assert predicted == report["bytes"] == (tmp_path / "s.g4d").stat().st_size

    def test_the_deliverable_scene_is_the_nine_and_a_half_megabytes_promised(self):
        """400,000 gaussians, 150,000 dynamic, 256 nodes, 90 frames.

        Measured by writing the file: 9,526,368 bytes = 9.526 MB, against the
        ~9.5 MB the module docstring claims. Recomputed here from the strides
        rather than written, because a 9.5 MB file per test run is not worth
        the two seconds. GEOM is 67 % of it and motion is 3.4 %, which is the
        ratio the whole format exists to achieve.
        """
        total = self.predicted_bytes(
            gaussians=400_000, dynamic=150_000, nodes=256,
            frames=90, skin_k=4, meta=352,
        )
        assert total == 9_526_368
        assert 9.4e6 < total < 9.7e6

        motion = 90 * 256 * TRAJ_STRIDE + 256 * 12
        assert motion / total < 0.04


class TestRefusals:
    """A reader that guesses is worse than a reader that stops."""

    def test_a_file_that_is_not_a_g4d_is_named_as_such(self, tmp_path):
        path = tmp_path / "not.g4d"
        path.write_bytes(b"PLYFILE\x00" + bytes(HEADER_SIZE))
        with pytest.raises(CaptureFormatError, match="magic"):
            read_scene4d_header(path)

    def test_a_file_shorter_than_its_header_is_refused(self, tmp_path):
        path = tmp_path / "stub.g4d"
        path.write_bytes(MAGIC + b"\x00" * 16)
        with pytest.raises(CaptureFormatError, match="header"):
            read_scene4d_header(path)

    def test_a_truncated_file_is_refused_rather_than_half_drawn(self, tmp_path):
        write_scene4d(small_scene(), tmp_path / "s.g4d")
        data = bytearray((tmp_path / "s.g4d").read_bytes())
        with pytest.raises(CaptureFormatError, match="truncated"):
            read_scene4d_header(bytes(data[: len(data) // 2]))

    def test_one_outlying_gaussian_that_would_coarsen_the_grid_is_refused(self, tmp_path):
        """The format's only absolute precision guard, and nothing had tested it.

        16 bits over the bounding box is ample for a room and hopeless for a
        room containing one stray gaussian a kilometre away: the box stretches,
        the step grows with it, and every gaussian in the subject lands on the
        same handful of grid points. Nothing else in the pipeline notices --
        the file is well formed, it opens, it draws, and the subject is a
        lattice. So the writer compares the step against the *median* gaussian
        scale, which an outlier cannot move.
        """
        scene = small_scene()
        scene.means = np.array(scene.means, copy=True)
        scene.means[0] = [1_000.0, 0.0, 0.0]

        with pytest.raises(CaptureFormatError, match="quantisation would be visible"):
            write_scene4d(scene, tmp_path / "outlier.g4d")

    def test_the_same_scene_without_the_outlier_writes_normally(self, tmp_path):
        """The guard has to be about the outlier, not about the scene."""
        report = write_scene4d(small_scene(), tmp_path / "fine.g4d")
        median_scale = float(np.median(small_scene().scales))
        assert report["position_step"] < median_scale / 4.0

    def test_an_unknown_breaking_flag_is_refused(self, tmp_path):
        """Low flag bits announce behaviour a reader cannot fake."""
        write_scene4d(small_scene(), tmp_path / "s.g4d")
        data = bytearray((tmp_path / "s.g4d").read_bytes())
        data[12] |= 0x08  # a low-half flag bit this reader does not implement
        with pytest.raises(CaptureFormatError, match="breaking flag"):
            read_scene4d_header(bytes(data))
