"""The deformation contract, checked across the two implementations we can run.

``export/deform_reference.py`` is the NumPy statement of how a scaffold moves a
gaussian; ``recon/deform/field.py`` is the PyTorch one the trainer optimises; the
GLSL in ``report/viewer_4d.js`` is the third. Nothing forces them to agree, and
every way they can disagree is silent -- a transposed rotation is a subject who
leans the wrong way, an unaligned quaternion blend is a head that snaps round
between two frames, a scaffold applied about the origin instead of about each
node's rest position is a face that swells as it turns. None of those raise.

So the two implementations that run in this process are checked against each
other and against closed forms that neither of them computes.
"""

from __future__ import annotations

import numpy as np
import pytest

# The trainer's half of this comparison is PyTorch, and torch is the ``train4d``
# extra rather than a core dependency -- CI installs the package with ``[dev]``,
# which does not carry it. A bare module-scope ``import torch`` therefore aborted
# collection for the whole session and ran zero tests, which is worse than
# running none of this file. ``test_deform_field.py`` already uses this pattern.
torch = pytest.importorskip("torch", reason="the deformation trainer needs the train4d extra")

from gausscapture.export.deform_reference import (  # noqa: E402
    deform_scene,
    deformed_depths,
    fold_view_row,
    quaternion_multiply,
    quaternion_to_matrix,
    sample_nodes,
    skin_positions,
    skin_quaternions,
)
from gausscapture.export.scene4d import Scene4D  # noqa: E402
from gausscapture.export.synthetic4d import (  # noqa: E402
    PIVOT,
    expected_position,
    synthetic_scene,
)
from gausscapture.recon.deform.field import (  # noqa: E402
    Skinning,
    deform_gaussians,
    interpolate_nodes,
)


def random_binding(count: int, nodes: int, k: int, seed: int):
    """Indices and descending weights, the order both readers rely on."""
    rng = np.random.default_rng(seed)
    idx = np.stack([rng.permutation(nodes)[:k] for _ in range(count)])
    weights = rng.uniform(0.05, 1.0, size=(count, k))
    weights /= weights.sum(axis=1, keepdims=True)
    # Influence 0 is the hemisphere reference in all three readers; none of them
    # searches for the largest weight, so the rows have to arrive sorted.
    order = np.argsort(-weights, axis=1)
    return np.take_along_axis(idx, order, axis=1), np.take_along_axis(weights, order, axis=1)


def random_quats(shape, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(*shape, 4))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


@pytest.fixture()
def bundle():
    """A small scaffold, some gaussians, and a binding between them."""
    rng = np.random.default_rng(2026)
    nodes, count, k = 9, 40, 4
    node_rest = rng.normal(scale=0.3, size=(nodes, 3))
    means = rng.normal(scale=0.3, size=(count, 3))
    quats = random_quats((count,), seed=7)
    # Small rotations: a scaffold node that tumbles is not a thing the protocol
    # can produce, and large ones make the two blends disagree for real reasons.
    node_quats = _shrink(random_quats((nodes,), seed=8), 0.15)
    node_trans = rng.normal(scale=0.05, size=(nodes, 3))
    idx, weights = random_binding(count, nodes, k, seed=9)
    return {
        "node_rest": node_rest, "means": means, "quats": quats,
        "node_quats": node_quats, "node_trans": node_trans,
        "idx": idx, "weights": weights,
    }


def _shrink(quats: np.ndarray, fraction: float) -> np.ndarray:
    """Scale each rotation's angle down, keeping its axis."""
    angle = 2.0 * np.arctan2(np.linalg.norm(quats[..., 1:], axis=-1), np.abs(quats[..., 0]))
    axis = quats[..., 1:] / np.maximum(
        np.linalg.norm(quats[..., 1:], axis=-1, keepdims=True), 1e-12
    )
    half = 0.5 * angle * fraction
    return np.concatenate([np.cos(half)[..., None], axis * np.sin(half)[..., None]], axis=-1)


class TestTheTwoImplementationsAgree:
    def test_positions(self, bundle):
        numpy_means = skin_positions(
            bundle["means"], bundle["node_rest"], bundle["node_quats"],
            bundle["node_trans"], bundle["idx"], bundle["weights"],
        )
        torch_means, _ = deform_gaussians(
            torch.tensor(bundle["means"], dtype=torch.float32),
            torch.tensor(bundle["quats"], dtype=torch.float32),
            torch.tensor(bundle["node_quats"], dtype=torch.float32),
            torch.tensor(bundle["node_trans"], dtype=torch.float32),
            torch.tensor(bundle["node_rest"], dtype=torch.float32),
            Skinning(torch.tensor(bundle["idx"]), torch.tensor(bundle["weights"],
                                                              dtype=torch.float32)),
        )
        assert numpy_means == pytest.approx(torch_means.numpy(), abs=1e-6)

    def test_orientations(self, bundle):
        numpy_quats = skin_quaternions(
            bundle["quats"], bundle["node_quats"], bundle["idx"], bundle["weights"]
        )
        _, torch_quats = deform_gaussians(
            torch.tensor(bundle["means"], dtype=torch.float32),
            torch.tensor(bundle["quats"], dtype=torch.float32),
            torch.tensor(bundle["node_quats"], dtype=torch.float32),
            torch.tensor(bundle["node_trans"], dtype=torch.float32),
            torch.tensor(bundle["node_rest"], dtype=torch.float32),
            Skinning(torch.tensor(bundle["idx"]), torch.tensor(bundle["weights"],
                                                              dtype=torch.float32)),
        )
        got = torch_quats.numpy()
        # q and -q are the same rotation, so compare on one hemisphere.
        sign = np.sign(np.sum(numpy_quats * got, axis=1, keepdims=True))
        assert numpy_quats == pytest.approx(got * sign, abs=1e-6)

    def test_interpolation_between_keyframes(self):
        frames, nodes = 6, 5
        node_quats = _shrink(random_quats((frames, nodes), seed=21), 0.2)
        node_trans = np.random.default_rng(22).normal(scale=0.05, size=(frames, nodes, 3))
        frame_times = np.linspace(0.0, 1.0, frames)

        for tau in (0.0, 0.5, 1.5, 2.25, float(frames - 1)):
            numpy_q, numpy_t = sample_nodes(node_quats, node_trans, tau)
            # The reference indexes time in frames, the field in normalised tau;
            # the two are the same instant, expressed in the two units that
            # actually appear in the file and in the trainer.
            t = np.interp(tau, np.arange(frames), frame_times)
            torch_q, torch_t = interpolate_nodes(
                torch.tensor(node_quats, dtype=torch.float64),
                torch.tensor(node_trans, dtype=torch.float64),
                torch.tensor(frame_times, dtype=torch.float64),
                torch.tensor([t], dtype=torch.float64),
            )
            got = torch_q[0].numpy()
            sign = np.sign(np.sum(numpy_q * got, axis=1, keepdims=True))
            assert numpy_q == pytest.approx(got * sign, abs=1e-9)
            assert numpy_t == pytest.approx(torch_t[0].numpy(), abs=1e-9)


class TestRigidity:
    """The property the whole representation exists to have."""

    def test_a_scaffold_carrying_one_rigid_motion_moves_every_gaussian_rigidly(self, bundle):
        rotation_q = _shrink(random_quats((1,), seed=31), 1.0)[0]
        rotation = quaternion_to_matrix(rotation_q)
        shift = np.array([0.02, -0.05, 0.11])
        rest = bundle["node_rest"]

        # A global p -> Rp + s is the pivot-absorbing per-node displacement, not
        # the constant s: with a constant the blend leaves the residual
        # sum_i w_i (n_i - R n_i), which vanishes only when every influence
        # shares a rest position.
        node_quats = np.repeat(rotation_q[None, :], len(rest), axis=0)
        node_trans = rest @ rotation.T + shift - rest

        moved = skin_positions(
            bundle["means"], rest, node_quats, node_trans, bundle["idx"], bundle["weights"]
        )
        expected = bundle["means"] @ rotation.T + shift
        assert moved == pytest.approx(expected, abs=1e-6)

    def test_a_constant_displacement_is_not_rigid_which_is_why_the_pivot_matters(self, bundle):
        """The guard that keeps the test above from being trivially satisfiable."""
        rotation_q = _shrink(random_quats((1,), seed=31), 1.0)[0]
        rotation = quaternion_to_matrix(rotation_q)
        shift = np.array([0.02, -0.05, 0.11])
        rest = bundle["node_rest"]

        moved = skin_positions(
            bundle["means"], rest,
            np.repeat(rotation_q[None, :], len(rest), axis=0),
            np.repeat(shift[None, :], len(rest), axis=0),
            bundle["idx"], bundle["weights"],
        )
        expected = bundle["means"] @ rotation.T + shift
        assert np.abs(moved - expected).max() > 1e-3

    def test_an_identity_scaffold_leaves_everything_exactly_where_it_was(self, bundle):
        rest = bundle["node_rest"]
        identity = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(rest), 1))
        moved = skin_positions(
            bundle["means"], rest, identity, np.zeros_like(rest),
            bundle["idx"], bundle["weights"],
        )
        assert moved == pytest.approx(bundle["means"], abs=1e-6)


class TestHemisphereAlignment:
    def test_a_sign_flipped_node_describes_the_same_motion(self, bundle):
        flipped = bundle["node_quats"].copy()
        flipped[0] = -flipped[0]

        base = skin_quaternions(
            bundle["quats"], bundle["node_quats"], bundle["idx"], bundle["weights"]
        )
        after = skin_quaternions(
            bundle["quats"], flipped, bundle["idx"], bundle["weights"]
        )
        sign = np.sign(np.sum(base * after, axis=1, keepdims=True))
        # A negated quaternion is the same rotation. Without the alignment the
        # blend cancels and the orientation becomes arbitrary, which is visible
        # as a head that snaps round between two frames.
        assert base == pytest.approx(after * sign, abs=1e-6)

    def test_it_also_holds_across_a_keyframe_boundary(self):
        node_quats = _shrink(random_quats((2, 4), seed=41), 0.2)
        flipped = node_quats.copy()
        flipped[1] = -flipped[1]
        trans = np.zeros((2, 4, 3))

        base, _ = sample_nodes(node_quats, trans, 0.5)
        after, _ = sample_nodes(flipped, trans, 0.5)
        sign = np.sign(np.sum(base * after, axis=1, keepdims=True))
        assert base == pytest.approx(after * sign, abs=1e-9)


class TestDepthFolding:
    """The optimisation the viewer's per-frame re-sort depends on."""

    def test_folded_depths_equal_the_depths_of_the_deformed_positions(self, bundle):
        view_row = np.array([0.31, -0.62, 0.72])
        moved = skin_positions(
            bundle["means"], bundle["node_rest"], bundle["node_quats"],
            bundle["node_trans"], bundle["idx"], bundle["weights"],
        )
        direct = moved.astype(np.float64) @ view_row

        g, h = fold_view_row(
            bundle["node_rest"], bundle["node_quats"], bundle["node_trans"], view_row
        )
        folded = deformed_depths(bundle["means"], bundle["idx"], bundle["weights"], g, h)
        assert folded == pytest.approx(direct, abs=1e-6)

    def test_the_fold_is_per_node_not_per_gaussian(self, bundle):
        g, h = fold_view_row(
            bundle["node_rest"], bundle["node_quats"], bundle["node_trans"],
            np.array([0.0, 0.0, 1.0]),
        )
        assert g.shape == (len(bundle["node_rest"]), 3)
        assert h.shape == (len(bundle["node_rest"]),)


class TestSampleNodes:
    def test_it_clamps_rather_than_extrapolating_past_the_clip(self):
        node_quats = _shrink(random_quats((4, 3), seed=51), 0.2)
        trans = np.random.default_rng(52).normal(scale=0.05, size=(4, 3, 3))

        first_q, first_t = sample_nodes(node_quats, trans, -5.0)
        assert first_q == pytest.approx(node_quats[0], abs=1e-12)
        assert first_t == pytest.approx(trans[0], abs=1e-12)

        last_q, last_t = sample_nodes(node_quats, trans, 99.0)
        assert last_q == pytest.approx(node_quats[-1], abs=1e-12)
        assert last_t == pytest.approx(trans[-1], abs=1e-12)

    def test_a_keyframe_is_reproduced_exactly(self):
        node_quats = _shrink(random_quats((5, 3), seed=53), 0.2)
        trans = np.random.default_rng(54).normal(scale=0.05, size=(5, 3, 3))
        for frame in range(5):
            q, t = sample_nodes(node_quats, trans, float(frame))
            assert q == pytest.approx(node_quats[frame], abs=1e-12)
            assert t == pytest.approx(trans[frame], abs=1e-12)


class TestQuaternionAlgebra:
    def test_the_product_composes_rotations_in_matrix_order(self):
        a, b = random_quats((2,), seed=61)
        assert quaternion_to_matrix(quaternion_multiply(a, b)) == pytest.approx(
            quaternion_to_matrix(a) @ quaternion_to_matrix(b), abs=1e-12
        )

    def test_the_matrix_is_a_rotation(self):
        for q in random_quats((8,), seed=62):
            m = quaternion_to_matrix(q)
            assert m @ m.T == pytest.approx(np.eye(3), abs=1e-12)
            assert np.linalg.det(m) == pytest.approx(1.0, abs=1e-12)


class TestSyntheticScene:
    """The generated scene has a known answer; the reader has to reproduce it."""

    @pytest.fixture()
    def scene(self) -> Scene4D:
        return synthetic_scene(gaussians=400, nodes=10, frames=8, yaw_degrees=12.0)

    def test_the_static_background_never_moves(self, scene: Scene4D):
        for tau in (0.0, 2.5, float(scene.frame_count - 1)):
            means, quats = deform_scene(scene, tau)
            assert means[scene.dynamic_count:] == pytest.approx(
                scene.means[scene.dynamic_count:], abs=0.0
            )
            assert quats[scene.dynamic_count:] == pytest.approx(
                scene.quats[scene.dynamic_count:], abs=0.0
            )

    def test_the_subject_moves_and_returns_to_where_it_started(self, scene: Scene4D):
        start, _ = deform_scene(scene, 0.0)
        middle, _ = deform_scene(scene, scene.frame_count / 4.0)
        assert start[: scene.dynamic_count] == pytest.approx(
            scene.means[: scene.dynamic_count], abs=1e-6
        )
        # A head turning through twelve degrees at ten centimetres moves about a
        # centimetre; anything under a millimetre would mean nothing happened.
        assert np.abs(middle[: scene.dynamic_count] - start[: scene.dynamic_count]).max() > 1e-3

    def test_every_gaussian_lands_on_the_blend_of_its_influences_exactly(self, scene: Scene4D):
        """Each summand is ``R_i (p - pivot) + pivot``, so the blend is closed form.

        The scene's node translations are built as ``(R_j - I)(n_j - pivot)``,
        which is exactly the pivot-absorbing form, so every influence maps a
        point to the same expression about the shared pivot and the weighted sum
        collapses to one matrix per gaussian.
        """
        frame = 3
        means, _ = deform_scene(scene, float(frame))
        idx, weights = scene.skin_indices, scene.skin_weights
        rotations = quaternion_to_matrix(scene.node_quats[frame])

        blended = np.einsum("nk,nkij->nij", weights.astype(np.float64), rotations[idx])
        local = scene.means[: scene.dynamic_count].astype(np.float64) - PIVOT
        expected = np.einsum("nij,nj->ni", blended, local) + PIVOT
        assert means[: scene.dynamic_count] == pytest.approx(expected, abs=1e-6)

    def test_it_tracks_the_closed_form_the_generator_publishes(self, scene: Scene4D):
        """``expected_position`` is the eye-level claim: a yaw ramped up the head.

        A gaussian bound to several nodes sees the weighted average of their
        ramps, and the blend of rotations is not exactly the rotation of the
        blended angle -- so this is checked to a tolerance rather than exactly.
        A tenth of a millimetre on a scene whose subject moves a centimetre is
        the difference between "the same motion" and "some other motion".
        """
        frame = 3
        means, _ = deform_scene(scene, float(frame))
        yaw = np.radians(scene.meta["yaw_amplitude_deg"]) * np.sin(
            2 * np.pi * frame / scene.frame_count
        )
        # Per-node ramp read back off the file rather than recomputed from the
        # generator's constants: the y component of the node quaternion is
        # sin(angle / 2) about +Y, and angle is yaw * ramp.
        angle = 2.0 * np.arcsin(np.clip(scene.node_quats[frame][:, 2], -1.0, 1.0))
        ramp = angle / yaw
        effective = (scene.skin_weights.astype(np.float64) * ramp[scene.skin_indices]).sum(axis=1)

        for i in (0, 17, 101, scene.dynamic_count - 1):
            reference = expected_position(scene, scene.means[i], frame, float(effective[i]))
            assert means[i] == pytest.approx(reference, abs=1e-4)
