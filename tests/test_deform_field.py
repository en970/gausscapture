"""Rotation algebra, exact KNN, and the SE(3) scaffold.

The assertions here are against closed-form answers -- a lattice whose neighbour
spacing is known by construction, a rotation composed by hand, a rigid motion
applied to every node at once -- rather than against whatever the code produced
on the day it was written. That is the only kind of test worth having for a
contract three separate implementations (PyTorch, NumPy, GLSL) have to agree on.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gausscapture.recon.deform.field import (  # noqa: E402
    ExplicitTrajectoryField,
    KPlanesField,
    Skinning,
    build_skinning,
    default_frame_times,
    deform_gaussians,
    interpolate_nodes,
    node_edges,
    scene_aabb,
    seed_nodes,
)
from gausscapture.recon.deform.knn import (  # noqa: E402
    MIN_SQ_DIST,
    init_log_scales,
    knn_indices,
    knn_mean_sq_dist,
)
from gausscapture.recon.deform.so3 import (  # noqa: E402
    align_hemisphere,
    axis_angle_to_quat,
    nlerp,
    quat_conjugate,
    quat_multiply,
    quat_normalise,
    quat_rotate,
    quat_to_axis_angle,
    quat_to_matrix,
)


def _lattice(n: int = 5, spacing: float = 0.25) -> torch.Tensor:
    grid = torch.arange(n, dtype=torch.float32) * spacing
    x, y, z = torch.meshgrid(grid, grid, grid, indexing="ij")
    return torch.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], dim=-1)


def _brute_force_mean_sq(points: np.ndarray, k: int) -> np.ndarray:
    """Deliberately naive reference: a Python loop over every pair."""
    out = np.empty(len(points))
    for i, p in enumerate(points):
        d = [float(np.sum((q - p) ** 2)) for j, q in enumerate(points) if j != i]
        out[i] = float(np.mean(sorted(d)[:k]))
    return out


# -- so3 --------------------------------------------------------------------


def test_quat_multiply_matches_matrix_product():
    """R(a*b) == R(a) @ R(b) -- the composition order the whole chain assumes."""
    g = torch.Generator().manual_seed(7)
    a = quat_normalise(torch.randn(64, 4, generator=g))
    b = quat_normalise(torch.randn(64, 4, generator=g))
    lhs = quat_to_matrix(quat_multiply(a, b))
    rhs = quat_to_matrix(a) @ quat_to_matrix(b)
    assert torch.allclose(lhs, rhs, atol=1e-5)


def test_quat_rotate_matches_matrix_apply():
    g = torch.Generator().manual_seed(11)
    q = quat_normalise(torch.randn(32, 4, generator=g))
    p = torch.randn(32, 3, generator=g)
    direct = quat_rotate(q, p)
    viamat = torch.einsum("nij,nj->ni", quat_to_matrix(q), p)
    assert torch.allclose(direct, viamat, atol=1e-5)


def test_quat_multiply_by_conjugate_is_identity():
    g = torch.Generator().manual_seed(3)
    q = quat_normalise(torch.randn(16, 4, generator=g))
    identity = quat_multiply(q, quat_conjugate(q))
    expected = torch.zeros(16, 4)
    expected[:, 0] = 1.0
    assert torch.allclose(identity, expected, atol=1e-6)


def test_axis_angle_round_trip_including_the_identity():
    """The zero rotation is where every trajectory parameter starts."""
    g = torch.Generator().manual_seed(5)
    aa = torch.randn(200, 3, generator=g) * 0.7
    aa[0] = 0.0
    q = axis_angle_to_quat(aa)
    assert torch.allclose(q.norm(dim=-1), torch.ones(200), atol=1e-6)
    assert torch.allclose(quat_to_axis_angle(q), aa, atol=1e-5)


def test_axis_angle_gradient_is_finite_at_zero():
    aa = torch.zeros(1, 3, requires_grad=True)
    axis_angle_to_quat(aa).sum().backward()
    assert torch.isfinite(aa.grad).all()


def test_axis_angle_known_quarter_turn():
    q = axis_angle_to_quat(torch.tensor([[0.0, 0.0, math.pi / 2]]))
    rotated = quat_rotate(q, torch.tensor([[1.0, 0.0, 0.0]]))
    assert torch.allclose(rotated, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6)


def test_align_hemisphere_and_nlerp_ignore_the_sign_ambiguity():
    a = quat_normalise(torch.tensor([[1.0, 0.1, 0.0, 0.0]]))
    b = -quat_normalise(torch.tensor([[1.0, 0.2, 0.0, 0.0]]))
    assert (align_hemisphere(b, a) * a).sum() > 0
    mid = nlerp(a, b, torch.tensor([[0.5]]))
    # Blending the flipped copy must not collapse towards the zero quaternion.
    assert float(mid.norm()) == pytest.approx(1.0, abs=1e-6)
    assert float((mid * a).sum()) > 0.99


# -- knn --------------------------------------------------------------------


def test_knn_mean_sq_dist_on_a_lattice_with_known_spacing():
    spacing = 0.25
    points = _lattice(5, spacing)
    mean_sq = knn_mean_sq_dist(points, k=3)
    # An interior lattice point has six neighbours at exactly `spacing`, so the
    # mean over the nearest three is spacing squared regardless of tie-breaking.
    interior = (points > spacing * 0.5).all(dim=1) & (points < spacing * 3.5).all(dim=1)
    assert bool(interior.any())
    assert torch.allclose(mean_sq[interior], torch.full((int(interior.sum()),), spacing**2), atol=1e-6)


def test_knn_matches_a_brute_force_reference():
    g = torch.Generator().manual_seed(19)
    points = torch.randn(120, 3, generator=g)
    ours = knn_mean_sq_dist(points, k=4, chunk=17).numpy()
    reference = _brute_force_mean_sq(points.numpy(), k=4)
    assert np.allclose(ours, reference, atol=1e-5)


def test_knn_duplicate_points_do_not_produce_infinite_log_scales():
    points = torch.zeros(8, 3)
    log_scales = init_log_scales(points)
    assert torch.isfinite(log_scales).all()
    assert float(knn_mean_sq_dist(points).min()) == pytest.approx(MIN_SQ_DIST)


def test_knn_handles_fewer_points_than_neighbours():
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert torch.allclose(knn_mean_sq_dist(points, k=3), torch.ones(2))
    assert knn_mean_sq_dist(torch.zeros(1, 3)).shape == (1,)


def test_knn_indices_pad_when_there_are_too_few_nodes():
    idx, dist = knn_indices(torch.randn(5, 3), torch.zeros(2, 3), k=4)
    assert idx.shape == (5, 4)
    assert dist.shape == (5, 4)


def test_init_log_scales_tracks_spacing():
    coarse = init_log_scales(_lattice(4, 1.0))
    fine = init_log_scales(_lattice(4, 0.1))
    assert float(coarse.mean() - fine.mean()) == pytest.approx(math.log(10.0), abs=1e-4)


# -- skinning and the deformation contract ----------------------------------


def test_skinning_weights_sum_to_one():
    g = torch.Generator().manual_seed(23)
    points = torch.randn(200, 3, generator=g)
    nodes = seed_nodes(points, 12)
    skin = build_skinning(points, nodes, k=4)
    assert skin.idx.shape == (200, 4)
    assert torch.allclose(skin.weights.sum(dim=1), torch.ones(200), atol=1e-6)
    assert bool((skin.weights >= 0).all())


def test_seed_nodes_cover_the_cloud_better_than_a_prefix():
    points = _lattice(6, 0.2)
    nodes = seed_nodes(points, 10)
    assert nodes.shape == (10, 3)
    covered = torch.cdist(points, nodes).min(dim=1).values.max()
    naive = torch.cdist(points, points[:10]).min(dim=1).values.max()
    assert float(covered) < float(naive)


def test_node_edges_are_undirected_and_unique():
    nodes = _lattice(3, 1.0)
    edges = node_edges(nodes, k=6)
    assert edges.shape[1] == 2
    assert bool((edges[:, 0] < edges[:, 1]).all())
    keys = {(int(a), int(b)) for a, b in edges}
    assert len(keys) == edges.shape[0]


def _rigid_node_transform(nodes, axis, shift):
    """The node parameters that encode one global motion ``p -> R p + shift``.

    The scaffold stores each node's *own* displacement and rotates about that
    node's rest position, so a global motion is ``d_k = R n_k + shift - n_k``.
    The constant ``d_k = shift`` is a different, non-rigid motion -- every node
    spinning about itself and the cloud shearing between them -- and getting
    this wrong is the single mistake the whole rigid-motion test exists to
    catch, so it is derived here once and shared.
    """
    q = axis_angle_to_quat(axis).reshape(1, 4).expand(nodes.shape[0], 4).contiguous()
    return q, quat_rotate(q, nodes) + shift - nodes


def test_deform_gaussians_reproduces_a_rigid_motion_exactly():
    """Every node carrying the same transform must move the cloud rigidly."""
    g = torch.Generator().manual_seed(29)
    points = torch.randn(64, 3, generator=g)
    nodes = seed_nodes(points, 8)
    skin = build_skinning(points, nodes, k=4)
    quats = torch.zeros(64, 4)
    quats[:, 0] = 1.0

    shift = torch.tensor([0.1, 0.2, -0.3])
    q, trans = _rigid_node_transform(nodes, torch.tensor([0.3, -0.5, 0.2]), shift)

    means_def, quats_def = deform_gaussians(points, quats, q, trans, nodes, skin)
    expected = quat_rotate(q[:1].expand(64, 4), points) + shift
    assert torch.allclose(means_def, expected, atol=1e-5)
    assert torch.allclose(quats_def, q[:1].expand(64, 4), atol=1e-5)


def test_deform_gaussians_is_rigid_for_any_weights_that_sum_to_one():
    """Rigidity comes from the partition of unity, not from the binding.

    ``build_skinning`` produces smooth, distance-ordered weights, so the test
    above could pass on a formula that happened to be right only for a nearly
    single-node influence. These weights are arbitrary -- random, and bound to
    randomly chosen nodes rather than near ones -- and the canonical
    orientations are arbitrary too, which is what pins the left-multiplication
    order of the rotation blend.
    """
    g = torch.Generator().manual_seed(37)
    points = torch.randn(50, 3, generator=g)
    nodes = torch.randn(6, 3, generator=g)
    canonical = quat_normalise(torch.randn(50, 4, generator=g))

    idx = torch.randint(0, nodes.shape[0], (50, 4), generator=g)
    w = torch.rand(50, 4, generator=g) + 0.05
    # Descending is part of the container contract: influence 0 is the
    # hemisphere reference that none of the three readers searches for.
    w = w.sort(dim=1, descending=True).values
    skin = Skinning(idx, w / w.sum(dim=1, keepdim=True))

    shift = torch.tensor([-0.4, 0.05, 0.7])
    q, trans = _rigid_node_transform(nodes, torch.tensor([-0.9, 0.2, 0.35]), shift)

    means_def, quats_def = deform_gaussians(points, canonical, q, trans, nodes, skin)
    q_all = q[:1].expand(50, 4)
    assert torch.allclose(means_def, quat_rotate(q_all, points) + shift, atol=1e-5)
    assert torch.allclose(quats_def, quat_multiply(q_all, canonical), atol=1e-5)


def test_deform_gaussians_is_not_rigid_when_the_pivots_are_not_absorbed():
    """The guard on the test above: constant node translations must *not* pass.

    If this ever starts producing a rigid motion, the skinning has stopped
    rotating about each node's rest position and the container, the NumPy
    reference and the shader are all describing something else.
    """
    g = torch.Generator().manual_seed(41)
    points = torch.randn(40, 3, generator=g)
    nodes = seed_nodes(points, 6)
    skin = build_skinning(points, nodes, k=4)
    quats = torch.zeros(40, 4)
    quats[:, 0] = 1.0

    shift = torch.tensor([0.1, 0.2, -0.3])
    q = axis_angle_to_quat(torch.tensor([0.3, -0.5, 0.2])).reshape(1, 4).expand(6, 4).contiguous()
    means_def, _ = deform_gaussians(points, quats, q, shift.expand(6, 3), nodes, skin)
    assert not torch.allclose(means_def, quat_rotate(q[:1].expand(40, 4), points) + shift, atol=1e-3)


def test_deform_gaussians_leaves_static_gaussians_untouched():
    points = torch.randn(20, 3, generator=torch.Generator().manual_seed(31))
    nodes = seed_nodes(points, 4)
    skin = build_skinning(points, nodes, k=2)
    quats = torch.zeros(20, 4)
    quats[:, 0] = 1.0
    q = axis_angle_to_quat(torch.tensor([0.4, 0.0, 0.0])).reshape(1, 4).expand(4, 4).contiguous()
    trans = torch.full((4, 3), 0.5)
    dynamic = torch.zeros(20)
    dynamic[:10] = 1.0

    means_def, quats_def = deform_gaussians(points, quats, q, trans, nodes, skin, dynamic=dynamic)
    assert torch.allclose(means_def[10:], points[10:], atol=0)
    assert torch.allclose(quats_def[10:], quats[10:], atol=0)
    assert not torch.allclose(means_def[:10], points[:10])


# -- fields -----------------------------------------------------------------


@pytest.mark.parametrize("kind", ["explicit", "kplanes"])
def test_field_is_the_identity_at_initialisation(kind):
    nodes = _lattice(3, 0.5)
    times = default_frame_times(9)
    if kind == "explicit":
        fld = ExplicitTrajectoryField(nodes, times)
    else:
        fld = KPlanesField(nodes, aabb=scene_aabb(nodes), base_resolution=8, time_resolution=5)
    quat, trans = fld(times)
    assert quat.shape == (9, nodes.shape[0], 4)
    assert trans.shape == (9, nodes.shape[0], 3)
    assert torch.allclose(trans, torch.zeros_like(trans), atol=1e-6)
    assert torch.allclose(quat[..., 0], torch.ones_like(quat[..., 0]), atol=1e-6)


@pytest.mark.parametrize("kind", ["explicit", "kplanes"])
def test_field_is_exactly_the_identity_at_the_anchor_after_training_moves_it(kind):
    """The anchor gauge is what makes the canonical splat mean a real instant."""
    nodes = _lattice(3, 0.5)
    times = default_frame_times(7)
    if kind == "explicit":
        fld = ExplicitTrajectoryField(nodes, times, anchor_time=0.0)
    else:
        fld = KPlanesField(
            nodes, aabb=scene_aabb(nodes), base_resolution=8, time_resolution=5, anchor_time=0.0
        )
    with torch.no_grad():
        for p in fld.parameters():
            p.add_(torch.randn(p.shape, generator=torch.Generator().manual_seed(2)) * 0.05)

    quat, trans = fld(torch.tensor([0.0]))
    # The tolerance is the float32 noise floor, not a fitting allowance: at the
    # anchor the composition subtracts a quantity from itself, so the only
    # residual is the rounding of `q * conj(q)` (a fused multiply-add spoils the
    # otherwise exact cancellation, by about 1e-9 on the quaternion). Anything
    # at 1e-5 would mean a canonical pose that genuinely drifts.
    assert torch.allclose(trans, torch.zeros_like(trans), atol=1e-8)
    assert torch.allclose(quat[..., 0].abs(), torch.ones_like(quat[..., 0]), atol=1e-8)
    # ... and something genuinely moves elsewhere, or the test above is vacuous.
    _, trans_late = fld(torch.tensor([1.0]))
    assert float(trans_late.abs().max().detach()) > 1e-6


def test_explicit_field_interpolates_between_keyframes():
    nodes = torch.zeros(2, 3)
    times = torch.tensor([0.0, 1.0])
    fld = ExplicitTrajectoryField(nodes, times, anchor_time=None)
    with torch.no_grad():
        fld.translation[1] = torch.tensor([1.0, 0.0, 0.0])
    _, trans = fld(torch.tensor([0.0, 0.25, 1.0]))
    assert torch.allclose(trans[:, 0, 0], torch.tensor([0.0, 0.25, 1.0]), atol=1e-6)


def test_explicit_field_rejects_unsorted_keyframes():
    with pytest.raises(ValueError):
        ExplicitTrajectoryField(torch.zeros(2, 3), torch.tensor([0.0, 0.0, 1.0]))


def test_kplanes_scales_are_hard_bounds_the_head_cannot_exceed():
    """Drive the head to saturation and check the stated bounds hold anyway.

    The head is an unconstrained MLP, so "bounded" is a claim about the squash
    in front of the scales and nothing else. The two scales differ here, and
    neither is 1, so a bound that came from somewhere other than the parameter
    it is named after would be visible.
    """
    nodes = _lattice(3, 0.5)
    rotation_scale, translation_scale = 0.2, 0.05
    fld = KPlanesField(
        nodes,
        aabb=scene_aabb(nodes),
        base_resolution=8,
        time_resolution=5,
        rotation_scale=rotation_scale,
        translation_scale=translation_scale,
        anchor_time=None,
    )
    with torch.no_grad():
        # Far past anything training would reach: an unbounded linear head puts
        # the translation in the hundreds of world units from here.
        for p in fld.head[-1].parameters():
            p.add_(50.0)
    with torch.no_grad():
        quat, trans = fld(torch.linspace(0, 1, 5))

    assert torch.isfinite(quat).all() and torch.isfinite(trans).all()
    assert torch.allclose(quat.norm(dim=-1), torch.ones(5, nodes.shape[0]), atol=1e-5)
    # The bound is checked at the precision the field computes in: float32 has
    # no exact 0.05, and demanding the double-precision literal would be a
    # test of Python's parser rather than of the field.
    assert float(trans.abs().max()) <= float(np.float32(translation_scale))
    assert float(quat_to_axis_angle(quat).abs().max()) <= float(np.float32(rotation_scale)) + 1e-6
    # ... and the bound is attained, so it is a bound on a live output rather
    # than on a field that has quietly stopped moving.
    assert float(trans.abs().max()) == pytest.approx(translation_scale, rel=1e-4)
    assert float(quat_to_axis_angle(quat).abs().max()) == pytest.approx(rotation_scale, rel=1e-4)


def test_kplanes_planes_split_into_spatial_and_spacetime():
    nodes = _lattice(2, 1.0)
    fld = KPlanesField(nodes, aabb=scene_aabb(nodes), base_resolution=8, multires=(1, 2))
    assert len(fld.spatial_planes()) == 6
    assert len(fld.spacetime_planes()) == 6
    # Time lives on the height axis of the space-time planes; the regulariser
    # differences dim=-2 and would silently smooth space if this flipped.
    assert all(p.shape[-2] == 25 for p in fld.spacetime_planes())


def test_kplanes_gradients_reach_every_plane():
    nodes = _lattice(2, 1.0)
    fld = KPlanesField(nodes, aabb=scene_aabb(nodes), base_resolution=8, anchor_time=None)
    with torch.no_grad():
        for p in fld.head[-1].parameters():
            p.add_(0.5)
    quat, trans = fld(torch.tensor([0.3]))
    (quat.sum() + trans.sum()).backward()
    for planes in fld.scales:
        for p in planes:
            assert p.grad is not None and float(p.grad.abs().sum()) > 0


def test_interpolate_nodes_matches_the_field_it_was_baked_from():
    nodes = _lattice(2, 1.0)
    times = default_frame_times(5)
    fld = ExplicitTrajectoryField(nodes, times, anchor_time=None)
    with torch.no_grad():
        fld.translation.copy_(torch.randn(fld.translation.shape, generator=torch.Generator().manual_seed(4)) * 0.1)
    quat, trans = fld.bake(times)
    q, d = interpolate_nodes(quat, trans, times, times)
    assert torch.allclose(d, trans, atol=1e-6)
    assert torch.allclose(q.abs(), quat.abs(), atol=1e-6)


def test_scene_aabb_pads_the_bounds():
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    box = scene_aabb(points, margin=0.1)
    assert float(box[0, 0]) < 0.0 and float(box[1, 2]) > 3.0
