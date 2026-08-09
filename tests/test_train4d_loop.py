"""What the training loop actually does, rather than what it reports having done.

Everything here was uncovered before it existed, and each gap had the same
shape: the suite asserted on numbers that were *inputs* passing through. The one
end-to-end run asserted the gaussian count (the input point count), the node and
frame counts (inputs), the device string, and a label substring -- so neutering
every optimiser step, in the parameters and in the field alike, left the whole
suite green. `final_loss`, `train_psnr_db` and `holdout_psnr_db` were never
asserted anywhere.

The densification half was worse than untested, it was unreachable: the
end-to-end fit runs four iterations while ``ScheduleConfig.refine_start`` is 500,
so ``should_densify`` is False at every step and ``ReferenceMcmcStrategy.refine``,
``add_noise``, ``CanonicalParams.relocate``/``duplicate`` and the whole
rebind-versus-carry-forward decision never ran. On a real 3,000 + 12,000
iteration run those are the paths that take a scene from ~200 COLMAP points to
400,000 gaussians.

Everything below is CPU, seconds, and synthesised in-process.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="the trainer is the train4d extra")

from gausscapture.errors import PipelineStateError  # noqa: E402
from gausscapture.recon.deform.field import (  # noqa: E402
    ExplicitTrajectoryField,
    KPlanesField,
    default_frame_times,
    scene_aabb,
)
from gausscapture.recon.deform.losses import (  # noqa: E402
    l1_time_planes,
    plane_tv_loss,
    psnr,
    time_smoothness_loss,
)
from gausscapture.recon.deform.params import CanonicalParams  # noqa: E402
from gausscapture.recon.deform.raster import ReferenceRasterizer  # noqa: E402
from gausscapture.recon.deform.schedule import (  # noqa: E402
    ReferenceMcmcStrategy,
    ScheduleConfig,
    TrainingSchedule,
)
from gausscapture.recon.deform.train4d import (  # noqa: E402
    SCENE4D_VERSION,
    FixedCameraClip,
    Train4DConfig,
    load_checkpoint,
    save_checkpoint,
    train,
)

WIDTH, HEIGHT, FRAMES, POINTS = 32, 24, 6, 200

INTRINSICS = torch.tensor(
    [[30.0, 0.0, WIDTH / 2], [0.0, 30.0, HEIGHT / 2], [0.0, 0.0, 1.0]]
)


def truth_cloud(seed: int = 3):
    """A cloud in front of the camera, with the colours the fit has to find."""
    generator = torch.Generator().manual_seed(seed)
    points = torch.rand(POINTS, 3, generator=generator) * 0.8 - 0.4
    points[:, 2] += 2.0
    colors = torch.rand(POINTS, 3, generator=generator)
    return points, colors, generator


def render(params: CanonicalParams, viewmat) -> torch.Tensor:
    with torch.no_grad():
        return (
            ReferenceRasterizer()
            .render(
                means=params.params["means"],
                quats=params.normalised_quats(),
                scales=params.scales(),
                opacities=params.opacities(),
                colors=params.colors(),
                viewmat=viewmat,
                intrinsics=INTRINSICS,
                width=WIDTH,
                height=HEIGHT,
            )
            .image
        )


@pytest.fixture()
def fittable():
    """A clip that *is* a render of a known scene, plus a wrong starting point.

    Fitting noise would measure nothing: any loss curve is compatible with an
    optimiser that works and with one that does not. Here the target is a real
    image of a real scene and the initialisation is that scene with grey colours
    and jittered positions, so a loop that discards its updates cannot reach it.
    """
    points, colors, generator = truth_cloud()
    viewmat = torch.eye(4)
    truth = CanonicalParams.from_points(points, colors, scene_scale=0.3)
    image = render(truth, viewmat)
    clip = FixedCameraClip(
        images=image.unsqueeze(0).repeat(FRAMES, 1, 1, 1),
        viewmat=viewmat,
        intrinsics=INTRINSICS,
        times=default_frame_times(FRAMES),
    )
    start = CanonicalParams.from_points(
        points + 0.05 * torch.randn(POINTS, 3, generator=generator),
        torch.full((POINTS, 3), 0.5),
        scene_scale=0.3,
    )
    field = ExplicitTrajectoryField(points[:8].clone(), clip.times)
    return clip, start, field


def short_config(**overrides) -> Train4DConfig:
    settings = {
        "schedule": ScheduleConfig(coarse_iters=60, fine_iters=20),
        "holdout_stride": 0,
        "checkpoint_every": 0,
    }
    settings.update(overrides)
    return Train4DConfig(**settings)


class TestTrainingChangesTheScene:
    """The one property no other test in the suite asserts: the fit fits."""

    def test_the_loss_falls_and_the_render_gets_closer_to_the_target(self, fittable):
        clip, params, field = fittable
        before = float(psnr(render(params, clip.viewmat), clip.images[0]))

        result = train(clip, params, field, config=short_config())
        history = result.metrics.loss_history

        assert len(history) == 80
        # Measured at this size: 0.051 -> 0.013, a factor of four. Half is the
        # bound worth asserting -- it is unreachable by an optimiser whose steps
        # are discarded, and comfortably clear of run-to-run variation.
        assert history[-1] < 0.5 * history[0], history[:1] + history[-1:]
        assert result.metrics.final_loss == pytest.approx(history[-1])
        # Measured: 29.8 dB at the initialisation, 38.6 dB after the fit.
        assert result.metrics.train_psnr > before + 3.0

    def test_a_withheld_frame_is_measured_and_labelled_as_temporal(self, fittable):
        """The hold-out number exists, is finite, and counts the frames it used."""
        clip, params, field = fittable
        result = train(clip, params, field, config=short_config(holdout_stride=2))

        assert result.metrics.holdout_frames == 2
        assert torch.isfinite(torch.tensor(result.metrics.holdout_psnr_temporal))
        assert result.metrics.holdout_psnr_temporal > 0.0


class TestDensification:
    """``refine`` is what turns 200 COLMAP points into a scene. It has to run."""

    def _params(self, opacities: list[float]) -> CanonicalParams:
        n = len(opacities)
        generator = torch.Generator().manual_seed(5)
        means = torch.rand(n, 3, generator=generator)
        quats = torch.zeros(n, 4)
        quats[:, 0] = 1.0
        logits = torch.logit(torch.tensor(opacities))
        return CanonicalParams(
            means=means,
            log_scales=torch.full((n, 3), -3.0),
            quats=quats,
            logit_opacities=logits,
            sh0=torch.zeros(n, 3),
        )

    def test_a_collapsed_gaussian_is_relocated_onto_one_that_survived(self):
        """Below ``min_opacity`` a gaussian is recycled, not left as clutter."""
        params = self._params([1e-6, 1e-6, 0.9, 0.8])
        before = params.params["means"].detach().clone()
        strategy = ReferenceMcmcStrategy(growth_factor=0.0)

        report = strategy.refine(params, cap_max=100)

        assert report["relocated"] == 2
        assert report["grown"] == 0
        # The two dead ones now sit on top of survivors, and the survivors are
        # exactly the two that were above the floor.
        moved = params.params["means"].detach()
        for dead in (0, 1):
            assert not torch.equal(moved[dead], before[dead])
            assert torch.any(torch.all(moved[dead] == before[2:], dim=1))
        assert params.opacities().min() > 1e-3

    def test_growth_reports_a_parent_row_for_every_gaussian_that_now_exists(self):
        """``parent`` is what the rebind is checked against; its shape is the contract."""
        params = self._params([0.5] * 20)
        strategy = ReferenceMcmcStrategy(growth_factor=0.5)

        report = strategy.refine(params, cap_max=1000)
        parent = report["parent"]

        assert report["grown"] == 10
        assert params.n == 30
        assert isinstance(parent, torch.Tensor)
        assert parent.shape == (30,)
        # Every entry indexes into the population as it was *before* the call.
        assert int(parent.min()) >= 0 and int(parent.max()) < 20
        assert torch.equal(parent[:20], torch.arange(20))

    def test_the_cap_is_a_hard_budget_and_not_a_target(self):
        """``cap_max`` is also the .g4d payload budget, so it cannot be overshot."""
        params = self._params([0.5] * 20)
        report = ReferenceMcmcStrategy(growth_factor=1.0).refine(params, cap_max=25)
        assert report["grown"] == 5
        assert params.n == 25

    def test_a_split_pair_composites_to_the_alpha_the_single_gaussian_had(self):
        """Otherwise every refine step brightens the scene and the loss undoes it."""
        params = self._params([0.64] * 8)
        strategy = ReferenceMcmcStrategy(growth_factor=1.0)
        report = strategy.refine(params, cap_max=100)
        grown = int(report["grown"])
        with torch.no_grad():
            opacities = params.opacities().tolist()

        # o' = 1 - sqrt(1 - o); a pair of those composites back to o.
        split = 1.0 - (1.0 - 0.64) ** 0.5
        assert split == pytest.approx(0.4, abs=1e-6)
        assert 1.0 - (1.0 - split) ** 2 == pytest.approx(0.64, abs=1e-9)

        # Sampling is with replacement, so an untouched gaussian keeps 0.64 --
        # but nothing may hold anything else, and every child is a half.
        for value in opacities:
            assert value == pytest.approx(0.64, abs=1e-4) or value == pytest.approx(
                split, abs=1e-4
            )
        for child in opacities[-grown:]:
            assert child == pytest.approx(split, abs=1e-4)

    def test_noise_moves_faint_gaussians_and_leaves_solid_ones_alone(self):
        """The gate is the point: noise explores with the recyclable, not the useful."""
        params = self._params([1e-4, 0.9])
        before = params.params["means"].detach().clone()
        ReferenceMcmcStrategy().add_noise(params, lr=1e-3, scale=1.0)
        delta = (params.params["means"].detach() - before).norm(dim=1)
        assert float(delta[0]) > 10.0 * float(delta[1])

    def test_a_scale_of_zero_is_exactly_no_noise(self):
        """The fine stage drives it to zero, and zero has to mean zero."""
        params = self._params([1e-4, 1e-4])
        before = params.params["means"].detach().clone()
        ReferenceMcmcStrategy().add_noise(params, lr=1e-3, scale=0.0)
        assert torch.equal(params.params["means"].detach(), before)

    def test_a_run_that_densifies_grows_the_population_and_rebinds(self, fittable):
        """The end-to-end path through refine, the surgery, and the rebind decision."""
        clip, params, field = fittable
        schedule = ScheduleConfig(
            coarse_iters=20, fine_iters=10, refine_start=2, refine_every=2, cap_max=400
        )
        result = train(clip, params, field, config=short_config(schedule=schedule))

        assert result.metrics.grown > 0
        assert result.params.n > POINTS
        assert result.metrics.rebinds > 0
        # Whichever way the decision went, it was made on a measurement.
        assert result.metrics.max_rebind_rms >= 0.0
        assert result.metrics.rebinds_rejected <= result.metrics.rebinds
        # The skinning has to cover the population that now exists, or the
        # exporter writes rows belonging to gaussians that are gone.
        assert result.skinning.idx.shape[0] == result.params.n

    def test_a_carried_forward_binding_is_taken_when_the_rebind_would_jump(
        self, fittable
    ):
        """``rebind_max_rms_ratio`` of 0 rejects every rebind; the run still finishes."""
        clip, params, field = fittable
        schedule = ScheduleConfig(
            coarse_iters=12, fine_iters=0, refine_start=2, refine_every=2, cap_max=400
        )
        result = train(
            clip,
            params,
            field,
            config=short_config(schedule=schedule, rebind_max_rms_ratio=0.0),
        )
        assert result.metrics.rebinds > 0
        assert result.skinning.idx.shape[0] == result.params.n

    def test_the_schedule_never_densifies_the_shipped_four_step_run(self):
        """Why this class exists: the defaults put every refine past the horizon."""
        schedule = TrainingSchedule(ScheduleConfig(coarse_iters=2, fine_iters=2))
        assert not any(schedule.should_densify(step) for step in range(4))


class TestCheckpoints:
    """A 15,000-iteration run on a recycled Colab runtime, made survivable."""

    def test_a_checkpoint_restores_the_tensors_and_the_step_it_was_written_at(
        self, fittable, tmp_path: Path
    ):
        clip, params, field = fittable
        path = tmp_path / "scene4d.ckpt"
        save_checkpoint(path, params, field, step=1234)
        saved = params.params["means"].detach().clone()

        params.params["means"].data.add_(1.0)
        assert not torch.equal(params.params["means"].detach(), saved)

        step = load_checkpoint(path, params, field)
        assert step == 1234
        assert torch.equal(params.params["means"].detach(), saved)

    def test_a_checkpoint_from_another_format_version_is_refused(
        self, fittable, tmp_path: Path
    ):
        clip, params, field = fittable
        path = tmp_path / "old.ckpt"
        save_checkpoint(path, params, field, step=1)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload["version"] = SCENE4D_VERSION + 1
        torch.save(payload, path)

        with pytest.raises(PipelineStateError, match="does not match"):
            load_checkpoint(path, params, field)

    def test_a_checkpoint_holding_a_different_field_is_refused(
        self, fittable, tmp_path: Path
    ):
        clip, params, field = fittable
        path = tmp_path / "other.ckpt"
        save_checkpoint(path, params, field, step=1)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload["field_class"] = "KPlanesField"
        torch.save(payload, path)

        with pytest.raises(PipelineStateError, match="KPlanesField"):
            load_checkpoint(path, params, field)

    def test_training_writes_one_periodically_and_one_at_the_end(
        self, fittable, tmp_path: Path
    ):
        clip, params, field = fittable
        path = tmp_path / "run.ckpt"
        config = short_config(
            schedule=ScheduleConfig(coarse_iters=8, fine_iters=0), checkpoint_every=4
        )
        train(clip, params, field, config=config, checkpoint=path)

        assert path.exists()
        assert int(torch.load(path, map_location="cpu", weights_only=True)["step"]) == 8

    def test_resuming_continues_from_the_recorded_step_rather_than_restarting(
        self, fittable, tmp_path: Path
    ):
        clip, params, field = fittable
        path = tmp_path / "run.ckpt"
        first = short_config(
            schedule=ScheduleConfig(coarse_iters=10, fine_iters=0), checkpoint_every=10
        )
        train(clip, params, field, config=first, checkpoint=path)

        second = short_config(schedule=ScheduleConfig(coarse_iters=16, fine_iters=0))
        result = train(clip, params, field, config=second, checkpoint=path, resume=True)

        # Six steps left of a sixteen-step schedule, not sixteen.
        assert len(result.metrics.loss_history) == 6
        assert result.metrics.steps == 16

    def test_a_checkpoint_past_the_end_of_the_schedule_is_refused_by_name(
        self, fittable, tmp_path: Path
    ):
        clip, params, field = fittable
        path = tmp_path / "run.ckpt"
        save_checkpoint(path, params, field, step=500)
        config = short_config(schedule=ScheduleConfig(coarse_iters=4, fine_iters=0))

        with pytest.raises(PipelineStateError, match="nothing left to resume"):
            train(clip, params, field, config=config, checkpoint=path, resume=True)


class TestTheBackendIsChosenRatherThanDefaulted:
    """`--device cuda` has to mean a CUDA kernel, and the summary has to say so.

    ``fit4d.train_4d`` called ``run_training`` with neither a rasterizer nor a
    strategy, so both defaulted -- and the default rasterizer is the CPU
    reference, which materialises an ``(N, H*W, 2)`` tensor. At the shipped
    ``--cap-max 400000`` and 1080p that is 7 TB for one intermediate, per
    iteration. ``GsplatRasterizer`` was defined and called by nothing. The CLI
    even refused ``--device cuda`` without gsplat installed and then never used
    it.
    """

    def test_a_cpu_run_selects_the_reference_pair(self):
        from gausscapture.recon.fit4d import select_backend

        rasterizer, strategy = select_backend("cpu")
        assert rasterizer.name == "reference"
        assert strategy.name == "reference-mcmc"

    def test_a_cuda_run_selects_gsplat_or_refuses(self):
        from gausscapture.errors import DependencyMissingError
        from gausscapture.recon.deform.raster import GsplatRasterizer
        from gausscapture.recon.fit4d import select_backend

        if GsplatRasterizer.available():
            rasterizer, _strategy = select_backend("cuda")
            assert rasterizer.name == "gsplat"
            return
        # No gsplat: refusing is the only honest answer. Falling back to the
        # reference kernel is what used to happen, silently, and it cannot
        # complete a forward pass at any real scene size.
        with pytest.raises(DependencyMissingError, match="gsplat is not installed"):
            select_backend("cuda")


class TestKPlanesRegularisers:
    """Reachable from `train4d --field kplanes`; closed forms, not recorded output."""

    def test_a_constant_plane_has_exactly_zero_total_variation(self):
        planes = [torch.full((1, 4, 6, 6), 0.7)]
        assert float(plane_tv_loss(planes)) == 0.0

    def test_a_unit_ramp_has_the_total_variation_a_unit_ramp_has(self):
        """One step of 1 per texel along each axis: mean square difference is 1."""
        ramp = torch.arange(6, dtype=torch.float32)
        both = ramp.reshape(1, 1, 6, 1) + ramp.reshape(1, 1, 1, 6)
        # dh and dw are each exactly 1 everywhere, so the sum is 2.
        assert float(plane_tv_loss([both])) == pytest.approx(2.0)

    def test_no_planes_is_zero_rather_than_an_error(self):
        assert float(plane_tv_loss([])) == 0.0
        assert float(time_smoothness_loss([])) == 0.0
        assert float(l1_time_planes([])) == 0.0

    def test_a_linear_ramp_in_time_is_free_and_a_kink_is_not(self):
        """Second differences: constant velocity costs nothing, acceleration costs."""
        # Time is the height axis of a space-time plane (PLANE_PAIRS pairs each
        # spatial axis with axis 3), so the difference is along dim -2.
        straight = torch.arange(5, dtype=torch.float32).reshape(1, 1, 5, 1).repeat(1, 1, 1, 3)
        assert float(time_smoothness_loss([straight])) == pytest.approx(0.0, abs=1e-6)

        kinked = straight.clone()
        kinked[0, 0, 2, :] += 1.0
        assert float(time_smoothness_loss([kinked])) > 0.1

    def test_a_plane_shorter_than_three_frames_contributes_nothing(self):
        assert float(time_smoothness_loss([torch.rand(1, 2, 2, 4)])) == 0.0

    def test_the_l1_prior_is_measured_from_the_multiplicative_identity(self):
        """The planes compose by product, so one -- not zero -- is 'no contribution'."""
        assert float(l1_time_planes([torch.ones(1, 2, 3, 4)])) == 0.0
        assert float(l1_time_planes([torch.full((1, 2, 3, 4), 1.25)])) == pytest.approx(0.25)

    def test_the_field_the_regularisers_belong_to_trains(self, fittable):
        """`--field kplanes` reaches `_regularise_field`'s K-Planes branch.

        It could not before: ``field_kind`` was a parameter of ``train_4d`` that
        no flag ever set, so the field, its head and these three losses were
        unreachable from any shipped command.
        """
        clip, params, _ = fittable
        field = KPlanesField(
            params.params["means"].detach()[:8].clone(),
            aabb=scene_aabb(params.params["means"].detach()),
        )
        before = [p.detach().clone() for p in field.parameters()]

        result = train(
            clip,
            params,
            field,
            config=short_config(schedule=ScheduleConfig(coarse_iters=2, fine_iters=8)),
        )

        assert torch.isfinite(torch.tensor(result.metrics.final_loss))
        assert any(
            not torch.equal(was, is_now)
            for was, is_now in zip(before, field.parameters(), strict=True)
        )
