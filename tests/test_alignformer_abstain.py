"""Per-pair abstention from the estimator's own covariance (task 24).

Task 23 left one gap: at low sigma the correct action is to not correct, and we
correct anyway. Neither existing guard has the right shape --
``MIN_MATCH_MASS`` reads evidence *volume* and never asks whether the estimated
correction is meaningful, and a single global shrinkage ``tau`` cannot express
that a dense 12-correspondence pair and a sparse 2-correspondence pair have
wildly different estimator variance.

This module tests the quantity that can: the weighted-least-squares estimator's
**own** covariance, and the Wald statistic of the emitted correction against it.
Four properties are load-bearing and each is pinned here.

1. **The statistic is a pure function of the fit, not of the weight scale.**
   ``weighted_se2_kabsch`` is scale-invariant in its weights and so is this; a
   statistic that moved when the weights were renormalized (``variance.py`` and
   ``robust.py`` both renormalize) would be measuring the plumbing.
2. **It is calibrated.** With no true offset the emitted correction is pure
   estimator noise, and the statistic must then sit on its nominal chi-square
   scale. This is the negative control: an implementation that forgot the
   residual-variance division, or used the nominal point count where the
   effective one belongs, still passes every "big offset gives a big number"
   test and fails this one.
3. **Sparse evidence abstains more readily than dense evidence for the same
   correction**, which is FreeAlign's minimum-node behaviour arrived at from
   the estimator rather than from a node count. If this failed, the whole
   argument for going per-pair would be empty.
4. **A disabled decision changes nothing**, so ``alignformer_irls`` as deployed
   reproduces bit for bit in the very same run that measures the new arms.

The equivalences between the three pre-registered variants are asserted rather
than described: ``both`` at a threshold of 0 -- and at 3, the James-Stein
dimension -- IS ``per_pair``, because the positive-part rule already returns
exactly zero there. Stating that in a test is what keeps the three arms from
silently being two.
"""

import math

import pytest
import torch

from embedding_aware_belt_fusion.alignformer.abstain import (
    ABSTAIN,
    ABSTENTION_MODES,
    BOTH,
    NONE,
    PER_PAIR,
    RESIDUAL_SD_FLOOR_M,
    AbstentionConfig,
    decide,
    decision_factor,
    effective_sample_size,
    residual_variance,
    wald_statistic,
)
from embedding_aware_belt_fusion.alignformer.model import PoseEstimate
from embedding_aware_belt_fusion.alignformer.shrinkage import (
    POSE_DIMENSIONS,
    shrinkage_factor,
)

# chi-square(3) upper quantiles, the only thresholds used below.
CHI2_3_95 = 7.8147
CHI2_3_999 = 16.2662


def _apply(psi, t, q):
    cos, sin = math.cos(psi), math.sin(psi)
    rotation = torch.tensor([[cos, -sin], [sin, cos]], dtype=q.dtype)
    return q @ rotation.T + torch.tensor(t, dtype=q.dtype)


def _scatter(count, generator, extent=60.0):
    return (torch.rand(count, 2, generator=generator) - 0.5) * extent


def _noisy_pair(count, *, sd, generator, psi=0.0, t=(0.0, 0.0)):
    """``(p, q, w)`` for a pair whose only disagreement is detector noise."""
    q = _scatter(count, generator)
    p = _apply(psi, t, q) + torch.randn(count, 2, generator=generator) * sd
    return p.unsqueeze(0), q.unsqueeze(0), torch.ones(1, count)


def _estimate(statistic, psi=0.3, t=(1.0, -2.0)):
    """A batch of identical corrections, one per supplied statistic."""
    values = torch.as_tensor(statistic, dtype=torch.float32)
    size = values.shape[0]
    return PoseEstimate(
        psi=torch.full((size,), psi),
        t=torch.tensor([list(t)]).expand(size, 2).clone(),
        confidence=torch.full((size,), 9.0),
        offset_statistic=values,
    )


# --------------------------------------------------------------------------
# 1. The statistic is a property of the fit, not of the weight scale.
# --------------------------------------------------------------------------


def test_the_statistic_is_invariant_to_the_scale_of_the_weights():
    generator = torch.Generator().manual_seed(3)
    p, q, w = _noisy_pair(12, sd=0.25, generator=generator, psi=0.02, t=(0.4, -0.3))

    plain = wald_statistic(p, q, w, torch.tensor([0.02]), torch.tensor([[0.4, -0.3]]))
    rescaled = wald_statistic(
        p, q, w * 7.5, torch.tensor([0.02]), torch.tensor([[0.4, -0.3]])
    )

    assert torch.allclose(plain, rescaled, rtol=1e-5, atol=1e-6)


def test_the_effective_sample_size_is_the_count_for_equal_weights():
    equal = torch.ones(1, 8)
    assert float(effective_sample_size(equal)[0]) == pytest.approx(8.0)

    # One dominant weight carries almost all the information: Kish's effective
    # count collapses towards one, which is what makes a lopsided soft match
    # abstain rather than pass for eight observations.
    lopsided = torch.tensor([[100.0] + [1e-3] * 7])
    assert float(effective_sample_size(lopsided)[0]) < 1.1


def test_an_all_zero_weight_row_yields_a_finite_statistic():
    generator = torch.Generator().manual_seed(5)
    p, q, _ = _noisy_pair(6, sd=0.25, generator=generator)
    zero = torch.zeros(1, 6)

    value = wald_statistic(p, q, zero, torch.tensor([0.0]), torch.tensor([[0.0, 0.0]]))

    assert torch.isfinite(value).all()
    assert float(value[0]) == pytest.approx(0.0)


def test_an_identity_correction_scores_exactly_zero():
    generator = torch.Generator().manual_seed(9)
    p, q, w = _noisy_pair(10, sd=0.25, generator=generator)

    value = wald_statistic(p, q, w, torch.tensor([0.0]), torch.tensor([[0.0, 0.0]]))

    assert float(value[0]) == 0.0


def test_a_perfect_fit_with_a_real_offset_scores_far_outside_the_noise():
    generator = torch.Generator().manual_seed(13)
    q = _scatter(10, generator).unsqueeze(0)
    psi, t = 0.05, (1.5, -2.0)
    p = _apply(psi, t, q[0]).unsqueeze(0)

    value = wald_statistic(
        p, q, torch.ones(1, 10), torch.tensor([psi]), torch.tensor([list(t)])
    )

    # Zero residual, so the scale floors at RESIDUAL_SD_FLOOR_M and the
    # correction is thousands of floored sigmas of displacement.
    assert float(value[0]) > 1e4


def test_a_lopsided_soft_match_is_charged_for_the_evidence_it_actually_has():
    """Kish's effective count, not the nominal one, sets the degrees of freedom.

    Eight rows of which six carry negligible mass are two observations, not
    eight. Using the nominal count here would divide the same residual sum by
    ``2 * 8 - 3`` instead of the floored ``2 * 2 - 3``, understate the
    estimator's variance by an order of magnitude, and make a lopsided match
    look confident -- which is the failure this whole statistic exists to
    avoid.
    """
    residual = torch.full((1, 8, 2), 0.3)
    lopsided = torch.tensor([[1.0, 1.0] + [1e-6] * 6])

    measured = float(residual_variance(residual, lopsided)[0])

    n_eff = float(effective_sample_size(lopsided)[0])
    assert n_eff == pytest.approx(2.0, abs=1e-3)
    # sum_n w_n |r_n|^2 with the weights normalized to total n_eff, over the
    # floored dof of max(2 * 2 - 3, 1) = 1.
    assert measured == pytest.approx(n_eff * 2 * 0.3 ** 2, rel=1e-3)
    # The nominal count would have been 13 degrees of freedom, i.e. 13x smaller.
    assert measured > 5.0 * (n_eff * 2 * 0.3 ** 2) / (2 * 8 - POSE_DIMENSIONS)


def test_the_residual_variance_floors_rather_than_dividing_by_zero():
    exact = torch.zeros(1, 6, 2)
    value = residual_variance(exact, torch.ones(1, 6))
    assert float(value[0]) == pytest.approx(RESIDUAL_SD_FLOOR_M ** 2)


# --------------------------------------------------------------------------
# 2. THE NEGATIVE CONTROL: with no true offset the statistic is calibrated.
# --------------------------------------------------------------------------


def _null_statistics(count, trials, sd=0.25, seed=101):
    """Wald statistics of the SOLVED correction when the truth is the identity.

    The pose is solved, not supplied, so what is measured is the estimator's
    own noise against its own covariance -- the sigma = 0 situation the whole
    intervention exists for.
    """
    from embedding_aware_belt_fusion.alignformer.procrustes import weighted_se2_kabsch

    generator = torch.Generator().manual_seed(seed)
    values = []
    for _ in range(trials):
        p, q, w = _noisy_pair(count, sd=sd, generator=generator)
        psi, t = weighted_se2_kabsch(p, q, w)
        values.append(float(wald_statistic(p, q, w, psi, t)[0]))
    return torch.tensor(values)


def test_under_the_null_the_statistic_sits_on_its_chi_square_scale():
    values = _null_statistics(count=24, trials=400)

    median = float(values.median())
    tail = float((values > CHI2_3_95).to(torch.float32).mean())

    # chi-square(3) has median 2.366 and 5% mass above 7.815. The plug-in
    # variance makes this 3 * F(3, 2n - 3), slightly heavier in the tail.
    assert 1.6 < median < 3.4, median
    assert 0.02 < tail < 0.12, tail


def test_a_real_offset_moves_the_same_statistic_off_that_scale():
    """The control's companion: a calibrated statistic must still FIRE.

    Without this, an implementation that returned a constant near 2.4 would
    pass the calibration test and abstain on everything.
    """
    from embedding_aware_belt_fusion.alignformer.procrustes import weighted_se2_kabsch

    generator = torch.Generator().manual_seed(29)
    fired = 0
    for _ in range(50):
        p, q, w = _noisy_pair(24, sd=0.25, generator=generator, t=(0.8, -0.6))
        psi, t = weighted_se2_kabsch(p, q, w)
        fired += int(float(wald_statistic(p, q, w, psi, t)[0]) > CHI2_3_95)

    assert fired == 50, fired


# --------------------------------------------------------------------------
# 3. Sparse evidence abstains more readily than dense, for the same correction.
# --------------------------------------------------------------------------


def test_sparse_evidence_scores_lower_than_dense_for_the_same_correction():
    generator = torch.Generator().manual_seed(17)
    psi, t = torch.tensor([0.004]), torch.tensor([[0.20, -0.15]])

    dense_p, dense_q, dense_w = _noisy_pair(16, sd=0.25, generator=generator)
    sparse_p, sparse_q, sparse_w = _noisy_pair(4, sd=0.25, generator=generator)

    dense = float(wald_statistic(dense_p, dense_q, dense_w, psi, t)[0])
    sparse = float(wald_statistic(sparse_p, sparse_q, sparse_w, psi, t)[0])

    assert dense > 2.0 * sparse, (dense, sparse)


def test_a_noisier_pair_scores_lower_than_a_cleaner_one():
    generator = torch.Generator().manual_seed(23)
    psi, t = torch.tensor([0.004]), torch.tensor([[0.20, -0.15]])

    clean_p, clean_q, w = _noisy_pair(16, sd=0.10, generator=generator)
    noisy_p, noisy_q, _ = _noisy_pair(16, sd=0.60, generator=generator)

    clean = float(wald_statistic(clean_p, clean_q, w, psi, t)[0])
    noisy = float(wald_statistic(noisy_p, noisy_q, w, psi, t)[0])

    assert clean > noisy, (clean, noisy)


# --------------------------------------------------------------------------
# 4. The decision rule, and the equivalences between its three variants.
# --------------------------------------------------------------------------


def test_a_disabled_decision_returns_the_estimate_untouched():
    estimate = _estimate([0.01])
    decided = decide(estimate, AbstentionConfig())

    assert decided is estimate


@pytest.mark.parametrize("mode", ABSTENTION_MODES)
def test_every_mode_is_constructible_and_reports_its_own_name(mode):
    threshold = 0.0 if mode in (NONE, PER_PAIR) else CHI2_3_95
    config = AbstentionConfig(mode=mode, threshold=threshold)
    assert config.enabled == (mode != NONE)
    assert config.to_dict()["mode"] == mode


def test_hard_abstention_zeroes_below_the_threshold_and_is_exact_above_it():
    estimate = _estimate([1.0, 400.0])
    config = AbstentionConfig(mode=ABSTAIN, threshold=CHI2_3_95)

    decided = decide(estimate, config)

    assert float(decided.psi[0]) == 0.0
    assert float(decided.t[0].abs().sum()) == 0.0
    assert float(decided.psi[1]) == float(estimate.psi[1])
    assert torch.equal(decided.t[1], estimate.t[1])


def test_the_threshold_is_inclusive_so_an_exact_hit_abstains():
    estimate = _estimate([CHI2_3_95])
    decided = decide(estimate, AbstentionConfig(mode=ABSTAIN, threshold=CHI2_3_95))
    assert float(decided.psi[0]) == 0.0


def test_the_per_pair_factor_is_the_james_stein_rule_on_this_statistic():
    statistic = torch.tensor([0.5, 3.0, 12.0, 300.0])

    factor = decision_factor(statistic, AbstentionConfig(mode=PER_PAIR))

    assert torch.equal(factor, shrinkage_factor(statistic, POSE_DIMENSIONS))


def test_both_at_the_james_stein_dimension_is_exactly_the_per_pair_rule():
    statistic = torch.tensor([0.5, 2.9, 3.0, 12.0, 300.0])

    per_pair = decision_factor(statistic, AbstentionConfig(mode=PER_PAIR))
    at_zero = decision_factor(statistic, AbstentionConfig(mode=BOTH, threshold=0.0))
    at_p = decision_factor(
        statistic, AbstentionConfig(mode=BOTH, threshold=float(POSE_DIMENSIONS))
    )

    assert torch.equal(per_pair, at_zero)
    assert torch.equal(per_pair, at_p)


def test_both_above_the_dimension_abstains_where_per_pair_would_have_shrunk():
    statistic = torch.tensor([5.0])

    per_pair = decision_factor(statistic, AbstentionConfig(mode=PER_PAIR))
    both = decision_factor(
        statistic, AbstentionConfig(mode=BOTH, threshold=CHI2_3_95)
    )

    assert float(per_pair[0]) > 0.0
    assert float(both[0]) == 0.0


def test_per_pair_refuses_a_threshold_because_it_has_no_use_for_one():
    with pytest.raises(ValueError, match="threshold"):
        AbstentionConfig(mode=PER_PAIR, threshold=5.0)


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="mode"):
        AbstentionConfig(mode="gate")


def test_a_negative_threshold_is_refused():
    with pytest.raises(ValueError, match="threshold"):
        AbstentionConfig(mode=ABSTAIN, threshold=-1.0)


def test_deciding_without_a_statistic_is_an_error_rather_than_a_silent_pass():
    estimate = PoseEstimate(
        psi=torch.tensor([0.3]),
        t=torch.tensor([[1.0, -2.0]]),
        confidence=torch.tensor([9.0]),
    )
    with pytest.raises(ValueError, match="statistic"):
        decide(estimate, AbstentionConfig(mode=ABSTAIN, threshold=CHI2_3_95))


def test_an_abstained_pair_is_counted_as_a_fallback_so_coverage_is_measurable():
    from embedding_aware_belt_fusion.alignformer.stage2 import is_fallback

    estimate = _estimate([1.0, 400.0])
    decided = decide(estimate, AbstentionConfig(mode=ABSTAIN, threshold=CHI2_3_95))

    assert is_fallback(decided).tolist() == [True, False]


# --------------------------------------------------------------------------
# 5. Specs, names and the sweep wiring.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec, mode, threshold",
    (
        ("abstain:7.8147", ABSTAIN, CHI2_3_95),
        ("both:16.2662", BOTH, CHI2_3_999),
        ("per_pair", PER_PAIR, 0.0),
    ),
)
def test_a_spec_parses_to_the_configuration_it_names(spec, mode, threshold):
    config = AbstentionConfig.parse(spec)
    assert config.mode == mode
    assert config.threshold == pytest.approx(threshold)


def test_a_spec_with_a_nonsense_threshold_is_refused():
    with pytest.raises(ValueError):
        AbstentionConfig.parse("abstain:wide")


def test_arm_names_are_distinct_and_carry_their_threshold():
    names = {
        AbstentionConfig.parse(spec).name
        for spec in ("abstain:3", "abstain:7.8147", "both:7.8147", "per_pair")
    }
    assert names == {
        "alignformer_abstain_3",
        "alignformer_abstain_7.8147",
        "alignformer_both_7.8147",
        "alignformer_per_pair",
    }


def test_the_sweep_adds_one_arm_per_configuration_after_the_irls_arm():
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import (
        ALIGNFORMER,
        ALIGNFORMER_IRLS,
        FREEALIGN,
        sweep_estimators,
    )
    from embedding_aware_belt_fusion.alignformer.robust import HUBER, RobustSolveConfig

    arms = (AbstentionConfig.parse("abstain:7.8147"), AbstentionConfig.parse("per_pair"))
    names = sweep_estimators(
        [], True, RobustSolveConfig(mode=HUBER, iterations=2), abstention=arms
    )

    assert names[0] == ALIGNFORMER
    assert names[1] == ALIGNFORMER_IRLS
    assert names[2:4] == [arm.name for arm in arms]
    assert names[-1] == FREEALIGN


def test_abstention_arms_need_the_irls_arm_they_are_built_on():
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import sweep_estimators
    from embedding_aware_belt_fusion.alignformer.robust import RobustSolveConfig

    with pytest.raises(ValueError, match="robust"):
        sweep_estimators(
            [], False, RobustSolveConfig(),
            abstention=(AbstentionConfig.parse("per_pair"),),
        )


def test_the_global_tau_never_reaches_an_abstention_arm():
    """They REPLACE the global-tau shrinkage; stacking both would confound H3."""
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import (
        FREEALIGN,
        unshrunk_conditions,
    )

    arms = (AbstentionConfig.parse("abstain:7.8147"), AbstentionConfig.parse("per_pair"))
    unshrunk = unshrunk_conditions(arms)

    assert FREEALIGN in unshrunk
    for arm in arms:
        assert arm.name in unshrunk


def test_draw_aligned_leaves_an_unshrunk_arm_exactly_where_it_found_it():
    from embedding_aware_belt_fusion.alignformer.boxes import AgentDetections
    from embedding_aware_belt_fusion.alignformer.draws import draw_aligned
    from embedding_aware_belt_fusion.alignformer.shrinkage import ShrinkageCalibration

    detections = AgentDetections(
        boxes=torch.zeros(1, 7),
        scores=torch.ones(1),
        corners=torch.zeros(1, 8, 3),
        gt_ids=[None],
        features=torch.zeros(1, 2, 2),
    )
    estimate = _estimate([50.0])
    calibration = ShrinkageCalibration(
        tau_translation_m=0.15, tau_yaw_rad=0.004, pairs=10, split="v", sigma_m=0.0
    )

    aligned = draw_aligned(
        detections, {"arm": estimate}, calibration, unshrunk=("arm",)
    )

    assert torch.equal(aligned["arm"].estimate.psi, estimate.psi)
    assert torch.equal(aligned["arm"].estimate.t, estimate.t)


# --------------------------------------------------------------------------
# 6. The solve carries the statistic without moving the pose it reports.
# --------------------------------------------------------------------------


def _pair_batch(count=6):
    from embedding_aware_belt_fusion.alignformer.boxes import BOX_YAW

    generator = torch.Generator().manual_seed(11)
    centres = _scatter(count, generator, extent=40.0)
    yaws = torch.rand(count, generator=generator) * 2 * math.pi

    ego = torch.zeros(1, count, 7)
    ego[0, :, :2] = centres
    ego[0, :, BOX_YAW] = yaws
    cav = ego.clone()
    cav[0, :, :2] = _apply(0.1, (1.0, -0.5), centres)
    cav[0, :, BOX_YAW] = yaws + 0.1

    batch = {
        "ego_boxes": ego,
        "cav_boxes": cav,
        "ego_scores": torch.full((1, count), 0.6),
        "cav_scores": torch.full((1, count), 0.6),
        "ego_mask": torch.ones(1, count, dtype=torch.bool),
        "cav_mask": torch.ones(1, count, dtype=torch.bool),
    }
    return batch, torch.eye(count).unsqueeze(0)


@pytest.mark.parametrize("robust_mode", (None, "huber"))
def test_asking_for_the_statistic_does_not_move_the_pose_it_is_computed_from(
    robust_mode,
):
    from embedding_aware_belt_fusion.alignformer.model import (
        reduce_correspondence,
        solve_pose,
    )
    from embedding_aware_belt_fusion.alignformer.robust import RobustSolveConfig
    from embedding_aware_belt_fusion.alignformer.variance import UNWEIGHTED

    batch, weights = _pair_batch()
    correspondence = reduce_correspondence(weights, batch, UNWEIGHTED)
    robust = (
        None if robust_mode is None
        else RobustSolveConfig(mode=robust_mode, iterations=2, min_evidence=3.0)
    )

    without = solve_pose(correspondence, batch, 2.0, UNWEIGHTED, robust)
    with_statistic = solve_pose(
        correspondence, batch, 2.0, UNWEIGHTED, robust, statistic=True
    )

    assert torch.equal(without.psi, with_statistic.psi)
    assert torch.equal(without.t, with_statistic.t)
    assert without.offset_statistic is None
    assert with_statistic.offset_statistic is not None
    assert torch.isfinite(with_statistic.offset_statistic).all()


def test_an_empty_object_set_carries_a_zero_statistic_rather_than_none():
    from embedding_aware_belt_fusion.alignformer.model import AlignFormerB

    model = AlignFormerB(embed_dim=8, model_dim=16, layers=1, heads=2)
    batch = {
        "ego_boxes": torch.zeros(1, 0, 7),
        "cav_boxes": torch.zeros(1, 0, 7),
        "ego_scores": torch.zeros(1, 0),
        "cav_scores": torch.zeros(1, 0),
        "ego_embeddings": torch.zeros(1, 0, 8),
        "cav_embeddings": torch.zeros(1, 0, 8),
        "ego_mask": torch.zeros(1, 0, dtype=torch.bool),
        "cav_mask": torch.zeros(1, 0, dtype=torch.bool),
    }

    estimate = model(batch, statistic=True)

    assert estimate.offset_statistic is not None
    assert float(estimate.offset_statistic[0]) == 0.0
    # And it therefore abstains, which is the right answer for no evidence.
    decided = decide(estimate, AbstentionConfig(mode=ABSTAIN, threshold=CHI2_3_95))
    assert float(decided.psi[0]) == 0.0


def test_the_sweep_asks_the_solve_for_the_statistic_its_arms_need():
    """End to end through ``_alignformer_estimates``, because that is where the
    request for the statistic is made and where forgetting it would be silent
    in every unit test above."""
    from embedding_aware_belt_fusion.alignformer.embedding import ObjectEmbedding
    from embedding_aware_belt_fusion.alignformer.model import AlignFormerB
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import (
        ALIGNFORMER,
        ALIGNFORMER_IRLS,
        _alignformer_estimates,
    )
    from embedding_aware_belt_fusion.alignformer.robust import (
        HUBER,
        RobustSolveConfig,
    )

    torch.manual_seed(0)
    channels, output_size, dim, count = 3, 2, 8, 6
    batch, _ = _pair_batch(count)
    batch["ego_roi"] = torch.rand(1, count, channels, output_size, output_size)
    batch["cav_roi"] = torch.rand(1, count, channels, output_size, output_size)

    modules = {
        "embedding": ObjectEmbedding(channels, output_size, dim),
        "pose": AlignFormerB(embed_dim=dim, model_dim=16, layers=1, heads=2),
    }
    robust = RobustSolveConfig(mode=HUBER, iterations=2, min_evidence=0.0)
    arms = (
        AbstentionConfig.parse("abstain:7.8147"),
        AbstentionConfig.parse("per_pair"),
    )

    estimates = _alignformer_estimates(modules, batch, False, robust, arms)

    assert set(estimates) == {ALIGNFORMER, ALIGNFORMER_IRLS} | {a.name for a in arms}
    irls = estimates[ALIGNFORMER_IRLS]
    assert irls.offset_statistic is not None
    # Each arm is that ONE estimate with a decision applied, never a re-solve.
    for arm in arms:
        factor = decision_factor(irls.offset_statistic, arm)
        assert torch.equal(estimates[arm.name].psi, irls.psi * factor)
        assert torch.equal(estimates[arm.name].t, irls.t * factor.unsqueeze(-1))


def test_the_deployed_arms_are_untouched_by_the_presence_of_abstention_arms():
    from embedding_aware_belt_fusion.alignformer.embedding import ObjectEmbedding
    from embedding_aware_belt_fusion.alignformer.model import AlignFormerB
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import (
        ALIGNFORMER,
        ALIGNFORMER_IRLS,
        _alignformer_estimates,
    )
    from embedding_aware_belt_fusion.alignformer.robust import (
        HUBER,
        RobustSolveConfig,
    )

    torch.manual_seed(0)
    channels, output_size, dim, count = 3, 2, 8, 6
    batch, _ = _pair_batch(count)
    batch["ego_roi"] = torch.rand(1, count, channels, output_size, output_size)
    batch["cav_roi"] = torch.rand(1, count, channels, output_size, output_size)
    modules = {
        "embedding": ObjectEmbedding(channels, output_size, dim),
        "pose": AlignFormerB(embed_dim=dim, model_dim=16, layers=1, heads=2),
    }
    robust = RobustSolveConfig(mode=HUBER, iterations=2, min_evidence=0.0)

    without = _alignformer_estimates(modules, batch, False, robust)
    with_arms = _alignformer_estimates(
        modules, batch, False, robust, (AbstentionConfig.parse("per_pair"),)
    )

    for name in (ALIGNFORMER, ALIGNFORMER_IRLS):
        assert torch.equal(without[name].psi, with_arms[name].psi)
        assert torch.equal(without[name].t, with_arms[name].t)
