"""Per-pair abstention from the estimator's own covariance (task 24).

Task 23 left one gap: at low sigma the correct action is to not correct, and we
correct anyway. Neither existing guard has the right shape --
``MIN_MATCH_MASS`` reads evidence *volume* and never asks whether the estimated
correction is meaningful, and a single global shrinkage ``tau`` cannot express
that a dense twelve-correspondence pair and a sparse two-correspondence pair
have wildly different estimator variance.

This module tests the quantity that can. Four properties are load-bearing and
each is pinned here.

1. **The statistic is calibrated ON THE GEOMETRY THE PIPELINE ACTUALLY
   PRODUCES.** That geometry is heading-augmented: object ``m``'s two points
   share their entire centre error (correlation about 0.84 at this project's
   fitted constants), so a covariance that treats ``2M`` points as ``2M``
   independent observations understates itself and the test over-fires -- worst
   where the evidence is thinnest, which *inverts* the behaviour the module
   exists to produce. ``_null_pair`` therefore builds pairs through
   ``procrustes.augment_with_heading`` with the r140 constants, and the control
   sweeps the matched-object count. An earlier draft of this module tested on
   centre-only iid points, where the naive covariance IS calibrated, and was
   blind to the defect by construction.
2. **The level means the same thing at two matched objects as at twenty.**
   Asserted per object count, not pooled, because a pooled false-fire rate can
   sit on its nominal value while every individual count misses it.
3. **The statistic is a pure function of the fit, not of the weight scale.**
   ``weighted_se2_kabsch`` is scale-invariant in its weights and so is this; a
   statistic that moved when ``variance.py`` or ``robust.py`` renormalized
   would be measuring the plumbing.
4. **A disabled decision changes nothing**, so ``alignformer_irls`` as deployed
   reproduces bit for bit in the very same run that measures the new arms.

The equivalence between the three pre-registered variants is asserted rather
than described: ``both`` at level 1.0 -- a threshold of zero -- IS ``per_pair``.
Stating that in a test is what keeps the three arms from silently being two.
"""

import math

import pytest
import torch

from embedding_aware_belt_fusion.alignformer.abstain import (
    ABSTAIN,
    ABSTENTION_MODES,
    BOTH,
    DIRECTIONAL,
    NONE,
    PER_PAIR,
    RESIDUAL_SD_FLOOR_M,
    AbstentionConfig,
    abstention_threshold,
    decide,
    decision_factor,
    effective_sample_size,
    residual_scale,
    wald_statistic,
)
from embedding_aware_belt_fusion.alignformer.model import PoseEstimate
from embedding_aware_belt_fusion.alignformer.procrustes import (
    augment_with_heading,
    weighted_se2_kabsch,
)
from embedding_aware_belt_fusion.alignformer.shrinkage import (
    POSE_DIMENSIONS,
    shrinkage_factor,
)

# configs/alignformer_r140.yaml, model.correspondence_variance: the fitted
# per-detection disagreement at the reference confidence, and the deployed
# heading offset. The calibration control has to run at these or it is testing
# a geometry this project does not deploy.
SIGMA_TRANSLATION_M = 0.2516
SIGMA_YAW_RAD = math.radians(4.6392)
HEADING_LAMBDA = 2.0
# Two detections per correspondence, so the disagreement variance is twice the
# per-detection one. `variance.correspondence_variances`' own arithmetic.
VARIANCE_CENTRE = 2.0 * SIGMA_TRANSLATION_M ** 2
VARIANCE_HEADING = VARIANCE_CENTRE + HEADING_LAMBDA ** 2 * 2.0 * SIGMA_YAW_RAD ** 2


def _apply(psi, t, q):
    cos, sin = math.cos(psi), math.sin(psi)
    rotation = torch.tensor([[cos, -sin], [sin, cos]], dtype=q.dtype)
    return q @ rotation.T + torch.tensor(t, dtype=q.dtype)


def _scatter(count, generator, extent=60.0):
    return (torch.rand(count, 2, generator=generator) - 0.5) * extent


def _variances(count, batch=1):
    return (
        torch.full((batch, count), VARIANCE_CENTRE),
        torch.full((batch, count), VARIANCE_HEADING),
    )


def _null_pair(objects, generator, *, psi=0.0, t=(0.0, 0.0)):
    """``(p, q, w)`` for one pair in the DEPLOYED heading-augmented geometry.

    Both agents detect the same objects and disagree only by the fitted
    per-detection centre and heading noise; the true relative correction is
    ``(psi, t)``, which the null uses as the identity. Points come out of
    ``augment_with_heading`` so an object's two rows share its centre error
    exactly as they do in the sweep.
    """
    centres = _scatter(objects, generator)
    yaws = (torch.rand(objects, generator=generator) - 0.5) * 2 * math.pi

    axis = SIGMA_TRANSLATION_M / math.sqrt(2.0)
    ego_centres = centres + torch.randn(objects, 2, generator=generator) * axis
    ego_yaws = yaws + torch.randn(objects, generator=generator) * SIGMA_YAW_RAD
    cav_centres = centres + torch.randn(objects, 2, generator=generator) * axis
    cav_yaws = yaws + torch.randn(objects, generator=generator) * SIGMA_YAW_RAD

    # The CAV's boxes arrive already projected by the (possibly wrong) pose;
    # the correction the solve looks for takes them back onto the ego's.
    inverse = _apply(-psi, (0.0, 0.0), cav_centres - torch.tensor(t, dtype=torch.float))

    p = augment_with_heading(
        ego_centres.unsqueeze(0), ego_yaws.unsqueeze(0), HEADING_LAMBDA
    )
    q = augment_with_heading(
        inverse.unsqueeze(0), (cav_yaws - psi).unsqueeze(0), HEADING_LAMBDA
    )
    return p, q, torch.ones(1, 2 * objects)


def _statistic(p, q, w, psi, t):
    centre, heading = _variances(p.shape[1] // 2)
    return wald_statistic(
        p, q, w, psi, t,
        variance_centre=centre,
        variance_heading=heading,
        heading_lambda=HEADING_LAMBDA,
    )


def _solved_statistic(p, q, w):
    psi, t = weighted_se2_kabsch(p, q, w)
    return _statistic(p, q, w, psi, t)


def _estimate(statistic, dof=1000.0, psi=0.3, t=(1.0, -2.0)):
    """A batch of identical corrections, one per supplied statistic."""
    values = torch.as_tensor(statistic, dtype=torch.float32)
    size = values.shape[0]
    return PoseEstimate(
        psi=torch.full((size,), psi),
        t=torch.tensor([list(t)]).expand(size, 2).clone(),
        confidence=torch.full((size,), 9.0),
        offset_statistic=values,
        offset_dof=torch.full((size,), float(dof)),
    )


# --------------------------------------------------------------------------
# 1. THE NEGATIVE CONTROL: calibrated on the geometry the pipeline produces.
# --------------------------------------------------------------------------


def _false_fire_rate(objects, level, trials, seed):
    """How often the rule answers when the truth is exactly the identity."""
    config = AbstentionConfig(mode=ABSTAIN, level=level)
    generator = torch.Generator().manual_seed(seed)
    fired = 0
    for _ in range(trials):
        p, q, w = _null_pair(objects, generator)
        statistic, dof = _solved_statistic(p, q, w)
        fired += int(float(decision_factor(statistic, dof, config)[0]) > 0.0)
    return fired / trials


@pytest.mark.parametrize("objects", (2, 4, 12, 24))
def test_the_level_holds_at_every_matched_object_count(objects):
    """The control the earlier draft did not have, and the defect it hid.

    A covariance that ignored the shared centre error fired on 64% of
    two-object pairs and 27% of twenty-four-object pairs at a nominal 5%. With
    the sandwich and the computed residual divisor the measured rate is 0.075
    at two objects falling to 0.045 at twenty-four (2000 trials each). The bar
    here is deliberately per-count: a pooled rate can sit on its nominal value
    while every individual count misses it, and sigma = 0 on the sparse slice
    is the one cell this whole intervention exists to fix.
    """
    measured = _false_fire_rate(objects, level=0.05, trials=400, seed=1000 + objects)
    assert 0.02 < measured < 0.13, (objects, measured)


def test_a_looser_level_answers_more_often_than_a_tighter_one():
    loose = _false_fire_rate(12, level=0.5, trials=300, seed=77)
    tight = _false_fire_rate(12, level=0.01, trials=300, seed=77)
    assert loose > 0.25
    assert tight < 0.06
    assert loose > tight


def test_a_real_offset_still_fires():
    """The control's companion: a calibrated statistic must not be inert.

    Without this, an implementation that returned a constant below every
    threshold would pass the calibration test and abstain on everything.
    """
    config = AbstentionConfig(mode=ABSTAIN, level=0.05)
    generator = torch.Generator().manual_seed(29)
    fired = 0
    for _ in range(60):
        p, q, w = _null_pair(12, generator, t=(1.2, -0.9))
        statistic, dof = _solved_statistic(p, q, w)
        fired += int(float(decision_factor(statistic, dof, config)[0]) > 0.0)
    assert fired == 60, fired


def test_a_sparse_pair_needs_a_larger_offset_than_a_dense_one_to_fire():
    """The mechanism the whole intervention is justified by, measured on the
    SOLVED correction rather than on a supplied one -- the sweep never supplies
    one, and a sparse pair solves a larger spurious correction."""
    config = AbstentionConfig(mode=ABSTAIN, level=0.05)

    def rate(objects, offset):
        generator = torch.Generator().manual_seed(4242)
        fired = 0
        for _ in range(300):
            p, q, w = _null_pair(objects, generator, t=(offset, 0.0))
            statistic, dof = _solved_statistic(p, q, w)
            fired += int(float(decision_factor(statistic, dof, config)[0]) > 0.0)
        return fired / 300

    dense = rate(16, 0.20)
    sparse = rate(2, 0.20)
    assert dense > sparse, (dense, sparse)


# --------------------------------------------------------------------------
# 2. The statistic is a property of the fit, not of the weight scale.
# --------------------------------------------------------------------------


def test_the_statistic_is_invariant_to_the_scale_of_the_weights():
    generator = torch.Generator().manual_seed(3)
    p, q, w = _null_pair(12, generator, t=(0.4, -0.3))
    psi, t = torch.tensor([0.02]), torch.tensor([[0.4, -0.3]])

    plain, plain_dof = _statistic(p, q, w, psi, t)
    rescaled, rescaled_dof = _statistic(p, q, w * 7.5, psi, t)

    assert torch.allclose(plain, rescaled, rtol=1e-4, atol=1e-6)
    assert torch.allclose(plain_dof, rescaled_dof, rtol=1e-5, atol=1e-6)


def test_the_effective_sample_size_is_the_count_for_equal_weights():
    assert float(effective_sample_size(torch.ones(1, 8))[0]) == pytest.approx(8.0)

    # One dominant weight carries almost all the information: Kish's effective
    # count collapses towards one, which is what makes a lopsided soft match
    # abstain rather than pass for eight observations.
    lopsided = torch.tensor([[100.0] + [1e-3] * 7])
    assert float(effective_sample_size(lopsided)[0]) < 1.1


def test_the_degrees_of_freedom_count_objects_and_not_points():
    """Three independent noise dimensions per object -- two centre, one heading
    -- however many augmented points it contributes."""
    generator = torch.Generator().manual_seed(5)
    p, q, w = _null_pair(8, generator)

    _, dof = _solved_statistic(p, q, w)

    assert float(dof[0]) == pytest.approx(POSE_DIMENSIONS * 8 - POSE_DIMENSIONS)
    # Not the point count: that would be 3 * 16 - 3 = 45.
    assert float(dof[0]) < 30.0


def test_a_lopsided_soft_match_is_charged_for_the_evidence_it_actually_has():
    generator = torch.Generator().manual_seed(6)
    p, q, _ = _null_pair(8, generator)
    lopsided = torch.tensor([[1.0, 1.0] + [1e-6] * 6] * 2).reshape(1, 16)

    _, dof = _solved_statistic(p, q, lopsided)

    # Two objects' worth of evidence, not eight.
    assert float(dof[0]) == pytest.approx(POSE_DIMENSIONS * 2 - POSE_DIMENSIONS, abs=0.1)


def test_an_all_zero_weight_row_yields_a_finite_statistic():
    generator = torch.Generator().manual_seed(5)
    p, q, _ = _null_pair(6, generator)
    zero = torch.zeros(1, 12)

    value, dof = _statistic(p, q, zero, torch.tensor([0.0]), torch.tensor([[0.0, 0.0]]))

    assert torch.isfinite(value).all()
    assert float(value[0]) == pytest.approx(0.0)
    assert float(dof[0]) > 0.0


def test_an_identity_correction_scores_exactly_zero():
    generator = torch.Generator().manual_seed(9)
    p, q, w = _null_pair(10, generator)

    value, _ = _statistic(p, q, w, torch.tensor([0.0]), torch.tensor([[0.0, 0.0]]))

    assert float(value[0]) == 0.0


def test_a_perfect_fit_with_a_real_offset_scores_far_outside_the_noise():
    generator = torch.Generator().manual_seed(13)
    centres = _scatter(10, generator)
    yaws = torch.rand(10, generator=generator) * 2 * math.pi
    q = augment_with_heading(centres.unsqueeze(0), yaws.unsqueeze(0), HEADING_LAMBDA)
    psi, t = 0.05, (1.5, -2.0)
    p = augment_with_heading(
        _apply(psi, t, centres).unsqueeze(0), (yaws + psi).unsqueeze(0), HEADING_LAMBDA
    )

    value, _ = _statistic(
        p, q, torch.ones(1, 20), torch.tensor([psi]), torch.tensor([list(t)])
    )

    assert float(value[0]) > 1e4


def test_the_residual_scale_floors_rather_than_dividing_by_zero():
    exact = torch.zeros(1, 6, 2)
    variance = torch.full((1, 6), VARIANCE_CENTRE)
    value = float(
        residual_scale(exact, torch.ones(1, 6), variance, torch.tensor([6.0]))[0]
    )
    # The floor is on the IMPLIED per-axis residual sd, which is where it keeps
    # its meaning in metres.
    assert value == pytest.approx(2.0 * RESIDUAL_SD_FLOOR_M ** 2 / VARIANCE_CENTRE)


def _null_residual_scale(objects, trials, seed):
    """``tau^2`` on pairs whose noise is exactly the model's, and its divisor."""
    from embedding_aware_belt_fusion.alignformer.abstain import (
        _normal_matrix,
        expected_residual_sum,
        residual_scale,
        sandwich_middle,
    )

    generator = torch.Generator().manual_seed(seed)
    centre, heading = _variances(objects)
    scales, sums, divisor = [], [], None
    for _ in range(trials):
        p, q, w = _null_pair(objects, generator)
        psi, t = weighted_se2_kabsch(p, q, w)
        cos, sin = math.cos(float(psi[0])), math.sin(float(psi[0]))
        rotated = (q[0] @ torch.tensor([[cos, -sin], [sin, cos]]).T).unsqueeze(0)
        residual = rotated + t.unsqueeze(1) - p
        variance = torch.cat([centre, heading], dim=1)

        curvature = _normal_matrix(rotated, w)
        middle = sandwich_middle(rotated, w, centre, heading, HEADING_LAMBDA)
        divisor = expected_residual_sum(
            rotated, w, centre, heading, HEADING_LAMBDA, curvature, middle
        )
        sums.append(
            float(((residual * residual).sum(-1) / variance).sum())
        )
        scales.append(float(residual_scale(residual, w, variance, divisor)[0]))
    return sum(scales) / len(scales), sum(sums) / len(sums), float(divisor[0])


@pytest.mark.parametrize("objects", (2, 4, 12, 24))
def test_the_residual_scale_is_unbiased_at_every_object_count(objects):
    """``tau^2`` is centred on 1 when the noise IS the model, at every count.

    The derivation this pins is ``expected_residual_sum``: the divisor is the
    model's own expectation for the numerator, computed from the fit. Counting
    points instead -- ``2 n_eff - 3`` -- is wrong by a factor of two at two
    matched objects, which halves tau^2, doubles the statistic and over-fires
    exactly where the evidence is thinnest. The companion assertion compares the
    computed divisor against the measured expectation directly, so a divisor
    that happened to be wrong in a way tau^2 absorbed could not pass.
    """
    scale, measured_sum, divisor = _null_residual_scale(objects, 400, 300 + objects)

    assert 0.8 < scale < 1.25, (objects, scale)
    assert divisor == pytest.approx(measured_sum, rel=0.12), (objects, divisor)
    # And the naive point count is NOT the answer at the thin end.
    if objects == 2:
        assert divisor < 0.75 * (2 * 2 * objects - POSE_DIMENSIONS)


# --------------------------------------------------------------------------
# 3. The threshold, and the decision rule.
# --------------------------------------------------------------------------


def test_the_threshold_falls_to_the_chi_square_quantile_at_large_dof():
    from scipy.stats import chi2

    config = AbstentionConfig(mode=ABSTAIN, level=0.05)
    large = abstention_threshold(torch.tensor([3500.0]), config)
    assert float(large[0]) == pytest.approx(chi2.ppf(0.95, 3), rel=0.02)


def test_the_threshold_is_far_larger_where_the_evidence_is_thin():
    config = AbstentionConfig(mode=ABSTAIN, level=0.05)
    thresholds = abstention_threshold(torch.tensor([3.0, 9.0, 69.0]), config)
    assert thresholds[0] > 3.0 * thresholds[2]
    assert thresholds[0] > thresholds[1] > thresholds[2]


def test_a_disabled_decision_returns_the_estimate_untouched():
    estimate = _estimate([0.01])
    assert decide(estimate, AbstentionConfig()) is estimate


@pytest.mark.parametrize("mode", ABSTENTION_MODES)
def test_every_mode_is_constructible_and_reports_its_own_name(mode):
    # per_pair and directional refuse a level rather than ignoring one.
    level = 0.0 if mode in (NONE, PER_PAIR, DIRECTIONAL) else 0.05
    config = AbstentionConfig(mode=mode, level=level)
    assert config.enabled == (mode != NONE)
    assert config.to_dict()["mode"] == mode


def test_hard_abstention_zeroes_below_the_threshold_and_is_exact_above_it():
    estimate = _estimate([1.0, 400.0])
    decided = decide(estimate, AbstentionConfig(mode=ABSTAIN, level=0.05))

    assert float(decided.psi[0]) == 0.0
    assert float(decided.t[0].abs().sum()) == 0.0
    assert float(decided.psi[1]) == float(estimate.psi[1])
    assert torch.equal(decided.t[1], estimate.t[1])


def test_the_threshold_is_inclusive_so_an_exact_hit_abstains():
    config = AbstentionConfig(mode=ABSTAIN, level=0.05)
    exact = float(abstention_threshold(torch.tensor([1000.0]), config)[0])
    decided = decide(_estimate([exact], dof=1000.0), config)
    assert float(decided.psi[0]) == 0.0


def test_the_per_pair_factor_is_the_james_stein_rule_on_this_statistic():
    statistic = torch.tensor([0.5, 3.0, 12.0, 300.0])
    factor = decision_factor(statistic, None, AbstentionConfig(mode=PER_PAIR))
    assert torch.equal(factor, shrinkage_factor(statistic, POSE_DIMENSIONS))


def test_both_at_a_level_of_one_is_exactly_the_per_pair_rule():
    statistic = torch.tensor([0.5, 2.9, 3.0, 12.0, 300.0])
    dof = torch.full((5,), 40.0)

    per_pair = decision_factor(statistic, dof, AbstentionConfig(mode=PER_PAIR))
    at_one = decision_factor(
        statistic, dof, AbstentionConfig(mode=BOTH, level=1.0)
    )

    assert torch.equal(per_pair, at_one)


def test_both_abstains_where_per_pair_would_have_shrunk():
    statistic = torch.tensor([5.0])
    dof = torch.tensor([40.0])

    per_pair = decision_factor(statistic, dof, AbstentionConfig(mode=PER_PAIR))
    both = decision_factor(statistic, dof, AbstentionConfig(mode=BOTH, level=0.05))

    assert float(per_pair[0]) > 0.0
    assert float(both[0]) == 0.0


def test_a_hard_rule_without_the_degrees_of_freedom_is_an_error():
    with pytest.raises(ValueError, match="degrees of freedom"):
        decision_factor(
            torch.tensor([5.0]), None, AbstentionConfig(mode=ABSTAIN, level=0.05)
        )


def test_per_pair_refuses_a_level_because_it_has_no_use_for_one():
    with pytest.raises(ValueError, match="level"):
        AbstentionConfig(mode=PER_PAIR, level=0.05)


def test_a_hard_rule_refuses_a_level_of_zero():
    with pytest.raises(ValueError, match="level"):
        AbstentionConfig(mode=ABSTAIN, level=0.0)


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="mode"):
        AbstentionConfig(mode="gate")


@pytest.mark.parametrize("level", (-0.1, 1.5, float("nan")))
def test_a_level_outside_the_unit_interval_is_refused(level):
    with pytest.raises(ValueError, match="level"):
        AbstentionConfig(mode=ABSTAIN, level=level)


def test_deciding_without_a_statistic_is_an_error_rather_than_a_silent_pass():
    estimate = PoseEstimate(
        psi=torch.tensor([0.3]),
        t=torch.tensor([[1.0, -2.0]]),
        confidence=torch.tensor([9.0]),
    )
    with pytest.raises(ValueError, match="statistic"):
        decide(estimate, AbstentionConfig(mode=ABSTAIN, level=0.05))


def test_an_abstained_pair_is_counted_as_a_fallback_so_coverage_is_measurable():
    from embedding_aware_belt_fusion.alignformer.stage2 import is_fallback

    decided = decide(
        _estimate([1.0, 400.0]), AbstentionConfig(mode=ABSTAIN, level=0.05)
    )
    assert is_fallback(decided).tolist() == [True, False]


# --------------------------------------------------------------------------
# 4. Specs, names and the sweep wiring.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec, mode, level",
    (
        ("abstain:0.05", ABSTAIN, 0.05),
        ("both:0.01", BOTH, 0.01),
        ("per_pair", PER_PAIR, 0.0),
    ),
)
def test_a_spec_parses_to_the_configuration_it_names(spec, mode, level):
    config = AbstentionConfig.parse(spec)
    assert config.mode == mode
    assert config.level == pytest.approx(level)


@pytest.mark.parametrize("spec", ("abstain:wide", "abstain:", "per_pair:0.2"))
def test_a_nonsense_spec_is_refused(spec):
    with pytest.raises(ValueError):
        AbstentionConfig.parse(spec)


def test_arm_names_are_distinct_and_carry_their_level():
    names = {
        AbstentionConfig.parse(spec).name
        for spec in ("abstain:0.5", "abstain:0.05", "both:0.05", "per_pair")
    }
    assert names == {
        "alignformer_abstain_0.5",
        "alignformer_abstain_0.05",
        "alignformer_both_0.05",
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

    arms = (AbstentionConfig.parse("abstain:0.05"), AbstentionConfig.parse("per_pair"))
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

    arms = (AbstentionConfig.parse("abstain:0.05"), AbstentionConfig.parse("per_pair"))
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
# 5. The solve carries the statistic without moving the pose it reports.
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

    without = solve_pose(correspondence, batch, HEADING_LAMBDA, UNWEIGHTED, robust)
    with_statistic = solve_pose(
        correspondence, batch, HEADING_LAMBDA, UNWEIGHTED, robust, statistic=True
    )

    assert torch.equal(without.psi, with_statistic.psi)
    assert torch.equal(without.t, with_statistic.t)
    assert without.offset_statistic is None and without.offset_dof is None
    assert torch.isfinite(with_statistic.offset_statistic).all()
    assert torch.isfinite(with_statistic.offset_dof).all()


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
    decided = decide(estimate, AbstentionConfig(mode=ABSTAIN, level=0.05))
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
    from embedding_aware_belt_fusion.alignformer.robust import HUBER, RobustSolveConfig

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
    arms = (AbstentionConfig.parse("abstain:0.05"), AbstentionConfig.parse("per_pair"))

    estimates = _alignformer_estimates(modules, batch, False, robust, arms)

    assert set(estimates) == {ALIGNFORMER, ALIGNFORMER_IRLS} | {a.name for a in arms}
    irls = estimates[ALIGNFORMER_IRLS]
    assert irls.offset_statistic is not None
    for arm in arms:
        factor = decision_factor(irls.offset_statistic, irls.offset_dof, arm)
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
    from embedding_aware_belt_fusion.alignformer.robust import HUBER, RobustSolveConfig

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
