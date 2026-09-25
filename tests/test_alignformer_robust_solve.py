"""The robust re-weighting loop wrapped around head B's closed-form solve.

Task 22's hypothesis is that a *single* weighted Kabsch over soft Sinkhorn mass
lets a wrong correspondence with small-but-nonzero mass bias the fit forever,
and that an IRLS loop recovers the dense-regime precision FreeAlign gets from a
hard robust fit. Four things have to be true for the measurement that follows
to mean anything, and they are what is tested here:

1. a disabled loop reproduces the deployed estimator **bit for bit**, so every
   prior measurement still reproduces and the new arm is the only thing that
   moved;
2. the loop actually discards outlier correspondences -- if it did not, a
   favourable AP would be an artefact of something else;
3. the mass guard suppresses the loop on sparse pairs *without* suppressing it
   everywhere, because a guard that never lets the loop run would also
   reproduce the deployed arm bit for bit and look like a null;
4. the residual weights carry no gradient, so the training graph is unchanged.

The negative control is
``test_corrupting_every_correspondence_destroys_the_robust_recovery``: a robust
solver is *supposed* to survive corruption, so the control has to corrupt more
than the loop can reject and assert the honest arm still recovers in the same
test, or it would pass with an inert estimator.
"""

import math

import pytest
import torch

from embedding_aware_belt_fusion.alignformer.model import solve_pose
from embedding_aware_belt_fusion.alignformer.procrustes import (
    MIN_MATCH_MASS,
    weighted_se2_kabsch,
)
from embedding_aware_belt_fusion.alignformer.robust import (
    GEMAN_MCCLURE,
    HUBER,
    ROBUST_MODES,
    RobustSolveConfig,
    robust_factor,
    robust_se2_kabsch,
    weighted_median,
)


def _apply(psi, t, q):
    cos, sin = math.cos(psi), math.sin(psi)
    rotation = torch.tensor([[cos, -sin], [sin, cos]], dtype=q.dtype)
    return q @ rotation.T + torch.tensor(t, dtype=q.dtype)


def _clean_problem(count: int = 10, psi: float = 0.15, t=(1.5, -2.0)):
    """``(p, q, w, psi, t)``: a consistent point set every weight agrees on."""
    generator = torch.Generator().manual_seed(7)
    q = (torch.rand(count, 2, generator=generator) - 0.5) * 60.0
    p = _apply(psi, t, q)
    return p.unsqueeze(0), q.unsqueeze(0), torch.ones(1, count), psi, t


def _with_outliers(outliers: int = 2, count: int = 12):
    """The same problem with ``outliers`` source points moved somewhere else.

    The outliers keep full weight, which is exactly the situation the brief
    diagnoses: soft assignment never removes a wrong correspondence, it only
    gives it a smaller number.
    """
    p, q, w, psi, t = _clean_problem(count)
    displacement = torch.tensor([[9.0, -11.0], [-13.0, 7.0], [15.0, 15.0]])
    for index in range(outliers):
        q[0, index] = q[0, index] + displacement[index % 3]
    return p, q, w, psi, t


def _error(psi_hat, t_hat, psi_true, t_true):
    translation = math.hypot(
        float(t_hat[0, 0]) - t_true[0], float(t_hat[0, 1]) - t_true[1]
    )
    difference = float(psi_hat[0]) - psi_true
    yaw = abs(math.degrees(math.atan2(math.sin(difference), math.cos(difference))))
    return translation, yaw


_ENGAGING = RobustSolveConfig(mode=HUBER, iterations=2, min_evidence=3.0)


# --------------------------------------------------------------------------
# 1. A disabled loop changes nothing at all.
# --------------------------------------------------------------------------


def test_a_disabled_mode_reproduces_the_plain_kabsch_bit_for_bit():
    p, q, w, _, _ = _with_outliers()
    expected_psi, expected_t = weighted_se2_kabsch(p, q, w)

    solution = robust_se2_kabsch(
        p, q, w, evidence=torch.tensor([99.0]), config=RobustSolveConfig()
    )

    assert torch.equal(solution.psi, expected_psi)
    assert torch.equal(solution.t, expected_t)
    assert torch.equal(solution.weights, w)
    assert not bool(solution.engaged.any())


def test_zero_iterations_reproduce_the_plain_kabsch_bit_for_bit():
    p, q, w, _, _ = _with_outliers()
    expected_psi, expected_t = weighted_se2_kabsch(p, q, w)

    solution = robust_se2_kabsch(
        p, q, w,
        evidence=torch.tensor([99.0]),
        config=RobustSolveConfig(mode=HUBER, iterations=0),
    )

    assert torch.equal(solution.psi, expected_psi)
    assert torch.equal(solution.t, expected_t)


def test_solve_pose_without_a_robust_config_is_the_deployed_solve_bit_for_bit():
    from embedding_aware_belt_fusion.alignformer.model import (
        Correspondence,
        reduce_correspondence,
    )
    from embedding_aware_belt_fusion.alignformer.variance import UNWEIGHTED

    batch, weights = _pair_batch()
    correspondence = reduce_correspondence(weights, batch, UNWEIGHTED)
    assert isinstance(correspondence, Correspondence)

    plain = solve_pose(correspondence, batch, 2.0, UNWEIGHTED)
    disabled = solve_pose(
        correspondence, batch, 2.0, UNWEIGHTED, robust=RobustSolveConfig()
    )

    assert torch.equal(plain.psi, disabled.psi)
    assert torch.equal(plain.t, disabled.t)


def _pair_batch():
    """A tiny ego/CAV object set plus a soft correspondence over it."""
    from embedding_aware_belt_fusion.alignformer.boxes import BOX_YAW

    count = 5
    generator = torch.Generator().manual_seed(11)
    centres = (torch.rand(count, 2, generator=generator) - 0.5) * 40.0
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
    }
    return batch, torch.eye(count).unsqueeze(0)


# --------------------------------------------------------------------------
# 2. The mechanism: the loop must actually reject an outlier correspondence.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", (HUBER, GEMAN_MCCLURE))
def test_the_loop_recovers_a_transform_a_single_solve_gets_wrong(mode):
    p, q, w, psi_true, t_true = _with_outliers(outliers=2, count=12)

    plain_psi, plain_t = weighted_se2_kabsch(p, q, w)
    plain_translation, _ = _error(plain_psi, plain_t, psi_true, t_true)

    solution = robust_se2_kabsch(
        p, q, w,
        evidence=torch.tensor([12.0]),
        config=RobustSolveConfig(mode=mode, iterations=3, min_evidence=3.0),
    )
    robust_translation, robust_yaw = _error(
        solution.psi, solution.t, psi_true, t_true
    )

    assert plain_translation > 0.5, "the single solve must be broken for this to test"
    assert robust_translation < 0.05
    assert robust_yaw < 0.5
    assert bool(solution.engaged.all())


def test_the_loop_leaves_an_outlier_free_fit_where_it_found_it():
    p, q, w, psi_true, t_true = _clean_problem()

    solution = robust_se2_kabsch(
        p, q, w, evidence=torch.tensor([10.0]),
        config=RobustSolveConfig(mode=HUBER, iterations=3, min_evidence=3.0),
    )

    translation, yaw = _error(solution.psi, solution.t, psi_true, t_true)
    assert translation < 1e-3
    assert yaw < 1e-2


def test_a_down_weighted_outlier_is_pushed_further_down_by_the_loop():
    p, q, w, _, _ = _with_outliers(outliers=1, count=12)

    solution = robust_se2_kabsch(
        p, q, w, evidence=torch.tensor([12.0]),
        config=RobustSolveConfig(mode=HUBER, iterations=2, min_evidence=3.0),
    )

    final = solution.weights[0]
    assert float(final[0]) < 0.2 * float(final[1:].mean())


# --------------------------------------------------------------------------
# 3. The guard -- both directions. It must fire on sparse evidence and it must
#    NOT fire everywhere, or a null result would be unfalsifiable.
# --------------------------------------------------------------------------


def test_the_mass_guard_leaves_a_sparse_pair_exactly_as_the_single_solve_left_it():
    p, q, w, _, _ = _with_outliers()
    expected_psi, expected_t = weighted_se2_kabsch(p, q, w)

    solution = robust_se2_kabsch(
        p, q, w, evidence=torch.tensor([2.0]), config=_ENGAGING
    )

    assert torch.equal(solution.psi, expected_psi)
    assert torch.equal(solution.t, expected_t)
    assert torch.equal(solution.weights, w)
    assert not bool(solution.engaged.any())


def test_the_guard_engages_as_soon_as_there_is_evidence():
    p, q, w, _, _ = _with_outliers()
    plain_psi, plain_t = weighted_se2_kabsch(p, q, w)

    solution = robust_se2_kabsch(
        p, q, w, evidence=torch.tensor([3.0]), config=_ENGAGING
    )

    assert bool(solution.engaged.all())
    assert not torch.equal(solution.psi, plain_psi)
    assert not torch.equal(solution.t, plain_t)


def test_the_guard_is_decided_per_sample_not_for_the_whole_batch():
    p, q, w, _, _ = _with_outliers()
    p = torch.cat([p, p])
    q = torch.cat([q, q])
    w = torch.cat([w, w])
    plain_psi, plain_t = weighted_se2_kabsch(p, q, w)

    solution = robust_se2_kabsch(
        p, q, w, evidence=torch.tensor([2.0, 12.0]), config=_ENGAGING
    )

    assert solution.engaged.tolist() == [False, True]
    assert float(solution.psi[0]) == float(plain_psi[0])
    assert torch.equal(solution.t[0], plain_t[0])
    assert float(solution.psi[1]) != float(plain_psi[1])


# --------------------------------------------------------------------------
# 4. Weight bookkeeping: MIN_MATCH_MASS must keep firing on the same pairs.
# --------------------------------------------------------------------------


def test_the_final_weights_keep_the_total_the_input_weights_had():
    p, q, w, _, _ = _with_outliers()
    w = w * 0.37

    solution = robust_se2_kabsch(
        p, q, w, evidence=torch.tensor([12.0]),
        config=RobustSolveConfig(mode=GEMAN_MCCLURE, iterations=3, min_evidence=3.0),
    )

    assert float(solution.weights.sum()) == pytest.approx(float(w.sum()), rel=1e-5)


def test_a_pair_below_min_match_mass_is_suppressed_with_the_loop_as_without_it():
    p, q, w, _, _ = _with_outliers()
    w = w * (0.5 * MIN_MATCH_MASS / float(w.sum()))
    assert float(w.sum()) < MIN_MATCH_MASS

    solution = robust_se2_kabsch(
        p, q, w, evidence=torch.tensor([99.0]),
        config=RobustSolveConfig(mode=HUBER, iterations=3, min_evidence=0.0),
    )

    assert float(solution.psi[0]) == 0.0
    assert torch.equal(solution.t[0], torch.zeros(2))


def test_zero_weight_points_stay_at_zero_weight():
    p, q, w, _, _ = _with_outliers()
    w = w.clone()
    w[0, -1] = 0.0

    solution = robust_se2_kabsch(
        p, q, w, evidence=torch.tensor([12.0]), config=_ENGAGING
    )

    assert float(solution.weights[0, -1]) == 0.0


# --------------------------------------------------------------------------
# 5. The gradient path stays the single-solve one.
# --------------------------------------------------------------------------


def test_the_residual_weights_are_detached_so_the_gradient_is_the_single_solve_one():
    p, q, w, _, _ = _with_outliers()
    p = p.clone().requires_grad_(True)

    solution = robust_se2_kabsch(
        p, q, w, evidence=torch.tensor([12.0]), config=_ENGAGING
    )
    (robust_grad,) = torch.autograd.grad(solution.psi.sum(), p, retain_graph=True)

    # The same solve with the final weights frozen. If the residual path were
    # live these two gradients would differ.
    frozen = solution.weights.detach()
    psi, _ = weighted_se2_kabsch(p, q, frozen)
    (frozen_grad,) = torch.autograd.grad(psi.sum(), p)

    assert torch.allclose(robust_grad, frozen_grad, atol=1e-6)


def test_the_robust_factor_itself_carries_no_gradient():
    residual = torch.tensor([[0.1, 5.0, 0.2]], requires_grad=True)
    scale = torch.tensor([[0.2]], requires_grad=True)

    factor = robust_factor(residual, scale, _ENGAGING)

    assert not factor.requires_grad


def test_the_loop_is_finite_when_every_residual_is_zero():
    p, q, w, _, _ = _clean_problem(count=4)

    solution = robust_se2_kabsch(
        p, q, w, evidence=torch.tensor([4.0]),
        config=RobustSolveConfig(mode=GEMAN_MCCLURE, iterations=3, min_evidence=3.0),
    )

    assert torch.isfinite(solution.psi).all()
    assert torch.isfinite(solution.t).all()
    assert torch.isfinite(solution.weights).all()


# --------------------------------------------------------------------------
# 6. NEGATIVE CONTROL.
# --------------------------------------------------------------------------


def test_corrupting_every_correspondence_destroys_the_robust_recovery():
    """A robust loop that "recovers" from total corruption is measuring nothing.

    The honest arm is asserted in the same test, so this cannot pass by the
    estimator being inert: the loop must recover the transform when the
    correspondences are real and must NOT when they are not.
    """
    p, q, w, psi_true, t_true = _clean_problem(count=12)
    config = RobustSolveConfig(mode=HUBER, iterations=3, min_evidence=3.0)

    honest = robust_se2_kabsch(p, q, w, evidence=torch.tensor([12.0]), config=config)
    honest_translation, _ = _error(honest.psi, honest.t, psi_true, t_true)

    # Every source point re-assigned to a different target: a derangement, so
    # no subset of the correspondences agrees on the true transform.
    order = torch.roll(torch.arange(12), shifts=5)
    corrupted = robust_se2_kabsch(
        p, q[:, order], w, evidence=torch.tensor([12.0]), config=config
    )
    corrupted_translation, corrupted_yaw = _error(
        corrupted.psi, corrupted.t, psi_true, t_true
    )

    assert honest_translation < 0.05
    assert corrupted_translation > 1.0
    assert corrupted_yaw > 5.0


# --------------------------------------------------------------------------
# 7. The pieces.
# --------------------------------------------------------------------------


def test_the_weighted_median_respects_the_weights():
    values = torch.tensor([[1.0, 2.0, 3.0, 100.0]])

    unweighted = weighted_median(values, torch.ones(1, 4))
    dominated = weighted_median(values, torch.tensor([[0.01, 0.01, 0.01, 10.0]]))

    assert float(unweighted) == pytest.approx(2.0)
    assert float(dominated) == pytest.approx(100.0)


def test_the_weighted_median_ignores_the_input_order():
    values = torch.tensor([[7.0, 1.0, 3.0, 2.0, 9.0]])
    weights = torch.tensor([[1.0, 2.0, 1.0, 1.0, 1.0]])
    order = torch.tensor([3, 0, 4, 1, 2])

    assert float(weighted_median(values, weights)) == pytest.approx(
        float(weighted_median(values[:, order], weights[:, order]))
    )


@pytest.mark.parametrize("mode", (HUBER, GEMAN_MCCLURE))
def test_the_robust_factor_is_one_at_zero_and_falls_with_the_residual(mode):
    residual = torch.tensor([[0.0, 0.5, 1.0, 4.0, 40.0]])
    scale = torch.ones(1, 1)

    factor = robust_factor(residual, scale, RobustSolveConfig(mode=mode, iterations=1))

    assert float(factor[0, 0]) == pytest.approx(1.0)
    assert (factor[0, 1:] <= factor[0, :-1] + 1e-9).all()
    assert float(factor[0, -1]) < 0.05
    assert (factor >= 0).all()


def test_huber_leaves_an_inlier_alone_and_geman_mcclure_does_not():
    residual = torch.tensor([[0.5]])
    scale = torch.ones(1, 1)

    huber = robust_factor(residual, scale, RobustSolveConfig(mode=HUBER, iterations=1))
    geman = robust_factor(
        residual, scale, RobustSolveConfig(mode=GEMAN_MCCLURE, iterations=1)
    )

    assert float(huber) == pytest.approx(1.0)
    assert float(geman) < 0.7


def test_an_unknown_mode_is_rejected_rather_than_silently_disabled():
    with pytest.raises(ValueError, match="mode"):
        RobustSolveConfig(mode="tukey")


def test_a_negative_iteration_count_is_rejected():
    with pytest.raises(ValueError, match="iterations"):
        RobustSolveConfig(mode=HUBER, iterations=-1)


def test_a_negative_evidence_threshold_is_rejected():
    with pytest.raises(ValueError, match="min_evidence"):
        RobustSolveConfig(mode=HUBER, iterations=1, min_evidence=-1.0)


def test_the_config_records_itself_for_the_result_file():
    config = RobustSolveConfig(mode=GEMAN_MCCLURE, iterations=2, min_evidence=4.0)

    recorded = config.to_dict()

    assert recorded["mode"] == GEMAN_MCCLURE
    assert recorded["iterations"] == 2
    assert recorded["min_evidence"] == 4.0
    assert "none" in ROBUST_MODES


# --------------------------------------------------------------------------
# 8. Head B end to end: the arm the sweep actually runs.
# --------------------------------------------------------------------------


def _oracle_head_b_batch(outlier_index=None):
    """Head B with unambiguous one-hot embeddings, optionally with one liar.

    ``outlier_index`` displaces one CAV box so that its (correct, confident)
    correspondence disagrees with every other one -- the detector-blunder case
    a single weighted solve cannot reject.
    """
    from embedding_aware_belt_fusion.alignformer.model import AlignFormerB

    torch.manual_seed(0)
    count = 8
    identity = torch.eye(count).unsqueeze(0)
    centres = torch.tensor([[[0.0, 0.0], [12.0, 3.0], [-8.0, 5.0], [20.0, -7.0],
                             [4.0, 9.0], [-15.0, -2.0], [7.0, -12.0], [-3.0, 16.0]]])
    yaws = torch.rand(1, count) * 2 * math.pi

    from test_alignformer_model import _batch, _boxes_from_centres

    cav_boxes = _boxes_from_centres(centres, yaws)
    true_psi, true_t = 0.15, torch.tensor([[1.2, -0.8]])
    cos, sin = math.cos(true_psi), math.sin(true_psi)
    rotation = torch.tensor([[cos, -sin], [sin, cos]])
    ego_boxes = cav_boxes.clone()
    ego_boxes[..., :2] = centres @ rotation.T + true_t
    ego_boxes[..., 6] = yaws + true_psi
    if outlier_index is not None:
        cav_boxes[0, outlier_index, :2] += torch.tensor([14.0, -17.0])

    model = AlignFormerB(embed_dim=count).eval()
    model.use_raw_embedding_scores = True
    batch = _batch(
        ego_boxes, cav_boxes,
        ego_embeddings=identity.clone(), cav_embeddings=identity.clone(),
    )
    return model, batch, true_psi, true_t


def test_head_b_with_no_robust_config_is_unchanged():
    model, batch, _, _ = _oracle_head_b_batch(outlier_index=0)

    with torch.no_grad():
        plain = model(batch)
        disabled = model(batch, robust=RobustSolveConfig())

    assert torch.equal(plain.psi, disabled.psi)
    assert torch.equal(plain.t, disabled.t)


def test_head_b_with_the_loop_rejects_a_blundered_detection():
    model, batch, true_psi, true_t = _oracle_head_b_batch(outlier_index=0)

    with torch.no_grad():
        plain = model(batch)
        robust = model(
            batch,
            robust=RobustSolveConfig(mode=HUBER, iterations=3, min_evidence=3.0),
        )

    plain_error = math.hypot(
        float(plain.t[0, 0] - true_t[0, 0]), float(plain.t[0, 1] - true_t[0, 1])
    )
    robust_error = math.hypot(
        float(robust.t[0, 0] - true_t[0, 0]), float(robust.t[0, 1] - true_t[0, 1])
    )
    assert plain_error > 0.5
    assert robust_error < 0.25 * plain_error
    assert float(robust.psi[0]) == pytest.approx(true_psi, abs=0.05)


def test_an_empty_object_set_answers_the_identity_rather_than_raising():
    # An agent that detected nothing is a real situation on this data, and
    # weighted_se2_kabsch already answers the identity for it. A weighted
    # median over zero residuals is undefined, so the loop has to decline
    # instead of indexing into an empty dimension.
    empty_p = torch.zeros(1, 0, 2)
    empty_q = torch.zeros(1, 0, 2)
    empty_w = torch.zeros(1, 0)

    solution = robust_se2_kabsch(
        empty_p, empty_q, empty_w,
        evidence=torch.tensor([0.0]),
        config=RobustSolveConfig(mode=HUBER, iterations=2, min_evidence=0.0),
    )

    expected_psi, expected_t = weighted_se2_kabsch(empty_p, empty_q, empty_w)
    assert torch.equal(solution.psi, expected_psi)
    assert torch.equal(solution.t, expected_t)
    assert not bool(solution.engaged.any())
