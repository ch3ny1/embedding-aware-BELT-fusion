"""The exact re-solve: hard nearest-neighbour correspondences under the soft
estimate, then an unweighted closed-form SE(2) fit, iterated.

Why it exists. On V2X-Real test the pairs sharing three or more objects are
two thirds of the frames; FreeAlign sits at 90 % of the oracle there at sigma
2 m while the soft solve loses 0.10 AP@0.7 to it with a 0.8 m mean residual.
A soft Sinkhorn row blends neighbours into one virtual point as the tokens get
noisier, and the weighted solve returns a blurred answer. The remedy is the
one FreeAlign's hard fit has: decide the correspondence, then solve exactly.

What has to hold, each pinned below:

1. a disabled refinement returns the input estimate itself;
2. with three or more unambiguous shared objects the refined fit is exact
   where the soft estimate was off by half a metre;
3. a pair with too few hard correspondences keeps the soft estimate
   untouched -- the sparse slice is where the soft fit beats FreeAlign;
4. the heading fold is applied, so a back-to-front detection does not pull
   the fit;
5. unmatched and gated-out objects do not enter the fit at all;
6. the arm names and the CLI wiring follow the IRLS/abstention conventions.
"""

from __future__ import annotations

import math

import pytest
import torch

from embedding_aware_belt_fusion.alignformer.boxes import BOX_YAW
from embedding_aware_belt_fusion.alignformer.model import PoseEstimate
from embedding_aware_belt_fusion.alignformer.refine import (
    ICP,
    REFINE_NONE,
    RefineConfig,
    icp_refine,
    mutual_nearest,
    refined_name,
)

HEADING_LAMBDA = 2.0


def _apply(psi, t, q):
    cos, sin = math.cos(psi), math.sin(psi)
    rotation = torch.tensor([[cos, -sin], [sin, cos]], dtype=q.dtype)
    return q @ rotation.T + torch.tensor(t, dtype=q.dtype)


def _pair(count=6, psi=0.08, t=(1.2, -0.7), flip_index=None, extra_cav=0, seed=3):
    """Ego boxes and CAV boxes that are the ego boxes moved by the INVERSE of (psi, t),
    so that the correction taking CAV onto ego is exactly (psi, t)."""
    generator = torch.Generator().manual_seed(seed)
    centres = (torch.rand(count, 2, generator=generator) - 0.5) * 50.0
    yaws = torch.rand(count, generator=generator) * 2 * math.pi
    ego = torch.zeros(1, count, 7)
    ego[0, :, :2] = centres
    ego[0, :, BOX_YAW] = yaws
    # q such that R(psi) q + t = p  ->  q = R(-psi) (p - t)
    cav_centres = _apply(-psi, (0.0, 0.0), centres - torch.tensor(t))
    cav_yaws = yaws - psi
    if flip_index is not None:
        cav_yaws = cav_yaws.clone()
        cav_yaws[flip_index] = cav_yaws[flip_index] + math.pi
    cav = torch.zeros(1, count + extra_cav, 7)
    cav[0, :count, :2] = cav_centres
    cav[0, :count, BOX_YAW] = cav_yaws
    for k in range(extra_cav):  # far-away objects the ego never saw
        cav[0, count + k, :2] = torch.tensor([200.0 + 10.0 * k, 200.0])
    batch = {
        "ego_boxes": ego,
        "cav_boxes": cav,
        "ego_mask": torch.ones(1, count, dtype=torch.bool),
        "cav_mask": torch.ones(1, count + extra_cav, dtype=torch.bool),
    }
    return batch, psi, t


def _estimate(psi, t, statistic=True):
    psi_t = torch.tensor([psi])
    return PoseEstimate(
        psi=psi_t,
        t=torch.tensor([list(t)]),
        confidence=torch.tensor([4.0]),
        offset_statistic=torch.tensor([30.0]) if statistic else None,
        offset_dof=torch.tensor([3.0]) if statistic else None,
    )


def _error(refined, psi, t):
    translation = math.hypot(float(refined.t[0, 0]) - t[0], float(refined.t[0, 1]) - t[1])
    d = float(refined.psi[0]) - psi
    return translation, abs(math.degrees(math.atan2(math.sin(d), math.cos(d))))


# ---------------------------------------------------------------------------
# 1. Disabled is identity
# ---------------------------------------------------------------------------


def test_a_disabled_refinement_returns_the_input_estimate_itself():
    batch, psi, t = _pair()
    estimate = _estimate(psi + 0.02, (t[0] + 0.5, t[1]))

    assert icp_refine(estimate, batch, RefineConfig(), HEADING_LAMBDA) is estimate


# ---------------------------------------------------------------------------
# 2. Exactness on a dense pair
# ---------------------------------------------------------------------------


def test_a_blurred_soft_estimate_is_refined_to_the_exact_transform():
    batch, psi, t = _pair(count=6)
    blurred = _estimate(psi + 0.02, (t[0] + 0.5, t[1] - 0.4))

    refined = icp_refine(blurred, batch, RefineConfig(mode=ICP), HEADING_LAMBDA)

    translation, yaw = _error(refined, psi, t)
    assert translation < 1e-4 and yaw < 1e-3
    assert refined.offset_statistic is blurred.offset_statistic  # the decision inputs are untouched
    assert refined.confidence is blurred.confidence


def test_objects_the_ego_never_saw_do_not_enter_the_fit():
    batch, psi, t = _pair(count=5, extra_cav=3)
    blurred = _estimate(psi, (t[0] + 0.6, t[1]))

    refined = icp_refine(blurred, batch, RefineConfig(mode=ICP), HEADING_LAMBDA)

    translation, yaw = _error(refined, psi, t)
    assert translation < 1e-4 and yaw < 1e-3


# ---------------------------------------------------------------------------
# 3. The evidence guard
# ---------------------------------------------------------------------------


def test_a_pair_with_too_few_hard_correspondences_keeps_the_soft_estimate():
    batch, psi, t = _pair(count=2)
    blurred = _estimate(psi + 0.02, (t[0] + 0.5, t[1]))

    refined = icp_refine(blurred, batch, RefineConfig(mode=ICP, min_pairs=3), HEADING_LAMBDA)

    torch.testing.assert_close(refined.psi, blurred.psi)
    torch.testing.assert_close(refined.t, blurred.t)


def test_a_soft_estimate_too_far_off_for_the_first_gate_is_kept_rather_than_matched_wrongly():
    batch, psi, t = _pair(count=6)
    hopeless = _estimate(psi, (t[0] + 9.0, t[1] + 9.0))

    refined = icp_refine(hopeless, batch, RefineConfig(mode=ICP, gates_m=(2.0, 1.0)), HEADING_LAMBDA)

    torch.testing.assert_close(refined.t, hopeless.t)


# ---------------------------------------------------------------------------
# 4. The heading fold
# ---------------------------------------------------------------------------


def test_a_back_to_front_detection_does_not_pull_the_refined_fit():
    batch, psi, t = _pair(count=5, flip_index=1)
    blurred = _estimate(psi, (t[0] + 0.4, t[1] + 0.3))

    refined = icp_refine(blurred, batch, RefineConfig(mode=ICP), HEADING_LAMBDA)

    translation, yaw = _error(refined, psi, t)
    assert translation < 1e-4 and yaw < 1e-3


# ---------------------------------------------------------------------------
# 5. The matching primitive
# ---------------------------------------------------------------------------


def test_mutual_nearest_is_one_to_one_and_respects_the_gate():
    ego = torch.tensor([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    cav = torch.tensor([[0.3, 0.0], [0.6, 0.0], [10.2, 0.0], [40.0, 0.0]])

    ego_idx, cav_idx = mutual_nearest(ego, cav, gate_m=1.0)

    # ego 0 <-> cav 0 (cav 1 is nearer to ego 0 than to anything else but
    # ego 0's nearest is cav 0, so cav 1 stays unmatched); ego 1 <-> cav 2;
    # ego 2 has nothing within the gate.
    assert ego_idx.tolist() == [0, 1]
    assert cav_idx.tolist() == [0, 2]


def test_mutual_nearest_with_an_empty_side_matches_nothing():
    ego_idx, cav_idx = mutual_nearest(torch.zeros(0, 2), torch.zeros(3, 2), gate_m=1.0)

    assert ego_idx.numel() == 0 and cav_idx.numel() == 0


# ---------------------------------------------------------------------------
# 6. Configuration and naming
# ---------------------------------------------------------------------------


def test_config_validates_its_constants():
    with pytest.raises(ValueError):
        RefineConfig(mode="warp")
    with pytest.raises(ValueError):
        RefineConfig(mode=ICP, gates_m=())
    with pytest.raises(ValueError):
        RefineConfig(mode=ICP, gates_m=(1.0, -0.5))
    with pytest.raises(ValueError):
        RefineConfig(mode=ICP, min_pairs=0)
    assert not RefineConfig().enabled and RefineConfig(mode=ICP).enabled
    assert RefineConfig(mode=ICP, gates_m=(2.0, 1.0)).to_dict()["gates_m"] == [2.0, 1.0]
    assert REFINE_NONE == "none"


def test_refined_names_follow_the_arm_they_refine():
    assert refined_name("alignformer_irls") == "alignformer_icp"
    assert refined_name("alignformer_per_pair") == "alignformer_per_pair_icp"
    assert refined_name("alignformer_abstain_0.2") == "alignformer_abstain_0.2_icp"


def test_sweep_estimators_add_one_refined_arm_per_irls_based_arm():
    from embedding_aware_belt_fusion.alignformer.abstain import AbstentionConfig
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import sweep_estimators
    from embedding_aware_belt_fusion.alignformer.robust import HUBER, RobustSolveConfig

    robust = RobustSolveConfig(mode=HUBER, iterations=2)
    arms = [AbstentionConfig.parse("per_pair"), AbstentionConfig.parse("abstain:0.2")]

    names = sweep_estimators([], True, robust, arms, refine=RefineConfig(mode=ICP))

    assert names == [
        "alignformer", "alignformer_irls", "alignformer_per_pair", "alignformer_abstain_0.2",
        "alignformer_icp", "alignformer_per_pair_icp", "alignformer_abstain_0.2_icp", "freealign",
    ]
    assert sweep_estimators([], False, robust, arms, refine=RefineConfig()) == names[:4]
    with pytest.raises(ValueError):
        sweep_estimators([], False, None, [], refine=RefineConfig(mode=ICP))


# ---------------------------------------------------------------------------
# 7. The agreement rule: two estimators, one decision
# ---------------------------------------------------------------------------

from embedding_aware_belt_fusion.alignformer.refine import (  # noqa: E402
    AgreementConfig,
    agree,
    agreement_name,
    disagreement_m,
)


def _two(psi_soft, t_soft, psi_exact, t_exact, engaged):
    soft = _estimate(psi_soft, t_soft)
    exact = PoseEstimate(
        psi=torch.tensor([psi_exact]), t=torch.tensor([list(t_exact)]), confidence=soft.confidence,
        offset_statistic=soft.offset_statistic, offset_dof=soft.offset_dof,
        refined=torch.tensor([engaged]),
    )
    return soft, exact


def test_icp_refine_reports_which_samples_it_engaged_on():
    dense, psi, t = _pair(count=6)
    sparse, _, _ = _pair(count=2)
    blurred = _estimate(psi, (t[0] + 0.5, t[1]))

    assert icp_refine(blurred, dense, RefineConfig(mode=ICP), HEADING_LAMBDA).refined.tolist() == [True]
    assert icp_refine(blurred, sparse, RefineConfig(mode=ICP), HEADING_LAMBDA).refined.tolist() == [False]


def test_disagreement_is_metres_of_translation_plus_lambda_times_heading():
    soft, exact = _two(0.0, (0.0, 0.0), 0.1, (0.3, 0.4), True)

    assert disagreement_m(soft, exact, HEADING_LAMBDA).item() == pytest.approx(0.5 + HEADING_LAMBDA * 0.1)


def test_disagreement_wraps_the_heading_difference():
    soft, exact = _two(math.pi - 0.05, (0.0, 0.0), -math.pi + 0.05, (0.0, 0.0), True)

    assert disagreement_m(soft, exact, HEADING_LAMBDA).item() == pytest.approx(HEADING_LAMBDA * 0.1, abs=1e-6)


def test_agreeing_estimates_answer_with_the_exact_one():
    soft, exact = _two(0.02, (1.0, 0.0), 0.0, (1.2, 0.1), True)
    fallback = _estimate(0.0, (0.0, 0.0))

    decided = agree(soft, exact, fallback, AgreementConfig(tolerance_m=0.5), HEADING_LAMBDA)

    torch.testing.assert_close(decided.t, exact.t)
    torch.testing.assert_close(decided.psi, exact.psi)


def test_disagreeing_estimates_abstain():
    soft, exact = _two(0.0, (1.0, 0.0), 0.0, (3.0, 0.0), True)
    fallback = _estimate(0.0, (9.0, 9.0))

    decided = agree(soft, exact, fallback, AgreementConfig(tolerance_m=0.5), HEADING_LAMBDA)

    assert decided.t.abs().max().item() == 0.0 and decided.psi.abs().max().item() == 0.0


def test_a_pair_the_refinement_did_not_engage_on_takes_the_fallback_decision():
    soft, exact = _two(0.0, (1.0, 0.0), 0.0, (1.0, 0.0), False)
    fallback = _estimate(0.0, (0.7, 0.0))

    decided = agree(soft, exact, fallback, AgreementConfig(tolerance_m=0.5), HEADING_LAMBDA)

    torch.testing.assert_close(decided.t, fallback.t)


def test_agreement_config_and_names():
    with pytest.raises(ValueError):
        AgreementConfig(tolerance_m=0.0)
    assert AgreementConfig(tolerance_m=0.5).to_dict() == {"tolerance_m": 0.5}
    assert agreement_name("alignformer_abstain_0.2", 0.5) == "alignformer_abstain_0.2_agree_0.5"
    assert agreement_name("alignformer_per_pair", 1.0) == "alignformer_per_pair_agree_1"


def test_sweep_estimators_add_one_agreement_arm_per_decision_arm_and_tolerance():
    from embedding_aware_belt_fusion.alignformer.abstain import AbstentionConfig
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import sweep_estimators
    from embedding_aware_belt_fusion.alignformer.robust import HUBER, RobustSolveConfig

    robust = RobustSolveConfig(mode=HUBER, iterations=2)
    arms = [AbstentionConfig.parse("abstain:0.2")]
    tolerances = [AgreementConfig(0.5), AgreementConfig(1.0)]

    names = sweep_estimators([], False, robust, arms, refine=RefineConfig(mode=ICP), agreement=tolerances)

    assert names == [
        "alignformer", "alignformer_irls", "alignformer_abstain_0.2", "alignformer_icp", "alignformer_abstain_0.2_icp",
        "alignformer_abstain_0.2_agree_0.5", "alignformer_abstain_0.2_agree_1",
    ]
    with pytest.raises(ValueError):
        sweep_estimators([], False, robust, arms, refine=RefineConfig(), agreement=tolerances)


# ----------------------------------------------------------------------------
# The robust exact fit: RANSAC over the hard pairs
# ----------------------------------------------------------------------------


def _boxes_from(centres, yaws):
    import torch

    boxes = torch.zeros(len(centres), 7)
    boxes[:, :2] = torch.tensor(centres, dtype=torch.float32)
    boxes[:, 6] = torch.tensor(yaws, dtype=torch.float32)
    return boxes


def test_two_point_se2_recovers_the_transform_between_two_pairs():
    import math

    import torch

    from embedding_aware_belt_fusion.alignformer.refine import _moved, se2_from_two_pairs

    psi, t = torch.tensor(0.3), torch.tensor([1.0, -2.0])
    source = torch.tensor([[0.0, 0.0], [10.0, 5.0]])
    target = _moved(source, psi, t)

    got_psi, got_t = se2_from_two_pairs(target, source)

    assert math.isclose(float(got_psi), 0.3, abs_tol=1e-5)
    torch.testing.assert_close(got_t, t, atol=1e-4, rtol=0)


def test_robust_exact_solve_ignores_a_lane_neighbour_the_plain_solve_cannot():
    import torch

    from embedding_aware_belt_fusion.alignformer.refine import _exact_solve, _moved, robust_exact_solve

    # Five true correspondences under psi = 0.05, t = (1.2, -0.4); the fifth
    # CAV box is a lane neighbour 3.5 m across from the real partner.
    psi, t = torch.tensor(0.05), torch.tensor([1.2, -0.4])
    ego_centres = [[5.0, 0.0], [15.0, 2.0], [25.0, -1.0], [35.0, 3.0], [45.0, 0.0]]
    ego = _boxes_from(ego_centres, [0.0] * 5)
    cav_true = _moved(ego[:, :2], -psi, torch.zeros(2))  # rough inverse is fine: we fit cav -> ego
    cav = _boxes_from(cav_true.tolist(), [0.0] * 5)
    # Define the truth as the transform that maps these cav centres onto ego.
    from embedding_aware_belt_fusion.alignformer.refine import se2_from_two_pairs

    true_psi, true_t = se2_from_two_pairs(ego[:2, :2], cav[:2, :2])
    cav_wrong = cav.clone()
    cav_wrong[4, 1] += 3.5  # the outlier
    idx = torch.arange(5)

    plain_psi, plain_t = _exact_solve(ego, cav_wrong, idx, idx, heading_lambda=2.0)
    rob_psi, rob_t, inliers = robust_exact_solve(ego, cav_wrong, idx, idx, heading_lambda=2.0, inlier_m=1.0)

    plain_err = float((_moved(cav_wrong[:4, :2], plain_psi, plain_t) - ego[:4, :2]).norm(dim=-1).mean())
    robust_err = float((_moved(cav_wrong[:4, :2], rob_psi, rob_t) - ego[:4, :2]).norm(dim=-1).mean())
    assert robust_err < 0.05 < plain_err
    assert inliers.tolist() == [True, True, True, True, False]


def test_robust_exact_solve_is_deterministic_with_many_pairs():
    import torch

    from embedding_aware_belt_fusion.alignformer.refine import robust_exact_solve

    g = torch.Generator().manual_seed(3)
    ego = _boxes_from(torch.rand(30, 2, generator=g).mul(60).tolist(), [0.0] * 30)
    cav = ego.clone()
    cav[:, 0] += 0.7
    idx = torch.arange(30)

    first = robust_exact_solve(ego, cav, idx, idx, heading_lambda=2.0, inlier_m=1.0)
    second = robust_exact_solve(ego, cav, idx, idx, heading_lambda=2.0, inlier_m=1.0)

    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])
    assert bool(first[2].all())


def test_refine_config_and_names_for_the_ransac_mode():
    from embedding_aware_belt_fusion.alignformer.refine import ICP, ICP_RANSAC, RefineConfig, refined_name

    config = RefineConfig(mode=ICP_RANSAC, inlier_m=0.5)

    assert config.enabled and config.to_dict()["inlier_m"] == 0.5
    assert refined_name("alignformer_abstain_0.2", ICP) == "alignformer_abstain_0.2_icp"
    assert refined_name("alignformer_abstain_0.2", ICP_RANSAC) == "alignformer_abstain_0.2_icpr"
    assert refined_name("alignformer_irls", ICP_RANSAC) == "alignformer_icpr"
    with pytest.raises(ValueError):
        RefineConfig(mode=ICP_RANSAC, inlier_m=0.0)


def test_icp_refine_in_ransac_mode_marks_engagement_and_survives_an_outlier():
    import torch

    from embedding_aware_belt_fusion.alignformer.model import PoseEstimate
    from embedding_aware_belt_fusion.alignformer.refine import ICP_RANSAC, RefineConfig, _moved, icp_refine

    ego = _boxes_from([[5.0, 0.0], [15.0, 2.0], [25.0, -1.0], [35.0, 3.0], [45.0, 0.0]], [0.0] * 5)
    cav = ego.clone()
    cav[:, 0] -= 0.8  # the true correction is +0.8 in x
    cav[4, 1] += 3.5  # one lane neighbour
    batch = {"ego_boxes": ego.unsqueeze(0), "cav_boxes": cav.unsqueeze(0)}
    estimate = PoseEstimate(psi=torch.zeros(1), t=torch.tensor([[0.3, 0.0]]), confidence=torch.ones(1))

    out = icp_refine(estimate, batch, RefineConfig(mode=ICP_RANSAC), heading_lambda=2.0)

    assert out.refined.tolist() == [True]
    err = float((_moved(cav[:4, :2], out.psi[0], out.t[0]) - ego[:4, :2]).norm(dim=-1).mean())
    assert err < 0.05
