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
