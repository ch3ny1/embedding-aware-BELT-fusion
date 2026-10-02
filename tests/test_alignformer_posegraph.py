"""A frame-level pose graph over all agents: the sparse pair solved through a third agent.

Every V2X-Real validation frame and 58 % of test frames carry four agents.
A CAV that shares one object with the ego often shares several with another
agent whose correction to the ego is well determined, and pairwise
corrections compose. The unknowns are each CAV's correction in the EGO
frame; the ego's own measurement of a CAV constrains it directly, and a
CAV i's measurement of a CAV j constrains ``T_i<-ego C_j C_i^-1 T_ego<-i``.
A weak prior at the identity keeps an unobserved node at "uncorrected".

What has to hold:

1. SE(2) composition, inverse and the wrap behave;
2. a graph with only direct measurements returns them unchanged;
3. a CAV the ego cannot measure is recovered exactly from a third agent's
   consistent measurements;
4. an unobserved CAV stays at the identity;
5. precision weights decide a conflict between two measurements;
6. the conjugation convention is the one the sweep uses (projecting CAV
   boxes with the noisy relative pose).
"""

from __future__ import annotations

import math

import pytest
import torch

from embedding_aware_belt_fusion.alignformer.posegraph import (
    SE2,
    Measurement,
    apply_se2,
    compose,
    inverse,
    solve_pose_graph,
    wrap,
)

EGO = "ego"


def _se2(psi, x, y):
    # float64: the truth is composed through 30 m lever arms, and float32
    # rounding there is 3e-4 m, which would hide a solver error of that size.
    return SE2(psi=torch.tensor(float(psi), dtype=torch.float64), t=torch.tensor([float(x), float(y)], dtype=torch.float64))


def _close(a: SE2, b: SE2, tol=1e-5):
    assert float(wrap(a.psi - b.psi).abs()) < tol, (a, b)
    torch.testing.assert_close(a.t, b.t, atol=tol, rtol=0)


# ---------------------------------------------------------------------------
# 1. SE(2) algebra
# ---------------------------------------------------------------------------


def test_compose_then_inverse_is_the_identity_and_apply_follows_compose():
    a, b = _se2(0.3, 1.0, -2.0), _se2(-0.7, 4.0, 0.5)
    points = torch.tensor([[1.0, 2.0], [-3.0, 0.5]], dtype=torch.float64)

    _close(compose(a, inverse(a)), _se2(0.0, 0.0, 0.0))
    torch.testing.assert_close(apply_se2(compose(a, b), points), apply_se2(a, apply_se2(b, points)))


def test_wrap_keeps_angles_in_minus_pi_to_pi():
    assert float(wrap(torch.tensor(math.pi + 0.1))) == pytest.approx(-math.pi + 0.1)
    assert float(wrap(torch.tensor(-math.pi - 0.1))) == pytest.approx(math.pi - 0.1)


# ---------------------------------------------------------------------------
# 2-4. The graph
# ---------------------------------------------------------------------------


def _frame():
    """True ego-frame corrections for two CAVs and the (noisy) relative poses."""
    truth = {"a": _se2(0.02, 1.0, -0.5), "b": _se2(-0.03, -0.8, 1.2)}
    # T_i<-ego for each CAV: where the ego sits in CAV i's frame.
    frames = {"a": _se2(0.4, 30.0, -5.0), "b": _se2(-1.1, -12.0, 20.0)}
    return truth, frames


def _cross(truth, frames, observer, target):
    """What CAV ``observer`` measures for CAV ``target``: T_i<-ego C_j C_i^-1 T_ego<-i."""
    t_i_ego = frames[observer]
    return compose(compose(compose(t_i_ego, truth[target]), inverse(truth[observer])), inverse(t_i_ego))


def test_direct_measurements_alone_come_back_unchanged():
    truth, frames = _frame()
    measurements = [Measurement(EGO, k, v, None) for k, v in truth.items()]

    solved = solve_pose_graph(EGO, ["a", "b"], frames, measurements)

    _close(solved["a"], truth["a"])
    _close(solved["b"], truth["b"])


def test_a_cav_the_ego_cannot_measure_is_recovered_through_a_third_agent():
    truth, frames = _frame()
    measurements = [
        Measurement(EGO, "a", truth["a"], None),            # ego sees a well
        Measurement("a", "b", _cross(truth, frames, "a", "b"), None),  # a sees b well; ego cannot see b
    ]

    solved = solve_pose_graph(EGO, ["a", "b"], frames, measurements)

    _close(solved["a"], truth["a"], tol=1e-4)
    _close(solved["b"], truth["b"], tol=1e-4)


def test_an_unobserved_cav_stays_at_the_identity():
    truth, frames = _frame()
    measurements = [Measurement(EGO, "a", truth["a"], None)]

    solved = solve_pose_graph(EGO, ["a", "b"], frames, measurements)

    _close(solved["b"], _se2(0.0, 0.0, 0.0), tol=1e-6)


# ---------------------------------------------------------------------------
# 5. Precision decides a conflict
# ---------------------------------------------------------------------------


def test_the_more_precise_of_two_conflicting_measurements_wins():
    truth, frames = _frame()
    wrong = _se2(0.02, 3.0, -0.5)  # 2 m off in x
    sharp = torch.eye(3, dtype=torch.float64) * 100.0
    blunt = torch.eye(3, dtype=torch.float64) * 1.0
    measurements = [Measurement(EGO, "a", truth["a"], sharp), Measurement(EGO, "a", wrong, blunt)]

    solved = solve_pose_graph(EGO, ["a"], frames, measurements)

    # 100:1 weighting pulls the answer to within 2 cm of the sharp one.
    assert abs(float(solved["a"].t[0]) - 1.0) < 0.03


# ---------------------------------------------------------------------------
# 6. The convention matches the sweep's projection
# ---------------------------------------------------------------------------


def test_the_cross_measurement_is_what_projecting_boxes_with_noisy_poses_produces():
    """A CAV i that projects CAV j's boxes into ITS frame with both noisy
    poses sees exactly the SE(2) the graph predicts, so a model estimate from
    that pair is a valid measurement."""
    truth, frames = _frame()
    boxes_j = torch.tensor([[5.0, 1.0], [12.0, -3.0], [20.0, 7.0]], dtype=torch.float64)  # in j's own frame
    # Ego-frame placements: true = T_ego<-j boxes; noisy = C_j^-1 true (C_j corrects noisy -> true).
    t_ego_j = inverse(frames["b"])
    true_in_ego = apply_se2(t_ego_j, boxes_j)
    noisy_in_ego = apply_se2(inverse(truth["b"]), true_in_ego)
    # Agent a's frame is itself displaced: its noisy frame places ego-frame points through C_a^-1 too.
    t_a_ego = frames["a"]
    true_in_a = apply_se2(t_a_ego, true_in_ego)
    noisy_in_a = apply_se2(t_a_ego, apply_se2(truth["a"], noisy_in_ego))  # a's own error undone... see below

    predicted = _cross(truth, frames, "a", "b")

    # The correction a must apply to its noisy view of j to reach the truth in its frame:
    torch.testing.assert_close(apply_se2(predicted, noisy_in_a), true_in_a, atol=1e-4, rtol=0)
