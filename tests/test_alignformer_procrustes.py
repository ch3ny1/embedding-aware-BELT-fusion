import math

import pytest
import torch

from embedding_aware_belt_fusion.alignformer.procrustes import (
    MIN_MATCH_MASS,
    augment_with_heading,
    weighted_se2_kabsch,
)


def _apply(psi, t, q):
    cos, sin = math.cos(psi), math.sin(psi)
    rotation = torch.tensor([[cos, -sin], [sin, cos]])
    return q @ rotation.T + torch.tensor(t)


def test_recovers_a_known_transform_exactly():
    # Arrange
    q = torch.tensor([[[0.0, 0.0], [5.0, 1.0], [2.0, -3.0], [-4.0, 2.0]]])
    true_psi, true_t = 0.3, (1.5, -2.0)
    p = _apply(true_psi, true_t, q[0]).unsqueeze(0)
    w = torch.ones(1, 4)

    # Act
    psi, t = weighted_se2_kabsch(p, q, w)

    # Assert
    assert psi.item() == pytest.approx(true_psi, abs=1e-5)
    assert t[0, 0].item() == pytest.approx(true_t[0], abs=1e-5)
    assert t[0, 1].item() == pytest.approx(true_t[1], abs=1e-5)


def test_zero_weight_points_are_ignored():
    q = torch.tensor([[[0.0, 0.0], [4.0, 0.0], [999.0, 999.0]]])
    true_psi, true_t = -0.2, (0.5, 0.25)
    p = _apply(true_psi, true_t, q[0]).unsqueeze(0)
    p[0, 2] = torch.tensor([-50.0, 70.0])  # an outlier, down-weighted to zero
    w = torch.tensor([[1.0, 1.0, 0.0]])

    psi, t = weighted_se2_kabsch(p, q, w)

    assert psi.item() == pytest.approx(true_psi, abs=1e-5)


def test_a_single_object_with_heading_determines_full_se2():
    # A lone centre is rank-deficient for yaw. Adding the heading virtual point
    # makes one matched object sufficient - the spec's low-overlap claim.
    centres_q = torch.tensor([[[3.0, -1.0]]])
    yaws_q = torch.tensor([[0.4]])
    true_psi, true_t = 0.25, (-1.0, 2.0)

    centres_p = _apply(true_psi, true_t, centres_q[0]).unsqueeze(0)
    yaws_p = yaws_q + true_psi

    q = augment_with_heading(centres_q, yaws_q, lam=2.0)
    p = augment_with_heading(centres_p, yaws_p, lam=2.0)
    w = torch.ones(1, q.shape[1])

    psi, t = weighted_se2_kabsch(p, q, w)

    assert psi.item() == pytest.approx(true_psi, abs=1e-5)
    assert t[0, 0].item() == pytest.approx(true_t[0], abs=1e-5)


def test_degenerate_input_returns_identity_without_nan():
    p = torch.zeros(1, 3, 2)
    q = torch.zeros(1, 3, 2)
    w = torch.zeros(1, 3)

    psi, t = weighted_se2_kabsch(p, q, w)

    assert torch.isfinite(psi).all() and torch.isfinite(t).all()
    assert psi.item() == pytest.approx(0.0)
    assert torch.allclose(t, torch.zeros_like(t))


def test_gradients_flow_to_weights_and_are_finite():
    q = torch.tensor([[[0.0, 0.0], [5.0, 1.0], [2.0, -3.0]]])
    p = _apply(0.3, (1.0, 1.0), q[0]).unsqueeze(0)
    w = torch.full((1, 3), 0.5, requires_grad=True)

    psi, t = weighted_se2_kabsch(p, q, w)
    (psi.sum() + t.sum()).backward()

    assert w.grad is not None
    assert torch.isfinite(w.grad).all()


def test_degenerate_gradients_are_finite_not_nan():
    # atan2(0, 0) has undefined gradient; the guard must prevent NaN reaching
    # the optimizer, which would silently poison a whole training run.
    p = torch.zeros(1, 2, 2)
    q = torch.zeros(1, 2, 2)
    w = torch.zeros(1, 2, requires_grad=True)

    psi, t = weighted_se2_kabsch(p, q, w)
    (psi.sum() + t.sum()).backward()

    assert torch.isfinite(w.grad).all()


def test_min_match_mass_is_one_effective_object():
    assert MIN_MATCH_MASS == 1.0
