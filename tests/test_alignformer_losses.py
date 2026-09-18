import math

import pytest
import torch

from embedding_aware_belt_fusion.alignformer.losses import (
    apply_se2,
    bev_corners,
    corner_loss,
    match_nll,
)


def test_bev_corners_of_an_axis_aligned_box():
    # hwl order: [x, y, z, h, w, l, yaw], so width = 2, length = 4.
    boxes = torch.tensor([[[0.0, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0]]])

    corners = bev_corners(boxes)

    assert corners.shape == (1, 1, 4, 2)
    assert corners[0, 0, :, 0].abs().max().item() == pytest.approx(2.0)
    assert corners[0, 0, :, 1].abs().max().item() == pytest.approx(1.0)


def test_corner_loss_is_zero_for_a_perfect_prediction():
    boxes = torch.randn(2, 5, 7)
    psi = torch.tensor([0.1, -0.3])
    t = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
    mask = torch.ones(2, 5, dtype=torch.bool)

    loss = corner_loss(boxes, psi, t, psi, t, mask)

    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_corner_loss_grows_with_yaw_error():
    boxes = torch.tensor([[[30.0, 10.0, 0.0, 1.5, 2.0, 4.0, 0.0]]])
    mask = torch.ones(1, 1, dtype=torch.bool)
    zero_t = torch.zeros(1, 2)
    truth = torch.zeros(1)

    small = corner_loss(boxes, torch.tensor([0.01]), zero_t, truth, zero_t, mask)
    large = corner_loss(boxes, torch.tensor([0.10]), zero_t, truth, zero_t, mask)

    assert large.item() > small.item()


def test_corner_loss_ignores_padded_objects():
    boxes = torch.zeros(1, 3, 7)
    boxes[0, 2] = torch.tensor([500.0, 500.0, 0.0, 1.5, 2.0, 4.0, 0.0])
    mask = torch.tensor([[True, True, False]])

    loss = corner_loss(
        boxes, torch.tensor([0.2]), torch.zeros(1, 2),
        torch.zeros(1), torch.zeros(1, 2), mask,
    )

    # The padded far-away object would dominate if it were counted.
    assert loss.item() < 1.0


def test_apply_se2_rotates_and_translates():
    points = torch.tensor([[[1.0, 0.0]]])

    moved = apply_se2(points, torch.tensor([math.pi / 2]), torch.tensor([[0.0, 1.0]]))

    assert moved[0, 0, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert moved[0, 0, 1].item() == pytest.approx(2.0, abs=1e-6)


def test_match_nll_is_low_for_a_confident_correct_assignment():
    log_assignment = torch.full((1, 3, 3), -20.0)
    log_assignment[0, 0, 1] = 0.0
    log_assignment[0, 1, 0] = 0.0
    ego_match = torch.tensor([[1, 0]])
    cav_match = torch.tensor([[1, 0]])

    loss = match_nll(log_assignment, ego_match, cav_match)

    assert loss.item() < 0.1


def test_match_nll_charges_unmatched_objects_to_the_dustbin():
    log_assignment = torch.full((1, 3, 3), -20.0)
    log_assignment[0, 0, 2] = 0.0  # ego object 0 -> dustbin
    ego_match = torch.tensor([[-1, -1]])
    cav_match = torch.tensor([[-1, -1]])

    loss = match_nll(log_assignment, ego_match, cav_match)

    assert torch.isfinite(loss)
