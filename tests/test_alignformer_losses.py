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


def test_corner_loss_matches_a_hand_computed_displacement():
    # hwl order: [x, y, z, h, w, l, yaw] -> half_length=2 (l=4), half_width=1
    # (w=2), yaw=0, so bev_corners gives (2,1), (2,-1), (-2,-1), (-2,1).
    #
    # psi_pred=0.5, t_pred=(0,0) vs psi_true=0.0, t_true=(1.0, 0.0):
    #   predicted_i = R(0.5) @ corner_i         (R(0.5) uses cos/sin(0.5))
    #   target_i    = corner_i + (1, 0)
    # cos(0.5)=0.8775826, sin(0.5)=0.4794255. Per corner, |predicted-target|_1:
    #   (2, 1)  -> predicted=(1.275740, 1.836435), target=(3, 1)
    #              diff=(-1.724260, 0.836435)   -> 2.560695
    #   (2,-1)  -> predicted=(2.234592, 0.081269), target=(3,-1)
    #              diff=(-0.765408, 1.081269)   -> 1.846677
    #  (-2,-1)  -> predicted=(-1.275740,-1.836435), target=(-1,-1)
    #              diff=(-0.275740,-0.836435)   -> 1.112175
    #  (-2, 1)  -> predicted=(-2.234592,-0.081269), target=(-1, 1)
    #              diff=(-1.234592,-1.081269)   -> 2.315861
    # Sum = 7.835408 (matches 7.835404 to float rounding); one real object,
    # so weights.sum() == 1 and the loss equals that sum directly.
    boxes = torch.tensor([[[0.0, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0]]])
    mask = torch.ones(1, 1, dtype=torch.bool)
    psi_pred = torch.tensor([0.5])
    t_pred = torch.zeros(1, 2)
    psi_true = torch.zeros(1)
    t_true = torch.tensor([[1.0, 0.0]])

    loss = corner_loss(boxes, psi_pred, t_pred, psi_true, t_true, mask)

    assert loss.item() == pytest.approx(7.835404, abs=1e-4)


def test_corner_loss_grows_with_yaw_error():
    # The box sits at the origin (not far from it, as the previous version of
    # this test had it) so its own extents -- not distance from the rotation
    # origin -- dominate the corner displacement. A pure yaw-only mismatch
    # with no translation error is, by the rotation-difference algebra,
    # provably blind to a width/length swap (the total corner displacement is
    # exactly symmetric under swapping which extent is which). A fixed
    # translation mismatch (t_true = (0.5, 0)) breaks that symmetry, so the
    # loss now depends on which extent is width and which is length.
    #
    # Hand-derived for width=2, length=4, t_true=(0.5, 0), t_pred=(0, 0),
    # psi_true=0 (full corner-by-corner working in the report):
    #   psi_pred = pi/6 (cos=sqrt(3)/2, sin=1/2)  -> loss = 6.535898...
    #   psi_pred = pi/3 (cos=1/2, sin=sqrt(3)/2)  -> loss = 11.660254...
    # A width/length swap gives 6.0 and 10.928203... instead -- still growing,
    # so ordering alone would not catch it, but the exact values do.
    boxes = torch.tensor([[[0.0, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0]]])
    mask = torch.ones(1, 1, dtype=torch.bool)
    zero_t = torch.zeros(1, 2)
    offset_t = torch.tensor([[0.5, 0.0]])
    truth = torch.zeros(1)

    small = corner_loss(boxes, torch.tensor([math.pi / 6]), zero_t, truth, offset_t, mask)
    large = corner_loss(boxes, torch.tensor([math.pi / 3]), zero_t, truth, offset_t, mask)

    assert small.item() == pytest.approx(6.535898, abs=1e-4)
    assert large.item() == pytest.approx(11.660254, abs=1e-4)
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
    # Non-square (M=2 ego objects, N=1 cav object) so the two dustbin target
    # indices differ: the ego dustbin column is cav_count == 1, the cav
    # dustbin row is ego_count == 2. With M == N those two indices coincide
    # and an ego/cav dustbin-target swap is byte-indistinguishable; here it
    # is not.
    log_assignment = torch.full((1, 3, 2), -20.0)
    log_assignment[0, 0, 1] = -1.0  # ego object 0 -> dustbin column (1)
    log_assignment[0, 1, 1] = -3.0  # ego object 1 -> dustbin column (1)
    log_assignment[0, 2, 0] = -2.0  # dustbin row (2) -> cav object 0
    ego_match = torch.tensor([[-1, -1]])
    cav_match = torch.tensor([[-1]])

    loss = match_nll(log_assignment, ego_match, cav_match)

    # ego_terms = [-1.0, -3.0] -> mean -2.0; cav_terms = [-2.0] -> mean -2.0
    # loss = -((-2.0) + (-2.0)) / 2 = 2.0
    assert loss.item() == pytest.approx(2.0, abs=1e-6)
