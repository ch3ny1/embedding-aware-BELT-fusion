import math

import pytest
import torch

from embedding_aware_belt_fusion.alignformer.metrics import (
    chance_top1_sums,
    dustbin_mass_fraction,
    dustbin_mass_sums,
    nearest_centre_top1_counts,
    top1_accuracy,
    top1_counts,
    translation_mae,
    yaw_mae_deg,
)


def test_top1_counts_only_objects_that_have_a_counterpart():
    log_assignment = torch.full((1, 3, 3), -10.0)
    log_assignment[0, 0, 0] = 0.0  # correct
    log_assignment[0, 1, 1] = 0.0  # this ego object has no true match
    ego_match = torch.tensor([[0, -1]])
    ego_mask = torch.ones(1, 2, dtype=torch.bool)

    assert top1_accuracy(log_assignment, ego_match, ego_mask) == pytest.approx(1.0)


def test_top1_is_zero_when_every_match_is_wrong():
    log_assignment = torch.full((1, 3, 3), -10.0)
    log_assignment[0, 0, 1] = 0.0
    ego_match = torch.tensor([[0, -1]])
    ego_mask = torch.ones(1, 2, dtype=torch.bool)

    assert top1_accuracy(log_assignment, ego_match, ego_mask) == pytest.approx(0.0)


def test_top1_ignores_padded_objects():
    log_assignment = torch.full((1, 3, 3), -10.0)
    log_assignment[0, 0, 0] = 0.0
    ego_match = torch.tensor([[0, 1]])
    ego_mask = torch.tensor([[True, False]])

    assert top1_accuracy(log_assignment, ego_match, ego_mask) == pytest.approx(1.0)


def test_translation_mae_is_the_mean_euclidean_residual():
    result = translation_mae(
        torch.zeros(2), torch.tensor([[3.0, 4.0], [0.0, 0.0]]),
        torch.zeros(2), torch.zeros(2, 2),
    )

    assert result == pytest.approx(2.5)


def test_yaw_mae_wraps_across_pi():
    result = yaw_mae_deg(torch.tensor([math.pi - 0.01]), torch.tensor([-math.pi + 0.01]))

    assert result == pytest.approx(math.degrees(0.02), abs=1e-4)


# --- additions beyond the task brief's five ---------------------------------
# The gate is measured over a whole split, not one batch, so the accumulator
# needs counts rather than per-batch means; and the failure diagnosis the task
# asks for needs the dustbin mass.


def test_top1_counts_returns_correct_and_countable_totals():
    log_assignment = torch.full((2, 3, 3), -10.0)
    log_assignment[0, 0, 0] = 0.0  # correct
    log_assignment[1, 0, 1] = 0.0  # wrong: true match is column 0
    log_assignment[1, 1, 0] = 0.0  # correct
    ego_match = torch.tensor([[0, -1], [0, 0]])
    ego_mask = torch.ones(2, 2, dtype=torch.bool)

    # Sample 1's two ego objects both claim CAV column 0; that is not a legal
    # one-to-one assignment, but top-1 is a per-row metric and scores each row
    # independently, so it is still well defined.
    assert top1_counts(log_assignment, ego_match, ego_mask) == (2, 3)


def test_top1_never_averages_batches_of_different_size():
    # Two batches, 1/1 then 0/3: the count-weighted answer is 0.25, whereas
    # averaging the two per-batch accuracies would give 0.5.
    first = torch.full((1, 2, 3), -10.0)
    first[0, 0, 0] = 0.0
    second = torch.full((3, 2, 3), -10.0)
    second[:, 0, 1] = 0.0  # every row's argmax is column 1, the wrong column

    correct, total = top1_counts(
        first, torch.tensor([[0]]), torch.ones(1, 1, dtype=torch.bool)
    )
    more_correct, more_total = top1_counts(
        second, torch.tensor([[0], [0], [0]]), torch.ones(3, 1, dtype=torch.bool)
    )

    assert (correct + more_correct) / (total + more_total) == pytest.approx(0.25)


def test_top1_is_nan_when_nothing_is_countable():
    log_assignment = torch.full((1, 3, 3), -10.0)
    ego_match = torch.tensor([[-1, -1]])
    ego_mask = torch.ones(1, 2, dtype=torch.bool)

    assert math.isnan(top1_accuracy(log_assignment, ego_match, ego_mask))


def test_top1_can_score_the_dustbin_as_a_competing_column():
    # Dustbin (last) column wins outright; excluding it, column 0 wins.
    log_assignment = torch.full((1, 3, 3), -10.0)
    log_assignment[0, 0, 0] = 0.0
    log_assignment[0, 0, 2] = 5.0
    ego_match = torch.tensor([[0, -1]])
    ego_mask = torch.ones(1, 2, dtype=torch.bool)

    assert top1_accuracy(log_assignment, ego_match, ego_mask) == pytest.approx(1.0)
    assert top1_accuracy(
        log_assignment, ego_match, ego_mask, include_dustbin=True
    ) == pytest.approx(0.0)


def test_dustbin_mass_fraction_averages_real_rows_only():
    # Row 0 puts all its mass on the dustbin, row 1 none; row 2 is padded and
    # must not count even though it too is all-dustbin.
    log_assignment = torch.full((1, 4, 3), -100.0)
    log_assignment[0, 0, 2] = 0.0
    log_assignment[0, 1, 0] = 0.0
    log_assignment[0, 2, 2] = 0.0
    ego_mask = torch.tensor([[True, True, False]])

    assert dustbin_mass_fraction(log_assignment, ego_mask) == pytest.approx(0.5, abs=1e-4)


def test_metrics_reject_a_log_assignment_that_does_not_match_the_masks():
    log_assignment = torch.full((1, 3, 3), -10.0)

    with pytest.raises(ValueError):
        top1_accuracy(
            log_assignment, torch.tensor([[0, -1, 0]]), torch.ones(1, 3, dtype=torch.bool)
        )


def test_translation_mae_rejects_mismatched_pose_shapes():
    with pytest.raises(ValueError):
        translation_mae(
            torch.zeros(2), torch.zeros(2, 2), torch.zeros(3), torch.zeros(3, 2)
        )


def test_dustbin_mass_sums_accumulate_across_batches():
    first = torch.full((1, 3, 3), -100.0)
    first[0, 0, 2] = 0.0  # all dustbin
    first[0, 1, 0] = 0.0  # no dustbin
    second = torch.full((1, 3, 3), -100.0)
    second[0, 0, 2] = 0.0
    second[0, 1, 2] = 0.0  # both rows all dustbin

    mass, rows = dustbin_mass_sums(first, torch.ones(1, 2, dtype=torch.bool))
    more_mass, more_rows = dustbin_mass_sums(second, torch.ones(1, 2, dtype=torch.bool))

    assert (mass + more_mass) / (rows + more_rows) == pytest.approx(0.75, abs=1e-4)


def test_chance_is_one_over_the_number_of_cav_objects():
    # Two ego objects with a counterpart against 4 real CAV columns, and one
    # ego object with no counterpart, which must not count.
    ego_match = torch.tensor([[0, 1, -1]])
    ego_mask = torch.ones(1, 3, dtype=torch.bool)
    cav_mask = torch.tensor([[True, True, True, True, False]])

    total, countable = chance_top1_sums(ego_match, ego_mask, cav_mask)

    assert countable == 2
    assert total / countable == pytest.approx(0.25)


def test_nearest_centre_matches_the_closest_cav_box():
    # Ego object at the origin; the closest CAV box is column 1, which is also
    # the true match, so this scores 1/1.
    ego_boxes = torch.zeros(1, 1, 7)
    cav_boxes = torch.zeros(1, 3, 7)
    cav_boxes[0, 0, :2] = torch.tensor([10.0, 0.0])
    cav_boxes[0, 1, :2] = torch.tensor([0.3, 0.0])
    cav_boxes[0, 2, :2] = torch.tensor([-8.0, 0.0])

    assert nearest_centre_top1_counts(
        ego_boxes,
        cav_boxes,
        torch.tensor([[1]]),
        torch.ones(1, 1, dtype=torch.bool),
        torch.ones(1, 3, dtype=torch.bool),
    ) == (1, 1)


def test_nearest_centre_never_picks_a_padded_cav_object():
    # The nearest box in raw distance is the padded column 0 at the origin;
    # masking it out leaves column 1, the true match.
    ego_boxes = torch.zeros(1, 1, 7)
    cav_boxes = torch.zeros(1, 2, 7)
    cav_boxes[0, 1, :2] = torch.tensor([4.0, 0.0])

    assert nearest_centre_top1_counts(
        ego_boxes,
        cav_boxes,
        torch.tensor([[1]]),
        torch.ones(1, 1, dtype=torch.bool),
        torch.tensor([[False, True]]),
    ) == (1, 1)


def test_the_averaging_limited_prediction_reproduces_the_pose_floor_decomposition():
    # docs/alignformer_pose_floor.md section 1: a per-axis cross-agent centre
    # disagreement of 0.2198 m over 10.92 matched objects predicts a 0.083 m
    # translation MAE from averaging alone. That prediction is what separates
    # "the detector disagrees more on this split" from "the estimator is worse
    # on this split", so it has to be one function, not a number re-derived by
    # hand in each diagnostic.
    from embedding_aware_belt_fusion.alignformer.metrics import (
        averaging_limited_translation_mae_m,
    )

    # Act
    predicted = averaging_limited_translation_mae_m(0.2198, 10.92)

    # Assert
    assert predicted == pytest.approx(0.083, abs=5e-4)


def test_the_averaging_limited_prediction_falls_as_more_objects_are_matched():
    from embedding_aware_belt_fusion.alignformer.metrics import (
        averaging_limited_translation_mae_m,
    )

    # Four times the objects halves the prediction: it is a 1/sqrt(n) law.
    assert averaging_limited_translation_mae_m(0.2, 4.0) == pytest.approx(
        averaging_limited_translation_mae_m(0.2, 16.0) * 2.0
    )


def test_the_averaging_limited_prediction_rejects_an_empty_correspondence_set():
    from embedding_aware_belt_fusion.alignformer.metrics import (
        averaging_limited_translation_mae_m,
    )

    with pytest.raises(ValueError, match="matched objects"):
        averaging_limited_translation_mae_m(0.2, 0.0)
