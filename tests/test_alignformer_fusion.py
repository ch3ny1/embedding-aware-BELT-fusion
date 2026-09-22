import math

import pytest
import torch

from embedding_aware_belt_fusion.alignformer.boxes import AgentDetections
from embedding_aware_belt_fusion.alignformer.fusion import (
    average_precision,
    correct_detections,
    late_fuse,
)


def _detections(boxes, scores=None):
    count = boxes.shape[0]
    return AgentDetections(
        boxes=boxes,
        scores=torch.ones(count) if scores is None else scores,
        corners=torch.zeros(count, 8, 3),
        gt_ids=[None] * count,
        features=torch.zeros(1, 4, 4),
    )


def test_correction_translates_and_rotates_boxes():
    detections = _detections(torch.tensor([[1.0, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0]]))

    corrected = correct_detections(
        detections, torch.tensor(math.pi / 2), torch.tensor([0.0, 1.0])
    )

    assert corrected.boxes[0, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert corrected.boxes[0, 1].item() == pytest.approx(2.0, abs=1e-6)
    assert corrected.boxes[0, 6].item() == pytest.approx(math.pi / 2, abs=1e-6)


def test_correction_does_not_mutate_its_input():
    original = torch.tensor([[1.0, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0]])
    detections = _detections(original.clone())

    correct_detections(detections, torch.tensor(0.5), torch.tensor([3.0, 3.0]))

    assert torch.allclose(detections.boxes, original)


def test_identity_correction_is_a_no_op():
    detections = _detections(torch.randn(4, 7))

    corrected = correct_detections(detections, torch.tensor(0.0), torch.zeros(2))

    assert torch.allclose(corrected.boxes, detections.boxes, atol=1e-6)


def test_average_precision_is_one_for_perfect_predictions():
    boxes = torch.tensor([
        [0.0, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0],
        [20.0, 5.0, 0.0, 1.5, 2.0, 4.0, 0.3],
    ])

    result = average_precision(
        [(boxes, torch.ones(2))], [boxes], iou_threshold=0.7, global_sort=True
    )

    assert result == pytest.approx(1.0, abs=1e-6)


def test_average_precision_is_zero_when_nothing_overlaps():
    predicted = torch.tensor([[0.0, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0]])
    truth = torch.tensor([[80.0, 30.0, 0.0, 1.5, 2.0, 4.0, 0.0]])

    result = average_precision(
        [(predicted, torch.ones(1))], [truth], iou_threshold=0.7, global_sort=True
    )

    assert result == pytest.approx(0.0, abs=1e-6)


def test_global_sort_and_frame_order_provably_differ():
    """A perfect implementation that ignores ``global_sort`` would pass every
    other test here (they either use a single frame, or an all-zero result).
    This constructs two frames whose scores interleave -- frame 1's true
    positive (0.9) and false positive (0.2) straddle frame 2's true positive
    (0.5) -- so frame-order accumulation and global confidence sorting visit
    hits/misses in a genuinely different order and must produce different AP
    values. Expected numbers are hand-computed from the VOC PR-curve formula.
    """
    frame1_boxes = torch.tensor([
        [0.0, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0],
        [20.0, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0],
    ])
    frame1_scores = torch.tensor([0.9, 0.2])
    frame1_truth = torch.tensor([[0.0, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0]])

    frame2_boxes = torch.tensor([[80.0, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0]])
    frame2_scores = torch.tensor([0.5])
    frame2_truth = torch.tensor([[80.0, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0]])

    predictions = [(frame1_boxes, frame1_scores), (frame2_boxes, frame2_scores)]
    ground_truth = [frame1_truth, frame2_truth]

    global_ap = average_precision(predictions, ground_truth, iou_threshold=0.7, global_sort=True)
    frame_order_ap = average_precision(predictions, ground_truth, iou_threshold=0.7, global_sort=False)

    assert global_ap == pytest.approx(1.0, abs=1e-6)
    assert frame_order_ap == pytest.approx(5.0 / 6.0, abs=1e-4)
    assert global_ap != pytest.approx(frame_order_ap, abs=1e-3)


def test_average_precision_envelope_matters_when_precision_would_otherwise_rise():
    """Negative control for the VOC monotonic-envelope step.

    Single frame, scores already descending (0.9, 0.8, 0.7, 0.6) so
    ``global_sort`` cannot be responsible for any difference here -- this
    isolates the envelope step specifically. Matching gives TP, FP, TP, TP:
    raw precision is [1, 0.5, 0.667, 0.75], which *rises* from 0.5 to 0.667 --
    a real implementation must replace 0.667 with the envelope value 0.75
    before integrating, or AP comes out lower than the correct value. Expected
    numbers are hand-computed from the VOC PR-curve formula.
    """
    predicted = torch.tensor([
        [0.0, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0],      # TP: matches truth[0]
        [200.0, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0],    # FP: matches nothing
        [40.0, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0],     # TP: matches truth[1]
        [80.0, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0],     # TP: matches truth[2]
    ])
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    truth = torch.tensor([
        [0.0, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0],
        [40.0, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0],
        [80.0, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0],
    ])

    result = average_precision(
        [(predicted, scores)], [truth], iou_threshold=0.7, global_sort=True
    )

    assert result == pytest.approx(5.0 / 6.0, abs=1e-4)


def test_late_fuse_suppresses_overlapping_cross_agent_duplicates():
    # Agent A and B both see the same object with slightly jittered boxes;
    # a real per-agent NMS would not have deduplicated this (they come from
    # different agents' own detectors), so cross-agent NMS in late_fuse must.
    agent_a = _detections(
        torch.tensor([[0.0, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0]]), torch.tensor([0.9])
    )
    agent_b = _detections(
        torch.tensor([[0.1, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0]]), torch.tensor([0.5])
    )
    agent_c = _detections(
        torch.tensor([[50.0, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0]]), torch.tensor([0.7])
    )

    boxes, scores = late_fuse([agent_a, agent_b, agent_c], nms_threshold=0.15)

    assert boxes.shape[0] == 2
    kept_scores = sorted(scores.tolist())
    assert kept_scores == pytest.approx([0.7, 0.9], abs=1e-6)


def test_late_fuse_returns_empty_when_every_agent_has_no_detections():
    empty = _detections(torch.zeros(0, 7), torch.zeros(0))

    boxes, scores = late_fuse([empty, empty], nms_threshold=0.15)

    assert boxes.shape == (0, 7)
    assert scores.shape == (0,)
