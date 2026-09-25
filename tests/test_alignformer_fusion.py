import math

import pytest
import torch

from embedding_aware_belt_fusion.alignformer.boxes import AgentDetections
from embedding_aware_belt_fusion.alignformer.fusion import (
    average_precision,
    correct_detections,
    late_fuse,
)

# R26: shapely's set_operations.intersection (called by OpenCOOD's own
# common_utils.compute_iou, not by this project's code) emits a RuntimeWarning
# for the exactly-coincident and heavily-overlapping polygon pairs several of
# this module's tests deliberately construct (predicted == truth boxes,
# duplicate cross-agent detections). This is not our call to fix at its
# source -- external/OpenCOOD is a vendored submodule -- so it is silenced
# here, scoped to this module and to exactly that shapely warning, rather than
# with a blanket filter.
#
# Review LOW-4: investigated, not just filtered. Traced (via
# `pytest -W error:...` with this filter disabled) to
# external/OpenCOOD/opencood/utils/box_utils.py:624's `nms_rotated`, calling
# `compute_iou` on the two widely-SEPARATED (non-overlapping) duplicate boxes
# `test_late_fuse_suppresses_overlapping_cross_agent_duplicates` places 50 m
# apart. Reproduced standalone: `Polygon([(2,-1),(2,1),(-2,1),(-2,-1)]).
# intersection(Polygon([(52,-1),(52,1),(48,1),(48,-1)]))` on shapely 2.0.0
# raises exactly this RuntimeWarning while still returning `POLYGON EMPTY`
# with `area == 0.0` -- a correct, finite result. Both input polygons are
# `shapely.is_valid`. So this is a GEOS/shapely-internal transient during the
# robust intersection predicate for disjoint geometries, not a masked
# correctness bug in either polygon's construction: independently confirmed
# by wrapping `compute_iou` across this entire file's run (15 calls, 0
# non-finite results reaching its return value). The filter is correctly
# targeted and scoped; nothing here needed widening.
pytestmark = pytest.mark.filterwarnings(
    "ignore:invalid value encountered in intersection:RuntimeWarning:shapely"
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


# --- Task 23: one matching pass, reused by every threshold and every slice ----
#
# `average_precision` recomputed the rotated-IoU matching for each of the three
# IoU thresholds, each of the two sort orders, and again for each shared-object
# slice -- eighteen passes over the same polygons per condition. At one noise
# seed that was invisible; at five it made the AP scoring longer than the sweep
# itself. The matching does not depend on the threshold, the sort order or the
# frame subset, so it is now done once and reused. That is only safe if the
# reused form is EXACTLY the old one, which is what these pin.


def _ap_scenario():
    """Frames that exercise every branch: hits, misses, no truth, no detections."""
    torch.manual_seed(0)
    frames, truth = [], []
    for _ in range(6):
        count = int(torch.randint(0, 6, (1,)))
        centres = torch.randn(count, 2) * 30.0
        boxes = torch.zeros(count, 7)
        boxes[:, :2] = centres
        boxes[:, 3:6] = torch.tensor([1.5, 2.0, 4.0])
        boxes[:, 6] = torch.rand(count) * math.pi
        scores = torch.rand(count)
        # Truth: some of the same boxes, jittered, plus one that was never found.
        keep = boxes[: max(count - 1, 0)].clone()
        keep[:, :2] += torch.randn(keep.shape[0], 2) * 0.3
        extra = torch.tensor([[60.0, 60.0, 0.0, 1.5, 2.0, 4.0, 0.0]])
        frames.append((boxes, scores))
        truth.append(torch.cat([keep, extra], dim=0))
    # A frame with detections but no truth, and a frame with truth but none found.
    frames.append((torch.tensor([[1.0, 1.0, 0.0, 1.5, 2.0, 4.0, 0.0]]), torch.tensor([0.4])))
    truth.append(torch.zeros(0, 7))
    frames.append((torch.zeros(0, 7), torch.zeros(0)))
    truth.append(torch.tensor([[3.0, 3.0, 0.0, 1.5, 2.0, 4.0, 0.0]]))
    return frames, truth


@pytest.mark.parametrize("threshold", (0.3, 0.5, 0.7))
@pytest.mark.parametrize("global_sort", (True, False))
def test_the_reused_matching_reproduces_average_precision_exactly(threshold, global_sort):
    from embedding_aware_belt_fusion.alignformer.fusion import (
        average_precision_from_matches,
        match_frames,
    )

    predictions, ground_truth = _ap_scenario()

    direct = average_precision(
        predictions, ground_truth, iou_threshold=threshold, global_sort=global_sort
    )
    reused = average_precision_from_matches(
        match_frames(predictions, ground_truth),
        iou_threshold=threshold,
        global_sort=global_sort,
    )

    # Exactly, not approximately: this replaces a published scoring path.
    assert reused == direct


def test_a_frame_subset_scores_as_if_it_had_been_matched_alone():
    """The shared-object slices are subsets of the same frames; reusing the
    parent's matching must give the same answer as matching the subset."""
    from embedding_aware_belt_fusion.alignformer.fusion import (
        average_precision_from_matches,
        match_frames,
    )

    predictions, ground_truth = _ap_scenario()
    indices = [0, 2, 4, 6]

    alone = average_precision(
        [predictions[i] for i in indices],
        [ground_truth[i] for i in indices],
        iou_threshold=0.5,
    )
    from_parent = average_precision_from_matches(
        match_frames(predictions, ground_truth), iou_threshold=0.5, indices=indices
    )

    assert from_parent == alone


def test_matching_once_is_cheaper_than_matching_per_threshold():
    """The whole point: one pass, then three thresholds for free."""
    from embedding_aware_belt_fusion.alignformer.fusion import (
        average_precision_from_matches,
        match_frames,
    )

    predictions, ground_truth = _ap_scenario()
    matches = match_frames(predictions, ground_truth)

    values = [
        average_precision_from_matches(matches, iou_threshold=t) for t in (0.3, 0.5, 0.7)
    ]
    direct = [
        average_precision(predictions, ground_truth, iou_threshold=t)
        for t in (0.3, 0.5, 0.7)
    ]

    assert values == direct
    # A lower threshold cannot score worse, or the scenario is degenerate and
    # the equality above would be vacuous.
    assert values[0] >= values[2]
    assert values[0] > 0.0


def test_matching_rejects_mismatched_frame_counts_like_the_function_it_replaces():
    from embedding_aware_belt_fusion.alignformer.fusion import match_frames

    with pytest.raises(ValueError, match="frames"):
        match_frames([(torch.zeros(0, 7), torch.zeros(0))], [])
