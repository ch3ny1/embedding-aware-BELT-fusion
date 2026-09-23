"""SE(2) box correction, cross-agent late fusion, and dataset-level AP.

``correct_detections`` reuses ``losses.apply_se2`` to move one agent's boxes
into another agent's frame. For the clean P0 gate the (psi, t) passed in is
the *exact* CAV-to-ego pose transform (no injected error, since there is
nothing to correct yet); AlignFormer's later noisy-pose tasks feed this same
function a learned or estimated correction instead. The function is generic
to both uses.

``late_fuse`` composes already-corrected ``AgentDetections`` (all expressed in
the same frame) into one fused set via OpenCOOD's rotated NMS -- the final
cross-agent step of the official ``VoxelPostprocessor.post_process``. The
task brief describes this step as "concatenate and NMS"; a final ego-frame
range filter (``box_utils.get_mask_for_boxes_within_range_torch``) is also
applied here because every OpenCOOD reference implementation of this step
does it after NMS (``VoxelPostprocessor.post_process``,
``evaluation/belt_fusion.py::_postprocess_fused_boxes``) and it is not
redundant with ``boxes.detect_agent``'s own local filtering: a CAV's own
detection range extends tens of metres past ego's evaluation range, so
skipping this would leave far-out-of-range boxes as uncontested false
positives in every frame's precision-recall curve. See task-12-report.md.

``average_precision`` mirrors ``opencood.utils.eval_utils.calculate_ap``
(``caluclate_tp_fp`` + ``calculate_ap``) exactly: the same greedy
per-frame highest-confidence-first matching, the same global-vs-frame-order
sort switch, and the same VOC monotonic-envelope integration. It is given in
full by the task brief; the P0 gate depends on this exact implementation.
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Sequence, Tuple

import torch
from torch import Tensor

from embedding_aware_belt_fusion.alignformer.boxes import BOX_YAW, AgentDetections
from embedding_aware_belt_fusion.alignformer.losses import apply_se2

# Box layout is OpenCOOD 'hwl': [x, y, z, h, w, l, yaw]. Every AgentDetections
# in this project commits to this order (see boxes.AgentDetections). R26: the
# yaw index itself is defined once, in boxes.py beside AgentDetections, and
# imported here rather than re-declared.
_ORDER = "hwl"


def correct_boxes(boxes: Tensor, psi: Tensor, t: Tensor) -> Tensor:
    """Return a **new** ``(N, 7)`` box tensor moved by SE(2) ``(psi, t)``.

    Centres are transformed by ``losses.apply_se2``; yaw is incremented by
    ``psi``. ``boxes`` is never mutated: a fresh tensor is built via
    ``.clone()`` before being written into.

    Separate from :func:`correct_detections` because the object sets AlignFormer
    consumes are bare box tensors (truncated to the trunk's token budget, with
    no corners or gt ids attached), and moving them by a second, parallel
    implementation is exactly how a projection convention drifts between the
    model's input and the boxes that are actually fused.
    """
    psi = psi.reshape(1).to(boxes.dtype)
    t = t.reshape(1, 2).to(boxes.dtype)

    centers = boxes[:, :2].unsqueeze(0)  # (1, N, 2), the batch dim apply_se2 expects
    corrected = boxes.clone()
    corrected[:, :2] = apply_se2(centers, psi, t).squeeze(0)
    corrected[:, BOX_YAW] = boxes[:, BOX_YAW] + psi
    return corrected


def correct_detections(detections: AgentDetections, psi: Tensor, t: Tensor) -> AgentDetections:
    """Return a **new** ``AgentDetections`` with boxes moved by SE(2) ``(psi, t)``.

    ``detections`` is never mutated: :func:`correct_boxes` returns a fresh boxes
    tensor and the new object is produced with ``dataclasses.replace``.
    """
    return replace(detections, boxes=correct_boxes(detections.boxes, psi, t))


def late_fuse(
    detections_by_agent: Sequence[AgentDetections], nms_threshold: float
) -> Tuple[Tensor, Tensor]:
    """Concatenate corrected per-agent boxes and rotated-NMS them into one set.

    Every entry of ``detections_by_agent`` must already be expressed in the
    same (typically ego) frame -- e.g. via :func:`correct_detections`.
    """
    from opencood.utils import box_utils

    nonempty = [d for d in detections_by_agent if len(d) > 0]
    if not nonempty:
        template = detections_by_agent[0].boxes if detections_by_agent else torch.empty(0, 7)
        return template.new_empty((0, 7)), template.new_empty((0,))

    all_boxes = torch.cat([d.boxes for d in nonempty], dim=0)
    all_scores = torch.cat([d.scores for d in nonempty], dim=0)

    corners = box_utils.boxes_to_corners_3d(all_boxes, order=_ORDER)
    keep = box_utils.nms_rotated(corners, all_scores, nms_threshold)
    keep = torch.as_tensor(keep, dtype=torch.long, device=all_boxes.device)
    corners, all_boxes, all_scores = corners[keep], all_boxes[keep], all_scores[keep]

    in_range = box_utils.get_mask_for_boxes_within_range_torch(corners)
    return all_boxes[in_range], all_scores[in_range]


def average_precision(
    predictions: List[Tuple[Tensor, Tensor]],
    ground_truth: List[Tensor],
    iou_threshold: float,
    *,
    global_sort: bool = True,
) -> float:
    """VOC-style AP over rotated BEV IoU.

    Parameters
    ----------
    predictions: one ``(boxes, scores)`` pair per frame.
    ground_truth: one ``boxes`` tensor per frame, same ordering.
    global_sort: sort every detection in the dataset by confidence before
        building the precision-recall curve. Required for a valid dataset-level
        comparison; per-frame accumulation silently inflates AP.
    """
    from opencood.utils import box_utils, common_utils

    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"{len(predictions)} predicted frames vs {len(ground_truth)} truth frames"
        )

    records: List[Tuple[float, int]] = []  # (score, is_true_positive)
    total_truth = 0

    for (boxes, scores), truth in zip(predictions, ground_truth):
        total_truth += int(truth.shape[0])
        if boxes.shape[0] == 0:
            continue
        if truth.shape[0] == 0:
            records.extend((float(s), 0) for s in scores)
            continue

        predicted_polygons = common_utils.convert_format(
            box_utils.boxes_to_corners_3d(boxes, order="hwl")[:, :4, :2]
            .detach().cpu().numpy()
        )
        truth_polygons = common_utils.convert_format(
            box_utils.boxes_to_corners_3d(truth, order="hwl")[:, :4, :2]
            .detach().cpu().numpy()
        )

        # Greedy highest-confidence-first matching, one truth box per detection.
        claimed = set()
        for index in torch.argsort(scores, descending=True).tolist():
            iou = common_utils.compute_iou(predicted_polygons[index], truth_polygons)
            best = int(iou.argmax())
            hit = iou[best] >= iou_threshold and best not in claimed
            if hit:
                claimed.add(best)
            records.append((float(scores[index]), int(hit)))

    if total_truth == 0 or not records:
        return 0.0

    if global_sort:
        records.sort(key=lambda row: row[0], reverse=True)

    hits = torch.tensor([row[1] for row in records], dtype=torch.float64)
    true_positives = torch.cumsum(hits, dim=0)
    ranks = torch.arange(1, len(records) + 1, dtype=torch.float64)
    precision = true_positives / ranks
    recall = true_positives / total_truth

    # VOC-style: integrate the monotonically decreasing precision envelope.
    precision = torch.flip(torch.cummax(torch.flip(precision, [0]), dim=0).values, [0])
    recall = torch.cat([torch.zeros(1, dtype=torch.float64), recall])
    return float(((recall[1:] - recall[:-1]) * precision).sum())
