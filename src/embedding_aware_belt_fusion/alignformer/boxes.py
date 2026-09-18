"""Per-agent detection, and the one place that knows about the detector.

A backbone swap (SWFormer, or a stronger encoder) changes this file only. Every
downstream module consumes :class:`AgentDetections` and never touches OpenCOOD.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from torch import Tensor

from embedding_aware_belt_fusion.features.opencood_proposals import (
    assign_proposals_to_ground_truth,
    decode_local_proposals,
    pointpillar_forward_with_features,
)

# IoU below which a proposal is treated as having no ground-truth counterpart.
DEFAULT_MINIMUM_IOU = 0.3


@dataclass(frozen=True)
class AgentDetections:
    """One agent's detections plus the BEV map they were decoded from.

    ``boxes`` are in the agent's own LiDAR frame, OpenCOOD ``hwl`` order
    ``[x, y, z, h, w, l, yaw]`` with yaw in radians. ``gt_ids`` holds the OPV2V
    physical object id each detection was matched to, or ``None`` when it matched
    nothing - these are supervision only and are never transmitted.
    """

    boxes: Tensor
    scores: Tensor
    corners: Tensor
    gt_ids: Sequence[Optional[str]]
    features: Tensor

    def __post_init__(self) -> None:
        count = self.boxes.shape[0]
        lengths = {
            "scores": self.scores.shape[0],
            "corners": self.corners.shape[0],
            "gt_ids": len(self.gt_ids),
        }
        mismatched = {name: n for name, n in lengths.items() if n != count}
        if mismatched:
            raise ValueError(
                f"inconsistent length against {count} boxes: {mismatched}"
            )
        if self.features.dim() != 3:
            raise ValueError(
                f"features must be (C, H, W), got shape {tuple(self.features.shape)}"
            )

    def __len__(self) -> int:
        return self.boxes.shape[0]


def detect_agent(
    detector,
    cav_content: Mapping,
    postprocessor,
    *,
    minimum_iou: float = DEFAULT_MINIMUM_IOU,
) -> AgentDetections:
    """Run the detector on one agent and match its proposals to local ground truth."""
    output = pointpillar_forward_with_features(detector, cav_content)
    decoded = decode_local_proposals(output, cav_content, postprocessor)

    assignment = assign_proposals_to_ground_truth(
        decoded["corners"],
        cav_content["object_bbx_center"][cav_content["object_bbx_mask"].bool()],
        cav_content["object_ids"],
        order=postprocessor.params["order"],
        minimum_iou=minimum_iou,
    )

    return AgentDetections(
        boxes=decoded["boxes"],
        scores=decoded["scores"],
        corners=decoded["corners"],
        gt_ids=assignment["gt_ids"],
        features=output["spatial_features_2d"][0],
    )
