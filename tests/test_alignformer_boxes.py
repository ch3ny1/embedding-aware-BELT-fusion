"""Tests for the AgentDetections dataclass - the detector/downstream boundary.

Only the dataclass is exercised here. ``detect_agent`` needs a live OpenCOOD
dataset and detector; it is covered end-to-end by a later task.
"""

from __future__ import annotations

import pytest
import torch

from embedding_aware_belt_fusion.alignformer.boxes import AgentDetections


def test_agent_detections_is_immutable_and_reports_count():
    detections = AgentDetections(
        boxes=torch.zeros(3, 7),
        scores=torch.zeros(3),
        corners=torch.zeros(3, 8, 3),
        gt_ids=["a", None, "c"],
        features=torch.zeros(256, 96, 256),
    )

    assert len(detections) == 3
    with pytest.raises(AttributeError):
        detections.boxes = torch.zeros(1, 7)


def test_agent_detections_rejects_inconsistent_lengths():
    with pytest.raises(ValueError, match="length"):
        AgentDetections(
            boxes=torch.zeros(3, 7),
            scores=torch.zeros(2),
            corners=torch.zeros(3, 8, 3),
            gt_ids=["a", None, "c"],
            features=torch.zeros(256, 96, 256),
        )


def test_agent_detections_rejects_corners_length_mismatch():
    with pytest.raises(ValueError, match="length"):
        AgentDetections(
            boxes=torch.zeros(3, 7),
            scores=torch.zeros(3),
            corners=torch.zeros(2, 8, 3),
            gt_ids=["a", None, "c"],
            features=torch.zeros(256, 96, 256),
        )


def test_agent_detections_rejects_gt_ids_length_mismatch():
    with pytest.raises(ValueError, match="length"):
        AgentDetections(
            boxes=torch.zeros(3, 7),
            scores=torch.zeros(3),
            corners=torch.zeros(3, 8, 3),
            gt_ids=["a", None],
            features=torch.zeros(256, 96, 256),
        )
