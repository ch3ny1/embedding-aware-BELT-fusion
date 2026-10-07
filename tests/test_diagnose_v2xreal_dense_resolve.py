"""Pure parts of the dense-pair re-solve diagnostic."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from diagnose_v2xreal_dense_resolve import box_residual_m, pair_precision_recall, summarize  # noqa: E402


def test_pair_precision_and_recall_use_the_ground_truth_ids():
    ego_ids, cav_ids = ["a", "b", None, "c"], ["b", "a", "x", None]

    precision, recall, correct, proposed = pair_precision_recall(ego_ids, cav_ids, [0, 1, 2], [1, 0, 2])

    assert (correct, proposed) == (2, 3)
    assert precision == 2 / 3 and recall == 1.0  # shared ids: a, b; both proposed correctly


def test_pair_precision_is_nan_without_proposals_and_recall_nan_without_shared_ids():
    precision, recall, _, _ = pair_precision_recall(["a"], ["z"], [], [])

    assert math.isnan(precision) and math.isnan(recall)


def test_box_residual_is_zero_for_the_true_correction_and_grows_with_the_lever_arm():
    boxes = torch.tensor([[10.0, 0.0, 0.0, 1.5, 2.0, 4.5, 0.0], [20.0, 0.0, 0.0, 1.5, 2.0, 4.5, 0.0]])
    psi, t = torch.tensor(0.1), torch.tensor([0.5, -0.2])
    from embedding_aware_belt_fusion.alignformer.fusion import correct_boxes

    truth = correct_boxes(boxes, psi, t)

    assert box_residual_m(boxes, truth, psi, t) < 1e-5
    assert box_residual_m(boxes, truth, torch.tensor(0.0), t) > box_residual_m(boxes, truth, psi, t + 0.1)


def test_summarize_pools_stage_precision_and_tail_fractions():
    rows = [
        {"residual_icp": 0.2, "icp_stages": [{"gate": 2.0, "pairs": 4, "correct": 3, "precision": 0.75, "recall": 1.0}], "fa_precision": 1.0, "fa_recall": 0.5},
        {"residual_icp": 1.5, "icp_stages": [{"gate": 2.0, "pairs": 2, "correct": 2, "precision": 1.0, "recall": 0.5}]},
    ]

    out = summarize(rows)

    assert out["pairs"] == 2 and out["icp_engaged"] == 2
    assert out["residuals"]["residual_icp"]["frac_over_1m"] == 0.5
    assert out["icp_stage_precision"]["0"]["precision_pooled"] == 5 / 6
    assert out["freealign"]["answered"] == 1
