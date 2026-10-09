"""The pure parts of the association-accuracy sweep: ground-truth match indices, Top-1 counting, pooling."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from association_accuracy_sweep import (  # noqa: E402
    bucket_of,
    ego_match_from_ids,
    nearest_top1,
    soft_top1,
    summarize_sigma,
    top1_counts_from_prediction,
)


def test_ego_match_points_at_the_cav_detection_of_the_same_object_and_minus_one_otherwise():
    assert ego_match_from_ids(["a", "b", None, "d"], ["b", None, "a"]) == [2, 0, -1, -1]


def test_top1_counts_only_ego_objects_that_have_a_counterpart():
    counts = top1_counts_from_prediction(predicted=[2, 1, 0, 0], ego_match=[2, 0, -1, -1])

    assert counts == {"correct": 1, "countable": 2}


def test_soft_top1_ignores_the_dustbin_column():
    # Arrange: two ego objects, two CAV objects plus a dustbin column that would win for row 0.
    log_assignment = torch.log(torch.tensor([[0.2, 0.1, 0.7], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]]))

    counts = soft_top1(log_assignment, ego_match=[0, 1])

    assert counts == {"correct": 2, "countable": 2}


def test_nearest_top1_matches_by_centre_distance():
    ego = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    cav = torch.tensor([[10.5, 0.0], [0.4, 0.0]])

    assert nearest_top1(ego, cav, ego_match=[1, 0]) == {"correct": 2, "countable": 2}
    assert nearest_top1(ego, cav, ego_match=[0, 1]) == {"correct": 0, "countable": 2}


def test_buckets_split_at_three_shared_objects():
    assert bucket_of(1) == "shared_1_2" and bucket_of(2) == "shared_1_2" and bucket_of(3) == "shared_3plus"


def test_summary_pools_counts_over_pairs_and_reports_coverage():
    rows = [
        {"bucket": "shared_3plus", "nearest": {"correct": 3, "countable": 4}, "soft": {"correct": 4, "countable": 4},
         "hard": {"precision": 1.0, "recall": 0.75, "correct": 3, "proposed": 3, "shared": 4},
         "freealign": {"precision": 0.5, "recall": 0.5, "correct": 2, "proposed": 4, "shared": 4},
         "freealign_inliers": {"precision": 1.0, "recall": 0.5, "correct": 2, "proposed": 2, "shared": 4},
         "residual_uncorrected": 2.0, "residual_soft": 0.5, "residual_hard": 0.2, "residual_freealign": 3.0},
        {"bucket": "shared_1_2", "nearest": {"correct": 1, "countable": 2}, "soft": {"correct": 2, "countable": 2},
         "residual_uncorrected": 1.0, "residual_soft": 0.4},
    ]

    out = summarize_sigma(rows)

    assert out["pairs"] == 2
    assert out["nearest"]["top1"] == 4 / 6 and out["soft"]["top1"] == 1.0
    assert out["hard"]["answered"] == 0.5 and out["hard"]["precision"] == 1.0 and out["hard"]["recall"] == 0.75
    assert out["freealign"]["answered"] == 0.5 and out["freealign_inliers"]["precision"] == 1.0
    assert out["residual"]["residual_hard"]["n"] == 1 and math.isclose(out["residual"]["residual_soft"]["mean"], 0.45)
    assert out["buckets"]["shared_1_2"]["pairs"] == 1 and math.isnan(out["buckets"]["shared_1_2"]["hard"]["precision"])
