"""Training toward the sparse pairs: the pure parts of the selection and verdict."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_v2xreal_appearance_head_sparse import pick, selection_metric, split_scenarios, verdict  # noqa: E402

from embedding_aware_belt_fusion.alignformer.appearance_head import AgentFrame  # noqa: E402


def _frame(scenario: str) -> AgentFrame:
    return AgentFrame(scenario, "1", "000010", ("a",), np.zeros((1, 2), np.float32), np.zeros((1, 2)), np.ones(1), ("a",))


def test_split_scenarios_holds_out_the_last_fifth_of_sorted_scenarios_whole():
    frames = [_frame(f"s{i:02d}") for i in range(10)] + [_frame("s09")]

    train, held = split_scenarios(frames, fraction=0.2)

    assert {f.scenario for f in held} == {"s08", "s09"}
    assert len(held) == 3 and len(train) == 8
    assert not ({f.scenario for f in train} & {f.scenario for f in held})


def test_selection_metric_is_the_mean_over_the_far_bands_and_none_when_a_band_is_empty():
    report = {"ambiguous_by_range": {"40-70m": {"head": {"auc_paired": 0.7}}, "70m+": {"head": {"auc_paired": 0.9}}}}
    missing = {"ambiguous_by_range": {"40-70m": {"head": {"auc_paired": 0.7}}}}

    assert selection_metric(report) == 0.8
    assert selection_metric(missing) is None


def test_pick_takes_the_best_held_out_metric_and_ignores_unscored_candidates():
    results = [{"best_holdout_metric": None, "config": {"name": "a"}}, {"best_holdout_metric": 0.7, "config": {"name": "b"}},
               {"best_holdout_metric": 0.75, "config": {"name": "c"}}]

    assert pick(results)["config"]["name"] == "c"


def test_verdict_needs_the_sparse_bar_and_the_dense_guard():
    def report(sparse, dense):
        return {"ambiguous_by_shared": {"shared_1_2": {"head": {"auc_paired": sparse}}, "shared_3plus": {"head": {"auc_paired": dense}}}}

    assert verdict(report(0.82, 0.85))["worth_building"] is True
    assert verdict(report(0.70, 0.85))["worth_building"] is False
    assert verdict(report(0.82, 0.75))["worth_building"] is False
