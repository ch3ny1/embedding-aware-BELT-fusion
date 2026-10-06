"""Training toward the sparse pairs: the pure parts of the selection and verdict."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_v2xreal_appearance_head_sparse import pick, selection_metric, split_scenarios, verdict  # noqa: E402

from embedding_aware_belt_fusion.alignformer.appearance_head import AgentFrame, frame_key  # noqa: E402


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


def test_merge_frames_concatenates_descriptors_per_vehicle_and_drops_what_either_cache_lacks():
    from train_v2xreal_appearance_head_sparse import merge_frames

    dino = AgentFrame("s", "1", "000010", ("a", "b"), np.array([[1, 1], [2, 2]], np.float32), np.array([[0, 0], [5, 0]]), np.array([10.0, 20.0]), ("a", "b", "c"))
    colour = AgentFrame("s", "1", "000010", ("b", "a"), np.array([[7, 7, 7], [9, 9, 9]], np.float32), np.array([[5, 0], [0, 0]]), np.array([20.0, 10.0]), ("a", "b", "c"))
    only_dino = AgentFrame("s", "1", "000012", ("a",), np.ones((1, 2), np.float32), np.zeros((1, 2)), np.ones(1), ("a",))

    merged = merge_frames([dino, only_dino], [colour])

    assert len(merged) == 1  # the frame missing from the colour cache is dropped
    frame = merged[0]
    assert frame.vids == ("a", "b") and frame.gt_vids == ("a", "b", "c")
    assert frame.features.tolist() == [[1, 1, 9, 9, 9], [2, 2, 7, 7, 7]]  # matched by id, not by position
    assert frame.range_m.tolist() == [10.0, 20.0]


def test_cache_script_accepts_a_descriptor_set():
    from cache_v2xreal_appearance_features import parse_args

    args = parse_args(["--root", "/x/val", "--output-dir", "/y", "--descriptors", "colour"])

    assert args.descriptors == "colour" and args.every == 1


def test_track_aggregation_averages_an_objects_embedding_over_the_agents_neighbouring_frames():
    from train_v2xreal_appearance_head_sparse import aggregate_tracks

    def frame(stamp, vids):
        return AgentFrame("s", "1", stamp, tuple(vids), np.zeros((len(vids), 2), np.float32), np.zeros((len(vids), 2)), np.ones(len(vids)), tuple(vids))

    frames = [frame("000010", ["a", "b"]), frame("000012", ["a"]), frame("000014", ["a"]), frame("000030", ["a"])]
    table = {
        frame_key("s", "1", "000010"): np.array([[1.0, 0.0], [0.0, 1.0]]),
        frame_key("s", "1", "000012"): np.array([[0.0, 1.0]]),
        frame_key("s", "1", "000014"): np.array([[1.0, 0.0]]),
        frame_key("s", "1", "000030"): np.array([[0.6, 0.8]]),
    }

    out = aggregate_tracks(frames, table, window=1)

    r = 1 / np.sqrt(2)
    np.testing.assert_allclose(out[frame_key("s", "1", "000010")], [[r, r], [0.0, 1.0]], atol=1e-6)  # a: frames 10+12; b alone
    np.testing.assert_allclose(out[frame_key("s", "1", "000012")], [[2 / np.sqrt(5), 1 / np.sqrt(5)]], atol=1e-6)  # a over 10, 12, 14
    v = np.array([1.6, 0.8]) / np.linalg.norm([1.6, 0.8])
    np.testing.assert_allclose(out[frame_key("s", "1", "000030")], [v], atol=1e-6)  # neighbours are cached positions: stamp 14 is one step before 30


def test_jsonable_turns_paths_inside_lists_into_strings():
    import json

    from train_v2xreal_appearance_head_sparse import jsonable

    out = jsonable({"a": Path("/x"), "b": [Path("/y"), 2], "c": (Path("/z"),), "d": 1.5})

    assert json.loads(json.dumps(out)) == {"a": "/x", "b": ["/y", 2], "c": ["/z"], "d": 1.5}
