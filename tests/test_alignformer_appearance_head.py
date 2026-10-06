"""A projection head trained on cross-agent pairs over frozen DINOv2 features.

Zero-shot DINOv2 failed the appearance signal bar on V2X-Real val (AUC 0.68
against 0.80) and its same-agent bound was 0.73: two cameras of one vehicle
at one instant barely separate it from its neighbour, so viewpoint
dominates. The one route left is to TRAIN for viewpoint invariance on the
cross-agent pairs the dataset provides. These tests cover the pure parts:
the head, the contrastive loss with in-frame hard negatives, the cached
agent-frame record, and the pair construction from cached frames.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from embedding_aware_belt_fusion.alignformer.appearance_head import (
    AgentFrame,
    AppearanceHead,
    PairSet,
    build_pairs,
    frame_key,
    info_nce,
)


def _frame(scenario: str, agent: str, stamp: str, vids, centres, seed: int) -> AgentFrame:
    rng = np.random.default_rng(seed)
    n = len(vids)
    return AgentFrame(
        scenario=scenario, agent=agent, timestamp=stamp,
        vids=tuple(vids), features=rng.normal(size=(n, 8)).astype(np.float32),
        centre_xy=np.asarray(centres, dtype=np.float64), range_m=np.full(n, 20.0),
    )


# ----------------------------------------------------------------------------
# Head and loss
# ----------------------------------------------------------------------------


def test_head_projects_to_unit_vectors_of_the_requested_width():
    head = AppearanceHead(in_dim=8, hidden_dim=16, out_dim=4)

    out = head(torch.randn(5, 8))

    assert out.shape == (5, 4)
    torch.testing.assert_close(out.norm(dim=1), torch.ones(5), atol=1e-5, rtol=0)


def test_info_nce_is_lower_when_anchors_match_their_positives():
    anchors = torch.nn.functional.normalize(torch.randn(16, 4), dim=1)
    aligned = anchors.clone()
    random = torch.nn.functional.normalize(torch.randn(16, 4), dim=1)
    no_hard = torch.zeros(16, 0, 4)
    no_mask = torch.zeros(16, 0, dtype=torch.bool)

    good = info_nce(anchors, aligned, no_hard, no_mask, temperature=0.1)
    bad = info_nce(anchors, random, no_hard, no_mask, temperature=0.1)

    assert good < bad


def test_hard_negatives_that_resemble_the_anchor_raise_the_loss_and_padding_does_not():
    anchors = torch.nn.functional.normalize(torch.randn(16, 4), dim=1)
    positives = anchors.clone()
    hard = anchors.unsqueeze(1).repeat(1, 3, 1)  # three look-alikes per anchor
    valid = torch.ones(16, 3, dtype=torch.bool)
    padding_only = torch.zeros(16, 3, dtype=torch.bool)

    without = info_nce(anchors, positives, torch.zeros(16, 0, 4), torch.zeros(16, 0, dtype=torch.bool), 0.1)
    padded = info_nce(anchors, positives, hard, padding_only, 0.1)
    with_hard = info_nce(anchors, positives, hard, valid, 0.1)

    torch.testing.assert_close(padded, without)
    assert with_hard > without


# ----------------------------------------------------------------------------
# Cached frames and pairs
# ----------------------------------------------------------------------------


def test_frame_key_is_filesystem_safe_and_unique_per_agent_frame():
    assert frame_key("2023-04-04-14-28-53_45_0", "-1", "000010") == "2023-04-04-14-28-53_45_0__-1__000010"
    assert frame_key("s", "1", "000010") != frame_key("s", "-1", "000010")


def test_build_pairs_uses_common_vehicles_and_nearest_in_frame_hard_negatives_first():
    ego = _frame("s", "1", "000010", ["a", "b", "c"], [[0, 0], [10, 0], [20, 0]], seed=1)
    cav = _frame("s", "2", "000010", ["a", "c", "d", "e"], [[0, 0], [20, 0], [3, 0], [50, 0]], seed=2)
    other_stamp = _frame("s", "2", "000011", ["a"], [[0, 0]], seed=3)

    pairs = build_pairs([ego, cav, other_stamp], max_hard_negatives=2)

    assert isinstance(pairs, PairSet)
    # (ego->cav) and (cav->ego) for the two common vehicles a and c: four pairs.
    assert pairs.anchors.shape == (4, 8) and pairs.positives.shape == (4, 8)
    assert pairs.hard_negatives.shape == (4, 2, 8) and pairs.hard_mask.shape == (4, 2)
    # ego 'a' -> cav 'a': the nearest other cav vehicles are d (3 m) then c (20 m), not e.
    row = pairs.rows.index(("s", "000010", "1", "2", "a"))
    np.testing.assert_array_equal(pairs.anchors[row], ego.features[0])
    np.testing.assert_array_equal(pairs.positives[row], cav.features[0])
    np.testing.assert_array_equal(pairs.hard_negatives[row, 0], cav.features[2])  # d
    np.testing.assert_array_equal(pairs.hard_negatives[row, 1], cav.features[1])  # c
    assert pairs.hard_mask[row].tolist() == [True, True]


def test_build_pairs_pads_hard_negatives_when_the_frame_has_too_few():
    ego = _frame("s", "1", "000010", ["a"], [[0, 0]], seed=1)
    cav = _frame("s", "2", "000010", ["a"], [[0, 0]], seed=2)

    pairs = build_pairs([ego, cav], max_hard_negatives=2)

    assert pairs.anchors.shape[0] == 2
    assert not pairs.hard_mask.any()


def test_build_pairs_with_shuffled_identities_pairs_different_vehicles():
    ego = _frame("s", "1", "000010", ["a", "b", "c"], [[0, 0], [10, 0], [20, 0]], seed=1)
    cav = _frame("s", "2", "000010", ["a", "b", "c"], [[0, 0], [10, 0], [20, 0]], seed=2)

    honest = build_pairs([ego, cav], max_hard_negatives=2)
    shuffled = build_pairs([ego, cav], max_hard_negatives=2, shuffle_identities=np.random.default_rng(0))

    assert honest.anchors.shape == shuffled.anchors.shape
    # Same anchors, positives drawn from the partner frame's OTHER vehicles.
    for row, key in enumerate(shuffled.rows):
        anchor_vid = key[-1]
        partner = cav if key[2] == "1" else ego
        matched = [v for v, f in zip(partner.vids, partner.features) if np.array_equal(f, shuffled.positives[row])]
        assert matched and matched[0] != anchor_vid


# ----------------------------------------------------------------------------
# The cache record from probe views
# ----------------------------------------------------------------------------


def test_frame_from_views_concatenates_descriptors_and_keeps_every_annotated_vehicle(tmp_path):
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"))
    from cache_v2xreal_appearance_features import frame_from_views

    from embedding_aware_belt_fusion.alignformer.appearance_head import load_frame, save_frame

    params = {
        "lidar_pose": [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "vehicles": {
            "7": {"location": [15.0, 2.0, 0.0], "center": [0, 0, 0], "extent": [2, 1, 0.75], "angle": [0, 0, 0], "obj_type": "Car"},
            "8": {"location": [40.0, 0.0, 0.0], "center": [0, 0, 0], "extent": [2, 1, 0.75], "angle": [0, 0, 0], "obj_type": "Car"},
            "9": {"location": [1.0, 0.0, 0.0], "center": [0, 0, 0], "extent": [1, 1, 1], "angle": [0, 0, 0], "obj_type": "Pedestrian"},
        },
    }
    views = {"7": {"a": np.ones(2), "b": np.zeros(3), "camera": "cam1"}}  # 8 is annotated but not seen

    frame = frame_from_views("s", "1", "000010", params, views, names=("a", "b"))
    reloaded = load_frame(save_frame(tmp_path, frame))

    assert frame.vids == ("7",) and frame.gt_vids == ("7", "8")
    assert frame.features.shape == (1, 5) and frame.features[0].tolist() == [1, 1, 0, 0, 0]
    np.testing.assert_allclose(frame.centre_xy, [[15.0, 2.0]])
    np.testing.assert_allclose(frame.range_m, [np.hypot(5.0, 2.0)])
    assert reloaded.vids == frame.vids and reloaded.gt_vids == frame.gt_vids
    np.testing.assert_array_equal(reloaded.features, frame.features)


# ----------------------------------------------------------------------------
# Review fixes: the control never repels the true match; in-batch false negatives are masked
# ----------------------------------------------------------------------------


def test_control_hard_negatives_never_contain_the_anchors_true_match():
    ego = _frame("s", "1", "000010", ["a", "b", "c"], [[0, 0], [10, 0], [20, 0]], seed=1)
    cav = _frame("s", "2", "000010", ["a", "b", "c"], [[0, 0], [10, 0], [20, 0]], seed=2)

    shuffled = build_pairs([ego, cav], max_hard_negatives=3, shuffle_identities=np.random.default_rng(0))

    for row, key in enumerate(shuffled.rows):
        partner = cav if key[2] == "1" else ego
        true_match = partner.features[partner.vids.index(key[-1])]
        for slot in range(3):
            if shuffled.hard_mask[row, slot]:
                assert not np.array_equal(shuffled.hard_negatives[row, slot], true_match)


def test_same_identity_mask_marks_off_diagonal_repeats_of_one_object():
    from embedding_aware_belt_fusion.alignformer.appearance_head import same_identity_mask

    mask = same_identity_mask([("s", "a"), ("s", "b"), ("s", "a"), ("t", "a")])

    assert mask.tolist() == [
        [False, False, True, False],
        [False, False, False, False],
        [True, False, False, False],
        [False, False, False, False],
    ]


def test_masked_in_batch_duplicates_do_not_count_as_negatives():
    from embedding_aware_belt_fusion.alignformer.appearance_head import same_identity_mask

    base = torch.nn.functional.normalize(torch.randn(8, 4), dim=1)
    anchors = torch.cat([base, base[:1]])  # row 8 repeats object 0
    positives = anchors.clone()
    identities = [("s", str(i)) for i in range(8)] + [("s", "0")]
    none = (torch.zeros(9, 0, 4), torch.zeros(9, 0, dtype=torch.bool))

    unmasked = info_nce(anchors, positives, *none, temperature=0.1)
    masked = info_nce(anchors, positives, *none, temperature=0.1, identity_mask=same_identity_mask(identities))
    all_false = info_nce(anchors, positives, *none, temperature=0.1, identity_mask=torch.zeros(9, 9, dtype=torch.bool))

    # Row 8 is a second view of object 0: unmasked, it is a perfect-score
    # "negative" for row 0 and vice versa; masked, both rows face only the
    # seven other objects, so the loss drops. An all-false mask changes nothing.
    assert masked < unmasked
    torch.testing.assert_close(all_false, unmasked)


# ----------------------------------------------------------------------------
# Training toward the sparse pairs: row metadata, cross-time partners, weighted draws
# ----------------------------------------------------------------------------


def test_build_pairs_records_the_shared_count_and_anchor_range_of_every_row():
    ego = _frame("s", "1", "000010", ["a", "b"], [[0, 0], [10, 0]], seed=1)._replace(range_m=np.array([55.0, 12.0]), gt_vids=("a", "b", "x", "y"))
    cav = _frame("s", "2", "000010", ["a", "b"], [[0, 0], [10, 0]], seed=2)._replace(gt_vids=("a", "b", "x"))

    pairs = build_pairs([ego, cav], max_hard_negatives=2)

    row = pairs.rows.index(("s", "000010", "1", "2", "a"))
    assert pairs.shared_counts[row] == 3  # a, b, x annotated in both
    assert pairs.anchor_ranges[row] == 55.0
    assert pairs.shared_counts.shape == (4,) and pairs.anchor_ranges.shape == (4,)


def test_cross_time_offsets_pair_an_anchor_with_the_partner_agents_neighbouring_frames():
    frames = [
        _frame("s", "1", "000010", ["a"], [[0, 0]], seed=1),
        _frame("s", "2", "000010", ["a"], [[0, 0]], seed=2),
        _frame("s", "2", "000012", ["a"], [[1, 0]], seed=3),  # the partner one cached step later
        _frame("s", "1", "000012", ["a"], [[1, 0]], seed=4),
    ]

    same_time = build_pairs(frames, max_hard_negatives=1)
    with_offsets = build_pairs(frames, max_hard_negatives=1, offsets=(-1, 0, 1))

    assert len(same_time.rows) == 4  # two timestamps, both directions
    assert len(with_offsets.rows) == 8  # plus 1@10->2@12, 2@10->1@12, 1@12->2@10, 2@12->1@10
    # Rows are keyed by the ANCHOR's timestamp: anchor 1@10 now has two rows
    # against agent 2, one with the partner's features at 10, one at 12.
    rows = [i for i, r in enumerate(with_offsets.rows) if r == ("s", "000010", "1", "2", "a")]
    assert len(rows) == 2
    partner_same, partner_later = frames[1].features[0], frames[2].features[0]
    assert sorted(np.array_equal(with_offsets.positives[i], partner_later) for i in rows) == [False, True]
    assert sorted(np.array_equal(with_offsets.positives[i], partner_same) for i in rows) == [False, True]


def test_cross_time_offsets_never_pair_an_agent_with_itself():
    frames = [_frame("s", "1", "000010", ["a"], [[0, 0]], seed=1), _frame("s", "1", "000012", ["a"], [[0, 0]], seed=2)]

    pairs = build_pairs(frames, max_hard_negatives=1, offsets=(-1, 0, 1))

    assert pairs.rows == []


def test_row_weights_emphasize_far_anchors_and_sparse_pairs():
    from embedding_aware_belt_fusion.alignformer.appearance_head import row_weights

    shared = np.array([5, 5, 2, 2])
    ranges = np.array([10.0, 60.0, 10.0, 60.0])

    weights = row_weights(shared, ranges, far_weight=3.0, sparse_weight=5.0, far_from_m=40.0)

    assert weights.tolist() == [1.0, 4.0, 6.0, 9.0]


def test_weighted_epoch_order_draws_rows_in_proportion_to_their_weight():
    from embedding_aware_belt_fusion.alignformer.appearance_head import epoch_order

    weights = np.array([1.0, 0.0, 9.0])
    order = epoch_order(weights, count=3, draws=3000, rng=np.random.default_rng(0))

    assert order.shape == (3000,)
    assert (order == 1).sum() == 0
    assert 0.85 < (order == 2).mean() < 0.95


def test_uniform_epoch_order_is_a_permutation():
    from embedding_aware_belt_fusion.alignformer.appearance_head import epoch_order

    order = epoch_order(None, count=10, draws=10, rng=np.random.default_rng(0))

    assert sorted(order.tolist()) == list(range(10))
