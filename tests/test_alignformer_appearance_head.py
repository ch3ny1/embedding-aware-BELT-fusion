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
