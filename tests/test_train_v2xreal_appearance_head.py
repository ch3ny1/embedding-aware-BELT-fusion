"""The appearance head's training script: the pure parts.

Scoring cached frames must reproduce the probe's statistics (partner versus
nearest distractor, ambiguous subset, shared-object bucket from every
annotated vehicle, not just the seen ones); the fixed-epoch trainer must
learn a separable toy problem; the verdict must apply the pre-registered
bars.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_v2xreal_colour_separability import Pair  # noqa: E402
from train_v2xreal_appearance_head import (  # noqa: E402
    head_embeddings,
    pairs_from_frames,
    score_pairs,
    train_head,
    verdict,
    zero_shot_embeddings,
)

from embedding_aware_belt_fusion.alignformer.appearance_head import AgentFrame, build_pairs, frame_key  # noqa: E402


def _frame(agent, stamp, vids, centres, feats, gt=None):
    return AgentFrame("s", agent, stamp, tuple(vids), np.asarray(feats, np.float32), np.asarray(centres, float),
                      np.full(len(vids), 30.0), tuple(gt if gt is not None else vids))


def test_score_pairs_gives_a_perfect_auc_to_identity_preserving_embeddings_and_buckets_by_annotations():
    ego = _frame("1", "000010", ["a", "b"], [[0, 0], [5, 0]], [[1, 0, 0], [0, 1, 0]], gt=["a", "b", "x", "y"])
    cav = _frame("2", "000010", ["a", "b"], [[0, 0], [5, 0]], [[1, 0, 0], [0, 1, 0]], gt=["a", "b", "x"])
    embeddings = {"head": {frame_key("s", "1", "000010"): ego.features, frame_key("s", "2", "000010"): cav.features}}

    report = score_pairs([ego, cav], [Pair("s", "000010", "1", "2")], embeddings, random.Random(0))

    assert report["pairs_scored"] == 1
    assert report["ambiguous"]["head"]["auc_paired"] == 1.0  # 5 m apart: ambiguous, and separated
    assert report["ambiguous_by_shared"]["shared_3plus"]["head"]["n"] == 2  # three annotated in common
    assert report["ambiguous_by_shared"]["shared_1_2"]["head"]["n"] == 0


def test_zero_shot_embeddings_split_the_cached_descriptor_halves():
    frame = _frame("1", "000010", ["a"], [[0, 0]], [[1, 2, 3, 4]])

    tables = zero_shot_embeddings([frame], split_at=2)

    assert tables["zero_shot_cls"][frame_key("s", "1", "000010")].tolist() == [[1, 2]]
    assert tables["zero_shot_patch_mean"][frame_key("s", "1", "000010")].tolist() == [[3, 4]]


def test_pairs_from_frames_lists_ordered_agent_pairs_per_timestamp():
    frames = [_frame("1", "000010", ["a"], [[0, 0]], [[1.0]]), _frame("2", "000010", ["a"], [[0, 0]], [[1.0]]),
              _frame("1", "000011", ["a"], [[0, 0]], [[1.0]])]

    pairs = pairs_from_frames(frames, count=10, seed=0)

    assert set(pairs) == {Pair("s", "000010", "1", "2"), Pair("s", "000010", "2", "1")}


def test_train_head_learns_a_toy_cross_view_problem():
    # Identity lives in the first two dims (a 2-D "signature" per vehicle);
    # each VIEW adds its own large random nuisance in the last four dims, so
    # zero-shot cosine between two views is mostly nuisance and near chance
    # against the distractor; the head must learn to keep the signature.
    rng = np.random.default_rng(0)
    frames = []
    for stamp in range(40):
        signatures = rng.normal(size=(4, 2))
        for agent in ("1", "2"):
            feats = np.concatenate([signatures, 3.0 * rng.normal(size=(4, 4))], axis=1)
            frames.append(_frame(agent, f"{stamp:06d}", ["a", "b", "c", "d"], [[0, 0], [4, 0], [8, 0], [12, 0]], feats))
    pairs = build_pairs(frames, max_hard_negatives=3)
    probe_pairs = [Pair("s", f"{stamp:06d}", "1", "2") for stamp in range(40)]
    before = score_pairs(frames, probe_pairs, {"raw": {frame_key(f.scenario, f.agent, f.timestamp): f.features for f in frames}}, random.Random(0))

    head, losses = train_head(pairs, epochs=20, batch_size=64, lr=3e-3, temperature=0.1, seed=0, device="cpu")
    after = score_pairs(frames, probe_pairs, {"head": head_embeddings(frames, head, "cpu")}, random.Random(0))

    assert losses[-1] < losses[0]
    assert before["ambiguous"]["raw"]["auc_paired"] < 0.75
    assert after["ambiguous"]["head"]["auc_paired"] > 0.9


def test_verdict_requires_the_signal_bar_and_a_null_control():
    good = {"ambiguous": {"head": {"auc_paired": 0.85}}}
    weak = {"ambiguous": {"head": {"auc_paired": 0.70}}}
    null_control = {"ambiguous": {"head": {"auc_paired": 0.52}}}
    leaky_control = {"ambiguous": {"head": {"auc_paired": 0.70}}}

    assert verdict(good, null_control)["worth_building"] is True
    assert verdict(weak, null_control)["worth_building"] is False
    assert verdict(good, leaky_control)["worth_building"] is False
