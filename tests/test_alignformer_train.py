"""Stage-1 trainer wiring.

The training *run* is measured by the P1 gate
(``outputs/alignformer/p1_gate.json``), not by a test. What is tested here is
the wiring around it, where a silent mistake would make the gate meaningless:
the scenario split actually being disjoint, the embedding control actually
zeroing the embedding, and the split-level tally actually weighting by count.
"""

from pathlib import Path

import pytest
import torch
import yaml

from embedding_aware_belt_fusion.alignformer.train import (
    _MatchingTally,
    _embed,
    build_modules,
    build_pair_split,
    forward_batch,
)

CONFIG_PATH = Path("configs/alignformer.yaml")


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def _batch(batch_size: int, ego: int, cav: int, channels: int, grid: int) -> dict:
    generator = torch.Generator().manual_seed(0)

    def side(count: int) -> dict:
        return {
            "boxes": torch.randn(batch_size, count, 7, generator=generator),
            "scores": torch.rand(batch_size, count, generator=generator),
            "roi": torch.randn(
                batch_size, count, channels, grid, grid, generator=generator
            ),
            "mask": torch.ones(batch_size, count, dtype=torch.bool),
        }

    ego_side, cav_side = side(ego), side(cav)
    return {
        "ego_boxes": ego_side["boxes"], "ego_scores": ego_side["scores"],
        "ego_roi": ego_side["roi"], "ego_mask": ego_side["mask"],
        "cav_boxes": cav_side["boxes"], "cav_scores": cav_side["scores"],
        "cav_roi": cav_side["roi"], "cav_mask": cav_side["mask"],
        "ego_match": torch.full((batch_size, ego), -1, dtype=torch.long),
        "cav_match": torch.full((batch_size, cav), -1, dtype=torch.long),
    }


def test_build_modules_rejects_a_max_objects_that_the_dataset_will_not_honour():
    config = _config()
    config["model"]["max_objects"] = 65

    with pytest.raises(ValueError, match="max_objects"):
        build_modules(config, 384, torch.device("cpu"))


def test_build_modules_sizes_the_embedding_head_to_the_cached_roi():
    config = _config()
    modules = build_modules(config, 16, torch.device("cpu"))

    grid = int(config["model"]["output_size"])
    embeddings = modules["embedding"](torch.randn(3, 16, grid, grid))

    assert embeddings.shape == (3, int(config["model"]["embed_dim"]))


def test_zeroing_the_embedding_leaves_the_trunk_only_box_geometry():
    config = _config()
    modules = build_modules(config, 8, torch.device("cpu"))
    roi = torch.randn(2, 5, 8, int(config["model"]["output_size"]),
                      int(config["model"]["output_size"]))

    zeroed = _embed(modules["embedding"], roi, True)
    trained = _embed(modules["embedding"], roi, False)

    assert zeroed.shape == trained.shape == (2, 5, int(config["model"]["embed_dim"]))
    assert torch.count_nonzero(zeroed) == 0
    assert torch.count_nonzero(trained) > 0


def test_forward_batch_returns_a_sinkhorn_assignment_with_both_dustbins():
    config = _config()
    grid = int(config["model"]["output_size"])
    modules = build_modules(config, 8, torch.device("cpu"))

    estimate = forward_batch(modules, _batch(2, 5, 7, 8, grid))

    # (B, M + 1, N + 1): the extra row and column are the dustbins.
    assert estimate.log_assignment.shape == (2, 6, 8)
    assert torch.isfinite(estimate.log_assignment).all()


def test_the_tally_weights_by_object_count_not_by_batch():
    # One object right in a 1-object batch, then three wrong in a 3-object
    # batch: 0.25, not the 0.5 that averaging the two batches would give.
    tally = _MatchingTally()

    first = torch.full((1, 2, 3), -10.0)
    first[0, 0, 0] = 0.0
    tally.update(
        {
            "ego_match": torch.tensor([[0]]),
            "ego_mask": torch.ones(1, 1, dtype=torch.bool),
            "cav_mask": torch.ones(1, 2, dtype=torch.bool),
            "ego_boxes": torch.zeros(1, 1, 7),
            "cav_boxes": torch.zeros(1, 2, 7),
        },
        first,
        loss=1.0,
    )
    second = torch.full((3, 2, 3), -10.0)
    second[:, 0, 1] = 0.0
    tally.update(
        {
            "ego_match": torch.zeros(3, 1, dtype=torch.long),
            "ego_mask": torch.ones(3, 1, dtype=torch.bool),
            "cav_mask": torch.ones(3, 2, dtype=torch.bool),
            "ego_boxes": torch.zeros(3, 1, 7),
            "cav_boxes": torch.zeros(3, 2, 7),
        },
        second,
        loss=3.0,
    )

    metrics = tally.compute()

    assert metrics["top1"] == pytest.approx(0.25)
    assert metrics["countable_objects"] == 4.0
    assert metrics["top1_chance"] == pytest.approx(0.5)


def test_the_scenario_split_is_disjoint_and_never_splits_a_scenario_by_frame():
    train_pairs, val_pairs, train_scenarios, val_scenarios = build_pair_split(_config())

    assert train_pairs and val_pairs
    assert set(train_scenarios).isdisjoint(val_scenarios)
    # The decisive property: no scenario appears on both sides, so no frame of
    # a validation scenario can have been trained on.
    assert {pair.scenario for pair in train_pairs}.isdisjoint(
        {pair.scenario for pair in val_pairs}
    )
    assert {pair.scenario for pair in val_pairs} == set(val_scenarios)
