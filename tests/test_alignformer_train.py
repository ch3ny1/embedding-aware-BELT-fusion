"""Stage-1 trainer wiring.

The training *run* is measured by the P1 gate
(``outputs/alignformer/p1_gate.json``), not by a test. What is tested here is
the wiring around it, where a silent mistake would make the gate meaningless:
the scenario split actually being disjoint, the embedding control actually
zeroing the embedding, and the split-level tally actually weighting by count.
"""

import math
from pathlib import Path

import pytest
import torch
import yaml

from embedding_aware_belt_fusion.alignformer.boxes import AgentDetections
from embedding_aware_belt_fusion.alignformer.fusion import correct_detections
from embedding_aware_belt_fusion.alignformer.losses import corner_loss
from embedding_aware_belt_fusion.alignformer.model import AlignFormerB
from embedding_aware_belt_fusion.alignformer.procrustes import MIN_MATCH_MASS
from embedding_aware_belt_fusion.alignformer.stage2 import stage2_output_dir
from embedding_aware_belt_fusion.alignformer.train import (
    _MatchingTally,
    _embed,
    build_modules,
    build_pair_split,
    forward_batch,
    zero_embeddings,
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

    trained = _embed(modules["embedding"], roi)
    zeroed = zero_embeddings({"ego_embeddings": trained})["ego_embeddings"]

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


# --- Stage 2: the boxes-only ablation and the pose objective (task 14) -------


def test_boxes_only_mode_removes_all_embedding_information():
    batch = {
        "ego_embeddings": torch.randn(2, 4, 16),
        "cav_embeddings": torch.randn(2, 5, 16),
        "ego_boxes": torch.randn(2, 4, 7),
    }

    ablated = zero_embeddings(batch)

    assert torch.count_nonzero(ablated["ego_embeddings"]) == 0
    assert torch.count_nonzero(ablated["cav_embeddings"]) == 0
    # Geometry must survive, or the ablation tests the wrong thing.
    assert torch.allclose(ablated["ego_boxes"], batch["ego_boxes"])


def test_boxes_only_mode_does_not_mutate_the_original_batch():
    batch = {
        "ego_embeddings": torch.randn(1, 2, 8),
        "cav_embeddings": torch.randn(1, 2, 8),
    }
    before = batch["ego_embeddings"].clone()

    zero_embeddings(batch)

    assert torch.allclose(batch["ego_embeddings"], before)


def test_a_model_step_reduces_the_loss_on_a_single_repeated_batch():
    # Overfitting one batch is the cheapest possible check that gradients flow
    # end to end through Sinkhorn and the closed-form solver.
    torch.manual_seed(0)
    model = AlignFormerB(embed_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = {
        "ego_boxes": torch.randn(4, 6, 7), "ego_scores": torch.rand(4, 6),
        "ego_embeddings": torch.randn(4, 6, 8),
        "ego_mask": torch.ones(4, 6, dtype=torch.bool),
        "cav_boxes": torch.randn(4, 6, 7), "cav_scores": torch.rand(4, 6),
        "cav_embeddings": torch.randn(4, 6, 8),
        "cav_mask": torch.ones(4, 6, dtype=torch.bool),
    }
    target_psi = torch.full((4,), 0.2)
    target_t = torch.full((4, 2), 1.0)

    def step():
        optimizer.zero_grad()
        estimate = model(batch)
        loss = (
            (estimate.psi - target_psi).abs().mean()
            + (estimate.t - target_t).abs().mean()
        )
        loss.backward()
        optimizer.step()
        return loss.item()

    first = step()
    for _ in range(50):
        last = step()

    assert last < first


def test_zeroing_the_embedding_survives_a_batch_that_carries_no_embedding_field():
    # forward_batch inserts the embeddings itself; zero_embeddings is also
    # called on raw batches in tests and analysis scripts, where the fields may
    # not be there yet. Missing is not an error, it is nothing to zero.
    batch = {"ego_boxes": torch.randn(1, 3, 7)}

    ablated = zero_embeddings(batch)

    assert set(ablated) == {"ego_boxes"}


def _pose_batch(psi_true: float, translation: float, ego: int, cav: int) -> dict:
    generator = torch.Generator().manual_seed(7)
    return {
        "ego_boxes": torch.randn(1, ego, 7, generator=generator),
        "ego_scores": torch.rand(1, ego, generator=generator),
        "ego_embeddings": torch.randn(1, ego, 8, generator=generator),
        "ego_mask": torch.ones(1, ego, dtype=torch.bool),
        "cav_boxes": torch.randn(1, cav, 7, generator=generator),
        "cav_scores": torch.rand(1, cav, generator=generator),
        "cav_embeddings": torch.randn(1, cav, 8, generator=generator),
        "cav_mask": torch.ones(1, cav, dtype=torch.bool),
        "psi_true": torch.full((1,), psi_true),
        "t_true": torch.full((1, 2), translation),
    }


def test_the_corner_loss_produces_finite_gradients_at_the_collapsed_temperature():
    # The decisive risk for stage 2: stage 1 drove the match temperature to
    # ~0.033, below the ~0.043 at which design spec 3.4 says the atan2(0, 0)
    # singularity in _soft_correspondence becomes reachable. match_nll never
    # backpropagated through that branch; corner_loss does. A finite LOSS is
    # not evidence -- a NaN gradient earlier in this project came with a
    # perfectly finite loss -- so the assertion has to be on the gradients.
    torch.manual_seed(0)
    model = AlignFormerB(embed_dim=8)
    with torch.no_grad():
        model.log_temperature.fill_(math.log(0.0330))
    batch = _pose_batch(0.2, 1.0, ego=6, cav=6)

    estimate = model(batch)
    loss = corner_loss(
        batch["cav_boxes"], estimate.psi, estimate.t,
        batch["psi_true"], batch["t_true"], batch["cav_mask"],
    )
    loss.backward()

    non_finite = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    assert non_finite == []


def test_a_pair_with_too_little_match_evidence_falls_back_to_the_uncorrected_pose():
    # 18.5% of ego-CAV pairs share no object at all. Those must keep the
    # UNCORRECTED relative pose, not receive a garbage SE(2): a correction
    # applied to a pair with no common object actively corrupts one fused set
    # in five. The guard is procrustes.MIN_MATCH_MASS, and this checks it end
    # to end, through the fusion path that actually consumes the estimate.
    torch.manual_seed(0)
    model = AlignFormerB(embed_dim=8)
    batch = _pose_batch(0.0, 0.0, ego=4, cav=4)
    # Force the correspondence to be all-dustbin: nothing matches anything.
    with torch.no_grad():
        model.dustbin.fill_(50.0)

    with torch.no_grad():
        estimate = model(batch)

    assert float(estimate.confidence.item()) < MIN_MATCH_MASS
    assert float(estimate.psi.abs().max()) == 0.0
    assert float(estimate.t.abs().max()) == 0.0

    boxes = torch.randn(5, 7)
    detections = AgentDetections(
        boxes=boxes,
        scores=torch.rand(5),
        corners=torch.randn(5, 8, 3),
        gt_ids=[None] * 5,
        features=torch.randn(2, 4, 4),
    )

    corrected = correct_detections(detections, estimate.psi[0], estimate.t[0])

    assert torch.allclose(corrected.boxes, detections.boxes)


def test_only_head_b_gets_the_nomatch_suffix():
    # "_nomatch" names head B's fairness control: B's architecture without B's
    # extra supervision. Head A exposes no assignment, so its match weight is 0
    # by necessity, and suffixing it would (a) name it after a control it
    # cannot be the counterpart of and (b) silently move its checkpoint out
    # from under anything that resolves the conventional path.
    assert stage2_output_dir("B", "boxes+embeddings", 1.0).name == "stage2_B_boxes+embeddings"
    assert (
        stage2_output_dir("B", "boxes+embeddings", 0.0).name
        == "stage2_B_boxes+embeddings_nomatch"
    )
    assert stage2_output_dir("A", "boxes+embeddings", 0.0).name == "stage2_A_boxes+embeddings"
    assert stage2_output_dir("A", "boxes_only", 0.0).name == "stage2_A_boxes_only"
