"""The LiDAR+camera trunk as a configuration, not a second code path.

``model.camera_dim`` in the AlignFormer config is the one switch: it sizes
the embedding head's camera branch, tells the pair dataset to emit the
camera fields, and tells ``embed_batch`` to feed them. Zero (the default,
and every OPV2V config) leaves every path byte-identical to before.

``data.val_root`` is the other addition: V2X-Real ships an official,
scenario-disjoint ``val/`` and its detector was trained on all of ``train/``,
so a holdout of ``train/`` would be scenarios the detector has seen. With
``val_root`` set, every pair under ``train_root`` trains and every pair under
``val_root`` validates and calibrates; without it the OPV2V holdout rule is
unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch


def _model_cfg(camera_dim: int = 0) -> dict:
    return {
        "embed_dim": 16,
        "output_size": 2,
        "model_dim": 32,
        "layers": 1,
        "heads": 2,
        "heading_lambda": 2.0,
        "sinkhorn_iterations": 5,
        "max_objects": 64,
        "camera_dim": camera_dim,
    }


def test_stage1_modules_size_the_camera_branch_from_the_config():
    from embedding_aware_belt_fusion.alignformer.train import build_modules

    without = build_modules({"model": _model_cfg(0)}, channels=4, device=torch.device("cpu"))
    with_camera = build_modules({"model": _model_cfg(256)}, channels=4, device=torch.device("cpu"))

    assert without["embedding"].camera_dim == 0
    assert with_camera["embedding"].camera_dim == 256


def test_stage2_modules_size_the_camera_branch_from_the_config():
    from embedding_aware_belt_fusion.alignformer.stage2 import build_stage2_modules

    modules = build_stage2_modules({"model": _model_cfg(256)}, 4, torch.device("cpu"), head="B")

    assert modules["embedding"].camera_dim == 256


def test_a_config_without_camera_dim_means_zero():
    from embedding_aware_belt_fusion.alignformer.train import build_modules, uses_camera

    cfg = _model_cfg()
    del cfg["camera_dim"]

    assert uses_camera({"model": cfg}) is False
    assert build_modules({"model": cfg}, channels=4, device=torch.device("cpu"))["embedding"].camera_dim == 0


def _batch(with_camera: bool, n: int = 3, channels: int = 4, size: int = 2) -> dict:
    g = torch.Generator().manual_seed(0)
    batch = {
        "ego_roi": torch.randn(1, n, channels, size, size, generator=g),
        "cav_roi": torch.randn(1, n, channels, size, size, generator=g),
    }
    if with_camera:
        batch.update(
            ego_camera=torch.randn(1, n, 256, generator=g),
            ego_has_camera=torch.tensor([[True, False, True]]),
            cav_camera=torch.randn(1, n, 256, generator=g),
            cav_has_camera=torch.ones(1, n, dtype=torch.bool),
        )
    return batch


def test_embed_batch_feeds_the_camera_fields_to_a_camera_head():
    from embedding_aware_belt_fusion.alignformer.embedding import ObjectEmbedding
    from embedding_aware_belt_fusion.alignformer.train import embed_batch

    torch.manual_seed(0)
    head = ObjectEmbedding(in_channels=4, output_size=2, dim=16, camera_dim=256).eval()
    batch = _batch(with_camera=True)

    seen = embed_batch(head, batch)
    blind = embed_batch(head, {**batch, "ego_has_camera": torch.zeros(1, 3, dtype=torch.bool)})

    assert seen["ego_embeddings"].shape == (1, 3, 16)
    assert not torch.allclose(seen["ego_embeddings"][0, 0], blind["ego_embeddings"][0, 0])
    torch.testing.assert_close(seen["ego_embeddings"][0, 1], blind["ego_embeddings"][0, 1])


def test_embed_batch_refuses_a_camera_head_on_a_batch_without_camera_fields():
    from embedding_aware_belt_fusion.alignformer.embedding import ObjectEmbedding
    from embedding_aware_belt_fusion.alignformer.train import embed_batch

    head = ObjectEmbedding(in_channels=4, output_size=2, dim=16, camera_dim=256)

    with pytest.raises(ValueError, match="camera"):
        embed_batch(head, _batch(with_camera=False))


def test_embed_batch_ignores_camera_fields_for_a_lidar_head():
    # A boxes+embeddings checkpoint evaluated on a cache built with --camera
    # must behave exactly as on a cache without: the head has no branch to
    # feed, so the fields are simply not consumed.
    from embedding_aware_belt_fusion.alignformer.embedding import ObjectEmbedding
    from embedding_aware_belt_fusion.alignformer.train import embed_batch

    torch.manual_seed(0)
    head = ObjectEmbedding(in_channels=4, output_size=2, dim=16).eval()
    batch = _batch(with_camera=True)

    with_fields = embed_batch(head, batch)
    without = embed_batch(head, {k: v for k, v in batch.items() if "camera" not in k})

    torch.testing.assert_close(with_fields["ego_embeddings"], without["ego_embeddings"])


def test_the_third_message_content_is_registered_with_its_own_stage1_checkpoint():
    from embedding_aware_belt_fusion.alignformer.stage2 import MESSAGE_CONTENTS, STAGE1_CHECKPOINTS

    assert "boxes+embeddings+camera" in MESSAGE_CONTENTS
    assert STAGE1_CHECKPOINTS["boxes+embeddings+camera"] != STAGE1_CHECKPOINTS["boxes+embeddings"]


# ----------------------------------------------------------------------------
# data.val_root
# ----------------------------------------------------------------------------


def _write_split(root: Path, scenarios, agents=("1", "2")) -> None:
    for scenario in scenarios:
        for agent in agents:
            folder = root / scenario / agent
            folder.mkdir(parents=True)
            x = 0.0 if agent == "1" else 10.0
            (folder / "000000.yaml").write_text(f"lidar_pose:\n- {x}\n- 0.0\n- 0.0\n- 0.0\n- 0.0\n- 0.0\n")
            (folder / "000000.bin").write_bytes(b"\x00" * 16)


def _data_cfg(tmp_path: Path, with_val_root: bool) -> dict:
    train_root, val_root = tmp_path / "train", tmp_path / "val"
    _write_split(train_root, ["s_a", "s_b", "s_c", "s_d"])
    _write_split(val_root, ["v_a"])
    data = {
        "train_root": str(train_root),
        "cache_root": str(tmp_path / "cache"),
        "pair_cache_dir": str(tmp_path / "pairs"),
        "comm_range_m": 70.0,
        "val_scenario_fraction": 0.25,
        "split_seed": 0,
    }
    if with_val_root:
        data["val_root"] = str(val_root)
    return {"data": data, "model": _model_cfg(0)}


def test_val_root_puts_every_train_root_pair_in_training_and_validates_on_val_root(tmp_path):
    from embedding_aware_belt_fusion.alignformer.train import build_pair_split

    train_pairs, val_pairs, train_scenarios, val_scenarios = build_pair_split(_data_cfg(tmp_path, True))

    assert set(train_scenarios) == {"s_a", "s_b", "s_c", "s_d"}
    assert val_scenarios == ["v_a"]
    assert {p.scenario for p in train_pairs} == {"s_a", "s_b", "s_c", "s_d"}
    assert {p.scenario for p in val_pairs} == {"v_a"}


def test_without_val_root_the_holdout_rule_is_unchanged(tmp_path):
    from embedding_aware_belt_fusion.alignformer.train import build_pair_split

    train_pairs, val_pairs, train_scenarios, val_scenarios = build_pair_split(_data_cfg(tmp_path, False))

    assert len(val_scenarios) == 1 and len(train_scenarios) == 3
    assert set(train_scenarios) | set(val_scenarios) == {"s_a", "s_b", "s_c", "s_d"}
    assert {p.scenario for p in val_pairs} == set(val_scenarios)


def test_the_validation_dataset_reads_the_val_root_cache_split(tmp_path):
    from embedding_aware_belt_fusion.alignformer.train import build_eval_dataset, build_pair_split

    config = _data_cfg(tmp_path, True)
    _, val_pairs, _, _ = build_pair_split(config)

    dataset = build_eval_dataset(config, val_pairs, 0.0)

    assert dataset.split == "val"
    assert dataset.use_camera is False


def test_the_validation_dataset_uses_the_train_split_without_val_root(tmp_path):
    from embedding_aware_belt_fusion.alignformer.train import build_eval_dataset, build_pair_split

    config = _data_cfg(tmp_path, False)
    _, val_pairs, _, _ = build_pair_split(config)

    assert build_eval_dataset(config, val_pairs, 0.0).split == "train"


def test_datasets_carry_use_camera_from_the_config(tmp_path):
    from embedding_aware_belt_fusion.alignformer.train import build_datasets

    config = _data_cfg(tmp_path, True)
    config["model"]["camera_dim"] = 256
    config["train"] = {"stage1_max_xy_std": 1.0, "seed": 0}

    train_set, val_set = build_datasets(config, epochs=1)

    assert train_set.use_camera is True and val_set.use_camera is True


# ----------------------------------------------------------------------------
# Inference: the sweep computes camera features live, like the ROI features
# ----------------------------------------------------------------------------


class _FakeDataset:
    len_record = [2, 5]
    scenario_database = {
        0: {"1": {"000000": {"yaml": "/s0/1/000000.yaml"}, "000001": {"yaml": "/s0/1/000001.yaml"}, "ego": True},
            "-1": {"000000": {"yaml": "/s0/-1/000000.yaml"}, "000001": {"yaml": "/s0/-1/000001.yaml"}, "ego": False}},
        1: {"2": {"000000": {"yaml": "/s1/2/000000.yaml"}, "ego": True},
            "-2": {"000000": {"yaml": "/s1/-2/000000.yaml"}, "ego": False}},
    }


def test_frame_yaml_resolves_ego_and_cav_keys_across_scenarios():
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import _frame_yaml

    assert _frame_yaml(_FakeDataset(), 1, "ego", "000001") == "/s0/1/000001.yaml"
    assert _frame_yaml(_FakeDataset(), 1, "-1", "000001") == "/s0/-1/000001.yaml"
    assert _frame_yaml(_FakeDataset(), 3, "ego", "000000") == "/s1/2/000000.yaml"
    assert _frame_yaml(_FakeDataset(), 3, "-2", "000000") == "/s1/-2/000000.yaml"


def test_object_set_carries_camera_fields_only_when_the_pack_has_them():
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import _object_set

    side = {"boxes": torch.zeros(2, 7), "scores": torch.ones(2), "roi": torch.zeros(2, 4, 2, 2)}
    with_cam = {**side, "camera": torch.zeros(2, 256), "has_camera": torch.tensor([True, False])}

    plain = _object_set(side, side)
    camera = _object_set(with_cam, side)

    assert not any("camera" in k for k in plain)
    assert camera["ego_camera"].shape == (1, 2, 256) and camera["ego_has_camera"].shape == (1, 2)
    assert "cav_camera" not in camera


def test_camera_like_the_cache_round_trips_float16_and_truncates_in_score_order():
    from embedding_aware_belt_fusion.alignformer.boxes import AgentDetections
    from embedding_aware_belt_fusion.alignformer.camera_features import CameraBackbone, Calibration
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import _camera_like_the_cache, _truncation_order

    backbone = CameraBackbone().eval()
    intrinsic = np.array([[1000.0, 0, 320.0], [0, 1000.0, 180.0], [0, 0, 1]])
    cameras = [(np.random.default_rng(0).integers(0, 255, (360, 640, 3), dtype=np.uint8), Calibration(intrinsic, np.eye(4)))]
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    corners = np.stack([np.array([0.0, 0.0, 10.0]) + signs, np.array([0.0, 0.0, -10.0]) + signs, np.array([1.0, 0.0, 12.0]) + signs])
    found = AgentDetections(
        boxes=torch.zeros(3, 7), scores=torch.tensor([0.2, 0.9, 0.5]), corners=torch.from_numpy(corners),
        gt_ids=[None] * 3, features=torch.zeros(4, 8, 8),
    )

    camera, has_camera = _camera_like_the_cache(found, cameras, backbone)
    order = _truncation_order(found, 2)

    assert has_camera.tolist() == [True, False, True]
    torch.testing.assert_close(camera, camera.half().float())
    assert order.tolist() == [1, 2]
    assert has_camera[order].tolist() == [False, True]


def test_camera_backbone_for_is_none_without_a_camera_branch_and_refuses_a_blind_dataset():
    from embedding_aware_belt_fusion.alignformer.evaluate import camera_backbone_for

    assert camera_backbone_for({"model": _model_cfg(0)}, object(), torch.device("cpu")) is None
    with pytest.raises(ValueError, match="frame_cameras"):
        camera_backbone_for({"model": _model_cfg(256)}, object(), torch.device("cpu"))


def test_camera_backbone_for_builds_a_backbone_for_a_camera_dataset():
    from embedding_aware_belt_fusion.alignformer.camera_features import CameraBackbone
    from embedding_aware_belt_fusion.alignformer.evaluate import camera_backbone_for

    class WithCameras:
        def frame_cameras(self, path):
            return []

    backbone = camera_backbone_for({"model": _model_cfg(256)}, WithCameras(), torch.device("cpu"))

    assert isinstance(backbone, CameraBackbone)


def test_embed_batch_handles_an_agent_frame_with_no_objects_on_a_camera_head():
    # The evaluator sees frames where the detector found nothing; the padded
    # stacks are then (1, 0, ...) and reshaping a zero-element camera tensor
    # with an inferred dimension is what crashed the camera trunk's val sweep.
    from embedding_aware_belt_fusion.alignformer.embedding import ObjectEmbedding
    from embedding_aware_belt_fusion.alignformer.train import embed_batch

    head = ObjectEmbedding(in_channels=4, output_size=2, dim=16, camera_dim=256).eval()
    batch = _batch(with_camera=True, n=0)
    batch["ego_has_camera"] = torch.zeros(1, 0, dtype=torch.bool)

    enriched = embed_batch(head, batch)

    assert enriched["ego_embeddings"].shape == (1, 0, 16)
    assert enriched["cav_embeddings"].shape == (1, 0, 16)
