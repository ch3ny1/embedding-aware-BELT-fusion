"""The DINO-head appearance embedding on the matcher's own inputs.

Pure parts: the per-box crop selection and head projection with an injected
foundation model and head, the agent's own track (mutual-nearest world-frame
matching with a gap-dependent gate), causal pooling with the has_camera
semantics, the earlier-timestamp lookup, and the config dispatch.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from embedding_aware_belt_fusion.alignformer.appearance_head import AppearanceHead
from embedding_aware_belt_fusion.alignformer.camera_features import Calibration, CameraFeatures
from embedding_aware_belt_fusion.alignformer.dino_features import (
    DinoHeadBackbone,
    TrackPooling,
    earlier_timestamps,
    features_for_frame,
    frame_dino_features,
    frame_ref_from_yaml,
    match_to_current,
    pool_embeddings,
    pooling_from_config,
    world_centres,
)


class _FakeFoundation:
    """Descriptors that encode the crop's mean intensity, so identity is testable."""

    embed_dim = 4

    def describe_many(self, crops, masks):
        values = np.array([[float(c.mean()) / 255.0] * 4 for c in crops], dtype=np.float64)
        unit = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
        return {"dino_cls": unit, "dino_patch_mean": unit}


def _backbone(pooling=None) -> DinoHeadBackbone:
    torch.manual_seed(0)
    return DinoHeadBackbone(_FakeFoundation(), AppearanceHead(in_dim=8, hidden_dim=6, out_dim=3), pooling)


def _calibration() -> Calibration:
    intrinsic = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
    return Calibration(intrinsic=intrinsic, camera_to_lidar=np.eye(4))


def _cube(center, half=1.0):
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    return np.asarray(center)[None, :] + half * signs


def test_backbone_is_frozen_and_reports_the_heads_output_width():
    backbone = _backbone()

    assert backbone.feature_dim == 3
    assert all(not p.requires_grad for p in backbone.head.parameters())
    backbone.train()
    assert not backbone.training


def test_frame_features_embed_the_visible_box_and_flag_the_one_behind_the_camera():
    image = np.full((1080, 1920, 3), 200, dtype=np.uint8)
    corners = np.stack([_cube([0.0, 0.0, 12.0]), _cube([0.0, 0.0, -12.0])])

    out = frame_dino_features(_backbone(), corners, [(image, _calibration())])

    assert isinstance(out, CameraFeatures)
    assert out.has_camera.tolist() == [True, False]
    assert out.camera_index.tolist() == [0, -1]
    assert out.features.shape == (2, 3)
    assert np.linalg.norm(out.features[0]) == pytest.approx(1.0, abs=1e-5)
    assert np.abs(out.features[1]).max() == 0.0


def test_features_for_frame_dispatches_on_the_backbone_type():
    image = np.full((1080, 1920, 3), 200, dtype=np.uint8)

    out = features_for_frame(_backbone(), np.stack([_cube([0.0, 0.0, 12.0])]), [(image, _calibration())])

    assert out.has_camera.tolist() == [True]


def test_world_centres_apply_the_agents_pose():
    boxes = np.array([[10.0, 0.0, 0.0, 1.5, 2.0, 4.5, 0.0]])

    xy = world_centres(boxes, [100.0, 50.0, 0.0, 0.0, 90.0, 0.0])  # yaw 90 deg: +x -> +y

    np.testing.assert_allclose(xy, [[100.0, 60.0]], atol=1e-6)


def test_match_to_current_is_mutual_nearest_within_the_gate():
    current = np.array([[0.0, 0.0], [10.0, 0.0], [30.0, 0.0]])
    past = np.array([[0.5, 0.0], [9.0, 0.0], [9.6, 0.0], [50.0, 0.0]])

    mapping = match_to_current(current, past, gate_m=2.0)

    # past 1 and 2 both want current 1; current 1's nearest past is 2, so only 2 matches.
    assert mapping.tolist() == [0, -1, 1, -1]


def test_pool_embeddings_sums_matched_views_and_inherits_has_camera_from_the_track():
    current = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    current_has = np.array([True, False])
    past_raw = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    past_has = np.array([True, True])
    mapping = np.array([0, 1])  # past 0 -> current 0, past 1 -> current 1

    pooled, has = pool_embeddings(current, current_has, [(past_raw, past_has, mapping)])

    r = 1 / np.sqrt(2)
    np.testing.assert_allclose(pooled, [[r, r], [0.0, 1.0]], atol=1e-6)
    assert has.tolist() == [True, True]


def test_pool_embeddings_ignores_past_views_without_a_camera_and_unmatched_boxes():
    current = np.array([[1.0, 0.0]], dtype=np.float32)
    pasts = [(np.array([[0.0, 1.0]], np.float32), np.array([False]), np.array([0])),
             (np.array([[0.0, 1.0]], np.float32), np.array([True]), np.array([-1]))]

    pooled, has = pool_embeddings(current, np.array([True]), pasts)

    np.testing.assert_allclose(pooled, [[1.0, 0.0]])
    assert has.tolist() == [True]


def test_earlier_timestamps_are_the_agents_own_preceding_frames_nearest_first():
    stamps = ["000010", "000011", "000012", "000013", "000014"]

    assert earlier_timestamps(stamps, "000013", window=2) == ["000012", "000011"]
    assert earlier_timestamps(stamps, "000010", window=4) == []


def test_frame_ref_from_yaml_reads_scenario_agent_and_timestamp():
    ref = frame_ref_from_yaml("/data/v2x-real/val/2023-03-17-16-03-02_11_1/-1/000012.yaml")

    assert ref == ("2023-03-17-16-03-02_11_1", "-1", "000012")


def test_pooling_from_config_reads_the_window_and_gates_and_is_none_without_a_window():
    assert pooling_from_config({}) is None
    pooling = pooling_from_config({"track_window": 4, "track_gate_m": 2.0, "track_gate_per_frame_m": 1.5})
    assert pooling == TrackPooling(4, 2.0, 1.5)
    assert pooling.gate(3) == pytest.approx(6.5)


# ----------------------------------------------------------------------------
# Cache round trip and the pooling pass
# ----------------------------------------------------------------------------


def test_frame_record_keeps_the_raw_camera_vector_through_the_cache(tmp_path):
    from embedding_aware_belt_fusion.alignformer.cache import FrameRecord, read_frame, write_frame

    record = FrameRecord(
        boxes=np.zeros((2, 7), np.float32), scores=np.ones(2, np.float32), gt_ids=["a", None],
        roi=np.zeros((2, 3, 2, 2), np.float16), camera=np.ones((2, 3), np.float32), has_camera=np.array([True, False]),
        camera_index=np.array([0, -1], np.int8), camera_raw=np.full((2, 3), 0.5, np.float32),
    )

    write_frame(tmp_path / "f.npz", record)
    back = read_frame(tmp_path / "f.npz")

    np.testing.assert_allclose(back.camera_raw, 0.5)
    assert back.camera_raw.dtype == np.float16
    with pytest.raises(ValueError):
        FrameRecord(boxes=np.zeros((2, 7)), scores=np.ones(2), gt_ids=["a", None], roi=np.zeros((2, 1, 1, 1)),
                    camera_raw=np.zeros((3, 3)))


def _write_agent(split_root, cache_root, split, scenario, agent, stamps, boxes_by_stamp, raw_by_stamp, has_by_stamp):
    from embedding_aware_belt_fusion.alignformer.cache import FrameRecord, cache_path, write_frame

    agent_dir = split_root / scenario / agent
    agent_dir.mkdir(parents=True, exist_ok=True)
    for stamp in stamps:
        (agent_dir / f"{stamp}.yaml").write_text("lidar_pose: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]\n")
        boxes = np.asarray(boxes_by_stamp[stamp], np.float32)
        n = boxes.shape[0]
        path = cache_path(cache_root, split, scenario, agent, stamp)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_frame(path, FrameRecord(boxes=boxes, scores=np.ones(n, np.float32), gt_ids=[None] * n, roi=np.zeros((n, 1, 1, 1), np.float16),
                                      camera=np.asarray(raw_by_stamp[stamp], np.float32), has_camera=np.asarray(has_by_stamp[stamp]),
                                      camera_index=np.zeros(n, np.int8), camera_raw=np.asarray(raw_by_stamp[stamp], np.float32)))


def test_pool_split_pools_each_box_with_its_tracked_predecessors_and_keeps_the_raw_vector(tmp_path):
    from embedding_aware_belt_fusion.alignformer.cache import cache_path, read_frame
    from embedding_aware_belt_fusion.alignformer.dino_features import pool_split

    split_root, cache_root = tmp_path / "val", tmp_path / "cache"
    box = lambda x: [x, 0.0, 0.0, 1.5, 2.0, 4.5, 0.0]  # noqa: E731
    _write_agent(split_root, cache_root, "val", "s", "1", ["000010", "000011", "000012"],
                 {"000010": [box(0.0)], "000011": [box(0.5), box(40.0)], "000012": [box(1.0), box(41.0)]},
                 {"000010": [[1.0, 0.0]], "000011": [[0.0, 1.0], [1.0, 0.0]], "000012": [[1.0, 0.0], [0.0, 1.0]]},
                 {"000010": [True], "000011": [True, False], "000012": [True, True]})

    pooled = pool_split(cache_root, "val", split_root, TrackPooling(window=2, gate_m=2.0, gate_per_frame_m=1.5))

    assert pooled == 3
    last = read_frame(cache_path(cache_root, "val", "s", "1", "000012"))
    r = 1 / np.sqrt(2)
    # Box 0 at 000012 tracks to 000011 (0.5 m) and 000010 (1.0 m): (1,0)+(0,1)+(1,0) -> (2,1)/sqrt5.
    np.testing.assert_allclose(last.camera[0], [2 / np.sqrt(5), 1 / np.sqrt(5)], atol=1e-3)
    # Box 1 tracks to 000011's second box, which had no camera: its own vector only.
    np.testing.assert_allclose(last.camera[1], [0.0, 1.0], atol=1e-3)
    np.testing.assert_allclose(last.camera_raw, [[1.0, 0.0], [0.0, 1.0]], atol=1e-3)  # raw untouched
    first = read_frame(cache_path(cache_root, "val", "s", "1", "000010"))
    np.testing.assert_allclose(first.camera, [[1.0, 0.0]], atol=1e-3)  # nothing earlier: unchanged
    del r


def test_camera_backbone_for_builds_the_dino_backbone_when_the_config_names_a_camera_source(monkeypatch):
    from embedding_aware_belt_fusion.alignformer import dino_features, evaluate

    calls = []
    monkeypatch.setattr(dino_features, "build_dino_backbone", lambda source, device, cache_root=None: calls.append((source, cache_root)) or "dino")

    class WithCameras:
        def frame_cameras(self, path):
            return []

    config = {"model": {"camera_dim": 128, "camera_source": {"backbone": "dino_head", "head_checkpoint": "h.pth"}},
              "data": {"cache_root": "/cache/dino"}}
    out = evaluate.camera_backbone_for(config, WithCameras(), torch.device("cpu"))

    assert out == "dino" and calls[0][0]["backbone"] == "dino_head" and str(calls[0][1]) == "/cache/dino"


def test_build_dino_backbone_rejects_an_unknown_backbone_name():
    from embedding_aware_belt_fusion.alignformer.dino_features import build_dino_backbone

    with pytest.raises(ValueError):
        build_dino_backbone({"backbone": "resnet"}, "cpu")
