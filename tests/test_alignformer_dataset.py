"""Tests for the pairwise object-set dataset with a localization-noise curriculum.

Step 5's ``test_applying_the_label_realigns_the_cav_boxes`` is the single most
important test in this file: it is the only thing standing between the project
and a silently-wrong SE(2) label convention (see task-11 brief).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from embedding_aware_belt_fusion.alignformer.cache import FrameRecord, cache_path, write_frame
from embedding_aware_belt_fusion.alignformer.dataset import (
    NoiseSchedule,
    OPV2VObjectSetDataset,
    collate,
    correspondence_indices,
)
from embedding_aware_belt_fusion.coloca.geometry import se2_matrix
from embedding_aware_belt_fusion.coloca.index import AgentPair


def test_noise_ramps_from_zero_to_the_maximum():
    schedule = NoiseSchedule(max_xy_std=2.0)

    assert schedule.sigma_for_epoch(0, 10) == pytest.approx(0.0)
    assert schedule.sigma_for_epoch(9, 10) == pytest.approx(2.0)
    assert schedule.sigma_for_epoch(5, 10) < 2.0


def test_correspondences_pair_shared_object_ids():
    ego_ids = ["7", "9", None, "3"]
    cav_ids = ["3", "7", None]

    ego_match, cav_match = correspondence_indices(ego_ids, cav_ids)

    assert ego_match.tolist() == [1, -1, -1, 0]
    assert cav_match.tolist() == [3, 0, -1]


def test_none_ids_never_match_each_other():
    # Two unmatched detections both carry None; treating that as a shared
    # identity would train the model towards a false correspondence.
    ego_match, cav_match = correspondence_indices([None, None], [None])

    assert ego_match.tolist() == [-1, -1]
    assert cav_match.tolist() == [-1]


def test_duplicate_ids_take_the_first_occurrence_only():
    ego_match, cav_match = correspondence_indices(["5", "5"], ["5"])

    assert (ego_match >= 0).sum() == 1
    assert (cav_match >= 0).sum() == 1


def _sample(ego_count, cav_count, ego_match, cav_match):
    return {
        "ego_boxes": torch.randn(ego_count, 7),
        "ego_scores": torch.rand(ego_count),
        "ego_roi": torch.randn(ego_count, 4, 4, 4),
        "cav_boxes": torch.randn(cav_count, 7),
        "cav_scores": torch.rand(cav_count),
        "cav_roi": torch.randn(cav_count, 4, 4, 4),
        "ego_match": torch.tensor(ego_match),
        "cav_match": torch.tensor(cav_match),
        "psi_true": torch.tensor(0.1),
        "t_true": torch.tensor([1.0, 2.0]),
    }


def test_collate_pads_to_the_batch_maximum_and_masks():
    batch = collate([_sample(2, 3, [-1, 0], [1, -1, -1]),
                     _sample(5, 1, [-1] * 5, [-1])])

    assert batch["ego_boxes"].shape == (2, 5, 7)
    assert batch["cav_boxes"].shape == (2, 3, 7)
    assert batch["ego_mask"][0].tolist() == [True, True, False, False, False]
    assert batch["cav_mask"][1].tolist() == [True, False, False]
    assert batch["psi_true"].shape == (2,)


def test_padded_match_targets_are_negative_one():
    batch = collate([_sample(1, 1, [0], [0]), _sample(3, 2, [-1, 1, -1], [-1, 1])])

    assert batch["ego_match"][0, 1:].tolist() == [-1, -1]


# --- Negative control for the padding test above -------------------------
# If collate padded ego_match/cav_match with 0 instead of -1 it would claim
# every padded slot matches object 0. Flip _PAD_VALUES's intent here and
# show the test above would then fail; this block is not itself a test, it
# documents the check performed manually (see task-11-report.md) since
# mutating module internals from a test would be its own footgun.


# --- Step 5: label-exactness integration test -----------------------------

_EGO_POSE = (10.0, 5.0, 0.0, 0.0, 30.0, 0.0)
_CAV_POSE_TRUE = (14.0, 2.0, 0.0, 0.0, -15.0, 0.0)


def _write_synthetic_frame(path, boxes, gt_ids):
    count = boxes.shape[0]
    write_frame(
        path,
        FrameRecord(
            boxes=boxes.astype(np.float32),
            scores=np.linspace(1.0, 0.5, count).astype(np.float32),
            gt_ids=gt_ids,
            roi=np.zeros((count, 2, 2, 2), dtype=np.float16),
        ),
    )


def _true_projection(cav_boxes: np.ndarray) -> np.ndarray:
    """Independently reproduce the true-pose ego-frame projection.

    Deliberately re-derived here (not by calling into dataset.py) so a bug in
    the dataset's own projection cannot also be baked into the check.
    """
    world_to_ego = np.linalg.inv(se2_matrix(_EGO_POSE[0], _EGO_POSE[1], _EGO_POSE[4]))
    cav_to_world = se2_matrix(_CAV_POSE_TRUE[0], _CAV_POSE_TRUE[1], _CAV_POSE_TRUE[4])
    transform = world_to_ego @ cav_to_world
    homogeneous = np.hstack([cav_boxes[:, :2], np.ones((cav_boxes.shape[0], 1))])
    return (homogeneous @ transform.T)[:, :2]


def _build_dataset(tmp_path, sign=1.0):
    cav_boxes = np.array(
        [[3.0, -2.0, 0.0, 1.5, 1.5, 4.0, 0.4], [-5.0, 6.0, 0.0, 1.5, 1.5, 4.0, -0.9]]
    )
    ego_boxes = np.array([[1.0, 1.0, 0.0, 1.5, 1.5, 4.0, 0.1]])

    scenario, timestamp, ego_id, cav_id = "scenario_0", "000069", "ego", "cav"
    _write_synthetic_frame(
        cache_path(tmp_path, "train", scenario, ego_id, timestamp), ego_boxes, [None]
    )
    _write_synthetic_frame(
        cache_path(tmp_path, "train", scenario, cav_id, timestamp), cav_boxes, [None, None]
    )

    pair = AgentPair(
        scenario=scenario,
        timestamp=timestamp,
        ego_id=ego_id,
        cav_id=cav_id,
        ego_pose=_EGO_POSE,
        cav_pose=_CAV_POSE_TRUE,
    )
    dataset = OPV2VObjectSetDataset(
        [pair],
        tmp_path,
        "train",
        noise_schedule=NoiseSchedule(max_xy_std=3.0),
        train=True,
    )
    return dataset, cav_boxes


def test_applying_the_label_realigns_the_cav_boxes(tmp_path):
    """The SE(2) label must exactly undo the injected pose noise.

    coloca/geometry.py guarantees ``C @ T_noisy == T_true`` to 1e-9, so a box
    projected with the noisy pose and then corrected must land where the
    true-pose projection puts it.
    """
    dataset, cav_boxes = _build_dataset(tmp_path)

    sample = dataset[0]

    cos_psi, sin_psi = torch.cos(sample["psi_true"]), torch.sin(sample["psi_true"])
    rotation = torch.tensor([[cos_psi, -sin_psi], [sin_psi, cos_psi]])
    corrected = sample["cav_boxes"][:, :2] @ rotation.T + sample["t_true"]

    expected = _true_projection(cav_boxes)
    np.testing.assert_allclose(corrected.numpy(), expected, atol=1e-4)


def test_negative_control_wrong_sign_correction_fails(tmp_path):
    """Flipping the correction's sign must break the exactness check above.

    This is the negative control required by the task: if this test could not
    be made to fail, the positive test above would be vacuous.
    """
    dataset, cav_boxes = _build_dataset(tmp_path)
    sample = dataset[0]

    cos_psi, sin_psi = torch.cos(sample["psi_true"]), torch.sin(sample["psi_true"])
    rotation = torch.tensor([[cos_psi, -sin_psi], [sin_psi, cos_psi]])
    # Sign flipped on purpose: t_true negated instead of added.
    wrongly_corrected = sample["cav_boxes"][:, :2] @ rotation.T - sample["t_true"]

    expected = _true_projection(cav_boxes)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(wrongly_corrected.numpy(), expected, atol=1e-4)


def test_projection_puts_boxes_in_the_ego_frame_not_the_cav_frame(tmp_path):
    """trunk.tokenize requires boxes already in the ego frame (its docstring).

    A dataset that forgot the projection would return cav_boxes unchanged
    (still in the CAV's own LiDAR frame); this test fails on that regression
    because the ego and CAV poses here are far apart, so an un-projected
    centre cannot coincide with the correctly-projected one.
    """
    dataset, cav_boxes = _build_dataset(tmp_path)

    sample = dataset[0]

    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_allclose,
        sample["cav_boxes"][:, :2].numpy(),
        cav_boxes[:, :2],
        atol=1e-3,
    )


def test_correspondence_ids_reflect_truncated_final_index_positions(tmp_path):
    """None gt_ids from the cache never match; real ids match across sets."""
    scenario, timestamp, ego_id, cav_id = "scenario_1", "000001", "ego", "cav"
    ego_boxes = np.array([[0.0, 0.0, 0.0, 1.5, 1.5, 4.0, 0.0], [1.0, 1.0, 0.0, 1.5, 1.5, 4.0, 0.0]])
    cav_boxes = np.array([[0.0, 0.0, 0.0, 1.5, 1.5, 4.0, 0.0]])

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_synthetic_frame(
            cache_path(root, "train", scenario, ego_id, timestamp), ego_boxes, ["42", None]
        )
        _write_synthetic_frame(
            cache_path(root, "train", scenario, cav_id, timestamp), cav_boxes, ["42"]
        )
        pair = AgentPair(
            scenario=scenario,
            timestamp=timestamp,
            ego_id=ego_id,
            cav_id=cav_id,
            ego_pose=_EGO_POSE,
            cav_pose=_CAV_POSE_TRUE,
        )
        dataset = OPV2VObjectSetDataset(
            [pair], root, "train", noise_schedule=NoiseSchedule(max_xy_std=0.0), train=False
        )
        sample = dataset[0]

    assert sample["ego_match"].tolist() == [0, -1]
    assert sample["cav_match"].tolist() == [0]
