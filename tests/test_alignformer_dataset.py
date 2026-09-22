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
    _project_boxes_to_ego,
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


def test_truncating_before_matching_drops_a_truncated_partner_cleanly(tmp_path, monkeypatch):
    """R26.7 / review HIGH-2: pin __getitem__'s truncate-THEN-match order FOR REAL.

    The first version of this test (caught by review) called
    ``_truncate_by_score`` and ``correspondence_indices`` directly and
    asserted what the CORRECT order *would* produce -- a restatement of the
    argument, never routed through ``__getitem__`` itself. The reviewer
    proved this vacuous by reordering ``__getitem__`` to match-before-truncate
    and showing the whole dataset test file, including that test, still
    passed unchanged.

    This version calls ``dataset[0]`` directly. ``trunk.MAX_OBJECTS`` (64) is
    monkeypatched down to 2 so a 3-detection cav frame actually triggers
    truncation without needing to write 65 synthetic boxes. The cav
    detection that matches an ego object ("42") is deliberately the
    lowest-scoring of the three, so it is exactly the one truncation drops.
    """
    monkeypatch.setattr("embedding_aware_belt_fusion.alignformer.dataset.MAX_OBJECTS", 2)

    scenario, timestamp, ego_id, cav_id = "scenario_trunc", "000001", "ego", "cav"
    ego_boxes = np.array(
        [[0.0, 0.0, 0.0, 1.5, 1.5, 4.0, 0.0], [1.0, 1.0, 0.0, 1.5, 1.5, 4.0, 0.0]]
    )
    # 3 cav detections > the monkeypatched MAX_OBJECTS=2. "42" (score 0.1) is
    # the LOWEST-scoring, so truncation to top-2 by score drops exactly it,
    # keeping "99" and "7".
    cav_boxes = np.array([
        [5.0, 5.0, 0.0, 1.5, 1.5, 4.0, 0.0],
        [6.0, 6.0, 0.0, 1.5, 1.5, 4.0, 0.0],
        [7.0, 7.0, 0.0, 1.5, 1.5, 4.0, 0.0],
    ])

    write_frame(
        cache_path(tmp_path, "train", scenario, ego_id, timestamp),
        FrameRecord(
            boxes=ego_boxes.astype(np.float32),
            scores=np.array([1.0, 1.0], dtype=np.float32),
            gt_ids=["42", None],
            roi=np.zeros((2, 2, 2, 2), dtype=np.float16),
        ),
    )
    write_frame(
        cache_path(tmp_path, "train", scenario, cav_id, timestamp),
        FrameRecord(
            boxes=cav_boxes.astype(np.float32),
            scores=np.array([1.0, 0.1, 0.9], dtype=np.float32),
            gt_ids=["99", "42", "7"],
            roi=np.zeros((3, 2, 2, 2), dtype=np.float16),
        ),
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
        [pair], tmp_path, "train", noise_schedule=NoiseSchedule(max_xy_std=0.0), train=False
    )
    sample = dataset[0]

    # "42"'s only cav-side match was truncated away, so ego index 0 ("42")
    # must report no match against the FINAL, truncated arrays -- not a
    # stale index that happens to still be in range but now names a
    # different, surviving object ("7", which "42" never matched).
    assert sample["ego_match"].tolist() == [-1, -1]
    assert sample["cav_match"].tolist() == [-1, -1]
    assert sample["cav_boxes"].shape[0] == 2  # truncation to MAX_OBJECTS=2 did happen


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


def _build_dataset(tmp_path, sign=1.0, total_epochs=1):
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
        total_epochs=total_epochs,
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


def test_projection_rotates_the_yaw_column_by_the_relative_heading():
    """R30: the yaw column (index 6) must be exactly ``yaw + (agent_yaw - ego_yaw)``.

    Poses chosen so the relative heading (agent -15deg minus ego 30deg, i.e.
    -15 - 30 = -45deg) is non-zero: a projection with zero relative rotation
    would pass even if the ``+ dyaw`` term were deleted entirely, so it is not
    a valid test of the yaw update. The expected value is hand-computed here
    as a literal, independent of the implementation under test:

        dyaw = radians(agent_yaw_deg - ego_yaw_deg) = radians(-15 - 30)
             = radians(-45) = -0.7853981633974483
        expected_yaw = input_yaw + dyaw = 0.4 + (-0.7853981633974483)
                     = -0.38539816339744826
    """
    boxes = np.array([[3.0, -2.0, 0.0, 1.5, 1.5, 4.0, 0.4]])

    projected = _project_boxes_to_ego(boxes, _CAV_POSE_TRUE, _EGO_POSE)

    assert projected[0, 6] == pytest.approx(-0.38539816339744826, abs=1e-12)
    # Sanity: the case is not degenerate -- the yaw actually changed.
    assert projected[0, 6] != pytest.approx(boxes[0, 6])


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


def test_set_epoch_actually_changes_the_sampled_noise(tmp_path):
    """R26: end-to-end check that set_epoch's epoch feeds the noise schedule.

    NoiseSchedule.sigma_for_epoch is unit-tested on its own above, but nothing
    end-to-end confirmed that OPV2VObjectSetDataset.set_epoch actually wires
    the epoch it is given through to the noise that gets sampled -- a
    disconnected set_epoch (e.g. one that updates self.epoch but a stale
    closure/copy is what __getitem__ reads) would still pass that unit test.
    At epoch 0 of a 5-epoch ramp the schedule returns sigma == 0, so the
    injected noise is deterministically zero regardless of the RNG draw; at
    the final epoch it returns the full max_xy_std, so the residual t_true
    must be measurably non-zero.
    """
    # Review LOW-6: total_epochs is passed through the public constructor
    # (not set as a post-construction attribute), so this test is coupled to
    # OPV2VObjectSetDataset's public API, not an internal attribute name.
    dataset, _ = _build_dataset(tmp_path, total_epochs=5)

    dataset.set_epoch(0)
    sample_first_epoch = dataset[0]

    dataset.set_epoch(4)
    sample_last_epoch = dataset[0]

    np.testing.assert_allclose(
        sample_first_epoch["t_true"].numpy(), np.zeros(2), atol=1e-6
    )
    assert sample_first_epoch["psi_true"].item() == pytest.approx(0.0, abs=1e-6)

    assert torch.linalg.norm(sample_last_epoch["t_true"]).item() > 1e-3
