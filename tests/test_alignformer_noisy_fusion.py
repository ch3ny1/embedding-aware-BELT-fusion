"""The noisy-AP sweep's wiring, where a silent mistake would look like a result.

The sweep itself is a multi-hour run over the OPV2V test split and is measured
by ``outputs/alignformer/p2_noisy_ap.json``, not by a test. What is tested here
are the four places a bug would be invisible in that output:

- the object set AlignFormer scores must pair each box with *its own* ROI
  features after truncation reorders by score;
- the boxes fed to the model and the boxes that are fused must be moved by one
  shared SE(2) implementation, not two;
- "shares an object" must not treat two unmatched detections as the same
  object, which would make structurally unalignable pairs look alignable;
- each sigma in the sweep must draw independent noise, or the sweep reports one
  perturbation rescaled rather than a curve.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embedding_aware_belt_fusion.alignformer.boxes import AgentDetections
from embedding_aware_belt_fusion.alignformer.fusion import correct_boxes, correct_detections
from embedding_aware_belt_fusion.alignformer.noisy_fusion import (
    ALIGNFORMER,
    ORACLE,
    ORACLE_MATCH_IVW,
    ORACLE_MATCH_UNIFORM,
    _object_set,
    _oracle_variance_models,
    _shares_an_object,
    _sweep_rng,
    _truncate_by_score,
    condition_key,
)
from embedding_aware_belt_fusion.alignformer.variance import (
    UNWEIGHTED,
    CorrespondenceVarianceModel,
)


def _detections(scores, *, channels: int = 2) -> AgentDetections:
    count = len(scores)
    boxes = torch.arange(count * 7, dtype=torch.float32).reshape(count, 7)
    return AgentDetections(
        boxes=boxes,
        scores=torch.tensor(scores, dtype=torch.float32),
        corners=torch.zeros(count, 8, 3),
        gt_ids=[str(index) for index in range(count)],
        features=torch.zeros(channels, 4, 4),
    )


def test_truncation_keeps_every_box_with_its_own_roi_features_and_id():
    detections = _detections([0.1, 0.9, 0.5, 0.7])
    # Row i of the ROI stack is filled with i, so a mispairing is visible.
    roi = torch.arange(4, dtype=torch.float32).reshape(4, 1, 1, 1).expand(4, 2, 4, 4)

    boxes, scores, kept_roi, gt_ids = _truncate_by_score(detections, roi, 2)

    assert torch.equal(scores, detections.scores[[1, 3]])
    # Original rows 1 and 3, in that order, everywhere.
    assert torch.equal(boxes, detections.boxes[[1, 3]])
    assert kept_roi[:, 0, 0, 0].tolist() == [1.0, 3.0]
    assert gt_ids == ["1", "3"]


def test_truncation_is_a_no_op_below_the_budget():
    detections = _detections([0.1, 0.9])
    roi = torch.zeros(2, 2, 4, 4)

    boxes, scores, kept_roi, gt_ids = _truncate_by_score(detections, roi, 64)

    assert torch.equal(boxes, detections.boxes)
    assert torch.equal(scores, detections.scores)
    assert kept_roi.shape == roi.shape
    assert gt_ids == list(detections.gt_ids)


def test_the_model_input_and_the_fused_boxes_move_by_the_same_se2():
    # The sweep projects the TRUNCATED box tensor for the model and the FULL
    # AgentDetections for fusion. If those two ever disagree, the model is
    # scoring a differently-placed object set than the one being fused, and
    # nothing in the AP number would show it.
    detections = _detections([0.9, 0.8, 0.7])
    psi = torch.tensor(0.37)
    translation = torch.tensor([1.5, -2.25])

    from_detections = correct_detections(detections, psi, translation).boxes
    from_boxes = correct_boxes(detections.boxes, psi, translation)

    assert torch.allclose(from_detections, from_boxes)
    # And neither mutated its input.
    assert torch.equal(detections.boxes, _detections([0.9, 0.8, 0.7]).boxes)


def test_two_unmatched_detections_are_not_the_same_object():
    # None means "matched no ground-truth object". Treating two of them as a
    # shared object would report a structurally unalignable pair as alignable.
    assert not _shares_an_object([None, None], [None])
    assert not _shares_an_object(["7"], ["9", None])
    assert _shares_an_object(["7", None], [None, "7"])


def test_each_sigma_draws_independent_noise():
    first = _sweep_rng(0, 0.5, 3, 0).normal(size=4)
    second = _sweep_rng(0, 1.0, 3, 0).normal(size=4)
    again = _sweep_rng(0, 0.5, 3, 0).normal(size=4)

    # Reproducible for a given (seed, sigma, frame, agent) ...
    assert np.array_equal(first, again)
    # ... and not merely the same draw rescaled across sigmas.
    assert not np.allclose(first / np.linalg.norm(first), second / np.linalg.norm(second))


def test_a_condition_key_names_its_sigma_except_for_the_oracle():
    assert condition_key(ORACLE, 1.5) == "oracle"
    assert condition_key(ALIGNFORMER, 1.5) == "alignformer_sigma_1.5m"
    assert condition_key(ALIGNFORMER, 0.0) == "alignformer_sigma_0m"
    # The oracle-correspondence ceiling is a per-sigma condition like
    # alignformer, not a sigma-independent one like `oracle`: its input is the
    # noisily-projected CAV box set, only its correspondence is free.
    assert condition_key(ORACLE_MATCH_IVW, 0.4) == "oracle_match_ivw_sigma_0.4m"
    assert condition_key(ORACLE_MATCH_UNIFORM, 2.0) == "oracle_match_uniform_sigma_2m"


def test_the_oracle_ceiling_reuses_the_checkpoints_own_inverse_variance_model():
    # Read off the loaded head, never rebuilt from a config: the IVW variant is
    # the like-for-like comparison only if it weights correspondences exactly
    # the way the checkpoint being compared against does.
    deployed = CorrespondenceVarianceModel(
        mode="scalar", sigma_translation_m=0.2516, translation_exponent=1.1281,
        sigma_yaw_rad=0.081, yaw_exponent=1.9322,
    )
    modules = {"pose": SimpleNamespace(variance_model=deployed)}

    models = _oracle_variance_models(modules)

    assert models[ORACLE_MATCH_IVW] is deployed
    assert models[ORACLE_MATCH_UNIFORM] is UNWEIGHTED
    assert not UNWEIGHTED.enabled  # weight 1.0 on each true pair


def test_a_head_with_no_correspondence_cannot_have_its_matching_oracled():
    # Head A regresses the pose from a pooled descriptor; there is no
    # correspondence matrix to substitute, so asking for the ceiling is a
    # mistake that must surface rather than silently measure something else.
    with pytest.raises(ValueError, match="head B"):
        _oracle_variance_models({"pose": SimpleNamespace()})


def test_the_object_set_batch_marks_every_real_object_valid():
    ego = {
        "boxes": torch.randn(3, 7),
        "scores": torch.rand(3),
        "roi": torch.randn(3, 2, 4, 4),
    }
    cav = {
        "boxes": torch.randn(5, 7),
        "scores": torch.rand(5),
        "roi": torch.randn(5, 2, 4, 4),
    }

    batch = _object_set(ego, cav)

    assert batch["ego_boxes"].shape == (1, 3, 7)
    assert batch["cav_roi"].shape == (1, 5, 2, 4, 4)
    assert bool(batch["ego_mask"].all()) and batch["ego_mask"].shape == (1, 3)
    assert bool(batch["cav_mask"].all()) and batch["cav_mask"].shape == (1, 5)


def test_the_pose_stats_split_pairs_by_the_training_communication_range():
    # The guard against a train/test range mismatch: whatever range the pair
    # index was built at, the pairs outside it are out of distribution by
    # construction and an aggregate hides them. configs/alignformer.yaml now
    # matches OpenCOOD's COM_RANGE, so this subset should be empty in practice
    # -- which is the point. It stays measured so a future divergence shows up
    # as a number rather than as a mystery.
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import _PoseStats

    stats = _PoseStats(40.0)
    stats.update(0.0, (0.0, 0.0), 0.0, (1.0, 0.0),
                 fell_back=True, alignable=True, distance_m=12.0)
    stats.update(0.0, (0.0, 0.0), 0.0, (3.0, 0.0),
                 fell_back=True, alignable=True, distance_m=55.0)

    metrics = stats.compute()

    assert metrics["pairs"] == 2.0
    assert metrics["training_comm_range_m"] == 40.0
    assert metrics["beyond_training_range_fraction"] == 0.5
    assert metrics["within_training_range_translation_mae_m"] == pytest.approx(1.0)
    assert metrics["beyond_training_range_translation_mae_m"] == pytest.approx(3.0)
    # A subset with no members reports None, not a zero it never measured.
    assert metrics["unalignable_translation_mae_m"] is None


def test_the_pose_stats_also_split_at_the_fixed_40m_diagnostic_boundary():
    # Separate from the training-range split above, which moves with the
    # config. Every earlier measurement in docs/alignformer_pose_floor.md is
    # reported either side of 40 m, and once the pair index is built at 70 m
    # the training-range split collapses to "everything" and that comparison
    # would silently vanish. A FIXED boundary keeps the before/after readable.
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import (
        DIAGNOSTIC_RANGE_M,
        _PoseStats,
    )

    # Arrange: a training range that no longer coincides with the boundary
    stats = _PoseStats(70.0)
    stats.update(0.0, (0.0, 0.0), 0.0, (1.0, 0.0),
                 fell_back=False, alignable=True, distance_m=12.0)
    stats.update(0.0, (0.0, 0.0), 0.0, (3.0, 0.0),
                 fell_back=False, alignable=True, distance_m=55.0)

    # Act
    metrics = stats.compute()

    # Assert
    assert DIAGNOSTIC_RANGE_M == 40.0
    # Both pairs are inside the training range now, so that split says nothing.
    assert metrics["within_training_range_pairs"] == 2.0
    assert metrics["beyond_training_range_pairs"] == 0.0
    # The fixed boundary still separates them.
    assert metrics["within_40m_translation_mae_m"] == pytest.approx(1.0)
    assert metrics["beyond_40m_translation_mae_m"] == pytest.approx(3.0)
    assert metrics["beyond_40m_fraction"] == 0.5
