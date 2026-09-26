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
from embedding_aware_belt_fusion.alignformer.abstain import (
    NONE as ABSTAIN_NONE,
    PER_PAIR,
    AbstentionConfig,
)
from embedding_aware_belt_fusion.alignformer.draws import sweep_draws
from embedding_aware_belt_fusion.alignformer.noisy_fusion import (
    ALIGNFORMER,
    ORACLE,
    ORACLE_MATCH_IVW,
    ORACLE_MATCH_UNIFORM,
    _object_set,
    _oracle_variance_models,
    _require_fitted_variance_model,
    _sweep_rng,
    _truncate_by_score,
    condition_key,
    shared_object_count,
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


def test_the_pose_stats_slice_pairs_by_shared_object_count_and_report_coverage():
    # Task 20. FreeAlign's evidence is pairwise distances, so one shared object
    # is a graph with no edge and two give a single scalar; three is where a
    # distance graph stops being degenerate, which is where the boundary sits.
    # Coverage is reported beside the error because a method that abstains
    # often looks good on the pairs it does answer.
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import (
        SHARED_BUCKETS,
        _PoseStats,
        shared_object_bucket,
    )

    assert SHARED_BUCKETS == ("shared_0", "shared_1_2", "shared_3plus")
    assert [shared_object_bucket(n) for n in (0, 1, 2, 3, 40)] == [
        "shared_0", "shared_1_2", "shared_1_2", "shared_3plus", "shared_3plus"
    ]

    stats = _PoseStats(70.0)
    # A declined sparse pair, an answered sparse pair, an answered dense one.
    stats.update(0.0, (0.0, 0.0), 0.0, (2.0, 0.0),
                 fell_back=True, shared_count=2, distance_m=10.0)
    stats.update(0.0, (0.4, 0.0), 0.0, (0.0, 0.0),
                 fell_back=False, shared_count=1, distance_m=10.0)
    stats.update(0.0, (0.1, 0.0), 0.0, (0.0, 0.0),
                 fell_back=False, shared_count=9, distance_m=10.0)

    metrics = stats.compute()

    assert metrics["coverage"] == pytest.approx(2 / 3)
    assert metrics["shared_1_2_pairs"] == 2.0
    assert metrics["shared_1_2_coverage"] == 0.5
    # Over ALL sparse pairs the declined one contributes its 2 m of uncorrected
    # error; over the ANSWERED ones only, it does not. Both are reported,
    # because FreeAlign's published pose figures are conditioned on the latter.
    assert metrics["shared_1_2_translation_mae_m"] == pytest.approx(1.2)
    assert metrics["shared_1_2_answered_translation_mae_m"] == pytest.approx(0.4)
    assert metrics["shared_3plus_coverage"] == 1.0
    assert metrics["shared_0_pairs"] == 0.0


def test_the_freealign_condition_is_named_and_keyed_like_every_other():
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import (
        FREEALIGN,
        condition_key,
    )

    assert FREEALIGN == "freealign"
    assert condition_key(FREEALIGN, 0.4) == "freealign_sigma_0.4m"


def test_two_unmatched_detections_are_not_the_same_object():
    # None means "matched no ground-truth object". Treating two of them as a
    # shared object would report a structurally unalignable pair as alignable.
    assert shared_object_count([None, None], [None]) == 0
    assert shared_object_count(["7"], ["9", None]) == 0
    assert shared_object_count(["7", None], [None, "7"]) == 1
    assert shared_object_count(["7", "9", "4"], ["9", "7", None]) == 2


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
                 fell_back=True, shared_count=4, distance_m=12.0)
    stats.update(0.0, (0.0, 0.0), 0.0, (3.0, 0.0),
                 fell_back=True, shared_count=4, distance_m=55.0)

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
                 fell_back=False, shared_count=4, distance_m=12.0)
    stats.update(0.0, (0.0, 0.0), 0.0, (3.0, 0.0),
                 fell_back=False, shared_count=4, distance_m=55.0)

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


def test_the_irls_condition_only_joins_the_sweep_when_a_robust_config_is_given():
    # Task 22. A fifth correction arm whose condition keys collide with, or
    # silently replace, the deployed arm's would make the bit-identity check
    # that proves nothing else moved impossible to run.
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import (
        ALIGNFORMER_IRLS,
        sweep_estimators,
    )
    from embedding_aware_belt_fusion.alignformer.robust import (
        HUBER,
        RobustSolveConfig,
    )

    without = sweep_estimators(oracle_match=[], freealign=False, robust=None)
    with_loop = sweep_estimators(
        oracle_match=[], freealign=False,
        robust=RobustSolveConfig(mode=HUBER, iterations=2),
    )
    disabled = sweep_estimators(
        oracle_match=[], freealign=False, robust=RobustSolveConfig(),
    )

    assert without == [ALIGNFORMER]
    assert with_loop == [ALIGNFORMER, ALIGNFORMER_IRLS]
    assert disabled == [ALIGNFORMER]
    assert condition_key(ALIGNFORMER_IRLS, 1.0) != condition_key(ALIGNFORMER, 1.0)


# --- Task 23: multi-seed draws, and the pairing they depend on ----------------
#
# Every AP cell on this branch was one noise draw, and the FreeAlign head-to-head
# at AP@0.7 turns on differences one draw cannot resolve. The sweep now runs
# several independent draws per sigma. Two properties make the resulting error
# bars mean anything, and neither is visible in the multi-hour output:
#
# - a draw is reproducible from its seed alone, so a rerun is the same
#   experiment and the single-seed rows still reproduce the published files;
# - every condition inside one draw sees the SAME perturbed poses, so the
#   per-seed difference between two conditions is a paired statistic. Unpaired
#   draws would leave the conditions varying independently and the error bar
#   would be several times too wide.


def test_the_same_seed_reproduces_the_same_draw():
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import sweep_noisy_poses

    poses = {"cav0": [10.0, -4.0, 1.0, 0.0, 30.0, 0.0], "cav1": [1.0, 2.0, 1.0, 0.0, -5.0, 0.0]}
    keys = ["cav0", "cav1"]

    first = sweep_noisy_poses(poses, keys, sigma=0.8, seed=1000, frame=7)
    again = sweep_noisy_poses(poses, keys, sigma=0.8, seed=1000, frame=7)

    assert first == again
    # and it actually moved the pose, or the test above is vacuous.
    assert first["cav0"] != poses["cav0"]


def test_independent_seeds_draw_independent_noise():
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import sweep_noisy_poses

    poses = {"cav0": [10.0, -4.0, 1.0, 0.0, 30.0, 0.0]}

    first = sweep_noisy_poses(poses, ["cav0"], sigma=0.8, seed=1000, frame=7)
    second = sweep_noisy_poses(poses, ["cav0"], sigma=0.8, seed=1001, frame=7)

    assert first["cav0"] != second["cav0"]


def test_every_agent_in_one_draw_gets_its_own_perturbation():
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import sweep_noisy_poses

    # Two agents at the SAME pose: only a per-agent key can separate them.
    poses = {"cav0": [1.0, 2.0, 1.0, 0.0, 3.0, 0.0], "cav1": [1.0, 2.0, 1.0, 0.0, 3.0, 0.0]}

    drawn = sweep_noisy_poses(poses, ["cav0", "cav1"], sigma=1.0, seed=1000, frame=0)

    assert drawn["cav0"] != drawn["cav1"]


def test_at_sigma_zero_every_seed_draws_the_identical_unperturbed_pose():
    """Which is why sigma = 0 is run once and reported without a spread."""
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import sweep_noisy_poses

    poses = {"cav0": [10.0, -4.0, 1.0, 0.0, 30.0, 0.0]}

    drawn = [
        sweep_noisy_poses(poses, ["cav0"], sigma=0.0, seed=seed, frame=3)["cav0"]
        for seed in (1000, 1001, 1002, 1003, 1004)
    ]

    assert all(pose == drawn[0] for pose in drawn)
    assert drawn[0] == pytest.approx(poses["cav0"])


def test_sigma_zero_is_drawn_once_and_every_other_sigma_once_per_seed():
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import (
        draw_seeds,
        sweep_draws,
    )

    seeds = [1000, 1001, 1002]

    assert draw_seeds(0.0, seeds) == [1000]
    assert draw_seeds(0.2, seeds) == seeds

    draws = sweep_draws([0.0, 0.2], seeds)
    assert draws == [(0.0, 1000), (0.2, 1000), (0.2, 1001), (0.2, 1002)]


def test_a_sweep_needs_at_least_one_seed():
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import sweep_draws

    with pytest.raises(ValueError, match="at least one seed"):
        sweep_draws([0.0, 1.0], [])


def test_a_sweep_refuses_a_repeated_seed():
    """Two identical draws reported as two seeds would halve the error bar."""
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import sweep_draws

    with pytest.raises(ValueError, match="distinct"):
        sweep_draws([1.0], [1000, 1000])


def _estimate(psi: float, tx: float, ty: float):
    from embedding_aware_belt_fusion.alignformer.model import PoseEstimate

    return PoseEstimate(
        psi=torch.tensor([psi]),
        t=torch.tensor([[tx, ty]]),
        confidence=torch.tensor([10.0]),
    )


def test_every_condition_in_one_draw_corrects_the_same_perturbed_detections():
    """The pairing, at the one place it could be broken.

    Each condition's fused boxes must be its own estimate applied to the *one*
    perturbed set the draw produced. If any condition re-drew, or was handed a
    different agent's set, the per-seed difference between conditions would stop
    being paired and every error bar downstream would be wrong.
    """
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import (
        ALIGNFORMER,
        ALIGNFORMER_IRLS,
        FREEALIGN,
        UNSHRUNK_CONDITIONS,
        draw_aligned,
    )

    noisy = _detections([0.9, 0.5, 0.3])
    estimates = {
        ALIGNFORMER: _estimate(0.10, 1.0, 2.0),
        ALIGNFORMER_IRLS: _estimate(0.05, 0.5, 1.0),
        FREEALIGN: _estimate(-0.02, -0.25, 0.75),
    }

    aligned = draw_aligned(
        noisy, estimates, shrinkage=None, unshrunk=UNSHRUNK_CONDITIONS
    )

    assert set(aligned) == set(estimates)
    for name, estimate in estimates.items():
        expected = correct_detections(noisy, estimate.psi[0], estimate.t[0])
        assert torch.equal(aligned[name].detections.boxes, expected.boxes)
        # Same scores, same features, same ids: one perturbed set, three views.
        assert torch.equal(aligned[name].detections.scores, noisy.scores)
        assert aligned[name].detections.gt_ids == list(noisy.gt_ids)
        # And the estimate carried alongside is the one the boxes were moved by,
        # so the pose statistics and the AP cannot diverge.
        assert aligned[name].estimate is estimate


def test_the_draw_leaves_the_perturbed_detections_untouched():
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import (
        ALIGNFORMER,
        UNSHRUNK_CONDITIONS,
        draw_aligned,
    )

    noisy = _detections([0.9, 0.5])
    before = noisy.boxes.clone()

    draw_aligned(
        noisy, {ALIGNFORMER: _estimate(0.3, 4.0, 5.0)},
        shrinkage=None, unshrunk=UNSHRUNK_CONDITIONS,
    )

    assert torch.equal(noisy.boxes, before)


def test_shrinkage_reaches_every_arm_of_ours_and_never_the_competitor():
    """FreeAlign has no such step; scoring it through our calibration would be
    scoring the competitor through our method."""
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import (
        ALIGNFORMER,
        ALIGNFORMER_IRLS,
        FREEALIGN,
        UNSHRUNK_CONDITIONS,
        draw_aligned,
    )
    from embedding_aware_belt_fusion.alignformer.shrinkage import (
        ShrinkageCalibration,
        shrink,
    )

    noisy = _detections([0.9, 0.5])
    calibration = ShrinkageCalibration(
        tau_translation_m=0.15, tau_yaw_rad=0.004, pairs=10, split="unit", sigma_m=0.0
    )
    estimates = {
        ALIGNFORMER: _estimate(0.01, 0.20, 0.10),
        ALIGNFORMER_IRLS: _estimate(0.01, 0.20, 0.10),
        FREEALIGN: _estimate(0.01, 0.20, 0.10),
    }

    aligned = draw_aligned(
        noisy, estimates, shrinkage=calibration, unshrunk=UNSHRUNK_CONDITIONS
    )

    for name in (ALIGNFORMER, ALIGNFORMER_IRLS):
        shrunk = shrink(estimates[name], calibration)
        expected = correct_detections(noisy, shrunk.psi[0], shrunk.t[0])
        assert torch.equal(aligned[name].detections.boxes, expected.boxes)
        assert torch.equal(aligned[name].estimate.t, shrunk.t)
    untouched = correct_detections(
        noisy, estimates[FREEALIGN].psi[0], estimates[FREEALIGN].t[0]
    )
    assert torch.equal(aligned[FREEALIGN].detections.boxes, untouched.boxes)
    assert aligned[FREEALIGN].estimate is estimates[FREEALIGN]
    # And the shrinkage did something, or the first assertion is vacuous.
    assert not torch.equal(aligned[ALIGNFORMER].detections.boxes, untouched.boxes)


def test_the_shrinkage_exemption_cannot_be_forgotten():
    """`unshrunk` is keyword-only and required, because a default of "shrink
    everything" would silently score the competitor through our calibration and
    nothing in the output would show it."""
    from embedding_aware_belt_fusion.alignformer.draws import draw_aligned

    with pytest.raises(TypeError):
        draw_aligned(_detections([0.9]), {}, None)  # type: ignore[call-arg]


def test_a_sigma_with_no_draw_gets_ONE_accumulator_shared_by_every_seed():
    """The layout behind both the predictions and the pose statistics."""
    from embedding_aware_belt_fusion.alignformer.draws import draw_slots

    slots = draw_slots(["a", "b"], [0.0, 1.0], [10, 11, 12], list)

    # sigma = 0 has no draw: one object, three references.
    assert slots[("a", 0.0, 10)] is slots[("a", 0.0, 11)] is slots[("a", 0.0, 12)]
    # sigma = 1 does: three independent accumulators.
    assert len({id(slots[("a", 1.0, seed)]) for seed in (10, 11, 12)}) == 3
    # and the two conditions never share one.
    assert slots[("a", 0.0, 10)] is not slots[("b", 0.0, 10)]


def test_an_append_to_a_shared_slot_reaches_every_seed_exactly_once():
    from embedding_aware_belt_fusion.alignformer.draws import draw_slots

    slots = draw_slots(["a"], [0.0], [10, 11], list)
    slots[("a", 0.0, 10)].append("frame")

    assert slots[("a", 0.0, 11)] == ["frame"]


# --- Review round 2: the statistic's scale is a fitted quantity --------------
#
# The Wald statistic the per-pair rules decide on is the correction measured
# against the fit's own covariance, and that covariance is built from the
# checkpoint's fitted correspondence-variance model. `UNWEIGHTED` carries
# sigma = 1.0 placeholders so a config with no fitted block still loads and
# reproduces its own numbers -- but those placeholders are not a calibration,
# and a statistic divided by them is off by whatever the real sigmas were.
# Nothing downstream can see that: the run completes, the result file looks
# ordinary, and every abstention decision in it crossed a mis-scaled bar.


def test_an_abstention_arm_on_an_unfitted_variance_model_is_refused():
    modules = {"pose": SimpleNamespace(variance_model=UNWEIGHTED)}

    with pytest.raises(ValueError, match="fitted correspondence-variance"):
        _require_fitted_variance_model(
            modules, [AbstentionConfig(mode=PER_PAIR)]
        )


def test_an_abstention_arm_on_a_fitted_variance_model_is_allowed():
    fitted = CorrespondenceVarianceModel(
        mode="scalar", sigma_translation_m=0.2516, translation_exponent=1.1281,
        sigma_yaw_rad=0.081, yaw_exponent=1.9322,
    )
    modules = {"pose": SimpleNamespace(variance_model=fitted)}

    _require_fitted_variance_model(modules, [AbstentionConfig(mode=PER_PAIR)])


def test_a_disabled_abstention_arm_does_not_require_a_fitted_model():
    # The guard must not reach past the arms actually enabled: the unweighted
    # sweep is a published configuration and has to keep running.
    modules = {"pose": SimpleNamespace(variance_model=UNWEIGHTED)}

    _require_fitted_variance_model(modules, [AbstentionConfig(mode=ABSTAIN_NONE)])
    _require_fitted_variance_model(modules, [])


def test_a_sweep_cannot_repeat_a_sigma():
    # draw_slots keys on (name, sigma, seed), so a repeated sigma collapses
    # into one accumulator and the second copy silently overwrites the first.
    with pytest.raises(ValueError, match="distinct"):
        sweep_draws([0.0, 0.2, 0.2], [0, 1])
