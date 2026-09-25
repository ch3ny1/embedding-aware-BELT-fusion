import json
import math

import pytest
import torch

from embedding_aware_belt_fusion.alignformer.model import PoseEstimate
from embedding_aware_belt_fusion.alignformer.shrinkage import (
    POSE_DIMENSIONS,
    ShrinkageCalibration,
    calibrate_shrinkage,
    shrink,
    shrinkage_factor,
    standardized_magnitude_squared,
)

TAU_T = 0.25
TAU_YAW_DEG = 0.3


def _estimate(psi, translations):
    return PoseEstimate(
        psi=torch.tensor(psi),
        t=torch.tensor(translations),
        confidence=torch.full((len(psi),), 8.0),
    )


def _calibration(tau_t=TAU_T, tau_yaw_deg=TAU_YAW_DEG):
    return ShrinkageCalibration(
        tau_translation_m=tau_t,
        tau_yaw_rad=math.radians(tau_yaw_deg),
        pairs=4176,
        split="validation (scenario-disjoint, split_seed 0)",
        sigma_m=0.0,
    )


def test_a_correction_far_larger_than_the_noise_is_kept_essentially_whole():
    # Arrange: a 5 m correction against a 0.25 m noise floor.
    estimate = _estimate([0.0], [[5.0, 0.0]])

    # Act
    factor = shrinkage_factor(
        standardized_magnitude_squared(estimate.psi, estimate.t, _calibration())
    )

    # Assert
    assert factor.item() == pytest.approx(1.0 - 3.0 / 800.0, abs=1e-6)
    assert factor.item() > 0.99


def test_a_correction_the_size_of_the_noise_itself_is_suppressed_entirely():
    # z^2 = p is exactly the expected value under no true error at all, so the
    # estimated sigma^2 is zero and the whole correction goes.
    calibration = _calibration()
    # 2 |t|^2 / tau_t^2 = 3 with no yaw component.
    magnitude = TAU_T * math.sqrt(POSE_DIMENSIONS / 2.0)
    estimate = _estimate([0.0, 0.0], [[magnitude, 0.0], [magnitude / 2, 0.0]])

    shrunk = shrink(estimate, calibration)

    assert shrunk.t.abs().sum().item() == pytest.approx(0.0, abs=1e-6)


def test_the_factor_is_monotone_in_the_standardized_magnitude():
    squared = torch.tensor([0.0, 1.0, 3.0, 6.0, 12.0, 50.0, 1000.0])

    factors = shrinkage_factor(squared)

    assert torch.all(factors[1:] >= factors[:-1])
    assert torch.all(factors >= 0.0) and torch.all(factors <= 1.0)


def test_the_identity_fallback_stays_the_identity_without_a_nan():
    # A suppressed pair emits exactly (0, 0); 1 - p / 0 must not divide by zero.
    estimate = _estimate([0.0], [[0.0, 0.0]])

    shrunk = shrink(estimate, _calibration())

    assert torch.isfinite(shrunk.psi).all() and torch.isfinite(shrunk.t).all()
    assert shrunk.psi.item() == 0.0
    assert shrunk.t.abs().sum().item() == 0.0


def test_translation_evidence_rescues_a_yaw_that_alone_would_be_suppressed():
    # The point of pooling. A yaw of half the calibrated yaw noise carries no
    # evidence by itself and a per-component rule would delete it; alongside a
    # 3 m translation the same yaw is plainly part of a real pose error and
    # must survive nearly intact.
    calibration = _calibration()
    small_yaw = calibration.tau_yaw_rad / 2

    alone = shrink(_estimate([small_yaw], [[0.0, 0.0]]), calibration)
    pooled = shrink(_estimate([small_yaw], [[3.0, 0.0]]), calibration)

    assert alone.psi.item() == 0.0
    assert pooled.psi.item() == pytest.approx(small_yaw, rel=0.05)


def test_one_factor_moves_the_yaw_and_the_translation_together():
    estimate = _estimate([0.05], [[3.0, -4.0]])

    shrunk = shrink(estimate, _calibration())

    yaw_ratio = shrunk.psi.item() / estimate.psi.item()
    translation_ratio = (
        torch.linalg.norm(shrunk.t) / torch.linalg.norm(estimate.t)
    ).item()
    assert yaw_ratio == pytest.approx(translation_ratio, abs=1e-6)


def test_shrinking_preserves_direction_and_never_amplifies():
    estimate = _estimate([0.05, -0.05], [[3.0, -4.0], [0.2, 0.1]])

    shrunk = shrink(estimate, _calibration())

    for row in range(2):
        original, moved = estimate.t[row], shrunk.t[row]
        assert torch.linalg.norm(moved) <= torch.linalg.norm(original) + 1e-9
        if torch.linalg.norm(moved) > 0:
            cosine = torch.dot(original, moved) / (
                torch.linalg.norm(original) * torch.linalg.norm(moved)
            )
            assert cosine.item() == pytest.approx(1.0, abs=1e-6)
    assert abs(shrunk.psi[0].item()) <= abs(estimate.psi[0].item())


def test_shrink_does_not_mutate_the_estimate_it_is_given():
    estimate = _estimate([0.05], [[3.0, -4.0]])
    before_psi = estimate.psi.clone()
    before_t = estimate.t.clone()

    shrink(estimate, _calibration())

    assert torch.equal(estimate.psi, before_psi)
    assert torch.equal(estimate.t, before_t)


def test_the_confidence_and_assignment_survive_shrinking():
    estimate = PoseEstimate(
        psi=torch.tensor([0.05]),
        t=torch.tensor([[3.0, -4.0]]),
        confidence=torch.tensor([7.5]),
        log_assignment=torch.zeros(1, 3, 3),
    )

    shrunk = shrink(estimate, _calibration())

    assert shrunk.confidence.item() == 7.5
    assert shrunk.log_assignment is estimate.log_assignment


def test_a_pure_noise_correction_is_suppressed_on_average():
    # The guarantee the fix exists for: at sigma = 0 the truth is the identity,
    # so a correction drawn from the estimator's own noise must mostly vanish.
    torch.manual_seed(0)
    count = 20000
    calibration = _calibration()
    per_axis = TAU_T / math.sqrt(2)
    estimate = PoseEstimate(
        psi=torch.randn(count) * calibration.tau_yaw_rad,
        t=torch.randn(count, 2) * per_axis,
        confidence=torch.full((count,), 8.0),
    )

    shrunk = shrink(estimate, calibration)

    raw = float((estimate.t ** 2).sum(dim=-1).mean())
    left = float((shrunk.t ** 2).sum(dim=-1).mean())
    assert left < 0.35 * raw


def test_calibration_recovers_the_scale_of_the_residuals_it_is_given():
    # Arrange: zero-mean residuals of a known scale, as measured at sigma = 0
    # where the true correction is exactly the identity.
    torch.manual_seed(0)
    residual_t = torch.randn(200000, 2) * 0.3        # per-axis std 0.3 m
    residual_psi = torch.randn(200000) * 0.004       # 0.004 rad

    # Act
    calibration = calibrate_shrinkage(
        residual_t, residual_psi, pairs=200000, split="synthetic", sigma_m=0.0
    )

    # Assert: tau is the RMS NORM for translation (2 axes) and the RMS for yaw.
    assert calibration.tau_translation_m == pytest.approx(0.3 * math.sqrt(2), rel=0.01)
    assert calibration.tau_yaw_rad == pytest.approx(0.004, rel=0.01)


def test_calibration_round_trips_through_json():
    calibration = _calibration()

    restored = ShrinkageCalibration.from_dict(json.loads(json.dumps(calibration.to_dict())))

    assert restored == calibration


def test_a_negative_tau_is_rejected():
    with pytest.raises(ValueError):
        ShrinkageCalibration(
            tau_translation_m=-1.0, tau_yaw_rad=0.005, pairs=10, split="x", sigma_m=0.0
        )


# --- Task 23 part B: calibrating tau for the IRLS arm ------------------------
#
# The deployed tau (0.151 m) was fitted on the *un*-robust solve's residuals.
# The IRLS arm is a different estimator with a different residual distribution,
# and scoring it through the old calibration under-corrects it by an unmeasured
# amount. Refitting means running the very same estimator the sweep runs when
# the residuals are collected, so the calibration path needs the same
# `robust=` switch the sweep has -- and must reproduce the deployed calibration
# exactly when that switch is off, or the old number stops being reproducible.


def _residual_modules_and_loader(outlier_index=None):
    """A head-B model, a frozen-shaped embedding stub, and a one-batch loader.

    The embedding head is the identity on the ROI stack so the test controls
    the correspondence exactly: the residual then depends on the pose solve
    alone, which is the thing `robust=` changes.
    """
    from torch import nn

    from embedding_aware_belt_fusion.alignformer.model import AlignFormerB

    from test_alignformer_model import _boxes_from_centres

    class _FlattenEmbedding(nn.Module):
        """Sums each ROI channel away, so a one-hot stack stays one-hot."""

        def __init__(self, dim: int):
            super().__init__()
            self.dim = dim

        def forward(self, roi):  # (N, C, H, W) -> (N, C)
            return roi.flatten(start_dim=1).reshape(roi.shape[0], self.dim, -1).sum(-1)

    torch.manual_seed(0)
    count = 8
    centres = torch.tensor([[[0.0, 0.0], [12.0, 3.0], [-8.0, 5.0], [20.0, -7.0],
                             [4.0, 9.0], [-15.0, -2.0], [7.0, -12.0], [-3.0, 16.0]]])
    yaws = torch.rand(1, count) * 2 * math.pi
    cav_boxes = _boxes_from_centres(centres, yaws)

    true_psi, true_t = 0.15, torch.tensor([[1.2, -0.8]])
    cos, sin = math.cos(true_psi), math.sin(true_psi)
    rotation = torch.tensor([[cos, -sin], [sin, cos]])
    ego_boxes = cav_boxes.clone()
    ego_boxes[..., :2] = centres @ rotation.T + true_t
    ego_boxes[..., 6] = yaws + true_psi
    if outlier_index is not None:
        cav_boxes[0, outlier_index, :2] += torch.tensor([14.0, -17.0])

    # One-hot ROI stacks: the flattening stub turns them into one-hot embeddings.
    roi = torch.eye(count).reshape(1, count, count, 1, 1)
    model = AlignFormerB(embed_dim=count).eval()
    model.use_raw_embedding_scores = True
    modules = nn.ModuleDict({"embedding": _FlattenEmbedding(count), "pose": model})
    batch = {
        "ego_boxes": ego_boxes,
        "ego_scores": torch.ones(1, count),
        "ego_roi": roi.clone(),
        "ego_mask": torch.ones(1, count, dtype=torch.bool),
        "cav_boxes": cav_boxes,
        "cav_scores": torch.ones(1, count),
        "cav_roi": roi.clone(),
        "cav_mask": torch.ones(1, count, dtype=torch.bool),
        "psi_true": torch.tensor([true_psi]),
        "t_true": true_t.clone(),
    }
    return modules, [batch]


def test_residual_collection_without_a_robust_config_is_the_deployed_one():
    """The published tau = 0.151 m must stay reproducible after this change."""
    from embedding_aware_belt_fusion.alignformer.robust import RobustSolveConfig
    from embedding_aware_belt_fusion.alignformer.stage2 import collect_pose_residuals

    modules, loader = _residual_modules_and_loader(outlier_index=0)
    device = torch.device("cpu")

    plain_t, plain_psi = collect_pose_residuals(modules, loader, device)
    disabled_t, disabled_psi = collect_pose_residuals(
        modules, loader, device, robust=RobustSolveConfig()
    )

    assert torch.equal(plain_t, disabled_t)
    assert torch.equal(plain_psi, disabled_psi)


def test_the_irls_arm_is_calibrated_on_its_own_residuals_not_the_deployed_ones():
    from embedding_aware_belt_fusion.alignformer.robust import (
        HUBER,
        RobustSolveConfig,
    )
    from embedding_aware_belt_fusion.alignformer.stage2 import collect_pose_residuals

    modules, loader = _residual_modules_and_loader(outlier_index=0)
    device = torch.device("cpu")
    loop = RobustSolveConfig(mode=HUBER, iterations=3, min_evidence=3.0)

    plain_t, _ = collect_pose_residuals(modules, loader, device)
    robust_t, _ = collect_pose_residuals(modules, loader, device, robust=loop)

    assert not torch.equal(plain_t, robust_t)
    # The loop rejects the blunder, so its residual is the smaller one -- which
    # is exactly why a tau fitted on the deployed arm over-shrinks it.
    assert float(torch.linalg.norm(robust_t)) < float(torch.linalg.norm(plain_t))


def test_the_calibration_helper_threads_the_loop_through_to_the_solve():
    from embedding_aware_belt_fusion.alignformer.robust import (
        HUBER,
        RobustSolveConfig,
    )
    from embedding_aware_belt_fusion.alignformer.stage2 import calibrate_from_loader

    modules, loader = _residual_modules_and_loader(outlier_index=0)
    device = torch.device("cpu")

    deployed, _ = calibrate_from_loader(modules, loader, device, split="unit")
    refitted, _ = calibrate_from_loader(
        modules, loader, device, split="unit",
        robust=RobustSolveConfig(mode=HUBER, iterations=3, min_evidence=3.0),
    )

    assert refitted.tau_translation_m < deployed.tau_translation_m
    assert refitted.pairs == deployed.pairs
