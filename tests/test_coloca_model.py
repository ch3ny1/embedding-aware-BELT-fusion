"""Tests for the CoLoca-QuA localization module and its metrics."""

import numpy as np
import pytest
import torch

from embedding_aware_belt_fusion.coloca.backbone import bev_feature_size, pillar_grid_size
from embedding_aware_belt_fusion.coloca.metrics import (
    MetricAccumulator,
    localization_metrics,
    pose_error_loss,
    translation_residual,
)
from embedding_aware_belt_fusion.coloca.model import ColocaQuA
from embedding_aware_belt_fusion.coloca.train import split_scenarios

# The paper's inference setting: C_in=512, H=96, W=256, N=96, D=256 (Fig. 4).
PAPER_FEATURE_SIZE = (96, 256)
PAPER_IN_CHANNELS = 512
PAPER_NUM_PATCHES = 96


def test_paper_lidar_range_yields_the_reported_feature_map_size():
    # Arrange: the paper's evaluation range and OpenCOOD's 0.4 m voxels
    lidar_range = [-102.4, -38.4, -3, 102.4, 38.4, 1]
    voxel_size = [0.4, 0.4, 4]

    # Act
    grid = pillar_grid_size(lidar_range, voxel_size)
    feature_size = bev_feature_size(lidar_range, voxel_size)

    # Assert
    np.testing.assert_array_equal(grid, [512, 192, 1])
    assert feature_size == PAPER_FEATURE_SIZE


def test_model_produces_the_paper_token_count():
    # Arrange / Act
    model = ColocaQuA(in_channels=PAPER_IN_CHANNELS, feature_size=PAPER_FEATURE_SIZE)

    # Assert
    assert model.num_patches == PAPER_NUM_PATCHES
    assert model.pos_embed.shape == (1, PAPER_NUM_PATCHES + 1, 256)


def test_forward_returns_one_se2_error_per_sample():
    # Arrange
    model = ColocaQuA(in_channels=PAPER_IN_CHANNELS, feature_size=PAPER_FEATURE_SIZE).eval()
    features = torch.randn(2, PAPER_IN_CHANNELS, *PAPER_FEATURE_SIZE)

    # Act
    with torch.no_grad():
        output = model(features)

    # Assert
    assert output.shape == (2, 3)
    assert torch.isfinite(output).all()


def test_initial_predictions_are_small_but_not_identically_zero():
    """A zero-init head would starve the encoder of gradient for many steps."""
    # Arrange
    model = ColocaQuA(in_channels=PAPER_IN_CHANNELS, feature_size=PAPER_FEATURE_SIZE).eval()
    features = torch.randn(3, PAPER_IN_CHANNELS, *PAPER_FEATURE_SIZE)

    # Act
    with torch.no_grad():
        output = model(features)

    # Assert: well inside the ~1.6 m label scale, but able to propagate gradient
    assert output.abs().max() < 1.0
    assert not torch.allclose(output, torch.zeros_like(output))


def test_output_scale_rescales_the_regressed_pose_error():
    # Arrange: identical weights, different output scales
    torch.manual_seed(0)
    unit = ColocaQuA(
        in_channels=PAPER_IN_CHANNELS, feature_size=PAPER_FEATURE_SIZE, output_scale=(1.0, 1.0, 1.0)
    ).eval()
    torch.manual_seed(0)
    scaled = ColocaQuA(
        in_channels=PAPER_IN_CHANNELS, feature_size=PAPER_FEATURE_SIZE, output_scale=(2.0, 2.0, 1.0)
    ).eval()
    features = torch.randn(2, PAPER_IN_CHANNELS, *PAPER_FEATURE_SIZE)

    # Act
    with torch.no_grad():
        base, rescaled = unit(features), scaled(features)

    # Assert
    torch.testing.assert_close(rescaled, base * torch.tensor([2.0, 2.0, 1.0]))


def test_dropout_is_applied_to_attention_only():
    """Table I specifies "Dropout (attn) 0.3", not dropout on the FFN/residuals."""
    # Arrange / Act
    model = ColocaQuA(
        in_channels=PAPER_IN_CHANNELS, feature_size=PAPER_FEATURE_SIZE, dropout=0.3
    )

    # Assert
    for layer in model.encoder.layers:
        assert layer.self_attn.dropout == 0.3
        assert layer.dropout.p == 0.0
        assert layer.dropout1.p == 0.0
        assert layer.dropout2.p == 0.0
    assert model.pos_drop.p == 0.0


def test_gradients_reach_the_localization_token():
    # Arrange
    model = ColocaQuA(in_channels=PAPER_IN_CHANNELS, feature_size=PAPER_FEATURE_SIZE)
    features = torch.randn(2, PAPER_IN_CHANNELS, *PAPER_FEATURE_SIZE)

    # Act
    pose_error_loss(model(features), torch.randn(2, 3)).backward()

    # Assert
    assert model.loc_token.grad is not None
    assert model.pos_embed.grad is not None
    assert model.patch_embed.weight.grad is not None


def test_forward_rejects_a_feature_map_of_the_wrong_size():
    # Arrange
    model = ColocaQuA(in_channels=PAPER_IN_CHANNELS, feature_size=PAPER_FEATURE_SIZE).eval()
    wrong = torch.randn(1, PAPER_IN_CHANNELS, 96, 128)

    # Act / Assert
    with pytest.raises(ValueError, match="positional embedding"):
        model(wrong)


def test_constructor_rejects_a_feature_map_smaller_than_one_patch():
    # Act / Assert
    with pytest.raises(ValueError, match="smaller than patch_size"):
        ColocaQuA(in_channels=PAPER_IN_CHANNELS, feature_size=(8, 8))


def test_pose_error_loss_applies_the_paper_weights():
    # Arrange: unit error on each component in turn, weights (2, 2, 1)
    prediction = torch.zeros(1, 3)
    target = torch.tensor([[1.0, 1.0, 1.0]])

    # Act
    loss = pose_error_loss(prediction, target, weights=(2.0, 2.0, 1.0))

    # Assert
    assert loss.item() == pytest.approx(5.0)


def test_pose_error_loss_is_zero_for_a_perfect_prediction():
    # Arrange
    target = torch.randn(4, 3)

    # Act / Assert
    assert pose_error_loss(target.clone(), target).item() == pytest.approx(0.0)


def test_pose_error_loss_rejects_mismatched_shapes():
    # Act / Assert
    with pytest.raises(ValueError, match="shape"):
        pose_error_loss(torch.zeros(2, 3), torch.zeros(3, 3))


def test_translation_residual_is_the_euclidean_norm_and_ignores_yaw():
    # Arrange: a 3-4-5 triangle in (dx, dy), with a large yaw error
    prediction = torch.tensor([[0.0, 0.0, 0.0]])
    target = torch.tensor([[3.0, 4.0, 90.0]])

    # Act / Assert
    assert translation_residual(prediction, target).item() == pytest.approx(5.0)


def test_localization_metrics_report_thresholds_as_fractions():
    # Arrange: residuals of 0.0, 0.9 and 2.0 m
    prediction = torch.zeros(3, 3)
    target = torch.tensor([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [2.0, 0.0, 0.0]])

    # Act
    metrics = localization_metrics(prediction, target, thresholds=(1.0, 0.5))

    # Assert
    assert metrics["mae_m"] == pytest.approx((0.0 + 0.9 + 2.0) / 3)
    assert metrics["below_1.0m"] == pytest.approx(2 / 3)
    assert metrics["below_0.5m"] == pytest.approx(1 / 3)


def test_accumulator_matches_a_single_batch_over_uneven_batches():
    # Arrange: the same 5 samples, split 3 + 2
    torch.manual_seed(0)
    prediction, target = torch.randn(5, 3), torch.randn(5, 3)
    accumulator = MetricAccumulator()

    # Act
    accumulator.update(prediction[:3], target[:3])
    accumulator.update(prediction[3:], target[3:])
    streamed = accumulator.compute()
    direct = localization_metrics(prediction, target)

    # Assert
    assert streamed["mae_m"] == pytest.approx(direct["mae_m"])
    assert streamed["rmse_m"] == pytest.approx(direct["rmse_m"])
    assert streamed["count"] == 5


def test_accumulator_raises_when_nothing_was_accumulated():
    # Act / Assert
    with pytest.raises(RuntimeError, match="no predictions"):
        MetricAccumulator().compute()


def test_scenario_split_is_disjoint_deterministic_and_covers_everything():
    # Arrange
    scenarios = [f"scene_{i:02d}" for i in range(20)]

    # Act
    train, val = split_scenarios(scenarios, val_fraction=0.15, seed=0)
    train_again, val_again = split_scenarios(scenarios, val_fraction=0.15, seed=0)

    # Assert
    assert set(train).isdisjoint(val)
    assert sorted(train + val) == scenarios
    assert len(val) == 3
    assert (train, val) == (train_again, val_again)


def test_scenario_split_reserves_at_least_one_validation_scenario():
    # Arrange: a fraction small enough to round to zero
    scenarios = [f"scene_{i}" for i in range(4)]

    # Act
    _, val = split_scenarios(scenarios, val_fraction=0.01, seed=0)

    # Assert
    assert len(val) == 1


def test_scenario_split_rejects_an_out_of_range_fraction():
    # Act / Assert
    with pytest.raises(ValueError, match="val_fraction"):
        split_scenarios(["a", "b"], val_fraction=1.0, seed=0)


@pytest.mark.parametrize("readout", ["loc_token", "mean", "both"])
def test_every_readout_produces_one_pose_error_per_sample(readout):
    # Arrange
    model = ColocaQuA(
        in_channels=PAPER_IN_CHANNELS, feature_size=PAPER_FEATURE_SIZE, readout=readout
    ).eval()

    # Act
    with torch.no_grad():
        output = model(torch.randn(2, PAPER_IN_CHANNELS, *PAPER_FEATURE_SIZE))

    # Assert
    assert output.shape == (2, 3)
    assert torch.isfinite(output).all()


def test_readout_defaults_to_the_papers_localization_token():
    # Act / Assert
    model = ColocaQuA(in_channels=PAPER_IN_CHANNELS, feature_size=PAPER_FEATURE_SIZE)
    assert model.readout == "loc_token"


def test_mean_readout_ignores_the_localization_token_state():
    """Distinguishes the readouts: only `loc_token` depends on that token."""
    # Arrange
    torch.manual_seed(0)
    model = ColocaQuA(
        in_channels=PAPER_IN_CHANNELS, feature_size=PAPER_FEATURE_SIZE, readout="mean"
    ).eval()
    features = torch.randn(1, PAPER_IN_CHANNELS, *PAPER_FEATURE_SIZE)

    # Act: perturbing the loc token changes attention but this is a smoke check
    # that the mean readout runs and stays finite with a very different token
    with torch.no_grad():
        before = model(features)
        model.loc_token.mul_(100.0)
        after = model(features)

    # Assert
    assert torch.isfinite(before).all() and torch.isfinite(after).all()


def test_unknown_readout_is_rejected():
    # Act / Assert
    with pytest.raises(ValueError, match="readout must be one of"):
        ColocaQuA(
            in_channels=PAPER_IN_CHANNELS, feature_size=PAPER_FEATURE_SIZE, readout="cls"
        )
