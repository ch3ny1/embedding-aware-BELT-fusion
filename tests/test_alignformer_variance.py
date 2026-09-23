"""The inverse-variance weighting of the Procrustes fit.

Cross-agent correspondence disagreement is heteroscedastic: measured on the
validation slice under the true pose projection, per-detection confidence
predicts it far more strongly than range does (see
``scripts/fit_correspondence_variance.py``). Weighting each correspondence by
its inverse variance is the statistically correct estimator for that, and this
file pins the properties that make it one rather than a tuning knob.
"""

from __future__ import annotations

import math

import pytest
import torch

from embedding_aware_belt_fusion.alignformer.variance import (
    VARIANCE_MODES,
    CorrespondenceVarianceModel,
    variance_model_from_config,
)

_FITTED = dict(
    mode="split",
    sigma_translation_m=0.2517,
    translation_exponent=1.128,
    sigma_yaw_rad=0.0810,
    yaw_exponent=1.933,
    score_reference=0.4,
)


def _model(**overrides) -> CorrespondenceVarianceModel:
    return CorrespondenceVarianceModel(**{**_FITTED, **overrides})


def _inputs(scores, mass=None):
    """One batch of B=1 with ``len(scores)`` ego rows, CAV scores equal to ego's."""
    ego_scores = torch.tensor([scores], dtype=torch.float32)
    model = _model()
    cav_var_t, cav_var_yaw = model.detection_variances(ego_scores)
    mass = torch.ones_like(ego_scores) if mass is None else torch.tensor([mass])
    return ego_scores, cav_var_t, cav_var_yaw, mass


def test_mode_none_reproduces_the_unweighted_augmented_mass_exactly():
    # Arrange
    ego_scores, cav_var_t, cav_var_yaw, mass = _inputs([0.25, 0.45, 0.7])
    model = _model(mode="none")

    # Act
    weights = model.augmented_weights(mass, ego_scores, cav_var_t, cav_var_yaw, 2.0)

    # Assert
    assert torch.equal(weights, torch.cat([mass, mass], dim=1))


def test_every_mode_preserves_the_total_augmented_weight_per_sample():
    """The MIN_MATCH_MASS gate reads the total, so re-weighting must not move it.

    ``weighted_se2_kabsch`` is scale-invariant in the weights but its
    ``MIN_MATCH_MASS`` suppression is not, so the correction would silently
    start firing on different pairs if the total drifted.
    """
    # Arrange
    ego_scores, cav_var_t, cav_var_yaw, mass = _inputs([0.21, 0.5, 0.77], mass=[0.9, 0.4, 0.2])
    expected = 2.0 * float(mass.sum())

    for mode in VARIANCE_MODES:
        # Act
        weights = _model(mode=mode).augmented_weights(
            mass, ego_scores, cav_var_t, cav_var_yaw, 2.0
        )
        # Assert
        assert float(weights.sum()) == pytest.approx(expected, rel=1e-5), mode


def test_a_less_confident_correspondence_is_weighted_down_relative_to_a_confident_one():
    # Arrange
    ego_scores, cav_var_t, cav_var_yaw, mass = _inputs([0.22, 0.75])

    for mode in ("scalar", "split"):
        # Act
        weights = _model(mode=mode).augmented_weights(
            mass, ego_scores, cav_var_t, cav_var_yaw, 2.0
        )
        centre = weights[:, :2]
        # Assert
        assert float(centre[0, 0]) < float(centre[0, 1]), mode


def test_scalar_mode_gives_a_correspondences_centre_and_heading_the_same_weight():
    # Arrange
    ego_scores, cav_var_t, cav_var_yaw, mass = _inputs([0.25, 0.45, 0.7])

    # Act
    weights = _model(mode="scalar").augmented_weights(
        mass, ego_scores, cav_var_t, cav_var_yaw, 2.0
    )

    # Assert
    assert torch.allclose(weights[:, :3], weights[:, 3:], atol=1e-6)


def test_split_mode_weights_the_heading_virtual_point_below_its_centre():
    """The heading tip carries the centre error PLUS lam times the yaw error."""
    # Arrange
    ego_scores, cav_var_t, cav_var_yaw, mass = _inputs([0.25, 0.45, 0.7])

    # Act
    weights = _model(mode="split").augmented_weights(
        mass, ego_scores, cav_var_t, cav_var_yaw, 2.0
    )
    centre, heading = weights[:, :3], weights[:, 3:]

    # Assert
    assert torch.all(heading < centre)


def test_split_mode_penalizes_the_heading_point_harder_as_confidence_falls():
    """Yaw degrades with confidence faster than the centre does (1.93 vs 1.13).

    So the heading-to-centre weight ratio must itself fall with confidence --
    a single scalar weight per correspondence cannot express that.
    """
    # Arrange
    ego_scores, cav_var_t, cav_var_yaw, mass = _inputs([0.21, 0.4, 0.77])

    # Act
    weights = _model(mode="split").augmented_weights(
        mass, ego_scores, cav_var_t, cav_var_yaw, 2.0
    )
    ratio = weights[0, 3:] / weights[0, :3]

    # Assert
    assert float(ratio[0]) < float(ratio[1]) < float(ratio[2])


def test_a_zero_mass_padded_row_stays_at_zero_weight_and_produces_no_nan():
    """Collated batches pad with score 0, which a raw 1/score would send to inf."""
    # Arrange
    ego_scores = torch.tensor([[0.5, 0.0]], dtype=torch.float32)
    mass = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    model = _model()
    cav_var_t, cav_var_yaw = model.detection_variances(ego_scores)

    # Act
    weights = model.augmented_weights(mass, ego_scores, cav_var_t, cav_var_yaw, 2.0)

    # Assert
    assert torch.isfinite(weights).all()
    assert float(weights[0, 1]) == 0.0
    assert float(weights[0, 3]) == 0.0


def test_a_wholly_empty_sample_yields_finite_zero_weights():
    # Arrange
    ego_scores = torch.zeros(1, 3)
    mass = torch.zeros(1, 3)
    model = _model()
    cav_var_t, cav_var_yaw = model.detection_variances(ego_scores)

    # Act
    weights = model.augmented_weights(mass, ego_scores, cav_var_t, cav_var_yaw, 2.0)

    # Assert
    assert torch.isfinite(weights).all()
    assert float(weights.abs().sum()) == 0.0


def test_the_weights_carry_a_finite_gradient_back_to_the_soft_match_mass():
    # Arrange
    ego_scores, cav_var_t, cav_var_yaw, _ = _inputs([0.3, 0.6])
    mass = torch.tensor([[0.8, 0.5]], requires_grad=True)
    model = _model()

    # Act
    model.augmented_weights(mass, ego_scores, cav_var_t, cav_var_yaw, 2.0).sum().backward()

    # Assert
    assert torch.isfinite(mass.grad).all()


def test_detection_variance_rises_as_confidence_falls():
    # Arrange
    scores = torch.tensor([[0.2, 0.4, 0.8]])

    # Act
    var_t, var_yaw = _model().detection_variances(scores)

    # Assert
    assert float(var_t[0, 0]) > float(var_t[0, 1]) > float(var_t[0, 2])
    assert float(var_yaw[0, 0]) > float(var_yaw[0, 1]) > float(var_yaw[0, 2])
    assert float(var_t[0, 1]) == pytest.approx(_FITTED["sigma_translation_m"] ** 2, rel=1e-5)
    assert float(var_yaw[0, 1]) == pytest.approx(_FITTED["sigma_yaw_rad"] ** 2, rel=1e-5)


def test_the_reference_correspondence_has_unit_centre_precision():
    """Normalizing at the fit's reference confidence is what keeps rho ~ 1."""
    # Arrange
    ego_scores = torch.tensor([[_FITTED["score_reference"]]])
    model = _model()
    cav_var_t, cav_var_yaw = model.detection_variances(ego_scores)

    # Act
    rho_centre, rho_heading = model.precisions(ego_scores, cav_var_t, cav_var_yaw, 2.0)

    # Assert
    assert float(rho_centre[0, 0]) == pytest.approx(1.0, rel=1e-5)
    expected = 1.0 / (
        1.0 + 4.0 * _FITTED["sigma_yaw_rad"] ** 2 / _FITTED["sigma_translation_m"] ** 2
    )
    assert float(rho_heading[0, 0]) == pytest.approx(expected, rel=1e-5)


def test_an_invalid_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        _model(mode="inverse")


@pytest.mark.parametrize(
    "field", ["sigma_translation_m", "sigma_yaw_rad", "score_reference"]
)
def test_a_non_positive_scale_is_rejected(field):
    with pytest.raises(ValueError, match=field):
        _model(**{field: 0.0})


def test_a_config_without_the_block_yields_the_unweighted_model():
    # Act
    model = variance_model_from_config({"heading_lambda": 2.0})

    # Assert
    assert model.mode == "none"


def test_a_config_block_is_read_and_its_mode_can_be_overridden():
    # Arrange
    config = {
        "correspondence_variance": {
            "mode": "none",
            "sigma_translation_m": 0.2517,
            "translation_exponent": 1.128,
            "sigma_yaw_deg": math.degrees(0.0810),
            "yaw_exponent": 1.933,
            "score_reference": 0.4,
        }
    }

    # Act
    default = variance_model_from_config(config)
    overridden = variance_model_from_config(config, mode="split")

    # Assert
    assert default.mode == "none"
    assert overridden.mode == "split"
    assert overridden.sigma_yaw_rad == pytest.approx(0.0810, rel=1e-6)


def test_selecting_a_weighted_mode_without_fitted_parameters_is_refused():
    """A mode is an experiment switch; the parameters are data and must exist."""
    with pytest.raises(ValueError, match="correspondence_variance"):
        variance_model_from_config({"heading_lambda": 2.0}, mode="split")


# --- head B end to end -------------------------------------------------------


def _head(mode: str):
    from embedding_aware_belt_fusion.alignformer.model import AlignFormerB

    torch.manual_seed(0)
    head = AlignFormerB(
        embed_dim=8, model_dim=16, layers=1, heads=2, variance_model=_model(mode=mode)
    )
    # Documented test hook: score on the raw embeddings so the correspondence is
    # the intended one-to-one rather than whatever an untrained trunk invents.
    head.use_raw_embedding_scores = True
    head.eval()
    return head


def _batch(scores, shift_last=0.0):
    """Two agents seeing ``len(scores)`` objects, the last one displaced on the CAV side."""
    count = len(scores)
    centres = torch.stack(
        [torch.arange(count, dtype=torch.float32) * 9.0, torch.zeros(count)], dim=-1
    )
    ego = torch.zeros(1, count, 7)
    ego[0, :, :2] = centres
    cav = ego.clone()
    cav[0, -1, 0] += shift_last
    score = torch.tensor([scores], dtype=torch.float32)
    return {
        "ego_boxes": ego,
        "cav_boxes": cav,
        "ego_scores": score,
        "cav_scores": score,
        "ego_embeddings": torch.eye(count, 8).unsqueeze(0),
        "cav_embeddings": torch.eye(count, 8).unsqueeze(0),
        "ego_mask": torch.ones(1, count, dtype=torch.bool),
        "cav_mask": torch.ones(1, count, dtype=torch.bool),
    }


def test_head_b_under_mode_none_is_bit_identical_to_the_unweighted_head():
    """Every measurement taken before this module must still reproduce."""
    # Arrange
    from embedding_aware_belt_fusion.alignformer.model import AlignFormerB

    torch.manual_seed(0)
    legacy = AlignFormerB(embed_dim=8, model_dim=16, layers=1, heads=2)
    legacy.use_raw_embedding_scores = True
    legacy.eval()
    weighted_none = _head("none")
    batch = _batch([0.3, 0.5, 0.7, 0.25])

    # Act
    with torch.no_grad():
        a = legacy(batch)
        b = weighted_none(batch)

    # Assert
    assert torch.equal(a.psi, b.psi)
    assert torch.equal(a.t, b.t)


def test_weighting_reduces_the_pull_of_a_low_confidence_displaced_correspondence():
    """The point of the estimator change, stated as a property.

    One object is displaced on the CAV side and is the least confident
    detection in the set. Inverse-variance weighting must let it move the
    recovered translation less than the unweighted fit does.
    """
    # Arrange
    batch = _batch([0.75, 0.75, 0.75, 0.2], shift_last=4.0)

    # Act
    with torch.no_grad():
        unweighted = _head("none")(batch)
        split = _head("split")(batch)

    # Assert
    assert float(split.t.norm()) < float(unweighted.t.norm())


def test_weighting_leaves_the_fit_alone_when_every_detection_is_equally_confident():
    """A homoscedastic set is the case where least squares was already optimal."""
    # Arrange
    batch = _batch([0.5, 0.5, 0.5, 0.5], shift_last=4.0)

    # Act
    with torch.no_grad():
        unweighted = _head("none")(batch)
        split = _head("split")(batch)

    # Assert
    assert torch.allclose(unweighted.t, split.t, atol=1e-4)
    assert torch.allclose(unweighted.psi, split.psi, atol=1e-4)


def test_an_empty_object_set_still_returns_the_identity_under_weighting():
    # Arrange
    head = _head("split")
    batch = _batch([0.5, 0.5])
    batch = {
        **batch,
        "ego_boxes": batch["ego_boxes"][:, :0],
        "ego_scores": batch["ego_scores"][:, :0],
        "ego_embeddings": batch["ego_embeddings"][:, :0],
        "ego_mask": batch["ego_mask"][:, :0],
    }

    # Act
    estimate = head(batch)

    # Assert
    assert float(estimate.psi.abs().sum()) == 0.0
    assert float(estimate.t.abs().sum()) == 0.0


# --- config plumbing ---------------------------------------------------------


_MODEL_CONFIG = {
    "output_size": 4,
    "embed_dim": 8,
    "model_dim": 16,
    "layers": 1,
    "heads": 2,
    "heading_lambda": 2.0,
    "sinkhorn_iterations": 4,
    "max_objects": 64,
    "correspondence_variance": {
        "mode": "none",
        "sigma_translation_m": 0.2516,
        "translation_exponent": 1.1281,
        "sigma_yaw_deg": 4.6392,
        "yaw_exponent": 1.9322,
        "score_reference": 0.4,
    },
}


def test_resolving_the_mode_never_mutates_the_caller_s_config():
    # Arrange
    from embedding_aware_belt_fusion.alignformer.stage2 import _with_variance_weighting

    config = {"model": _MODEL_CONFIG}

    # Act
    resolved = _with_variance_weighting(config, "split")

    # Assert
    assert resolved["model"]["correspondence_variance"]["mode"] == "split"
    assert config["model"]["correspondence_variance"]["mode"] == "none"


def test_the_head_built_from_a_config_carries_that_config_s_mode():
    """`load_stage2` rebuilds from the checkpoint's config, so this is the path
    by which a trained estimator is reproduced at evaluation time."""
    # Arrange
    from embedding_aware_belt_fusion.alignformer.stage2 import (
        _with_variance_weighting,
        build_stage2_modules,
    )

    config = {"model": _MODEL_CONFIG}

    # Act
    modules = build_stage2_modules(
        _with_variance_weighting(config, "split"),
        channels=4,
        device=torch.device("cpu"),
        head="B",
    )

    # Assert
    assert modules["pose"].variance_model.mode == "split"


def test_the_variance_model_adds_nothing_to_the_state_dict():
    """An r140 checkpoint trained before this module must still load strictly."""
    # Arrange
    from embedding_aware_belt_fusion.alignformer.model import AlignFormerB

    torch.manual_seed(0)
    plain = AlignFormerB(embed_dim=8, model_dim=16, layers=1, heads=2)
    torch.manual_seed(0)
    weighted = AlignFormerB(
        embed_dim=8, model_dim=16, layers=1, heads=2, variance_model=_model(mode="split")
    )

    # Act / Assert
    assert set(plain.state_dict()) == set(weighted.state_dict())
    weighted.load_state_dict(plain.state_dict(), strict=True)


def test_the_output_directory_names_the_weighting_only_when_it_is_on():
    # Arrange
    from embedding_aware_belt_fusion.alignformer.stage2 import stage2_output_dir

    # Act / Assert
    assert stage2_output_dir("B", "boxes+embeddings", 1.0).name == "stage2_B_boxes+embeddings"
    assert (
        stage2_output_dir("B", "boxes+embeddings", 1.0, "split").name
        == "stage2_B_boxes+embeddings_ivw_split"
    )
