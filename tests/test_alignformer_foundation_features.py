"""Frozen DINOv2 crop descriptors for the cross-agent appearance probe.

The camera trunk's null was about a frozen ImageNet ResNet-18 pooled over
projected boxes (write-up caveat 5), and the colour probe's null was about
paint. Neither says appearance cannot carry cross-agent identity. Before any
training, the cheapest test is a vision foundation model's zero-shot crop
embedding run through the same separability probe. These tests cover the
pure parts: the letterbox crop, the two descriptors (pooled class token and
the silhouette-masked mean of patch tokens), their normalization, and that
the backbone is frozen.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from embedding_aware_belt_fusion.alignformer.foundation_features import (
    DESCRIPTOR_NAMES,
    INPUT_SIZE,
    FoundationBackbone,
    letterbox,
    patch_grid_mask,
)


def _image(height: int, width: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, size=(height, width, 3), dtype=np.uint8)


# ----------------------------------------------------------------------------
# Letterbox and mask
# ----------------------------------------------------------------------------


def test_letterbox_keeps_the_aspect_ratio_and_pads_to_a_square():
    crop = np.full((50, 200, 3), 200, dtype=np.uint8)

    square = letterbox(crop, INPUT_SIZE)

    assert square.shape == (INPUT_SIZE, INPUT_SIZE, 3)
    # The content is 224 wide and 56 tall, centred; the bands above and
    # below are padding, not image.
    rows_with_content = np.where((square != square[0, 0]).any(axis=(1, 2)))[0]
    assert 50 <= rows_with_content.max() - rows_with_content.min() + 1 <= 60


def test_letterbox_of_a_square_crop_is_a_plain_resize():
    crop = _image(100, 100)

    square = letterbox(crop, INPUT_SIZE)

    assert square.shape == (INPUT_SIZE, INPUT_SIZE, 3)
    assert not (square == square[0, 0]).all(axis=2).any(axis=1).all()  # no padding band


def test_patch_grid_mask_marks_the_patches_the_silhouette_covers():
    mask = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=bool)
    mask[:, : INPUT_SIZE // 2] = True  # left half

    grid = patch_grid_mask(mask, grid=16)

    assert grid.shape == (16, 16)
    assert grid[:, :8].all() and not grid[:, 8:].any()


# ----------------------------------------------------------------------------
# The backbone
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def backbone() -> FoundationBackbone:
    return FoundationBackbone(device="cpu")


def test_the_backbone_is_frozen_and_in_eval_mode(backbone):
    assert all(not p.requires_grad for p in backbone.parameters())
    assert not backbone.training
    backbone.train()  # a caller cannot un-freeze it by accident
    assert not backbone.training


def test_descriptors_have_the_declared_names_shapes_and_unit_norm(backbone):
    crop = _image(120, 180)
    mask = np.ones((120, 180), dtype=bool)

    descriptors = backbone.describe(crop, mask)

    assert set(descriptors) == set(DESCRIPTOR_NAMES) == {"dino_cls", "dino_patch_mean"}
    for name in DESCRIPTOR_NAMES:
        assert descriptors[name].shape == (backbone.embed_dim,)
        assert np.linalg.norm(descriptors[name]) == pytest.approx(1.0, abs=1e-4)


def test_descriptors_are_deterministic_and_depend_on_the_crop(backbone):
    first = backbone.describe(_image(100, 100, seed=1), np.ones((100, 100), dtype=bool))
    again = backbone.describe(_image(100, 100, seed=1), np.ones((100, 100), dtype=bool))
    other = backbone.describe(_image(100, 100, seed=2), np.ones((100, 100), dtype=bool))

    for name in DESCRIPTOR_NAMES:
        np.testing.assert_allclose(first[name], again[name], atol=1e-5)
        assert float(np.dot(first[name], other[name])) < 0.999


def test_the_masked_patch_mean_ignores_patches_outside_the_silhouette(backbone):
    # Left half noise, right half flat grey. A mask on the left half must
    # give the same masked mean whatever the right half holds.
    left = _image(112, 56, seed=3)
    crop_a = np.concatenate([left, np.full((112, 56, 3), 128, np.uint8)], axis=1)
    crop_b = np.concatenate([left, _image(112, 56, seed=4)], axis=1)
    mask = np.zeros((112, 112), dtype=bool)
    mask[:, :56] = True

    a = backbone.describe(crop_a, mask)["dino_patch_mean"]
    b = backbone.describe(crop_b, mask)["dino_patch_mean"]
    full = backbone.describe(crop_a, np.ones((112, 112), dtype=bool))["dino_patch_mean"]

    # Attention mixes the halves inside the network, so the two are close
    # but not identical; the full-mask mean is clearly different.
    assert float(np.dot(a, b)) > float(np.dot(a, full))


def test_an_empty_mask_falls_back_to_the_mean_over_every_patch(backbone):
    crop = _image(64, 64, seed=5)

    empty = backbone.describe(crop, np.zeros((64, 64), dtype=bool))["dino_patch_mean"]
    full = backbone.describe(crop, np.ones((64, 64), dtype=bool))["dino_patch_mean"]

    np.testing.assert_allclose(empty, full, atol=1e-5)


def test_a_silhouette_too_small_for_any_patch_keeps_its_best_covered_patch():
    mask = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=bool)
    mask[0:4, 0:4] = True  # a sliver inside the top-left patch

    grid = patch_grid_mask(mask, grid=16)

    assert grid.sum() == 1 and grid[0, 0]


def test_the_mask_letterbox_pads_with_zeros_so_padding_is_never_silhouette():
    mask = np.ones((50, 200), dtype=bool)
    square = letterbox(np.repeat(mask[..., None].astype(np.uint8) * 255, 3, axis=2), INPUT_SIZE, fill=(0, 0, 0))

    assert (square[0] == 0).all() and (square[INPUT_SIZE // 2] == 255).all()
