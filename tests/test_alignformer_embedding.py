import math

import pytest
import torch

from embedding_aware_belt_fusion.alignformer.embedding import (
    ObjectEmbedding,
    rotated_roi_align,
)

LIDAR_RANGE = [-102.4, -38.4, -3.0, 102.4, 38.4, 1.0]


def _stamp(features, x, y, value, lidar_range=LIDAR_RANGE):
    """Write `value` into the BEV cell containing world point (x, y)."""
    _, height, width = features.shape
    x_min, y_min, _, x_max, y_max, _ = lidar_range
    column = int((x - x_min) / (x_max - x_min) * width)
    row = int((y - y_min) / (y_max - y_min) * height)
    features[:, row, column] = value


def test_output_shape_is_per_box_grid():
    features = torch.randn(8, 96, 256)
    boxes = torch.tensor([[10.0, 5.0, 0.0, 1.6, 2.0, 4.5, 0.3]])

    roi = rotated_roi_align(features, boxes, LIDAR_RANGE, output_size=4)

    assert roi.shape == (1, 8, 4, 4)


def test_empty_box_set_returns_empty_without_error():
    features = torch.randn(8, 96, 256)

    roi = rotated_roi_align(features, torch.zeros(0, 7), LIDAR_RANGE, output_size=4)

    assert roi.shape == (0, 8, 4, 4)


def test_a_leading_batch_dim_of_one_is_squeezed_automatically():
    # R26: an un-squeezed detector-batch leading dim (forgetting the [0]
    # AgentDetections normally applies) must be accepted, not fail deep
    # inside grid_sample with an unhelpful shape-mismatch message.
    features_3d = torch.randn(8, 96, 256)
    boxes = torch.tensor([[10.0, 5.0, 0.0, 1.6, 2.0, 4.5, 0.3]])

    roi_from_4d = rotated_roi_align(
        features_3d.unsqueeze(0), boxes, LIDAR_RANGE, output_size=4
    )
    roi_from_3d = rotated_roi_align(features_3d, boxes, LIDAR_RANGE, output_size=4)

    assert roi_from_4d.shape == (1, 8, 4, 4)
    assert torch.equal(roi_from_4d, roi_from_3d)


def test_a_real_batch_of_feature_maps_is_rejected_with_a_clear_error():
    # There is no per-box map to select for batch size > 1 -- this function
    # pairs ONE shared BEV map with every box -- so this must raise, and name
    # the shape it actually got, rather than fail unhelpfully inside
    # grid_sample.
    features = torch.randn(2, 8, 96, 256)
    boxes = torch.tensor([[10.0, 5.0, 0.0, 1.6, 2.0, 4.5, 0.3]])

    with pytest.raises(ValueError, match=r"\(2, 8, 96, 256\)"):
        rotated_roi_align(features, boxes, LIDAR_RANGE, output_size=4)


def test_an_unexpected_rank_is_rejected_with_a_clear_error():
    features = torch.randn(96, 256)
    boxes = torch.tensor([[10.0, 5.0, 0.0, 1.6, 2.0, 4.5, 0.3]])

    with pytest.raises(ValueError, match=r"\(96, 256\)"):
        rotated_roi_align(features, boxes, LIDAR_RANGE, output_size=4)


def test_sampling_is_canonical_so_placement_does_not_change_the_patch():
    # The same synthetic object placed at two different positions AND
    # orientations must yield the same canonical ROI patch. This is the
    # pose-invariance property the whole matching design rests on.
    #
    # The marker sits at the box's forward-right corner of the 4x4 canonical
    # sampling grid (along = +length/2, across = +width/2), with length and
    # width chosen so that corner falls exactly on a BEV pixel *centre*
    # (world coordinate = x_min + (k + 0.5) * pixel_size for an integer k).
    # Both box centres (0, 0) and (40, 12) already sit on pixel boundaries
    # (multiples of the 0.8 m pixel size), so the marker offset from each
    # centre only needs to itself be an odd multiple of half a pixel:
    # 0.5 * length = 2.8 = (3 + 0.5) * 0.8 and 0.5 * width = 1.2 =
    # (1 + 0.5) * 0.8. This makes `grid_sample` land exactly on the marked
    # pixel with no bilinear blending, so a genuinely wrong sampling
    # convention (wrong axis, wrong rotation direction, ...) reliably misses
    # it rather than accidentally smearing onto it.
    length = 5.6
    width = 2.4
    box_a = torch.tensor([[0.0, 0.0, 0.0, 1.6, width, length, 0.0]])
    box_b = torch.tensor([[40.0, 12.0, 0.0, 1.6, width, length, math.pi / 2]])

    features_a = torch.zeros(1, 96, 256)
    _stamp(features_a, x=2.8, y=1.2, value=1.0)
    features_b = torch.zeros(1, 96, 256)
    _stamp(features_b, x=38.8, y=14.8, value=1.0)

    roi_a = rotated_roi_align(features_a, box_a, LIDAR_RANGE, output_size=4)
    roi_b = rotated_roi_align(features_b, box_b, LIDAR_RANGE, output_size=4)

    # Guard against vacuity: if the marker misses the sampled grid entirely,
    # both patches are all-zero and any comparison trivially "passes".
    assert roi_a.abs().max() > 0
    assert roi_b.abs().max() > 0

    # atol accounts only for float32 grid-normalization round-trip error
    # (observed ~4e-6 on one boundary-adjacent cell), not for any real
    # geometric mismatch.
    assert torch.allclose(roi_a, roi_b, atol=1e-4)


def test_roi_align_is_differentiable_wrt_features():
    features = torch.randn(4, 96, 256, requires_grad=True)
    boxes = torch.tensor([[10.0, 5.0, 0.0, 1.6, 2.0, 4.5, 0.3]])

    rotated_roi_align(features, boxes, LIDAR_RANGE, output_size=4).sum().backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_embedding_is_unit_norm():
    head = ObjectEmbedding(in_channels=8, output_size=4, dim=128)
    roi = torch.randn(5, 8, 4, 4)

    embeddings = head(roi)

    assert embeddings.shape == (5, 128)
    assert torch.allclose(embeddings.norm(dim=1), torch.ones(5), atol=1e-5)


def test_embedding_handles_empty_object_set():
    head = ObjectEmbedding(in_channels=8, output_size=4, dim=128)

    embeddings = head(torch.zeros(0, 8, 4, 4))

    assert embeddings.shape == (0, 128)
