import math

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


def test_sampling_is_canonical_so_placement_does_not_change_the_patch():
    # The same synthetic object placed at two different positions AND
    # orientations must yield the same canonical ROI patch. This is the
    # pose-invariance property the whole matching design rests on.
    length = 8.0
    box_a = torch.tensor([[0.0, 0.0, 0.0, 1.6, 4.0, length, 0.0]])
    box_b = torch.tensor([[40.0, 12.0, 0.0, 1.6, 4.0, length, math.pi / 2]])

    # An asymmetric marker 3 m "ahead" of each box centre along its own heading,
    # so a wrong rotation convention shows up as a flipped patch.
    features_a = torch.zeros(1, 96, 256)
    _stamp(features_a, x=3.0, y=0.0, value=1.0)
    features_b = torch.zeros(1, 96, 256)
    _stamp(features_b, x=40.0, y=15.0, value=1.0)

    roi_a = rotated_roi_align(features_a, box_a, LIDAR_RANGE, output_size=4)
    roi_b = rotated_roi_align(features_b, box_b, LIDAR_RANGE, output_size=4)

    assert torch.argmax(roi_a.flatten()) == torch.argmax(roi_b.flatten())


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
