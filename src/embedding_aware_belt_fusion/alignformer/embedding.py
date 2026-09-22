"""The transmit-side per-object embedding.

Features are sampled over each box's footprint **in that box's own canonical
frame**, so the resulting descriptor is invariant to where the sender believes
the object is. That is the separation the method depends on: the embedding
carries appearance for matching, the box carries geometry for solving.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Box layout is OpenCOOD 'hwl': [x, y, z, h, w, l, yaw].
_BOX_WIDTH = 4
_BOX_LENGTH = 5
_BOX_YAW = 6


def rotated_roi_align(
    features: Tensor,
    boxes: Tensor,
    lidar_range: Sequence[float],
    output_size: int,
) -> Tensor:
    """Sample a ``k x k`` grid over each rotated box footprint.

    Parameters
    ----------
    features: ``(C, H, W)`` BEV map, H indexing y and W indexing x. R26: a
        ``(1, C, H, W)`` tensor -- an un-squeezed detector-batch leading dim,
        e.g. forgetting the ``[0]`` that :class:`AgentDetections` normally
        applies -- is also accepted and squeezed automatically. Any other
        4-D shape (a genuine batch of more than one BEV map) is rejected with
        a ``ValueError`` naming the received shape: this function pairs ONE
        shared BEV map with every box in ``boxes``, so there is no per-box
        map to select for batch size > 1.
    boxes: ``(M, 7)`` in ``hwl`` order, in the same frame as ``lidar_range``.
    lidar_range: ``[x_min, y_min, z_min, x_max, y_max, z_max]``.
    output_size: the ``k`` of the ``k x k`` sampling grid.

    Returns
    -------
    Tensor
        ``(M, C, k, k)``. Row index runs along the box's length (forward) axis
        and column index across its width, both in the box frame.
    """
    if features.dim() == 4:
        if features.shape[0] != 1:
            raise ValueError(
                "features with a leading batch dimension must have batch size 1 "
                f"(one BEV map shared by every box); got {tuple(features.shape)}"
            )
        features = features.squeeze(0)
    if features.dim() != 3:
        raise ValueError(
            f"features must be (C, H, W) or (1, C, H, W), got {tuple(features.shape)}"
        )
    if boxes.dim() != 2 or boxes.shape[1] != 7:
        raise ValueError(f"boxes must be (M, 7), got {tuple(boxes.shape)}")

    channels = features.shape[0]
    count = boxes.shape[0]
    if count == 0:
        return features.new_zeros((0, channels, output_size, output_size))

    x_min, y_min, _, x_max, y_max, _ = lidar_range

    # Canonical grid in box-local units, spanning [-0.5, 0.5] of each extent.
    axis = torch.linspace(-0.5, 0.5, output_size, device=boxes.device, dtype=boxes.dtype)
    along, across = torch.meshgrid(axis, axis, indexing="ij")
    along = along.reshape(1, -1) * boxes[:, _BOX_LENGTH : _BOX_LENGTH + 1]
    across = across.reshape(1, -1) * boxes[:, _BOX_WIDTH : _BOX_WIDTH + 1]

    cosine = torch.cos(boxes[:, _BOX_YAW]).unsqueeze(1)
    sine = torch.sin(boxes[:, _BOX_YAW]).unsqueeze(1)
    world_x = boxes[:, 0:1] + along * cosine - across * sine
    world_y = boxes[:, 1:2] + along * sine + across * cosine

    # grid_sample expects normalized coordinates in [-1, 1], last dim (x, y).
    grid_x = 2.0 * (world_x - x_min) / (x_max - x_min) - 1.0
    grid_y = 2.0 * (world_y - y_min) / (y_max - y_min) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).reshape(
        count, output_size, output_size, 2
    )

    batched = features.unsqueeze(0).expand(count, -1, -1, -1)
    return F.grid_sample(batched, grid, align_corners=False, padding_mode="zeros")


class ObjectEmbedding(nn.Module):
    """Map pooled ROI features to an L2-normalized per-object descriptor."""

    def __init__(self, in_channels: int, output_size: int, dim: int) -> None:
        super().__init__()
        flat = in_channels * output_size * output_size
        self.net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(flat, 2 * dim),
            nn.LayerNorm(2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        self.dim = dim

    def forward(self, roi: Tensor) -> Tensor:
        """Return ``(M, dim)`` unit-norm embeddings for ``(M, C, k, k)`` ROI features."""
        if roi.shape[0] == 0:
            return roi.new_zeros((0, self.dim))
        return F.normalize(self.net(roi), p=2.0, dim=1)
