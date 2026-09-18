"""PointPillars BEV feature extractor for CoLoca-QuA.

The paper (Section III-A, Eq. 3) reuses the intermediate BEV feature map that a
cooperative-perception backbone already computes, so no extra bandwidth or
compute is spent on localization.  Concretely this is OpenCOOD's F-Cooper
PointPillars stack up to and including the shrink header, with the detection
heads and the fusion module removed.

With ``cav_lidar_range = [-102.4, -38.4, -3, 102.4, 38.4, 1]`` and
``voxel_size = 0.4`` the pillar grid is 512x192, the BEV backbone emits a
stride-2 map, and the shrink header compresses 384 -> 256 channels.  That gives
exactly the 256x96x256 feature map reported in the paper's Fig. 4 caption.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

# Keys of the pretrained F-Cooper checkpoint that belong to the detection head
# rather than the shared backbone.  They are dropped when loading.
_DETECTION_HEAD_PREFIXES = ("cls_head.", "reg_head.")
# OpenCOOD's BaseBEVBackbone upsamples every stage back to stride 2 of the
# pillar grid, so the emitted BEV map is always half the grid resolution.
BEV_FEATURE_STRIDE = 2


def pillar_grid_size(lidar_range: Sequence[float], voxel_size: Sequence[float]) -> np.ndarray:
    """Number of pillars along (x, y, z) for a lidar range and voxel size."""
    extent = np.array(lidar_range[3:6]) - np.array(lidar_range[0:3])
    return np.round(extent / np.array(voxel_size)).astype(np.int64)


def bev_feature_size(
    lidar_range: Sequence[float], voxel_size: Sequence[float]
) -> tuple[int, int]:
    """Return the ``(H, W)`` of the BEV feature map the encoder will emit.

    For the paper's range and voxel size this is ``(96, 256)``.
    """
    grid = pillar_grid_size(lidar_range, voxel_size)
    return int(grid[1] // BEV_FEATURE_STRIDE), int(grid[0] // BEV_FEATURE_STRIDE)


class PointPillarsBEVEncoder(nn.Module):
    """Wraps OpenCOOD's PointPillars sub-modules into a plain BEV encoder.

    Parameters
    ----------
    args:
        The ``model.args`` block of an OpenCOOD PointPillars config.  Must
        contain ``pillar_vfe``, ``point_pillar_scatter``, ``base_bev_backbone``,
        ``voxel_size`` and ``lidar_range``; ``shrink_header`` is optional but
        required to reproduce the paper's 256-channel feature map.
    """

    def __init__(self, args: Mapping[str, Any]) -> None:
        super().__init__()
        # Imported lazily so the package stays importable without OpenCOOD on
        # the path (the rest of this repo follows the same convention).
        from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
        from opencood.models.sub_modules.downsample_conv import DownsampleConv
        from opencood.models.sub_modules.pillar_vfe import PillarVFE
        from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter

        for required in ("pillar_vfe", "point_pillar_scatter", "base_bev_backbone"):
            if required not in args:
                raise ValueError(f"model args missing required block '{required}'")

        self.grid_size = pillar_grid_size(args["lidar_range"], args["voxel_size"])

        self.pillar_vfe = PillarVFE(
            args["pillar_vfe"],
            num_point_features=4,
            voxel_size=args["voxel_size"],
            point_cloud_range=args["lidar_range"],
        )
        # OpenCOOD normally injects `grid_size` via its yaml parser; derive it
        # here so the config stays a plain, self-contained dict.
        self.scatter = PointPillarScatter(
            {**args["point_pillar_scatter"], "grid_size": self.grid_size}
        )
        self.backbone = BaseBEVBackbone(args["base_bev_backbone"], 64)

        self.shrink_conv = (
            DownsampleConv(args["shrink_header"]) if "shrink_header" in args else None
        )
        self.out_channels = (
            args["shrink_header"]["dim"][-1] if self.shrink_conv is not None else 384
        )

    def forward(self, processed_lidar: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Encode a batch of voxelized point clouds into a BEV feature map.

        Parameters
        ----------
        processed_lidar:
            Collated output of OpenCOOD's ``SpVoxelPreprocessor``, i.e. keys
            ``voxel_features``, ``voxel_coords`` and ``voxel_num_points``.

        Returns
        -------
        torch.Tensor
            BEV features of shape ``(B, out_channels, H, W)``.
        """
        batch_dict = {
            "voxel_features": processed_lidar["voxel_features"],
            "voxel_coords": processed_lidar["voxel_coords"],
            "voxel_num_points": processed_lidar["voxel_num_points"],
        }
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        features = batch_dict["spatial_features_2d"]
        if self.shrink_conv is not None:
            features = self.shrink_conv(features)
        return features

    def load_fcooper_checkpoint(self, checkpoint_path: str, strict: bool = True) -> None:
        """Load the shared backbone weights from a trained F-Cooper checkpoint.

        The detection heads are discarded, matching the paper's Section IV-B
        protocol: "we remove the perception head, and add the proposed
        CoLoca-QuA model".
        """
        state = torch.load(checkpoint_path, map_location="cpu")
        state = state.get("model_state_dict", state)
        backbone_state = {
            key: value
            for key, value in state.items()
            if not key.startswith(_DETECTION_HEAD_PREFIXES)
        }

        missing, unexpected = self.load_state_dict(backbone_state, strict=False)
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"F-Cooper checkpoint does not match the encoder. "
                f"Missing keys: {sorted(missing)}. Unexpected keys: {sorted(unexpected)}."
            )

    def freeze(self) -> None:
        """Freeze every parameter and put normalization layers in eval mode.

        Used for the paper's stage-2 training, where the backbone is fixed and
        only the localization module is optimized.
        """
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def unfreeze(self) -> None:
        """Re-enable gradients for stage-3 end-to-end fine-tuning."""
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        self.train()

    def train(self, mode: bool = True) -> PointPillarsBEVEncoder:
        """Keep a frozen encoder in eval mode so its BatchNorm stats stay fixed."""
        frozen = not any(parameter.requires_grad for parameter in self.parameters())
        return super().train(False if frozen else mode)
