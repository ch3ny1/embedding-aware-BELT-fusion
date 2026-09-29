"""Per-object camera features: a frozen ImageNet trunk pooled over projected boxes.

The LiDAR+camera trunk needs, for each detected object, one vector that
comes from the agent's cameras. This module produces it once, offline,
into the cache beside the LiDAR ROI feature, so that the three trunks of
the V2X-Real comparison train through an identical pipeline and differ
only in what the embedding head is fed.

Projection is ``K @ inv(extrinsic) @ p_lidar``. V2X-Real stores
``extrinsic`` as **camera->LiDAR**; that convention was checked by eye (the
other one puts the boxes in the sky; memory ``v2x-real-dataset-state``). No
axis permutation: the stored matrices already map into OpenCV camera
coordinates.

An object is *visible* in a camera when every corner is at least
:data:`MIN_DEPTH_M` in front of it and the clipped 2-D box is at least
:data:`MIN_BOX_SIDE_PX` on each side. The camera with the larger visible
area wins. An object visible nowhere gets a zero vector and
``has_camera = False``; the flag travels with the feature so absence is
information rather than a zero pretending to be a feature.

The trunk is ResNet-18 cut after ``layer3``: stride 16, 256 channels, so a
2 m car at 60 m (~11 px at 1920 wide) still covers most of a cell.
``roi_align`` with a 2x2 output averaged to one vector. Frozen and in eval
mode; nothing here is ever trained.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn

STRIDE = 16
FEATURE_DIM = 256
ROI_OUTPUT = 2
MIN_DEPTH_M = 0.5
MIN_BOX_SIDE_PX = 8.0

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ImageSize = Tuple[int, int]  # (width, height)


class Calibration(NamedTuple):
    intrinsic: np.ndarray  # (3, 3)
    camera_to_lidar: np.ndarray  # (4, 4), as stored in the yaml


class Box2D(NamedTuple):
    x0: float
    y0: float
    x1: float
    y1: float
    visible_fraction: float  # clipped area over unclipped area

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)


@dataclass(frozen=True)
class CameraFeatures:
    """One agent-frame's per-object camera features."""

    features: np.ndarray  # (N, FEATURE_DIM) float32, zero where has_camera is False
    has_camera: np.ndarray  # (N,) bool
    camera_index: np.ndarray  # (N,) int8, -1 where has_camera is False

    def __post_init__(self) -> None:
        count = self.features.shape[0]
        if self.has_camera.shape != (count,) or self.camera_index.shape != (count,):
            raise ValueError("features, has_camera and camera_index disagree on the object count")


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------


def calibration_from_yaml(block: dict) -> Calibration:
    """The ``cam1`` / ``cam2`` block of a V2X-Real frame yaml."""
    return Calibration(
        intrinsic=np.asarray(block["intrinsic"], dtype=np.float64).reshape(3, 3),
        camera_to_lidar=np.asarray(block["extrinsic"], dtype=np.float64).reshape(4, 4),
    )


def project_points(points_lidar: np.ndarray, calib: Calibration) -> Tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates and depths of LiDAR-frame points; depth may be <= 0."""
    lidar_to_camera = np.linalg.inv(calib.camera_to_lidar)
    homogeneous = np.c_[points_lidar, np.ones(len(points_lidar))]
    camera = (lidar_to_camera @ homogeneous.T).T[:, :3]
    depth = camera[:, 2]
    safe_depth = np.where(np.abs(depth) < 1e-9, 1e-9, depth)
    pixels = (calib.intrinsic @ camera.T).T
    uv = pixels[:, :2] / safe_depth[:, None]
    return uv, depth


def box_2d(corners_lidar: np.ndarray, calib: Calibration, image_size: ImageSize) -> Optional[Box2D]:
    """Clipped 2-D bounding box of a 3-D box, or ``None`` if not visible."""
    uv, depth = project_points(np.asarray(corners_lidar, dtype=np.float64), calib)
    if depth.min() < MIN_DEPTH_M:
        return None
    width, height = image_size
    x0, y0 = uv.min(axis=0)
    x1, y1 = uv.max(axis=0)
    full_area = (x1 - x0) * (y1 - y0)
    cx0, cy0 = max(x0, 0.0), max(y0, 0.0)
    cx1, cy1 = min(x1, float(width)), min(y1, float(height))
    if cx1 - cx0 < MIN_BOX_SIDE_PX or cy1 - cy0 < MIN_BOX_SIDE_PX:
        return None
    visible = (cx1 - cx0) * (cy1 - cy0) / full_area if full_area > 0 else 0.0
    return Box2D(float(cx0), float(cy0), float(cx1), float(cy1), float(visible))


def choose_camera(candidates: Sequence[Sequence[Optional[Box2D]]]) -> List[int]:
    """Per object, the index of the camera with the largest visible box, or -1."""
    chosen = []
    for per_camera in candidates:
        best, best_area = -1, 0.0
        for index, box in enumerate(per_camera):
            if box is not None and box.area > best_area:
                best, best_area = index, box.area
        chosen.append(best)
    return chosen


# ----------------------------------------------------------------------------
# Appearance
# ----------------------------------------------------------------------------


class CameraBackbone(nn.Module):
    """ImageNet ResNet-18 through ``layer3``, frozen: stride 16, 256 channels."""

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        trunk = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(trunk.conv1, trunk.bn1, trunk.relu, trunk.maxpool)
        self.stages = nn.Sequential(trunk.layer1, trunk.layer2, trunk.layer3)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.eval()

    def train(self, mode: bool = True) -> "CameraBackbone":  # noqa: D401 - frozen
        return super().train(False)

    @torch.no_grad()
    def feature_map(self, image_rgb_uint8: np.ndarray) -> Tensor:
        """``(1, FEATURE_DIM, H/16, W/16)`` for one ``(H, W, 3)`` RGB uint8 image."""
        if image_rgb_uint8.ndim != 3 or image_rgb_uint8.shape[2] != 3:
            raise ValueError(f"expected an (H, W, 3) image, got {image_rgb_uint8.shape}")
        device = self.mean.device
        image = torch.from_numpy(np.ascontiguousarray(image_rgb_uint8)).to(device)
        image = image.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        image = (image - self.mean) / self.std
        return self.stages(self.stem(image))


def pooled_features(feature_map: Tensor, boxes_xyxy: Tensor) -> Tensor:
    """``(M, FEATURE_DIM)``: ``roi_align`` over each pixel-space box, averaged."""
    from torchvision.ops import roi_align

    if boxes_xyxy.shape[0] == 0:
        return feature_map.new_zeros((0, feature_map.shape[1]))
    rois = torch.cat(
        [boxes_xyxy.new_zeros((boxes_xyxy.shape[0], 1)), boxes_xyxy.to(feature_map.dtype)], dim=1
    ).to(feature_map.device)
    pooled = roi_align(
        feature_map, rois, output_size=ROI_OUTPUT, spatial_scale=1.0 / STRIDE, sampling_ratio=2, aligned=True
    )
    return pooled.mean(dim=(2, 3))


def load_image(path: Union[Path, str]) -> np.ndarray:
    """``(H, W, 3)`` RGB uint8."""
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


# ----------------------------------------------------------------------------
# One agent-frame
# ----------------------------------------------------------------------------


def frame_camera_features(
    backbone: CameraBackbone,
    corners_lidar: np.ndarray,
    cameras: Sequence[Tuple[np.ndarray, Calibration]],
) -> CameraFeatures:
    """Features for ``(N, 8, 3)`` box corners from ``(image, calibration)`` pairs."""
    corners_lidar = np.asarray(corners_lidar, dtype=np.float64)
    count = corners_lidar.shape[0]
    candidates = [
        [box_2d(corners, calib, (image.shape[1], image.shape[0])) for image, calib in cameras]
        for corners in corners_lidar
    ]
    chosen = np.asarray(choose_camera(candidates), dtype=np.int8) if count else np.zeros((0,), dtype=np.int8)
    features = np.zeros((count, FEATURE_DIM), dtype=np.float32)
    for camera_index, (image, _) in enumerate(cameras):
        objects = np.flatnonzero(chosen == camera_index)
        if objects.size == 0:
            continue
        boxes = torch.tensor([candidates[i][camera_index][:4] for i in objects], dtype=torch.float32)
        pooled = pooled_features(backbone.feature_map(image), boxes)
        features[objects] = pooled.cpu().numpy().astype(np.float32)
    return CameraFeatures(features=features, has_camera=chosen >= 0, camera_index=chosen)


__all__ = [
    "FEATURE_DIM",
    "MIN_BOX_SIDE_PX",
    "MIN_DEPTH_M",
    "STRIDE",
    "Box2D",
    "Calibration",
    "CameraBackbone",
    "CameraFeatures",
    "box_2d",
    "calibration_from_yaml",
    "choose_camera",
    "frame_camera_features",
    "load_image",
    "pooled_features",
    "project_points",
]
