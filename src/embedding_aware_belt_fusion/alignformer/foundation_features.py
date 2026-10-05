"""Frozen DINOv2 crop descriptors: does a foundation model's appearance
embedding carry cross-agent object identity on real traffic?

Where this sits
---------------
Two appearance nulls precede it. The LiDAR+camera trunk used a frozen
ImageNet ResNet-18 pooled over projected boxes and cost AP (write-up caveat
5: a null about that feature, not about cameras). The colour probe used
hue-saturation histograms and failed its signal bar on both datasets. The
cheapest next question is whether a self-supervised vision foundation
model, zero-shot, separates the same vehicle seen from two agents from its
nearest neighbour -- asked through the same pre-registered probe
(``scripts/analyze_v2xreal_colour_separability.py``) before any training.

Two descriptors, because they fail differently. ``dino_cls`` is the pooled
class token of the letterboxed crop with a little context around the box.
``dino_patch_mean`` is the mean of the patch tokens whose patch the vehicle's
projected silhouette covers, so a neighbour sharing the crop contributes
less. Both are L2-normalized; cosine is the score.

Frozen on purpose: if zero-shot clears the bar, a projection head trained
on cross-agent pairs is the follow-up; if it does not, the null is about
frozen DINOv2 features on these crops at this resolution.
"""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor, nn

MODEL_NAMES = {
    "small": "vit_small_patch14_dinov2.lvd142m",
    "base": "vit_base_patch14_dinov2.lvd142m",
}
DEFAULT_MODEL = "small"
INPUT_SIZE = 224
PATCH_SIZE = 14
PATCH_GRID = INPUT_SIZE // PATCH_SIZE  # 16
CONTEXT_FRACTION = 0.10  # box padding on each side before the crop
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DESCRIPTOR_NAMES = ("dino_cls", "dino_patch_mean")
_EPSILON = 1e-12


def letterbox(crop_rgb: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """Resize the longer side to ``size`` and pad the other to a square.

    Aspect is kept: a 2:1 car squashed to a square is a different object to
    the network than the same car seen square-on.
    """
    height, width = crop_rgb.shape[:2]
    scale = size / max(height, width)
    new_w, new_h = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
    resized = cv2.resize(crop_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    fill = tuple(int(v) for v in np.rint(np.array(IMAGENET_MEAN) * 255))
    canvas = np.empty((size, size, 3), dtype=np.uint8)
    canvas[:] = fill
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def patch_grid_mask(mask_square: np.ndarray, grid: int = PATCH_GRID) -> np.ndarray:
    """``(grid, grid)`` booleans: patches at least half covered by the mask."""
    fraction = cv2.resize(mask_square.astype(np.float32), (grid, grid), interpolation=cv2.INTER_AREA)
    return fraction >= 0.5


def context_box(box: Tuple[int, int, int, int], image_shape, fraction: float = CONTEXT_FRACTION):
    """The projected box grown by ``fraction`` of its size on each side, clipped."""
    x1, y1, x2, y2 = box
    pad_x, pad_y = int(round((x2 - x1) * fraction)), int(round((y2 - y1) * fraction))
    height, width = image_shape[:2]
    return (max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x + 1), min(height, y2 + pad_y + 1))


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > _EPSILON else vector


class FoundationBackbone(nn.Module):
    """DINOv2 ViT (timm), frozen, at a 224-px letterboxed input."""

    def __init__(self, model: str = DEFAULT_MODEL, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> None:
        super().__init__()
        import timm

        if model not in MODEL_NAMES:
            raise ValueError(f"unknown DINOv2 size {model!r}; one of {sorted(MODEL_NAMES)}")
        self.name = MODEL_NAMES[model]
        self.vit = timm.create_model(self.name, pretrained=True, num_classes=0, img_size=INPUT_SIZE)
        for parameter in self.vit.parameters():
            parameter.requires_grad_(False)
        self.embed_dim = int(self.vit.num_features)
        self.device_name = device
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.to(device)
        self.eval()

    def train(self, mode: bool = True) -> "FoundationBackbone":  # noqa: D401 - frozen
        """Always evaluation mode; the backbone is never trained here."""
        return super().train(False)

    def _tokens(self, squares_rgb: np.ndarray) -> Tuple[Tensor, Tensor]:
        """Pooled class token and ``(N, grid*grid, D)`` patch tokens."""
        batch = torch.from_numpy(np.ascontiguousarray(squares_rgb)).permute(0, 3, 1, 2).float().div_(255.0)
        batch = (batch.to(self.mean.device) - self.mean) / self.std
        with torch.no_grad():
            features = self.vit.forward_features(batch)
            pooled = self.vit.forward_head(features, pre_logits=True)
        return pooled, features[:, self.vit.num_prefix_tokens :]

    def describe(self, crop_rgb: np.ndarray, crop_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """Both descriptors of one crop; ``crop_mask`` is the silhouette inside it."""
        square = letterbox(crop_rgb)
        square_mask = letterbox(np.repeat(crop_mask[..., None].astype(np.uint8) * 255, 3, axis=2))[..., 0] > 127
        pooled, patches = self._tokens(square[None])
        grid = torch.from_numpy(patch_grid_mask(square_mask).reshape(-1)).to(patches.device)
        selected = patches[0][grid] if bool(grid.any()) else patches[0]
        return {
            "dino_cls": _unit(pooled[0].cpu().numpy().astype(np.float64)),
            "dino_patch_mean": _unit(selected.mean(dim=0).cpu().numpy().astype(np.float64)),
        }

    def describe_projection(self, image_rgb: np.ndarray, silhouette: np.ndarray, box) -> Dict[str, np.ndarray]:
        """Descriptors of the vehicle at ``box`` in a full image, with context."""
        x1, y1, x2, y2 = context_box(box, image_rgb.shape)
        return self.describe(image_rgb[y1:y2, x1:x2], silhouette[y1:y2, x1:x2])
