"""Per-detection appearance embeddings for the matcher: frozen DINOv2 crop
descriptors through the trained cross-agent head, pooled over the sending
agent's own track.

Where this sits
---------------
The appearance probe (``scripts/train_v2xreal_appearance_head_sparse.py``)
found that a two-layer head over frozen DINOv2 descriptors, stopped early
and pooled over a few of the agent's own frames, separates a vehicle from
its nearest neighbour across agents on the pairs where the matcher loses
(V2X-Real val, shared 1-2 bucket AUC 0.885). That was on annotated boxes
with ground-truth tracks. This module brings it to the matcher's own
inputs: the detector's boxes, and a track the agent builds itself.

It plugs into the existing LiDAR+camera trunk unchanged: the embedding
head (``alignformer.embedding``) takes a per-object ``camera`` vector of
``camera_dim`` and a ``has_camera`` mask; here the vector is the pooled
128-d head output instead of a ResNet ROI feature. The cache
(``alignformer.cache``) stores the pooled vector in ``camera`` for training
and the per-frame vector in ``camera_raw`` so evaluation can pool the live
frame with its cached predecessors the same way.

Pooling is CAUSAL: the current frame and up to ``window`` earlier frames of
the same agent, matched by mutual-nearest world-frame centre within a gate
that grows with the frame gap (the agent's own pose moves its own boxes;
no other agent's pose is involved). Deployable as is: the sender pools
before it transmits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from embedding_aware_belt_fusion.alignformer.camera import MIN_CROP_PIXELS, silhouette_mask, world_from_pose
from embedding_aware_belt_fusion.alignformer.camera_features import (
    Calibration,
    CameraFeatures,
    box_2d,
    choose_camera,
    project_points,
)
from embedding_aware_belt_fusion.alignformer.foundation_features import DESCRIPTOR_NAMES, context_box

CAMERA_SOURCE_KEY = "camera_source"
DINO_HEAD_BACKBONE = "dino_head"
DEFAULT_TRACK_WINDOW = 4  # earlier frames pooled with the current one (10 Hz: 0.4 s)
DEFAULT_GATE_M = 2.0
DEFAULT_GATE_PER_FRAME_M = 1.5
_EPSILON = 1e-12


class TrackPooling(NamedTuple):
    window: int = DEFAULT_TRACK_WINDOW
    gate_m: float = DEFAULT_GATE_M
    gate_per_frame_m: float = DEFAULT_GATE_PER_FRAME_M

    def gate(self, frames_apart: int) -> float:
        return self.gate_m + self.gate_per_frame_m * frames_apart


# ----------------------------------------------------------------------------
# Per-frame features
# ----------------------------------------------------------------------------


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms > _EPSILON, norms, 1.0)


class DinoHeadBackbone(nn.Module):
    """Frozen DINOv2 + the trained appearance head; ``frame_features`` mirrors ``frame_camera_features``."""

    def __init__(self, foundation, head: nn.Module, pooling: Optional[TrackPooling] = None,
                 cache_root: Optional[Path] = None) -> None:
        super().__init__()
        self.foundation = foundation
        self.head = head.eval()
        for parameter in self.head.parameters():
            parameter.requires_grad_(False)
        self.pooling = pooling
        self.cache_root = cache_root
        with torch.no_grad():
            probe = torch.zeros(1, int(self.head.net[1].in_features), device=next(head.parameters()).device)
            self.feature_dim = int(self.head(probe).shape[1])

    def train(self, mode: bool = True) -> "DinoHeadBackbone":  # noqa: D401 - frozen
        return super().train(False)

    def embed(self, descriptors: Dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate the descriptors in the training order and project with the head."""
        stacked = np.concatenate([descriptors[name] for name in DESCRIPTOR_NAMES], axis=1).astype(np.float32)
        device = next(self.head.parameters()).device
        with torch.no_grad():
            return self.head(torch.from_numpy(stacked).to(device)).cpu().numpy().astype(np.float32)

    def frame_features(self, corners_lidar: np.ndarray, cameras: Sequence[Tuple[np.ndarray, Calibration]]) -> CameraFeatures:
        return frame_dino_features(self, corners_lidar, cameras)


def _crop_for(image: np.ndarray, corners: np.ndarray, calib: Calibration) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """The context crop and its silhouette for one box in one camera, or None if too small."""
    pixels, _ = project_points(corners, calib)
    mask = silhouette_mask(pixels, image.shape)
    if int(mask.sum()) < MIN_CROP_PIXELS:
        return None
    x1, y1, x2, y2 = context_box(_bounds(pixels, image.shape), image.shape)
    return image[y1:y2, x1:x2], mask[y1:y2, x1:x2]


def _bounds(pixels: np.ndarray, image_shape) -> Tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x1, y1 = max(0, int(np.floor(pixels[:, 0].min()))), max(0, int(np.floor(pixels[:, 1].min())))
    x2, y2 = min(width - 1, int(np.ceil(pixels[:, 0].max()))), min(height - 1, int(np.ceil(pixels[:, 1].max())))
    return x1, y1, x2, y2


def frame_dino_features(backbone: DinoHeadBackbone, corners_lidar: np.ndarray,
                        cameras: Sequence[Tuple[np.ndarray, Calibration]]) -> CameraFeatures:
    """Per-frame (unpooled) head embeddings for ``(N, 8, 3)`` LiDAR-frame corners."""
    corners_lidar = np.asarray(corners_lidar, dtype=np.float64)
    count = corners_lidar.shape[0]
    candidates = [[box_2d(c, calib, (image.shape[1], image.shape[0])) for image, calib in cameras] for c in corners_lidar]
    chosen = np.asarray(choose_camera(candidates), dtype=np.int8) if count else np.zeros((0,), dtype=np.int8)
    crops, masks, rows = [], [], []
    for row in np.flatnonzero(chosen >= 0):
        image, calib = cameras[int(chosen[row])]
        crop = _crop_for(image, corners_lidar[row], calib)
        if crop is None:
            chosen[row] = -1
            continue
        crops.append(crop[0])
        masks.append(crop[1])
        rows.append(row)
    features = np.zeros((count, backbone.feature_dim), dtype=np.float32)
    if rows:
        features[rows] = backbone.embed(backbone.foundation.describe_many(crops, masks))
    return CameraFeatures(features=features, has_camera=chosen >= 0, camera_index=chosen)


def features_for_frame(backbone, corners_lidar: np.ndarray, cameras) -> CameraFeatures:
    """ResNet or DINO-head: whichever backbone the cache or the evaluation was given."""
    if hasattr(backbone, "frame_features"):
        return backbone.frame_features(corners_lidar, cameras)
    from embedding_aware_belt_fusion.alignformer.camera_features import frame_camera_features

    return frame_camera_features(backbone, corners_lidar, cameras)


# ----------------------------------------------------------------------------
# The agent's own track, and pooling along it
# ----------------------------------------------------------------------------


def world_centres(boxes: np.ndarray, lidar_pose: Sequence[float]) -> np.ndarray:
    """``(N, 2)`` world-frame xy of box centres given in the agent's LiDAR frame."""
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.shape[0] == 0:
        return np.zeros((0, 2))
    homogeneous = np.concatenate([boxes[:, :3], np.ones((boxes.shape[0], 1))], axis=1)
    return (world_from_pose(lidar_pose) @ homogeneous.T).T[:, :2]


def match_to_current(current_xy: np.ndarray, past_xy: np.ndarray, gate_m: float) -> np.ndarray:
    """For each past box the index of its mutual-nearest current box within ``gate_m``, else -1."""
    mapping = -np.ones(len(past_xy), dtype=np.int64)
    if len(current_xy) == 0 or len(past_xy) == 0:
        return mapping
    distance = np.linalg.norm(past_xy[:, None, :] - current_xy[None, :, :], axis=2)
    nearest_current = distance.argmin(axis=1)
    nearest_past = distance.argmin(axis=0)
    for p, c in enumerate(nearest_current):
        if nearest_past[c] == p and distance[p, c] <= gate_m:
            mapping[p] = c
    return mapping


def pool_embeddings(current: np.ndarray, current_has: np.ndarray,
                    pasts: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """Unit-normalized sum of the current embedding and every matched past one that had a camera.

    ``pasts`` are ``(raw, has_camera, mapping)`` per earlier frame, mapping as
    ``match_to_current``. A box seen now without a camera view but tracked
    to earlier views gets those views' pooled embedding and ``has_camera``.
    """
    pooled = np.where(current_has[:, None], current, 0.0).astype(np.float64)
    has = current_has.copy()
    for raw, past_has, mapping in pasts:
        for p, c in enumerate(mapping):
            if c >= 0 and past_has[p]:
                pooled[c] += raw[p]
                has[c] = True
    return _unit_rows(pooled).astype(np.float32), has


class FrameRef(NamedTuple):
    scenario: str
    cav_id: str
    timestamp: str


def frame_ref_from_yaml(yaml_path) -> FrameRef:
    path = Path(yaml_path)
    return FrameRef(path.parents[1].name, path.parent.name, path.stem)


def earlier_timestamps(all_timestamps: Sequence[str], timestamp: str, window: int) -> List[str]:
    """Up to ``window`` timestamps before ``timestamp`` in the agent's own ordering, nearest first."""
    ordered = sorted(all_timestamps)
    here = ordered.index(timestamp)
    return list(reversed(ordered[max(0, here - window) : here]))


def pooled_from_cache(current: CameraFeatures, boxes: np.ndarray, lidar_pose: Sequence[float], *, cache_root: Path,
                      split_name: str, ref: FrameRef, agent_dir: Path, pooling: TrackPooling,
                      pose_of) -> CameraFeatures:
    """Pool the live frame with its cached earlier frames (``camera_raw``), the agent's own track."""
    from embedding_aware_belt_fusion.alignformer.cache import cache_path, read_frame

    stamps = sorted(p.stem for p in agent_dir.glob("*.yaml") if not p.name.startswith("._"))
    current_xy = world_centres(boxes, lidar_pose)
    pasts = []
    for gap, stamp in enumerate(earlier_timestamps(stamps, ref.timestamp, pooling.window), start=1):
        path = cache_path(cache_root, split_name, ref.scenario, ref.cav_id, stamp)
        if not path.exists():
            continue
        record = read_frame(path)
        if record.camera_raw is None:
            continue
        past_xy = world_centres(record.boxes, pose_of(agent_dir / f"{stamp}.yaml"))
        pasts.append((record.camera_raw.astype(np.float32), record.has_camera, match_to_current(current_xy, past_xy, pooling.gate(gap))))
    pooled, has = pool_embeddings(current.features, current.has_camera, pasts)
    return CameraFeatures(features=pooled, has_camera=has, camera_index=current.camera_index)


def _lidar_pose(yaml_path) -> List[float]:
    from embedding_aware_belt_fusion.alignformer.v2xreal import fast_load_yaml

    return list(fast_load_yaml(str(yaml_path))["lidar_pose"])


def pooled_live(backbone: DinoHeadBackbone, features: CameraFeatures, boxes: np.ndarray, yaml_path) -> CameraFeatures:
    """Evaluation: pool the live frame's features with the cached earlier frames of the same agent."""
    if backbone.pooling is None or backbone.cache_root is None:
        return features
    path = Path(yaml_path)
    return pooled_from_cache(features, boxes, _lidar_pose(path), cache_root=backbone.cache_root,
                             split_name=path.parents[2].name, ref=frame_ref_from_yaml(path), agent_dir=path.parent,
                             pooling=backbone.pooling, pose_of=_lidar_pose)


def pool_split(cache_root: Path, split_name: str, split_root: Path, pooling: TrackPooling) -> int:
    """Cache build: rewrite every record's ``camera`` as the causal track pool of its ``camera_raw``."""
    from dataclasses import replace

    from embedding_aware_belt_fusion.alignformer.cache import cache_path, read_frame, write_frame

    pooled_count = 0
    for scenario_dir in sorted(p for p in Path(split_root).iterdir() if p.is_dir()):
        for agent_dir in sorted(p for p in scenario_dir.iterdir() if p.is_dir()):
            stamps = sorted(p.stem for p in agent_dir.glob("*.yaml") if not p.name.startswith("._"))
            records = {t: read_frame(cache_path(cache_root, split_name, scenario_dir.name, agent_dir.name, t))
                       for t in stamps if cache_path(cache_root, split_name, scenario_dir.name, agent_dir.name, t).exists()}
            poses = {t: _lidar_pose(agent_dir / f"{t}.yaml") for t in records}
            for stamp, record in records.items():
                raw = record.camera_raw if record.camera_raw is not None else record.camera
                if raw is None:
                    continue
                current = CameraFeatures(raw.astype(np.float32), record.has_camera, record.camera_index)
                pasts = [(records[t].camera_raw.astype(np.float32) if records[t].camera_raw is not None else records[t].camera.astype(np.float32),
                          records[t].has_camera,
                          match_to_current(world_centres(record.boxes, poses[stamp]), world_centres(records[t].boxes, poses[t]), pooling.gate(gap)))
                         for gap, t in enumerate(earlier_timestamps(list(records), stamp, pooling.window), start=1)]
                pooled, has = pool_embeddings(current.features, current.has_camera, pasts)
                write_frame(cache_path(cache_root, split_name, scenario_dir.name, agent_dir.name, stamp),
                            replace(record, camera=pooled, has_camera=has, camera_raw=raw))
                pooled_count += 1
    return pooled_count


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------


def pooling_from_config(source: Dict) -> Optional[TrackPooling]:
    window = int(source.get("track_window", 0))
    if window <= 0:
        return None
    return TrackPooling(window, float(source.get("track_gate_m", DEFAULT_GATE_M)),
                        float(source.get("track_gate_per_frame_m", DEFAULT_GATE_PER_FRAME_M)))


def build_dino_backbone(source: Dict, device, cache_root: Optional[Path] = None) -> DinoHeadBackbone:
    """``model.camera_source`` -> the DINO-head backbone (``backbone: dino_head``)."""
    from embedding_aware_belt_fusion.alignformer.appearance_head import load_head
    from embedding_aware_belt_fusion.alignformer.foundation_features import FoundationBackbone

    if source.get("backbone") != DINO_HEAD_BACKBONE:
        raise ValueError(f"unknown camera_source.backbone {source.get('backbone')!r}; expected {DINO_HEAD_BACKBONE!r}")
    device_name = str(device)
    foundation = FoundationBackbone(model=str(source.get("dino", "base")), device=device_name)
    head = load_head(Path(source["head_checkpoint"]), device=device_name)
    return DinoHeadBackbone(foundation, head, pooling_from_config(source), cache_root)
