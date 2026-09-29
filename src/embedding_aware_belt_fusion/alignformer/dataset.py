"""Pairwise object-set dataset with a localization-noise curriculum.

Splits are by scenario, never by frame: consecutive OPV2V frames are
near-duplicates and a frame-level split leaks. Reuses coloca/index.py for the
pair index and coloca/geometry.py for the exact SE(2) label.

The CAV's boxes arrive from the cache in the CAV's own LiDAR frame (see
``alignformer.boxes.AgentDetections``). ``trunk.tokenize`` requires both
object sets already in the ego frame, so ``__getitem__`` projects the CAV
boxes into the ego frame with the *noisy* CAV pose before returning them --
see ``_project_boxes_to_ego`` below. The two sets are then misaligned by
exactly the SE(2) error the model must learn to undo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from embedding_aware_belt_fusion.alignformer.boxes import BOX_YAW
from embedding_aware_belt_fusion.alignformer.cache import cache_path, read_frame
from embedding_aware_belt_fusion.alignformer.trunk import MAX_OBJECTS
from embedding_aware_belt_fusion.coloca.dataset import sample_rng
from embedding_aware_belt_fusion.coloca.geometry import (
    perturb_pose_2d,
    relative_pose_error,
    se2_matrix,
)
from embedding_aware_belt_fusion.coloca.index import AgentPair

# Yaw noise in degrees equals xy noise in metres, the convention this
# repository's existing sweeps already use (0.2 m / 0.2 deg, ...).
YAW_STD_PER_XY_STD = 1.0

# Box layout is OpenCOOD's "hwl" order [x, y, z, h, w, l, yaw] (see
# alignformer.boxes.AgentDetections); BOX_YAW (imported above) is the 7th
# (index 6) field, defined once in boxes.py. R34.
# Pose layout is OPV2V/CARLA's [x, y, z, roll, yaw, pitch] with angles in
# degrees (see coloca.geometry); x, y, yaw are indices 0, 1, 4.
_POSE_X, _POSE_Y, _POSE_YAW = 0, 1, 4

# Default epoch count when a dataset is constructed without a training-loop
# schedule (e.g. for evaluation): with total_epochs == 1, NoiseSchedule
# returns max_xy_std unconditionally, i.e. a fixed, non-curriculum noise
# level, which is what evaluating "at sigma" requires.
_DEFAULT_TOTAL_EPOCHS = 1
_DEFAULT_SEED = 20


@dataclass(frozen=True)
class NoiseSchedule:
    """Linear ramp of localization noise across training epochs."""

    max_xy_std: float

    def sigma_for_epoch(self, epoch: int, total_epochs: int) -> float:
        """Return the xy noise std for ``epoch``, ramping 0 -> ``max_xy_std``."""
        if total_epochs < 1:
            raise ValueError(f"total_epochs must be >= 1, got {total_epochs}")
        if total_epochs == 1:
            return self.max_xy_std
        fraction = min(max(epoch, 0), total_epochs - 1) / (total_epochs - 1)
        return self.max_xy_std * fraction


def correspondence_indices(
    ego_ids: List[Optional[str]], cav_ids: List[Optional[str]]
) -> Tuple[Tensor, Tensor]:
    """Match detections across agents by OPV2V physical object id.

    ``None`` means a detection matched no ground-truth object, so it can never
    correspond to anything - two ``None`` detections are different objects, not
    the same one. Duplicate ids keep only the first occurrence, so the result is
    a genuine one-to-one assignment.
    """
    ego_match = torch.full((len(ego_ids),), -1, dtype=torch.long)
    cav_match = torch.full((len(cav_ids),), -1, dtype=torch.long)

    first_cav: Dict[str, int] = {}
    for index, identifier in enumerate(cav_ids):
        if identifier is not None and identifier not in first_cav:
            first_cav[identifier] = index

    claimed = set()
    for ego_index, identifier in enumerate(ego_ids):
        if identifier is None:
            continue
        cav_index = first_cav.get(identifier)
        if cav_index is None or cav_index in claimed:
            continue
        ego_match[ego_index] = cav_index
        cav_match[cav_index] = ego_index
        claimed.add(cav_index)

    return ego_match, cav_match


# Per-object fields are padded to the batch maximum; the rest are stacked.
_EGO_FIELDS = ("ego_boxes", "ego_scores", "ego_roi", "ego_match")
_CAV_FIELDS = ("cav_boxes", "cav_scores", "cav_roi", "cav_match")
# Present only when the dataset was built with ``use_camera`` (the
# LiDAR+camera trunk); padded like the fields above when present.
_CAMERA_FIELDS = ("camera", "has_camera")
# Match targets pad with -1 ("no counterpart"), everything else with 0
# (which is ``False`` for the has_camera mask).
_PAD_VALUES = {"ego_match": -1, "cav_match": -1}


def _optional_fields(samples: List[Dict[str, Tensor]], prefix: str) -> Tuple[str, ...]:
    """The camera fields, if every sample carries them; none if none does."""
    names = tuple(f"{prefix}_{field}" for field in _CAMERA_FIELDS)
    carrying = [all(name in s for name in names) for s in samples]
    if all(carrying):
        return names
    if any(carrying):
        raise ValueError("some samples carry camera fields and others do not; one dataset per batch")
    return ()


def _pad(tensors: List[Tensor], size: int, value: float) -> Tensor:
    padded = []
    for tensor in tensors:
        deficit = size - tensor.shape[0]
        if deficit:
            shape = (deficit,) + tuple(tensor.shape[1:])
            tensor = torch.cat([tensor, tensor.new_full(shape, value)], dim=0)
        padded.append(tensor)
    return torch.stack(padded)


def collate(samples: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """Pad both object sets to the batch maximum and emit validity masks."""
    if not samples:
        raise ValueError("cannot collate an empty batch")

    ego_counts = [int(s["ego_boxes"].shape[0]) for s in samples]
    cav_counts = [int(s["cav_boxes"].shape[0]) for s in samples]
    ego_size, cav_size = max(ego_counts), max(cav_counts)

    batch: Dict[str, Tensor] = {}
    for field in _EGO_FIELDS + _optional_fields(samples, "ego"):
        batch[field] = _pad([s[field] for s in samples], ego_size, _PAD_VALUES.get(field, 0))
    for field in _CAV_FIELDS + _optional_fields(samples, "cav"):
        batch[field] = _pad([s[field] for s in samples], cav_size, _PAD_VALUES.get(field, 0))

    indices = torch.arange(ego_size)
    batch["ego_mask"] = indices.unsqueeze(0) < torch.tensor(ego_counts).unsqueeze(1)
    indices = torch.arange(cav_size)
    batch["cav_mask"] = indices.unsqueeze(0) < torch.tensor(cav_counts).unsqueeze(1)

    batch["psi_true"] = torch.stack([s["psi_true"] for s in samples])
    batch["t_true"] = torch.stack([s["t_true"] for s in samples])
    return batch


def _project_boxes_to_ego(boxes: np.ndarray, agent_pose: Sequence[float],
                           ego_pose: Sequence[float]) -> np.ndarray:
    """Project ``boxes`` (agent-frame, hwl order) into the ego frame.

    Uses the same SE(2)-only machinery as
    ``coloca.geometry.relative_pose_error`` (dropping z, roll, pitch), so a box
    projected here with a noisy pose and then corrected by that function's
    label lands exactly where projecting with the true pose would -- see
    ``tests/test_alignformer_dataset.py::test_applying_the_label_realigns_the_cav_boxes``.
    """
    if boxes.shape[0] == 0:
        return boxes.copy()

    world_to_ego = np.linalg.inv(
        se2_matrix(ego_pose[_POSE_X], ego_pose[_POSE_Y], ego_pose[_POSE_YAW])
    )
    agent_to_world = se2_matrix(
        agent_pose[_POSE_X], agent_pose[_POSE_Y], agent_pose[_POSE_YAW]
    )
    transform = world_to_ego @ agent_to_world

    projected = boxes.copy()
    projected[:, :2] = boxes[:, :2] @ transform[:2, :2].T + transform[:2, 2]
    yaw_delta = np.arctan2(transform[1, 0], transform[0, 0])
    projected[:, BOX_YAW] = boxes[:, BOX_YAW] + yaw_delta
    return projected


def _top_indices(scores: np.ndarray, max_objects: int) -> Optional[np.ndarray]:
    """Indices of the top ``max_objects`` by descending score, or ``None`` for "keep all"."""
    if scores.shape[0] <= max_objects:
        return None
    return np.argsort(-scores, kind="stable")[:max_objects]


def _truncate_by_score(
    boxes: np.ndarray, scores: np.ndarray, roi: np.ndarray, gt_ids: Sequence[Optional[str]],
    max_objects: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Optional[str]]]:
    """Keep the top ``max_objects`` detections by descending score."""
    order = _top_indices(scores, max_objects)
    if order is None:
        return boxes, scores, roi, list(gt_ids)
    return boxes[order], scores[order], roi[order], [gt_ids[i] for i in order]


def _camera_fields(record, order: Optional[np.ndarray], prefix: str) -> Dict[str, Tensor]:
    """``{prefix}_camera`` and ``{prefix}_has_camera`` in the truncation order."""
    if record.camera is None:
        raise ValueError(
            f"use_camera was requested but the cached {prefix} frame carries no camera "
            "fields; rebuild the cache with --camera"
        )
    camera, has_camera = record.camera, record.has_camera
    if order is not None:
        camera, has_camera = camera[order], has_camera[order]
    return {
        f"{prefix}_camera": torch.from_numpy(camera.astype(np.float32)),
        f"{prefix}_has_camera": torch.from_numpy(has_camera.astype(bool)),
    }


class OPV2VObjectSetDataset(Dataset):
    """Yields (ego objects, noisily-projected CAV objects, SE(2) label) samples.

    Parameters
    ----------
    pairs:
        Pre-built :class:`~embedding_aware_belt_fusion.coloca.index.AgentPair`
        list, e.g. from ``load_or_build_pairs`` filtered to scenarios in this
        split -- splits are by scenario, never by frame.
    cache_root, split:
        Locate each agent-frame's cached detections via
        ``alignformer.cache.cache_path``.
    noise_schedule:
        Ramps the CAV pose noise's xy std across training epochs.
    train:
        When True the noise is redrawn every epoch (see ``sample_rng``); when
        False it is deterministic per sample so evaluation is reproducible.
    total_epochs:
        Denominator for the noise ramp. Left at 1 (the default), the schedule
        returns ``max_xy_std`` unconditionally, i.e. a fixed evaluation noise
        level; a training loop should pass the real epoch count and call
        ``set_epoch`` each epoch.
    seed:
        Base seed forwarded to ``sample_rng``.
    """

    def __init__(
        self,
        pairs: Sequence[AgentPair],
        cache_root: Union[Path, str],
        split: str,
        *,
        noise_schedule: NoiseSchedule,
        train: bool,
        total_epochs: int = _DEFAULT_TOTAL_EPOCHS,
        seed: int = _DEFAULT_SEED,
        use_camera: bool = False,
    ) -> None:
        if not pairs:
            raise ValueError("pairs must be non-empty")
        self.pairs = list(pairs)
        self.cache_root = Path(cache_root)
        self.split = split
        self.noise_schedule = noise_schedule
        self.train = train
        self.total_epochs = total_epochs
        self.seed = seed
        self.use_camera = use_camera
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.pairs)

    def set_epoch(self, epoch: int) -> None:
        """Re-seed training noise per epoch so runs stay reproducible."""
        self.epoch = epoch

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        pair = self.pairs[index]
        rng = sample_rng(self.seed, self.epoch, index, train=self.train)

        sigma_xy = self.noise_schedule.sigma_for_epoch(self.epoch, self.total_epochs)
        sigma_yaw_deg = sigma_xy * YAW_STD_PER_XY_STD
        noisy_cav_pose = perturb_pose_2d(pair.cav_pose, sigma_xy, sigma_yaw_deg, rng)

        ego_record = read_frame(
            cache_path(self.cache_root, self.split, pair.scenario, pair.ego_id, pair.timestamp)
        )
        cav_record = read_frame(
            cache_path(self.cache_root, self.split, pair.scenario, pair.cav_id, pair.timestamp)
        )

        cav_boxes_ego = _project_boxes_to_ego(cav_record.boxes, noisy_cav_pose, pair.ego_pose)

        dx, dy, dpsi_deg = relative_pose_error(pair.ego_pose, pair.cav_pose, noisy_cav_pose)
        psi_true = float(np.radians(dpsi_deg))
        t_true = np.array([dx, dy], dtype=np.float32)

        # Truncate before matching so ego_match/cav_match index the final,
        # returned arrays directly -- matching first and truncating after
        # would require remapping indices for objects dropped by truncation.
        ego_order = _top_indices(ego_record.scores, MAX_OBJECTS)
        cav_order = _top_indices(cav_record.scores, MAX_OBJECTS)
        ego_boxes, ego_scores, ego_roi, ego_ids = _truncate_by_score(
            ego_record.boxes, ego_record.scores, ego_record.roi, ego_record.gt_ids, MAX_OBJECTS
        )
        cav_boxes_ego, cav_scores, cav_roi, cav_ids = _truncate_by_score(
            cav_boxes_ego, cav_record.scores, cav_record.roi, cav_record.gt_ids, MAX_OBJECTS
        )

        ego_match, cav_match = correspondence_indices(ego_ids, cav_ids)

        camera: Dict[str, Tensor] = {}
        if self.use_camera:
            camera.update(_camera_fields(ego_record, ego_order, "ego"))
            camera.update(_camera_fields(cav_record, cav_order, "cav"))

        return {
            **camera,
            "ego_boxes": torch.from_numpy(ego_boxes.astype(np.float32)),
            "ego_scores": torch.from_numpy(ego_scores.astype(np.float32)),
            "ego_roi": torch.from_numpy(ego_roi.astype(np.float32)),
            "cav_boxes": torch.from_numpy(cav_boxes_ego.astype(np.float32)),
            "cav_scores": torch.from_numpy(cav_scores.astype(np.float32)),
            "cav_roi": torch.from_numpy(cav_roi.astype(np.float32)),
            "ego_match": ego_match,
            "cav_match": cav_match,
            "psi_true": torch.tensor(psi_true, dtype=torch.float32),
            "t_true": torch.from_numpy(t_true),
        }
