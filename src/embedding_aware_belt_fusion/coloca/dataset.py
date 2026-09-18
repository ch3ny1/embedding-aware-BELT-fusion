"""OPV2V pairwise dataset for CoLoca-QuA.

Implements the paper's data pipeline (Section III-A, steps 1-2):

1. The CAV's self-reported pose is corrupted with Gaussian localization noise.
2. The CAV point cloud is projected into the ego frame with that *noisy*
   relative transform (Eq. 2), so the two clouds are deliberately misaligned.
3. Both clouds are cropped to the ego sensing range and voxelized.
4. The label is the SE(2) correction that undoes the misalignment (Eq. 5).

The ego pose is left clean: the paper adds noise only to the CAV/infrastructure
pose, and estimates the *relative* pose error.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from embedding_aware_belt_fusion.coloca.geometry import perturb_pose_2d, relative_pose_error
from embedding_aware_belt_fusion.coloca.index import AgentPair, load_or_build_pairs


def sample_rng(seed: int, epoch: int, index: int, train: bool) -> np.random.Generator:
    """Per-sample RNG: fresh noise each epoch when training, fixed otherwise.

    Seeding from ``(seed, epoch, index)`` keeps a run reproducible while still
    showing the model a different perturbation of each pair every epoch.

    Note this only varies per epoch if the worker processes actually observe the
    updated epoch, which is why the training loader must not use persistent
    workers -- see the comment in :mod:`embedding_aware_belt_fusion.coloca.train`.
    """
    if train:
        return np.random.default_rng([seed, epoch, index])
    return np.random.default_rng([seed, index])


class OPV2VPoseErrorDataset(Dataset):
    """Yields (ego cloud, noisily-projected CAV cloud, SE(2) error) triples.

    Parameters
    ----------
    root_dir:
        An OPV2V split directory (``train`` / ``test``).
    preprocess_params:
        The ``preprocess`` block of the OpenCOOD config, forwarded to
        ``SpVoxelPreprocessor``.
    xy_std:
        Position-noise standard deviation in metres.  A sequence draws one of
        the listed values uniformly per sample, which is how the paper trains a
        single model: "pose noises are ... randomly sampled from an error
        dataset that includes Gaussian noise with ... sigma = 2 m and sigma =
        1 m" (Section IV-B), then evaluated at each sigma separately.
    yaw_std_deg:
        Heading-noise standard deviation in degrees (1 deg in the paper).
    train:
        When True the noise is redrawn every epoch (the paper's training-time
        augmentation).  When False the noise is deterministic per sample so
        evaluation is reproducible.
    scenarios:
        Optional whitelist of scenario names, used to carve a validation split
        out of ``train`` (OPV2V ships no separate validation split here).
    pcd_cache_root:
        Optional directory holding the binary point-cloud cache built by
        :mod:`embedding_aware_belt_fusion.coloca.pcd_cache`.  Without it the
        loader decodes OPV2V's ASCII pcd files directly, which is ~20x slower.
    """

    def __init__(
        self,
        root_dir: str | Path,
        preprocess_params: Mapping[str, Any],
        cache_path: str | Path,
        xy_std: float | Sequence[float] = 2.0,
        yaw_std_deg: float = 1.0,
        comm_range_m: float = 40.0,
        train: bool = True,
        seed: int = 20,
        scenarios: Sequence[str] | None = None,
        pcd_cache_root: str | Path | None = None,
    ) -> None:
        from opencood.data_utils.pre_processor.sp_voxel_preprocessor import SpVoxelPreprocessor

        self.root_dir = Path(root_dir)
        self.pcd_cache_root = Path(pcd_cache_root) if pcd_cache_root else None
        self.xy_std_choices = (
            (float(xy_std),) if np.isscalar(xy_std) else tuple(float(s) for s in xy_std)
        )
        if not self.xy_std_choices:
            raise ValueError("xy_std must be a float or a non-empty sequence of floats")
        self.yaw_std_deg = yaw_std_deg
        self.train = train
        self.seed = seed
        self.epoch = 0

        self.pairs = load_or_build_pairs(self.root_dir, cache_path, comm_range_m)
        if scenarios is not None:
            allowed = set(scenarios)
            self.pairs = [pair for pair in self.pairs if pair.scenario in allowed]
            if not self.pairs:
                raise ValueError(f"no pairs left after filtering to scenarios {sorted(allowed)}")

        self.preprocessor = SpVoxelPreprocessor(dict(preprocess_params), train=train)
        self.lidar_range = list(preprocess_params["cav_lidar_range"])

    def __len__(self) -> int:
        return len(self.pairs)

    def scenario_names(self) -> list[str]:
        return sorted({pair.scenario for pair in self.pairs})

    def set_epoch(self, epoch: int) -> None:
        """Re-seed training noise per epoch so runs stay reproducible."""
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, Any]:
        from opencood.utils.pcd_utils import mask_points_by_range
        from opencood.utils.transformation_utils import x1_to_x2

        pair = self.pairs[index]
        rng = self._rng_for(index)

        # Step 1: corrupt the CAV's reported pose (Section IV-B).
        xy_std = float(rng.choice(self.xy_std_choices))
        noisy_cav_pose = perturb_pose_2d(pair.cav_pose, xy_std, self.yaw_std_deg, rng)

        # Step 2: project the CAV cloud with the *noisy* transform (Eq. 2).
        ego_points = self._load_points(pair, pair.ego_id)
        cav_points = self._load_points(pair, pair.cav_id)
        noisy_transform = x1_to_x2(noisy_cav_pose, list(pair.ego_pose))
        cav_points = _transform_points(cav_points, noisy_transform)

        # Step 3: crop both clouds to the ego sensing range and voxelize.
        ego_points = mask_points_by_range(ego_points, self.lidar_range)
        cav_points = mask_points_by_range(cav_points, self.lidar_range)

        # Step 4: the label is the correction that undoes the misalignment.
        pose_error = relative_pose_error(pair.ego_pose, pair.cav_pose, noisy_cav_pose)

        return {
            "ego_lidar": self.preprocessor.preprocess(ego_points),
            "cav_lidar": self.preprocessor.preprocess(cav_points),
            "pose_error": np.asarray(pose_error, dtype=np.float32),
            "meta": {
                "scenario": pair.scenario,
                "timestamp": pair.timestamp,
                "ego_id": pair.ego_id,
                "cav_id": pair.cav_id,
                "xy_std": xy_std,
            },
        }

    def _rng_for(self, index: int) -> np.random.Generator:
        return sample_rng(self.seed, self.epoch, index, train=self.train)

    def _pcd_path(self, pair: AgentPair, cav_id: str) -> Path:
        return self.root_dir / pair.scenario / cav_id / f"{pair.timestamp}.pcd"

    def _load_points(self, pair: AgentPair, cav_id: str) -> np.ndarray:
        from embedding_aware_belt_fusion.coloca.pcd_cache import load_points

        return load_points(self._pcd_path(pair, cav_id), self.pcd_cache_root, self.root_dir)

    def collate(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate into the dict consumed by :class:`ColocaQuANet`."""
        return {
            "ego_lidar": _to_tensors(
                self.preprocessor.collate_batch([item["ego_lidar"] for item in batch])
            ),
            "cav_lidar": _to_tensors(
                self.preprocessor.collate_batch([item["cav_lidar"] for item in batch])
            ),
            "pose_error": torch.from_numpy(
                np.stack([item["pose_error"] for item in batch])
            ),
            "meta": [item["meta"] for item in batch],
        }


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to (N, 4) xyz+intensity points, preserving intensity."""
    if points.shape[0] == 0:
        return points
    xyz = points[:, :3] @ transform[:3, :3].T + transform[:3, 3]
    return np.hstack([xyz, points[:, 3:]]).astype(np.float32)


def _to_tensors(collated: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Normalize OpenCOOD's collate output to tensors regardless of backend version."""
    return {
        key: value if torch.is_tensor(value) else torch.from_numpy(np.asarray(value))
        for key, value in collated.items()
    }
