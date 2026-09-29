"""Builds and caches the OPV2V agent-pair index used by the CoLoca-QuA dataset.

Each OPV2V frame yaml is several hundred kilobytes (camera matrices, every
vehicle in the scene), but the localization task only needs ``lidar_pose``.
Parsing them with PyYAML on every ``__getitem__`` would make the data loader the
bottleneck, so the poses are extracted once with a targeted line scan and cached
to a compressed npz keyed by the split directory.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from embedding_aware_belt_fusion.coloca.geometry import POSE_LENGTH

_LIDAR_POSE_KEY = "lidar_pose:"
# Paper Section IV-B: "The maximum communication distance between the ego
# vehicle and other agents is limited to 40 m."
DEFAULT_COMM_RANGE_M = 40.0
CACHE_VERSION = 1


@dataclass(frozen=True)
class AgentPair:
    """One (ego, cav) sample at a single timestamp."""

    scenario: str
    timestamp: str
    ego_id: str
    cav_id: str
    ego_pose: tuple[float, ...]
    cav_pose: tuple[float, ...]


def parse_lidar_pose(yaml_path: Path) -> list[float]:
    """Extract ``lidar_pose`` from an OPV2V frame yaml without a full parse.

    The block is a top-level key followed by exactly six ``- <float>`` lines,
    so a line scan is both correct and ~100x faster than ``yaml.safe_load``.
    """
    with open(yaml_path, "r") as handle:
        for line in handle:
            if not line.startswith(_LIDAR_POSE_KEY):
                continue
            pose = []
            for _ in range(POSE_LENGTH):
                entry = handle.readline()
                if not entry.startswith("- "):
                    raise ValueError(f"malformed lidar_pose block in {yaml_path}: {entry!r}")
                pose.append(float(entry[2:]))
            return pose
    raise ValueError(f"no lidar_pose key found in {yaml_path}")


# OPV2V ships <ts>.pcd; V2X-Real ships <ts>.bin (float32 x, y, z, i). A frame
# without either has no point cloud and is not a usable frame.
LIDAR_SUFFIXES = (".pcd", ".bin")


def _has_lidar(cav_dir: Path, stem: str) -> bool:
    return any((cav_dir / f"{stem}{suffix}").exists() for suffix in LIDAR_SUFFIXES)


def scan_split(root_dir: Path) -> dict[str, dict[str, dict[str, list[float]]]]:
    """Walk an OPV2V split and return ``{scenario: {cav_id: {timestamp: pose}}}``."""
    if not root_dir.is_dir():
        raise FileNotFoundError(f"OPV2V split directory not found: {root_dir}")

    poses: dict[str, dict[str, dict[str, list[float]]]] = {}
    for scenario in sorted(os.listdir(root_dir)):
        scenario_dir = root_dir / scenario
        if not scenario_dir.is_dir():
            continue
        agents: dict[str, dict[str, list[float]]] = {}
        for cav_id in sorted(os.listdir(scenario_dir)):
            cav_dir = scenario_dir / cav_id
            if not cav_dir.is_dir():
                continue
            frames = {
                path.stem: parse_lidar_pose(path)
                for path in sorted(cav_dir.glob("*.yaml"))
                if _has_lidar(cav_dir, path.stem)
            }
            if frames:
                agents[cav_id] = frames
        if agents:
            poses[scenario] = agents
    if not poses:
        raise ValueError(f"no usable OPV2V scenarios found under {root_dir}")
    return poses


def build_pairs(
    poses: dict[str, dict[str, dict[str, list[float]]]],
    comm_range_m: float = DEFAULT_COMM_RANGE_M,
) -> list[AgentPair]:
    """Enumerate every ordered (ego, cav) pair inside the communication range.

    Ordered rather than unordered: the localization problem is not symmetric
    (the correction is expressed in the ego frame), and using each agent as ego
    in turn is what the paper's "two-agent samples" reformulation amounts to.
    """
    if comm_range_m <= 0:
        raise ValueError(f"comm_range_m must be positive, got {comm_range_m}")

    pairs: list[AgentPair] = []
    for scenario, agents in poses.items():
        cav_ids = sorted(agents)
        if len(cav_ids) < 2:
            continue
        shared_timestamps = sorted(set.intersection(*(set(agents[c]) for c in cav_ids)))
        for timestamp in shared_timestamps:
            for ego_id in cav_ids:
                ego_pose = agents[ego_id][timestamp]
                for cav_id in cav_ids:
                    if cav_id == ego_id:
                        continue
                    cav_pose = agents[cav_id][timestamp]
                    distance = float(
                        np.hypot(cav_pose[0] - ego_pose[0], cav_pose[1] - ego_pose[1])
                    )
                    if distance > comm_range_m:
                        continue
                    pairs.append(
                        AgentPair(
                            scenario=scenario,
                            timestamp=timestamp,
                            ego_id=ego_id,
                            cav_id=cav_id,
                            ego_pose=tuple(ego_pose),
                            cav_pose=tuple(cav_pose),
                        )
                    )
    if not pairs:
        raise ValueError(
            f"no agent pairs within {comm_range_m} m; check the split or widen comm_range_m"
        )
    return pairs


def opencood_ego_id(cav_ids: Sequence[str]) -> str:
    """Which agent OpenCOOD's ``BaseDataset`` makes the ego of a scenario.

    Reproduces ``basedataset.py``'s own rule, because a measurement that wants
    the population the fused-AP sweep actually evaluates has to pick the same
    ego it does: the agents are sorted as **strings**, a roadside unit (a
    negative id) is rotated off the front so it can never be the ego, and the
    first of what remains is it.

    The pair index itself does not use this -- it enumerates every ordered
    pair, which is the right training population. It is the *diagnostics* that
    need to narrow to OpenCOOD's ego before comparing themselves against the
    live evaluation path.
    """
    ordered = sorted(cav_ids)
    if not ordered:
        raise ValueError("a scenario needs at least one agent to have an ego")
    if int(ordered[0]) < 0:
        ordered = ordered[1:] + [ordered[0]]
    return ordered[0]


def pair_cache_path(
    cache_dir: str | Path, split_name: str, comm_range_m: float
) -> Path:
    """Where the pair index for one (split, communication range) is cached.

    The range is part of the FILENAME, not only of the signature inside the
    file. A signature can tell a reader that the cached index was built for a
    different range; it cannot hold both. Two consumers that disagree about the
    range -- AlignFormer at 70 m, matching OpenCOOD's ``COM_RANGE``, and the
    CoLoca-QuA baseline at the 40 m its paper specifies -- would otherwise
    evict each other's index on every alternating run and pay a full split
    rescan each time.
    """
    return Path(cache_dir) / f"{split_name}_pairs_{comm_range_m:g}.npz"


def load_or_build_pairs(
    root_dir: str | Path,
    cache_path: str | Path,
    comm_range_m: float = DEFAULT_COMM_RANGE_M,
) -> list[AgentPair]:
    """Return the pair index, rebuilding and caching it when the cache is stale."""
    root_dir = Path(root_dir)
    cache_path = Path(cache_path)
    signature = {
        "version": CACHE_VERSION,
        "root_dir": str(root_dir.resolve()),
        "comm_range_m": comm_range_m,
    }

    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if json.loads(str(cached["signature"])) == signature:
            return _pairs_from_arrays(cached)

    pairs = build_pairs(scan_split(root_dir), comm_range_m)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        signature=json.dumps(signature),
        keys=np.array(
            [[p.scenario, p.timestamp, p.ego_id, p.cav_id] for p in pairs], dtype=object
        ).astype("U"),
        ego_poses=np.array([p.ego_pose for p in pairs], dtype=np.float64),
        cav_poses=np.array([p.cav_pose for p in pairs], dtype=np.float64),
    )
    return pairs


def _pairs_from_arrays(cached) -> list[AgentPair]:
    keys = cached["keys"]
    ego_poses = cached["ego_poses"]
    cav_poses = cached["cav_poses"]
    return [
        AgentPair(
            scenario=str(keys[i, 0]),
            timestamp=str(keys[i, 1]),
            ego_id=str(keys[i, 2]),
            cav_id=str(keys[i, 3]),
            ego_pose=tuple(ego_poses[i].tolist()),
            cav_pose=tuple(cav_poses[i].tolist()),
        )
        for i in range(keys.shape[0])
    ]
