"""Per-frame detection and ROI-feature cache.

Running the detector inside the training loop would make it the bottleneck, the
same failure the point-cloud cache fixed for CoLoca-QuA (0.62 -> 11.2 it/s).
Detections and their ROI features are computed once and stored on the NVMe.

ROI features are cached rather than finished embeddings so that the embedding
head remains trainable downstream.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

# Sentinel for "this detection matched no ground-truth object". npz has no null
# for a string array, and writing "None" would fabricate a shared identity that
# the matching loss would then train towards.
_NO_GT_ID = ""

# ROI grid size used when none is given on the command line. Matches the
# default used elsewhere in AlignFormer's per-object embedding pipeline.
DEFAULT_OUTPUT_SIZE = 4

# How many cached/source frame pairs the pcd-cache preflight check compares
# before trusting the mirror for a whole split. Cheap relative to the
# multi-hour build, so this stays generous.
_PCD_CACHE_PREFLIGHT_SAMPLES = 5

# Print a progress line after this many frames within a split.
_PROGRESS_INTERVAL = 500


@dataclass(frozen=True)
class FrameRecord:
    """One agent's cached detections at one timestamp."""

    boxes: np.ndarray
    scores: np.ndarray
    gt_ids: Sequence[Optional[str]]
    roi: np.ndarray

    def __post_init__(self) -> None:
        count = self.boxes.shape[0]
        lengths = {
            "scores": self.scores.shape[0],
            "gt_ids": len(self.gt_ids),
            "roi": self.roi.shape[0],
        }
        mismatched = {name: n for name, n in lengths.items() if n != count}
        if mismatched:
            raise ValueError(f"inconsistent length against {count} boxes: {mismatched}")


def cache_path(
    cache_root: Union[Path, str],
    split: str,
    scenario: str,
    cav_id: str,
    timestamp: str,
) -> Path:
    """Return the npz path for one agent-frame."""
    return Path(cache_root) / split / scenario / cav_id / f"{timestamp}.npz"


def write_frame(path: Union[Path, str], record: FrameRecord) -> None:
    """Persist one frame, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gt_ids = np.array(
        [_NO_GT_ID if value is None else str(value) for value in record.gt_ids],
        dtype=np.str_,
    )
    np.savez(
        path,
        boxes=record.boxes.astype(np.float32),
        scores=record.scores.astype(np.float32),
        gt_ids=gt_ids,
        roi=record.roi.astype(np.float16),
    )


def read_frame(path: Union[Path, str]) -> FrameRecord:
    """Load one cached frame."""
    with np.load(path, allow_pickle=False) as data:
        return FrameRecord(
            boxes=data["boxes"],
            scores=data["scores"],
            gt_ids=[
                None if value == _NO_GT_ID else str(value) for value in data["gt_ids"]
            ],
            roi=data["roi"],
        )


class PcdCacheStats:
    """Hit/miss counters for the binary point-cloud mirror.

    Single-process (the detector forward pass, not point-cloud decode, is the
    bottleneck here, so this build never forks workers) -- unlike
    ``scripts/train_late_fusion.py``'s ``ShimStats``, plain ints are enough;
    there is no ``DataLoader`` worker pool whose counts need to survive a fork.
    """

    def __init__(self) -> None:
        self.hits = 0
        self.misses = 0

    def record(self, hit: bool) -> None:
        if hit:
            self.hits += 1
        else:
            self.misses += 1

    def snapshot(self) -> Tuple[int, int]:
        return self.hits, self.misses


def verify_pcd_cache_equivalence(
    cache_root: Path, split_root: Path, sample_size: int = _PCD_CACHE_PREFLIGHT_SAMPLES
) -> int:
    """Assert a sample of cached point clouds exactly match the real decoder.

    Run once per split before trusting the mirror for a multi-hour build --
    see ``scripts/train_late_fusion.py``'s ``_check_shim_equivalence`` for the
    same check in the training shim this reuses the approach of.
    """
    from opencood.utils.pcd_utils import pcd_to_np

    from embedding_aware_belt_fusion.coloca.pcd_cache import cached_pcd_path

    checked = 0
    for cached_file in cache_root.rglob("*.npy"):
        relative = cached_file.relative_to(cache_root).with_suffix(".pcd")
        source = split_root / relative
        if not source.exists():
            continue
        cached_array = np.load(cached_file)
        real_array = pcd_to_np(str(source))
        if not np.array_equal(cached_array, real_array):
            raise AssertionError(
                f"pcd cache mismatch on {source}: cached shape "
                f"{cached_array.shape} vs real shape {real_array.shape}"
            )
        checked += 1
        if checked >= sample_size:
            break
    if checked == 0:
        raise RuntimeError(
            f"could not find any overlapping cache/source pcd pair under {cache_root} "
            "to sanity-check"
        )
    return checked


def _load_agent_points(
    pcd_path: Path, pcd_cache_root: Path, split_root: Path, stats: PcdCacheStats
) -> np.ndarray:
    """Load one agent-frame's point cloud, preferring the binary mirror."""
    from embedding_aware_belt_fusion.coloca.pcd_cache import cached_pcd_path, load_points

    cached = cached_pcd_path(pcd_cache_root, pcd_path, split_root)
    stats.record(hit=cached.exists())
    return load_points(pcd_path, pcd_cache_root, split_root)


def _build_detector(hypes: Dict, device) -> "torch.nn.Module":  # noqa: F821
    """Load the trained per-agent detector named in ``hypes['detector']``."""
    import torch

    from opencood.tools.train_utils import create_model

    detector = create_model(hypes)
    checkpoint_path = hypes["detector"]["checkpoint"]
    state = torch.load(checkpoint_path, map_location="cpu")
    state = state.get("model_state_dict", state) if isinstance(state, dict) else state
    detector.load_state_dict(state)
    return detector.to(device).eval()


def _cav_content_for_frame(
    dataset, params: Dict, lidar_np: np.ndarray, device
) -> Dict[str, object]:
    """Assemble one agent-frame's ``cav_content`` for ``boxes.detect_agent``.

    Reuses ``LateFusionDataset.get_item_single_car`` -- the exact per-agent
    voxelization/anchor/ground-truth pipeline the detector was trained and
    evaluated with -- rather than re-deriving it, then converts its numpy
    output into the torch layout ``detect_agent`` expects (normally supplied
    by ``collate_batch_test``, which this bypasses to keep one agent-frame at
    a time addressable by scenario/cav/timestamp).
    """
    import numpy as np
    import torch

    raw = dataset.get_item_single_car({"lidar_np": lidar_np, "params": params})
    processed_lidar = dataset.pre_processor.collate_batch([raw["processed_lidar"]])
    processed_lidar = {key: value.to(device) for key, value in processed_lidar.items()}
    return {
        "processed_lidar": processed_lidar,
        "anchor_box": torch.from_numpy(np.array(raw["anchor_box"])).float(),
        "object_bbx_center": torch.from_numpy(raw["object_bbx_center"]).float(),
        "object_bbx_mask": torch.from_numpy(raw["object_bbx_mask"]),
        "object_ids": raw["object_ids"],
    }


def _iter_agent_frames(poses: Dict[str, Dict[str, Dict[str, list]]]):
    """Yield ``(scenario, cav_id, timestamp)`` for every agent-frame in a split."""
    for scenario, agents in poses.items():
        for cav_id, frames in agents.items():
            for timestamp in frames:
                yield scenario, cav_id, timestamp


def _cache_one_frame(
    *,
    dataset,
    detector,
    postprocessor,
    lidar_range: Sequence[float],
    split_root: Path,
    pcd_cache_root: Path,
    scenario: str,
    cav_id: str,
    timestamp: str,
    destination: Path,
    output_size: int,
    device,
    stats: PcdCacheStats,
) -> None:
    """Detect, ROI-align, and write the cache record for one agent-frame."""
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from torch import no_grad

    from embedding_aware_belt_fusion.alignformer.boxes import detect_agent
    from embedding_aware_belt_fusion.alignformer.embedding import rotated_roi_align

    cav_dir = split_root / scenario / cav_id
    params = load_yaml(str(cav_dir / f"{timestamp}.yaml"), None)
    lidar_np = _load_agent_points(cav_dir / f"{timestamp}.pcd", pcd_cache_root, split_root, stats)
    cav_content = _cav_content_for_frame(dataset, params, lidar_np, device)

    with no_grad():
        detections = detect_agent(detector, cav_content, postprocessor)
        roi = rotated_roi_align(detections.features, detections.boxes, lidar_range, output_size)

    write_frame(
        destination,
        FrameRecord(
            boxes=detections.boxes.detach().cpu().numpy(),
            scores=detections.scores.detach().cpu().numpy(),
            gt_ids=list(detections.gt_ids),
            roi=roi.detach().cpu().numpy().astype(np.float16),
        ),
    )


def _log_progress(
    split_name: str, done: int, total: int, started: float, stats: PcdCacheStats
) -> None:
    elapsed = time.time() - started
    remaining = (total - done) / (done / elapsed) / 60
    hits, misses = stats.snapshot()
    hit_total = hits + misses
    rate = hits / hit_total if hit_total else 0.0
    print(
        f"  {split_name}: {done}/{total}  {done / elapsed:.1f} frames/s  "
        f"~{remaining:.1f} min left  pcd cache {rate:.1%} ({hits}/{hit_total})",
        flush=True,
    )


def build_split_cache(
    *,
    dataset,
    detector,
    postprocessor,
    lidar_range: Sequence[float],
    split_root: Path,
    split_name: str,
    cache_root: Path,
    pcd_cache_root: Path,
    output_size: int,
    device,
) -> PcdCacheStats:
    """Detect and cache every agent-frame under one OPV2V split directory."""
    from embedding_aware_belt_fusion.coloca.index import scan_split

    checked = verify_pcd_cache_equivalence(pcd_cache_root, split_root)
    print(f"{split_name}: pcd cache equivalence check passed on {checked} frames", flush=True)

    frames = list(_iter_agent_frames(scan_split(split_root)))
    print(f"{split_name}: {len(frames)} agent-frames to cache", flush=True)

    stats = PcdCacheStats()
    started, done = time.time(), 0
    for scenario, cav_id, timestamp in frames:
        destination = cache_path(cache_root, split_name, scenario, cav_id, timestamp)
        if not destination.exists():
            _cache_one_frame(
                dataset=dataset,
                detector=detector,
                postprocessor=postprocessor,
                lidar_range=lidar_range,
                split_root=split_root,
                pcd_cache_root=pcd_cache_root,
                scenario=scenario,
                cav_id=cav_id,
                timestamp=timestamp,
                destination=destination,
                output_size=output_size,
                device=device,
                stats=stats,
            )

        done += 1
        if done % _PROGRESS_INTERVAL == 0:
            _log_progress(split_name, done, len(frames), started, stats)

    print(f"{split_name}: done in {(time.time() - started) / 60:.1f} min", flush=True)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache per-agent detections and ROI features")
    parser.add_argument("--config", type=Path, required=True, help="OpenCOOD-format detector hypes yaml")
    parser.add_argument("--splits", nargs="+", required=True, help="OPV2V split directories")
    parser.add_argument("--cache-root", type=Path, required=True, help="destination on fast storage")
    parser.add_argument(
        "--pcd-cache-root",
        type=Path,
        default=Path("/media/chenyi/basement2/cache/opv2v_coloca"),
        help="root of the binary point-cloud mirror, one subdirectory per split name",
    )
    parser.add_argument("--output-size", type=int, default=DEFAULT_OUTPUT_SIZE)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main() -> None:
    import torch
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml.yaml_utils import load_yaml

    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)

    hypes = load_yaml(str(args.config), None)
    lidar_range = hypes["preprocess"]["cav_lidar_range"]

    # Only the pre/post-processors and get_item_single_car are used below --
    # this dataset's own root_dir/validate_dir indexing is never touched, so
    # one instance covers every split passed on the command line.
    dataset = build_dataset(hypes, visualize=False, train=False)
    detector = _build_detector(hypes, device)

    for split in args.splits:
        split_root = Path(split)
        split_name = split_root.name
        stats = build_split_cache(
            dataset=dataset,
            detector=detector,
            postprocessor=dataset.post_processor,
            lidar_range=lidar_range,
            split_root=split_root,
            split_name=split_name,
            cache_root=args.cache_root,
            pcd_cache_root=args.pcd_cache_root / split_name,
            output_size=args.output_size,
            device=device,
        )
        hits, misses = stats.snapshot()
        total = hits + misses
        rate = hits / total if total else 0.0
        print(f"{split_name}: pcd cache {hits}/{total} hits ({rate:.1%})", flush=True)

    total_bytes = sum(f.stat().st_size for f in Path(args.cache_root).rglob("*.npz"))
    print(f"cache size: {total_bytes / 1e9:.2f} GB at {args.cache_root}")


if __name__ == "__main__":
    main()
