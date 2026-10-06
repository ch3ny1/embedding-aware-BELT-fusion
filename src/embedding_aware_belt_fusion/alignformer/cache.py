"""Per-frame detection and ROI-feature cache.

Running the detector inside the training loop would make it the bottleneck, the
same failure the point-cloud cache fixed for CoLoca-QuA (0.62 -> 11.2 it/s).
Detections and their ROI features are computed once and stored on the NVMe.

ROI features are cached rather than finished embeddings so that the embedding
head remains trainable downstream.
"""

from __future__ import annotations

import argparse
import hashlib
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
    # Per-object camera features (alignformer.camera_features), present only
    # in caches built with --camera. OPV2V caches predate them and read back
    # as None; the pair dataset treats None as "this trunk has no camera".
    camera: Optional[np.ndarray] = None
    has_camera: Optional[np.ndarray] = None
    camera_index: Optional[np.ndarray] = None
    # The per-frame (unpooled) camera vector, kept beside the track-pooled
    # ``camera`` by caches built with a DINO-head backbone and a track window
    # (alignformer.dino_features), so evaluation can pool a live frame with
    # its cached predecessors exactly as training saw them.
    camera_raw: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        count = self.boxes.shape[0]
        lengths = {
            "scores": self.scores.shape[0],
            "gt_ids": len(self.gt_ids),
            "roi": self.roi.shape[0],
        }
        if self.camera_raw is not None:
            lengths["camera_raw"] = self.camera_raw.shape[0]
        present = [f is not None for f in (self.camera, self.has_camera, self.camera_index)]
        if any(present) and not all(present):
            raise ValueError("camera, has_camera and camera_index must be given together")
        if all(present):
            lengths["camera"] = self.camera.shape[0]
            lengths["has_camera"] = self.has_camera.shape[0]
            lengths["camera_index"] = self.camera_index.shape[0]
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
    """Persist one frame, creating parent directories as needed.

    Raises if a real (non-``None``) ``gt_id`` collides with the ``""``
    sentinel used for "matched nothing" -- that would otherwise decode back
    as ``None`` and silently fabricate a shared identity across agents that
    the matching loss would then train towards. Today this can only happen
    if an upstream id is itself the empty string (OPV2V's own ids are
    integers stringified by ``opencood_proposals.assign_proposals_to_ground_truth``,
    so it can't), but checking it here means that invariant is enforced, not
    just remembered.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for value in record.gt_ids:
        if value == _NO_GT_ID:
            raise ValueError(
                f"gt_id is the empty string, which collides with the "
                f"'no match' sentinel and would silently decode back as None: {record.gt_ids!r}"
            )
    gt_ids = np.array(
        [_NO_GT_ID if value is None else str(value) for value in record.gt_ids],
        dtype=np.str_,
    )
    arrays = dict(
        boxes=record.boxes.astype(np.float32),
        scores=record.scores.astype(np.float32),
        gt_ids=gt_ids,
        roi=record.roi.astype(np.float16),
    )
    if record.camera is not None:
        arrays.update(
            camera=record.camera.astype(np.float16),
            has_camera=record.has_camera.astype(bool),
            camera_index=record.camera_index.astype(np.int8),
        )
    if record.camera_raw is not None:
        arrays["camera_raw"] = record.camera_raw.astype(np.float16)
    np.savez(path, **arrays)


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
            camera=data["camera"] if "camera" in data else None,
            has_camera=data["has_camera"] if "has_camera" in data else None,
            camera_index=data["camera_index"] if "camera_index" in data else None,
            camera_raw=data["camera_raw"] if "camera_raw" in data else None,
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


# WHY this seeding exists: LateFusionDataset.get_item_single_car calls
# shuffle_points (external/OpenCOOD/opencood/utils/pcd_utils.py), which is a
# bare np.random.permutation with no seed of its own. Left unseeded, two runs
# of the same frame draw a different point order and, via
# max_points_per_voxel truncation, a slightly different detection -- both
# this cache and evaluate.py's live re-derivation of the same frames are
# source-of-truth artifacts, so that build-to-build (or cache-vs-live) drift
# is not acceptable.
#
# WHY hashlib.sha256 rather than builtin hash(): Python randomizes string
# hashing per process unless PYTHONHASHSEED is set at interpreter start,
# which would silently defeat the entire fix -- hashlib is stable across
# processes and interpreter versions with no environment dependency.
#
# This used to be duplicated byte-for-byte in cache.py and evaluate.py while
# both were being written concurrently (R31); it now lives here, the
# lower-level module, and evaluate.py imports it from here.
def frame_seed(scenario: str, cav_id: str, timestamp: str) -> int:
    """Stable, frame-derived seed for OpenCOOD's unseeded point shuffle.

    Deriving the seed from ``(scenario, cav_id, timestamp)`` (rather than one
    fixed constant for the whole run) keeps each frame's shuffle independent
    of every other frame's and of processing order, while making a given
    frame reproducible byte-for-byte on any rebuild.
    """
    digest = hashlib.sha256(f"{scenario}/{cav_id}/{timestamp}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big")


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
    dataset,
    params: Dict,
    lidar_np: np.ndarray,
    device,
    *,
    scenario: str,
    cav_id: str,
    timestamp: str,
) -> Dict[str, object]:
    """Assemble one agent-frame's ``cav_content`` for ``boxes.detect_agent``.

    Reuses ``LateFusionDataset.get_item_single_car`` -- the exact per-agent
    voxelization/anchor/ground-truth pipeline the detector was trained and
    evaluated with -- rather than re-deriving it, then converts its numpy
    output into the torch layout ``detect_agent`` expects (normally supplied
    by ``collate_batch_test``, which this bypasses to keep one agent-frame at
    a time addressable by scenario/cav/timestamp).

    Seeds numpy's global RNG with a frame-derived seed immediately before the
    call, since that is what ``get_item_single_car`` -> ``shuffle_points``
    consumes (see ``frame_seed``) -- this is what makes the cache
    byte-reproducible across rebuilds.
    """
    import numpy as np
    import torch

    np.random.seed(frame_seed(scenario, cav_id, timestamp))
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
    camera_backbone=None,
    source_path: Optional[Path] = None,
) -> None:
    """Detect, ROI-align, and write the cache record for one agent-frame.

    A dataset that exposes ``frame_params`` / ``frame_points`` (the V2X-Real
    adapter) supplies its own yaml filtering and ``.bin`` reading; OpenCOOD's
    stock dataset gets the OPV2V path, unchanged. With ``camera_backbone``
    set, the dataset's ``frame_cameras`` supplies the images and the record
    carries per-object camera features. With ``source_path`` set, the
    detections, ROI features and ids are taken from that existing record
    and only the camera features are computed: the detector is deterministic,
    so a second camera backbone need not pay for it again.
    """
    from torch import no_grad

    from embedding_aware_belt_fusion.alignformer.boxes import detect_agent
    from embedding_aware_belt_fusion.alignformer.embedding import rotated_roi_align

    cav_dir = split_root / scenario / cav_id
    yaml_path = cav_dir / f"{timestamp}.yaml"
    if source_path is not None:
        record = _record_without_camera(read_frame(source_path))
        corners = _corners_from_boxes(record.boxes)
        write_frame(destination, _with_camera(record, camera_backbone, dataset, yaml_path, corners))
        return
    params = _frame_params(dataset, yaml_path)
    lidar_np = _frame_points(dataset, cav_dir / f"{timestamp}.pcd", pcd_cache_root, split_root, stats)
    cav_content = _cav_content_for_frame(
        dataset, params, lidar_np, device, scenario=scenario, cav_id=cav_id, timestamp=timestamp
    )

    with no_grad():
        detections = detect_agent(detector, cav_content, postprocessor)
        roi = rotated_roi_align(detections.features, detections.boxes, lidar_range, output_size)

    record = FrameRecord(
        boxes=detections.boxes.detach().cpu().numpy(),
        scores=detections.scores.detach().cpu().numpy(),
        gt_ids=list(detections.gt_ids),
        roi=roi.detach().cpu().numpy().astype(np.float16),
    )
    if camera_backbone is not None:
        record = _with_camera(record, camera_backbone, dataset, yaml_path, detections.corners.detach().cpu().numpy())
    write_frame(destination, record)


def _record_without_camera(record: FrameRecord) -> FrameRecord:
    return FrameRecord(boxes=record.boxes, scores=record.scores, gt_ids=list(record.gt_ids), roi=record.roi)


def _corners_from_boxes(boxes: np.ndarray) -> np.ndarray:
    """``(N, 8, 3)`` corners of cached ``(N, 7)`` boxes in the detector's ``hwl`` order."""
    from opencood.utils.box_utils import boxes_to_corners_3d

    if boxes.shape[0] == 0:
        return np.zeros((0, 8, 3), dtype=np.float64)
    return np.asarray(boxes_to_corners_3d(np.asarray(boxes, dtype=np.float32), order="hwl"), dtype=np.float64)


def _with_camera(record: FrameRecord, backbone, dataset, yaml_path: Path, corners: np.ndarray) -> FrameRecord:
    """Camera features for the record's boxes; the per-frame vector is kept in ``camera_raw`` too."""
    from dataclasses import replace

    from embedding_aware_belt_fusion.alignformer.dino_features import features_for_frame

    cameras = dataset.frame_cameras(yaml_path)
    features = features_for_frame(backbone, corners, cameras)
    return replace(
        record,
        camera=features.features,
        has_camera=features.has_camera,
        camera_index=features.camera_index,
        camera_raw=features.features if hasattr(backbone, "frame_features") else None,
    )


def _frame_params(dataset, yaml_path: Path) -> Dict:
    if hasattr(dataset, "frame_params"):
        return dataset.frame_params(yaml_path)
    from opencood.hypes_yaml.yaml_utils import load_yaml

    return load_yaml(str(yaml_path), None)


def _frame_points(dataset, pcd_path: Path, pcd_cache_root: Path, split_root: Path, stats: PcdCacheStats):
    if hasattr(dataset, "frame_points"):
        return dataset.frame_points(pcd_path)
    return _load_agent_points(pcd_path, pcd_cache_root, split_root, stats)


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
    camera_backbone=None,
    source_cache_root: Optional[Path] = None,
    pooling=None,
) -> PcdCacheStats:
    """Detect and cache every agent-frame under one split directory.

    ``source_cache_root`` reuses an existing cache's detections and recomputes
    only the camera features; ``pooling`` (``dino_features.TrackPooling``)
    then pools each agent's camera vectors along its own track, in place.
    """
    from embedding_aware_belt_fusion.coloca.index import scan_split

    if hasattr(dataset, "frame_points"):
        print(f"{split_name}: dataset reads its own LiDAR; no pcd mirror involved", flush=True)
    else:
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
                camera_backbone=camera_backbone,
                source_path=None if source_cache_root is None
                else cache_path(source_cache_root, split_name, scenario, cav_id, timestamp),
            )

        done += 1
        if done % _PROGRESS_INTERVAL == 0:
            _log_progress(split_name, done, len(frames), started, stats)

    print(f"{split_name}: done in {(time.time() - started) / 60:.1f} min", flush=True)
    if pooling is not None:
        from embedding_aware_belt_fusion.alignformer.dino_features import pool_split

        pooled = pool_split(cache_root, split_name, split_root, pooling)
        print(f"{split_name}: track-pooled {pooled} records (window {pooling.window})", flush=True)
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
    parser.add_argument(
        "--camera",
        action="store_true",
        help="also store per-object camera features (alignformer.camera_features); "
        "needs a dataset that exposes frame_cameras, i.e. the V2X-Real adapter",
    )
    parser.add_argument("--camera-backbone", choices=("resnet", "dino"), default="resnet",
                        help="resnet: frozen ImageNet ROI features; dino: frozen DINOv2 through the trained appearance head")
    parser.add_argument("--camera-head", type=Path, default=None, help="the appearance head checkpoint (dino)")
    parser.add_argument("--dino", choices=("small", "base"), default="base")
    parser.add_argument("--from-cache", type=Path, default=None,
                        help="reuse this cache's detections and ROI features; only the camera features are computed")
    parser.add_argument("--track-window", type=int, default=0,
                        help="pool each agent's camera vectors over this many earlier frames of its own track (dino)")
    parser.add_argument("--track-gate-m", type=float, default=2.0)
    parser.add_argument("--track-gate-per-frame-m", type=float, default=1.5)
    return parser.parse_args()


def _camera_backbone_from_args(args, dataset, device):
    if not args.camera:
        return None
    if not hasattr(dataset, "frame_cameras"):
        raise ValueError("--camera needs a dataset that exposes frame_cameras (the V2X-Real adapter)")
    if args.camera_backbone == "resnet":
        from embedding_aware_belt_fusion.alignformer.camera_features import CameraBackbone

        return CameraBackbone().to(device)
    if args.camera_head is None:
        raise ValueError("--camera-backbone dino needs --camera-head")
    from embedding_aware_belt_fusion.alignformer.dino_features import DINO_HEAD_BACKBONE, build_dino_backbone

    source = {"backbone": DINO_HEAD_BACKBONE, "head_checkpoint": str(args.camera_head), "dino": args.dino,
              "track_window": args.track_window, "track_gate_m": args.track_gate_m,
              "track_gate_per_frame_m": args.track_gate_per_frame_m}
    return build_dino_backbone(source, device, cache_root=args.cache_root)


def main() -> None:
    import torch
    from opencood.hypes_yaml.yaml_utils import load_yaml

    from embedding_aware_belt_fusion.alignformer.v2xreal import build_dataset

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
    detector = None if args.from_cache is not None else _build_detector(hypes, device)
    camera_backbone = _camera_backbone_from_args(args, dataset, device)
    pooling = getattr(camera_backbone, "pooling", None)

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
            camera_backbone=camera_backbone,
            source_cache_root=args.from_cache,
            pooling=pooling,
        )
        hits, misses = stats.snapshot()
        total = hits + misses
        rate = hits / total if total else 0.0
        print(f"{split_name}: pcd cache {hits}/{total} hits ({rate:.1%})", flush=True)

    total_bytes = sum(f.stat().st_size for f in Path(args.cache_root).rglob("*.npz"))
    print(f"cache size: {total_bytes / 1e9:.2f} GB at {args.cache_root}")


if __name__ == "__main__":
    main()
