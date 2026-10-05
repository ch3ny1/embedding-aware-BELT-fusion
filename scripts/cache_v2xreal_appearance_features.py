"""Cache frozen DINOv2 crop descriptors of every usable vehicle view, per
agent-frame, for the cross-agent appearance head.

The views are exactly the separability probe's (same projection, silhouette,
occlusion and pixel gates, best camera per vehicle), so a head trained on
them is evaluated on the probe's own candidates. One ``.npz`` per
agent-frame (``alignformer.appearance_head.AgentFrame``): the concatenated
descriptors, the vehicles' world centres and LiDAR ranges, and every
annotated vehicle id for the shared-object bucket. Resumable; ``--every``
strides timestamps (10 Hz frames are near-duplicates for this purpose).

Usage::

    python -u scripts/cache_v2xreal_appearance_features.py \\
        --root /media/chenyi/basement2/dataset/v2x-real/train --every 2 --model base \\
        --output-dir /media/chenyi/basement2/cache/alignformer_v2xreal_appearance/train
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_v2xreal_colour_separability import (  # noqa: E402
    Describer,
    _lidar_range,
    _load_views,
    _timestamps,
    _vehicles,
    foundation_describer,
)
from embedding_aware_belt_fusion.alignformer.appearance_head import AgentFrame, frame_key, save_frame  # noqa: E402
from embedding_aware_belt_fusion.alignformer.camera import vehicle_world_corners  # noqa: E402

PROGRESS_EVERY = 200


def frame_from_views(scenario: str, agent: str, timestamp: str, params: Dict, views: Dict[str, Dict],
                     names: Sequence[str]) -> AgentFrame:
    """Pure conversion of one agent-frame's probe views into the cached record."""
    vehicles = _vehicles(params)
    vids = tuple(sorted(v for v in views if v in vehicles))
    dim = sum(int(np.asarray(views[vids[0]][n]).size) for n in names) if vids else 0
    features = np.zeros((len(vids), dim), dtype=np.float32)
    for row, vid in enumerate(vids):
        features[row] = np.concatenate([np.asarray(views[vid][n], dtype=np.float32).ravel() for n in names])
    centres = np.asarray([vehicle_world_corners(vehicles[v]).mean(axis=0)[:2] for v in vids], dtype=np.float64)
    ranges = np.asarray([_lidar_range(vehicles[v], params["lidar_pose"]) for v in vids], dtype=np.float64)
    return AgentFrame(scenario, agent, timestamp, vids, features, centres.reshape(len(vids), 2), ranges,
                      tuple(sorted(vehicles)))


def cache_split(root: Path, output_dir: Path, describe: Describer, every: int) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    written, started = 0, time.time()
    for scenario in sorted(p for p in root.iterdir() if p.is_dir()):
        for agent_dir in sorted(p for p in scenario.iterdir() if p.is_dir()):
            for timestamp in _timestamps(agent_dir)[::every]:
                if (output_dir / f"{frame_key(scenario.name, agent_dir.name, timestamp)}.npz").exists():
                    continue
                params, views = _load_views(root, agent_dir, timestamp, describe)
                save_frame(output_dir, frame_from_views(scenario.name, agent_dir.name, timestamp, params, views, describe.names))
                written += 1
                if written % PROGRESS_EVERY == 0:
                    print(f"  {written} agent-frames, {(time.time() - started) / written:.2f} s each", flush=True)
    return written


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("small", "base"), default="base")
    parser.add_argument("--every", type=int, default=1)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.root.name == "test":
        raise ValueError("the appearance head is trained on train and judged on val; test is not read here")
    written = cache_split(args.root, args.output_dir, foundation_describer(args.model), args.every)
    print(f"wrote {written} agent-frames to {args.output_dir}")


if __name__ == "__main__":
    main()
