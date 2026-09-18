"""Binary point-cloud cache for OPV2V.

OPV2V ships ~3 MB **ASCII** ``.pcd`` files.  On this machine the dataset lives on
a USB spinning disk, and decoding two of them per sample caps the CoLoca-QuA
data loader at roughly 10 samples/s against a GPU that can absorb 223.

This module mirrors a split into flat ``.npy`` arrays of ``(N, 4)`` float32
``[x, y, z, intensity]`` on fast local storage, which is byte-for-byte what
``opencood.utils.pcd_utils.pcd_to_np`` returns.  The cache is ~3.4x smaller than
the ASCII source and needs no parsing at read time.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

CACHE_SUFFIX = ".npy"


def cached_pcd_path(cache_root: Path, source_pcd: Path, split_root: Path) -> Path:
    """Map a source ``.pcd`` path to its cache location, preserving the layout."""
    return (cache_root / source_pcd.relative_to(split_root)).with_suffix(CACHE_SUFFIX)


def load_points(source_pcd: Path, cache_root: Path | None, split_root: Path | None) -> np.ndarray:
    """Load a point cloud, preferring the binary cache and falling back to the pcd."""
    if cache_root is not None and split_root is not None:
        cached = cached_pcd_path(cache_root, source_pcd, split_root)
        if cached.exists():
            return np.load(cached)

    from opencood.utils.pcd_utils import pcd_to_np

    return pcd_to_np(str(source_pcd))


def convert_one(task: tuple[str, str]) -> int:
    """Convert a single pcd to npy.  Returns the number of points written."""
    source, destination = Path(task[0]), Path(task[1])
    if destination.exists():
        return 0

    from opencood.utils.pcd_utils import pcd_to_np

    points = pcd_to_np(str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Write then rename so an interrupted run never leaves a truncated file that
    # a later run would happily treat as valid cache.
    # np.save appends ".npy" unless the path already ends in it, so the temp
    # name keeps the suffix and carries the pid before it.
    temporary = destination.with_suffix(f".tmp{os.getpid()}{CACHE_SUFFIX}")
    with open(temporary, "wb") as handle:
        np.save(handle, points)
    temporary.replace(destination)
    return points.shape[0]


def build_cache(split_root: Path, cache_root: Path, workers: int) -> None:
    """Convert every pcd under ``split_root`` into ``cache_root``."""
    if not split_root.is_dir():
        raise FileNotFoundError(f"split directory not found: {split_root}")

    sources = sorted(split_root.rglob("*.pcd"))
    if not sources:
        raise ValueError(f"no .pcd files found under {split_root}")

    tasks = [
        (str(source), str(cached_pcd_path(cache_root, source, split_root)))
        for source in sources
    ]
    pending = [task for task in tasks if not Path(task[1]).exists()]
    print(f"{split_root.name}: {len(sources)} clouds, {len(pending)} to convert", flush=True)
    if not pending:
        return

    started, done, points = time.time(), 0, 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for written in pool.map(convert_one, pending, chunksize=16):
            done += 1
            points += written
            if done % 500 == 0:
                elapsed = time.time() - started
                remaining = (len(pending) - done) / (done / elapsed) / 60
                print(
                    f"  {done}/{len(pending)}  {done / elapsed:.1f} clouds/s  "
                    f"~{remaining:.1f} min left",
                    flush=True,
                )
    elapsed = time.time() - started
    print(
        f"{split_root.name}: done in {elapsed / 60:.1f} min "
        f"({done / max(elapsed, 1e-9):.1f} clouds/s, {points / 1e6:.1f}M points)",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache OPV2V pcd files as binary npy")
    parser.add_argument("--splits", nargs="+", required=True, help="OPV2V split directories")
    parser.add_argument("--cache-root", required=True, help="destination on fast storage")
    parser.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 8))
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    for split in args.splits:
        split_root = Path(split)
        build_cache(split_root, cache_root / split_root.name, args.workers)

    total = sum(f.stat().st_size for f in cache_root.rglob(f"*{CACHE_SUFFIX}"))
    print(f"cache size: {total / 1e9:.1f} GB at {cache_root}")


if __name__ == "__main__":
    sys.exit(main())
