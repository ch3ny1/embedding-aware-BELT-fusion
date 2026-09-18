"""Train OpenCOOD's PointPillars late-fusion detector, fast, on this machine.

No late-fusion checkpoint survived on this machine and OpenCOOD's published
one (Google Drive and UCLA Box) is unreachable (see
``.superpowers/sdd/2026-09-17-alignformer-p0-p2/task-2-report.md``), so this
trains ``external/OpenCOOD/opencood/hypes_yaml/point_pillar_late_fusion.yaml``
from scratch instead.

OPV2V lives on a USB spinning disk here as ~3 MB **ASCII** ``.pcd`` files.
Decoding them caps OpenCOOD's stock data loader near 10 samples/s against a
GPU that absorbs 223 -- that is where the framework's "about a day" training
estimate comes from. A byte-identical binary ``.npy`` mirror of the whole
dataset already exists on fast local storage
(``embedding_aware_belt_fusion.coloca.pcd_cache``, built for the CoLoca-QuA
baseline). This script monkey-patches ``opencood.utils.pcd_utils.pcd_to_np``
at runtime to read that cache when a match exists, falling back to the real
pcd decoder otherwise -- **no file under external/OpenCOOD is modified**, the
patch lives entirely here.

Two things this script checks before trusting the shim:
  - Equivalence: the cached array and ``pcd_to_np`` must agree byte-for-byte
    on a sample of real frames (see ``_check_shim_equivalence``), checked
    once at startup before any training happens.
  - Hit rate: a counter tracks cache hits/misses across the whole run,
    logged when training ends (or the process is interrupted), so a
    silently-idle shim (e.g. from a bad path mapping) shows up as a hit rate
    near zero rather than as an unexplained lack of speedup. The it/s
    reported by OpenCOOD's own tqdm progress bar is the faster live signal
    while training is running.

OPV2V ships no usable ``validate`` split on this machine (broken symlink), so
validation is a 15% scenario-level holdout of ``train``, split the same way
(by scenario, seed 0) as the CoLoca-QuA baseline
(``embedding_aware_belt_fusion.coloca.train.split_scenarios``). Because
OpenCOOD's dataset just ``os.listdir()`` its ``root_dir``, the holdout is
materialized as two directories of symlinks into the real scenario folders,
built once and reused.

Usage
-----
    python -u scripts/train_late_fusion.py --hypes_yaml configs/alignformer_detector.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from embedding_aware_belt_fusion.coloca.pcd_cache import cached_pcd_path
from embedding_aware_belt_fusion.coloca.train import split_scenarios

# This machine's real OPV2V train tree and its binary pcd cache (see
# ``pcd_cache.py`` and the ``opv2v-pcd-cache-on-basement`` project memory).
OPV2V_TRAIN_ROOT = Path("/media/chenyi/Elements1/Dataset/OPV2V/train")
PCD_CACHE_ROOT = Path("/media/chenyi/basement2/cache/opv2v_coloca/train")

# Scenario-level 15% holdout, materialized as symlink directories so
# OpenCOOD's stock ``BaseDataset`` (which just lists ``root_dir``) needs no
# changes.
SPLIT_ROOT = Path("outputs/alignformer/opv2v_splits")
TRAIN_SPLIT_DIR = SPLIT_ROOT / "train"
VAL_SPLIT_DIR = SPLIT_ROOT / "val"
VAL_FRACTION = 0.15
SPLIT_SEED = 0

# Fixed (non-timestamped) output directory so the checkpoint path can be
# recorded in configs/alignformer_detector.yaml before training finishes.
OUTPUT_DIR = Path("outputs/alignformer/point_pillar_late_fusion")

_SHIM_STATS = {"hits": 0, "misses": 0}


def build_scenario_split(train_root: Path, train_dir: Path, val_dir: Path) -> None:
    """Materialize a scenario-level train/val split as symlink directories.

    Idempotent: safe to call every run. Rebuilds from scratch if the set of
    scenarios on disk has ever changed, so it never silently trains on a
    stale split.
    """
    scenarios = sorted(p.name for p in train_root.iterdir() if p.is_dir())
    if not scenarios:
        raise FileNotFoundError(f"no scenario directories under {train_root}")

    train_scenarios, val_scenarios = split_scenarios(scenarios, VAL_FRACTION, SPLIT_SEED)

    for split_dir, names in ((train_dir, train_scenarios), (val_dir, val_scenarios)):
        existing = {p.name for p in split_dir.iterdir()} if split_dir.exists() else set()
        if existing == set(names):
            continue
        split_dir.mkdir(parents=True, exist_ok=True)
        for stale in existing - set(names):
            (split_dir / stale).unlink()
        for name in names:
            link = split_dir / name
            if not link.exists():
                link.symlink_to(train_root / name)

    print(
        f"scenario split: {len(train_scenarios)} train, {len(val_scenarios)} val "
        f"(of {len(scenarios)}, seed={SPLIT_SEED})",
        flush=True,
    )


def _check_shim_equivalence(real_pcd_to_np: Callable[[str], np.ndarray], sample_size: int = 5) -> None:
    """Assert the cache and the real decoder agree exactly on real frames."""
    checked = 0
    for cached_file in PCD_CACHE_ROOT.rglob("*.npy"):
        relative = cached_file.relative_to(PCD_CACHE_ROOT).with_suffix(".pcd")
        source = OPV2V_TRAIN_ROOT / relative
        if not source.exists():
            continue
        cached_array = np.load(cached_file)
        real_array = real_pcd_to_np(str(source))
        if not np.array_equal(cached_array, real_array):
            raise AssertionError(
                f"pcd cache mismatch on {source}: cached shape {cached_array.shape} "
                f"vs real shape {real_array.shape}"
            )
        checked += 1
        if checked >= sample_size:
            break
    if checked == 0:
        raise RuntimeError("could not find any overlapping cache/source pair to sanity-check")
    print(f"pcd cache equivalence check passed on {checked} frames", flush=True)


def install_pcd_cache_shim() -> None:
    """Monkey-patch ``opencood.utils.pcd_utils.pcd_to_np`` to prefer the cache.

    Only one call site in OpenCOOD resolves ``pcd_to_np`` (``basedataset.py``,
    via ``import opencood.utils.pcd_utils as pcd_utils`` then
    ``pcd_utils.pcd_to_np(...)``), so patching the module attribute is
    sufficient -- there is no ``from ... import pcd_to_np`` binding elsewhere
    in the codebase to miss.
    """
    from opencood.utils import pcd_utils as opencood_pcd_utils

    real_pcd_to_np = opencood_pcd_utils.pcd_to_np
    _check_shim_equivalence(real_pcd_to_np)

    train_root = OPV2V_TRAIN_ROOT.resolve()

    def shimmed_pcd_to_np(pcd_file: str) -> np.ndarray:
        # Resolve first: root_dir/validate_dir point through the scenario
        # symlinks built by build_scenario_split(), so the raw path carries
        # the split-dir prefix rather than the real OPV2V tree.
        source = Path(pcd_file).resolve()
        try:
            cached = cached_pcd_path(PCD_CACHE_ROOT, source, train_root)
        except ValueError:
            _SHIM_STATS["misses"] += 1
            return real_pcd_to_np(pcd_file)

        if cached.exists():
            _SHIM_STATS["hits"] += 1
            return np.load(cached)

        _SHIM_STATS["misses"] += 1
        return real_pcd_to_np(pcd_file)

    opencood_pcd_utils.pcd_to_np = shimmed_pcd_to_np
    print("pcd cache shim installed", flush=True)


def install_fixed_output_dir() -> None:
    """Make OpenCOOD's trainer save to a fixed, predictable directory.

    Stock ``train_utils.setup_train`` names the output folder after a
    timestamp taken at call time, which this project's config cannot
    reference in advance. This keeps its exact behaviour (create the
    directory, dump ``hypes`` as ``config.yaml`` inside it) but at a path we
    choose, so ``configs/alignformer_detector.yaml`` can point at
    ``detector.checkpoint`` before training finishes.
    """
    from opencood.tools import train_utils

    def fixed_setup_train(hypes: dict) -> str:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_DIR / "config.yaml", "w") as handle:
            yaml.dump(hypes, handle)
        return str(OUTPUT_DIR)

    train_utils.setup_train = fixed_setup_train


def log_shim_stats() -> None:
    total = _SHIM_STATS["hits"] + _SHIM_STATS["misses"]
    rate = _SHIM_STATS["hits"] / total if total else 0.0
    print(f"pcd cache shim: {_SHIM_STATS['hits']}/{total} hits ({rate:.1%})", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hypes_yaml",
        type=Path,
        default=Path("configs/alignformer_detector.yaml"),
        help="OpenCOOD-format hypes yaml (a machine-local copy of point_pillar_late_fusion.yaml).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    build_scenario_split(OPV2V_TRAIN_ROOT, TRAIN_SPLIT_DIR, VAL_SPLIT_DIR)
    install_pcd_cache_shim()
    install_fixed_output_dir()

    # Import after the shims are installed but before any dataset is built,
    # so DataLoader worker processes (forked from this one) inherit the
    # patched module state.
    from opencood.tools import train as opencood_train

    sys.argv = ["train.py", "--hypes_yaml", str(args.hypes_yaml)]
    try:
        opencood_train.main()
    finally:
        log_shim_stats()


if __name__ == "__main__":
    sys.exit(main())
