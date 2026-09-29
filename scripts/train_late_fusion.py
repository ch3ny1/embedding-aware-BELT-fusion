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

Three things this script checks before trusting the shim, every run:
  - Content equivalence: the cached array and the real ``pcd_to_np`` must
    agree byte-for-byte on a sample of real frames, read directly from the
    real OPV2V tree (see ``_check_shim_equivalence``).
  - Path-resolution equivalence: calling the *patched* function through a
    train/val split symlink (the only kind of path it ever actually sees at
    runtime) must register as a cache hit, not just return the right bytes
    by falling back to the real decoder (see ``_check_shim_resolves_symlink``).
    A systematically wrong path mapping would pass the content check above
    while silently missing every real call -- this is what would have
    caught that.
  - Hit rate: ``ShimStats`` tracks cache hits/misses in a
    ``multiprocessing.Value`` (real shared memory, not a plain dict), so
    counts recorded inside ``DataLoader``'s forked worker processes
    aggregate back to the parent that logs them. Logged when training ends,
    so a silently-idle shim (e.g. from a cache-root mixup after this file is
    edited) shows up as a hit rate near zero rather than as an unexplained
    lack of speedup. Run with ``--verify-shim-iters`` to check this before
    committing to a multi-hour run (see ``run_shim_verification``).

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

    # Sanity-check the shim (a few dozen batches, no training) before a real run:
    python -u scripts/train_late_fusion.py --verify-shim-iters 30
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
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
# Overridable with --output-dir: the +/-70.4 m and +/-140.8 m detectors have
# to coexist, since the comparison between the two ranges is itself a result
# (see .superpowers/sdd/2026-09-17-alignformer-p0-p2/task-18-report.md).
OUTPUT_DIR = Path("outputs/alignformer/point_pillar_late_fusion")


class ShimStats:
    """Cache hit/miss counters that survive ``DataLoader`` worker forks.

    A plain dict here would live independently in each of ``num_workers``
    forked processes (copy-on-write) and never aggregate back to the parent
    that logs it at the end -- which is exactly what happened before this
    fix: the counter printed ``0/0`` on every run, whether the shim was
    working perfectly or silently misconfigured. ``multiprocessing.Value``
    is backed by real shared memory, so increments made inside a worker are
    visible here as long as ``ShimStats`` is constructed *before*
    ``DataLoader`` forks its workers (see ``install_pcd_cache_shim``, called
    before ``opencood_train.main()`` builds any dataset).
    """

    def __init__(self) -> None:
        self.hits = mp.Value("L", 0)
        self.misses = mp.Value("L", 0)

    def record_hit(self) -> None:
        with self.hits.get_lock():
            self.hits.value += 1

    def record_miss(self) -> None:
        with self.misses.get_lock():
            self.misses.value += 1

    def snapshot(self) -> tuple[int, int]:
        return int(self.hits.value), int(self.misses.value)

    def reset(self) -> None:
        with self.hits.get_lock():
            self.hits.value = 0
        with self.misses.get_lock():
            self.misses.value = 0


def build_scenario_split(train_root: Path, train_dir: Path, val_dir: Path) -> None:
    """Materialize a scenario-level train/val split as symlink directories.

    Idempotent: safe to call every run. Rebuilds from scratch if the set of
    scenarios on disk has ever changed, so it never silently trains on a
    stale split.

    Every link is checked for where it actually points, not merely for whether
    a name is present. ``Path.exists()`` follows symlinks, so a dangling link
    reads as absent and the naive ``if not link.exists(): symlink_to(...)``
    raises ``FileExistsError`` instead of repairing it -- and the name-set
    fast path this used to take would not even look. That is not theoretical:
    this repository lives on an NTFS volume that has stopped round-tripping
    symlinks (they read back as ``unsupported reparse tag``), which is why the
    wide-range run materializes its split on the ext4 disk via
    ``--split-root``.
    """
    scenarios = sorted(p.name for p in train_root.iterdir() if p.is_dir())
    if not scenarios:
        raise FileNotFoundError(f"no scenario directories under {train_root}")

    train_scenarios, val_scenarios = split_scenarios(scenarios, VAL_FRACTION, SPLIT_SEED)

    for split_dir, names in ((train_dir, train_scenarios), (val_dir, val_scenarios)):
        split_dir.mkdir(parents=True, exist_ok=True)
        existing = {p.name for p in split_dir.iterdir()}
        for stale in existing - set(names):
            (split_dir / stale).unlink()
        for name in names:
            link, target = split_dir / name, train_root / name
            if link.is_symlink() or link.exists():
                if link.is_symlink() and link.resolve() == target.resolve():
                    continue
                link.unlink()
            link.symlink_to(target)

    print(
        f"scenario split: {len(train_scenarios)} train, {len(val_scenarios)} val "
        f"(of {len(scenarios)}, seed={SPLIT_SEED})",
        flush=True,
    )


def _check_shim_equivalence(
    real_pcd_to_np: Callable[[str], np.ndarray],
    cache_root: Path,
    train_root: Path,
    sample_size: int = 5,
) -> None:
    """Assert the cache and the real decoder agree exactly on real frames.

    This reads straight from ``train_root`` (the real OPV2V tree), not
    through a split symlink -- it checks cache *content* only. Path
    *resolution* through the symlinks the shim actually sees at runtime is
    a separate check: ``_check_shim_resolves_symlink``.
    """
    checked = 0
    for cached_file in cache_root.rglob("*.npy"):
        relative = cached_file.relative_to(cache_root).with_suffix(".pcd")
        source = train_root / relative
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
        raise RuntimeError(f"could not find any overlapping cache/source pair under {cache_root} to sanity-check")
    print(f"pcd cache equivalence check passed on {checked} frames", flush=True)


def _iter_pcd_files_through_symlinks(root: Path):
    """Yield ``.pcd`` files under ``root``, descending into symlinked dirs.

    ``Path.rglob`` follows a symlink when it *is* the glob root, but does not
    descend into one encountered mid-traversal -- and ``build_scenario_split``'s
    split directories are exactly that: a real directory whose immediate
    children are symlinks to the real scenario directories. Globbing
    ``split_dir.rglob("*.pcd")`` directly silently finds nothing.
    """
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            yield from entry.rglob("*.pcd")


def _check_shim_resolves_symlink(
    shimmed_pcd_to_np: Callable[[str], np.ndarray],
    stats: ShimStats,
    split_dir: Path,
    sample_size: int = 5,
) -> None:
    """Assert the *patched* function registers a hit through a split symlink.

    ``_check_shim_equivalence`` calls the real decoder directly against
    ``train_root`` paths, which validates cache content but not path
    resolution. At runtime the shim only ever sees paths through
    ``TRAIN_SPLIT_DIR``/``VAL_SPLIT_DIR`` (the symlinks ``build_scenario_split``
    builds), which then have to resolve back through ``train_root`` before
    ``cached_pcd_path`` can look anything up. A systematically wrong mapping
    here would silently fall back to the real decoder on every call --
    still producing correct results, just slowly, and the content check
    above would not notice. This calls the shim itself (not the real
    decoder) on real symlinked paths and requires a hit, not just a
    matching array.
    """
    if not split_dir.exists():
        raise RuntimeError(f"split directory does not exist yet: {split_dir}")

    checked = 0
    for pcd_file in _iter_pcd_files_through_symlinks(split_dir):
        hits_before, _ = stats.snapshot()
        shimmed_pcd_to_np(str(pcd_file))
        hits_after, _ = stats.snapshot()
        if hits_after != hits_before + 1:
            raise AssertionError(
                f"pcd cache shim fell back to the real decoder for a symlinked "
                f"path with a known cache entry: {pcd_file}. A wrong path "
                f"mapping would silently do this on every call and still "
                f"produce correct-but-slow results."
            )
        checked += 1
        if checked >= sample_size:
            break
    if checked == 0:
        raise RuntimeError(f"no .pcd files found under {split_dir} to check symlink resolution")
    print(f"pcd cache shim symlink-resolution check passed on {checked} frames under {split_dir}", flush=True)


def install_pcd_cache_shim(
    cache_root: Path = PCD_CACHE_ROOT,
    train_root: Path = OPV2V_TRAIN_ROOT,
    split_dir: Path = TRAIN_SPLIT_DIR,
    skip_preflight: bool = False,
) -> ShimStats:
    """Monkey-patch ``opencood.utils.pcd_utils.pcd_to_np`` to prefer the cache.

    Only one call site in OpenCOOD resolves ``pcd_to_np`` (``basedataset.py``,
    via ``import opencood.utils.pcd_utils as pcd_utils`` then
    ``pcd_utils.pcd_to_np(...)``), so patching the module attribute is
    sufficient -- there is no ``from ... import pcd_to_np`` binding elsewhere
    in the codebase to miss.

    ``skip_preflight`` exists only to demonstrate, in isolation, that the
    hit/miss counter itself correctly attributes misses when pointed at a
    wrong ``cache_root`` -- with preflight checks enabled (the default, and
    the only mode ``main()`` uses for real training), a wrong ``cache_root``
    fails loudly here before any training starts, which is strictly better
    than "reports misses" for production use.
    """
    from opencood.utils import pcd_utils as opencood_pcd_utils

    real_pcd_to_np = opencood_pcd_utils.pcd_to_np
    stats = ShimStats()
    resolved_train_root = train_root.resolve()

    def shimmed_pcd_to_np(pcd_file: str) -> np.ndarray:
        # Resolve first: root_dir/validate_dir point through the scenario
        # symlinks built by build_scenario_split(), so the raw path carries
        # the split-dir prefix rather than the real OPV2V tree.
        source = Path(pcd_file).resolve()
        try:
            cached = cached_pcd_path(cache_root, source, resolved_train_root)
        except ValueError:
            stats.record_miss()
            return real_pcd_to_np(pcd_file)

        if cached.exists():
            stats.record_hit()
            return np.load(cached)

        stats.record_miss()
        return real_pcd_to_np(pcd_file)

    if not skip_preflight:
        _check_shim_equivalence(real_pcd_to_np, cache_root, resolved_train_root)
        _check_shim_resolves_symlink(shimmed_pcd_to_np, stats, split_dir)
        stats.reset()  # the checks above are self-tests, not real training hits

    opencood_pcd_utils.pcd_to_np = shimmed_pcd_to_np
    print("pcd cache shim installed", flush=True)
    return stats


def install_fixed_output_dir(output_dir: Path = OUTPUT_DIR) -> None:
    """Make OpenCOOD's trainer save to a fixed, predictable directory.

    Stock ``train_utils.setup_train`` names the output folder after a
    timestamp taken at call time, which this project's config cannot
    reference in advance. This keeps its exact behaviour (create the
    directory, dump ``hypes`` as ``config.yaml`` inside it) but at a path we
    choose, so ``configs/alignformer_detector.yaml`` can point at
    ``detector.checkpoint`` before training finishes.

    ``output_dir`` is a parameter rather than the module constant so two
    ranges' detectors can be trained without one overwriting the other.
    """
    from opencood.tools import train_utils

    def fixed_setup_train(hypes: dict) -> str:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "config.yaml", "w") as handle:
            yaml.dump(hypes, handle)
        return str(output_dir)

    train_utils.setup_train = fixed_setup_train


def install_v2xreal_training() -> None:
    """Route OpenCOOD's trainer through the V2X-Real dataset builder.

    ``opencood/tools/train.py`` binds ``build_dataset`` by ``from ... import``
    at import time, so the module attribute is the slot to replace; the
    adapter delegates every non-V2X-Real ``core_method`` to OpenCOOD
    unchanged. The LiDAR ``.bin`` shim and the range constants are installed
    by the builder itself the first time a V2X-Real dataset is built, which
    happens inside ``opencood_train.main()`` before any DataLoader forks.
    """
    from opencood.tools import train as opencood_train

    from embedding_aware_belt_fusion.alignformer import v2xreal

    opencood_train.build_dataset = v2xreal.build_dataset
    print("v2xreal dataset builder installed", flush=True)


def log_shim_stats(stats: ShimStats) -> None:
    hits, misses = stats.snapshot()
    total = hits + misses
    rate = hits / total if total else 0.0
    print(f"pcd cache shim: {hits}/{total} hits ({rate:.1%})", flush=True)


def run_shim_verification(hypes_yaml: Path, num_iters: int, stats: ShimStats) -> None:
    """Run a few dozen real training batches through the shim, then stop.

    Builds the exact same dataset/DataLoader ``opencood_train.main()`` would
    (same ``num_workers``), so the shim is exercised inside real forked
    worker processes, not just in this process -- but iterates only
    ``num_iters`` batches and does no training (no model, no optimizer, no
    checkpoint writes). Use this to catch a broken shim in seconds instead
    of discovering it hours into a real run.
    """
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from torch.utils.data import DataLoader

    hypes = load_yaml(str(hypes_yaml))
    dataset = build_dataset(hypes, visualize=False, train=True)
    loader = DataLoader(
        dataset,
        batch_size=hypes["train_params"]["batch_size"],
        num_workers=8,
        collate_fn=dataset.collate_batch_train,
        shuffle=True,
    )

    seen = 0
    for _ in loader:
        seen += 1
        if seen >= num_iters:
            break

    print(f"shim verification: ran {seen} batches (num_workers=8)", flush=True)
    log_shim_stats(stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hypes_yaml",
        type=Path,
        default=Path("configs/alignformer_detector.yaml"),
        help="OpenCOOD-format hypes yaml (a machine-local copy of point_pillar_late_fusion.yaml).",
    )
    parser.add_argument(
        "--dataset",
        choices=("opv2v", "v2xreal"),
        default="opv2v",
        help="opv2v (default): materialize the scenario-disjoint holdout and install the "
        "pcd cache shim, exactly as before. v2xreal: skip both (the tree has no .pcd "
        "files and ships its own val/ split) and route OpenCOOD's trainer through "
        "alignformer.v2xreal.build_dataset.",
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=SPLIT_ROOT,
        help="Where the train/val split symlink directories are materialized. The "
        "default lives in the repository, which is on an NTFS volume that no longer "
        "round-trips symlinks -- point this at a POSIX filesystem when that bites.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Where checkpoints and config.yaml are written. Give each cav_lidar_range "
        "its own directory: the narrow and wide detectors are separate results and "
        "neither may overwrite the other.",
    )
    parser.add_argument(
        "--verify-shim-iters",
        type=int,
        default=0,
        help="If > 0, build the real training DataLoader (num_workers=8), run this many "
        "batches through it, print pcd-cache shim hit/miss counts, and exit without "
        "training. Use to sanity-check the shim before committing to a multi-hour run.",
    )
    parser.add_argument(
        "--pcd-cache-root",
        type=Path,
        default=PCD_CACHE_ROOT,
        help="Override the pcd cache root. Only intended for demonstrating shim behavior "
        "(e.g. pointed at a nonexistent path with --skip-shim-preflight to show the "
        "counter correctly reports 100%% misses) -- real training should use the default.",
    )
    parser.add_argument(
        "--skip-shim-preflight",
        action="store_true",
        help="Skip the equivalence/symlink-resolution preflight checks. Only meaningful "
        "together with --pcd-cache-root pointed at a deliberately wrong path: with a real "
        "cache root, preflight failing loudly is the desired behavior, not something to "
        "bypass.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "v2xreal":
        return _main_v2xreal(args)

    train_split_dir = args.split_root / "train"
    val_split_dir = args.split_root / "val"
    build_scenario_split(OPV2V_TRAIN_ROOT, train_split_dir, val_split_dir)
    stats = install_pcd_cache_shim(
        cache_root=args.pcd_cache_root,
        split_dir=train_split_dir,
        skip_preflight=args.skip_shim_preflight,
    )

    if args.verify_shim_iters > 0:
        run_shim_verification(args.hypes_yaml, args.verify_shim_iters, stats)
        return

    install_fixed_output_dir(args.output_dir)

    # Import after the shims are installed but before any dataset is built,
    # so DataLoader worker processes (forked from this one) inherit the
    # patched module state.
    from opencood.tools import train as opencood_train

    sys.argv = ["train.py", "--hypes_yaml", str(args.hypes_yaml)]
    try:
        opencood_train.main()
    finally:
        log_shim_stats(stats)


def _main_v2xreal(args: argparse.Namespace) -> None:
    """The V2X-Real path: no OPV2V split, no pcd shim, the adapter installed.

    Installed before ``opencood.tools.train`` builds anything, so DataLoader
    workers forked from this process inherit the patched module state, the
    same ordering the OPV2V path relies on for its shim.
    """
    install_v2xreal_training()
    install_fixed_output_dir(args.output_dir)
    from opencood.tools import train as opencood_train

    sys.argv = ["train.py", "--hypes_yaml", str(args.hypes_yaml)]
    opencood_train.main()


if __name__ == "__main__":
    sys.exit(main())
