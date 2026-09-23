"""Why is AlignFormer's pose residual larger on test than on validation?

``docs/alignformer_pose_floor.md`` section 6 left one number unexplained. At
sigma = 0, before shrinkage, the translation residual was 0.1192 m on the
scenario-disjoint validation slice and 0.2208 m on the *within-40 m* part of
the test split -- a 1.85x gap that the communication-range mismatch does not
account for, because both of those populations are inside 40 m. Three causes
were candidates:

1. **a cached-vs-live code-path difference** -- validation was measured through
   ``alignformer.dataset`` (the npz cache) and test through
   ``alignformer.noisy_fusion`` (OpenCOOD's ``LateFusionDataset``, detector run
   live). Two paths that are meant to agree, and a silent disagreement between
   them would be a bug, not a result.
2. **scenario difficulty** -- different towns, densities and occlusion on the
   held-out split.
3. **detector quality** -- the same estimator fed noisier correspondences.

This script separates them, using **one** code path so nothing else can move:
it runs the cached path over both populations and reports, per population, the
pose residual, the per-correspondence cross-agent centre disagreement, the
number of matched objects per pair, and the translation MAE that disagreement
and that count predict on their own
(:func:`~embedding_aware_belt_fusion.alignformer.metrics.averaging_limited_translation_mae_m`).

- Cause 1 shows up as **cached test (within range) != the live test number**
  supplied by ``--live-noisy-ap``.
- Causes 2 and 3 are separated by the predicted floor: if the *predicted*
  ratio between two populations matches their *measured* ratio, the estimator
  is behaving identically and the split simply hands it worse correspondences.

Usage::

    python scripts/diagnose_split_gap.py \\
        --config configs/alignformer.yaml \\
        --checkpoint outputs/alignformer/stage2_B_heading_fold/best.pth \\
        --live-noisy-ap outputs/alignformer/p2_heading_fold_noisy_ap_result.json \\
        --output outputs/alignformer/split_gap_diagnostic.json

Shrinkage is deliberately NOT applied: it is calibrated on one of the two
populations being compared, so it would rescale them differently and make the
comparison meaningless. Sigma is 0, the one level at which the emitted
correction IS the residual.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from embedding_aware_belt_fusion.alignformer.dataset import (
    NoiseSchedule,
    OPV2VObjectSetDataset,
    collate,
)
from embedding_aware_belt_fusion.alignformer.metrics import (
    averaging_limited_translation_mae_m,
)
from embedding_aware_belt_fusion.alignformer.stage2 import load_stage2
from embedding_aware_belt_fusion.alignformer.train import (
    _split_name,
    build_pair_split,
    embed_batch,
)
from embedding_aware_belt_fusion.coloca.index import (
    load_or_build_pairs,
    opencood_ego_id,
    pair_cache_path,
)

BATCH_SIZE = 64
# The boundary the 40 m pair index used to draw. Kept as a reporting split even
# now that the index is built at 70 m, because the live numbers this script
# compares against were measured with it.
LEGACY_RANGE_M = 40.0


def pair_distance_m(pair) -> float:
    """Planar ego-CAV separation, the quantity both range filters threshold."""
    return float(np.hypot(pair.cav_pose[0] - pair.ego_pose[0],
                          pair.cav_pose[1] - pair.ego_pose[1]))


class _Accumulator:
    """Per-pair sums for one population, plus the correspondence statistics.

    Sums rather than running means: the last batch is short and averaging
    per-batch means would weight it as heavily as a full one.
    """

    def __init__(self) -> None:
        self.translation = 0.0
        self.yaw = 0.0
        self.pairs = 0
        self.matched = 0
        self.pairs_with_a_match = 0
        self.disagreement_sq = 0.0
        self.disagreement_terms = 0
        self.disagreement_sum = np.zeros(2)
        # Kept per pair, not only summed: a mean cannot tell "uniformly worse"
        # apart from "a tail of failures", and the two have different fixes.
        self.per_pair: List[float] = []
        self.per_scenario: Dict[str, List[float]] = {}

    def add_pair(
        self, translation: float, yaw_deg: float, matched: int, scenario: str
    ) -> None:
        self.translation += translation
        self.yaw += yaw_deg
        self.pairs += 1
        self.matched += matched
        self.pairs_with_a_match += int(matched > 0)
        self.per_pair.append(translation)
        self.per_scenario.setdefault(scenario, []).append(translation)

    def add_disagreements(self, offsets: np.ndarray) -> None:
        """``offsets`` is ``(k, 2)``: CAV centre minus ego centre, per correspondence."""
        if offsets.size == 0:
            return
        self.disagreement_sq += float((offsets ** 2).sum())
        self.disagreement_terms += int(offsets.shape[0])
        self.disagreement_sum += offsets.sum(axis=0)

    def compute(self) -> Dict[str, Optional[float]]:
        if not self.pairs:
            return {"pairs": 0.0}
        # Per-AXIS standard deviation: the sum of squares is over both axes, so
        # the divisor counts both, matching the isotropic model the prediction
        # in metrics.averaging_limited_translation_mae_m assumes.
        axes = 2 * self.disagreement_terms
        per_axis = math.sqrt(self.disagreement_sq / axes) if axes else float("nan")
        matched_per_pair = self.matched / self.pairs
        predicted = (
            averaging_limited_translation_mae_m(per_axis, matched_per_pair)
            if matched_per_pair > 0 and axes
            else None
        )
        mean_offset = (
            (self.disagreement_sum / self.disagreement_terms).tolist()
            if self.disagreement_terms
            else [None, None]
        )
        residuals = np.array(self.per_pair)
        return {
            "pairs": float(self.pairs),
            "translation_mae_m": self.translation / self.pairs,
            "translation_median_m": float(np.median(residuals)),
            "translation_p90_m": float(np.quantile(residuals, 0.9)),
            "scenarios": float(len(self.per_scenario)),
            "per_scenario": {
                scenario: {"pairs": len(values), "translation_mae_m": float(np.mean(values))}
                for scenario, values in sorted(self.per_scenario.items())
            },
            "yaw_mae_deg": self.yaw / self.pairs,
            "matched_objects_per_pair": matched_per_pair,
            "pairs_with_no_match_fraction": 1.0 - self.pairs_with_a_match / self.pairs,
            "correspondences": float(self.disagreement_terms),
            "per_axis_centre_disagreement_m": per_axis,
            "mean_centre_disagreement_m": mean_offset,
            "averaging_limited_translation_mae_m": predicted,
        }


def opencood_ego_by_scenario(pairs) -> Dict[str, str]:
    """The agent OpenCOOD would make ego in each scenario the pairs cover.

    The pair index enumerates every ORDERED pair, so it contains roughly three
    times what the fused-AP sweep evaluates: the sweep keeps only OpenCOOD's
    one ego per scenario. Comparing the cached path's aggregate against the
    live path's aggregate without this restriction compares two different
    populations and would attribute a population difference to the code path.
    """
    agents: Dict[str, set] = {}
    for pair in pairs:
        agents.setdefault(pair.scenario, set()).update((pair.ego_id, pair.cav_id))
    return {scenario: opencood_ego_id(sorted(ids)) for scenario, ids in agents.items()}


@torch.no_grad()
def measure(modules, pairs, config, split: str, device) -> Dict[str, Dict]:
    """Run the cached path at sigma = 0 and tally it several ways.

    ``all`` / ``within_40m`` / ``beyond_40m``: the legacy boundary is kept as a
    reporting split so the numbers line up with the live sweep's own split.
    ``opencood_ego*``: the same, restricted to the pairs whose ego is the one
    OpenCOOD's ``LateFusionDataset`` selects, which is the exact population the
    live fused-AP sweep measures its pose error over.
    """
    # Constructed directly rather than through ``train.build_eval_dataset``,
    # which hard-codes the train split's cache subdirectory: this diagnostic
    # has to point the SAME reader at the test split's cache.
    dataset = OPV2VObjectSetDataset(
        pairs,
        cache_root=config["data"]["cache_root"],
        split=split,
        noise_schedule=NoiseSchedule(max_xy_std=0.0),
        train=False,
        total_epochs=1,
    )
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=8, collate_fn=collate, pin_memory=True
    )
    names = (
        "all", "within_40m", "beyond_40m",
        "opencood_ego_all", "opencood_ego_within_40m", "opencood_ego_beyond_40m",
    )
    groups = {name: _Accumulator() for name in names}
    distances = [pair_distance_m(pair) for pair in pairs]
    egos = opencood_ego_by_scenario(pairs)
    is_ego_pair = [pair.ego_id == egos[pair.scenario] for pair in pairs]

    index = 0
    for batch in loader:
        on_device = {key: value.to(device) for key, value in batch.items()}
        enriched = embed_batch(modules["embedding"], on_device)
        estimate = modules["pose"](enriched)

        residual_t = (estimate.t - on_device["t_true"]).cpu().numpy()
        residual_psi = (estimate.psi - on_device["psi_true"]).cpu().numpy()
        yaw = np.abs(np.degrees(np.arctan2(np.sin(residual_psi), np.cos(residual_psi))))
        translation = np.linalg.norm(residual_t, axis=-1)

        ego_boxes = batch["ego_boxes"].numpy()
        cav_boxes = batch["cav_boxes"].numpy()
        ego_match = batch["ego_match"].numpy()
        ego_mask = batch["ego_mask"].numpy()

        for row in range(translation.shape[0]):
            selected = (ego_match[row] >= 0) & ego_mask[row]
            rows = np.nonzero(selected)[0]
            offsets = (
                cav_boxes[row, ego_match[row][rows], :2] - ego_boxes[row, rows, :2]
                if rows.size
                else np.zeros((0, 2))
            )
            bucket = "within_40m" if distances[index] <= LEGACY_RANGE_M else "beyond_40m"
            selected_groups = ["all", bucket]
            if is_ego_pair[index]:
                selected_groups += ["opencood_ego_all", "opencood_ego_" + bucket]
            for name in selected_groups:
                groups[name].add_pair(
                    float(translation[row]), float(yaw[row]), int(rows.size),
                    pairs[index].scenario,
                )
                groups[name].add_disagreements(offsets)
            index += 1

    if index != len(pairs):
        raise RuntimeError(f"scored {index} pairs but the population has {len(pairs)}")
    return {name: group.compute() for name, group in groups.items()}


def _live_reference(path: Optional[Path]) -> Dict[str, Optional[float]]:
    """The live-path test numbers this diagnostic is contrasted against."""
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    pose = payload["pose"]["sigma_0m"]
    return {
        "source": str(path),
        "shrinkage": payload.get("shrinkage"),
        "all_translation_mae_m": pose["translation_mae_m"],
        "within_40m_translation_mae_m": pose["within_training_range_translation_mae_m"],
        "beyond_40m_translation_mae_m": pose["beyond_training_range_translation_mae_m"],
        "all_yaw_mae_deg": pose["yaw_mae_deg"],
        "within_40m_yaw_mae_deg": pose["within_training_range_yaw_mae_deg"],
        "pairs": pose["pairs"],
    }


def _report(name: str, values: Dict[str, Optional[float]]) -> None:
    if not values.get("pairs"):
        print(f"  {name:<28} (empty)")
        return
    print(
        f"  {name:<28} n={int(values['pairs']):>6}  "
        f"t_mae={values['translation_mae_m']:.4f} m  "
        f"yaw_mae={values['yaw_mae_deg']:.4f} deg  "
        f"matched/pair={values['matched_objects_per_pair']:.2f}  "
        f"per-axis disagreement={values['per_axis_centre_disagreement_m']:.4f} m  "
        f"predicted floor={values['averaging_limited_translation_mae_m']:.4f} m  "
        f"median={values['translation_median_m']:.4f} p90={values['translation_p90_m']:.4f}  "
        f"scenarios={int(values['scenarios'])}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/alignformer.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True, help="a stage-2 checkpoint")
    parser.add_argument(
        "--live-noisy-ap", type=Path, default=None,
        help="a --metric noisy_ap result JSON, measured WITHOUT shrinkage, whose "
             "sigma = 0 pose block is the live-path reference",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = yaml.safe_load(args.config.read_text())
    data = config["data"]
    comm_range_m = float(data["comm_range_m"])

    modules, checkpoint = load_stage2(args.checkpoint, device)
    if checkpoint["message_content"] != "boxes+embeddings":
        raise ValueError(
            "this diagnostic compares the deployed configuration; pass a "
            f"boxes+embeddings checkpoint, not {checkpoint['message_content']}"
        )

    train_pairs, val_pairs, _, val_scenarios = build_pair_split(config)
    test_root = Path(data["test_root"])
    test_pairs = load_or_build_pairs(
        test_root,
        pair_cache_path(data["pair_cache_dir"], _split_name(test_root), comm_range_m),
        comm_range_m,
    )

    populations = {
        "validation (held-out train scenarios)": (val_pairs, _split_name(Path(data["train_root"]))),
        "train (the fitted scenarios)": (train_pairs, _split_name(Path(data["train_root"]))),
        "test (official split)": (test_pairs, _split_name(test_root)),
    }

    results: Dict[str, Dict] = {}
    for name, (pairs, split) in populations.items():
        print(f"{name}: {len(pairs)} pairs", flush=True)
        results[name] = measure(modules, pairs, config, split, device)
        for bucket, values in results[name].items():
            _report(bucket, values)

    payload = {
        "diagnostic": "validation-vs-test pose residual gap",
        "config": str(args.config),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_provenance": {
            key: checkpoint[key] for key in ("head", "message_content", "match_weight", "epoch")
        },
        "comm_range_m": comm_range_m,
        "sigma_m": 0.0,
        "shrinkage": None,
        "val_scenarios": val_scenarios,
        "cached_path": results,
        "live_path_reference": _live_reference(args.live_noisy_ap),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
