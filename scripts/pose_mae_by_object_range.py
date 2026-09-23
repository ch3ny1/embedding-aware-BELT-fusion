"""Pose MAE split by the range of the objects the estimate was built from.

Inverse-variance weighting targets a specific defect: far, low-confidence
correspondences entering the Procrustes fit with the same weight as near,
precise ones. An aggregate pose MAE averages that defect away, so this script
reports the same validation pose error the P2 gate reports, bucketed by **how
far away the matched objects behind each estimate were**.

Pose is a per-pair quantity while range is a per-object one, so the bucketing
key is each pair's *median* matched-object range in the ego frame -- the
population of correspondences that pair's Kabsch solve actually saw. Pairs with
no matched object at all are reported separately: no SE(2) is recoverable from
them and the identity is the only safe answer, so they belong in neither
bucket.

Every configuration passed is measured over the same deterministic noise draws
(``evaluate.run_pose``'s seeds), so a before/after comparison is paired.

Usage::

    python scripts/pose_mae_by_object_range.py \\
        --config configs/alignformer_r140.yaml \\
        --checkpoint outputs/alignformer/r140/stage2_B_boxes+embeddings/best.pth \\
                     outputs/alignformer/r140/stage2_B_ivw_split/best.pth \\
        --output outputs/alignformer/r140/pose_by_range_result.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader

from embedding_aware_belt_fusion.alignformer.dataset import collate
from embedding_aware_belt_fusion.alignformer.shrinkage import ShrinkageCalibration, shrink
from embedding_aware_belt_fusion.alignformer.stage2 import _yaw_error_deg, load_stage2
from embedding_aware_belt_fusion.alignformer.train import (
    build_eval_dataset,
    build_pair_split,
    embed_batch,
)

# The 70.4 m edge is the superseded detector range: the bucket beyond it is the
# population the widened detector added and the one task 18 measured as 52%
# noisier in translation.
RANGE_EDGES = (0.0, 25.0, 40.0, 70.4, 1.0e9)
_DEFAULT_SWEEP = (0.0, 0.2, 0.4, 1.0, 2.0)
_POSE_SEED_BASE = 1000
_BATCH_SIZE = 64


class _Bucket:
    """Running sums for one (configuration, sigma, range bucket) cell."""

    def __init__(self) -> None:
        self.n = 0
        self.translation = 0.0
        self.yaw = 0.0
        self.zero_translation = 0.0
        self.zero_yaw = 0.0

    def add(
        self, translation: Tensor, yaw: Tensor, zero_translation: Tensor, zero_yaw: Tensor
    ) -> None:
        self.n += int(translation.numel())
        self.translation += float(translation.sum())
        self.yaw += float(yaw.sum())
        self.zero_translation += float(zero_translation.sum())
        self.zero_yaw += float(zero_yaw.sum())

    def compute(self) -> Dict[str, float]:
        if not self.n:
            return {"pairs": 0}
        return {
            "pairs": self.n,
            "translation_mae_m": self.translation / self.n,
            "yaw_mae_deg": self.yaw / self.n,
            "predict_zero_translation_mae_m": self.zero_translation / self.n,
            "predict_zero_yaw_mae_deg": self.zero_yaw / self.n,
        }


def median_matched_range(batch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
    """``(median range per pair, has-any-match mask)``, both ``(B,)``.

    A pair's key is the median ego-frame range of the objects both agents
    matched to the same physical id -- the correspondences its Kabsch solve
    was actually built from.
    """
    matched = (batch["ego_match"] >= 0) & batch["ego_mask"]
    ranges = batch["ego_boxes"][..., :2].norm(dim=-1)
    # Push unmatched entries above every real range so they sort to the end;
    # the median is then taken over the matched prefix only.
    sentinel = torch.where(matched, ranges, torch.full_like(ranges, math.inf))
    ordered, _ = torch.sort(sentinel, dim=1)
    counts = matched.sum(dim=1)
    index = ((counts - 1).clamp_min(0) // 2).unsqueeze(1)
    median = ordered.gather(1, index).squeeze(1)
    return median, counts > 0


def measure(
    modules,
    loader: DataLoader,
    device: torch.device,
    *,
    ablate_embeddings: bool,
    shrinkage: Optional[ShrinkageCalibration],
    buckets: List[_Bucket],
    unmatched: _Bucket,
) -> None:
    """Accumulate one pass of ``loader`` into ``buckets`` and ``unmatched``."""
    modules.eval()
    edges = torch.tensor(RANGE_EDGES)
    with torch.no_grad():
        for batch in loader:
            key, has_match = median_matched_range(batch)
            batch = {
                name: value.to(device) if isinstance(value, Tensor) else value
                for name, value in batch.items()
            }
            enriched = embed_batch(modules["embedding"], batch, ablate=ablate_embeddings)
            estimate = modules["pose"](enriched)
            if shrinkage is not None:
                estimate = shrink(estimate, shrinkage)

            translation = torch.linalg.norm(
                estimate.t - batch["t_true"], dim=-1
            ).cpu()
            yaw = _yaw_error_deg(estimate.psi, batch["psi_true"]).cpu()
            zero_translation = torch.linalg.norm(batch["t_true"], dim=-1).cpu()
            zero_yaw = _yaw_error_deg(
                torch.zeros_like(batch["psi_true"]), batch["psi_true"]
            ).cpu()

            selection = ~has_match
            if bool(selection.any()):
                unmatched.add(
                    translation[selection], yaw[selection],
                    zero_translation[selection], zero_yaw[selection],
                )
            for index in range(len(RANGE_EDGES) - 1):
                selection = (
                    has_match & (key >= edges[index]) & (key < edges[index + 1])
                )
                if bool(selection.any()):
                    buckets[index].add(
                        translation[selection], yaw[selection],
                        zero_translation[selection], zero_yaw[selection],
                    )


def _shrinkage_for(path: Optional[Path]) -> Optional[ShrinkageCalibration]:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text())
    return ShrinkageCalibration.from_dict(payload["calibration"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/alignformer_r140.yaml"))
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--shrinkage", type=Path, nargs="*", default=None,
        help="one calibration JSON per checkpoint, in the same order, or omitted "
             "for the raw estimator. One tau cannot describe two estimators, so "
             "there is no single-value form.",
    )
    parser.add_argument("--sweep", type=float, nargs="+", default=list(_DEFAULT_SWEEP))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.shrinkage and len(args.shrinkage) != len(args.checkpoint):
        raise SystemExit(
            f"--shrinkage takes one calibration per checkpoint: got "
            f"{len(args.shrinkage)} for {len(args.checkpoint)} checkpoints"
        )

    device = torch.device(args.device)
    config = yaml.safe_load(args.config.read_text())
    _, val_pairs, _, val_scenarios = build_pair_split(config)

    loaded = []
    for index, path in enumerate(args.checkpoint):
        modules, checkpoint = load_stage2(path, device)
        calibration = _shrinkage_for(args.shrinkage[index] if args.shrinkage else None)
        loaded.append((path, modules, checkpoint, calibration))

    seeds = [_POSE_SEED_BASE + offset for offset in range(args.seeds)]
    results: Dict[str, Dict] = {}
    for sigma in args.sweep:
        per_configuration: Dict[str, Dict] = {}
        for path, modules, checkpoint, calibration in loaded:
            buckets = [_Bucket() for _ in range(len(RANGE_EDGES) - 1)]
            unmatched = _Bucket()
            for seed in seeds:
                dataset = build_eval_dataset(config, val_pairs, sigma, seed=seed)
                loader = DataLoader(
                    dataset, batch_size=_BATCH_SIZE, num_workers=args.num_workers,
                    collate_fn=collate, pin_memory=True,
                )
                measure(
                    modules, loader, device,
                    ablate_embeddings=checkpoint["message_content"] == "boxes_only",
                    shrinkage=calibration, buckets=buckets, unmatched=unmatched,
                )
            name = (
                f"{checkpoint['head']} / {checkpoint['message_content']} / "
                f"ivw {checkpoint.get('variance_weighting', 'none')}"
            )
            per_configuration[name] = {
                "checkpoint": str(Path(path).resolve()),
                "shrinkage": None if calibration is None else calibration.to_dict(),
                "buckets": [
                    {
                        "low_m": RANGE_EDGES[index],
                        "high_m": RANGE_EDGES[index + 1],
                        **bucket.compute(),
                    }
                    for index, bucket in enumerate(buckets)
                ],
                "no_matched_object": unmatched.compute(),
            }
            print(f"sigma={sigma:g} m  {name}")
            for entry in per_configuration[name]["buckets"]:
                if not entry["pairs"]:
                    continue
                high = entry["high_m"]
                print(
                    f"   median matched range {entry['low_m']:6.1f}-"
                    f"{'inf' if high > 1e8 else format(high, '6.1f')} m  "
                    f"n={entry['pairs']:6d}  t={entry['translation_mae_m']:.4f} m  "
                    f"yaw={entry['yaw_mae_deg']:.4f} deg"
                )
        results[f"sigma_{sigma:g}m"] = {
            "sigma_xy_m": sigma,
            "configurations": per_configuration,
        }

    payload = {
        "method": "alignformer_pose_by_object_range",
        "config": str(args.config),
        "split": (
            f"{config['data']['train_root']} :: validation scenarios "
            f"(scenario-disjoint, val_scenario_fraction="
            f"{config['data']['val_scenario_fraction']}, split_seed="
            f"{config['data']['split_seed']})"
        ),
        "val_pairs": len(val_pairs),
        "val_scenarios": val_scenarios,
        "range_edges_m": list(RANGE_EDGES),
        "bucket_key": "median ego-frame range of the pair's matched objects",
        "noise_seeds": seeds,
        "sweep_sigmas_m": list(args.sweep),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
