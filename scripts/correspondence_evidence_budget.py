"""How much correspondence evidence the learned matcher gathers, against the oracle's.

Task 21 substitutes the ground-truth one-to-one assignment for the learned
Sinkhorn correspondence and finds the substitution worth far more on test than
on validation. This script is the diagnostic that explains why, and it also
bounds what the substitution can mean at all.

Two facts it measures, per ego-CAV pair, at sigma = 0 (the true projection, so
nothing here is about pose error):

1. **The oracle is not a superset of the learned correspondence.** A detection
   only carries a ``gt_id`` when it matched its *own* agent's local ground
   truth at IoU >= 0.3, so an ego object that both agents genuinely saw but
   that one of them failed that gate on is invisible to the oracle and gets
   zero mass. The ``no ground-truth id`` line is how much of the ego object set
   that removes.
2. **Total match mass, learned vs oracle.** The oracle puts weight 1.0 on each
   true pair, so its total mass IS its number of correspondences. Comparing
   that with the Sinkhorn mass says whether the learned matcher is gathering
   more or less evidence than the ground truth offers -- which is the variable
   that separates the two splits.

It also breaks the sigma = 0 estimated translation down by how many true
correspondences the pair has. At sigma = 0 the correct correction is exactly the
identity, so ``|t|`` **is** the estimator's error. A hard one-to-one assignment
on a pair with one or two correspondences has no averaging to fall back on and
is far worse than the soft match there; that tail is the second reason a
perfect assignment can lose.

``--stride`` subsamples frames; the printed ``pairs`` count is the real
denominator. Consecutive OPV2V frames are near-duplicates, so a stride costs
much less information than it looks like it should.

Usage::

    python scripts/correspondence_evidence_budget.py \\
        --config configs/alignformer_detector_r140.yaml \\
        --alignformer-config configs/alignformer_r140.yaml \\
        --checkpoint outputs/alignformer/r140/stage2_B_ivw_scalar/best.pth \\
        --split /media/chenyi/basement2/cache/opv2v_splits/val --stride 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from embedding_aware_belt_fusion.alignformer.splits import resolve_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="OpenCOOD detector hypes yaml")
    parser.add_argument("--alignformer-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="stage-2 checkpoint")
    parser.add_argument("--split", type=Path, required=True, help="OPV2V split directory")
    parser.add_argument("--stride", type=int, default=8, help="take every Nth frame")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main() -> None:
    from embedding_aware_belt_fusion.alignformer.v2xreal import build_dataset
    from opencood.hypes_yaml.yaml_utils import load_yaml

    from embedding_aware_belt_fusion.alignformer.boxes import detect_agent
    from embedding_aware_belt_fusion.alignformer.dataset import correspondence_indices
    from embedding_aware_belt_fusion.alignformer.evaluate import (
        _build_detector,
        _build_test_frame,
        _cav_content,
        _frame_identity,
        _pose_correction,
    )
    from embedding_aware_belt_fusion.alignformer.fusion import correct_boxes
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import (
        _estimate,
        _object_set,
        _roi_like_the_cache,
        _truncate_by_score,
    )
    from embedding_aware_belt_fusion.alignformer.stage2 import load_stage2
    from embedding_aware_belt_fusion.alignformer.trunk import MAX_OBJECTS

    args = parse_args()
    device = torch.device(args.device)

    hypes = load_yaml(str(args.config), None)
    hypes["validate_dir"] = str(resolve_split(args.split, allow_test=False))
    dataset = build_dataset(hypes, visualize=False, train=False)
    detector = _build_detector(hypes, device)
    model_config = yaml.safe_load(args.alignformer_config.read_text())
    modules, checkpoint = load_stage2(args.checkpoint, device)
    ablate = checkpoint["message_content"] == "boxes_only"
    lidar_range = hypes["preprocess"]["cav_lidar_range"]
    output_size = int(model_config["model"]["output_size"])

    from embedding_aware_belt_fusion.alignformer.oracle import (
        oracle_assignment,
        oracle_pose_estimate,
    )

    heading_lambda = float(modules["pose"].heading_lambda)
    variance_model = modules["pose"].variance_model

    oracle_mass, learned_mass, unmatched_ego, ego_objects = [], [], [], []
    oracle_error, learned_error = [], []
    with torch.no_grad():
        for index in range(0, len(dataset), max(1, args.stride)):
            scenario, timestamp = _frame_identity(dataset, index)
            sample, _, _, _ = _build_test_frame(dataset, index, scenario, timestamp)
            batch = dataset.collate_batch_test([sample])

            packs = {}
            for key, entry in batch.items():
                found = detect_agent(
                    detector, _cav_content(entry, device), dataset.post_processor
                )
                roi = _roi_like_the_cache(found, lidar_range, output_size)
                boxes, scores, roi, gt_ids = _truncate_by_score(found, roi, MAX_OBJECTS)
                # The TRUE projection: this diagnostic is about correspondence,
                # never about pose error.
                psi, translation = _pose_correction(entry["transformation_matrix"].to(device))
                packs[key] = {
                    "boxes": correct_boxes(boxes, psi, translation),
                    "scores": scores,
                    "roi": roi,
                    "gt_ids": gt_ids,
                }

            ego = packs["ego"]
            for key in (k for k in packs if k != "ego"):
                ego_match, _ = correspondence_indices(ego["gt_ids"], packs[key]["gt_ids"])
                oracle_mass.append(float((ego_match >= 0).sum()))
                ego_objects.append(len(ego["gt_ids"]))
                unmatched_ego.append(sum(1 for i in ego["gt_ids"] if i is None))

                pair = _object_set(ego, packs[key])
                learned = _estimate(modules, pair, ablate)
                learned_mass.append(float(learned.confidence.item()))
                learned_error.append(float(torch.norm(learned.t[0])))
                oracle = oracle_pose_estimate(
                    pair,
                    oracle_assignment(
                        ego["gt_ids"], packs[key]["gt_ids"], device=device
                    ),
                    heading_lambda=heading_lambda,
                    variance_model=variance_model,
                )
                oracle_error.append(float(torch.norm(oracle.t[0])))

    oracle = np.array(oracle_mass)
    learned = np.array(learned_mass)
    print(f"split  {args.split}")
    print(f"stride {args.stride}  pairs {len(oracle)}")
    print(f"ego detections per pair             : {np.mean(ego_objects):.2f}")
    print(
        f"  of which no ground-truth id       : {np.mean(unmatched_ego):.2f} "
        f"({100 * np.mean(unmatched_ego) / np.mean(ego_objects):.1f}%, invisible to the oracle)"
    )
    print(f"oracle correspondences (unit mass)  : {oracle.mean():.2f}")
    print(f"learned Sinkhorn total mass         : {learned.mean():.2f}")
    print(f"pairs where learned mass > oracle   : {100 * np.mean(learned > oracle):.1f}%")
    print()
    print("estimated |t| at sigma = 0, by the pair's number of true correspondences")
    print("(the true correction is exactly the identity here, so |t| IS the error)")
    print("| correspondences | pairs | share | oracle-IVW |t| (m) | learned |t| (m) |")
    print("|---|---:|---:|---:|---:|")
    oracle_error_a, learned_error_a = np.array(oracle_error), np.array(learned_error)
    for low, high in ((0, 1), (2, 3), (4, 7), (8, 10_000)):
        selected = (oracle >= low) & (oracle <= high)
        if not selected.any():
            continue
        label = f"{low}-{high}" if high < 10_000 else f"{low}+"
        print(
            f"| {label} | {int(selected.sum())} | {100 * selected.mean():.1f}% "
            f"| {oracle_error_a[selected].mean():.3f} "
            f"| {learned_error_a[selected].mean():.3f} |"
        )


if __name__ == "__main__":
    main()
