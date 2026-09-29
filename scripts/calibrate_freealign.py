"""Choose the FreeAlign port's free parameters on the VALIDATION slice.

A reimplementation that loses can always be dismissed as a strawman, and the
paper leaves four quantities unspecified: the edge-discrepancy threshold, the
anchor-list limit ``gamma``, the exponent ``p`` and the constant in the
selection score ``eps = (offset + sum_e eps_e) / r^p``, and the minimum node
count below which a collaborative message is discarded. Two of those have a
value in the authors' shipped ``greedy_match.py`` (``max_error = 0.5`` m,
``min_nodes = 3``) and the rest do not. This script picks them the same way any
of this project's own hyperparameters are picked: by grid search on the
scenario-disjoint 15% validation slice of ``train/``, never on test.

**The criterion is deliberately not sigma = 0.** At zero localization noise the
true correction is exactly the identity, so a method that abstains on every
pair scores a perfect zero -- calibrating there would select for silence. The
grid is scored at a non-zero sigma on translation MAE over **all** pairs,
counting a discarded message as the uncorrected pose it actually produces, so
coverage and accuracy trade off against each other the way they do in the
sweep.

The detector runs once per frame and every configuration in the grid reuses
those detections, exactly as ``noisy_fusion`` does, so the grid measures the
alignment algorithm and nothing else.

Usage::

    python scripts/calibrate_freealign.py \\
        --config configs/alignformer_detector_r140.yaml \\
        --alignformer-config configs/alignformer_r140.yaml \\
        --split /media/chenyi/basement2/cache/opv2v_splits/val \\
        --stride 8 --sigma 1.0 \\
        --output outputs/alignformer/r140/freealign_calibration_result.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from embedding_aware_belt_fusion.alignformer.freealign import (
    EDGE_DISTANCE,
    EDGE_DISTANCE_YAW,
    LMEDS,
    RANSAC,
    FreeAlignConfig,
    freealign_estimate,
)
from embedding_aware_belt_fusion.alignformer.splits import resolve_split
from embedding_aware_belt_fusion.alignformer.stage2 import is_fallback

# The grid. Every axis the paper leaves open, plus the two the shipped code
# fixes (which are included so the authors' own values are measured, not
# assumed to be best).
_GRID = {
    "edge_feature": (EDGE_DISTANCE, EDGE_DISTANCE_YAW),
    "edge_threshold_m": (0.3, 0.5, 1.0, 1.5),
    "anchor_limit": (1, 2, 3),
    "epsilon_offset": (0.0, 100.0),
    "epsilon_power": (1.0, 3.0),
    "robust_estimator": (RANSAC, LMEDS),
}
# Probed on the winning configuration only: it is the abstain policy, not a
# matching parameter, and sweeping it inside the grid would let the search buy
# accuracy by declining the hard pairs.
_MIN_NODES = (2, 3, 4)

# FreeAlign's own headline matching metric (their Table II): the fraction of
# pairs whose transformation error exceeds 3 m, "which makes an accurately
# detected box be regarded as a false positive even in AP 0.3". Reported here
# beside the median because the MEAN is the wrong summary for this estimator --
# a handful of coincidental subgraphs produce tens of metres and drag it, while
# AP responds to the bulk of the distribution.
_ERROR_RATE_THRESHOLD_M = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--alignformer-config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True,
                        help="the VALIDATION split directory; never OPV2V/test")
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="--selected-only: a stage-2 checkpoint. Given, AlignFormer is scored "
             "on the SAME pairs so the median and the error rate are paired.",
    )
    parser.add_argument(
        "--shrinkage", type=Path, default=None,
        help="--checkpoint's shrinkage calibration, so the AlignFormer arm is the "
             "deployed one and not an undeployed variant of it",
    )
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="localization noise the grid is scored at; must be "
                             "non-zero, see the module docstring")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--selected-only", action="store_true",
        help="skip the grid and score the shipped FreeAlignConfig defaults alone. "
             "This is a MEASUREMENT, not a selection, so it is the only mode that "
             "may point at the test split -- the report needs the median and the "
             "error rate there to explain why a method with ten times the mean "
             "error scores a higher AP.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.sigma <= 0:
        parser.error("--sigma must be non-zero: at sigma = 0 abstaining is optimal")
    return args


def _collect_pairs(args, device):
    """One ego-CAV pair per CAV per strided frame, set up exactly as the sweep does."""
    from embedding_aware_belt_fusion.alignformer.v2xreal import build_dataset
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.utils.transformation_utils import x1_to_x2

    from embedding_aware_belt_fusion.alignformer.boxes import detect_agent
    from embedding_aware_belt_fusion.alignformer.dataset import YAW_STD_PER_XY_STD
    from embedding_aware_belt_fusion.alignformer.evaluate import (
        _build_detector,
        _build_test_frame,
        _cav_content,
        _frame_identity,
        _pose_correction,
    )
    from embedding_aware_belt_fusion.alignformer.fusion import correct_boxes
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import (
        _roi_like_the_cache,
        _sweep_rng,
        _truncate_by_score,
        shared_object_count,
    )
    from embedding_aware_belt_fusion.alignformer.trunk import MAX_OBJECTS
    from embedding_aware_belt_fusion.coloca.geometry import (
        perturb_pose_2d,
        relative_pose_error,
    )

    hypes = load_yaml(str(args.config), None)
    # The grid SELECTS hyperparameters and must never see test;
    # --selected-only scores one frozen choice and may.
    hypes["validate_dir"] = str(
        resolve_split(args.split, allow_test=args.selected_only)
    )
    dataset = build_dataset(hypes, visualize=False, train=False)
    detector = _build_detector(hypes, device)
    model_config = yaml.safe_load(args.alignformer_config.read_text())
    output_size = int(model_config["model"]["output_size"])
    lidar_range = hypes["preprocess"]["cav_lidar_range"]

    pairs = []
    with torch.no_grad():
        for index in range(0, len(dataset), max(1, args.stride)):
            scenario, timestamp = _frame_identity(dataset, index)
            sample, poses, ego_pose, _ = _build_test_frame(
                dataset, index, scenario, timestamp
            )
            batch = dataset.collate_batch_test([sample])

            packs = {}
            for key, entry in batch.items():
                found = detect_agent(
                    detector, _cav_content(entry, device), dataset.post_processor
                )
                roi = _roi_like_the_cache(found, lidar_range, output_size)
                boxes, scores, roi, gt_ids = _truncate_by_score(found, roi, MAX_OBJECTS)
                packs[key] = {
                    "boxes": boxes, "scores": scores, "roi": roi, "gt_ids": gt_ids
                }

            ego = packs["ego"]
            for agent, key in enumerate(sorted(k for k in packs if k != "ego")):
                rng = _sweep_rng(args.seed, args.sigma, index, agent)
                noisy_pose = perturb_pose_2d(
                    poses[key], args.sigma, args.sigma * YAW_STD_PER_XY_STD, rng
                )
                noisy_transform = torch.as_tensor(
                    x1_to_x2(noisy_pose, ego_pose), dtype=torch.float32, device=device
                )
                psi_noisy, t_noisy = _pose_correction(noisy_transform)
                dx, dy, dpsi_deg = relative_pose_error(ego_pose, poses[key], noisy_pose)
                pairs.append({
                    "ego": ego,
                    "cav": dict(packs[key]),
                    "ego_boxes": ego["boxes"],
                    "cav_boxes": correct_boxes(packs[key]["boxes"], psi_noisy, t_noisy),
                    "true_t": (dx, dy),
                    "true_psi": float(np.radians(dpsi_deg)),
                    "shared": shared_object_count(ego["gt_ids"], packs[key]["gt_ids"]),
                })
    return pairs


def _batch(pair) -> dict:
    """The one-sample object-set batch both estimators consume."""
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import _object_set

    ego = dict(pair["ego"])
    cav = dict(pair["cav"])
    cav["boxes"] = pair["cav_boxes"]
    return _object_set(ego, cav)


def _score(pairs, config: FreeAlignConfig, alignformer=None) -> dict:
    """Translation and yaw error over all pairs, and over answered pairs only.

    ``alignformer``, when given, is a callable scoring the SAME pairs with the
    deployed estimator, so the two columns differ in the algorithm alone.
    """
    translation, yaw, answered = [], [], []
    for pair in pairs:
        batch = _batch(pair)
        estimate = (
            freealign_estimate(batch, config) if alignformer is None
            else alignformer(batch)
        )
        psi, t = float(estimate.psi[0]), estimate.t[0]
        translation.append(float(np.hypot(
            float(t[0]) - pair["true_t"][0], float(t[1]) - pair["true_t"][1]
        )))
        difference = psi - pair["true_psi"]
        yaw.append(abs(float(np.degrees(
            np.arctan2(np.sin(difference), np.cos(difference))
        ))))
        answered.append(not bool(is_fallback(estimate)[0]))

    return _summarize(np.array(translation), np.array(yaw), np.array(answered))


def _summarize(translation, yaw, answered) -> dict:
    """Mean, median and FreeAlign's own >3 m error rate, overall and answered."""
    def over(values, mask):
        return (
            float((values[mask] > _ERROR_RATE_THRESHOLD_M).mean())
            if mask.any() else None
        )

    return {
        "translation_mae_m": float(translation.mean()),
        "translation_median_m": float(np.median(translation)),
        "yaw_mae_deg": float(yaw.mean()),
        "yaw_median_deg": float(np.median(yaw)),
        "coverage": float(answered.mean()),
        "error_rate_over_3m": over(translation, np.ones_like(answered, dtype=bool)),
        "answered_translation_mae_m": (
            float(translation[answered].mean()) if answered.any() else None
        ),
        "answered_translation_median_m": (
            float(np.median(translation[answered])) if answered.any() else None
        ),
        "answered_yaw_mae_deg": float(yaw[answered].mean()) if answered.any() else None,
        "answered_error_rate_over_3m": over(translation, answered),
    }


def _alignformer_scorer(args, device):
    """The deployed estimator as a one-batch callable, or ``None``.

    Built here rather than imported wholesale so that the diagnostic runs the
    same frozen embedding head, pose head and shrinkage the sweep deploys --
    anything else would compare the port against an estimator nobody ships.
    """
    if args.checkpoint is None:
        return None

    from embedding_aware_belt_fusion.alignformer.evaluate import _load_shrinkage
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import _estimate
    from embedding_aware_belt_fusion.alignformer.shrinkage import shrink
    from embedding_aware_belt_fusion.alignformer.stage2 import load_stage2

    modules, checkpoint = load_stage2(args.checkpoint, device)
    ablate = checkpoint["message_content"] == "boxes_only"
    calibration = _load_shrinkage(args)

    def score(batch):
        estimate = _estimate(modules, batch, ablate)
        return estimate if calibration is None else shrink(estimate, calibration)

    return score


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    pairs = _collect_pairs(args, device)
    print(f"{len(pairs)} pairs from {args.split} at sigma = {args.sigma} m", flush=True)

    if args.selected_only:
        config = FreeAlignConfig()
        deployed = _alignformer_scorer(args, device)
        slices = [
            ("shared_0", [p for p in pairs if p["shared"] == 0]),
            ("shared_1_2", [p for p in pairs if 1 <= p["shared"] <= 2]),
            ("shared_3plus", [p for p in pairs if p["shared"] >= 3]),
        ]
        payload = {
            "method": "freealign_reimplementation",
            "metric": "freealign_pose_diagnostic",
            "note": (
                "A MEASUREMENT of the shipped configuration, not a selection. "
                "Reimplementation of Lei et al., FreeAlign, ICRA 2024."
            ),
            "split": str(args.split),
            "stride": args.stride,
            "sigma_m": args.sigma,
            "pairs": len(pairs),
            "config": config.to_dict(),
            "overall": _score(pairs, config),
            "by_shared_objects": {
                label: _score(subset, config) for label, subset in slices if subset
            },
            "alignformer_overall": (
                None if deployed is None else _score(pairs, config, deployed)
            ),
            "alignformer_by_shared_objects": (
                None if deployed is None else {
                    label: _score(subset, config, deployed)
                    for label, subset in slices if subset
                }
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload["overall"], indent=2))
        for label, row in payload["by_shared_objects"].items():
            print(f"freealign {label}: {json.dumps(row)}")
        if deployed is not None:
            print("alignformer:", json.dumps(payload["alignformer_overall"], indent=2))
            for label, row in payload["alignformer_by_shared_objects"].items():
                print(f"alignformer {label}: {json.dumps(row)}")
        return

    names = list(_GRID)
    results = []
    for values in itertools.product(*(_GRID[name] for name in names)):
        config = FreeAlignConfig(**dict(zip(names, values)))
        row = {"config": config.to_dict(), **_score(pairs, config)}
        results.append(row)
        print(
            f"  {config.edge_feature:<13} thr={config.edge_threshold_m:<4} "
            f"gamma={config.anchor_limit} off={config.epsilon_offset:<5} "
            f"p={config.epsilon_power} {config.robust_estimator:<6} "
            f"-> {row['translation_mae_m']:.4f} m  {row['yaw_mae_deg']:.4f} deg  "
            f"cov {row['coverage']:.3f}",
            flush=True,
        )

    results.sort(key=lambda row: row["translation_mae_m"])
    best = FreeAlignConfig(**{
        name: results[0]["config"][key]
        for name, key in (
            ("edge_feature", "edge_feature"),
            ("edge_threshold_m", "edge_threshold_m"),
            ("anchor_limit", "anchor_limit_gamma"),
            ("epsilon_offset", "epsilon_offset"),
            ("epsilon_power", "epsilon_power"),
            ("robust_estimator", "robust_estimator"),
        )
    })
    abstain = [
        {"min_nodes": count, **_score(pairs, replace(best, min_nodes=count))}
        for count in _MIN_NODES
    ]

    payload = {
        "method": "freealign_reimplementation",
        "metric": "freealign_calibration",
        "note": (
            "Parameters of THIS PROJECT'S reimplementation of Lei et al., "
            "FreeAlign, ICRA 2024 (arXiv 2405.02965). Not the authors' code. "
            "Chosen on the scenario-disjoint validation slice only."
        ),
        "split": str(args.split),
        "stride": args.stride,
        "sigma_m": args.sigma,
        "pairs": len(pairs),
        "selected": best.to_dict(),
        "selected_score": results[0],
        "grid": results,
        "abstain_threshold_probe": abstain,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nselected: {best}")
    for row in abstain:
        print(
            f"  min_nodes={row['min_nodes']}  {row['translation_mae_m']:.4f} m  "
            f"cov {row['coverage']:.3f}  answered {row['answered_translation_mae_m']}"
        )


if __name__ == "__main__":
    main()
