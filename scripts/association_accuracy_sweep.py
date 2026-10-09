"""Association accuracy against localization error: AlignFormer beside FreeAlign.

The stage-1 gate reports one number, cross-agent Top-1 at the stage-1
validation noise (0.5 m). This script asks the same question across the AP
sweep's noise levels and for both aligners, on every ego-CAV pair that
shares at least ``--min-shared`` ground-truth objects:

- ``nearest``: nearest CAV centre under the reported (noisy) pose, the
  geometry-only control; Top-1 over ego objects that have a counterpart.
- ``soft``: AlignFormer's Sinkhorn assignment, Top-1 the way the gate counts it.
- ``hard``: the correspondences behind AlignFormer's verified fit (mutual
  nearest at the last gate under the re-solved pose): precision / recall
  against the ground-truth identities, and the fraction of pairs engaged.
- ``freealign``: the matched common subgraph: precision / recall of its
  pairs and of its RANSAC inliers, and the fraction of pairs answered.

Beside each, the mean centre residual of the corrected CAV boxes against
their true placement, so association and pose error are read together.
Reads val by default; ``--allow-test`` reports on test (a diagnostic, never
a selection).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "external" / "OpenCOOD"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_v2xreal_dense_resolve import box_residual_m, pair_precision_recall  # noqa: E402

DEFAULT_SIGMAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0)
DEFAULT_GATES = (6.0, 3.0, 1.5, 0.75, 0.5)
BUCKETS = (("shared_1_2", 1, 2), ("shared_3plus", 3, 10 ** 6))


def ego_match_from_ids(ego_ids: Sequence[Optional[str]], cav_ids: Sequence[Optional[str]]) -> List[int]:
    """Per ego object, the index of the CAV detection of the same ground-truth object, else -1."""
    where = {g: j for j, g in enumerate(cav_ids) if g is not None}
    return [where.get(g, -1) if g is not None else -1 for g in ego_ids]


def top1_counts_from_prediction(predicted: Sequence[int], ego_match: Sequence[int]) -> Dict[str, int]:
    """``correct`` and ``countable`` over the ego objects that have a counterpart."""
    countable = [i for i, m in enumerate(ego_match) if m >= 0]
    correct = sum(1 for i in countable if predicted[i] == ego_match[i])
    return {"correct": correct, "countable": len(countable)}


def soft_top1(log_assignment: torch.Tensor, ego_match: Sequence[int]) -> Dict[str, int]:
    """Top-1 of the Sinkhorn assignment ``(M+1, N+1)`` with the dustbin column excluded, as the gate counts it."""
    scores = log_assignment[: len(ego_match), :-1]
    return top1_counts_from_prediction(scores.argmax(dim=1).tolist(), ego_match)


def nearest_top1(ego_centres: torch.Tensor, cav_centres: torch.Tensor, ego_match: Sequence[int]) -> Dict[str, int]:
    """Top-1 of nearest-centre matching under the pose the CAV boxes currently sit in."""
    if cav_centres.shape[0] == 0:
        return {"correct": 0, "countable": sum(1 for m in ego_match if m >= 0)}
    predicted = torch.cdist(ego_centres, cav_centres).argmin(dim=1).tolist()
    return top1_counts_from_prediction(predicted, ego_match)


def bucket_of(shared: int) -> str:
    for name, low, high in BUCKETS:
        if low <= shared <= high:
            return name
    return "shared_0"


def _pooled(rows: List[Dict], prefix: str) -> Dict[str, float]:
    correct = sum(r[prefix]["correct"] for r in rows if prefix in r)
    countable = sum(r[prefix]["countable"] for r in rows if prefix in r)
    return {"top1": correct / countable if countable else float("nan"), "countable": countable}


def _pooled_pairs(rows: List[Dict], key: str) -> Dict[str, float]:
    present = [r[key] for r in rows if key in r]
    proposed = sum(p["proposed"] for p in present)
    correct = sum(p["correct"] for p in present)
    shared = sum(p["shared"] for p in present)
    return {"answered": len(present) / len(rows) if rows else float("nan"),
            "precision": correct / proposed if proposed else float("nan"),
            "recall": correct / shared if shared else float("nan"),
            "pairs_mean": proposed / len(present) if present else float("nan")}


def _residual(rows: List[Dict], key: str) -> Dict[str, float]:
    v = np.array([r[key] for r in rows if key in r], dtype=float)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "median": float(np.median(v)), "mean": float(v.mean()), "frac_over_1m": float((v > 1.0).mean())}


def summarize_sigma(rows: List[Dict]) -> Dict:
    """Pooled Top-1, pair precision / recall, coverage and residuals, overall and per shared-object bucket."""
    def block(sub: List[Dict]) -> Dict:
        return {
            "pairs": len(sub),
            "nearest": _pooled(sub, "nearest"),
            "soft": _pooled(sub, "soft"),
            "hard": _pooled_pairs(sub, "hard"),
            "freealign": _pooled_pairs(sub, "freealign"),
            "freealign_inliers": _pooled_pairs(sub, "freealign_inliers"),
            "residual": {k: _residual(sub, k) for k in ("residual_uncorrected", "residual_soft", "residual_hard", "residual_freealign")},
        }
    out = block(rows)
    out["buckets"] = {name: block([r for r in rows if r["bucket"] == name]) for name, _, _ in BUCKETS}
    return out


def _pairs_record(ego_ids, cav_ids, ego_idx, cav_idx) -> Dict:
    precision, recall, correct, proposed = pair_precision_recall(ego_ids, cav_ids, ego_idx, cav_idx)
    shared = len({g for g in ego_ids if g is not None} & {g for g in cav_ids if g is not None})
    return {"precision": precision, "recall": recall, "correct": correct, "proposed": proposed, "shared": shared}


def score_pair(pair, ego_ids, cav_ids, true_cav_boxes, estimates, refine, fa_config, device, heading_lambda) -> Dict:
    """Every association metric and residual for one ego-CAV pair at one draw."""
    from embedding_aware_belt_fusion.alignformer.freealign import edge_features, mass_common_subgraph, robust_se2
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import ALIGNFORMER_IRLS
    from embedding_aware_belt_fusion.alignformer.refine import _moved, mutual_nearest, refined_name

    ego_boxes, cav_boxes = pair["ego_boxes"][0], pair["cav_boxes"][0]
    ego_match = ego_match_from_ids(ego_ids, cav_ids)
    soft = estimates[ALIGNFORMER_IRLS]
    hard = estimates[refined_name(ALIGNFORMER_IRLS, refine.mode)]
    zero = torch.zeros((), device=device), torch.zeros(2, device=device)
    out: Dict = {
        "n_ego": int(ego_boxes.shape[0]), "n_cav": int(cav_boxes.shape[0]),
        "nearest": nearest_top1(ego_boxes[:, :2], cav_boxes[:, :2], ego_match),
        "soft": soft_top1(soft.log_assignment[0], ego_match),
        "residual_uncorrected": box_residual_m(cav_boxes, true_cav_boxes, *zero),
        "residual_soft": box_residual_m(cav_boxes, true_cav_boxes, soft.psi[0], soft.t[0]),
    }
    if hard.refined is not None and bool(hard.refined[0]):
        e_idx, c_idx = mutual_nearest(ego_boxes[:, :2], _moved(cav_boxes[:, :2], hard.psi[0], hard.t[0]), refine.gates_m[-1])
        out["hard"] = _pairs_record(ego_ids, cav_ids, e_idx.tolist(), c_idx.tolist())
        out["residual_hard"] = box_residual_m(cav_boxes, true_cav_boxes, hard.psi[0], hard.t[0])
    match = mass_common_subgraph(edge_features(ego_boxes, fa_config), edge_features(cav_boxes, fa_config), fa_config)
    if len(match) >= fa_config.min_nodes:
        e_list, c_list = match.ego_indices.tolist(), match.cav_indices.tolist()
        out["freealign"] = _pairs_record(ego_ids, cav_ids, e_list, c_list)
        psi, t, inliers = robust_se2(ego_boxes[match.ego_indices, :2], cav_boxes[match.cav_indices, :2], fa_config)
        kept = [k for k, keep in enumerate(inliers.tolist()) if keep]
        out["freealign_inliers"] = _pairs_record(ego_ids, cav_ids, [e_list[k] for k in kept], [c_list[k] for k in kept])
        out["residual_freealign"] = box_residual_m(cav_boxes, true_cav_boxes, psi, t)
    return out


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--alignformer-config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--freealign-calibration", type=Path, required=True)
    parser.add_argument("--sigmas", type=float, nargs="+", default=list(DEFAULT_SIGMAS))
    parser.add_argument("--gates", type=float, nargs="+", default=list(DEFAULT_GATES))
    parser.add_argument("--min-shared", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--allow-test", action="store_true", help="report on the test split (a diagnostic, never a selection)")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--start", type=int, default=0, help="first frame index, for smoke tests")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:  # noqa: C901 - one evaluation loop
    args = parse_args(argv)
    import yaml
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.utils.transformation_utils import x1_to_x2

    from embedding_aware_belt_fusion.alignformer.boxes import detect_agent
    from embedding_aware_belt_fusion.alignformer.cache import _build_detector
    from embedding_aware_belt_fusion.alignformer.draws import sweep_noisy_poses
    from embedding_aware_belt_fusion.alignformer.evaluate import _build_test_frame, _cav_content, _frame_identity, _pose_correction
    from embedding_aware_belt_fusion.alignformer.freealign_calibration import freealign_config_from_calibration
    from embedding_aware_belt_fusion.alignformer.fusion import correct_boxes
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import (
        MAX_OBJECTS, _alignformer_estimates, _object_set, _roi_like_the_cache, _truncate_by_score, shared_object_count,
    )
    from embedding_aware_belt_fusion.alignformer.refine import ICP_RANSAC, RefineConfig
    from embedding_aware_belt_fusion.alignformer.robust import RobustSolveConfig
    from embedding_aware_belt_fusion.alignformer.splits import resolve_split
    from embedding_aware_belt_fusion.alignformer.stage2 import load_stage2
    from embedding_aware_belt_fusion.alignformer.v2xreal import build_dataset

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    hypes = load_yaml(str(args.config), None)
    hypes["validate_dir"] = str(resolve_split(args.split, allow_test=args.allow_test))
    dataset = build_dataset(hypes, visualize=False, train=False)
    detector = _build_detector(hypes, device)
    modules, checkpoint = load_stage2(args.checkpoint, device)
    model_config = yaml.safe_load(args.alignformer_config.read_text())
    ablate = checkpoint["message_content"] == "boxes_only"
    robust = RobustSolveConfig(mode="huber", iterations=2, min_evidence=3.0)
    refine = RefineConfig(mode=ICP_RANSAC, gates_m=tuple(args.gates))
    fa_config = freealign_config_from_calibration(args.freealign_calibration)
    heading_lambda = float(getattr(modules["pose"], "heading_lambda", 2.0))
    lidar_range = hypes["preprocess"]["cav_lidar_range"]
    output_size = int(model_config["model"]["output_size"])
    base_seed = int(model_config["train"]["seed"]) + args.seed
    stop = len(dataset) if args.max_frames is None else min(len(dataset), args.start + args.max_frames * args.stride)
    frames = range(args.start, stop, args.stride)

    rows: Dict[str, List[Dict]] = {f"{s:g}": [] for s in args.sigmas}
    for index in frames:
        scenario, timestamp = _frame_identity(dataset, index)
        sample, poses, ego_pose, _ = _build_test_frame(dataset, index, scenario, timestamp)
        batch = dataset.collate_batch_test([sample])
        packs, transforms = {}, {}
        for key, entry in batch.items():
            found = detect_agent(detector, _cav_content(entry, device), dataset.post_processor)
            roi = _roi_like_the_cache(found, lidar_range, output_size)
            boxes, scores, roi, gt_ids = _truncate_by_score(found, roi, MAX_OBJECTS)
            packs[key] = {"boxes": boxes, "scores": scores, "roi": roi, "gt_ids": gt_ids}
            transforms[key] = entry["transformation_matrix"].to(device)
        ego_pack = packs["ego"]
        cav_keys = sorted(k for k in packs if k != "ego")
        for sigma in args.sigmas:
            drawn = sweep_noisy_poses(poses, cav_keys, sigma=sigma, seed=base_seed, frame=index)
            for key in cav_keys:
                shared = shared_object_count(ego_pack["gt_ids"], packs[key]["gt_ids"])
                if shared < args.min_shared:
                    continue
                noisy_transform = torch.as_tensor(x1_to_x2(drawn[key], ego_pose), dtype=torch.float32, device=device)
                psi_n, t_n = _pose_correction(noisy_transform)
                psi_true, t_true = _pose_correction(transforms[key])
                cav_pack = dict(packs[key])
                cav_pack["boxes"] = correct_boxes(packs[key]["boxes"], psi_n, t_n)
                true_boxes = correct_boxes(packs[key]["boxes"], psi_true, t_true)
                pair = _object_set(ego_pack, cav_pack)
                estimates = _alignformer_estimates(modules, pair, ablate, robust, [], refine, ())
                row = score_pair(pair, ego_pack["gt_ids"], cav_pack["gt_ids"], true_boxes, estimates, refine, fa_config, device, heading_lambda)
                row.update(frame=index, cav=key, sigma=sigma, shared=shared, bucket=bucket_of(shared))
                rows[f"{sigma:g}"].append(row)
        if (index // args.stride) % 50 == 0:
            print(f"  frame {index}/{len(dataset)}: {sum(len(v) for v in rows.values())} pair draws", flush=True)
    report = {"method": "association_accuracy_sweep", "split": str(args.split), "stride": args.stride, "seed": args.seed,
              "min_shared": args.min_shared, "gates_m": list(args.gates), "freealign": fa_config.to_dict() if hasattr(fa_config, "to_dict") else str(fa_config),
              "summary": {s: summarize_sigma(r) for s, r in rows.items()}, "pairs": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print_table(report["summary"])
    print(f"wrote {args.output}")


def print_table(summary: Dict) -> None:
    print(f"\n{'sigma':>5s} {'pairs':>5s} | {'nearest':>7s} {'soft':>6s} | {'hard P':>6s} {'hard R':>6s} {'engd':>5s} | {'FA P':>6s} {'FA R':>6s} {'FA ans':>6s} {'inl P':>6s} | "
          f"{'res soft':>8s} {'res hard':>8s} {'res FA':>7s}")
    for s, b in summary.items():
        r = b["residual"]
        print(f"{s:>5s} {b['pairs']:5d} | {b['nearest']['top1']:7.3f} {b['soft']['top1']:6.3f} | {b['hard']['precision']:6.3f} {b['hard']['recall']:6.3f} "
              f"{b['hard']['answered']:5.2f} | {b['freealign']['precision']:6.3f} {b['freealign']['recall']:6.3f} {b['freealign']['answered']:6.2f} "
              f"{b['freealign_inliers']['precision']:6.3f} | {r['residual_soft'].get('mean', float('nan')):8.3f} {r['residual_hard'].get('mean', float('nan')):8.3f} "
              f"{r['residual_freealign'].get('mean', float('nan')):7.3f}")


if __name__ == "__main__":
    main()
