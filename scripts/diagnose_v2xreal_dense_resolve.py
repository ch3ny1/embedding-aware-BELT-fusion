"""Where does the exact re-solve lose to FreeAlign on dense pairs at large noise?

On V2X-Real test, pairs sharing three or more objects are two thirds of the
frames, and at sigma 2 m the re-solve sits at .370 AP@0.7 against FreeAlign's
.395 with an answered residual of 0.55 m. Two causes are possible and this
script separates them on val, pair by pair, using the ground-truth id every
cached detection carries:

- **wrong correspondences**: the ICP gates admit a lane neighbour, and an
  unweighted least-squares fit is pulled by it. Measured as the precision of
  the hard pairs at each gate, and by re-solving the same pair over ONLY its
  correct pairs and over the oracle correspondence.
- **the fit**: FreeAlign fits RANSAC (512 minimal samples, 1 m inlier gate)
  over its matches; the re-solve fits all its pairs at once. Measured by
  fitting RANSAC over the re-solve's own pairs, and the re-solve's exact fit
  over FreeAlign's matches.

Every number is the residual on the CAV's boxes after correction, in metres
(mean centre distance between the corrected noisy boxes and the boxes under
the true pose), so rotation counts through its lever arm as it does in AP.

Usage::

    python scripts/diagnose_v2xreal_dense_resolve.py --config configs/v2xreal_detector.yaml \\
        --alignformer-config configs/alignformer_v2xreal.yaml --split .../v2x-real/val \\
        --checkpoint outputs/v2xreal/stage2_B_boxes_only/best.pth \\
        --freealign-calibration outputs/v2xreal/freealign_calibration_valap_thr1_result.json \\
        --sigmas 0.8 2.0 --stride 3 --output outputs/v2xreal/dense_resolve_diagnosis_val_result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "external" / "OpenCOOD"))

MIN_SHARED = 3
RESIDUAL_BARS_M = (0.5, 1.0)


def pair_precision_recall(ego_ids: Sequence[Optional[str]], cav_ids: Sequence[Optional[str]],
                          ego_idx: Sequence[int], cav_idx: Sequence[int]) -> Tuple[float, float, int, int]:
    """Of the proposed (ego, cav) pairs, the fraction whose ids agree; of the shared ids, the fraction proposed."""
    shared = {g for g in ego_ids if g is not None} & {g for g in cav_ids if g is not None}
    correct = sum(1 for e, c in zip(ego_idx, cav_idx) if ego_ids[e] is not None and ego_ids[e] == cav_ids[c])
    proposed = len(ego_idx)
    precision = correct / proposed if proposed else float("nan")
    recall = correct / len(shared) if shared else float("nan")
    return precision, recall, correct, proposed


def box_residual_m(noisy_boxes: torch.Tensor, true_boxes: torch.Tensor, psi: torch.Tensor, t: torch.Tensor) -> float:
    """Mean centre distance between the noisy boxes corrected by ``(psi, t)`` and the true ones."""
    from embedding_aware_belt_fusion.alignformer.fusion import correct_boxes

    corrected = correct_boxes(noisy_boxes, psi, t)
    return float((corrected[:, :2] - true_boxes[:, :2]).norm(dim=-1).mean())


def _icp_stages(psi, t, ego_boxes, cav_boxes, gates, min_pairs, heading_lambda, ego_ids, cav_ids):
    """Replay the re-solve's gates, recording each stage's pairs and precision."""
    from embedding_aware_belt_fusion.alignformer.refine import _exact_solve, _moved, mutual_nearest

    stages, last = [], None
    for gate in gates:
        e_idx, c_idx = mutual_nearest(ego_boxes[:, :2], _moved(cav_boxes[:, :2], psi, t), gate)
        if e_idx.numel() < min_pairs:
            break
        precision, recall, correct, proposed = pair_precision_recall(ego_ids, cav_ids, e_idx.tolist(), c_idx.tolist())
        psi, t = _exact_solve(ego_boxes, cav_boxes, e_idx, c_idx, heading_lambda)
        stages.append({"gate": gate, "pairs": proposed, "correct": correct, "precision": precision, "recall": recall})
        last = (e_idx, c_idx)
    return psi, t, stages, last


def _correct_only(ego_ids, cav_ids, e_idx, c_idx):
    keep = [k for k, (e, c) in enumerate(zip(e_idx.tolist(), c_idx.tolist())) if ego_ids[e] is not None and ego_ids[e] == cav_ids[c]]
    return e_idx[keep], c_idx[keep]


def _oracle_pairs(ego_ids, cav_ids, device):
    from embedding_aware_belt_fusion.alignformer.oracle import oracle_assignment

    assignment = oracle_assignment(ego_ids, cav_ids, device=device)[0]
    rows, cols = torch.nonzero(assignment > 0.5, as_tuple=True)
    return rows, cols


def diagnose_pair(pair, ego_ids, cav_ids, true_cav_boxes, estimates, modules, refine, freealign_config, device) -> Dict:
    """One ego-CAV pair at one draw: residuals of every solve variant."""
    from embedding_aware_belt_fusion.alignformer.freealign import edge_features, mass_common_subgraph, robust_se2
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import ALIGNFORMER_IRLS
    from embedding_aware_belt_fusion.alignformer.refine import _exact_solve, refined_name

    ego_boxes, cav_boxes = pair["ego_boxes"][0], pair["cav_boxes"][0]
    heading_lambda = float(getattr(modules["pose"], "heading_lambda", 2.0))
    irls = estimates[ALIGNFORMER_IRLS]
    out: Dict = {"n_ego": int(ego_boxes.shape[0]), "n_cav": int(cav_boxes.shape[0])}
    out["residual_uncorrected"] = box_residual_m(cav_boxes, true_cav_boxes, torch.zeros((), device=device), torch.zeros(2, device=device))
    out["residual_irls"] = box_residual_m(cav_boxes, true_cav_boxes, irls.psi[0], irls.t[0])
    psi, t, stages, last = _icp_stages(irls.psi[0], irls.t[0], ego_boxes, cav_boxes, refine.gates_m, refine.min_pairs, heading_lambda, ego_ids, cav_ids)
    out["icp_stages"] = stages
    out["residual_icp"] = box_residual_m(cav_boxes, true_cav_boxes, psi, t)
    shipped = estimates[refined_name(ALIGNFORMER_IRLS)]
    out["residual_icp_shipped"] = box_residual_m(cav_boxes, true_cav_boxes, shipped.psi[0], shipped.t[0])
    if last is not None:
        e_idx, c_idx = last
        ce, cc = _correct_only(ego_ids, cav_ids, e_idx, c_idx)
        if ce.numel() >= 2:
            p2, t2 = _exact_solve(ego_boxes, cav_boxes, ce, cc, heading_lambda)
            out["residual_icp_correct_pairs_only"] = box_residual_m(cav_boxes, true_cav_boxes, p2, t2)
        if e_idx.numel() >= 2:
            p3, t3, inliers = robust_se2(ego_boxes[e_idx, :2], cav_boxes[c_idx, :2], freealign_config)
            out["residual_icp_pairs_ransac"] = box_residual_m(cav_boxes, true_cav_boxes, p3, t3)
            out["icp_ransac_inliers"] = int(inliers.sum())
    rows, cols = _oracle_pairs(ego_ids, cav_ids, device)
    out["oracle_pairs"] = int(rows.numel())
    if rows.numel() >= 2:
        p4, t4 = _exact_solve(ego_boxes, cav_boxes, rows, cols, heading_lambda)
        out["residual_oracle_pairs_exact"] = box_residual_m(cav_boxes, true_cav_boxes, p4, t4)
        p5, t5, _ = robust_se2(ego_boxes[rows, :2], cav_boxes[cols, :2], freealign_config)
        out["residual_oracle_pairs_ransac"] = box_residual_m(cav_boxes, true_cav_boxes, p5, t5)
    match = mass_common_subgraph(edge_features(ego_boxes, freealign_config), edge_features(cav_boxes, freealign_config), freealign_config)
    out["fa_pairs"] = len(match)
    if len(match) >= freealign_config.min_nodes:
        precision, recall, correct, proposed = pair_precision_recall(ego_ids, cav_ids, match.ego_indices.tolist(), match.cav_indices.tolist())
        out["fa_precision"], out["fa_recall"] = precision, recall
        p6, t6, inliers = robust_se2(ego_boxes[match.ego_indices, :2], cav_boxes[match.cav_indices, :2], freealign_config)
        out["residual_freealign"] = box_residual_m(cav_boxes, true_cav_boxes, p6, t6)
        out["fa_inliers"] = int(inliers.sum())
        p7, t7 = _exact_solve(ego_boxes, cav_boxes, match.ego_indices, match.cav_indices, heading_lambda)
        out["residual_fa_pairs_exact"] = box_residual_m(cav_boxes, true_cav_boxes, p7, t7)
    return out


def summarize(rows: List[Dict]) -> Dict:
    """Medians, means and tail fractions per residual kind; stage precision pooled."""
    keys = sorted({k for r in rows for k in r if k.startswith("residual_")})
    out: Dict = {"pairs": len(rows), "residuals": {}}
    for k in keys:
        v = np.array([r[k] for r in rows if k in r], dtype=float)
        out["residuals"][k] = {"n": int(v.size), "median": float(np.median(v)), "mean": float(v.mean()),
                               **{f"frac_over_{b:g}m": float((v > b).mean()) for b in RESIDUAL_BARS_M}}
    stages: Dict[int, List[Dict]] = {}
    for r in rows:
        for i, s in enumerate(r.get("icp_stages", [])):
            stages.setdefault(i, []).append(s)
    out["icp_stage_precision"] = {str(i): {"gate": s[0]["gate"], "pairs_mean": float(np.mean([x["pairs"] for x in s])),
                                           "precision_pooled": float(sum(x["correct"] for x in s) / max(sum(x["pairs"] for x in s), 1)),
                                           "recall_mean": float(np.nanmean([x["recall"] for x in s]))}
                                  for i, s in stages.items()}
    fa = [r for r in rows if "fa_precision" in r]
    out["freealign"] = {"answered": len(fa), "precision_mean": float(np.mean([r["fa_precision"] for r in fa])) if fa else None,
                        "recall_mean": float(np.nanmean([r["fa_recall"] for r in fa])) if fa else None}
    engaged = [r for r in rows if r.get("icp_stages")]
    out["icp_engaged"] = len(engaged)
    return out


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--alignformer-config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--freealign-calibration", type=Path, required=True)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.8, 2.0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:  # noqa: C901 - one evaluation loop
    args = parse_args(argv)
    import yaml
    from opencood.hypes_yaml.yaml_utils import load_yaml

    from embedding_aware_belt_fusion.alignformer.robust import RobustSolveConfig
    from embedding_aware_belt_fusion.alignformer.boxes import detect_agent
    from embedding_aware_belt_fusion.alignformer.cache import _build_detector
    from embedding_aware_belt_fusion.alignformer.draws import sweep_noisy_poses
    from embedding_aware_belt_fusion.alignformer.evaluate import _build_test_frame, _cav_content, _frame_identity, _pose_correction
    from embedding_aware_belt_fusion.alignformer.freealign_calibration import freealign_config_from_calibration
    from embedding_aware_belt_fusion.alignformer.fusion import correct_boxes
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import (
        MAX_OBJECTS, _alignformer_estimates, _object_set, _roi_like_the_cache, _truncate_by_score, shared_object_count,
    )
    from embedding_aware_belt_fusion.alignformer.refine import RefineConfig
    from embedding_aware_belt_fusion.alignformer.splits import resolve_split
    from embedding_aware_belt_fusion.alignformer.stage2 import load_stage2
    from embedding_aware_belt_fusion.alignformer.v2xreal import build_dataset
    from opencood.utils.transformation_utils import x1_to_x2

    if args.split.name == "test":
        raise ValueError("this diagnostic selects a fix; it reads val only")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hypes = load_yaml(str(args.config), None)
    hypes["validate_dir"] = str(resolve_split(args.split, allow_test=False))
    dataset = build_dataset(hypes, visualize=False, train=False)
    detector = _build_detector(hypes, device)
    modules, checkpoint = load_stage2(args.checkpoint, device)
    model_config = yaml.safe_load(args.alignformer_config.read_text())
    ablate = checkpoint["message_content"] == "boxes_only"
    robust = RobustSolveConfig(mode="huber", iterations=2, min_evidence=3.0)
    refine = RefineConfig(mode="icp")
    fa_config = freealign_config_from_calibration(args.freealign_calibration)
    lidar_range = hypes["preprocess"]["cav_lidar_range"]
    output_size = int(model_config["model"]["output_size"])
    base_seed = int(model_config["train"]["seed"]) + args.seed

    rows: Dict[str, List[Dict]] = {f"{s:g}": [] for s in args.sigmas}
    for index in range(0, len(dataset), args.stride):
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
                if shared_object_count(ego_pack["gt_ids"], packs[key]["gt_ids"]) < MIN_SHARED:
                    continue
                noisy_transform = torch.as_tensor(x1_to_x2(drawn[key], ego_pose), dtype=torch.float32, device=device)
                psi_n, t_n = _pose_correction(noisy_transform)
                psi_true, t_true = _pose_correction(transforms[key])
                cav_pack = dict(packs[key])
                cav_pack["boxes"] = correct_boxes(packs[key]["boxes"], psi_n, t_n)
                true_boxes = correct_boxes(packs[key]["boxes"], psi_true, t_true)
                pair = _object_set(ego_pack, cav_pack)
                estimates = _alignformer_estimates(modules, pair, ablate, robust, [], refine, ())
                row = diagnose_pair(pair, ego_pack["gt_ids"], cav_pack["gt_ids"], true_boxes, estimates, modules, refine, fa_config, device)
                row.update(frame=index, cav=key, sigma=sigma, shared=shared_object_count(ego_pack["gt_ids"], packs[key]["gt_ids"]))
                rows[f"{sigma:g}"].append(row)
        if (index // args.stride) % 50 == 0:
            print(f"  frame {index}/{len(dataset)}: {sum(len(v) for v in rows.values())} dense pairs", flush=True)
    report = {"method": "v2xreal_dense_resolve_diagnosis", "split": str(args.split), "stride": args.stride, "seed": args.seed,
              "min_shared": MIN_SHARED, "summary": {s: summarize(r) for s, r in rows.items()}, "pairs": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    for s, summ in report["summary"].items():
        print(f"\nsigma {s}: {summ['pairs']} dense pairs, ICP engaged {summ['icp_engaged']}")
        for k, v in summ["residuals"].items():
            print(f"  {k:36s} n {v['n']:4d} median {v['median']:.3f} mean {v['mean']:.3f} >0.5m {v['frac_over_0.5m']:.2f} >1m {v['frac_over_1m']:.2f}")
        print("  icp stages:", summ["icp_stage_precision"]); print("  freealign:", summ["freealign"])
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
