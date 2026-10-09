"""Dump one V2X-Real frame for the paper's teaser: the ego's boxes, one CAV's
boxes as received under a localization-noise draw, the same boxes under the
true pose, and where the selected AlignFormer arm and FreeAlign put them.

The frame is built exactly as the noisy-AP sweep and the dense-bucket
diagnosis build it (same detector, same ROI, same seed formula, same noise
draw), so the picture is a frame of the reported experiment and not a
re-enactment. Reads val only; the figure is illustration, not a result.

    python scripts/dump_v2xreal_teaser_frame.py --config configs/v2xreal_detector.yaml \\
        --alignformer-config configs/alignformer_v2xreal.yaml --split $D/val \\
        --checkpoint outputs/v2xreal/stage2_B_boxes_only/best.pth \\
        --freealign-calibration outputs/v2xreal/freealign_calibration_valap_thr1_result.json \\
        --frame 468 --cav 2 --sigma 2.0 --device cpu --output outputs/v2xreal/teaser_frame_468.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "external" / "OpenCOOD"))

SELECTED_GATES_M = (6.0, 3.0, 1.5, 0.75, 0.5)
SELECTED_INLIER_M = 1.0
SELECTED_MIN_PAIRS = 3
SELECTED_WALD_LEVEL = 0.2
SELECTED_AGREE_TOLERANCE_M = 1.0
SELECTED_CONSENSUS_FLOOR = 4
LIDAR_POINTS_KEPT = 40000


def box_corners(boxes: np.ndarray) -> np.ndarray:
    """``(N, 4, 2)`` BEV corners of ``[x, y, z, h, w, l, yaw]`` boxes, counter-clockwise."""
    boxes = np.asarray(boxes, dtype=np.float64)
    half_l, half_w, yaw = boxes[:, 5] / 2, boxes[:, 4] / 2, boxes[:, 6]
    local = np.stack([np.stack([half_l, half_w], -1), np.stack([-half_l, half_w], -1),
                      np.stack([-half_l, -half_w], -1), np.stack([half_l, -half_w], -1)], 1)
    cos, sin = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    x = boxes[:, :1] + local[..., 0] * cos - local[..., 1] * sin
    y = boxes[:, 1:2] + local[..., 0] * sin + local[..., 1] * cos
    return np.stack([x, y], -1)


def _lidar_xy(base_data_dict, ego_id: str, kept: int, seed: int) -> Optional[List[List[float]]]:
    """The ego's raw point cloud in its own LiDAR frame, subsampled; ``None`` if the adapter keeps none."""
    points = base_data_dict[ego_id].get("lidar_np")
    if points is None:
        return None
    points = np.asarray(points)[:, :2]
    if points.shape[0] > kept:
        points = points[np.random.default_rng(seed).choice(points.shape[0], kept, replace=False)]
    return np.round(points, 2).tolist()


def _freealign_pose(ego_boxes, cav_boxes, fa_config):
    """FreeAlign's ``(psi, t)`` for the pair, or ``None`` where its graph is too small."""
    from embedding_aware_belt_fusion.alignformer.freealign import edge_features, mass_common_subgraph, robust_se2

    match = mass_common_subgraph(edge_features(ego_boxes, fa_config), edge_features(cav_boxes, fa_config), fa_config)
    if len(match) < fa_config.min_nodes:
        return None
    psi, t, _ = robust_se2(ego_boxes[match.ego_indices, :2], cav_boxes[match.cav_indices, :2], fa_config)
    return psi, t


def _np(tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _moved(boxes, pose) -> List[List[float]]:
    from embedding_aware_belt_fusion.alignformer.fusion import correct_boxes

    if pose is None:
        return _np(boxes).round(3).tolist()
    return _np(correct_boxes(boxes, pose[0], pose[1])).round(3).tolist()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--alignformer-config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--freealign-calibration", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True, help="dataset index, as the diagnosis reports it")
    parser.add_argument("--cav", required=True, help="the CAV key, as the diagnosis reports it")
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    import torch

    with torch.no_grad():
        _dump(parse_args(argv))


def _dump(args: argparse.Namespace) -> None:  # noqa: C901 - one frame, every stage of the sweep's pipeline
    import torch
    import yaml
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.utils.transformation_utils import x1_to_x2

    from embedding_aware_belt_fusion.alignformer.abstain import AbstentionConfig
    from embedding_aware_belt_fusion.alignformer.boxes import detect_agent
    from embedding_aware_belt_fusion.alignformer.cache import _build_detector
    from embedding_aware_belt_fusion.alignformer.draws import sweep_noisy_poses
    from embedding_aware_belt_fusion.alignformer.evaluate import _build_test_frame, _cav_content, _frame_identity, _pose_correction
    from embedding_aware_belt_fusion.alignformer.freealign_calibration import freealign_config_from_calibration
    from embedding_aware_belt_fusion.alignformer.fusion import correct_boxes
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import (
        ALIGNFORMER_IRLS, MAX_OBJECTS, _alignformer_estimates, _object_set, _roi_like_the_cache, _truncate_by_score,
        shared_object_count,
    )
    from embedding_aware_belt_fusion.alignformer.refine import AgreementConfig, RefineConfig, agreement_name
    from embedding_aware_belt_fusion.alignformer.robust import RobustSolveConfig
    from embedding_aware_belt_fusion.alignformer.splits import resolve_split
    from embedding_aware_belt_fusion.alignformer.stage2 import load_stage2
    from embedding_aware_belt_fusion.alignformer.v2xreal import build_dataset

    if args.split.name == "test":
        raise ValueError("the teaser is drawn from val; test is for reporting")
    device = torch.device(args.device)
    hypes = load_yaml(str(args.config), None)
    hypes["validate_dir"] = str(resolve_split(args.split, allow_test=False))
    dataset = build_dataset(hypes, visualize=False, train=False)
    detector = _build_detector(hypes, device)
    modules, checkpoint = load_stage2(args.checkpoint, device)
    model_config = yaml.safe_load(args.alignformer_config.read_text())
    ablate = checkpoint["message_content"] == "boxes_only"
    robust = RobustSolveConfig(mode="huber", iterations=2, min_evidence=3.0)
    refine = RefineConfig(mode="icp_ransac", gates_m=SELECTED_GATES_M, min_pairs=SELECTED_MIN_PAIRS,
                          inlier_m=SELECTED_INLIER_M, weighted=True)
    decision = AbstentionConfig(mode="abstain", level=SELECTED_WALD_LEVEL)
    agreement = AgreementConfig(tolerance_m=SELECTED_AGREE_TOLERANCE_M, consensus_floor=SELECTED_CONSENSUS_FLOOR)
    fa_config = freealign_config_from_calibration(args.freealign_calibration)
    lidar_range = hypes["preprocess"]["cav_lidar_range"]
    output_size = int(model_config["model"]["output_size"])
    base_seed = int(model_config["train"]["seed"]) + args.seed

    scenario, timestamp = _frame_identity(dataset, args.frame)
    sample, poses, ego_pose, base = _build_test_frame(dataset, args.frame, scenario, timestamp)
    batch = dataset.collate_batch_test([sample])
    packs, transforms = {}, {}
    for key, entry in batch.items():
        found = detect_agent(detector, _cav_content(entry, device), dataset.post_processor)
        roi = _roi_like_the_cache(found, lidar_range, output_size)
        boxes, scores, roi, gt_ids = _truncate_by_score(found, roi, MAX_OBJECTS)
        packs[key] = {"boxes": boxes, "scores": scores, "roi": roi, "gt_ids": gt_ids}
        transforms[key] = entry["transformation_matrix"].to(device)
    if args.cav not in packs:
        raise KeyError(f"frame {args.frame} has agents {sorted(packs)}, not {args.cav!r}")
    ego_pack, cav_keys = packs["ego"], sorted(k for k in packs if k != "ego")
    drawn = sweep_noisy_poses(poses, cav_keys, sigma=args.sigma, seed=base_seed, frame=args.frame)
    noisy_transform = torch.as_tensor(x1_to_x2(drawn[args.cav], ego_pose), dtype=torch.float32, device=device)
    psi_n, t_n = _pose_correction(noisy_transform)
    psi_true, t_true = _pose_correction(transforms[args.cav])
    cav_pack = dict(packs[args.cav])
    cav_pack["boxes"] = correct_boxes(packs[args.cav]["boxes"], psi_n, t_n)
    true_boxes = correct_boxes(packs[args.cav]["boxes"], psi_true, t_true)
    pair = _object_set(ego_pack, cav_pack)
    estimates = _alignformer_estimates(modules, pair, ablate, robust, [decision], refine, [agreement])
    ours = estimates[agreement_name(decision.name, agreement.tolerance_m, agreement.consensus_floor)]
    soft = estimates[ALIGNFORMER_IRLS]
    ego_boxes, cav_boxes = pair["ego_boxes"][0], pair["cav_boxes"][0]
    fa_pose = _freealign_pose(ego_boxes, cav_boxes, fa_config)

    def residual(boxes_list) -> float:
        return float(np.linalg.norm(np.asarray(boxes_list)[:, :2] - _np(true_boxes)[:, :2], axis=-1).mean())

    record: Dict = {
        "method": "v2xreal_teaser_frame", "split": str(args.split), "frame": args.frame, "scenario": scenario,
        "timestamp": timestamp, "cav_key": args.cav, "sigma_m": args.sigma, "seed": args.seed,
        "shared": shared_object_count(ego_pack["gt_ids"], packs[args.cav]["gt_ids"]),
        "lidar_range": list(lidar_range), "box_layout": "x y z h w l yaw (ego LiDAR frame)",
        "ego": {"boxes": _np(ego_boxes).round(3).tolist(), "scores": _np(ego_pack["scores"]).round(3).tolist(),
                "gt_ids": list(ego_pack["gt_ids"]), "lidar_xy": _lidar_xy(base, next(k for k, v in base.items() if v["ego"]), LIDAR_POINTS_KEPT, args.seed)},
        "cav": {"gt_ids": list(packs[args.cav]["gt_ids"]), "scores": _np(packs[args.cav]["scores"]).round(3).tolist(),
                "boxes_received": _np(cav_boxes).round(3).tolist(), "boxes_true": _np(true_boxes).round(3).tolist(),
                "boxes_soft_proposal": _moved(cav_boxes, (soft.psi[0], soft.t[0])),
                "boxes_alignformer": _moved(cav_boxes, (ours.psi[0], ours.t[0])),
                "boxes_freealign": _moved(cav_boxes, fa_pose), "freealign_answered": fa_pose is not None},
    }
    record["residual_m"] = {k: residual(record["cav"][f"boxes_{k}"]) for k in ("received", "soft_proposal", "alignformer", "freealign")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record) + "\n")
    print(f"{scenario}/{timestamp} cav {args.cav}: shared {record['shared']}, residuals {record['residual_m']}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
