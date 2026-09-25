"""AlignFormer evaluation CLI: ``--method`` selects the fusion pipeline,
``--metric`` selects what to report (ruling R1).

Five metrics are wired up:

- ``--metric ap`` needs ``--config`` (an OpenCOOD detector hypes yaml) and
  ``--split``, runs the full detect/correct/fuse pipeline over an OPV2V split
  directory, and reports AP. This is the P0 gate.
- ``--metric top1`` needs ``--config`` (``configs/alignformer.yaml``) and
  ``--checkpoint`` (a stage-1 checkpoint), and reports cross-agent association
  Top-1 on the held-out *validation scenarios* carved out of the train split.
  This is the P1 gate. It touches neither OpenCOOD nor the raw point clouds:
  everything it needs is in the detection/ROI cache.

- ``--metric pose`` needs ``configs/alignformer.yaml`` and one or more stage-2
  checkpoints, and reports translation and yaw MAE against the **predict-zero**
  baseline over a sigma sweep on the same scenario-disjoint validation
  scenarios. This is the P2 gate.
- ``--metric noisy_ap`` needs an OpenCOOD detector hypes yaml, ``--split``, a
  stage-2 checkpoint and ``--alignformer-config``, and reports fused AP under
  localization error for three conditions at each sigma: plain late fusion on
  the noisy pose, AlignFormer-corrected, and the true-pose oracle. This is the
  number the method exists for; see ``alignformer.noisy_fusion``.
- ``--metric shrinkage`` needs ``configs/alignformer.yaml`` and one stage-2
  checkpoint, and writes the calibration ``--metric pose`` and
  ``--metric noisy_ap`` then consume via ``--shrinkage``. It is measured on the
  validation split at sigma = 0 only -- the one noise level where the true
  correction is exactly the identity, so the emitted correction *is* the
  estimator's residual. See ``alignformer.shrinkage``.

``ransac`` is a later task's work; the dispatch tables below exist so it slots
in as a new entry without reshaping the CLI.

``run_late_fusion_clean`` reuses OpenCOOD's own ``LateFusionDataset`` (the
same dataset class ``alignformer/cache.py`` and
``evaluation/belt_fusion.py`` use) so ego selection, the CAV communication
range, and ground-truth construction (``post_processor.generate_gt_bbx``)
match the official evaluation protocol exactly. Per-agent detection and
cross-agent fusion go through this project's own
``boxes.detect_agent`` / ``fusion.correct_detections`` / ``fusion.late_fuse``
instead of OpenCOOD's monolithic ``post_process`` -- the P0 gate is exactly
the claim that this decomposition reproduces the same number.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor

from embedding_aware_belt_fusion.alignformer.boxes import detect_agent
from embedding_aware_belt_fusion.alignformer.cache import frame_seed
from embedding_aware_belt_fusion.alignformer.freealign import (
    DEFAULT_MIN_NODES,
    EDGE_DISTANCE,
    EDGE_DISTANCE_YAW,
)
from embedding_aware_belt_fusion.alignformer.fusion import (
    average_precision,
    average_precision_from_matches,
    correct_detections,
    late_fuse,
    match_frames,
)
from embedding_aware_belt_fusion.alignformer.abstain import AbstentionConfig
from embedding_aware_belt_fusion.alignformer.robust import (
    DEFAULT_ITERATIONS,
    DEFAULT_MIN_EVIDENCE,
    ROBUST_MODES,
    ROBUST_NONE,
    RobustSolveConfig,
)

# AP is reported at all three; only AP@0.7 is the P0 gate.
_AP_IOU_THRESHOLDS = (0.3, 0.5, 0.7)
_PROGRESS_INTERVAL = 200

# Spec section 4: the P1 gate is cross-agent Top-1 at or above this, measured
# at stage 1's own training-noise maximum (train.stage1_max_xy_std).
P1_TOP1_GATE = 0.85
# Reported alongside the gate. Stage 1 never trains above its own maximum, so
# 1.0 and 2.0 m are out-of-distribution here; they are the levels stage 2's
# curriculum reaches, and they are what separates the embedding's contribution
# from geometry's -- at sub-metre error, nearest-centre matching alone is
# already near-perfect, so the gate's own sigma cannot make that distinction.
_TOP1_CONTEXT_SIGMAS = (0.0, 1.0, 2.0)
_TOP1_BATCH_SIZE = 64
_DEFAULT_TOP1_WORKERS = 10

# Spec section 4 / task 14: the P2 gate is head B's yaw MAE strictly below the
# predict-zero value at every NON-ZERO sigma. Predicting the identity
# correction gives E|dpsi| = sigma_yaw * sqrt(2/pi) for zero-mean Gaussian yaw
# noise; at sigma = 0 that baseline is 0 and nothing can be strictly below it,
# so sigma = 0 is reported as context (does the model damage the clean case?)
# rather than gated on.
_DEFAULT_POSE_SWEEP = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0)
_DEFAULT_POSE_SEEDS = 3
# Seeds are consecutive from here, so a sweep is reproducible by name.
_POSE_SEED_BASE = 1000
_HALF_NORMAL_MEAN = math.sqrt(2.0 / math.pi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True,
        help="OpenCOOD detector hypes yaml for --metric ap; configs/alignformer.yaml for top1",
    )
    parser.add_argument(
        "--split", type=Path, default=None,
        help="OPV2V split directory, e.g. .../OPV2V/test. Required by --metric ap only; "
             "--metric top1 takes its split from the config's scenario-disjoint val slice.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, nargs="+", default=None,
        help="stage-1 checkpoint for --metric top1; one or more stage-2 checkpoints for "
             "--metric pose (each is measured over the whole sweep, so one run produces "
             "the whole head x message-content table); one stage-2 checkpoint for "
             "--metric noisy_ap. --metric ap reads the detector checkpoint named in its "
             "own hypes yaml instead.",
    )
    parser.add_argument(
        "--method", choices=("late_fusion_clean",), default="late_fusion_clean",
        help="Fusion pipeline to evaluate. Ignored by --metric top1, which has no "
             "fusion step: it scores the correspondence, not fused boxes.",
    )
    parser.add_argument(
        "--metric", choices=("ap", "top1", "pose", "noisy_ap", "shrinkage"), default="ap",
        help="What to report.",
    )
    parser.add_argument(
        "--shrinkage", type=Path, default=None,
        help="calibration JSON from --metric shrinkage. Given, --metric pose and "
             "--metric noisy_ap shrink every estimated correction by "
             "max(0, 1 - tau^2 / |x|^2) before scoring or fusing it. Calibrate on "
             "the validation split only.",
    )
    parser.add_argument(
        "--sweep", type=float, nargs="+", default=list(_DEFAULT_POSE_SWEEP),
        help="localization-noise levels in metres (yaw noise in degrees matches); "
             "--metric pose and --metric noisy_ap",
    )
    parser.add_argument(
        "--seeds", type=int, default=_DEFAULT_POSE_SEEDS,
        help="independent noise draws per sigma; --metric pose only",
    )
    parser.add_argument(
        "--alignformer-config", type=Path, default=Path("configs/alignformer.yaml"),
        help="AlignFormer model/data config; --metric noisy_ap only (--config there is "
             "the OpenCOOD detector hypes yaml)",
    )
    parser.add_argument(
        "--oracle-match", action="store_true",
        help="--metric noisy_ap only: also fuse the two oracle-correspondence "
             "conditions, i.e. the same estimator with the learned Sinkhorn "
             "correspondence replaced by the ground-truth assignment (weighted "
             "uniformly, and by the checkpoint's own inverse-variance model). "
             "This is the ceiling on what any matching improvement can deliver.",
    )
    parser.add_argument(
        "--freealign", action="store_true",
        help="--metric noisy_ap only: also fuse the FreeAlign condition -- this "
             "project's REIMPLEMENTATION of Lei et al., ICRA 2024 "
             "(arXiv 2405.02965), not the authors' code, which cannot be run "
             "here (see alignformer.freealign). It is applied to the same late "
             "fusion, the same detections and the same evaluator as "
             "--metric noisy_ap's alignformer arm, so the only thing that "
             "differs between the two rows is the alignment algorithm. That "
             "late-fusion pairing is the row their own paper never reports.",
    )
    parser.add_argument(
        "--freealign-edge-feature", choices=(EDGE_DISTANCE, EDGE_DISTANCE_YAW),
        default=EDGE_DISTANCE,
        help="the paper's training-free edge attribute is the relative distance "
             "alone; their shipped greedy_match.py adds the relative yaw. "
             "Selected on validation, never on test.",
    )
    parser.add_argument(
        "--freealign-min-nodes", type=int, default=DEFAULT_MIN_NODES,
        help="FreeAlign discards a collaborative message whose common subgraph "
             "has fewer nodes than this; 3 is their shipped default and the "
             "count below which a distance graph cannot determine SE(2).",
    )
    parser.add_argument(
        "--robust-solve", choices=ROBUST_MODES, default=ROBUST_NONE,
        help="--metric shrinkage: calibrate tau on the IRLS arm's own residuals "
             "instead of the deployed solve's. --metric noisy_ap: also fuse the "
             "alignformer_irls condition, "
             "i.e. the deployed estimator with an IRLS re-weighting loop wrapped "
             "around its closed-form solve (alignformer.robust). The deployed "
             "alignformer arm is produced unchanged in the same run, so the two "
             "differ in the pose solve alone. 'none' adds nothing.",
    )
    parser.add_argument(
        "--robust-iterations", type=int, default=DEFAULT_ITERATIONS,
        help="re-weight/re-solve rounds in --robust-solve; chosen on validation",
    )
    parser.add_argument(
        "--robust-min-evidence", type=float, default=DEFAULT_MIN_EVIDENCE,
        help="effective matched correspondences below which --robust-solve is "
             "suppressed and the single solve is kept untouched. The sparse "
             "slice is where AlignFormer beats a distance-graph method, and a "
             "robust loop with three effective points will trim its way to "
             "nonsense; chosen on validation, never on test.",
    )
    parser.add_argument(
        "--abstain-arm", action="append", default=None, metavar="MODE[:THRESHOLD]",
        help="--metric noisy_ap only, repeatable: add one per-pair abstention "
             "arm (alignformer.abstain) built on the --robust-solve arm. "
             "'abstain:C' zeroes the correction at or below Wald statistic C; "
             "'per_pair' applies the James-Stein positive-part factor with the "
             "pair's own covariance in place of the global tau; 'both:C' does "
             "both. Each arm REPLACES the --shrinkage calibration rather than "
             "stacking on it, so the deployed arms in the same run still "
             "reproduce bit for bit.",
    )
    parser.add_argument(
        "--ap-seeds", type=int, default=1,
        help="--metric noisy_ap only: how many INDEPENDENT noise draws to run "
             "per sigma. Every condition inside one draw sees the same "
             "perturbed poses, so the per-seed difference between conditions "
             "is paired and the error bar on it is far tighter than either "
             "condition's own spread. The first seed is the model config's own "
             "train seed, so 1 reproduces every single-seed sweep on this "
             "branch bit for bit. sigma = 0 perturbs nothing and is drawn once "
             "however many seeds are asked for.",
    )
    parser.add_argument("--output", type=Path, required=True, help="destination JSON result file")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-frames", type=int, default=None, help="limit frames, for smoke-testing")
    parser.add_argument(
        "--num-workers", type=int, default=_DEFAULT_TOP1_WORKERS,
        help="cache-reading data loader workers; --metric top1 only",
    )

    args = parser.parse_args()
    required = {
        "ap": ("split",),
        "top1": ("checkpoint",),
        "pose": ("checkpoint",),
        "noisy_ap": ("split", "checkpoint"),
        "shrinkage": ("checkpoint",),
    }[args.metric]
    for name in required:
        if getattr(args, name) is None:
            parser.error(f"--metric {args.metric} requires --{name}")
    if args.metric in ("top1", "noisy_ap", "shrinkage") and len(args.checkpoint) != 1:
        parser.error(f"--metric {args.metric} takes exactly one --checkpoint")
    return args


def _frame_identity(dataset, index: int) -> Tuple[str, str]:
    """Return ``(scenario, timestamp)`` for a global dataset index.

    Mirrors ``BaseDataset.retrieve_base_data``'s own index accounting so the
    seed derived here refers to the same frame OpenCOOD is about to load.
    """
    scenario_index = next(i for i, end in enumerate(dataset.len_record) if index < end)
    scenario_database = dataset.scenario_database[scenario_index]
    timestamp_index = (
        index if scenario_index == 0 else index - dataset.len_record[scenario_index - 1]
    )
    timestamp_key = dataset.return_timestamp_key(scenario_database, timestamp_index)
    timestamps = next(iter(scenario_database.values()))
    scenario_name = Path(timestamps[timestamp_key]["yaml"]).parents[1].name
    return scenario_name, timestamp_key


def _build_test_frame(
    dataset, index: int, scenario: str, timestamp: str
) -> Tuple[Dict[str, dict], Dict[str, List[float]], List[float], Dict]:
    """Reimplement ``LateFusionDataset.get_item_test`` with a reproducible seed.

    Returns ``(frame, lidar_poses, ego_lidar_pose, base_data_dict)``. The raw
    poses come back alongside the processed frame because
    ``collate_batch_test`` keeps only the fields it knows about, and the noisy
    sweep needs the un-collated pose to perturb -- re-reading it would mean a
    second ``retrieve_base_data`` and a second point-cloud decode per frame.
    The raw ``base_data_dict`` comes back for the same reason: the fused-AP
    sweep scores its predictions against a second, intermediate-fusion ground
    truth as well, and that one is built from the un-processed per-CAV object
    lists rather than from the collated frame.

    OpenCOOD's own ``__getitem__`` gives no hook to reseed between agents, so
    this replicates its ego-selection / communication-range / per-agent
    processing loop directly, calling ``dataset.get_item_single_car`` (the
    exact method the ROI cache builder uses) and reseeding immediately before
    each call -- the same point (and the same seed formula) ``alignformer.
    cache._cache_one_frame`` reseeds at.
    """
    import opencood.data_utils.datasets as opencood_datasets
    from opencood.utils.transformation_utils import x1_to_x2

    base_data_dict = dataset.retrieve_base_data(index)

    ego_id, ego_lidar_pose = None, None
    for cav_id, cav_content in base_data_dict.items():
        if cav_content["ego"]:
            ego_id, ego_lidar_pose = cav_id, cav_content["params"]["lidar_pose"]
            break
    if ego_id is None:
        raise RuntimeError(f"frame {index} ({scenario}/{timestamp}) has no ego vehicle")

    frame: Dict[str, dict] = {}
    poses: Dict[str, List[float]] = {}
    for cav_id, selected_cav_base in base_data_dict.items():
        cav_pose = selected_cav_base["params"]["lidar_pose"]
        distance = math.hypot(cav_pose[0] - ego_lidar_pose[0], cav_pose[1] - ego_lidar_pose[1])
        if distance > opencood_datasets.COM_RANGE:
            continue

        np.random.seed(frame_seed(scenario, cav_id, timestamp))
        processed = dataset.get_item_single_car(selected_cav_base)
        processed["transformation_matrix"] = x1_to_x2(cav_pose, ego_lidar_pose)
        key = "ego" if cav_id == ego_id else cav_id
        frame[key] = processed
        poses[key] = list(cav_pose)

    return frame, poses, list(ego_lidar_pose), base_data_dict


def _cav_content(entry: Dict, device) -> Dict:
    """Move one collated agent's fields onto ``device`` for ``detect_agent``."""
    return {
        "processed_lidar": {k: v.to(device) for k, v in entry["processed_lidar"].items()},
        "anchor_box": entry["anchor_box"].to(device),
        "object_bbx_center": entry["object_bbx_center"].to(device),
        "object_bbx_mask": entry["object_bbx_mask"].to(device),
        "object_ids": entry["object_ids"],
    }


def _pose_correction(transformation_matrix: Tensor) -> Tuple[Tensor, Tensor]:
    """Extract the SE(2) ``(psi, t)`` a 4x4 CAV-to-ego matrix implies.

    For the clean condition ``transformation_matrix`` already *is* the true
    CAV-to-ego pose transform (``x1_to_x2``), so this correction is exact, not
    estimated -- there is no localization error to remove yet. The ego
    agent's own matrix is the identity, so its correction is a no-op (see
    ``test_identity_correction_is_a_no_op``), letting every agent in a frame
    go through :func:`correct_detections` uniformly.
    """
    psi = torch.atan2(transformation_matrix[1, 0], transformation_matrix[0, 0])
    t = transformation_matrix[:2, 3]
    return psi, t


def run_late_fusion_clean(
    dataset, detector, postprocessor, device, max_frames: int = None
) -> Tuple[List[Tuple[Tensor, Tensor]], List[Tensor]]:
    """Detect, correct into the ego frame, and fuse every frame of a split."""
    from opencood.utils import box_utils

    predictions: List[Tuple[Tensor, Tensor]] = []
    ground_truth: List[Tensor] = []

    frame_count = len(dataset) if max_frames is None else min(max_frames, len(dataset))
    started = time.time()

    for index in range(frame_count):
        scenario, timestamp = _frame_identity(dataset, index)
        sample, _, _, _ = _build_test_frame(dataset, index, scenario, timestamp)
        batch = dataset.collate_batch_test([sample])

        corrected = []
        for cav_id, entry in batch.items():
            cav_content = _cav_content(entry, device)
            with torch.no_grad():
                detections = detect_agent(detector, cav_content, postprocessor)
            psi, t = _pose_correction(entry["transformation_matrix"].to(device))
            corrected.append(correct_detections(detections, psi, t))

        boxes, scores = late_fuse(corrected, postprocessor.params["nms_thresh"])
        predictions.append((boxes.detach().cpu(), scores.detach().cpu()))

        gt_corners = postprocessor.generate_gt_bbx(batch)
        gt_boxes = box_utils.corner_to_center(
            gt_corners.detach().cpu().numpy(), order=postprocessor.params["order"]
        )
        ground_truth.append(torch.from_numpy(gt_boxes).float())

        if (index + 1) % _PROGRESS_INTERVAL == 0:
            rate = (index + 1) / (time.time() - started)
            print(f"  {index + 1}/{frame_count} frames  {rate:.2f} fr/s", flush=True)

    return predictions, ground_truth


def _build_detector(hypes: Dict, device):
    """Load the trained per-agent detector named in ``hypes['detector']``."""
    from opencood.tools.train_utils import create_model

    detector = create_model(hypes)
    checkpoint_path = hypes["detector"]["checkpoint"]
    state = torch.load(checkpoint_path, map_location="cpu")
    state = state.get("model_state_dict", state) if isinstance(state, dict) else state
    detector.load_state_dict(state)
    return detector.to(device).eval()


# Method -> pipeline function; metric -> the AP report built from its output.
# Later tasks add entries here (alignformer_a/b/ransac methods; top1/pose
# metrics) without touching the CLI or this dispatch shape (ruling R1).
_METHODS = {"late_fusion_clean": run_late_fusion_clean}


def _ap_report(predictions, ground_truth) -> Dict[str, Dict[str, float]]:
    return {
        f"ap_{int(threshold * 100):02d}": {
            "global_sorted": average_precision(
                predictions, ground_truth, threshold, global_sort=True
            ),
            "frame_order": average_precision(
                predictions, ground_truth, threshold, global_sort=False
            ),
        }
        for threshold in _AP_IOU_THRESHOLDS
    }


def _ap_report_from_matches(matches, *, indices=None) -> Dict[str, Dict[str, float]]:
    """:func:`_ap_report`, replayed from one :func:`match_frames` pass.

    Same keys, same numbers; the rotated-IoU matching is simply not redone for
    each threshold, sort order and frame slice.
    """
    return {
        f"ap_{int(threshold * 100):02d}": {
            "global_sorted": average_precision_from_matches(
                matches, threshold, global_sort=True, indices=indices
            ),
            "frame_order": average_precision_from_matches(
                matches, threshold, global_sort=False, indices=indices
            ),
        }
        for threshold in _AP_IOU_THRESHOLDS
    }


def _score_by_seed(predictions, seeds, ground_truth, buckets):
    """AP per seed, full and sliced, with ONE rotated-IoU matching per condition.

    The matching is replayed for the three thresholds, the two sort orders and
    the three shared-object slices; matching per reading instead would be
    eighteen passes over the same polygons and, at five seeds, would cost more
    than the sweep did.

    A cell aliased across seeds -- the oracle, and every condition at sigma = 0,
    which perturbs nothing -- is the SAME list object under every seed, so
    ``seen`` scores it once. Identity is the right key precisely because the
    aliasing is deliberate: two seeds collide only when they are literally the
    same run, never when they merely agree.
    """
    ap_by_seed: Dict[str, Dict] = {str(seed): {} for seed in seeds}
    ap_by_shared_by_seed: Dict[str, Dict] = {
        str(seed): {
            bucket: {"frames": len(indices), "conditions": {}}
            for bucket, indices in buckets.items()
        }
        for seed in seeds
    }
    seen: Dict[int, Tuple[Any, Dict, Dict]] = {}
    for seed in seeds:
        for name, frames in predictions[seed].items():
            scored = seen.get(id(frames))
            if scored is None:
                matches = match_frames(frames, ground_truth)
                scored = (
                    frames,  # held so the id cannot be recycled under the cache
                    _ap_report_from_matches(matches),
                    {
                        bucket: _ap_report_from_matches(matches, indices=indices)
                        for bucket, indices in buckets.items()
                    },
                )
                seen[id(frames)] = scored
            _, full, sliced = scored
            ap_by_seed[str(seed)][name] = full
            for bucket in buckets:
                ap_by_shared_by_seed[str(seed)][bucket]["conditions"][name] = sliced[
                    bucket
                ]
            print(
                f"  seed {seed}  {name:<32} "
                f"AP@0.3={full['ap_30']['global_sorted']:.4f} "
                f"AP@0.5={full['ap_50']['global_sorted']:.4f} "
                f"AP@0.7={full['ap_70']['global_sorted']:.4f}",
                flush=True,
            )
    return ap_by_seed, ap_by_shared_by_seed


def _sigma_key(sigma: float) -> str:
    """Result-dict key for one evaluation noise level, e.g. ``sigma_0.5m``."""
    return f"sigma_{sigma:g}m"


def run_top1(args: argparse.Namespace, device) -> Dict:
    """Score cross-agent association on the held-out validation scenarios.

    The split is the same scenario-disjoint 15% slice stage 1 validated on, and
    the same slice the per-agent detector held out, so nothing in the pipeline
    has seen these scenarios. Splitting by scenario rather than by frame is not
    a nicety: consecutive OPV2V frames are near-duplicates, so a frame split
    would report a leaked number.
    """
    import yaml
    from torch.utils.data import DataLoader

    from embedding_aware_belt_fusion.alignformer.dataset import collate
    from embedding_aware_belt_fusion.alignformer.train import (
        build_eval_dataset,
        build_pair_split,
        evaluate_matching,
        load_stage1,
    )

    config = yaml.safe_load(args.config.read_text())
    checkpoint_path = args.checkpoint[0]
    modules, checkpoint = load_stage1(checkpoint_path, device)
    _, val_pairs, train_scenarios, val_scenarios = build_pair_split(config)

    gate_sigma = float(config["train"]["stage1_max_xy_std"])
    results: Dict[str, Dict[str, float]] = {}
    for sigma in (gate_sigma,) + _TOP1_CONTEXT_SIGMAS:
        dataset = build_eval_dataset(config, val_pairs, sigma)
        loader = DataLoader(
            dataset,
            batch_size=_TOP1_BATCH_SIZE,
            num_workers=args.num_workers,
            collate_fn=collate,
            pin_memory=True,
        )
        metrics = evaluate_matching(modules, loader, device)
        results[_sigma_key(sigma)] = metrics
        print(
            f"  sigma={sigma:g} m: top1={metrics['top1']:.4f} "
            f"(nearest-centre control {metrics['top1_nearest_centre']:.4f}, "
            f"chance {metrics['top1_chance']:.4f})",
            flush=True,
        )

    measured = results[_sigma_key(gate_sigma)]["top1"]
    return {
        "method": "alignformer_b_stage1",
        "metric": "top1",
        "config": str(args.config),
        "split": (
            f"{config['data']['train_root']} :: validation scenarios "
            f"(scenario-disjoint, val_scenario_fraction="
            f"{config['data']['val_scenario_fraction']}, split_seed="
            f"{config['data']['split_seed']})"
        ),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint["epoch"],
        "cache_root": config["data"]["cache_root"],
        "comm_range_m": config["data"]["comm_range_m"],
        "train_scenarios": len(train_scenarios),
        "val_scenarios": val_scenarios,
        "val_pairs": len(val_pairs),
        "gate": {
            "name": "P1",
            "definition": (
                "cross-agent Top-1: of the ego objects that have a true CAV "
                "counterpart, the fraction whose highest-scoring real CAV column "
                "is the correct one. Ego objects with no counterpart, and padded "
                "positions, are excluded."
            ),
            "threshold": P1_TOP1_GATE,
            "sigma_m": gate_sigma,
            "value": measured,
            "passed": bool(measured >= P1_TOP1_GATE),
        },
        "results": results,
    }


def _mean(values: List[float]) -> float:
    return sum(values) / len(values)


def _pose_provenance(checkpoint: Dict) -> Dict:
    """What a stage-2 checkpoint was trained under.

    ``variance_weighting`` defaults to ``"none"`` for checkpoints written before
    ``alignformer.variance`` existed, which is what they were trained under.
    """
    return {
        "head": checkpoint["head"],
        "message_content": checkpoint["message_content"],
        "match_weight": checkpoint["match_weight"],
        "variance_weighting": checkpoint.get("variance_weighting", "none"),
        "stage1_checkpoint": checkpoint["stage1_checkpoint"],
        "epoch": checkpoint["epoch"],
    }


def _configuration_name(checkpoint: Dict) -> str:
    """The table label for one stage-2 configuration.

    Deliberately blind to ``variance_weighting``: the P2 gate selects its
    subject by this name, and appending the estimator variant to it would make
    the gate's definition depend on which estimator was being measured.
    ``_reject_duplicate_configurations`` is what stops two variants from being
    silently merged into one row instead.
    """
    suffix = "" if checkpoint["match_weight"] else " (match_weight 0)"
    return f"{checkpoint['head']} / {checkpoint['message_content']}{suffix}"


def _reject_duplicate_configurations(loaded) -> None:
    """Refuse a ``--metric pose`` run whose checkpoints share a table label.

    Two estimator variants of the same (head, message content) are a legitimate
    comparison, but they must be measured in separate runs, not collapsed onto
    one row. The noise draws are keyed on (seed, sigma, sample index) alone, so
    separate runs still see byte-identical perturbations and the comparison
    stays paired.
    """
    seen: Dict[str, str] = {}
    for path, _, checkpoint in loaded:
        name = _configuration_name(checkpoint)
        if name in seen:
            raise ValueError(
                f"{path} and {seen[name]} both report as {name!r}; measure them in "
                "separate runs (the noise draws are identical across runs)"
            )
        seen[name] = str(path)


def run_pose(args: argparse.Namespace, device) -> Dict:
    """Sweep localization noise and report pose MAE against the predict-zero baseline.

    The split is the same scenario-disjoint 15% slice stages 1 and 2 validated
    on, and the same slice the per-agent detector held out. Each sigma is
    measured under ``--seeds`` independent deterministic noise draws of the same
    pairs; the tables report the mean, and the per-seed values are kept so a
    spread can be read off rather than assumed.

    Every configuration passed via ``--checkpoint`` is measured over the same
    draws, so the head and message-content comparisons are paired -- they see
    byte-identical perturbations, not merely the same distribution.
    """
    import yaml
    from torch.utils.data import DataLoader

    from embedding_aware_belt_fusion.alignformer.dataset import (
        YAW_STD_PER_XY_STD,
        collate,
    )
    from embedding_aware_belt_fusion.alignformer.stage2 import evaluate_pose, load_stage2
    from embedding_aware_belt_fusion.alignformer.train import (
        build_eval_dataset,
        build_pair_split,
    )

    config = yaml.safe_load(args.config.read_text())
    _, val_pairs, train_scenarios, val_scenarios = build_pair_split(config)
    shrinkage = _load_shrinkage(args)

    loaded = []
    for path in args.checkpoint:
        modules, checkpoint = load_stage2(path, device)
        loaded.append((path, modules, checkpoint))
    _reject_duplicate_configurations(loaded)

    seeds = [_POSE_SEED_BASE + offset for offset in range(args.seeds)]
    results: Dict[str, Dict[str, Dict]] = {}

    for sigma in args.sweep:
        per_configuration: Dict[str, List[Dict[str, float]]] = {
            _configuration_name(checkpoint): [] for _, _, checkpoint in loaded
        }
        for seed in seeds:
            dataset = build_eval_dataset(config, val_pairs, sigma, seed=seed)
            loader = DataLoader(
                dataset, batch_size=_TOP1_BATCH_SIZE, num_workers=args.num_workers,
                collate_fn=collate, pin_memory=True,
            )
            for _, modules, checkpoint in loaded:
                metrics = evaluate_pose(
                    modules, loader, device,
                    ablate_embeddings=checkpoint["message_content"] == "boxes_only",
                    shrinkage=shrinkage,
                )
                per_configuration[_configuration_name(checkpoint)].append(metrics)

        cell: Dict[str, Dict] = {}
        for name, runs in per_configuration.items():
            keys = [key for key, value in runs[0].items() if isinstance(value, float)]
            cell[name] = {
                "mean": {key: _mean([run[key] for run in runs]) for key in keys},
                "per_seed_yaw_mae_deg": [run["yaw_mae_deg"] for run in runs],
                "per_seed_translation_mae_m": [run["translation_mae_m"] for run in runs],
            }
        analytic = sigma * YAW_STD_PER_XY_STD * _HALF_NORMAL_MEAN
        results[_sigma_key(sigma)] = {
            "sigma_xy_m": sigma,
            "sigma_yaw_deg": sigma * YAW_STD_PER_XY_STD,
            "analytic_predict_zero_yaw_mae_deg": analytic,
            "configurations": cell,
        }
        for name, values in cell.items():
            print(
                f"  sigma={sigma:g} m  {name:<34} "
                f"t_mae={values['mean']['translation_mae_m']:.4f} m "
                f"(zero {values['mean']['predict_zero_translation_mae_m']:.4f}) "
                f"yaw_mae={values['mean']['yaw_mae_deg']:.4f} deg "
                f"(zero {values['mean']['predict_zero_yaw_mae_deg']:.4f}, "
                f"analytic {analytic:.4f})",
                flush=True,
            )

    return {
        "method": "alignformer_stage2",
        "metric": "pose",
        "config": str(args.config),
        "split": (
            f"{config['data']['train_root']} :: validation scenarios "
            f"(scenario-disjoint, val_scenario_fraction="
            f"{config['data']['val_scenario_fraction']}, split_seed="
            f"{config['data']['split_seed']})"
        ),
        "checkpoints": {
            _configuration_name(checkpoint): {
                "path": str(Path(path).resolve()),
                **_pose_provenance(checkpoint),
            }
            for path, _, checkpoint in loaded
        },
        "cache_root": config["data"]["cache_root"],
        "comm_range_m": config["data"]["comm_range_m"],
        "train_scenarios": len(train_scenarios),
        "val_scenarios": val_scenarios,
        "val_pairs": len(val_pairs),
        "sweep_sigmas_m": list(args.sweep),
        "noise_seeds": seeds,
        "shrinkage": None if shrinkage is None else shrinkage.to_dict(),
        "gate": _p2_gate(results),
        "results": results,
    }


def _p2_gate(results: Dict[str, Dict]) -> Dict:
    """The P2 verdict: head B's yaw MAE below predict-zero at every non-zero sigma.

    Measured against the EMPIRICAL predict-zero value on the same samples, not
    against the analytic ``sigma * sqrt(2/pi)``: the two agree to within the
    sample noise, but only the empirical one is a statement about the data that
    was actually scored. The analytic value is reported beside it.
    """
    name = next(
        (
            key
            for key in next(iter(results.values()))["configurations"]
            if key == "B / boxes+embeddings"
        ),
        None,
    )
    if name is None:
        return {"name": "P2", "passed": None, "reason": "no B / boxes+embeddings run supplied"}

    cells = []
    for sigma_key, entry in results.items():
        if entry["sigma_xy_m"] == 0.0:
            continue
        mean = entry["configurations"][name]["mean"]
        cells.append(
            {
                "sigma": sigma_key,
                "yaw_mae_deg": mean["yaw_mae_deg"],
                "predict_zero_yaw_mae_deg": mean["predict_zero_yaw_mae_deg"],
                "analytic_predict_zero_yaw_mae_deg": entry[
                    "analytic_predict_zero_yaw_mae_deg"
                ],
                "below": mean["yaw_mae_deg"] < mean["predict_zero_yaw_mae_deg"],
            }
        )
    return {
        "name": "P2",
        "definition": (
            "head B (boxes+embeddings) yaw MAE strictly below the predict-zero "
            "yaw MAE at every non-zero sigma. sigma = 0 is excluded because the "
            "predict-zero baseline is exactly 0 there and nothing can be below "
            "it; it is reported in `results` as a clean-case diagnostic."
        ),
        "configuration": name,
        "cells": cells,
        "passed": bool(cells) and all(cell["below"] for cell in cells),
    }


def _load_shrinkage(args: argparse.Namespace):
    """Read the calibration named by ``--shrinkage``, or ``None`` if it was omitted."""
    from embedding_aware_belt_fusion.alignformer.shrinkage import ShrinkageCalibration

    if args.shrinkage is None:
        return None
    payload = json.loads(args.shrinkage.read_text())
    return ShrinkageCalibration.from_dict(payload["calibration"])


def run_shrinkage(args: argparse.Namespace, device) -> Dict:
    """Calibrate the shrinkage constant on validation at sigma = 0.

    Sigma = 0 is the only noise level at which the true correction is exactly
    the identity, so the emitted correction IS the estimator's residual and
    ``tau`` can be read off it without having to subtract a label. The signed
    mean of that residual is reported beside ``tau``: shrinkage is the correct
    treatment for an unbiased-but-imprecise estimator, and if the residual
    turned out to be biased instead, the right response would be to find the
    bias, not to shrink it.

    The split is the scenario-disjoint validation slice, never the test split.
    """
    import yaml
    from torch.utils.data import DataLoader

    from embedding_aware_belt_fusion.alignformer.dataset import collate
    from embedding_aware_belt_fusion.alignformer.stage2 import (
        calibrate_from_loader,
        load_stage2,
    )
    from embedding_aware_belt_fusion.alignformer.train import (
        build_eval_dataset,
        build_pair_split,
    )

    config = yaml.safe_load(args.config.read_text())
    _, val_pairs, _, val_scenarios = build_pair_split(config)
    checkpoint_path = args.checkpoint[0]
    modules, checkpoint = load_stage2(checkpoint_path, device)

    split = (
        f"{config['data']['train_root']} :: validation scenarios "
        f"(scenario-disjoint, val_scenario_fraction="
        f"{config['data']['val_scenario_fraction']}, split_seed="
        f"{config['data']['split_seed']})"
    )
    dataset = build_eval_dataset(config, val_pairs, 0.0)
    loader = DataLoader(
        dataset, batch_size=_TOP1_BATCH_SIZE, num_workers=args.num_workers,
        collate_fn=collate, pin_memory=True,
    )
    # Which estimator is being calibrated. tau is a property of ONE arm's
    # residual distribution: the deployed closed-form solve and the IRLS arm
    # (alignformer.robust) have different ones, and scoring the second through
    # the first's tau over-shrinks it by an unmeasured amount. 'none' keeps the
    # deployed calibration reproducible.
    robust = RobustSolveConfig(
        mode=args.robust_solve,
        iterations=args.robust_iterations,
        min_evidence=args.robust_min_evidence,
    )
    calibration, diagnostics = calibrate_from_loader(
        modules, loader, device,
        ablate_embeddings=checkpoint["message_content"] == "boxes_only",
        split=split,
        sigma_m=0.0,
        robust=robust,
    )

    for name in ("dx_m", "dy_m", "dpsi_deg"):
        mean, sem = diagnostics[f"signed_mean_{name}"], diagnostics[f"sem_{name}"]
        print(
            f"  signed mean {name:<9} {mean:+.5f}  95% CI "
            f"[{mean - 1.96 * sem:+.5f}, {mean + 1.96 * sem:+.5f}]",
            flush=True,
        )
    print(
        f"  tau_translation={calibration.tau_translation_m:.4f} m  "
        f"tau_yaw={calibration.tau_yaw_deg:.4f} deg  "
        f"(translation MAE {diagnostics['translation_mae_m']:.4f} m, "
        f"yaw MAE {diagnostics['yaw_mae_deg']:.4f} deg)",
        flush=True,
    )

    return {
        "method": "alignformer_stage2",
        "metric": "shrinkage",
        "config": str(args.config),
        "split": split,
        "val_scenarios": val_scenarios,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_provenance": _pose_provenance(checkpoint),
        # WHICH arm's residuals this tau was measured on. A calibration without
        # this field is ambiguous the moment a second estimator exists.
        "robust_solve_config": robust.to_dict(),
        "calibration": calibration.to_dict(),
        "residual_diagnostics": diagnostics,
    }


def _run_noisy_ap(args: argparse.Namespace, device) -> Dict:
    """Fused AP under localization error: uncorrected vs AlignFormer vs oracle."""
    import yaml
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml.yaml_utils import load_yaml

    from embedding_aware_belt_fusion.alignformer.freealign import FreeAlignConfig
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import (
        ALIGNFORMER,
        ALIGNFORMER_IRLS,
        FREEALIGN,
        ORACLE,
        ORACLE_MATCH_CONDITIONS,
        SHARED_BUCKETS,
        UNCORRECTED,
        condition_key,
        draw_seeds,
        run_noise_sweep,
    )
    from embedding_aware_belt_fusion.alignformer.stage2 import load_stage2

    hypes = load_yaml(str(args.config), None)
    hypes["validate_dir"] = str(args.split.resolve())
    dataset = build_dataset(hypes, visualize=False, train=False)
    detector = _build_detector(hypes, device)

    model_config = yaml.safe_load(args.alignformer_config.read_text())
    checkpoint_path = args.checkpoint[0]
    modules, checkpoint = load_stage2(checkpoint_path, device)
    shrinkage = _load_shrinkage(args)

    freealign = (
        FreeAlignConfig(
            edge_feature=args.freealign_edge_feature,
            min_nodes=args.freealign_min_nodes,
        )
        if args.freealign
        else None
    )
    robust = RobustSolveConfig(
        mode=args.robust_solve,
        iterations=args.robust_iterations,
        min_evidence=args.robust_min_evidence,
    )
    abstention = [
        AbstentionConfig.parse(spec) for spec in (args.abstain_arm or [])
    ]
    # A disabled arm produces no rows, so recording one in the result file
    # would advertise a condition nothing measured. Refuse it here rather than
    # filtering it away silently.
    disabled = [config.name for config in abstention if not config.enabled]
    if disabled:
        raise ValueError(
            f"--abstain-arm none produces no arm; drop {disabled} or give it a mode"
        )
    if len({config.name for config in abstention}) != len(abstention):
        raise ValueError(
            f"--abstain-arm names must be distinct, got "
            f"{[config.name for config in abstention]}"
        )

    # Independent noise draws of the same frames. The FIRST seed is the config's
    # own -- the one every single-seed sweep on this branch was run under -- so
    # `--ap-seeds 1` reproduces those files bit for bit and a longer run carries
    # the published draw as its first row rather than replacing it.
    base_seed = int(model_config["train"]["seed"])
    if args.ap_seeds < 1:
        raise ValueError(f"--ap-seeds must be at least 1, got {args.ap_seeds}")
    seeds = [base_seed + offset for offset in range(args.ap_seeds)]

    (
        predictions,
        ground_truth,
        pose_stats,
        intermediate_truth,
        frame_shared_counts,
    ) = run_noise_sweep(
        dataset, detector, dataset.post_processor, device,
        modules=modules,
        ablate_embeddings=checkpoint["message_content"] == "boxes_only",
        lidar_range=hypes["preprocess"]["cav_lidar_range"],
        output_size=int(model_config["model"]["output_size"]),
        sigmas=list(args.sweep),
        seeds=seeds,
        training_comm_range_m=float(model_config["data"]["comm_range_m"]),
        max_frames=args.max_frames,
        shrinkage=shrinkage,
        oracle_match=args.oracle_match,
        freealign=freealign,
        robust=robust,
        abstention=abstention,
    )

    # AP sliced by how many ground-truth objects the frame's agents share --
    # the row the FreeAlign comparison turns on, because a relative-distance
    # graph is structurally degenerate below three and AlignFormer's heading
    # virtual points are not. Averaged over the whole split that slice is
    # diluted eightfold and the difference disappears into noise.
    grouped = _frames_by_shared_bucket(frame_shared_counts)
    buckets = {bucket: indices for bucket, indices in grouped.items() if indices}

    ap_by_seed, ap_by_shared_by_seed = _score_by_seed(
        predictions, seeds, ground_truth, buckets
    )

    # The ground truth OpenCOOD's IntermediateFusionDataset would report is a
    # second matching pass over the same predictions, and it exists to say how
    # much the CHOICE OF CONVENTION is worth -- a question about the dataset,
    # not about the noise draw. It is therefore scored for the primary seed
    # only: five copies of it would double the scoring cost of every sweep to
    # put an error bar on a number nothing reads one from.
    ap_intermediate_gt = {
        name: _ap_report_from_matches(match_frames(frames, intermediate_truth))
        for name, frames in predictions[seeds[0]].items()
    }

    # The primary seed reproduces the single-seed runs this branch published,
    # under exactly the keys those files carry. `..._by_seed` is the new
    # measurement; these three stay the rows a reader already knows how to find.
    ap = ap_by_seed[str(seeds[0])]
    ap_by_shared = ap_by_shared_by_seed[str(seeds[0])]

    return {
        "method": "alignformer_noisy_late_fusion",
        "metric": "noisy_ap",
        "config": str(args.config),
        "alignformer_config": str(args.alignformer_config),
        "split": str(args.split),
        "detector_checkpoint": hypes["detector"]["checkpoint"],
        "pose_checkpoint": str(checkpoint_path.resolve()),
        "pose_checkpoint_provenance": _pose_provenance(checkpoint),
        "shrinkage": None if shrinkage is None else shrinkage.to_dict(),
        "cav_lidar_range": hypes["preprocess"]["cav_lidar_range"],
        "nms_thresh": dataset.post_processor.params["nms_thresh"],
        "frames": len(ground_truth),
        "sweep_sigmas_m": list(args.sweep),
        "conditions": {
            "oracle": ORACLE,
            "uncorrected": UNCORRECTED,
            "alignformer": ALIGNFORMER,
            "oracle_match": list(ORACLE_MATCH_CONDITIONS) if args.oracle_match else [],
            "freealign": FREEALIGN if freealign is not None else None,
            "robust_solve": ALIGNFORMER_IRLS if robust.enabled else None,
            # One name per per-pair abstention arm (task 24). Each is the IRLS
            # solve with a decision applied, and each carries its own
            # calibration instead of the global tau, so `shrinkage` above does
            # NOT describe them.
            "abstention": [config.name for config in abstention],
            "key_format": condition_key(ALIGNFORMER, 1.0),
        },
        # Labelled unambiguously: this row is OUR reimplementation of the
        # published method, on our detections, never the authors' code.
        "freealign_config": None if freealign is None else freealign.to_dict(),
        # The IRLS arm's frozen configuration. Chosen on the scenario-disjoint
        # validation slice (scripts/calibrate_robust_solve.py); recorded here so
        # a result file states what it was measured under.
        "robust_solve_config": robust.to_dict(),
        # Every constant each abstention arm ran under, so a result file states
        # what it was measured with rather than leaving it to a shell history.
        "abstention_arms": [config.to_dict() for config in abstention],
        "ap": ap,
        # The multi-seed measurement (task 23). Independent noise draws of the
        # same frames, shared across conditions within a draw, so the per-seed
        # difference between two conditions is paired; see
        # alignformer.seedstats. `ap` above is seed `noise_seeds[0]`, unchanged.
        "noise_seeds": seeds,
        # sigma = 0 perturbs nothing, so it is drawn ONCE and every seed's entry
        # for it is the same run. Reporting a spread there would be fabricated.
        "sigmas_drawn_once_m": [
            sigma for sigma in args.sweep if len(draw_seeds(sigma, seeds)) == 1
        ],
        "ap_by_seed": ap_by_seed,
        # The same predictions scored against the ground truth OpenCOOD's
        # IntermediateFusionDataset would report, so the head-to-head table
        # against intermediate-fusion baselines can say how much the choice of
        # convention is worth. `ap` above is unchanged and remains the headline.
        "ap_intermediate_convention_gt": ap_intermediate_gt,
        "ground_truth_boxes": sum(int(g.shape[0]) for g in ground_truth),
        "intermediate_convention_ground_truth_boxes": sum(
            int(g.shape[0]) for g in intermediate_truth
        ),
        "frames_where_gt_conventions_differ": sum(
            1 for a, b in zip(ground_truth, intermediate_truth)
            if a.shape[0] != b.shape[0]
        ),
        # Unchanged in shape and meaning: the deployed estimator's own pose
        # error per sigma, which every earlier result file carries under this
        # key. The oracle conditions are reported beside it, never folded in.
        "pose": pose_stats[seeds[0]][ALIGNFORMER],
        "pose_by_condition": pose_stats[seeds[0]],
        "pose_by_condition_by_seed": {
            str(seed): pose_stats[seed] for seed in seeds
        },
        "ap_by_shared_objects": ap_by_shared,
        "ap_by_shared_objects_by_seed": ap_by_shared_by_seed,
        "shared_object_buckets": list(SHARED_BUCKETS),
        "frames_with_mixed_shared_object_buckets": len(grouped["mixed"]),
        "pairs_by_shared_object_bucket": _pairs_by_shared_bucket(frame_shared_counts),
    }


def _frames_by_shared_bucket(frame_shared_counts) -> Dict[str, List[int]]:
    """Frame indices grouped by the shared-object bucket of ALL their pairs.

    AP is a per-frame quantity but the shared-object count is a per-*pair* one,
    so a frame with two CAVs in different buckets belongs to neither. Those
    frames go to ``mixed`` and are reported as a count rather than silently
    assigned, and a frame with no CAV at all is in no bucket: nothing about it
    distinguishes the conditions.
    """
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import (
        SHARED_BUCKETS,
        shared_object_bucket,
    )

    grouped: Dict[str, List[int]] = {name: [] for name in SHARED_BUCKETS}
    grouped["mixed"] = []
    for index, counts in enumerate(frame_shared_counts):
        if not counts:
            continue
        names = {shared_object_bucket(count) for count in counts}
        grouped[names.pop() if len(names) == 1 else "mixed"].append(index)
    return grouped


def _pairs_by_shared_bucket(frame_shared_counts) -> Dict[str, int]:
    """How many ego-CAV PAIRS fall in each bucket, the denominator for coverage."""
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import (
        SHARED_BUCKETS,
        shared_object_bucket,
    )

    tally = {name: 0 for name in SHARED_BUCKETS}
    for counts in frame_shared_counts:
        for count in counts:
            tally[shared_object_bucket(count)] += 1
    return tally


def _run_ap(args: argparse.Namespace, device) -> Dict:
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml.yaml_utils import load_yaml

    hypes = load_yaml(str(args.config), None)
    hypes["validate_dir"] = str(args.split.resolve())

    dataset = build_dataset(hypes, visualize=False, train=False)
    detector = _build_detector(hypes, device)

    pipeline = _METHODS[args.method]
    predictions, ground_truth = pipeline(
        dataset, detector, dataset.post_processor, device, max_frames=args.max_frames
    )

    return {
        "method": args.method,
        "metric": args.metric,
        "config": str(args.config),
        "split": str(args.split),
        "checkpoint": hypes["detector"]["checkpoint"],
        "cav_lidar_range": hypes["preprocess"]["cav_lidar_range"],
        "nms_thresh": dataset.post_processor.params["nms_thresh"],
        "frames": len(predictions),
        "ap": _ap_report(predictions, ground_truth),
    }


# Metric -> the runner that produces its result dict. Each one is a new entry
# here, not a change to the CLI or to any other runner (ruling R1).
_METRIC_RUNNERS = {
    "ap": _run_ap,
    "top1": run_top1,
    "pose": run_pose,
    "noisy_ap": _run_noisy_ap,
    "shrinkage": run_shrinkage,
}


def _json_safe(value):
    """Replace NaN with ``null`` so the written file is valid JSON.

    An empty subset (e.g. the structurally unalignable pairs, of which the
    validation split has none) yields NaN rather than 0.0, because nothing was
    scored is not the same as scoring nothing correctly. ``json.dumps`` writes
    that as the bare token ``NaN``, which Python reads back but which is not
    JSON and which several readers reject outright -- and these result files
    are committed evidence that other tools read.
    """
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)

    result = _json_safe(_METRIC_RUNNERS[args.metric](args, device))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
