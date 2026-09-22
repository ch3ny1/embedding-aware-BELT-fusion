"""AlignFormer evaluation CLI: ``--method`` selects the fusion pipeline,
``--metric`` selects what to report (ruling R1).

Only ``--method late_fusion_clean --metric ap`` is wired up by this task (the
P0 gate). ``alignformer_a``, ``alignformer_b``, ``ransac`` methods and the
``top1``/``pose`` metrics are later tasks' work; the dispatch tables below
exist so those slot in as new entries without reshaping the CLI.

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
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor

from embedding_aware_belt_fusion.alignformer.boxes import detect_agent
from embedding_aware_belt_fusion.alignformer.fusion import (
    average_precision,
    correct_detections,
    late_fuse,
)

# AP is reported at all three; only AP@0.7 is the P0 gate.
_AP_IOU_THRESHOLDS = (0.3, 0.5, 0.7)
_PROGRESS_INTERVAL = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="OpenCOOD-format detector hypes yaml")
    parser.add_argument("--split", type=Path, required=True, help="OPV2V split directory, e.g. .../OPV2V/test")
    parser.add_argument(
        "--method", choices=("late_fusion_clean",), default="late_fusion_clean",
        help="Fusion pipeline to evaluate.",
    )
    parser.add_argument(
        "--metric", choices=("ap",), default="ap",
        help="What to report for the selected method.",
    )
    parser.add_argument("--output", type=Path, required=True, help="destination JSON result file")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-frames", type=int, default=None, help="limit frames, for smoke-testing")
    return parser.parse_args()


def _frame_seed(scenario: str, cav_id: str, timestamp: str) -> int:
    """Stable, frame-derived seed for OpenCOOD's unseeded point shuffle.

    Deliberately reimplemented rather than imported from
    ``alignformer.cache._frame_seed``: that module is being rebuilt
    concurrently by another task, and this evaluation must not couple to its
    churn. The formula is identical on purpose -- same inputs
    (``scenario``, ``cav_id``, ``timestamp``), same ``hashlib.sha256``
    construction, same big-endian truncation to 4 bytes -- so a live
    per-agent detection and a cached one for the same agent-frame draw the
    same point order and cannot silently diverge into an "unexplained delta"
    when a later task compares them in one table.
    """
    digest = hashlib.sha256(f"{scenario}/{cav_id}/{timestamp}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big")


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


def _build_test_frame(dataset, index: int, scenario: str, timestamp: str) -> Dict[str, dict]:
    """Reimplement ``LateFusionDataset.get_item_test`` with a reproducible seed.

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
    for cav_id, selected_cav_base in base_data_dict.items():
        cav_pose = selected_cav_base["params"]["lidar_pose"]
        distance = math.hypot(cav_pose[0] - ego_lidar_pose[0], cav_pose[1] - ego_lidar_pose[1])
        if distance > opencood_datasets.COM_RANGE:
            continue

        np.random.seed(_frame_seed(scenario, cav_id, timestamp))
        processed = dataset.get_item_single_car(selected_cav_base)
        processed["transformation_matrix"] = x1_to_x2(cav_pose, ego_lidar_pose)
        frame["ego" if cav_id == ego_id else cav_id] = processed

    return frame


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
        sample = _build_test_frame(dataset, index, scenario, timestamp)
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


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)

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

    result = {
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
