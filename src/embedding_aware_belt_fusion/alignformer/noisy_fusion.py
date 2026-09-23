"""Fused AP under CAV localization error: uncorrected, AlignFormer, oracle.

This is the measurement the whole method exists for. The P0 gate established
that clean late fusion scores 0.8764 AP@0.7 when every agent's pose is exact.
Real localization error breaks that: a CAV that believes it is two metres from
where it is projects every one of its boxes two metres off, and those boxes
stop matching the ego's own detections and start being counted as false
positives. AlignFormer's claim is that it can recover the SE(2) error well
enough to undo that degradation.

Three conditions are fused from **one** set of detections per frame, so nothing
but the correction differs between them:

- ``oracle`` -- every agent corrected by its true relative pose. Independent of
  sigma, and identical to the P0 pipeline; it is the ceiling, and its
  appearance in the sweep is also a consistency check against ``p0_gate.json``.
- ``uncorrected`` -- every agent corrected by its *noisy* relative pose, which
  is what plain late fusion does when the pose it is given is wrong. The floor.
- ``alignformer`` -- the noisy correction, then AlignFormer's estimated
  residual SE(2) on top. The result is the gap it closes between the two.

The detector runs once per agent per frame and every condition and sigma reuses
those detections. Without that the sweep would be a detector benchmark: the
forward pass dominates everything else here by two orders of magnitude.

The per-agent object sets fed to AlignFormer are built exactly as
``alignformer.dataset`` builds them for training -- top-``MAX_OBJECTS`` by
score, ROI features sampled in each agent's *own* frame and round-tripped
through float16 as the cache stores them, CAV boxes projected into the ego
frame with the noisy pose. A mismatch there would be a silent train/test skew
that looks like a modelling result.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from embedding_aware_belt_fusion.alignformer.boxes import AgentDetections, detect_agent
from embedding_aware_belt_fusion.alignformer.dataset import YAW_STD_PER_XY_STD
from embedding_aware_belt_fusion.alignformer.embedding import rotated_roi_align
from embedding_aware_belt_fusion.alignformer.fusion import (
    correct_boxes,
    correct_detections,
    late_fuse,
)
from embedding_aware_belt_fusion.alignformer.shrinkage import ShrinkageCalibration, shrink
from embedding_aware_belt_fusion.alignformer.stage2 import is_fallback
from embedding_aware_belt_fusion.alignformer.train import embed_batch
from embedding_aware_belt_fusion.alignformer.trunk import MAX_OBJECTS
from embedding_aware_belt_fusion.coloca.geometry import perturb_pose_2d, relative_pose_error

# A FIXED reporting boundary, independent of whatever range the pair index was
# built at. Every measurement in docs/alignformer_pose_floor.md is reported
# either side of 40 m, and now that the index matches OpenCOOD's COM_RANGE the
# training-range split collapses to "everything"; without this constant the
# before/after comparison would quietly disappear.
DIAGNOSTIC_RANGE_M = 40.0

ORACLE = "oracle"
UNCORRECTED = "uncorrected"
ALIGNFORMER = "alignformer"

_PROGRESS_INTERVAL = 100


def condition_key(condition: str, sigma: float) -> str:
    """Result-dict key for one (condition, sigma) cell; the oracle has no sigma."""
    return condition if condition == ORACLE else f"{condition}_sigma_{sigma:g}m"


def _sweep_rng(seed: int, sigma: float, frame: int, agent: int) -> np.random.Generator:
    """Deterministic per-(sigma, frame, agent) noise draw.

    Keyed on sigma as well as on the frame so that the sweep's levels are
    independent draws rather than one draw rescaled -- a rescaled draw would
    make every level's error point in the same direction and turn the sweep
    into a single sample.
    """
    return np.random.default_rng([seed, int(round(sigma * 1000)), frame, agent])


def _truncate_by_score(
    detections: AgentDetections, roi: Tensor, max_objects: int
) -> Tuple[Tensor, Tensor, Tensor, List[Optional[str]]]:
    """Keep the top ``max_objects`` detections by descending score.

    Mirrors ``alignformer.dataset._truncate_by_score``: the trunk's token budget
    is ``MAX_OBJECTS``, and training never saw a longer set.
    """
    count = detections.boxes.shape[0]
    if count <= max_objects:
        return detections.boxes, detections.scores, roi, list(detections.gt_ids)
    order = torch.argsort(detections.scores, descending=True, stable=True)[:max_objects]
    return (
        detections.boxes[order],
        detections.scores[order],
        roi[order],
        [detections.gt_ids[i] for i in order.tolist()],
    )


def _roi_like_the_cache(
    detections: AgentDetections, lidar_range: Sequence[float], output_size: int
) -> Tensor:
    """ROI features in the agent's own frame, round-tripped through float16.

    ``cache.write_frame`` stores the ROI stack as float16 and the dataset reads
    it back as float32, so training only ever saw half-precision features. The
    round-trip here is not cosmetic: skipping it would feed the frozen
    embedding head slightly different inputs at test time than it was fitted on.
    """
    roi = rotated_roi_align(detections.features, detections.boxes, lidar_range, output_size)
    return roi.half().float()


def _object_set(
    ego: Mapping[str, Tensor], cav: Mapping[str, Tensor]
) -> Dict[str, Tensor]:
    """A one-sample collated batch in the layout the trunk expects."""
    batch: Dict[str, Tensor] = {}
    for prefix, side in (("ego", ego), ("cav", cav)):
        batch[f"{prefix}_boxes"] = side["boxes"].unsqueeze(0)
        batch[f"{prefix}_scores"] = side["scores"].unsqueeze(0)
        batch[f"{prefix}_roi"] = side["roi"].unsqueeze(0)
        batch[f"{prefix}_mask"] = torch.ones(
            (1, side["boxes"].shape[0]), dtype=torch.bool, device=side["boxes"].device
        )
    return batch


class _PoseSubset:
    """Per-pair sums for one subset of the test split.

    Sums rather than running means, so a subset with a handful of members is
    weighted by its size and not by how the frames happened to batch.
    """

    def __init__(self) -> None:
        self.translation = 0.0
        self.zero_translation = 0.0
        self.yaw = 0.0
        self.zero_yaw = 0.0
        self.fallbacks = 0
        self.pairs = 0

    def add(self, translation, zero_translation, yaw, zero_yaw, fell_back) -> None:
        self.translation += translation
        self.zero_translation += zero_translation
        self.yaw += yaw
        self.zero_yaw += zero_yaw
        self.fallbacks += int(fell_back)
        self.pairs += 1

    def compute(self) -> Dict[str, Optional[float]]:
        def mean(total: float) -> Optional[float]:
            return total / self.pairs if self.pairs else None

        return {
            "pairs": float(self.pairs),
            "translation_mae_m": mean(self.translation),
            "predict_zero_translation_mae_m": mean(self.zero_translation),
            "yaw_mae_deg": mean(self.yaw),
            "predict_zero_yaw_mae_deg": mean(self.zero_yaw),
            "fallback_fraction": mean(float(self.fallbacks)),
        }


class _PoseStats:
    """Per-sigma pose error of the estimated correction, on the test split.

    Reported next to the predict-zero baseline on the same pairs, and split
    several ways, because an aggregate hides distinct ways of being wrong.

    - ``unalignable``: the two agents detected no object in common, so no pose
      is recoverable and the only safe answer is the identity correction.
      Whether it got one is read off the emitted ``(psi, t)``
      (``stage2.is_fallback``), not inferred from a threshold.
    - ``beyond_training_range``: the pair is farther apart than the
      ``comm_range_m`` the pair index was built with, so the model was never
      trained or validated on anything like it. This is the guard against a
      train/test range mismatch -- ``configs/alignformer.yaml`` once built
      pairs at 40 m while OpenCOOD's ``LateFusionDataset`` admitted every CAV
      within ``COM_RANGE = 70`` m, which put a third of the evaluated
      population out of distribution by construction. The two now agree, so
      this subset should be empty; it stays measured because a future
      divergence should show up as a number rather than as a mystery.
    - ``within_40m`` / ``beyond_40m``: the same split at a FIXED boundary
      (:data:`DIAGNOSTIC_RANGE_M`), which does not move with the config, so the
      far-pair numbers stay comparable with everything measured before the
      range was widened.
    """

    def __init__(self, training_range_m: float) -> None:
        self.training_range_m = training_range_m
        self.all = _PoseSubset()
        self.unalignable = _PoseSubset()
        self.within_training_range = _PoseSubset()
        self.beyond_training_range = _PoseSubset()
        self.within_40m = _PoseSubset()
        self.beyond_40m = _PoseSubset()

    def update(
        self,
        psi_hat: float,
        t_hat: Sequence[float],
        psi_true: float,
        t_true: Sequence[float],
        *,
        fell_back: bool,
        alignable: bool,
        distance_m: float,
    ) -> None:
        translation = float(np.hypot(t_hat[0] - t_true[0], t_hat[1] - t_true[1]))
        zero_translation = float(np.hypot(t_true[0], t_true[1]))
        yaw = abs(float(np.degrees(np.arctan2(
            np.sin(psi_hat - psi_true), np.cos(psi_hat - psi_true)
        ))))
        zero_yaw = abs(float(np.degrees(np.arctan2(np.sin(psi_true), np.cos(psi_true)))))

        terms = (translation, zero_translation, yaw, zero_yaw, fell_back)
        self.all.add(*terms)
        if not alignable:
            self.unalignable.add(*terms)
        if distance_m <= self.training_range_m:
            self.within_training_range.add(*terms)
        else:
            self.beyond_training_range.add(*terms)
        if distance_m <= DIAGNOSTIC_RANGE_M:
            self.within_40m.add(*terms)
        else:
            self.beyond_40m.add(*terms)

    def compute(self) -> Dict[str, Optional[float]]:
        metrics: Dict[str, Optional[float]] = dict(self.all.compute())
        metrics["training_comm_range_m"] = self.training_range_m
        metrics["unalignable_fraction"] = (
            self.unalignable.pairs / self.all.pairs if self.all.pairs else None
        )
        metrics["beyond_training_range_fraction"] = (
            self.beyond_training_range.pairs / self.all.pairs if self.all.pairs else None
        )
        metrics["diagnostic_range_m"] = DIAGNOSTIC_RANGE_M
        metrics["beyond_40m_fraction"] = (
            self.beyond_40m.pairs / self.all.pairs if self.all.pairs else None
        )
        for name, subset in (
            ("unalignable", self.unalignable),
            ("within_training_range", self.within_training_range),
            ("beyond_training_range", self.beyond_training_range),
            ("within_40m", self.within_40m),
            ("beyond_40m", self.beyond_40m),
        ):
            for key, value in subset.compute().items():
                metrics[f"{name}_{key}"] = value
        return metrics


def _shares_an_object(left: Sequence[Optional[str]], right: Sequence[Optional[str]]) -> bool:
    """True when both agents matched at least one common ground-truth object.

    ``None`` means "matched no ground-truth object", so two ``None`` detections
    are different objects, not the same one -- the same rule
    ``dataset.correspondence_indices`` applies.
    """
    return bool(
        {value for value in left if value is not None}
        & {value for value in right if value is not None}
    )


@torch.no_grad()
def run_noise_sweep(
    dataset,
    detector,
    postprocessor,
    device,
    *,
    modules,
    ablate_embeddings: bool,
    lidar_range: Sequence[float],
    output_size: int,
    sigmas: Sequence[float],
    seed: int,
    training_comm_range_m: float,
    max_frames: Optional[int] = None,
    shrinkage: Optional[ShrinkageCalibration] = None,
) -> Tuple[Dict[str, List[Tuple[Tensor, Tensor]]], List[Tensor], Dict[str, Dict[str, float]]]:
    """Fuse every frame under every condition; return predictions, truth and pose stats.

    Returns ``(predictions_by_condition, ground_truth, pose_stats_by_sigma)``.
    ``ground_truth`` is shared by every condition -- the frames and their labels
    do not change, only the correction applied to the CAV boxes does.

    ``shrinkage``, when given, is applied to every estimate before the boxes
    are moved, so the ``alignformer`` condition measures what the method would
    actually deploy. The calibration is fitted on the validation split at
    sigma = 0 and is the same one at every sigma here -- a per-sigma factor
    would be fitting the sweep it is being scored on.
    """
    from opencood.utils import box_utils
    from opencood.utils.transformation_utils import x1_to_x2

    from embedding_aware_belt_fusion.alignformer.evaluate import (
        _build_test_frame,
        _cav_content,
        _frame_identity,
        _pose_correction,
    )

    conditions = [ORACLE] + [
        condition_key(name, sigma)
        for sigma in sigmas
        for name in (UNCORRECTED, ALIGNFORMER)
    ]
    predictions: Dict[str, List[Tuple[Tensor, Tensor]]] = {key: [] for key in conditions}
    ground_truth: List[Tensor] = []
    intermediate_ground_truth: List[Tensor] = []
    stats = {sigma: _PoseStats(training_comm_range_m) for sigma in sigmas}
    nms_threshold = postprocessor.params["nms_thresh"]

    frame_count = len(dataset) if max_frames is None else min(max_frames, len(dataset))
    started = time.time()

    for index in range(frame_count):
        scenario, timestamp = _frame_identity(dataset, index)
        sample, poses, ego_pose, base = _build_test_frame(
            dataset, index, scenario, timestamp
        )
        batch = dataset.collate_batch_test([sample])

        detections: Dict[str, AgentDetections] = {}
        packs: Dict[str, Dict[str, Any]] = {}
        transforms: Dict[str, Tensor] = {}
        for key, entry in batch.items():
            found = detect_agent(detector, _cav_content(entry, device), postprocessor)
            detections[key] = found
            roi = _roi_like_the_cache(found, lidar_range, output_size)
            boxes, scores, roi, gt_ids = _truncate_by_score(found, roi, MAX_OBJECTS)
            packs[key] = {"boxes": boxes, "scores": scores, "roi": roi, "gt_ids": gt_ids}
            transforms[key] = entry["transformation_matrix"].to(device)

        # Oracle: every agent moved by its TRUE relative pose (the P0 pipeline).
        oracle = []
        for key, found in detections.items():
            psi, translation = _pose_correction(transforms[key])
            oracle.append(correct_detections(found, psi, translation))
        predictions[ORACLE].append(_fuse(oracle, nms_threshold))

        ego_pack = packs["ego"]
        for sigma in sigmas:
            uncorrected = [detections["ego"]]  # ego's own transform is the identity
            aligned = [detections["ego"]]
            for agent, key in enumerate(sorted(k for k in detections if k != "ego")):
                rng = _sweep_rng(seed, sigma, index, agent)
                noisy_pose = perturb_pose_2d(
                    poses[key], sigma, sigma * YAW_STD_PER_XY_STD, rng
                )
                noisy_transform = torch.as_tensor(
                    x1_to_x2(noisy_pose, ego_pose), dtype=torch.float32, device=device
                )
                psi_noisy, t_noisy = _pose_correction(noisy_transform)
                noisy_detections = correct_detections(detections[key], psi_noisy, t_noisy)
                uncorrected.append(noisy_detections)

                # Project the TRUNCATED set directly rather than slicing the
                # projected full one: truncation reorders by score, so a slice
                # would silently pair each box with another box's features.
                cav_pack = dict(packs[key])
                cav_pack["boxes"] = correct_boxes(packs[key]["boxes"], psi_noisy, t_noisy)

                estimate = _estimate(modules, ego_pack, cav_pack, ablate_embeddings)
                if shrinkage is not None:
                    estimate = shrink(estimate, shrinkage)
                aligned.append(
                    correct_detections(noisy_detections, estimate.psi[0], estimate.t[0])
                )

                dx, dy, dpsi_deg = relative_pose_error(ego_pose, poses[key], noisy_pose)
                stats[sigma].update(
                    float(estimate.psi[0].item()),
                    (float(estimate.t[0, 0]), float(estimate.t[0, 1])),
                    float(np.radians(dpsi_deg)),
                    (dx, dy),
                    fell_back=bool(is_fallback(estimate)[0].item()),
                    alignable=_shares_an_object(
                        packs["ego"]["gt_ids"], packs[key]["gt_ids"]
                    ),
                    distance_m=float(np.hypot(
                        poses[key][0] - ego_pose[0], poses[key][1] - ego_pose[1]
                    )),
                )

            predictions[condition_key(UNCORRECTED, sigma)].append(
                _fuse(uncorrected, nms_threshold)
            )
            predictions[condition_key(ALIGNFORMER, sigma)].append(
                _fuse(aligned, nms_threshold)
            )

        gt_corners = postprocessor.generate_gt_bbx(batch)
        gt_boxes = box_utils.corner_to_center(
            gt_corners.detach().cpu().numpy(), order=postprocessor.params["order"]
        )
        ground_truth.append(torch.from_numpy(gt_boxes).float())
        intermediate_ground_truth.append(
            _intermediate_convention_ground_truth(dataset, base, ego_pose)
        )

        if (index + 1) % _PROGRESS_INTERVAL == 0:
            rate = (index + 1) / (time.time() - started)
            print(f"  {index + 1}/{frame_count} frames  {rate:.2f} fr/s", flush=True)

    return (
        predictions,
        ground_truth,
        {f"sigma_{sigma:g}m": tally.compute() for sigma, tally in stats.items()},
        intermediate_ground_truth,
    )


def _intermediate_convention_ground_truth(dataset, base_data_dict, ego_pose) -> Tensor:
    """The ground truth ``IntermediateFusionDataset`` would report for this frame.

    Every in-range CAV's object list is referenced to the **ego** pose (rather
    than to the CAV's own, which is what late fusion does), unioned by object
    id, and masked to ``GT_RANGE`` in x and y -- exactly
    ``IntermediateFusionDataset.__getitem__`` followed by
    ``generate_gt_bbx``. Measured, not assumed, so the baseline comparison can
    state how much the convention is worth instead of hoping it is nothing.
    """
    import math

    import opencood.data_utils.datasets as opencood_datasets
    from opencood.utils import box_utils

    order = dataset.post_processor.params["order"]
    centers, object_ids = [], []
    for content in base_data_dict.values():
        pose = content["params"]["lidar_pose"]
        if math.hypot(pose[0] - ego_pose[0], pose[1] - ego_pose[1]) > \
                opencood_datasets.COM_RANGE:
            continue
        boxes, mask, ids = dataset.post_processor.generate_object_center(
            [content], list(ego_pose)
        )
        centers.append(boxes[mask == 1])
        object_ids += ids

    if not centers:
        return torch.zeros((0, 7), dtype=torch.float32)

    stacked = torch.from_numpy(np.vstack(centers)).float()
    stacked = stacked[[object_ids.index(x) for x in set(object_ids)]]
    if stacked.shape[0] == 0:
        return torch.zeros((0, 7), dtype=torch.float32)
    corners = box_utils.boxes_to_corners_3d(stacked, order)
    return stacked[box_utils.get_mask_for_boxes_within_range_torch(corners)]


def _estimate(modules, ego_pack, cav_pack, ablate: bool):
    """Run the frozen embedding head and the pose head over one ego-CAV pair."""
    batch = _object_set(ego_pack, cav_pack)
    enriched = embed_batch(modules["embedding"], batch, ablate=ablate)
    return modules["pose"](enriched)


def _fuse(
    detections_by_agent: Sequence[AgentDetections], nms_threshold: float
) -> Tuple[Tensor, Tensor]:
    """Late-fuse one frame and move the result to the CPU for accumulation."""
    boxes, scores = late_fuse(detections_by_agent, nms_threshold)
    return boxes.detach().cpu(), scores.detach().cpu()
