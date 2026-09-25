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

With ``freealign`` a fourth condition joins them: this project's
REIMPLEMENTATION of Lei et al.'s FreeAlign (ICRA 2024), the closest published
competitor, aligning the same agents from the same boxes by a salient-object
graph instead of by a learned correspondence. Running it here rather than in
the authors' repository is what makes the row comparable at all -- same
detector, same fusion, same AP -- and **late fusion + FreeAlign is the pairing
their own paper never reports**, since every result there sits on an
intermediate-fusion backbone still shipping feature maps.

With ``oracle_match`` two further conditions join them,
``oracle_match_uniform`` and ``oracle_match_ivw``: the *same* estimator with
the learned correspondence replaced by the ground-truth assignment. They are
not a method, they are a ceiling -- the most any improvement to cross-agent
matching, by camera or by anything else, could be worth on this data.

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
from embedding_aware_belt_fusion.alignformer.freealign import (
    FreeAlignConfig,
    freealign_estimate,
)
from embedding_aware_belt_fusion.alignformer.fusion import (
    correct_boxes,
    correct_detections,
    late_fuse,
)
from embedding_aware_belt_fusion.alignformer.model import DEFAULT_HEADING_LAMBDA
from embedding_aware_belt_fusion.alignformer.oracle import (
    oracle_assignment,
    oracle_pose_estimate,
)
from embedding_aware_belt_fusion.alignformer.shrinkage import ShrinkageCalibration, shrink
from embedding_aware_belt_fusion.alignformer.stage2 import is_fallback
from embedding_aware_belt_fusion.alignformer.train import embed_batch
from embedding_aware_belt_fusion.alignformer.trunk import MAX_OBJECTS
from embedding_aware_belt_fusion.alignformer.variance import (
    UNWEIGHTED,
    CorrespondenceVarianceModel,
)
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
# The FreeAlign REIMPLEMENTATION (task 20), applied to this same late-fusion
# pipeline on these same detections. That pairing -- late fusion + FreeAlign --
# is the row FreeAlign's own paper never reports: every result there pairs it
# with an intermediate-fusion backbone, so the system still ships feature maps.
# See alignformer.freealign for what is and is not ported.
FREEALIGN = "freealign"
# The oracle-correspondence ceiling (task 21): AlignFormer's own estimator with
# the learned Sinkhorn correspondence replaced by the ground-truth one-to-one
# assignment. Two weightings, because the deployed model weights its
# correspondences by inverse variance and a ceiling that bounded only the
# matching would leave the weighting scheme unbounded. See alignformer.oracle.
ORACLE_MATCH_UNIFORM = "oracle_match_uniform"
ORACLE_MATCH_IVW = "oracle_match_ivw"
ORACLE_MATCH_CONDITIONS = (ORACLE_MATCH_UNIFORM, ORACLE_MATCH_IVW)

# Slices by how many ground-truth objects the two agents both detected. The
# boundary at 3 is not arbitrary: it is exactly where a relative-distance graph
# stops being degenerate (see shared_object_count).
SHARED_NONE = "shared_0"
SHARED_SPARSE = "shared_1_2"
SHARED_DENSE = "shared_3plus"
SHARED_BUCKETS = (SHARED_NONE, SHARED_SPARSE, SHARED_DENSE)

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
        # Task 20. ``answered`` drops the pairs the estimator declined, which
        # is the conditioning FreeAlign's published pose figures are reported
        # under; without it the two are not comparable as printed. The
        # shared-object slices are where the two methods differ structurally.
        self.answered = _PoseSubset()
        self.shared = {name: _PoseSubset() for name in SHARED_BUCKETS}
        self.shared_answered = {name: _PoseSubset() for name in SHARED_BUCKETS}

    def update(
        self,
        psi_hat: float,
        t_hat: Sequence[float],
        psi_true: float,
        t_true: Sequence[float],
        *,
        fell_back: bool,
        shared_count: int,
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
        if shared_count == 0:
            self.unalignable.add(*terms)
        bucket = shared_object_bucket(shared_count)
        self.shared[bucket].add(*terms)
        if not fell_back:
            self.answered.add(*terms)
            self.shared_answered[bucket].add(*terms)
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
        metrics["coverage"] = (
            self.answered.pairs / self.all.pairs if self.all.pairs else None
        )
        subsets = [
            ("unalignable", self.unalignable),
            ("within_training_range", self.within_training_range),
            ("beyond_training_range", self.beyond_training_range),
            ("within_40m", self.within_40m),
            ("beyond_40m", self.beyond_40m),
            ("answered", self.answered),
        ]
        for name in SHARED_BUCKETS:
            subsets.append((name, self.shared[name]))
            subsets.append((f"{name}_answered", self.shared_answered[name]))
            answered = self.shared_answered[name].pairs
            total = self.shared[name].pairs
            metrics[f"{name}_coverage"] = answered / total if total else None
            metrics[f"{name}_fraction"] = (
                total / self.all.pairs if self.all.pairs else None
            )
        for name, subset in subsets:
            for key, value in subset.compute().items():
                metrics[f"{name}_{key}"] = value
        return metrics


def shared_object_count(
    left: Sequence[Optional[str]], right: Sequence[Optional[str]]
) -> int:
    """How many ground-truth objects both agents detected.

    ``None`` means "matched no ground-truth object", so two ``None`` detections
    are different objects, not the same one -- the same rule
    ``dataset.correspondence_indices`` applies.

    This count, not merely whether it is positive, is what separates the two
    methods structurally (task 20). FreeAlign's evidence is pairwise
    *distances*: one shared object is a 1-node graph with no edge, two give a
    single scalar that fixes neither rotation nor the reflection, and three are
    needed before a distance graph rigidly determines SE(2). AlignFormer's
    heading virtual points solve the full SE(2) from one. Averaged over the
    whole split that difference is diluted eightfold, so it is reported as its
    own slice.
    """
    return len(
        {value for value in left if value is not None}
        & {value for value in right if value is not None}
    )


def shared_object_bucket(count: int) -> str:
    """The slice name for a pair sharing ``count`` ground-truth objects."""
    if count == 0:
        return SHARED_NONE
    return SHARED_SPARSE if count <= 2 else SHARED_DENSE


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
    oracle_match: bool = False,
    freealign: Optional[FreeAlignConfig] = None,
) -> Tuple[
    Dict[str, List[Tuple[Tensor, Tensor]]],
    List[Tensor],
    Dict[str, Dict[str, Dict[str, float]]],
    List[Tensor],
    List[List[int]],
]:
    """Fuse every frame under every condition; return predictions, truth and pose stats.

    Returns ``(predictions_by_condition, ground_truth, pose_stats_by_condition,
    intermediate_convention_ground_truth, shared_object_counts_per_frame)``.
    ``pose_stats_by_condition`` is ``{estimator: {sigma key: metrics}}``.
    ``ground_truth`` is shared by every condition -- the frames and their
    labels do not change, only the correction applied to the CAV boxes does --
    and the last entry carries, per frame, how many ground-truth objects each
    of its ego-CAV pairs shares, which is what lets AP be sliced the way the
    pose error is.

    ``shrinkage``, when given, is applied to every estimate before the boxes
    are moved, so the ``alignformer`` condition measures what the method would
    actually deploy. The calibration is fitted on the validation split at
    sigma = 0 and is the same one at every sigma here -- a per-sigma factor
    would be fitting the sweep it is being scored on.

    ``oracle_match`` adds the two :data:`ORACLE_MATCH_CONDITIONS`, which are
    the deployed estimator run on the ground-truth correspondence. They share
    the frame's detections, noise draws and shrinkage with ``alignformer``, so
    the difference between them is the matching and nothing else.

    ``freealign``, when given, adds the :data:`FREEALIGN` condition -- the
    reimplementation of Lei et al.'s ICRA 2024 method (see
    ``alignformer.freealign``) -- on the SAME detections, the SAME noise draws
    and through the SAME fusion and evaluator, so the only thing that differs
    between that row and ``alignformer`` is the alignment algorithm.
    ``shrinkage`` is deliberately NOT applied to it: the calibration is fitted
    on AlignFormer's own residuals and FreeAlign has no such step, so applying
    it would be scoring the competitor through our calibration.
    """
    from opencood.utils import box_utils
    from opencood.utils.transformation_utils import x1_to_x2

    from embedding_aware_belt_fusion.alignformer.evaluate import (
        _build_test_frame,
        _cav_content,
        _frame_identity,
        _pose_correction,
    )

    variance_models = _oracle_variance_models(modules) if oracle_match else {}
    estimators = [ALIGNFORMER] + list(variance_models)
    if freealign is not None:
        estimators.append(FREEALIGN)
    heading_lambda = float(
        getattr(modules["pose"], "heading_lambda", DEFAULT_HEADING_LAMBDA)
    )

    conditions = [ORACLE] + [
        condition_key(name, sigma)
        for sigma in sigmas
        for name in [UNCORRECTED] + estimators
    ]
    predictions: Dict[str, List[Tuple[Tensor, Tensor]]] = {key: [] for key in conditions}
    ground_truth: List[Tensor] = []
    intermediate_ground_truth: List[Tensor] = []
    # One entry per frame: how many ground-truth objects each of the frame's
    # ego-CAV pairs shares. Independent of sigma (the detections do not move
    # with the noise), and the key that lets AP be sliced the way the pose
    # error already is.
    frame_shared_counts: List[List[int]] = []
    stats = {
        name: {sigma: _PoseStats(training_comm_range_m) for sigma in sigmas}
        for name in estimators
    }
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
        cav_keys = sorted(key for key in detections if key != "ego")
        # Fixed for the frame: the detections do not move with sigma, so the
        # shared-object count of each pair is a property of the frame alone.
        shared_this_frame = [
            shared_object_count(ego_pack["gt_ids"], packs[key]["gt_ids"])
            for key in cav_keys
        ]
        for sigma in sigmas:
            uncorrected = [detections["ego"]]  # ego's own transform is the identity
            # Every estimator corrects the SAME noisy detections, so the fused
            # sets differ by the estimate alone.
            aligned = {name: [detections["ego"]] for name in estimators}
            for agent, key in enumerate(cav_keys):
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

                pair = _object_set(ego_pack, cav_pack)
                estimates = {
                    ALIGNFORMER: _estimate(modules, pair, ablate_embeddings)
                }
                if freealign is not None:
                    estimates[FREEALIGN] = freealign_estimate(pair, freealign)
                if variance_models:
                    assignment = oracle_assignment(
                        ego_pack["gt_ids"],
                        cav_pack["gt_ids"],
                        device=pair["ego_boxes"].device,
                    )
                    for name, variance_model in variance_models.items():
                        estimates[name] = oracle_pose_estimate(
                            pair,
                            assignment,
                            heading_lambda=heading_lambda,
                            variance_model=variance_model,
                        )

                dx, dy, dpsi_deg = relative_pose_error(ego_pose, poses[key], noisy_pose)
                shared = shared_this_frame[agent]
                for name, estimate in estimates.items():
                    # FreeAlign is scored as published: no shrinkage, which is
                    # AlignFormer's calibration and not part of their method.
                    if shrinkage is not None and name != FREEALIGN:
                        estimate = shrink(estimate, shrinkage)
                    aligned[name].append(
                        correct_detections(
                            noisy_detections, estimate.psi[0], estimate.t[0]
                        )
                    )
                    stats[name][sigma].update(
                        float(estimate.psi[0].item()),
                        (float(estimate.t[0, 0]), float(estimate.t[0, 1])),
                        float(np.radians(dpsi_deg)),
                        (dx, dy),
                        fell_back=bool(is_fallback(estimate)[0].item()),
                        shared_count=shared,
                        distance_m=float(np.hypot(
                            poses[key][0] - ego_pose[0], poses[key][1] - ego_pose[1]
                        )),
                    )

            predictions[condition_key(UNCORRECTED, sigma)].append(
                _fuse(uncorrected, nms_threshold)
            )
            for name, corrected in aligned.items():
                predictions[condition_key(name, sigma)].append(
                    _fuse(corrected, nms_threshold)
                )

        gt_corners = postprocessor.generate_gt_bbx(batch)
        gt_boxes = box_utils.corner_to_center(
            gt_corners.detach().cpu().numpy(), order=postprocessor.params["order"]
        )
        ground_truth.append(torch.from_numpy(gt_boxes).float())
        frame_shared_counts.append(shared_this_frame)
        intermediate_ground_truth.append(
            _intermediate_convention_ground_truth(dataset, base, ego_pose)
        )

        if (index + 1) % _PROGRESS_INTERVAL == 0:
            rate = (index + 1) / (time.time() - started)
            print(f"  {index + 1}/{frame_count} frames  {rate:.2f} fr/s", flush=True)

    return (
        predictions,
        ground_truth,
        {
            name: {f"sigma_{sigma:g}m": tally.compute() for sigma, tally in by_sigma.items()}
            for name, by_sigma in stats.items()
        },
        intermediate_ground_truth,
        frame_shared_counts,
    )


def _oracle_variance_models(modules) -> Dict[str, CorrespondenceVarianceModel]:
    """The correspondence weightings the two oracle conditions are measured under.

    The IVW one is read off the loaded pose head rather than rebuilt from a
    config, so it is by construction the weighting this very checkpoint
    deploys: the two conditions then differ in the correspondence alone, which
    is the whole claim. A pose head with no ``variance_model`` is head A, which
    has no correspondence to replace.
    """
    variance_model = getattr(modules["pose"], "variance_model", None)
    if variance_model is None:
        raise ValueError(
            "the oracle-correspondence conditions need head B; the loaded "
            "checkpoint's pose head builds no correspondence to replace"
        )
    return {
        ORACLE_MATCH_UNIFORM: UNWEIGHTED,
        ORACLE_MATCH_IVW: variance_model,
    }


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


def _estimate(modules, batch: Mapping[str, Tensor], ablate: bool):
    """Run the frozen embedding head and the pose head over one ego-CAV pair."""
    enriched = embed_batch(modules["embedding"], batch, ablate=ablate)
    return modules["pose"](enriched)


def _fuse(
    detections_by_agent: Sequence[AgentDetections], nms_threshold: float
) -> Tuple[Tensor, Tensor]:
    """Late-fuse one frame and move the result to the CPU for accumulation."""
    boxes, scores = late_fuse(detections_by_agent, nms_threshold)
    return boxes.detach().cpu(), scores.detach().cpu()
