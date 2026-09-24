"""Head B's solver on the GROUND-TRUTH correspondence: a ceiling on matching.

A camera-augmented embedding has been proposed to "enhance semantics" for
cross-agent association. This module exists to bound what any such improvement
could possibly be worth on OPV2V, before one is built. It replaces the learned
Sinkhorn correspondence with the one-to-one assignment ``gt_ids`` already
carries -- the best any matcher could ever produce -- and leaves every other
part of the estimator exactly as deployed.

"Exactly as deployed" is load-bearing, so it is enforced structurally rather
than by inspection: the reduction to virtual points and the closed-form solve
are :func:`~embedding_aware_belt_fusion.alignformer.model.reduce_correspondence`
and :func:`~embedding_aware_belt_fusion.alignformer.model.solve_pose`, the same
two functions ``AlignFormerB.forward`` calls. Only the ``(B, M, N)`` matrix
handed to them differs. The heading pi-fold, the inverse-variance weighting,
the ``MIN_MATCH_MASS`` fallback and the shrinkage applied downstream are all
therefore whatever the deployed model does, by construction and not by
duplication.

Two weightings are worth measuring and both are offered through
``variance_model``:

- ``variance.UNWEIGHTED`` puts weight 1.0 on every true pair. This bounds the
  weighting scheme as well as the matching: it is what a perfect matcher with
  no notion of per-correspondence precision would give.
- the deployed inverse-variance model is the like-for-like comparison against
  the shipped estimator, which differs from it in the correspondence alone.

An ego object whose ground-truth id appears in no CAV row gets an all-zero row
and therefore zero mass -- the same thing the Sinkhorn dustbin does to an
object the CAV never saw, and the reason this is a ceiling on *matching* rather
than a ceiling on detection recall.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from torch import Tensor

from embedding_aware_belt_fusion.alignformer.dataset import correspondence_indices
from embedding_aware_belt_fusion.alignformer.model import (
    PoseEstimate,
    _is_empty,
    _zero_estimate,
    reduce_correspondence,
    solve_pose,
)
from embedding_aware_belt_fusion.alignformer.variance import CorrespondenceVarianceModel


def oracle_assignment(
    ego_ids: Sequence[Optional[str]],
    cav_ids: Sequence[Optional[str]],
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """``(1, M, N)`` one-hot ground-truth assignment for one ego-CAV pair.

    Built from :func:`alignformer.dataset.correspondence_indices`, which is the
    same matching rule the training loss and the Top-1 metric use, so the
    ceiling is measured against the target the matcher was actually trained
    towards. ``None`` ids match nothing and duplicate ids keep their first
    occurrence, so the result is one-to-one in both directions.
    """
    ego_match, _ = correspondence_indices(list(ego_ids), list(cav_ids))
    assignment = torch.zeros((1, len(ego_ids), len(cav_ids)), dtype=dtype, device=device)
    rows: List[int] = [i for i, target in enumerate(ego_match.tolist()) if target >= 0]
    if rows:
        columns = [int(ego_match[i]) for i in rows]
        assignment[0, rows, columns] = 1.0
    return assignment


def oracle_pose_estimate(
    batch,
    assignment: Tensor,
    *,
    heading_lambda: float,
    variance_model: CorrespondenceVarianceModel,
) -> PoseEstimate:
    """Solve the SE(2) from a GIVEN correspondence matrix instead of a learned one.

    Parameters
    ----------
    batch: the same one-sample object-set batch ``AlignFormerB`` consumes; only
        the boxes, scores and masks are read, because a given correspondence
        makes the trunk's output unreachable.
    assignment: ``(B, M, N)`` non-negative correspondence weights, e.g. from
        :func:`oracle_assignment`. Masked to valid positions here, so a caller
        need not pre-mask padded columns.
    heading_lambda, variance_model: taken from the deployed pose head, so the
        only difference from ``AlignFormerB.forward`` is ``assignment``.
    """
    # Same guard, and for the same reason, as AlignFormerB.forward: with an
    # empty object set there is no correspondence to weigh and the identity
    # correction is the correct answer, not a crash-avoidance shortcut.
    if _is_empty(batch):
        return _zero_estimate(batch)

    expected = (
        batch["ego_boxes"].shape[0],
        batch["ego_boxes"].shape[1],
        batch["cav_boxes"].shape[1],
    )
    if tuple(assignment.shape) != expected:
        raise ValueError(
            f"assignment {tuple(assignment.shape)} must be {expected} "
            "(batch, ego objects, CAV objects)"
        )

    valid = batch["ego_mask"].unsqueeze(2) & batch["cav_mask"].unsqueeze(1)
    weights = assignment.to(batch["ego_boxes"].dtype) * valid
    correspondence = reduce_correspondence(weights, batch, variance_model)
    return solve_pose(correspondence, batch, heading_lambda, variance_model)
