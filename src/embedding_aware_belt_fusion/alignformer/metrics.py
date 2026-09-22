"""Association and pose metrics for AlignFormer.

Cross-agent **Top-1** is the P1 gate: of the ego objects that genuinely have a
counterpart in the CAV's object set, how many score their true counterpart
highest. Two exclusions are what make the number mean that, and both are easy
to get wrong:

- Ego objects with ``ego_match == -1`` are excluded. 18.5% of ego-CAV pairs in
  OPV2V's train split share no object at all, and a large further share of
  individual objects are seen by one agent only; those rows have no correct
  answer, so counting them would dilute the metric with rows no model could
  ever get right and put the gate out of reach for reasons unrelated to the
  model.
- Padded rows (``ego_mask == False``, from :func:`alignformer.dataset.collate`
  padding both object sets to the batch maximum) are excluded for the same
  reason, plus they carry a ``-1`` match target that would otherwise look like
  a genuine unmatched object.

The pose metrics mirror ``docs/coloca_qua_baseline.md``'s definitions exactly,
so AlignFormer's numbers are directly comparable to the CoLoca-QuA baseline
table: MAE is the mean Euclidean norm of the translation residual, and the yaw
error wraps across +/-pi.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor


def _check_ego_shapes(log_assignment: Tensor, ego_match: Tensor, ego_mask: Tensor) -> int:
    """Validate the three tensors agree, and return the ego-object count.

    ``log_assignment`` is ``(B, M + 1, N + 1)`` -- the Sinkhorn output
    *including* its dustbin row and column -- so the real ego rows are the
    first ``M``, which must be exactly what ``ego_match``/``ego_mask`` describe.
    Checking it here turns an off-by-one over the dustbin row (the obvious
    mistake) into an error rather than a silently wrong metric.
    """
    if log_assignment.dim() != 3:
        raise ValueError(
            f"log_assignment must be (B, M + 1, N + 1), got {tuple(log_assignment.shape)}"
        )
    batch, rows, _ = log_assignment.shape
    ego_count = rows - 1
    expected = (batch, ego_count)
    for name, tensor in (("ego_match", ego_match), ("ego_mask", ego_mask)):
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"{name} must be {expected} for a log_assignment of shape "
                f"{tuple(log_assignment.shape)}, got {tuple(tensor.shape)}"
            )
    return ego_count


def top1_counts(
    log_assignment: Tensor,
    ego_match: Tensor,
    ego_mask: Tensor,
    *,
    include_dustbin: bool = False,
) -> Tuple[int, int]:
    """Return ``(correct, countable)`` Top-1 hits over one batch.

    Counts rather than a ratio, because the gate is measured over a whole
    split whose batches differ in size: averaging per-batch accuracies would
    silently weight a short final batch as heavily as a full one.

    Parameters
    ----------
    log_assignment: ``(B, M + 1, N + 1)`` from
        :func:`~embedding_aware_belt_fusion.alignformer.head_match.log_sinkhorn`.
    ego_match: ``(B, M)`` true CAV index per ego object, ``-1`` if it has none.
    ego_mask: ``(B, M)`` bool, True for real (non-padded) ego objects.
    include_dustbin: when True the dustbin column competes for the argmax, so
        a row that prefers "no match" over its true counterpart scores as a
        miss. Default False: the gate asks whether the *correct CAV column*
        outranks the other CAV columns, which is the association question the
        embedding is responsible for.
    """
    ego_count = _check_ego_shapes(log_assignment, ego_match, ego_mask)

    columns = log_assignment.shape[2] if include_dustbin else log_assignment.shape[2] - 1
    scores = log_assignment[:, :ego_count, :columns]
    predicted = scores.argmax(dim=2)

    countable = ego_mask & (ego_match >= 0)
    correct = countable & (predicted == ego_match)
    return int(correct.sum().item()), int(countable.sum().item())


def top1_accuracy(
    log_assignment: Tensor,
    ego_match: Tensor,
    ego_mask: Tensor,
    *,
    include_dustbin: bool = False,
) -> float:
    """Fraction of ego objects with a counterpart that rank it highest.

    Returns NaN, not 0.0, when no ego object in the batch has a counterpart:
    there is nothing to be right or wrong about, and 0.0 would read as a total
    failure. Callers accumulating over a split should use :func:`top1_counts`.
    """
    correct, countable = top1_counts(
        log_assignment, ego_match, ego_mask, include_dustbin=include_dustbin
    )
    if countable == 0:
        return math.nan
    return correct / countable


def dustbin_mass_sums(log_assignment: Tensor, ego_mask: Tensor) -> Tuple[float, int]:
    """Return ``(total dustbin mass, real ego rows)`` over one batch.

    Sums rather than a ratio, for the same reason :func:`top1_counts` returns
    counts: batches differ in how many real ego rows they contain.

    Padded rows are excluded -- their score row is masked to a large negative
    everywhere except the dustbin, so they are all-dustbin by construction and
    would drag the average towards 1.0 for purely structural reasons.
    """
    if log_assignment.dim() != 3:
        raise ValueError(
            f"log_assignment must be (B, M + 1, N + 1), got {tuple(log_assignment.shape)}"
        )
    ego_count = log_assignment.shape[1] - 1
    if tuple(ego_mask.shape) != (log_assignment.shape[0], ego_count):
        raise ValueError(
            f"ego_mask must be {(log_assignment.shape[0], ego_count)}, "
            f"got {tuple(ego_mask.shape)}"
        )

    weights = ego_mask.to(log_assignment.dtype)
    dustbin = log_assignment[:, :ego_count, -1].exp()
    return float((dustbin * weights).sum().item()), int(ego_mask.sum().item())


def dustbin_mass_fraction(log_assignment: Tensor, ego_mask: Tensor) -> float:
    """Mean share of each real ego row's assignment mass sitting on the dustbin.

    The diagnostic for a collapsed matcher: a model that has learned to declare
    everything unmatched drives this towards 1.0 while its ``match_nll`` still
    falls, because most objects genuinely are unmatched.
    """
    mass, rows = dustbin_mass_sums(log_assignment, ego_mask)
    if rows == 0:
        return math.nan
    return mass / rows


def chance_top1_sums(
    ego_match: Tensor, ego_mask: Tensor, cav_mask: Tensor
) -> Tuple[float, int]:
    """Return ``(expected correct, countable)`` for uniform random matching.

    The reference the gate must be read against: a row facing ``n`` real CAV
    columns is right with probability ``1 / n`` by luck alone. Summing per row
    rather than using one ``1 / mean(n)`` keeps the baseline exact for a split
    whose object counts vary from frame to frame.
    """
    if tuple(ego_match.shape) != tuple(ego_mask.shape):
        raise ValueError(
            f"ego_match {tuple(ego_match.shape)} and ego_mask {tuple(ego_mask.shape)} disagree"
        )
    if cav_mask.shape[0] != ego_mask.shape[0]:
        raise ValueError(
            f"cav_mask {tuple(cav_mask.shape)} and ego_mask {tuple(ego_mask.shape)} "
            "disagree on batch size"
        )

    countable = ego_mask & (ego_match >= 0)
    cav_counts = cav_mask.sum(dim=1).clamp_min(1).to(torch.float64)
    per_row = countable.to(torch.float64) / cav_counts.unsqueeze(1)
    return float(per_row.sum().item()), int(countable.sum().item())


def nearest_centre_top1_counts(
    ego_boxes: Tensor,
    cav_boxes: Tensor,
    ego_match: Tensor,
    ego_mask: Tensor,
    cav_mask: Tensor,
) -> Tuple[int, int]:
    """Top-1 counts for the geometry-only control: nearest CAV box centre.

    Matching by proximity needs no embedding and no training, so it bounds how
    much of a learned Top-1 score is actually attributable to the embedding. At
    the low noise stage 1 trains under, vehicles are metres apart while the
    localization error is sub-metre, so this control is expected to be strong --
    which is exactly why it has to be reported next to the gate rather than
    left implicit.
    """
    if ego_boxes.shape[:2] != ego_mask.shape or cav_boxes.shape[:2] != cav_mask.shape:
        raise ValueError(
            f"boxes {tuple(ego_boxes.shape)}/{tuple(cav_boxes.shape)} disagree with "
            f"masks {tuple(ego_mask.shape)}/{tuple(cav_mask.shape)}"
        )

    distance = torch.cdist(ego_boxes[..., :2].float(), cav_boxes[..., :2].float())
    # Padded CAV columns sit at the origin and would otherwise win outright for
    # any ego object near it.
    distance = distance.masked_fill(~cav_mask.unsqueeze(1), float("inf"))

    predicted = distance.argmin(dim=2)
    countable = ego_mask & (ego_match >= 0)
    correct = countable & (predicted == ego_match)
    return int(correct.sum().item()), int(countable.sum().item())


def _check_pose_shapes(psi: Tensor, t: Tensor, name: str) -> None:
    if psi.dim() != 1:
        raise ValueError(f"{name}_psi must be (B,), got {tuple(psi.shape)}")
    if t.dim() != 2 or t.shape[1] != 2:
        raise ValueError(f"{name}_t must be (B, 2), got {tuple(t.shape)}")
    if psi.shape[0] != t.shape[0]:
        raise ValueError(
            f"{name}_psi and {name}_t disagree on batch size: "
            f"{tuple(psi.shape)} vs {tuple(t.shape)}"
        )


def translation_mae(
    psi_pred: Tensor, t_pred: Tensor, psi_true: Tensor, t_true: Tensor
) -> float:
    """Mean Euclidean norm of the translation residual, in metres.

    This is the reading of "MAE" that ``docs/coloca_qua_baseline.md`` settled on
    for the CoLoca-QuA table, so the two are directly comparable.

    The yaw arguments are validated but do not enter the value: the translation
    residual is ``|t_pred - t_true|`` by definition, and yaw is reported
    separately by :func:`yaw_mae_deg`. They are part of the signature so that
    both pose metrics are called with the same ``(psi, t)`` pair and neither can
    be handed one half of a prediction by mistake.
    """
    _check_pose_shapes(psi_pred, t_pred, "pred")
    _check_pose_shapes(psi_true, t_true, "true")
    if t_pred.shape != t_true.shape:
        raise ValueError(
            f"prediction {tuple(t_pred.shape)} and target {tuple(t_true.shape)} "
            "disagree on batch size"
        )
    return float(torch.linalg.norm(t_pred - t_true, dim=-1).mean().item())


def yaw_mae_deg(psi_pred: Tensor, psi_true: Tensor) -> float:
    """Mean absolute yaw residual in degrees, wrapped onto ``(-pi, pi]``.

    Wrapping matters at the seam: a prediction of ``pi - 0.01`` against a truth
    of ``-pi + 0.01`` is off by 0.02 rad, not by very nearly a full turn.
    """
    if psi_pred.shape != psi_true.shape:
        raise ValueError(
            f"psi_pred {tuple(psi_pred.shape)} and psi_true {tuple(psi_true.shape)} disagree"
        )
    residual = psi_pred - psi_true
    wrapped = torch.atan2(torch.sin(residual), torch.cos(residual))
    return float(torch.rad2deg(wrapped.abs()).mean().item())
