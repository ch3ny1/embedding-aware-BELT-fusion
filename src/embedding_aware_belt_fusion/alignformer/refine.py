"""The exact re-solve: hard correspondences under the soft estimate, then an
unweighted closed-form SE(2) fit, iterated with a shrinking gate.

What it answers. On V2X-Real test the pairs sharing three or more objects are
two thirds of the frames. There FreeAlign at sigma 2 m sits at 90 % of the
oracle while the deployed soft solve loses 0.10 AP@0.7 to it, with a mean
answered residual of 0.8 m against 0.4 m in the clean case. The soft solve's
input is one virtual CAV point per ego object, a Sinkhorn-mass mixture of the
candidates; as the tokens get noisier the row spreads over neighbours and the
mixture lands between them, so the least-squares answer is blurred rather than
wrong. FreeAlign's advantage in that regime is that it *decides* the
correspondence and then fits exactly. This module gives the deployed estimate
the same second step.

The step is plain ICP, initialised by the soft estimate: move the CAV boxes by
the current (psi, t), take the mutually-nearest ego/CAV centre pairs within a
gate, solve SE(2) exactly over them with the heading augmentation and the
heading fold head B already uses, and repeat with a tighter gate. Everything
here is inference-time and parameter-free apart from the gate schedule and
the evidence floor, both chosen on validation.

Two things are deliberate. The refinement **keeps the soft fit's decision
inputs** (the Wald statistic, its degrees of freedom and the precision) and
changes only ``(psi, t)``, so the deployed decision rules apply to the refined
correction unchanged and the arm differs from the one it refines in the solve
alone. And it **engages only above an evidence floor** of hard pairs, because
the pairs sharing one or two objects are where the soft fit beats FreeAlign
(+0.05 AP@0.7 over doing nothing at sigma 2 on test, FreeAlign abstains) and a
nearest-neighbour step with two points will happily snap to the wrong
neighbour.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence, Tuple

import torch
from torch import Tensor

from embedding_aware_belt_fusion.alignformer.boxes import BOX_YAW
from embedding_aware_belt_fusion.alignformer.procrustes import (
    augment_with_heading,
    heading_orientation,
    weighted_se2_kabsch,
)

REFINE_NONE = "none"
ICP = "icp"
REFINE_MODES = (REFINE_NONE, ICP)
# The gate schedule: the first gate must admit the soft fit's own residual
# (0.8 m mean on dense pairs at sigma 2), the last should be near the
# detector's per-detection centre error (0.25 m RMS, variance.py) so a
# neighbour a lane away (3.5 m) can never be taken for the object itself.
DEFAULT_GATES_M = (2.0, 1.0, 0.5)
# Hard correspondences below which the soft estimate is kept: three is where a
# relative-distance graph stops being degenerate and is the boundary the
# sweep's sparse slice (``shared_1_2``) is drawn at.
DEFAULT_MIN_PAIRS = 3
REFINED_SUFFIX = "_icp"


@dataclass(frozen=True)
class RefineConfig:
    """Every tunable of the re-solve, in one immutable place."""

    mode: str = REFINE_NONE
    gates_m: Tuple[float, ...] = DEFAULT_GATES_M
    min_pairs: int = DEFAULT_MIN_PAIRS

    def __post_init__(self) -> None:
        if self.mode not in REFINE_MODES:
            raise ValueError(f"mode must be one of {REFINE_MODES}, got {self.mode!r}")
        if len(self.gates_m) == 0:
            raise ValueError("gates_m needs at least one gate")
        if any(not (g > 0.0 and math.isfinite(g)) for g in self.gates_m):
            raise ValueError(f"every gate must be finite and positive, got {self.gates_m}")
        if self.min_pairs < 1:
            raise ValueError(f"min_pairs must be >= 1, got {self.min_pairs}")

    @property
    def enabled(self) -> bool:
        return self.mode != REFINE_NONE

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "gates_m": list(self.gates_m),
            "min_pairs": self.min_pairs,
            "enabled": self.enabled,
        }


def refined_name(condition: str) -> str:
    """The refined arm's condition name: ``alignformer_irls`` -> ``alignformer_icp``,
    any decision arm -> that name plus ``_icp``."""
    if condition == "alignformer_irls":
        return "alignformer" + REFINED_SUFFIX
    return condition + REFINED_SUFFIX


def _moved(centres: Tensor, psi: Tensor, t: Tensor) -> Tensor:
    """``R(psi) centres + t`` for one sample: ``(N, 2)``."""
    cos, sin = torch.cos(psi), torch.sin(psi)
    x, y = centres[:, 0], centres[:, 1]
    return torch.stack([cos * x - sin * y, sin * x + cos * y], dim=-1) + t


def mutual_nearest(ego: Tensor, cav: Tensor, gate_m: float) -> Tuple[Tensor, Tensor]:
    """Indices of the mutually-nearest ``(ego, cav)`` centre pairs closer than ``gate_m``.

    One-to-one by construction: a CAV object is taken only by the ego object
    it is nearest to, and only if that ego object is nearest to it in turn.
    """
    if ego.shape[0] == 0 or cav.shape[0] == 0:
        empty = torch.zeros(0, dtype=torch.long, device=ego.device)
        return empty, empty
    distance = torch.cdist(ego, cav)
    nearest_cav = distance.argmin(dim=1)
    nearest_ego = distance.argmin(dim=0)
    ego_idx = torch.arange(ego.shape[0], device=ego.device)
    mutual = nearest_ego[nearest_cav] == ego_idx
    within = distance[ego_idx, nearest_cav] < gate_m
    keep = mutual & within
    return ego_idx[keep], nearest_cav[keep]


def _exact_solve(
    ego_boxes: Tensor, cav_boxes: Tensor, ego_idx: Tensor, cav_idx: Tensor, heading_lambda: float
) -> Tuple[Tensor, Tensor]:
    """Unweighted SE(2) fit over the hard pairs, heading-augmented and folded."""
    ego_yaw = ego_boxes[ego_idx, BOX_YAW]
    cav_yaw = cav_boxes[cav_idx, BOX_YAW]
    # Fold each CAV heading onto its ego partner's half-plane (procrustes.py:
    # the detector reports an axis, not a direction).
    sign = heading_orientation(ego_yaw.unsqueeze(0), cav_yaw.unsqueeze(0))[0].diagonal()
    folded_yaw = torch.where(sign < 0, cav_yaw + math.pi, cav_yaw)
    target = augment_with_heading(ego_boxes[ego_idx, :2].unsqueeze(0), ego_yaw.unsqueeze(0), heading_lambda)
    source = augment_with_heading(cav_boxes[cav_idx, :2].unsqueeze(0), folded_yaw.unsqueeze(0), heading_lambda)
    weights = torch.ones(1, target.shape[1], dtype=target.dtype, device=target.device)
    psi, t = weighted_se2_kabsch(target, source, weights)
    return psi[0], t[0]


def _refine_one(
    psi: Tensor, t: Tensor, ego_boxes: Tensor, cav_boxes: Tensor, config: RefineConfig, heading_lambda: float
) -> Tuple[Tensor, Tensor, bool]:
    """ICP over one sample's valid boxes; ``(psi, t, engaged)``."""
    ego_centres, cav_centres = ego_boxes[:, :2], cav_boxes[:, :2]
    engaged = False
    for gate in config.gates_m:
        ego_idx, cav_idx = mutual_nearest(ego_centres, _moved(cav_centres, psi, t), gate)
        if ego_idx.numel() < config.min_pairs:
            break
        psi, t = _exact_solve(ego_boxes, cav_boxes, ego_idx, cav_idx, heading_lambda)
        engaged = True
    return psi, t, engaged


def icp_refine(estimate, batch, config: RefineConfig, heading_lambda: float):
    """A **new** estimate with ``(psi, t)`` re-solved exactly; the decision inputs unchanged.

    A disabled ``config`` returns the input object itself. A sample the
    evidence floor declines keeps its ``(psi, t)`` bit for bit.
    """
    if not config.enabled:
        return estimate
    ego_mask = batch.get("ego_mask")
    cav_mask = batch.get("cav_mask")
    psis, ts, engaged = [], [], []
    for b in range(estimate.psi.shape[0]):
        ego_boxes = batch["ego_boxes"][b]
        cav_boxes = batch["cav_boxes"][b]
        if ego_mask is not None:
            ego_boxes = ego_boxes[ego_mask[b]]
        if cav_mask is not None:
            cav_boxes = cav_boxes[cav_mask[b]]
        psi, t, did = _refine_one(estimate.psi[b], estimate.t[b], ego_boxes, cav_boxes, config, heading_lambda)
        psis.append(psi)
        ts.append(t)
        engaged.append(did)
    return replace(
        estimate,
        psi=torch.stack(psis),
        t=torch.stack(ts),
        refined=torch.tensor(engaged, dtype=torch.bool, device=estimate.psi.device),
    )


# ---------------------------------------------------------------------------
# The agreement rule
# ---------------------------------------------------------------------------
#
# With the re-solve in hand every pair has two estimates of its correction:
# the soft weighted fit and the exact hard fit. They are built from the same
# detections but reach the answer by different routes (a Sinkhorn mixture
# against a decided correspondence), and the clean-case loss on V2X-Real is
# pairs answered with a wrong match (-0.09 AP@0.7 on the 1-2-shared bucket at
# sigma 0). Two routes landing in the same place within detection noise is
# evidence the correction is real; landing apart is grounds to abstain. On a
# sample the re-solve did not engage on there is no second estimate, and the
# rule hands back the decision arm it wraps.

AGREEMENT_SUFFIX = "_agree_"


@dataclass(frozen=True)
class AgreementConfig:
    """``tolerance_m``: the soft/exact disagreement, in metres, above which the pair abstains."""

    tolerance_m: float

    def __post_init__(self) -> None:
        if not (self.tolerance_m > 0.0 and math.isfinite(self.tolerance_m)):
            raise ValueError(f"tolerance_m must be finite and positive, got {self.tolerance_m}")

    def to_dict(self) -> dict:
        return {"tolerance_m": self.tolerance_m}


def agreement_name(decision_arm: str, tolerance_m: float) -> str:
    return f"{decision_arm}{AGREEMENT_SUFFIX}{tolerance_m:g}"


def _wrap(angle: Tensor) -> Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def disagreement_m(soft, exact, heading_lambda: float) -> Tensor:
    """``(B,)`` metres: translation gap plus ``heading_lambda`` times the heading gap.

    The same metre-equivalent the heading augmentation uses, so one tolerance
    covers both components.
    """
    translation = (exact.t - soft.t).norm(dim=-1)
    heading = _wrap(exact.psi - soft.psi).abs()
    return translation + heading_lambda * heading


def agree(soft, exact, fallback, config: AgreementConfig, heading_lambda: float):
    """A **new** estimate: the exact fit where the two agree, zero where they
    disagree, ``fallback`` where the re-solve did not engage."""
    if exact.refined is None:
        raise ValueError("the agreement rule needs an estimate the re-solve produced (refined is None)")
    engaged = exact.refined
    agreeing = engaged & (disagreement_m(soft, exact, heading_lambda) <= config.tolerance_m)
    zero_t, zero_psi = torch.zeros_like(exact.t), torch.zeros_like(exact.psi)
    t = torch.where(engaged.unsqueeze(-1), torch.where(agreeing.unsqueeze(-1), exact.t, zero_t), fallback.t)
    psi = torch.where(engaged, torch.where(agreeing, exact.psi, zero_psi), fallback.psi)
    return replace(fallback, psi=psi, t=t)


__all__ = [
    "AGREEMENT_SUFFIX",
    "AgreementConfig",
    "agree",
    "agreement_name",
    "disagreement_m",
    "DEFAULT_GATES_M",
    "DEFAULT_MIN_PAIRS",
    "ICP",
    "REFINED_SUFFIX",
    "REFINE_MODES",
    "REFINE_NONE",
    "RefineConfig",
    "icp_refine",
    "mutual_nearest",
    "refined_name",
]
