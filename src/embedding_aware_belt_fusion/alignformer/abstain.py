"""Per-pair abstention from the estimator's OWN covariance.

At low localization error the correct action is to not correct, and head B
corrects anyway. Two guards already exist and neither has the right shape.
``procrustes.MIN_MATCH_MASS`` suppresses on **evidence volume** -- fewer than
one effective matched object -- and never asks whether the *estimated
correction* is meaningful. Positive-part shrinkage (:mod:`alignformer.shrinkage`)
does ask, but through a single global ``tau`` fitted once at sigma = 0: it can
only scale the whole population by one number, and a dense twelve-correspondence
pair and a sparse two-correspondence pair do not have the same estimator
variance. Task 23 measured exactly that dead end -- refitting the global tau
bought the low-noise end and gave back the clean case, because a smaller tau
shrinks less *everywhere*.

This module is the per-pair replacement. Nothing is learned and nothing new is
fitted: the quantity it uses is one the weighted least-squares solve already
computes and currently discards.

**The covariance.** Head B minimizes ``sum_n w_n |R(psi) q_n + t - p_n|^2`` over
``theta = (t_x, t_y, psi)``. Writing ``u_n = R(psi) q_n``, the residual's
Jacobian is ``J_n = [I_2 | perp(u_n)]`` with ``perp(a, b) = (-b, a)``, so the
weighted normal matrix is ``A = sum_n w_n J_n^T J_n`` and the estimator's
covariance is ``Cov(theta) = sigma^2 A^{-1}``. The Wald statistic of the emitted
correction against the null "no correction is needed" is then

    T^2 = theta^T Cov(theta)^{-1} theta = (1 / sigma^2) sum_n w_n |J_n theta|^2

-- no matrix inverse is ever formed, and the quadratic form reads, in plain
terms, *how far this correction moves the matched points, measured in units of
how far the fit already disagrees with itself*. Under the null it is
approximately chi-square on :data:`~alignformer.shrinkage.POSE_DIMENSIONS`
degrees of freedom.

**The scale.** ``sigma^2`` is the plug-in weighted-residual estimate rather
than a number carried over from a calibration file: a sandwich estimator would
be the alternative and is not needed, because the weights here are already a
precision model (:mod:`alignformer.variance`) and the residuals are what that
model failed to explain. Its degrees of freedom are ``2 * n_eff - 3`` -- two
coordinates per point, three parameters -- where ``n_eff`` is **Kish's**
effective sample size ``(sum w)^2 / sum w^2`` and not the nominal point count.
That distinction is the whole reason a sparse or lopsided soft match abstains:
a row whose soft-match mass sits almost entirely on one candidate carries the
information of roughly one observation, not of the eight it has entries for.

**Scale invariance.** ``weighted_se2_kabsch`` is scale-invariant in its weights,
and both :func:`variance.augmented_weights` and :func:`robust._renormalized`
rescale them. ``T^2`` inherits that invariance exactly -- numerator and
denominator are both linear in the weights -- so the weights are normalized to
``sum w = n_eff`` here only to give :data:`RESIDUAL_SD_FLOOR_M` a meaning in
metres. Pinned by a test, because a statistic that moved when the plumbing
renormalized would be measuring the plumbing.

Three decision rules are built over the statistic and no fourth:

``abstain``
    Zero the correction at or below a threshold, apply it unchanged above.
``per_pair``
    The James-Stein positive-part factor ``max(0, 1 - p / T^2)`` -- the SAME
    rule :mod:`alignformer.shrinkage` applies, with the pair's own standardized
    statistic in place of the global-tau one. :func:`shrinkage.shrinkage_factor`
    is reused rather than re-derived, so the two cannot drift apart.
``both``
    A hard threshold, then the positive-part factor above it.

``both`` at a threshold of 0 -- and at 3, the pooled dimension -- IS
``per_pair``, because the positive-part rule already returns exactly zero
there. That equivalence is asserted in the tests so the three arms cannot
silently become two.

An abstained pair emits ``(psi, t) = (0, 0)``, which is what
:func:`stage2.is_fallback` reads, so **coverage falls out of the existing pose
statistics** and can be reported beside every AP number. It has to be: an
abstaining method answers less often and therefore looks better on what it does
answer, which is the reporting trap this project criticized in the published
competitor's own numbers and which now applies to us.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from embedding_aware_belt_fusion.alignformer.shrinkage import (
    POSE_DIMENSIONS,
    shrinkage_factor,
)

# ``none`` is the deployed behaviour: no per-pair decision at all, so an arm
# configured with it reproduces what it wraps bit for bit.
NONE = "none"
ABSTAIN = "abstain"
PER_PAIR = "per_pair"
BOTH = "both"
ABSTENTION_MODES = (NONE, ABSTAIN, PER_PAIR, BOTH)

# Lower bound on the fitted per-coordinate residual sd, in metres. Same number
# and same reason as ``robust.SCALE_FLOOR_M``: the detector's own fitted
# per-detection RMS centre error is 0.2516 m (``variance.py``), so nothing real
# is resolved at 5 cm, and a fit that happens to be exact must not divide by
# zero. It is the only place the statistic is not scale-invariant, which is why
# the weights are normalized before it is applied.
RESIDUAL_SD_FLOOR_M = 0.05
# Floor on the residual degrees of freedom. Below about 1.5 effective points
# the SE(2) is not identified at all; such a pair gets a large variance, a
# small statistic, and therefore abstains -- which is the intended answer, not
# a guard against one.
DOF_FLOOR = 1.0


@dataclass(frozen=True)
class AbstentionConfig:
    """One per-pair decision rule, and the name the sweep reports it under.

    Parameters
    ----------
    mode: one of :data:`ABSTENTION_MODES`.
    threshold: the statistic at or below which the correction is abstained
        from. Meaningful for ``abstain`` and ``both`` only; ``per_pair``
        refuses a non-zero one rather than ignoring it, because a threshold
        silently dropped is a constant a reader would believe was in force.
    dimensions: the pooled problem's dimension in the positive-part rule.
    """

    mode: str = NONE
    threshold: float = 0.0
    dimensions: int = POSE_DIMENSIONS

    def __post_init__(self) -> None:
        if self.mode not in ABSTENTION_MODES:
            raise ValueError(
                f"mode must be one of {ABSTENTION_MODES}, got {self.mode!r}"
            )
        if not (self.threshold >= 0.0 and torch.isfinite(torch.tensor(self.threshold))):
            raise ValueError(
                f"threshold must be finite and non-negative, got {self.threshold}"
            )
        if self.mode in (NONE, PER_PAIR) and self.threshold != 0.0:
            raise ValueError(
                f"mode {self.mode!r} has no use for a threshold, got {self.threshold}"
            )
        if self.dimensions < 1:
            raise ValueError(f"dimensions must be at least 1, got {self.dimensions}")

    @property
    def enabled(self) -> bool:
        return self.mode != NONE

    @property
    def name(self) -> str:
        """The sweep condition key: ``alignformer_<mode>[_<threshold>]``."""
        if self.mode == PER_PAIR:
            return f"alignformer_{self.mode}"
        return f"alignformer_{self.mode}_{self.threshold:g}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "threshold": self.threshold,
            "dimensions": self.dimensions,
            "name": self.name,
            "enabled": self.enabled,
        }

    @classmethod
    def parse(cls, spec: str) -> "AbstentionConfig":
        """``"abstain:7.8147"``, ``"both:16.2662"`` or ``"per_pair"``."""
        mode, _, threshold = str(spec).partition(":")
        if threshold == "":
            return cls(mode=mode)
        try:
            value = float(threshold)
        except ValueError as error:
            raise ValueError(
                f"{spec!r} is not a <mode>[:<threshold>] abstention spec"
            ) from error
        return cls(mode=mode, threshold=value)


def effective_sample_size(weights: Tensor) -> Tensor:
    """``(B,)`` Kish effective sample size ``(sum w)^2 / sum w^2``.

    Equal to the point count when every weight is equal, and collapsing towards
    one when a single correspondence carries the mass. A row with no weight at
    all returns zero, which floors the degrees of freedom downstream instead of
    dividing by nothing.
    """
    weights = weights.clamp_min(0.0)
    total = weights.sum(dim=1)
    squared = (weights * weights).sum(dim=1)
    tiny = torch.finfo(weights.dtype).tiny
    return torch.where(
        squared > 0, total * total / squared.clamp_min(tiny), torch.zeros_like(total)
    )


def _normalized(weights: Tensor) -> Tensor:
    """``weights`` rescaled to total ``n_eff``, so the mean weight is one.

    The statistic is invariant to this (see the module docstring); it is done
    so that :data:`RESIDUAL_SD_FLOOR_M` is a length in metres rather than a
    number whose meaning depends on how the caller happened to scale its
    weights.
    """
    weights = weights.clamp_min(0.0)
    total = weights.sum(dim=1, keepdim=True)
    tiny = torch.finfo(weights.dtype).tiny
    scale = torch.where(
        total > 0,
        effective_sample_size(weights).unsqueeze(-1) / total.clamp_min(tiny),
        torch.zeros_like(total),
    )
    return weights * scale


def residual_variance(residual: Tensor, weights: Tensor) -> Tensor:
    """``(B,)`` per-coordinate residual variance of the weighted fit.

    Parameters
    ----------
    residual: ``(B, N, 2)`` signed residuals of the final solve.
    weights: ``(B, N)`` the weights that solve actually used.

    ``sum_n w_n |r_n|^2 / (2 n_eff - 3)``: two coordinates per point, three
    parameters, and Kish's effective count rather than the nominal one. Floored
    at ``RESIDUAL_SD_FLOOR_M^2``.
    """
    if residual.dim() != 3 or residual.shape[-1] != 2:
        raise ValueError(f"residual must be (B, N, 2), got {tuple(residual.shape)}")
    if weights.shape != residual.shape[:2]:
        raise ValueError(
            f"weights {tuple(weights.shape)} must be {tuple(residual.shape[:2])}"
        )
    normalized = _normalized(weights)
    weighted = (normalized * (residual * residual).sum(dim=-1)).sum(dim=1)
    dof = (2.0 * effective_sample_size(weights) - POSE_DIMENSIONS).clamp_min(DOF_FLOOR)
    return (weighted / dof).clamp_min(RESIDUAL_SD_FLOOR_M ** 2)


def wald_statistic(
    p: Tensor, q: Tensor, weights: Tensor, psi: Tensor, t: Tensor
) -> Tensor:
    """``(B,)`` ``T^2`` of the emitted correction against the fit's own covariance.

    Parameters
    ----------
    p: ``(B, N, 2)`` target points and ``q``: ``(B, N, 2)`` source points,
        exactly as :func:`procrustes.weighted_se2_kabsch` took them -- the
        heading-augmented vectors, so ``N = 2M``.
    weights: ``(B, N)`` the weights the FINAL solve used. For the IRLS arm that
        is the re-weighted vector (:attr:`robust.RobustSolution.weights`), not
        the one it started from: the covariance belongs to the fit that was
        actually reported.
    psi, t: the emitted correction, ``(B,)`` and ``(B, 2)``.

    An identity correction scores exactly zero, which abstains under every
    rule; that is the same answer ``MIN_MATCH_MASS`` already gave such a pair.
    """
    if p.shape != q.shape:
        raise ValueError(f"p {tuple(p.shape)} and q {tuple(q.shape)} must match")
    if weights.shape != p.shape[:2]:
        raise ValueError(
            f"weights {tuple(weights.shape)} must be {tuple(p.shape[:2])}"
        )

    normalized = _normalized(weights)
    cos, sin = torch.cos(psi).unsqueeze(-1), torch.sin(psi).unsqueeze(-1)
    x, y = q[..., 0], q[..., 1]
    rotated = torch.stack([cos * x - sin * y, sin * x + cos * y], dim=-1)
    residual = rotated + t.unsqueeze(1) - p

    # J_n theta = t + psi * perp(R(psi) q_n): the displacement this correction
    # applies to each matched point, linearized at the reported solution.
    perpendicular = torch.stack([-rotated[..., 1], rotated[..., 0]], dim=-1)
    displacement = t.unsqueeze(1) + psi.reshape(-1, 1, 1) * perpendicular
    quadratic = (normalized * (displacement * displacement).sum(dim=-1)).sum(dim=1)

    return quadratic / residual_variance(residual, weights)


def decision_factor(statistic: Tensor, config: AbstentionConfig) -> Tensor:
    """``(B,)`` multiplier in ``[0, 1]`` this rule applies to the correction.

    One factor for the whole SE(2), exactly as :func:`shrinkage.shrink` does,
    so the result always lies on the path between the identity and the
    correction the solver proposed.
    """
    if not config.enabled:
        return torch.ones_like(statistic)

    # Inclusive at the threshold: an exact hit abstains. A rule stated as "at
    # or below" has to be implemented as one, or the constant a report quotes
    # is not the constant that ran.
    answered = (statistic > config.threshold).to(statistic.dtype)
    if config.mode == ABSTAIN:
        return answered
    factor = shrinkage_factor(statistic, config.dimensions)
    if config.mode == PER_PAIR:
        return factor
    return factor * answered


def decide(estimate, config: AbstentionConfig):
    """Return a **new** ``PoseEstimate`` with this pair's decision applied.

    A disabled configuration returns the input object itself, so an arm that
    is meant to reproduce what it wraps does so by identity rather than by
    arithmetic that happens to be a no-op.
    """
    if not config.enabled:
        return estimate
    statistic: Optional[Tensor] = getattr(estimate, "offset_statistic", None)
    if statistic is None:
        raise ValueError(
            "this estimate carries no offset statistic; solve it with "
            "solve_pose(..., statistic=True) before asking for a decision"
        )
    factor = decision_factor(statistic, config)
    return replace(
        estimate, psi=estimate.psi * factor, t=estimate.t * factor.unsqueeze(-1)
    )


__all__ = [
    "ABSTAIN",
    "ABSTENTION_MODES",
    "BOTH",
    "DOF_FLOOR",
    "NONE",
    "PER_PAIR",
    "RESIDUAL_SD_FLOOR_M",
    "AbstentionConfig",
    "decide",
    "decision_factor",
    "effective_sample_size",
    "residual_variance",
    "wald_statistic",
]
