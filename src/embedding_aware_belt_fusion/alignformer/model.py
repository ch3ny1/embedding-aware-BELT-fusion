"""The two interchangeable pose heads, over a shared trunk.

Head A regresses the pose from a mean-pooled descriptor. Head B builds a soft
correspondence and solves for the pose in closed form. They share tokenization
and trunk exactly, so a comparison between them isolates the head - the same
methodology that isolated CoLoca-QuA's two deficiencies.

Head A deliberately mean-pools rather than reading out a learned query token:
the reproduction showed a content-free token entering the residual stream makes
the prediction nearly input-independent and collapses it to the conditional mean.

Head B never regresses yaw: the same reproduction showed a regressed yaw never
leaves the conditional mean on this data, so B solves the pose analytically
from a Sinkhorn correspondence instead (see procrustes.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from embedding_aware_belt_fusion.alignformer.boxes import BOX_YAW
from embedding_aware_belt_fusion.alignformer.head_match import (
    DEFAULT_SINKHORN_ITERATIONS,
    log_sinkhorn,
)
from embedding_aware_belt_fusion.alignformer.procrustes import (
    augment_with_heading,
    heading_orientation,
    weighted_se2_kabsch,
)
from embedding_aware_belt_fusion.alignformer.trunk import AlignFormerTrunk, tokenize

# Heading virtual-point offset in metres, about half a vehicle length so that
# headings and centres contribute comparably to the Kabsch fit.
DEFAULT_HEADING_LAMBDA = 2.0
# Score assigned to padded positions so they can never win a match.
_MASKED_SCORE = -1e4
# Below this, a soft-match row's total weight is treated as zero (no matched
# evidence): guards the /safe_mass divisions from a literal 0/0 without
# perturbing any row that has real, non-negligible weight.
_MIN_MASS = 1e-6


@dataclass(frozen=True)
class PoseEstimate:
    """A CAV's estimated SE(2) correction, with the evidence behind it."""

    psi: Tensor
    t: Tensor
    confidence: Tensor
    log_assignment: Optional[Tensor] = None


def _tokens(batch: Mapping[str, Tensor], prefix: str) -> Tensor:
    return tokenize(
        batch[f"{prefix}_boxes"], batch[f"{prefix}_scores"], batch[f"{prefix}_embeddings"]
    )


def _masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    weights = mask.unsqueeze(-1).to(x.dtype)
    return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _is_empty(batch: Mapping[str, Tensor]) -> bool:
    """True when either object set's token dimension is zero.

    ``AlignFormerTrunk`` raises inside ``nn.MultiheadAttention`` when handed a
    ``(B, 0, D)`` token tensor - a real situation (an agent that detected
    nothing), not a synthetic edge case. This MUST be checked before the trunk
    is invoked, not after: there is no exception to catch usefully once
    ``nn.MultiheadAttention`` has already failed its internal reshape.
    """
    return batch["ego_boxes"].shape[1] == 0 or batch["cav_boxes"].shape[1] == 0


def _zero_estimate(batch: Mapping[str, Tensor]) -> PoseEstimate:
    """The identity correction with zero confidence, for an empty object set."""
    reference = batch["ego_boxes"]
    batch_size, device, dtype = reference.shape[0], reference.device, reference.dtype
    psi = torch.zeros(batch_size, device=device, dtype=dtype)
    t = torch.zeros(batch_size, 2, device=device, dtype=dtype)
    confidence = torch.zeros(batch_size, device=device, dtype=dtype)
    return PoseEstimate(psi=psi, t=t, confidence=confidence)


class _Base(nn.Module):
    def __init__(self, embed_dim: int, **trunk_kwargs) -> None:
        super().__init__()
        self.trunk = AlignFormerTrunk(embed_dim=embed_dim, **trunk_kwargs)

    def _encode(self, batch: Mapping[str, Tensor]) -> Tuple[Tensor, Tensor]:
        return self.trunk(
            _tokens(batch, "ego"),
            _tokens(batch, "cav"),
            batch["ego_mask"],
            batch["cav_mask"],
        )


class AlignFormerA(_Base):
    """Direct regression from a mean-pooled descriptor of both sets."""

    def __init__(self, embed_dim: int, **trunk_kwargs) -> None:
        super().__init__(embed_dim, **trunk_kwargs)
        model_dim = self.trunk.model_dim
        self.head = nn.Sequential(
            nn.Linear(2 * model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 3),
        )

    def forward(self, batch: Mapping[str, Tensor]) -> PoseEstimate:
        # Guard BEFORE the trunk call: see _is_empty's docstring.
        if _is_empty(batch):
            return _zero_estimate(batch)

        ego, cav = self._encode(batch)
        pooled = torch.cat(
            [_masked_mean(ego, batch["ego_mask"]), _masked_mean(cav, batch["cav_mask"])],
            dim=-1,
        )
        output = self.head(pooled)

        present = batch["ego_mask"].any(dim=1) & batch["cav_mask"].any(dim=1)
        psi = torch.where(present, output[:, 2], torch.zeros_like(output[:, 2]))
        t = torch.where(
            present.unsqueeze(-1), output[:, :2], torch.zeros_like(output[:, :2])
        )
        return PoseEstimate(psi=psi, t=t, confidence=present.to(psi.dtype))


class AlignFormerB(_Base):
    """Soft correspondence, then a closed-form weighted SE(2) Kabsch solve."""

    def __init__(
        self,
        embed_dim: int,
        heading_lambda: float = DEFAULT_HEADING_LAMBDA,
        sinkhorn_iterations: int = DEFAULT_SINKHORN_ITERATIONS,
        **trunk_kwargs,
    ) -> None:
        super().__init__(embed_dim, **trunk_kwargs)
        model_dim = self.trunk.model_dim
        self.match_projection = nn.Linear(model_dim, model_dim)
        self.dustbin = nn.Parameter(torch.tensor(1.0))
        self.log_temperature = nn.Parameter(torch.tensor(0.1).log())
        self.heading_lambda = heading_lambda
        self.sinkhorn_iterations = sinkhorn_iterations
        # Test hook: score on the raw embeddings, bypassing an untrained trunk.
        self.use_raw_embedding_scores = False

    def _soft_correspondence(
        self,
        batch: Mapping[str, Tensor],
        ego: Tensor,
        cav: Tensor,
        ego_mask: Tensor,
        cav_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Score, Sinkhorn-normalize, and reduce to one virtual CAV point per ego object.

        Returns ``(virtual_centres, virtual_yaws, mass, log_assignment)``:
        ``virtual_centres``/``virtual_yaws`` are the per-ego-object soft-matched
        CAV centre/heading (``(B, M, 2)`` / ``(B, M)``), ``mass`` is each row's
        total soft-match weight (``(B, M)``), and ``log_assignment`` is passed
        through unchanged for :class:`PoseEstimate`.

        LOW-7 (review): extracted out of ``forward`` so the pose-solving code
        below is not buried under this block's own comments.
        """
        if self.use_raw_embedding_scores:
            ego_features = batch["ego_embeddings"]
            cav_features = batch["cav_embeddings"]
        else:
            ego_features = F.normalize(self.match_projection(ego), dim=-1)
            cav_features = F.normalize(self.match_projection(cav), dim=-1)

        scores = ego_features @ cav_features.transpose(1, 2) / self.log_temperature.exp()
        valid = ego_mask.unsqueeze(2) & cav_mask.unsqueeze(1)
        scores = scores.masked_fill(~valid, _MASKED_SCORE)

        log_assignment = log_sinkhorn(scores, self.dustbin, self.sinkhorn_iterations)
        weights = log_assignment[:, :-1, :-1].exp() * valid

        mass = weights.sum(dim=2)
        safe_mass = mass.clamp_min(_MIN_MASS).unsqueeze(-1)
        cav_centres = batch["cav_boxes"][..., :2]
        cav_yaws = batch["cav_boxes"][..., BOX_YAW]

        virtual_centres = weights @ cav_centres / safe_mass
        # Fold each CAV heading onto the half-plane of the ego object it is
        # being matched to BEFORE averaging. The detector reports no direction
        # (procrustes.heading_orientation), so 20.3% of cross-agent detections
        # of the same object point the opposite way along the same axis;
        # averaging those raw would cancel real heading evidence and push the
        # heading virtual point up to 2 * heading_lambda off. The fold has to
        # happen per (ego row, CAV object) rather than once per CAV object,
        # because a single row can soft-match CAV boxes that are flipped
        # differently from one another.
        orientation = heading_orientation(batch["ego_boxes"][..., BOX_YAW], cav_yaws)
        cav_direction = torch.stack([torch.cos(cav_yaws), torch.sin(cav_yaws)], dim=-1)
        virtual_direction = (weights * orientation) @ cav_direction / safe_mass
        # R27 / review MEDIUM-1: atan2's gradient is undefined at exactly
        # (0, 0) -- a mathematical singularity -- and any row with zero
        # soft-match weight (an ordinary padded ego position in a collated
        # batch, or a wholly empty sample) lands exactly there, since
        # `weights` is identically 0 for it. Route those rows through a
        # placeholder direction with a well-defined gradient before atan2,
        # mirroring weighted_se2_kabsch's own atan2 guard below.
        #
        # The placeholder's value is never read, but for two DIFFERENT
        # reasons depending on which case fired: for the dominant case (one
        # padded row inside an otherwise ordinary, non-empty sample) that
        # row's own Kabsch WEIGHT (`augmented_mass` below) is 0, so it drops
        # out of weighted_se2_kabsch's sums regardless of its yaw --
        # MIN_MATCH_MASS never even applies to a single row, it is a
        # per-SAMPLE total-mass gate (procrustes.py). For a wholly empty
        # sample specifically, MIN_MATCH_MASS zeroes the entire correction
        # instead. Threshold at `_MIN_MASS`, not 0, to agree with the
        # `safe_mass` clamp above. The reachability analysis for whether this
        # singularity is achievable end-to-end (it has not been reproduced
        # here) is in docs/superpowers/specs/2026-09-17-alignformer-design.md,
        # section 3.4.
        has_weight = mass > _MIN_MASS
        placeholder_direction = virtual_direction.new_tensor([1.0, 0.0]).expand_as(
            virtual_direction
        )
        safe_direction = torch.where(
            has_weight.unsqueeze(-1), virtual_direction, placeholder_direction
        )
        virtual_yaws = torch.atan2(safe_direction[..., 1], safe_direction[..., 0])

        return virtual_centres, virtual_yaws, mass, log_assignment

    def forward(self, batch: Mapping[str, Tensor]) -> PoseEstimate:
        # Guard BEFORE the trunk call: see _is_empty's docstring. With an empty
        # object set there is no correspondence to build anyway, so the zero
        # correction below is not just a crash-avoidance shortcut - it is the
        # correct answer.
        if _is_empty(batch):
            return _zero_estimate(batch)

        ego_mask, cav_mask = batch["ego_mask"], batch["cav_mask"]
        ego, cav = self._encode(batch)

        virtual_centres, virtual_yaws, mass, log_assignment = self._soft_correspondence(
            batch, ego, cav, ego_mask, cav_mask
        )

        source = augment_with_heading(virtual_centres, virtual_yaws, self.heading_lambda)
        target = augment_with_heading(
            batch["ego_boxes"][..., :2],
            batch["ego_boxes"][..., BOX_YAW],
            self.heading_lambda,
        )
        augmented_mass = torch.cat([mass, mass], dim=1)

        psi, t = weighted_se2_kabsch(target, source, augmented_mass)
        return PoseEstimate(
            psi=psi, t=t, confidence=mass.sum(dim=1), log_assignment=log_assignment
        )
