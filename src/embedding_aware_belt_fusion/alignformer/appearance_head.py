"""A projection head trained for cross-agent viewpoint invariance over frozen
DINOv2 crop features.

Why train at all
----------------
Zero-shot DINOv2 failed the appearance signal bar on V2X-Real val (AUC 0.68
on the ambiguous subset against the pre-registered 0.80), and its
same-agent bound was 0.73: two cameras of ONE vehicle at ONE instant barely
separate it from its neighbour. Viewpoint dominates the raw feature, which
is the hypothesis a frozen backbone cannot answer. The dataset itself
supplies the supervision for the fix: every shared vehicle seen by two
agents at one timestamp is a cross-view positive pair with known identity,
and the other vehicles in the partner's frame are the exact distractors the
matcher will face.

What this module holds (pure, tested): the cached per-agent-frame record
written by ``scripts/cache_v2xreal_appearance_features.py``; the pair
construction (anchor, cross-agent positive, nearest in-frame hard negatives);
the head; the InfoNCE loss with in-batch and hard negatives. Training and
the validation verdict live in ``scripts/train_v2xreal_appearance_head.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

HIDDEN_DIM = 512
OUT_DIM = 128
TEMPERATURE = 0.07
MAX_HARD_NEGATIVES = 8


class AgentFrame(NamedTuple):
    """Cached DINOv2 descriptors of every usable vehicle view in one agent-frame."""

    scenario: str
    agent: str
    timestamp: str
    vids: Tuple[str, ...]
    features: np.ndarray  # (N, D) float32, concatenated descriptors
    centre_xy: np.ndarray  # (N, 2) world frame, for the distractor geometry
    range_m: np.ndarray  # (N,) from this agent's LiDAR
    gt_vids: Tuple[str, ...] = ()  # every annotated vehicle, seen or not: the shared-object bucket


def frame_key(scenario: str, agent: str, timestamp: str) -> str:
    return f"{scenario}__{agent}__{timestamp}"


def save_frame(directory: Path, frame: AgentFrame) -> Path:
    path = directory / f"{frame_key(frame.scenario, frame.agent, frame.timestamp)}.npz"
    partial = path.with_suffix(".npz.tmp")
    with open(partial, "wb") as handle:  # write whole, then rename: a crash leaves no half file to resume past
        np.savez(handle, vids=np.asarray(frame.vids, dtype=str), features=frame.features.astype(np.float32),
                 centre_xy=frame.centre_xy, range_m=frame.range_m, gt_vids=np.asarray(frame.gt_vids, dtype=str))
    partial.replace(path)
    return path


def feature_dim(frames: Sequence[AgentFrame]) -> int:
    """Descriptor width from the first frame that saw a vehicle (empty frames carry none)."""
    return next((int(f.features.shape[1]) for f in frames if len(f.vids) > 0), 0)


def load_frame(path: Path) -> AgentFrame:
    scenario, agent, timestamp = path.stem.rsplit("__", 2)
    with np.load(path) as data:
        return AgentFrame(scenario, agent, timestamp, tuple(str(v) for v in data["vids"]),
                          data["features"], data["centre_xy"], data["range_m"],
                          tuple(str(v) for v in data["gt_vids"]) if "gt_vids" in data else ())


class PairSet(NamedTuple):
    """Training pairs: ``rows[i] = (scenario, timestamp, anchor agent, partner agent, vid)``;
    ``identities[i] = (scenario, vid)`` names the object, which persists through a scenario."""

    rows: List[Tuple[str, str, str, str, str]]
    identities: List[Tuple[str, str]]
    anchors: np.ndarray  # (P, D)
    positives: np.ndarray  # (P, D)
    hard_negatives: np.ndarray  # (P, K, D), zero where padded
    hard_mask: np.ndarray  # (P, K) bool
    shared_counts: np.ndarray  # (P,) annotated vehicles in common between the two frames
    anchor_ranges: np.ndarray  # (P,) the anchor's LiDAR range, metres


def _group_by_timestamp(frames: Sequence[AgentFrame]) -> Dict[Tuple[str, str], List[AgentFrame]]:
    groups: Dict[Tuple[str, str], List[AgentFrame]] = {}
    for frame in frames:
        groups.setdefault((frame.scenario, frame.timestamp), []).append(frame)
    return groups


def _partners(frame: AgentFrame, group_of: Dict[Tuple[str, str], List[AgentFrame]], stamps: Dict[str, List[str]],
              offsets: Sequence[int]) -> List[AgentFrame]:
    """The other agents' frames at the anchor's timestamp and at ``offsets`` cached steps around it."""
    ordered = stamps[frame.scenario]
    here = ordered.index(frame.timestamp)
    seen, partners = set(), []
    for offset in offsets:
        if not 0 <= here + offset < len(ordered):
            continue
        for partner in group_of.get((frame.scenario, ordered[here + offset]), []):
            key = (partner.agent, partner.timestamp)
            if partner.agent != frame.agent and key not in seen:
                seen.add(key)
                partners.append(partner)
    return partners


def _hard_negatives(partner: AgentFrame, vid: str, exclude: Sequence[str], k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Up to ``k`` partner-frame features nearest to ``vid``'s centre, ``exclude`` left out.

    The true match is always excluded, in the control too: a control that
    pushed the anchor away from its own counterpart would be a negative
    signal, not a null."""
    index = {v: i for i, v in enumerate(partner.vids)}
    centre = partner.centre_xy[index[vid]]
    others = sorted((v for v in partner.vids if v not in exclude),
                    key=lambda v: float(np.linalg.norm(partner.centre_xy[index[v]] - centre)))[:k]
    dim = partner.features.shape[1]
    block, mask = np.zeros((k, dim), dtype=np.float32), np.zeros(k, dtype=bool)
    for slot, v in enumerate(others):
        block[slot], mask[slot] = partner.features[index[v]], True
    return block, mask


def _positive_vid(common: Sequence[str], vid: str, shuffle: Optional[np.random.Generator]) -> Optional[str]:
    if shuffle is None:
        return vid
    others = [v for v in common if v != vid]
    return str(shuffle.choice(others)) if others else None


def build_pairs(frames: Sequence[AgentFrame], max_hard_negatives: int = MAX_HARD_NEGATIVES,
                shuffle_identities: Optional[np.random.Generator] = None, offsets: Sequence[int] = (0,)) -> PairSet:
    """Every (anchor agent, partner agent, shared vehicle) at every timestamp, both directions.

    ``offsets`` adds the partner agent's frames at neighbouring cached steps
    as further cross-view positives (the view change is the same, the data
    multiplies). ``shuffle_identities`` builds the control: the positive is
    another shared vehicle of the partner frame, so a head trained on it
    learns nothing about identity.
    """
    group_of = _group_by_timestamp(frames)
    stamps = {s: sorted(t for (sc, t) in group_of if sc == s) for s in {sc for (sc, _) in group_of}}
    acc = {k: [] for k in ("rows", "identities", "anchors", "positives", "negatives", "masks", "shared", "ranges")}
    for key in sorted(group_of):
        for anchor_frame in group_of[key]:
            for partner in _partners(anchor_frame, group_of, stamps, offsets):
                _append_rows(acc, anchor_frame, partner, max_hard_negatives, shuffle_identities)
    dim, k = feature_dim(frames), max_hard_negatives
    return PairSet(acc["rows"], acc["identities"], _stack(acc["anchors"], (0, dim)), _stack(acc["positives"], (0, dim)),
                   _stack(acc["negatives"], (0, k, dim)), _stack(acc["masks"], (0, k), bool),
                   np.asarray(acc["shared"], dtype=np.int64), np.asarray(acc["ranges"], dtype=np.float64))


def _append_rows(acc: Dict[str, list], anchor_frame: AgentFrame, partner: AgentFrame, k: int,
                 shuffle: Optional[np.random.Generator]) -> None:
    common = sorted(set(anchor_frame.vids) & set(partner.vids))
    shared = len(set(anchor_frame.gt_vids) & set(partner.gt_vids))
    for vid in common:
        positive = _positive_vid(common, vid, shuffle)
        if positive is None:
            continue
        block, mask = _hard_negatives(partner, vid, exclude=(vid, positive), k=k)
        row = anchor_frame.vids.index(vid)
        acc["rows"].append((anchor_frame.scenario, anchor_frame.timestamp, anchor_frame.agent, partner.agent, vid))
        acc["identities"].append((anchor_frame.scenario, vid))
        acc["anchors"].append(anchor_frame.features[row])
        acc["positives"].append(partner.features[partner.vids.index(positive)])
        acc["negatives"].append(block)
        acc["masks"].append(mask)
        acc["shared"].append(shared)
        acc["ranges"].append(float(anchor_frame.range_m[row]))


def row_weights(shared_counts: np.ndarray, anchor_ranges: np.ndarray, far_weight: float, sparse_weight: float,
                far_from_m: float) -> np.ndarray:
    """``1 + far_weight`` for anchors beyond ``far_from_m``, ``+ sparse_weight`` for pairs sharing <= 2 annotations."""
    far = (np.asarray(anchor_ranges) > far_from_m).astype(np.float64)
    sparse = (np.asarray(shared_counts) <= 2).astype(np.float64)
    return 1.0 + far_weight * far + sparse_weight * sparse


def epoch_order(weights: Optional[np.ndarray], count: int, draws: int, rng: np.random.Generator) -> np.ndarray:
    """A permutation when unweighted; otherwise ``draws`` rows with replacement, probability ∝ weight."""
    if weights is None:
        return rng.permutation(count)
    probability = np.asarray(weights, dtype=np.float64)
    return rng.choice(count, size=draws, replace=True, p=probability / probability.sum())


def _stack(items, empty_shape, dtype=np.float32) -> np.ndarray:
    return np.stack(items).astype(dtype) if items else np.zeros(empty_shape, dtype=dtype)


class AppearanceHead(nn.Module):
    """Two-layer MLP onto the unit sphere."""

    def __init__(self, in_dim: int, hidden_dim: int = HIDDEN_DIM, out_dim: int = OUT_DIM, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, out_dim))

    def forward(self, features: Tensor) -> Tensor:
        return F.normalize(self.net(features), dim=-1)


def same_identity_mask(identities: Sequence[Tuple[str, str]]) -> Tensor:
    """``(B, B)`` True off the diagonal where two rows are the same object.

    One vehicle appears in many rows (both directions, every agent pair,
    every timestamp of its scenario), so a random batch holds about one such
    collision per row; counted as negatives they would push the head apart
    from its own positives. Masked out of the in-batch softmax instead.
    """
    codes = {key: i for i, key in enumerate(dict.fromkeys(identities))}
    ids = torch.tensor([codes[key] for key in identities])
    same = ids[:, None] == ids[None, :]
    return same & ~torch.eye(len(identities), dtype=torch.bool)


def info_nce(anchors: Tensor, positives: Tensor, hard_negatives: Tensor, hard_mask: Tensor,
             temperature: float = TEMPERATURE, identity_mask: Optional[Tensor] = None) -> Tensor:
    """Symmetric InfoNCE: in-batch negatives both ways, in-frame hard negatives on the anchor side.

    ``hard_negatives`` is ``(B, K, D)`` already on the sphere, ``hard_mask``
    ``(B, K)``; padded slots are excluded from the softmax, so a frame with
    no other vehicle contributes only its in-batch term. ``identity_mask``
    (``same_identity_mask``) removes in-batch false negatives.
    """
    batch = anchors.shape[0]
    logits_ap = anchors @ positives.T / temperature  # (B, B)
    if identity_mask is not None:
        logits_ap = logits_ap.masked_fill(identity_mask.to(logits_ap.device), float("-inf"))
    targets = torch.arange(batch, device=anchors.device)
    if hard_negatives.shape[1] > 0:
        hard = torch.einsum("bd,bkd->bk", anchors, hard_negatives) / temperature
        hard = hard.masked_fill(~hard_mask, float("-inf"))
        logits_anchor = torch.cat([logits_ap, hard], dim=1)
    else:
        logits_anchor = logits_ap
    return 0.5 * (F.cross_entropy(logits_anchor, targets) + F.cross_entropy(logits_ap.T, targets))
