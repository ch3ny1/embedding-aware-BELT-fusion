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
    np.savez(path, vids=np.asarray(frame.vids, dtype=str), features=frame.features.astype(np.float32),
             centre_xy=frame.centre_xy, range_m=frame.range_m, gt_vids=np.asarray(frame.gt_vids, dtype=str))
    return path


def load_frame(path: Path) -> AgentFrame:
    scenario, agent, timestamp = path.stem.split("__")
    with np.load(path) as data:
        return AgentFrame(scenario, agent, timestamp, tuple(str(v) for v in data["vids"]),
                          data["features"], data["centre_xy"], data["range_m"],
                          tuple(str(v) for v in data["gt_vids"]) if "gt_vids" in data else ())


class PairSet(NamedTuple):
    """Training pairs: ``rows[i] = (scenario, timestamp, anchor agent, partner agent, vid)``."""

    rows: List[Tuple[str, str, str, str, str]]
    anchors: np.ndarray  # (P, D)
    positives: np.ndarray  # (P, D)
    hard_negatives: np.ndarray  # (P, K, D), zero where padded
    hard_mask: np.ndarray  # (P, K) bool


def _group_by_timestamp(frames: Sequence[AgentFrame]) -> Dict[Tuple[str, str], List[AgentFrame]]:
    groups: Dict[Tuple[str, str], List[AgentFrame]] = {}
    for frame in frames:
        groups.setdefault((frame.scenario, frame.timestamp), []).append(frame)
    return groups


def _hard_negatives(partner: AgentFrame, vid: str, exclude: str, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Up to ``k`` partner-frame features nearest to ``vid``'s centre, ``exclude`` left out."""
    index = {v: i for i, v in enumerate(partner.vids)}
    centre = partner.centre_xy[index[vid]]
    others = sorted((v for v in partner.vids if v != exclude),
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
                shuffle_identities: Optional[np.random.Generator] = None) -> PairSet:
    """Every (anchor agent, partner agent, shared vehicle) at every timestamp, both directions.

    ``shuffle_identities`` builds the control: the positive is another shared
    vehicle of the partner frame, so a head trained on it learns nothing
    about identity and its validation AUC must stay near one half.
    """
    rows, anchors, positives, negatives, masks = [], [], [], [], []
    for (scenario, stamp), group in sorted(_group_by_timestamp(frames).items()):
        for anchor_frame in group:
            for partner in group:
                if partner.agent == anchor_frame.agent:
                    continue
                common = sorted(set(anchor_frame.vids) & set(partner.vids))
                for vid in common:
                    positive = _positive_vid(common, vid, shuffle_identities)
                    if positive is None:
                        continue
                    block, mask = _hard_negatives(partner, vid, exclude=positive, k=max_hard_negatives)
                    rows.append((scenario, stamp, anchor_frame.agent, partner.agent, vid))
                    anchors.append(anchor_frame.features[anchor_frame.vids.index(vid)])
                    positives.append(partner.features[partner.vids.index(positive)])
                    negatives.append(block)
                    masks.append(mask)
    dim = frames[0].features.shape[1] if frames else 0
    return PairSet(rows, _stack(anchors, (0, dim)), _stack(positives, (0, dim)),
                   _stack(negatives, (0, max_hard_negatives, dim)), _stack(masks, (0, max_hard_negatives), bool))


def _stack(items, empty_shape, dtype=np.float32) -> np.ndarray:
    return np.stack(items).astype(dtype) if items else np.zeros(empty_shape, dtype=dtype)


class AppearanceHead(nn.Module):
    """Two-layer MLP onto the unit sphere."""

    def __init__(self, in_dim: int, hidden_dim: int = HIDDEN_DIM, out_dim: int = OUT_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim))

    def forward(self, features: Tensor) -> Tensor:
        return F.normalize(self.net(features), dim=-1)


def info_nce(anchors: Tensor, positives: Tensor, hard_negatives: Tensor, hard_mask: Tensor,
             temperature: float = TEMPERATURE) -> Tensor:
    """Symmetric InfoNCE: in-batch negatives both ways, in-frame hard negatives on the anchor side.

    ``hard_negatives`` is ``(B, K, D)`` already on the sphere, ``hard_mask``
    ``(B, K)``; padded slots are excluded from the softmax, so a frame with
    no other vehicle contributes only its in-batch term.
    """
    batch = anchors.shape[0]
    logits_ap = anchors @ positives.T / temperature  # (B, B)
    targets = torch.arange(batch, device=anchors.device)
    if hard_negatives.shape[1] > 0:
        hard = torch.einsum("bd,bkd->bk", anchors, hard_negatives) / temperature
        hard = hard.masked_fill(~hard_mask, float("-inf"))
        logits_anchor = torch.cat([logits_ap, hard], dim=1)
    else:
        logits_anchor = logits_ap
    return 0.5 * (F.cross_entropy(logits_anchor, targets) + F.cross_entropy(logits_ap.T, targets))
