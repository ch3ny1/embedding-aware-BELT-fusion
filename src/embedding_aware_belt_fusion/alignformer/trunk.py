"""Shared transformer over the ego and CAV object sets.

Both heads consume this trunk unchanged, so an A-vs-B comparison isolates the
head. Cross-attention between the two sets supplies the multiplicative ego x CAV
interaction whose absence made CoLoca-QuA's linear tokenization unable to
register; see docs/coloca_qua_baseline.md.

Object sets are unordered, so the trunk carries no positional encoding - spatial
information enters only through each token's own box geometry.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

DEFAULT_MODEL_DIM = 256
DEFAULT_LAYERS = 4
DEFAULT_HEADS = 4
# Token budget per agent; objects beyond this are dropped by descending score.
MAX_OBJECTS = 64

# 8 geometry features plus the detector score.
GEOMETRY_FEATURES = 9
_BOX_YAW = 6


def tokenize(boxes: Tensor, scores: Tensor, embeddings: Tensor) -> Tensor:
    """Build raw token features from boxes, scores and embeddings.

    Yaw is encoded as ``(cos, sin)`` so the representation is continuous across
    the +/-pi wrap.

    Parameters
    ----------
    boxes: ``(B, N, 7)`` in ``hwl`` order, already in the ego frame.
    scores: ``(B, N)`` detector confidences.
    embeddings: ``(B, N, d)`` unit-norm per-object descriptors.

    Returns
    -------
    Tensor
        ``(B, N, GEOMETRY_FEATURES + d)``.
    """
    yaw = boxes[..., _BOX_YAW]
    geometry = torch.cat(
        [
            boxes[..., :6],
            torch.cos(yaw).unsqueeze(-1),
            torch.sin(yaw).unsqueeze(-1),
            scores.unsqueeze(-1),
        ],
        dim=-1,
    )
    return torch.cat([geometry, embeddings], dim=-1)


class _Layer(nn.Module):
    """One self-attention pass within each set, then cross-attention across."""

    def __init__(self, model_dim: int, heads: int) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(model_dim, heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(model_dim, heads, batch_first=True)
        self.norm_self = nn.LayerNorm(model_dim)
        self.norm_cross = nn.LayerNorm(model_dim)
        self.norm_ff = nn.LayerNorm(model_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim),
            nn.GELU(),
            nn.Linear(4 * model_dim, model_dim),
        )

    def forward(self, x: Tensor, y: Tensor, x_mask: Tensor, y_mask: Tensor) -> Tensor:
        # key_padding_mask marks positions to IGNORE, so invert the validity mask.
        normed = self.norm_self(x)
        attended, _ = self.self_attention(
            normed, normed, normed, key_padding_mask=~x_mask, need_weights=False
        )
        x = x + attended

        normed_x, normed_y = self.norm_cross(x), self.norm_cross(y)
        attended, _ = self.cross_attention(
            normed_x, normed_y, normed_y, key_padding_mask=~y_mask, need_weights=False
        )
        x = x + attended

        return x + self.feed_forward(self.norm_ff(x))


class AlignFormerTrunk(nn.Module):
    """Interleaved self- and cross-attention over the two object sets."""

    def __init__(
        self,
        embed_dim: int,
        model_dim: int = DEFAULT_MODEL_DIM,
        layers: int = DEFAULT_LAYERS,
        heads: int = DEFAULT_HEADS,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(GEOMETRY_FEATURES + embed_dim, model_dim)
        self.set_embedding = nn.Parameter(torch.zeros(2, model_dim))
        self.ego_layers = nn.ModuleList(_Layer(model_dim, heads) for _ in range(layers))
        self.cav_layers = nn.ModuleList(_Layer(model_dim, heads) for _ in range(layers))
        self.output_norm = nn.LayerNorm(model_dim)
        self.model_dim = model_dim

    def forward(
        self, ego_tokens: Tensor, cav_tokens: Tensor, ego_mask: Tensor, cav_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Encode both sets, each attending to itself and to the other."""
        ego = self.input_projection(ego_tokens) + self.set_embedding[0]
        cav = self.input_projection(cav_tokens) + self.set_embedding[1]

        for ego_layer, cav_layer in zip(self.ego_layers, self.cav_layers):
            next_ego = ego_layer(ego, cav, ego_mask, cav_mask)
            next_cav = cav_layer(cav, ego, cav_mask, ego_mask)
            ego, cav = next_ego, next_cav

        ego = self.output_norm(ego) * ego_mask.unsqueeze(-1)
        cav = self.output_norm(cav) * cav_mask.unsqueeze(-1)
        return ego, cav
