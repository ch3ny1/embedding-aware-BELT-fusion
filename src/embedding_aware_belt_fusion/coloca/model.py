"""The CoLoca-QuA localization module (paper Section III-B, Fig. 4).

A ViT-style encoder over BEV patches with one extra learnable "Localization
Token".  That token queries every patch through self-attention and its final
state is mapped to the SE(2) pose error ``[dx, dy, dpsi]`` (Eq. 10-13).

Default hyperparameters are Table I of the paper:

    patch size P       16
    embedding dim D    256
    heads H            4
    MLP ratio r        4
    encoder layers L   6
    dropout            0.3
    positional enc.    learnable
    loc token          learnable, dim D
    patch embedding    Conv-PxP, stride P
    normalization      Pre-LN (LayerNorm)
"""

from __future__ import annotations

import torch
from torch import nn

# Paper Table I / Section IV-E.
DEFAULT_PATCH_SIZE = 16
DEFAULT_EMBED_DIM = 256
DEFAULT_NUM_HEADS = 4
DEFAULT_MLP_RATIO = 4
DEFAULT_DEPTH = 6
DEFAULT_DROPOUT = 0.3
POSE_ERROR_DIM = 3  # dx, dy, dpsi
# Roughly the per-component standard deviation of the SE(2) pose-error labels
# under the paper's noise model, measured on OPV2V: 1.57 m / 1.65 m / 1.00 deg.
# Used only to condition the regression head, not to change the loss.
DEFAULT_OUTPUT_SCALE = (2.0, 2.0, 1.0)
# Width of the optional convolutional stem. Its first layer is strided, so the
# stem halves the spatial resolution exactly once regardless of its depth.
DEFAULT_STEM_CHANNELS = 128
CONV_STEM_STRIDE = 2

# How the pose error is read out of the encoded sequence.
#   "loc_token" - the paper's Eq. (13): the localization token's final state.
#   "mean"      - average of the final patch-token states.
#   "both"      - their sum, keeping the localization token in the loop.
# Measured on OPV2V, "loc_token" never leaves the predict-zero solution while
# "mean" trains readily on identical features; see docs/coloca_qua_baseline.md.
READOUTS = ("loc_token", "mean", "both")
DEFAULT_READOUT = "loc_token"


def _build_conv_stem(
    in_channels: int, depth: int, channels: int
) -> tuple[nn.Module, int, int]:
    """Return ``(stem, total_stride, output_channels)``.

    ``depth = 0`` reproduces the paper exactly: an identity stem, so the patch
    embedding sees the raw concatenated feature map.
    """
    if depth < 0:
        raise ValueError(f"conv_stem_depth must be non-negative, got {depth}")
    if depth == 0:
        return nn.Identity(), 1, in_channels

    layers: list[nn.Module] = []
    previous = in_channels
    for index in range(depth):
        layers += [
            nn.Conv2d(
                previous,
                channels,
                kernel_size=3,
                stride=CONV_STEM_STRIDE if index == 0 else 1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        ]
        previous = channels
    return nn.Sequential(*layers), CONV_STEM_STRIDE, channels


class ColocaQuA(nn.Module):
    """Query-based localization head over a concatenated pair of BEV maps.

    Parameters
    ----------
    in_channels:
        Channels of the concatenated ego+CAV feature map (``2C`` in Eq. 9).
    feature_size:
        ``(H, W)`` of the incoming feature map, used to size the learnable
        positional embedding.  The paper's setting is ``(96, 256)``.
    """

    def __init__(
        self,
        in_channels: int,
        feature_size: tuple[int, int],
        patch_size: int = DEFAULT_PATCH_SIZE,
        embed_dim: int = DEFAULT_EMBED_DIM,
        depth: int = DEFAULT_DEPTH,
        num_heads: int = DEFAULT_NUM_HEADS,
        mlp_ratio: int = DEFAULT_MLP_RATIO,
        dropout: float = DEFAULT_DROPOUT,
        output_scale: tuple[float, float, float] = DEFAULT_OUTPUT_SCALE,
        conv_stem_depth: int = 0,
        conv_stem_channels: int = DEFAULT_STEM_CHANNELS,
        readout: str = DEFAULT_READOUT,
    ) -> None:
        super().__init__()
        height, width = feature_size
        if height < patch_size or width < patch_size:
            raise ValueError(
                f"feature_size {feature_size} is smaller than patch_size {patch_size}"
            )

        if readout not in READOUTS:
            raise ValueError(f"readout must be one of {READOUTS}, got {readout!r}")
        self.readout = readout
        self.feature_size = (height, width)
        self.patch_size = patch_size
        self.num_patches = (height // patch_size) * (width // patch_size)

        # Optional convolutional stem (see `conv_stem` in the config and
        # docs/coloca_qua_baseline.md).  The paper tokenizes the concatenated
        # map with a single *linear* convolution, but a linear map cannot form
        # the ego x CAV product that registration needs -- measured on OPV2V,
        # the paper-faithful head never leaves the predict-zero solution.  A
        # shallow conv stack supplies that local nonlinear interaction before
        # the 512x patch compression, and keeps N, D, L and the loc token
        # exactly as specified because its stride is folded into the patch size.
        self.conv_stem, stem_stride, stem_channels = _build_conv_stem(
            in_channels, conv_stem_depth, conv_stem_channels
        )
        if patch_size % stem_stride:
            raise ValueError(
                f"patch_size {patch_size} must be divisible by the conv-stem stride {stem_stride}"
            )

        # Eq. (10): a strided convolution tokenizes the (optionally stemmed) map.
        self.patch_embed = nn.Conv2d(
            stem_channels,
            embed_dim,
            kernel_size=patch_size // stem_stride,
            stride=patch_size // stem_stride,
        )
        # Eq. (12): the learnable localization token prepended to the sequence.
        self.loc_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Eq. (12): learnable positional embeddings over token + patches.
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(0.0)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN, per Table I
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        # Table I specifies "Dropout (attn) 0.3", i.e. dropout on the attention
        # only.  PyTorch's `dropout=` argument also wires it into the FFN and
        # both residual branches, which is far heavier regularization than the
        # paper asks for, so the non-attention dropouts are switched off.
        for layer in self.encoder.layers:
            layer.self_attn.dropout = dropout
            layer.dropout.p = 0.0
            layer.dropout1.p = 0.0
            layer.dropout2.p = 0.0

        self.norm = nn.LayerNorm(embed_dim)

        # Eq. (13): the localization token's final state -> [dx, dy, dpsi].
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, POSE_ERROR_DIM),
        )
        # The head reads a LayerNorm'd (unit-scale) vector but must emit metres
        # and degrees, whose targets have std ~1.6 m / ~1.0 deg.  Without this
        # the final layer has to grow its weights by an order of magnitude
        # before predictions even reach the right scale, which at lr 1e-4 wastes
        # thousands of steps.  Regressing a normalized value and rescaling here
        # leaves the loss of Eq. (14) untouched but conditions it far better.
        self.register_buffer(
            "output_scale", torch.tensor(output_scale, dtype=torch.float32), persistent=False
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.loc_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """Regress the SE(2) pose error from a concatenated BEV feature map.

        Parameters
        ----------
        fused_features:
            ``(B, 2C, H, W)`` tensor, the channel-wise concatenation of the ego
            and CAV BEV maps (Eq. 9).

        Returns
        -------
        torch.Tensor
            ``(B, 3)`` tensor of ``[dx, dy, dpsi]``; translation in metres and
            yaw in degrees.
        """
        if fused_features.dim() != 4:
            raise ValueError(
                f"expected a 4D (B, C, H, W) tensor, got shape {tuple(fused_features.shape)}"
            )

        patches = self.patch_embed(self.conv_stem(fused_features))  # (B, D, H/P, W/P)
        patches = patches.flatten(2).transpose(1, 2)  # Eq. (11): (B, N, D)
        if patches.shape[1] != self.num_patches:
            raise ValueError(
                f"got {patches.shape[1]} patches but the positional embedding was built "
                f"for {self.num_patches}; feature map size changed from {self.feature_size}"
            )

        loc_token = self.loc_token.expand(patches.shape[0], -1, -1)
        sequence = torch.cat([loc_token, patches], dim=1)  # Eq. (12)
        sequence = self.pos_drop(sequence + self.pos_embed)

        encoded = self.norm(self.encoder(sequence))
        if self.readout == "loc_token":
            pooled = encoded[:, 0]  # Eq. (13), localization token only
        elif self.readout == "mean":
            pooled = encoded[:, 1:].mean(dim=1)
        else:
            pooled = encoded[:, 0] + encoded[:, 1:].mean(dim=1)
        return self.head(pooled) * self.output_scale


class ColocaQuANet(nn.Module):
    """End-to-end model: shared PointPillars encoder + CoLoca-QuA head.

    Both agents are encoded by the *same* backbone weights, matching the paper's
    Fig. 3 where "Ego Backbone" and "CAV Backbone" are the shared cooperative
    perception backbone run on each agent's own point cloud.
    """

    def __init__(self, encoder: nn.Module, feature_size: tuple[int, int], **head_kwargs) -> None:
        super().__init__()
        self.encoder = encoder
        self.localizer = ColocaQuA(
            in_channels=2 * encoder.out_channels,
            feature_size=feature_size,
            **head_kwargs,
        )

    def forward(self, batch: dict) -> torch.Tensor:
        ego_features = self.encoder(batch["ego_lidar"])
        cav_features = self.encoder(batch["cav_lidar"])
        fused = torch.cat([ego_features, cav_features], dim=1)  # Eq. (9)
        return self.localizer(fused)
