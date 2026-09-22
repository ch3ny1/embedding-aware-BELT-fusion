import torch

from embedding_aware_belt_fusion.alignformer.trunk import (
    DEFAULT_HEADS,
    DEFAULT_LAYERS,
    DEFAULT_MODEL_DIM,
    AlignFormerTrunk,
    tokenize,
)


def _trunk(embed_dim=16):
    return AlignFormerTrunk(
        embed_dim=embed_dim,
        model_dim=DEFAULT_MODEL_DIM,
        layers=DEFAULT_LAYERS,
        heads=DEFAULT_HEADS,
    ).eval()


def test_tokenize_concatenates_geometry_score_and_embedding():
    boxes = torch.randn(2, 5, 7)
    scores = torch.rand(2, 5)
    embeddings = torch.randn(2, 5, 16)

    tokens = tokenize(boxes, scores, embeddings)

    # 8 geometry features (x, y, z, h, w, l, cos yaw, sin yaw) + score + embedding
    assert tokens.shape == (2, 5, 9 + 16)


def test_forward_preserves_per_set_shapes():
    trunk = _trunk()
    ego = torch.randn(2, 5, 25)
    cav = torch.randn(2, 7, 25)
    ego_mask = torch.ones(2, 5, dtype=torch.bool)
    cav_mask = torch.ones(2, 7, dtype=torch.bool)

    ego_out, cav_out = trunk(ego, cav, ego_mask, cav_mask)

    assert ego_out.shape == (2, 5, DEFAULT_MODEL_DIM)
    assert cav_out.shape == (2, 7, DEFAULT_MODEL_DIM)


def test_output_is_permutation_equivariant():
    # Object sets have no intrinsic order, so permuting the input must permute
    # the output identically. A positional leak would break matching.
    torch.manual_seed(0)
    trunk = _trunk()
    ego = torch.randn(1, 4, 25)
    cav = torch.randn(1, 3, 25)
    ego_mask = torch.ones(1, 4, dtype=torch.bool)
    cav_mask = torch.ones(1, 3, dtype=torch.bool)

    with torch.no_grad():
        base, _ = trunk(ego, cav, ego_mask, cav_mask)
        order = torch.tensor([2, 0, 3, 1])
        permuted, _ = trunk(ego[:, order], cav, ego_mask[:, order], cav_mask)

    assert torch.allclose(base[:, order], permuted, atol=1e-5)


def test_output_is_permutation_equivariant_on_the_cav_side():
    # R26: the mirror of test_output_is_permutation_equivariant above, which
    # only ever permutes the EGO side, leaving the CAV-side equivariance
    # unverified even though tokenize/AlignFormerTrunk carry no positional
    # encoding and must be equivariant to permuting either set.
    torch.manual_seed(0)
    trunk = _trunk()
    ego = torch.randn(1, 4, 25)
    cav = torch.randn(1, 3, 25)
    ego_mask = torch.ones(1, 4, dtype=torch.bool)
    cav_mask = torch.ones(1, 3, dtype=torch.bool)

    with torch.no_grad():
        _, base_cav = trunk(ego, cav, ego_mask, cav_mask)
        order = torch.tensor([2, 0, 1])
        _, permuted_cav = trunk(ego, cav[:, order], ego_mask, cav_mask[:, order])

    assert torch.allclose(base_cav[:, order], permuted_cav, atol=1e-5)


def test_padded_objects_do_not_affect_real_ones():
    torch.manual_seed(0)
    trunk = _trunk()
    ego = torch.randn(1, 3, 25)
    cav = torch.randn(1, 2, 25)
    ego_mask = torch.ones(1, 3, dtype=torch.bool)
    cav_mask = torch.ones(1, 2, dtype=torch.bool)

    padded_ego = torch.cat([ego, torch.randn(1, 4, 25)], dim=1)
    padded_ego_mask = torch.cat([ego_mask, torch.zeros(1, 4, dtype=torch.bool)], dim=1)

    with torch.no_grad():
        base, _ = trunk(ego, cav, ego_mask, cav_mask)
        padded, _ = trunk(padded_ego, cav, padded_ego_mask, cav_mask)

    assert torch.allclose(base, padded[:, :3], atol=1e-5)
