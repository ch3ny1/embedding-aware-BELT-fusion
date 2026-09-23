import math

import pytest
import torch

from embedding_aware_belt_fusion.alignformer.model import AlignFormerA, AlignFormerB


def _batch(ego_boxes, cav_boxes, embed_dim=8, ego_embeddings=None, cav_embeddings=None):
    batch_size = ego_boxes.shape[0]
    ego_count, cav_count = ego_boxes.shape[1], cav_boxes.shape[1]
    return {
        "ego_boxes": ego_boxes,
        "ego_scores": torch.ones(batch_size, ego_count),
        "ego_embeddings": (
            ego_embeddings if ego_embeddings is not None
            else torch.zeros(batch_size, ego_count, embed_dim)
        ),
        "ego_mask": torch.ones(batch_size, ego_count, dtype=torch.bool),
        "cav_boxes": cav_boxes,
        "cav_scores": torch.ones(batch_size, cav_count),
        "cav_embeddings": (
            cav_embeddings if cav_embeddings is not None
            else torch.zeros(batch_size, cav_count, embed_dim)
        ),
        "cav_mask": torch.ones(batch_size, cav_count, dtype=torch.bool),
    }


def _boxes_from_centres(centres, yaws):
    """Build (B, N, 7) boxes with fixed height/width/length/z from centres+yaws."""
    batch_size, count = centres.shape[:2]
    return torch.cat(
        [
            centres,
            torch.zeros(batch_size, count, 1),
            torch.full((batch_size, count, 1), 1.5),
            torch.full((batch_size, count, 1), 2.0),
            torch.full((batch_size, count, 1), 4.0),
            yaws.unsqueeze(-1),
        ],
        dim=-1,
    )


def test_head_b_returns_a_pose_estimate_with_an_assignment():
    model = AlignFormerB(embed_dim=8).eval()
    batch = _batch(torch.randn(1, 4, 7), torch.randn(1, 5, 7))

    estimate = model(batch)

    assert estimate.psi.shape == (1,)
    assert estimate.t.shape == (1, 2)
    assert estimate.confidence.shape == (1,)
    assert estimate.log_assignment.shape == (1, 5, 6)


def test_head_a_returns_a_pose_estimate_without_an_assignment():
    model = AlignFormerA(embed_dim=8).eval()
    batch = _batch(torch.randn(1, 4, 7), torch.randn(1, 5, 7))

    estimate = model(batch)

    assert estimate.psi.shape == (1,)
    assert estimate.t.shape == (1, 2)
    assert estimate.log_assignment is None


def test_head_b_recovers_the_true_transform_given_oracle_embeddings():
    # Orthogonal one-hot embeddings make the correspondence unambiguous, so the
    # closed-form solver must recover the transform almost exactly. This is the
    # end-to-end sanity check that the wiring - not just each part - is correct.
    torch.manual_seed(0)
    count = 6
    identity = torch.eye(count).unsqueeze(0)

    centres = torch.tensor([[[0.0, 0.0], [12.0, 3.0], [-8.0, 5.0],
                             [20.0, -7.0], [4.0, 9.0], [-15.0, -2.0]]])
    yaws = torch.rand(1, count) * 2 * math.pi
    cav_boxes = _boxes_from_centres(centres, yaws)

    true_psi, true_t = 0.15, torch.tensor([[1.2, -0.8]])
    cos, sin = math.cos(true_psi), math.sin(true_psi)
    rotation = torch.tensor([[cos, -sin], [sin, cos]])
    ego_boxes = cav_boxes.clone()
    ego_boxes[..., :2] = centres @ rotation.T + true_t
    ego_boxes[..., 6] = yaws + true_psi

    model = AlignFormerB(embed_dim=count).eval()
    batch = _batch(
        ego_boxes, cav_boxes,
        ego_embeddings=identity.clone(), cav_embeddings=identity.clone(),
    )
    # Bypass the untrained trunk: score directly on the oracle embeddings.
    model.use_raw_embedding_scores = True

    with torch.no_grad():
        estimate = model(batch)

    assert estimate.psi.item() == pytest.approx(true_psi, abs=1e-3)
    assert estimate.t[0, 0].item() == pytest.approx(1.2, abs=1e-2)
    assert estimate.t[0, 1].item() == pytest.approx(-0.8, abs=1e-2)


def test_head_b_recovers_independent_transforms_across_a_batch():
    # Regression guard for batch-mixing: procrustes.py's own suite is B=1 only,
    # so a bug that mixed samples across the batch dimension (e.g. broadcasting
    # sample 0's transform onto sample 1) would slip past it undetected. Two
    # samples with different true transforms and different geometry make such
    # mixing visible: if the solver used the wrong sample's data for either,
    # at least one of these recoveries fails.
    torch.manual_seed(2)
    count = 6
    identity = torch.eye(count).unsqueeze(0).expand(2, -1, -1)

    centres = torch.stack(
        [
            torch.tensor([[0.0, 0.0], [12.0, 3.0], [-8.0, 5.0],
                          [20.0, -7.0], [4.0, 9.0], [-15.0, -2.0]]),
            torch.tensor([[2.0, -3.0], [-10.0, 4.0], [7.0, -9.0],
                          [15.0, 6.0], [-4.0, -11.0], [9.0, 2.0]]),
        ],
        dim=0,
    )
    yaws = torch.rand(2, count) * 2 * math.pi
    cav_boxes = _boxes_from_centres(centres, yaws)

    true_psi = torch.tensor([0.15, -0.4])
    true_t = torch.tensor([[1.2, -0.8], [-2.0, 3.0]])
    cos, sin = torch.cos(true_psi), torch.sin(true_psi)
    rotation = torch.stack(
        [torch.stack([cos, -sin], dim=-1), torch.stack([sin, cos], dim=-1)], dim=1
    )  # (2, 2, 2)

    ego_boxes = cav_boxes.clone()
    ego_boxes[..., :2] = torch.bmm(centres, rotation.transpose(1, 2)) + true_t.unsqueeze(1)
    ego_boxes[..., 6] = yaws + true_psi.unsqueeze(-1)

    model = AlignFormerB(embed_dim=count).eval()
    batch = _batch(
        ego_boxes, cav_boxes,
        ego_embeddings=identity.clone(), cav_embeddings=identity.clone(),
    )
    model.use_raw_embedding_scores = True

    with torch.no_grad():
        estimate = model(batch)

    for sample in range(2):
        assert estimate.psi[sample].item() == pytest.approx(
            true_psi[sample].item(), abs=1e-3
        )
        assert estimate.t[sample, 0].item() == pytest.approx(
            true_t[sample, 0].item(), abs=1e-2
        )
        assert estimate.t[sample, 1].item() == pytest.approx(
            true_t[sample, 1].item(), abs=1e-2
        )


def test_confidence_is_zero_when_the_ego_set_is_empty():
    model = AlignFormerB(embed_dim=8).eval()
    batch = _batch(torch.zeros(1, 0, 7), torch.randn(1, 3, 7))

    estimate = model(batch)

    assert estimate.confidence.item() == pytest.approx(0.0)
    assert estimate.psi.item() == pytest.approx(0.0)
    assert torch.allclose(estimate.t, torch.zeros(1, 2))


def test_head_a_confidence_is_zero_when_the_ego_set_is_empty():
    # Both heads sit before the trunk's empty-set crash, so both need the
    # guard - not just Head B, which the brief's own test exercises.
    model = AlignFormerA(embed_dim=8).eval()
    batch = _batch(torch.zeros(1, 0, 7), torch.randn(1, 3, 7))

    estimate = model(batch)

    assert estimate.confidence.item() == pytest.approx(0.0)
    assert estimate.psi.item() == pytest.approx(0.0)
    assert torch.allclose(estimate.t, torch.zeros(1, 2))


def test_gradients_reach_the_trunk_through_the_closed_form_solver():
    model = AlignFormerB(embed_dim=8)
    batch = _batch(torch.randn(1, 4, 7), torch.randn(1, 4, 7))

    estimate = model(batch)
    (estimate.psi.sum() + estimate.t.sum()).backward()

    gradients = [p.grad for p in model.trunk.parameters() if p.grad is not None]
    assert gradients, "no gradient reached the trunk"
    assert all(torch.isfinite(g).all() for g in gradients)


def _mixed_batch(embed_dim=8):
    """A 2-sample batch where sample 1's ego set is entirely masked out.

    ``_is_empty`` in model.py only fires when a whole batch's token dimension
    is 0 -- it does not catch a *mixed* batch where some individual rows
    (samples) have zero real objects but the batch is still padded to a
    non-zero size by the other, real samples. Sample 0 here has real ego and
    cav objects; sample 1's ego_mask is all-False (an ego that detected
    nothing this frame) while its cav_mask has real objects, so it takes the
    ordinary forward path, not the pre-trunk guard.
    """
    torch.manual_seed(0)
    ego_boxes = torch.randn(2, 3, 7)
    cav_boxes = torch.randn(2, 4, 7)
    ego_mask = torch.tensor([[True, True, True], [False, False, False]])
    cav_mask = torch.ones(2, 4, dtype=torch.bool)
    return {
        "ego_boxes": ego_boxes,
        "ego_scores": torch.ones(2, 3),
        "ego_embeddings": torch.randn(2, 3, embed_dim),
        "ego_mask": ego_mask,
        "cav_boxes": cav_boxes,
        "cav_scores": torch.ones(2, 4),
        "cav_embeddings": torch.randn(2, 4, embed_dim),
        "cav_mask": cav_mask,
    }


def _partially_padded_batch(embed_dim=8):
    """A single, ORDINARY sample with one padded ego position.

    Review HIGH-1: ``collate`` pads every sample's object sets to the batch
    maximum, so this is not an exotic input -- it is what essentially every
    real multi-object training batch looks like. Ego position 2 here is
    masked out (padding), while positions 0-1 and every cav position are
    real. For AlignFormerB, that padded row has zero valid (ego, cav) pairs,
    hence ``mass == 0`` for that row alone -- the atan2(0, 0) singularity --
    even though the *sample* as a whole is completely ordinary, non-empty,
    and has plenty of total match mass. This is the reviewer's minimal
    reproduction, and it is what makes HIGH-1 broader than
    ``_mixed_batch`` above: it fires on a per-ROW basis, not a per-SAMPLE one.
    """
    torch.manual_seed(1)
    ego_boxes = torch.randn(1, 3, 7)
    cav_boxes = torch.randn(1, 4, 7)
    ego_mask = torch.tensor([[True, True, False]])
    cav_mask = torch.ones(1, 4, dtype=torch.bool)
    return {
        "ego_boxes": ego_boxes,
        "ego_scores": torch.ones(1, 3),
        "ego_embeddings": torch.randn(1, 3, embed_dim),
        "ego_mask": ego_mask,
        "cav_boxes": cav_boxes,
        "cav_scores": torch.ones(1, 4),
        "cav_embeddings": torch.randn(1, 4, embed_dim),
        "cav_mask": cav_mask,
    }


@pytest.mark.parametrize("model_cls", [AlignFormerA, AlignFormerB])
@pytest.mark.parametrize(
    "batch_fn",
    [_mixed_batch, _partially_padded_batch],
    ids=["fully_masked_row", "partially_padded_row"],
)
def test_mixed_batch_with_an_empty_row_produces_no_nan(model_cls, batch_fn):
    """R27: neither an entirely-empty row NOR an ordinary padded row may NaN.

    ``nn.MultiheadAttention`` returns NaN for every query in a batch item
    whose entire key set is masked (softmax over an all -inf row is 0/0).
    The forward guards already in place (``torch.where(present, ..., 0)`` in
    AlignFormerA, ``MIN_MATCH_MASS`` in weighted_se2_kabsch) mask that NaN out
    of the returned *values*, but masking a NaN with a multiply-by-zero does
    not clean the computation graph: ``0 * NaN == NaN``, so the corrupted
    row's NaN reappears in the *gradient* of every shared parameter once you
    call ``.backward()``, even though every forward value is finite. This is
    the regression this test is for; see also AlignFormerB's own
    atan2(0, 0) gradient singularity for a zero-weight row -- exercised by
    ``_partially_padded_batch`` on essentially every real training batch, not
    only by ``_mixed_batch``'s wholly-empty-sample case (review HIGH-1).
    """
    model = model_cls(embed_dim=8)
    batch = batch_fn()

    estimate = model(batch)

    assert torch.isfinite(estimate.psi).all()
    assert torch.isfinite(estimate.t).all()
    assert torch.isfinite(estimate.confidence).all()

    (estimate.psi.sum() + estimate.t.sum() + estimate.confidence.sum()).backward()

    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients, "no gradient reached any parameter"
    assert all(torch.isfinite(g).all() for g in gradients)


@pytest.mark.parametrize("model_cls", [AlignFormerA, AlignFormerB])
def test_fully_empty_row_is_the_identity_correction_with_zero_confidence(model_cls):
    """The wholly-empty-sample-specific guarantee, kept separate from the
    finite-gradient parametrization above: exact identity correction and
    zero confidence are only meaningful for a sample with NO real objects at
    all. The partially-padded case has a real, non-zero correction to
    estimate and no such guarantee applies to it.
    """
    model = model_cls(embed_dim=8)
    batch = _mixed_batch()

    estimate = model(batch)

    assert estimate.psi[1].item() == pytest.approx(0.0)
    assert torch.allclose(estimate.t[1], torch.zeros(2))
    assert estimate.confidence[1].item() == pytest.approx(0.0)


def test_head_b_is_unmoved_by_flipped_cav_box_headings():
    # The detector has no direction classifier, so the heading it reports is
    # only determined modulo pi: measured on the validation cache, 20.3% of
    # cross-agent detections of the SAME physical object disagree by ~180 deg.
    # Each such flip displaces a heading virtual point by 2 * heading_lambda,
    # which is the dominant term in AlignFormer's pose error floor. Flipping a
    # heading describes the same box, so it must not move the estimate.
    torch.manual_seed(0)
    count = 6
    identity = torch.eye(count).unsqueeze(0)

    centres = torch.tensor([[[0.0, 0.0], [12.0, 3.0], [-8.0, 5.0],
                             [20.0, -7.0], [4.0, 9.0], [-15.0, -2.0]]])
    yaws = torch.rand(1, count) * 2 * math.pi
    cav_boxes = _boxes_from_centres(centres, yaws)

    true_psi, true_t = 0.03, torch.tensor([[0.6, -0.4]])
    cos, sin = math.cos(true_psi), math.sin(true_psi)
    rotation = torch.tensor([[cos, -sin], [sin, cos]])
    ego_boxes = cav_boxes.clone()
    ego_boxes[..., :2] = centres @ rotation.T + true_t
    ego_boxes[..., 6] = yaws + true_psi

    # Flip the heading of half the CAV boxes: the same boxes, reported the
    # other way round along their own axis.
    flipped = cav_boxes.clone()
    flipped[0, ::2, 6] = flipped[0, ::2, 6] + math.pi

    model = AlignFormerB(embed_dim=count).eval()
    model.use_raw_embedding_scores = True

    with torch.no_grad():
        clean = model(_batch(ego_boxes, cav_boxes,
                             ego_embeddings=identity.clone(),
                             cav_embeddings=identity.clone()))
        halved = model(_batch(ego_boxes, flipped,
                              ego_embeddings=identity.clone(),
                              cav_embeddings=identity.clone()))

    assert clean.psi.item() == pytest.approx(true_psi, abs=1e-3)
    assert halved.psi.item() == pytest.approx(clean.psi.item(), abs=1e-4)
    assert halved.t[0, 0].item() == pytest.approx(clean.t[0, 0].item(), abs=1e-3)
    assert halved.t[0, 1].item() == pytest.approx(clean.t[0, 1].item(), abs=1e-3)
