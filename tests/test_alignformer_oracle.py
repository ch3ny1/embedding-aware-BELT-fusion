"""The oracle-correspondence ceiling: head B's solver on the ground-truth match.

Task 21 measures a hard upper bound on what ANY improvement to cross-agent
matching can deliver, by replacing the learned Sinkhorn correspondence with the
one-to-one assignment ``gt_ids`` already carries and re-running the sweep. The
number is only a ceiling if the substitution is exactly that substitution, so
what is pinned here is:

- the oracle assignment recovers a known SE(2) to the same tolerance the
  learned head reaches on oracle embeddings, and it does so from *ids*, not
  from row order;
- an ego object whose ground-truth id appears in no CAV row contributes zero
  mass, exactly as the dustbin would, and therefore cannot move the fit;
- the heading pi-fold is still applied, so a back-to-front CAV detection does
  not displace its heading virtual point;
- the two weighting variants coincide when every detection is equally
  confident, which is the only case in which inverse-variance weighting is a
  no-op;
- nothing but the correspondence differs from ``AlignFormerB.forward``.

And the NEGATIVE CONTROL, which this project has been bitten by the absence of
nine times: permuting the CAV ids within a frame must destroy the recovery. A
version of these tests that passes with shuffled ids would be measuring the
geometry, not the correspondence, and the ceiling it reported would be a
fiction.
"""

import math

import pytest
import torch

from embedding_aware_belt_fusion.alignformer.model import AlignFormerB
from embedding_aware_belt_fusion.alignformer.oracle import (
    oracle_assignment,
    oracle_pose_estimate,
)
from embedding_aware_belt_fusion.alignformer.variance import (
    UNWEIGHTED,
    CorrespondenceVarianceModel,
)

HEADING_LAMBDA = 2.0
TRUE_PSI = 0.15
TRUE_T = (1.2, -0.8)

# The deployed r140 inverse-variance model (configs/alignformer_r140.yaml,
# fitted in outputs/alignformer/r140/correspondence_variance_result.json).
DEPLOYED_IVW = CorrespondenceVarianceModel(
    mode="scalar",
    sigma_translation_m=0.2516,
    translation_exponent=1.1281,
    sigma_yaw_rad=math.radians(4.6392),
    yaw_exponent=1.9322,
    score_reference=0.4,
)

_CENTRES = [[0.0, 0.0], [12.0, 3.0], [-8.0, 5.0], [20.0, -7.0], [4.0, 9.0], [-15.0, -2.0]]


def _boxes_from_centres(centres, yaws):
    """(B, N, 7) boxes with fixed z/h/w/l, matching tests/test_alignformer_model.py."""
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


def _batch(ego_boxes, cav_boxes, *, ego_scores=None, cav_scores=None, embed_dim=8):
    batch_size = ego_boxes.shape[0]
    ego_count, cav_count = ego_boxes.shape[1], cav_boxes.shape[1]
    return {
        "ego_boxes": ego_boxes,
        "ego_scores": (
            ego_scores if ego_scores is not None else torch.full((batch_size, ego_count), 0.4)
        ),
        "ego_embeddings": torch.zeros(batch_size, ego_count, embed_dim),
        "ego_mask": torch.ones(batch_size, ego_count, dtype=torch.bool),
        "cav_boxes": cav_boxes,
        "cav_scores": (
            cav_scores if cav_scores is not None else torch.full((batch_size, cav_count), 0.4)
        ),
        "cav_embeddings": torch.zeros(batch_size, cav_count, embed_dim),
        "cav_mask": torch.ones(batch_size, cav_count, dtype=torch.bool),
    }


def _paired_scene(seed: int = 0, cav_order=None):
    """An ego/CAV object set related by exactly (TRUE_PSI, TRUE_T).

    ``cav_order`` permutes the CAV rows, so a test that passes only because row
    *i* happens to face row *i* fails here.
    """
    torch.manual_seed(seed)
    centres = torch.tensor([_CENTRES])
    count = centres.shape[1]
    yaws = torch.rand(1, count) * 2 * math.pi

    cos, sin = math.cos(TRUE_PSI), math.sin(TRUE_PSI)
    rotation = torch.tensor([[cos, -sin], [sin, cos]])
    ego_centres = centres @ rotation.T + torch.tensor([list(TRUE_T)])
    ego_boxes = _boxes_from_centres(ego_centres, yaws + TRUE_PSI)

    order = list(range(count)) if cav_order is None else list(cav_order)
    cav_boxes = _boxes_from_centres(centres[:, order], yaws[:, order])
    ego_ids = [str(index) for index in range(count)]
    cav_ids = [str(index) for index in order]
    return ego_boxes, cav_boxes, ego_ids, cav_ids


def _recover(batch, ego_ids, cav_ids, variance_model=UNWEIGHTED):
    assignment = oracle_assignment(ego_ids, cav_ids)
    return oracle_pose_estimate(
        batch, assignment, heading_lambda=HEADING_LAMBDA, variance_model=variance_model
    )


def _translation_error(estimate) -> float:
    return float(
        torch.hypot(estimate.t[0, 0] - TRUE_T[0], estimate.t[0, 1] - TRUE_T[1]).item()
    )


def test_the_oracle_assignment_is_one_to_one_and_drops_unmatched_ids():
    # `None` means "matched no ground-truth object", so two of them are
    # different objects; a duplicated id keeps only its first occurrence.
    assignment = oracle_assignment(["a", None, "b", "zz"], ["b", None, "a", "a"])

    assert assignment.shape == (1, 4, 4)
    assert assignment[0].tolist() == [
        [0.0, 0.0, 1.0, 0.0],   # ego "a" -> the FIRST cav "a"
        [0.0, 0.0, 0.0, 0.0],   # ego None matches nothing
        [1.0, 0.0, 0.0, 0.0],   # ego "b"
        [0.0, 0.0, 0.0, 0.0],   # ego "zz" has no counterpart
    ]
    # One-to-one in both directions.
    assert assignment[0].sum(dim=0).max().item() <= 1.0
    assert assignment[0].sum(dim=1).max().item() <= 1.0


def test_the_oracle_assignment_recovers_the_true_transform():
    # Same tolerance as
    # test_head_b_recovers_the_true_transform_given_oracle_embeddings.
    ego_boxes, cav_boxes, ego_ids, cav_ids = _paired_scene(cav_order=[3, 0, 5, 1, 4, 2])

    estimate = _recover(_batch(ego_boxes, cav_boxes), ego_ids, cav_ids)

    assert estimate.psi.item() == pytest.approx(TRUE_PSI, abs=1e-3)
    assert estimate.t[0, 0].item() == pytest.approx(TRUE_T[0], abs=1e-2)
    assert estimate.t[0, 1].item() == pytest.approx(TRUE_T[1], abs=1e-2)
    # Six true pairs, each carrying unit mass.
    assert estimate.confidence.item() == pytest.approx(6.0, abs=1e-5)


def test_shuffling_the_cav_ids_destroys_the_recovered_pose():
    # NEGATIVE CONTROL. The geometry is unchanged and only the id labelling is
    # permuted, so a test suite that still recovers the transform here is
    # measuring the scene, not the correspondence.
    ego_boxes, cav_boxes, ego_ids, cav_ids = _paired_scene(cav_order=[3, 0, 5, 1, 4, 2])
    deranged = [cav_ids[index] for index in (1, 2, 3, 4, 5, 0)]

    honest = _recover(_batch(ego_boxes, cav_boxes), ego_ids, cav_ids)
    shuffled = _recover(_batch(ego_boxes, cav_boxes), ego_ids, deranged)

    assert _translation_error(honest) < 0.01
    assert _translation_error(shuffled) > 1.0
    assert abs(shuffled.psi.item() - TRUE_PSI) > math.radians(5.0)


def test_an_ego_object_with_no_cav_counterpart_contributes_zero_mass():
    # Exactly what the dustbin would do: the extra ego object is 60 m away and
    # points the other way, so if it carried any weight at all the fit would
    # visibly move.
    ego_boxes, cav_boxes, ego_ids, cav_ids = _paired_scene()
    orphan = _boxes_from_centres(
        torch.tensor([[[60.0, -45.0]]]), torch.tensor([[2.7]])
    )
    widened = torch.cat([ego_boxes, orphan], dim=1)

    matched = _recover(_batch(ego_boxes, cav_boxes), ego_ids, cav_ids)
    with_orphan = _recover(
        _batch(widened, cav_boxes), ego_ids + ["no-such-object"], cav_ids
    )

    assert with_orphan.psi.item() == pytest.approx(matched.psi.item(), abs=1e-6)
    assert torch.allclose(with_orphan.t, matched.t, atol=1e-5)
    assert with_orphan.confidence.item() == pytest.approx(matched.confidence.item(), abs=1e-5)


def test_the_oracle_is_unmoved_by_a_back_to_front_cav_heading():
    # The detector reports no direction, so 20.3% of cross-agent detections of
    # the same object disagree by ~180 degrees (procrustes.py). The fold must
    # still be applied on the oracle path or the ceiling would be measured with
    # the very defect the fold removes.
    ego_boxes, cav_boxes, ego_ids, cav_ids = _paired_scene()
    flipped = cav_boxes.clone()
    flipped[0, 2, 6] += math.pi

    honest = _recover(_batch(ego_boxes, cav_boxes), ego_ids, cav_ids)
    with_flip = _recover(_batch(ego_boxes, flipped), ego_ids, cav_ids)

    assert with_flip.psi.item() == pytest.approx(honest.psi.item(), abs=1e-5)
    assert torch.allclose(with_flip.t, honest.t, atol=1e-4)


def test_inverse_variance_weighting_is_a_no_op_at_equal_confidence():
    # rho depends on the score alone, so equal scores make every weight equal
    # and the two variants must coincide. If they diverge here the IVW variant
    # is reading something it should not.
    ego_boxes, cav_boxes, ego_ids, cav_ids = _paired_scene()
    batch = _batch(ego_boxes, cav_boxes)

    uniform = _recover(batch, ego_ids, cav_ids, UNWEIGHTED)
    weighted = _recover(batch, ego_ids, cav_ids, DEPLOYED_IVW)

    assert weighted.psi.item() == pytest.approx(uniform.psi.item(), abs=1e-6)
    assert torch.allclose(weighted.t, uniform.t, atol=1e-5)


def test_unequal_confidence_makes_the_two_variants_differ():
    # The converse of the test above: with a spread of detection scores the
    # deployed weights are NOT uniform, so a run that reported both variants as
    # identical would mean the variance model was never applied.
    ego_boxes, cav_boxes, ego_ids, cav_ids = _paired_scene()
    # Displace one CAV box so the fit is over-determined and the weights matter.
    perturbed = cav_boxes.clone()
    perturbed[0, 0, :2] += torch.tensor([1.5, -1.0])
    scores = torch.tensor([[0.95, 0.9, 0.85, 0.8, 0.75, 0.22]])
    batch = _batch(ego_boxes, perturbed, ego_scores=scores, cav_scores=scores)

    uniform = _recover(batch, ego_ids, cav_ids, UNWEIGHTED)
    weighted = _recover(batch, ego_ids, cav_ids, DEPLOYED_IVW)

    assert abs(weighted.psi.item() - uniform.psi.item()) > 1e-4


def test_nothing_but_the_correspondence_differs_from_head_b():
    # Orthogonal one-hot embeddings drive the Sinkhorn correspondence to the
    # ground-truth assignment, so head B's own forward and the oracle path must
    # agree. This is what makes the substitution a *ceiling on matching* rather
    # than a different estimator.
    ego_boxes, cav_boxes, ego_ids, cav_ids = _paired_scene(cav_order=[3, 0, 5, 1, 4, 2])
    count = ego_boxes.shape[1]
    batch = _batch(ego_boxes, cav_boxes, embed_dim=count)
    identity = torch.eye(count).unsqueeze(0)
    batch["ego_embeddings"] = identity.clone()
    # Row i of the CAV side is ego object `cav_ids[i]`, so its one-hot must be
    # that ego object's basis vector.
    batch["cav_embeddings"] = identity[:, [int(i) for i in cav_ids]].clone()

    model = AlignFormerB(embed_dim=count, heading_lambda=HEADING_LAMBDA).eval()
    model.use_raw_embedding_scores = True
    with torch.no_grad():
        learned = model(batch)
    oracle = _recover(batch, ego_ids, cav_ids)

    assert oracle.psi.item() == pytest.approx(learned.psi.item(), abs=1e-3)
    assert torch.allclose(oracle.t, learned.t, atol=1e-2)


def test_an_empty_object_set_is_the_identity_correction():
    ego_boxes, cav_boxes, ego_ids, _ = _paired_scene()
    empty = _batch(ego_boxes, cav_boxes[:, :0])

    estimate = _recover(empty, ego_ids, [])

    assert estimate.psi.item() == 0.0
    assert torch.equal(estimate.t, torch.zeros(1, 2))
    assert estimate.confidence.item() == 0.0


def test_a_single_matched_pair_is_below_the_minimum_match_mass():
    # procrustes.MIN_MATCH_MASS suppresses a fit with less than one effective
    # matched object; the augmented weight vector doubles the mass, so one true
    # pair sits exactly at the floor and is still solved. Zero pairs is the
    # fallback. Pinned so the ceiling run's fallback fraction is interpretable.
    ego_boxes, cav_boxes, ego_ids, cav_ids = _paired_scene()
    batch = _batch(ego_boxes, cav_boxes)

    lonely = _recover(batch, ego_ids, ["0"] + [None] * 5)
    orphaned = _recover(batch, ego_ids, [None] * 6)

    assert lonely.confidence.item() == pytest.approx(1.0, abs=1e-5)
    assert orphaned.psi.item() == 0.0
    assert torch.equal(orphaned.t, torch.zeros(1, 2))
