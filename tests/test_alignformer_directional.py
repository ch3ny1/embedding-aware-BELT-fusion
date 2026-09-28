"""Shrinkage that knows which DIRECTION of the SE(2) the evidence supports.

Task 24's per-pair rule replaced one global ``tau`` with each pair's own
standardized statistic, and that was the right first move. It still applies
**one scalar to all of SE(2)**, and the measured failure is that this is the
wrong shape.

Where it breaks, measured
-------------------------

At sigma = 0 the truth is the identity and every metre we emit is damage. The
emitted translation at sigma = 0 is 0.032 m on validation and **0.120 m on
test**, at almost the same coverage (0.433 vs 0.482) -- we do not fire more
often on test, we fire 3.7x harder. Split by object range at sigma = 0 the
pose MAE is 0.021 m under 25 m, 0.029 m at 25-40 m and **0.112 m at 40-70 m**,
while per-correspondence disagreement over that same span rises only 1.7x
(0.299 -> 0.517 m). The noise did not get 5x worse; the GEOMETRY did. A far
pair's matched objects sit in a narrow angular wedge, the normal matrix ``A``
goes ill-conditioned, and the fit runs away along its weak eigendirection.

A scalar factor cannot express that. Two directions well determined and a
third unidentified leaves ``max(0, 1 - p / T^2)`` one choice: keep the
runaway, or throw away the two good directions with it. The validation slice
holds 579 such pairs (3%), so the cell that costs test 3 points of clean mAP
had almost no weight when the scalar arm was selected on the validation mean.

What this module asserts
------------------------

The formulation that does NOT work
----------------------------------

The obvious move -- keep ``max(0, 1 - p / z^2)`` and apply it per axis at
``p = 1`` -- was tried first and is wrong. James-Stein earns its shrinkage by
POOLING ``p`` dimensions; split into three ``p = 1`` problems it is barely
better than the raw estimate. Measured on this project's own geometry it
emitted **more** spurious correction than the scalar rule it was meant to
improve on (1.35x on a ten-object wedge, 1.69x on an open scatter) and only
broke even on the sparsest wedges. ``test_pooling_is_not_thrown_away`` keeps
that from being retried.

The formulation that ALSO does not work
---------------------------------------

The posterior mean under the already-fitted prior -- whiten by ``shrinkage``'s
tau, then one ridge factor ``mu / (1 + mu)`` per eigen-axis -- is the textbook
anisotropic answer and is the best rule here at sigma = 0 by a wide margin. It
is unusable anyway, and the sigma sweep is what shows it: that tau was
calibrated at sigma = 0, so against a true correction of two metres a prior
saying "corrections are about 0.15 m" destroys an estimate that was already
accurate. On an open ten-object scatter at sigma = 2 it turned a 0.116 m error
into 0.967 m. ``test_a_fixed_prior_is_not_smuggled_back_in`` keeps that from
being retried too.

What works
----------

Both factors scale-free, composed: the pooled positive-part rule that ships
today, then the per-axis positive-part rule in the eigenbasis of the
precision. No constant is introduced, so there is nothing to be calibrated at
one sigma and applied at another. At sigma = 0 it beats the deployed scalar
rule in every geometry cell (2.5x to 4.9x less spurious translation) and at
larger sigma it is far better where the geometry is ill-conditioned and worse
by at most 0.01 m where it is not.

Four properties carry it and each is pinned below.

1. **The statistic is the precision's quadratic form**, ``theta^T M theta ==
   T^2``. If it is not, the arm is not a refinement of the deployed statistic
   but a different one, and every calibration argument made for task 24 stops
   transferring.
2. **A weak axis is crushed while a strong one in the same pair survives.**
   This is the whole point; a rule that could only move both together would be
   the scalar rule with extra arithmetic.
3. **It never lengthens the correction**, so it stays a shrinkage and the
   result stays on the convex hull between the identity and the solve.
4. **It does not depend on the eigenvector sign convention.** ``eigh`` fixes
   no sign, so a rule that read one would be a rule whose answer changed with
   the LAPACK build underneath it.
"""

import math

import pytest
import torch

from embedding_aware_belt_fusion.alignformer.abstain import (
    ABSTENTION_MODES,
    DIRECTIONAL,
    PER_PAIR,
    AbstentionConfig,
    decide,
    directional_theta,
    wald_parts,
)
from embedding_aware_belt_fusion.alignformer.shrinkage import shrinkage_factor
from embedding_aware_belt_fusion.alignformer.model import PoseEstimate
from embedding_aware_belt_fusion.alignformer.procrustes import (
    augment_with_heading,
    weighted_se2_kabsch,
)

# configs/alignformer_r140.yaml, as in test_alignformer_abstain.py: an arm
# tested off the deployed constants is an arm tested on a different problem.
SIGMA_TRANSLATION_M = 0.2516
SIGMA_YAW_RAD = math.radians(4.6392)
HEADING_LAMBDA = 2.0
VARIANCE_CENTRE = 2.0 * SIGMA_TRANSLATION_M ** 2
VARIANCE_HEADING = VARIANCE_CENTRE + HEADING_LAMBDA ** 2 * 2.0 * SIGMA_YAW_RAD ** 2



def _wedge(count, generator, *, radius=55.0, half_angle_deg=6.0):
    """Object centres in a NARROW angular wedge -- the far-pair geometry.

    This is the shape that costs clean mAP: everything the pair can see lies
    along one bearing, so the component of the correction across that bearing
    and the yaw are barely identified while the along-bearing one is fine.
    """
    half = math.radians(half_angle_deg)
    bearing = (torch.rand(count, generator=generator) - 0.5) * 2.0 * half
    span = radius + (torch.rand(count, generator=generator) - 0.5) * 8.0
    return torch.stack([span * torch.cos(bearing), span * torch.sin(bearing)], dim=-1)


def _spread(count, generator, extent=60.0):
    """Object centres scattered all around -- the well-conditioned near pair."""
    return (torch.rand(count, 2, generator=generator) - 0.5) * extent


def _pair(centres, generator, *, psi=0.0, t=(0.0, 0.0)):
    """``(p, q, w)`` in the deployed heading-augmented geometry."""
    count = centres.shape[0]
    yaws = (torch.rand(count, generator=generator) - 0.5) * 2 * math.pi
    axis = SIGMA_TRANSLATION_M / math.sqrt(2.0)

    ego_centres = centres + torch.randn(count, 2, generator=generator) * axis
    ego_yaws = yaws + torch.randn(count, generator=generator) * SIGMA_YAW_RAD
    cav_centres = centres + torch.randn(count, 2, generator=generator) * axis
    cav_yaws = yaws + torch.randn(count, generator=generator) * SIGMA_YAW_RAD

    cos, sin = math.cos(-psi), math.sin(-psi)
    rotation = torch.tensor([[cos, -sin], [sin, cos]], dtype=torch.float)
    shifted = cav_centres - torch.tensor(t, dtype=torch.float)

    p = augment_with_heading(
        ego_centres.unsqueeze(0), ego_yaws.unsqueeze(0), HEADING_LAMBDA
    )
    q = augment_with_heading(
        (shifted @ rotation.T).unsqueeze(0),
        (cav_yaws - psi).unsqueeze(0),
        HEADING_LAMBDA,
    )
    return p, q, torch.ones(1, 2 * count)


def _parts(p, q, w, psi=None, t=None):
    if psi is None:
        psi, t = weighted_se2_kabsch(p, q, w)
    count = p.shape[1] // 2
    return wald_parts(
        p,
        q,
        w,
        psi,
        t,
        variance_centre=torch.full((p.shape[0], count), VARIANCE_CENTRE),
        variance_heading=torch.full((p.shape[0], count), VARIANCE_HEADING),
        heading_lambda=HEADING_LAMBDA,
    )


# --------------------------------------------------------------------------
# 1. THE STATISTIC IS THE PRECISION'S QUADRATIC FORM.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("objects", [2, 3, 6, 12])
def test_the_precision_reproduces_the_deployed_statistic(objects):
    # theta^T M theta == T^2 is what makes this a refinement of task 24's
    # statistic rather than a second, uncalibrated one. Every level argument
    # made there transfers only through this identity.
    generator = torch.Generator().manual_seed(11 + objects)
    p, q, w = _pair(_spread(objects, generator), generator, psi=0.05, t=(0.6, -0.4))
    psi, t = weighted_se2_kabsch(p, q, w)
    theta = torch.cat([t, psi.reshape(-1, 1)], dim=-1)

    parts = _parts(p, q, w, psi, t)

    quadratic = torch.einsum("bi,bij,bj->b", theta, parts.precision, theta)
    assert torch.allclose(quadratic, parts.statistic, rtol=1e-4, atol=1e-5)


def test_the_precision_is_symmetric_and_positive_semidefinite():
    generator = torch.Generator().manual_seed(5)
    p, q, w = _pair(_spread(8, generator), generator, psi=0.1, t=(1.0, 0.5))

    precision = _parts(p, q, w).precision

    assert torch.allclose(precision, precision.transpose(-1, -2), atol=1e-6)
    assert float(torch.linalg.eigvalsh(precision).min()) >= -1e-5


def test_the_deployed_statistic_is_unchanged_by_the_refactor():
    # wald_statistic keeps its signature and its VALUE: the shipped per_pair
    # arm has to reproduce bit for bit in the same run that measures this one.
    from embedding_aware_belt_fusion.alignformer.abstain import wald_statistic

    generator = torch.Generator().manual_seed(23)
    p, q, w = _pair(_spread(7, generator), generator, psi=0.2, t=(1.5, -1.0))
    psi, t = weighted_se2_kabsch(p, q, w)
    count = p.shape[1] // 2
    kwargs = dict(
        variance_centre=torch.full((1, count), VARIANCE_CENTRE),
        variance_heading=torch.full((1, count), VARIANCE_HEADING),
        heading_lambda=HEADING_LAMBDA,
    )

    statistic, dof = wald_statistic(p, q, w, psi, t, **kwargs)
    parts = wald_parts(p, q, w, psi, t, **kwargs)

    assert torch.equal(statistic, parts.statistic)
    assert torch.equal(dof, parts.dof)


# --------------------------------------------------------------------------
# 2. A WEAK AXIS IS CRUSHED WHILE A STRONG ONE IN THE SAME PAIR SURVIVES.
# --------------------------------------------------------------------------


def test_a_weak_axis_is_crushed_while_a_strong_axis_in_the_same_pair_survives():
    # Hand-built so the claim is about the RULE and not about whether a
    # particular random wedge happened to be ill-conditioned. Eigenvalues span
    # four orders of magnitude, which is the wedge case in miniature.
    evidences = (1.0, 1e-12, 1e-6)
    precision = torch.diag(torch.tensor(evidences, dtype=torch.float64)).unsqueeze(0)
    theta = torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float64)
    statistic = float(torch.einsum("bi,bij,bj->b", theta, precision, theta))
    ridge = 3.0 * (evidences[0] * evidences[1] * evidences[2]) ** (1.0 / 3.0) / statistic

    shrunk = directional_theta(theta, precision)

    for axis, evidence in enumerate(evidences):
        assert shrunk[0, axis] == pytest.approx(
            2.0 * evidence / (evidence + ridge), rel=1e-4
        )
    # The unidentified axis keeps under 2% of itself; the well-determined one
    # keeps over 99%. That separation is the entire point.
    assert shrunk[0, 1] / 2.0 < 0.02 < 0.99 < shrunk[0, 0] / 2.0
    # And the scale that separates them is the fit's own, not a constant: the
    # ridge is p det(M)^(1/3) / T^2, so an axis is weak RELATIVE to this pair.
    assert ridge == pytest.approx(3.0 * 1e-6 / statistic, rel=1e-9)


def test_pooling_is_not_thrown_away():
    # The formulation this module rejects: per-axis max(0, 1 - 1/z^2). On the
    # narrow-wedge geometry at sigma = 0 -- where the truth is the identity and
    # every metre emitted is damage -- it must not beat the rule that ships,
    # or the reason for the posterior-mean form would be gone. Kept as a test
    # so the dead end is not quietly retried.
    generator = torch.Generator().manual_seed(404)
    centres = _wedge(10, generator)
    per_axis = scalar = 0.0
    for _ in range(96):
        p, q, w = _pair(centres, generator)
        psi, t = weighted_se2_kabsch(p, q, w)
        parts = _parts(p, q, w, psi, t)
        theta = torch.cat([t, psi.reshape(-1, 1)], dim=-1)

        evidence, basis = torch.linalg.eigh(parts.precision)
        components = torch.matmul(
            basis.transpose(-1, -2), theta.unsqueeze(-1)
        ).squeeze(-1)
        standardized = evidence.clamp_min(0.0).sqrt() * components
        split = torch.matmul(
            basis,
            (shrinkage_factor(standardized.pow(2), dimensions=1) * components)
            .unsqueeze(-1),
        ).squeeze(-1)

        per_axis += float(split[:, :2].norm())
        scalar += float((theta * shrinkage_factor(parts.statistic).unsqueeze(-1))[:, :2].norm())

    assert per_axis > scalar


def test_the_narrow_wedge_keeps_less_of_its_runaway_than_the_scalar_rule():
    # The measured failure, as a test. At sigma = 0 the true correction is the
    # identity, so a smaller emitted translation is strictly better, and the
    # wedge is the geometry that costs test 3 points of clean mAP.
    generator = torch.Generator().manual_seed(404)
    centres = _wedge(10, generator)
    directional = scalar = 0.0
    for _ in range(96):
        p, q, w = _pair(centres, generator)
        psi, t = weighted_se2_kabsch(p, q, w)
        parts = _parts(p, q, w, psi, t)
        theta = torch.cat([t, psi.reshape(-1, 1)], dim=-1)

        directional += float(
            directional_theta(theta, parts.precision)[:, :2].norm()
        )
        scalar += float(
            (theta * shrinkage_factor(parts.statistic).unsqueeze(-1))[:, :2].norm()
        )

    assert directional < scalar


def test_every_axis_far_outside_the_prior_is_left_almost_alone():
    prior = torch.ones(3)
    precision = torch.diag(torch.tensor([1e4, 1e4, 1e4])).unsqueeze(0)
    theta = torch.tensor([[3.0, -1.0, 0.4]])

    assert torch.allclose(
        directional_theta(theta, precision), theta, rtol=2e-3
    )


# --------------------------------------------------------------------------
# 3. IT IS A SHRINKAGE.
# --------------------------------------------------------------------------


def test_the_correction_is_never_lengthened():
    generator = torch.Generator().manual_seed(77)
    for objects in (2, 3, 5, 9):
        p, q, w = _pair(_spread(objects, generator), generator, psi=0.3, t=(2.0, 1.0))
        psi, t = weighted_se2_kabsch(p, q, w)
        parts = _parts(p, q, w, psi, t)
        theta = torch.cat([t, psi.reshape(-1, 1)], dim=-1)

        shrunk = directional_theta(theta, parts.precision)

        assert float(shrunk.norm()) <= float(theta.norm()) + 1e-6


def test_an_identity_correction_stays_the_identity():
    precision = torch.eye(3).unsqueeze(0)
    theta = torch.zeros(1, 3)

    assert torch.allclose(directional_theta(theta, precision), theta)


def test_it_is_invariant_to_the_eigenvector_sign_convention():
    # torch.linalg.eigh fixes no sign. A rule that read one would give a
    # different answer on a different LAPACK build. Conjugating the precision
    # by a sign flip is that same ambiguity, expressed in the input.
    generator = torch.Generator().manual_seed(909)
    p, q, w = _pair(_wedge(5, generator), generator, psi=0.15, t=(1.2, -0.8))
    psi, t = weighted_se2_kabsch(p, q, w)
    parts = _parts(p, q, w, psi, t)
    theta = torch.cat([t, psi.reshape(-1, 1)], dim=-1)

    flip = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    flipped = directional_theta(
        theta @ flip, flip @ parts.precision @ flip
    ) @ flip

    assert torch.allclose(
        directional_theta(theta, parts.precision), flipped, atol=1e-6
    )


def test_an_isotropic_precision_gives_one_factor_to_every_axis():
    # With nothing to distinguish the axes the rule must not invent a
    # distinction; it collapses to the single factor T^2 / (T^2 + p).
    precision = (4.0 * torch.eye(3)).unsqueeze(0)
    theta = torch.tensor([[1.0, -2.0, 0.5]])
    statistic = float(torch.einsum("bi,bij,bj->b", theta, precision, theta))

    assert torch.allclose(
        directional_theta(theta, precision),
        theta * statistic / (statistic + 3.0),
        rtol=1e-5,
    )


def test_it_does_not_depend_on_which_eigenbasis_lapack_returns():
    # THE defect that eliminated the per-component formulation. On the
    # well-conditioned pairs that are the majority, the two translation axes
    # sit at mu_2 / mu_1 = 1.09 to 1.15 -- near enough to degenerate that the
    # eigenbasis is arbitrary -- and a rule whose factor read the components
    # returned (0.565, -1.568, 0.000) in one basis and (0.549, -1.340, 0.794)
    # in a rotated one, for the same fit. Conjugating by a rotation is that
    # ambiguity written down.
    generator = torch.Generator().manual_seed(3)
    rotation = torch.linalg.qr(torch.randn(3, 3, generator=generator))[0]
    for precision in (
        (4.0 * torch.eye(3)).unsqueeze(0),
        torch.diag(torch.tensor([9.0, 9.0, 0.02])).unsqueeze(0),
    ):
        theta = torch.tensor([[1.0, -2.0, 0.5]])

        rotated = directional_theta(
            theta @ rotation, rotation.T @ precision @ rotation
        ) @ rotation.T

        assert torch.allclose(
            directional_theta(theta, precision), rotated, atol=1e-5
        )


def test_a_fixed_prior_is_not_smuggled_back_in():
    # The rule must be scale-free: scaling the correction and the precision so
    # the statistic is unchanged must scale the answer by the same factor. A
    # rule carrying a calibrated tau cannot do this, and that is exactly how
    # the posterior-mean form destroyed a good two-metre correction.
    precision = torch.diag(torch.tensor([9.0, 0.5, 2.0])).unsqueeze(0)
    theta = torch.tensor([[1.0, -2.0, 0.5]])
    alpha = 100.0

    scaled = directional_theta(alpha * theta, precision / alpha ** 2)

    assert torch.allclose(
        scaled, alpha * directional_theta(theta, precision), rtol=1e-4
    )


def test_directional_is_a_mode_and_names_itself_without_a_level():
    config = AbstentionConfig(mode=DIRECTIONAL)

    assert DIRECTIONAL in ABSTENTION_MODES
    assert config.name == "alignformer_directional"
    assert config.enabled


def test_directional_refuses_a_level_rather_than_ignoring_one():
    # Same reason per_pair does: a constant silently dropped is a constant a
    # reader would believe was in force.
    with pytest.raises(ValueError, match="level"):
        AbstentionConfig(mode=DIRECTIONAL, level=0.05)


def test_deciding_directionally_without_the_precision_is_an_error():
    estimate = PoseEstimate(
        psi=torch.tensor([0.3]),
        t=torch.tensor([[1.0, -2.0]]),
        confidence=torch.tensor([9.0]),
        offset_statistic=torch.tensor([9.0]),
        offset_dof=torch.tensor([100.0]),
    )

    with pytest.raises(ValueError, match="precision"):
        decide(estimate, AbstentionConfig(mode=DIRECTIONAL))


def test_deciding_directionally_rewrites_both_the_yaw_and_the_translation():
    estimate = PoseEstimate(
        psi=torch.tensor([0.4]),
        t=torch.tensor([[2.0, 2.0]]),
        confidence=torch.tensor([9.0]),
        offset_statistic=torch.tensor([16.25]),
        offset_dof=torch.tensor([100.0]),
        offset_precision=torch.diag(
            torch.tensor([1.0, 1e-12, 1e-12])
        ).unsqueeze(0),
    )

    decided = decide(estimate, AbstentionConfig(mode=DIRECTIONAL))

    # One axis carries essentially all the evidence and survives; the two that
    # carry none are crushed, yaw included.
    assert float(decided.t[0, 0]) == pytest.approx(2.0, rel=1e-3)
    assert float(decided.t[0, 1]) == pytest.approx(0.0, abs=1e-3)
    assert float(decided.psi[0]) == pytest.approx(0.0, abs=1e-3)


def test_a_disabled_decision_still_returns_the_estimate_untouched():
    estimate = PoseEstimate(
        psi=torch.tensor([0.4]),
        t=torch.tensor([[2.0, 2.0]]),
        confidence=torch.tensor([9.0]),
        offset_statistic=torch.tensor([16.0]),
        offset_dof=torch.tensor([100.0]),
    )

    assert decide(estimate, AbstentionConfig(mode="none")) is estimate


def test_the_per_pair_arm_is_unaffected_by_the_new_field():
    # per_pair reads only the scalar statistic, so carrying a precision must
    # not change what it emits -- the shipped arm and the new one run side by
    # side in one sweep.
    common = dict(
        psi=torch.tensor([0.4]),
        t=torch.tensor([[2.0, 2.0]]),
        confidence=torch.tensor([9.0]),
        offset_statistic=torch.tensor([16.25]),
        offset_dof=torch.tensor([100.0]),
    )
    bare = decide(PoseEstimate(**common), AbstentionConfig(mode=PER_PAIR))
    carrying = decide(
        PoseEstimate(**common, offset_precision=torch.eye(3).unsqueeze(0)),
        AbstentionConfig(mode=PER_PAIR),
    )

    assert torch.equal(bare.t, carrying.t)
    assert torch.equal(bare.psi, carrying.psi)


def test_the_directional_factor_has_no_scalar_form():
    # decision_factor returns ONE number; this rule is not one number, and
    # saying so is better than returning something plausible.
    from embedding_aware_belt_fusion.alignformer.abstain import decision_factor

    with pytest.raises(ValueError, match="one factor|decide"):
        decision_factor(
            torch.tensor([9.0]),
            torch.tensor([100.0]),
            AbstentionConfig(mode=DIRECTIONAL),
        )
