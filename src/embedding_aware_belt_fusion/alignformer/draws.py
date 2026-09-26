"""One noise draw of the AP sweep, and which draws a sweep actually makes.

Every AP cell on this branch before task 23 was a single noise seed. The
head-to-head against :mod:`alignformer.freealign` at AP@0.7 turns on
differences of 0.0005-0.011, which one draw cannot resolve, so the sweep now
runs several independent draws per sigma and the per-seed *difference* between
two conditions becomes the statistic (:mod:`alignformer.seedstats`).

That difference is only paired if every condition inside one draw sees the
**same** perturbed poses. :func:`sweep_noisy_poses` is the one place a
perturbation is produced, exactly once per ``(sigma, seed, frame)``; everything
downstream consumes its result. Re-drawing per condition would leave the
conditions varying independently and inflate every error bar by roughly
``sqrt(2)`` even where the two conditions are identical.

``sigma = 0`` perturbs nothing -- ``perturb_pose_2d`` adds ``N(0, 0)``, which is
exactly zero for every generator state -- so all seeds would produce
byte-identical predictions there. :func:`draw_seeds` runs that level **once**
and :func:`draw_slots` gives every seed the *same* accumulator for it, so the
cell is reported as one deterministic run rather than as five copies of one
number with a fabricated standard deviation of 0.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import numpy as np

from embedding_aware_belt_fusion.alignformer.boxes import AgentDetections
from embedding_aware_belt_fusion.alignformer.dataset import YAW_STD_PER_XY_STD
from embedding_aware_belt_fusion.alignformer.fusion import correct_detections
from embedding_aware_belt_fusion.alignformer.shrinkage import (
    ShrinkageCalibration,
    shrink,
)
from embedding_aware_belt_fusion.coloca.geometry import perturb_pose_2d

T = TypeVar("T")


def sweep_rng(seed: int, sigma: float, frame: int, agent: int) -> np.random.Generator:
    """Deterministic per-(seed, sigma, frame, agent) noise draw.

    Keyed on sigma as well as on the frame so that the sweep's levels are
    independent draws rather than one draw rescaled -- a rescaled draw would
    make every level's error point in the same direction and turn the sweep
    into a single sample. Keyed on the agent INDEX rather than on the agent's
    pose, so two CAVs that happen to sit at the same pose still get independent
    perturbations.
    """
    return np.random.default_rng([seed, int(round(sigma * 1000)), frame, agent])


def sweep_noisy_poses(
    poses: Mapping[str, Sequence[float]],
    cav_keys: Sequence[str],
    *,
    sigma: float,
    seed: int,
    frame: int,
) -> Dict[str, List[float]]:
    """The perturbed pose of every CAV in ONE draw, keyed by agent.

    This is the single object every condition in a draw is built from, which is
    what makes the per-seed difference between two conditions paired. It is
    reproducible from ``(seed, sigma, frame, agent)`` alone, so a rerun is the
    same experiment and the seed that matches an earlier single-seed sweep
    reproduces it bit for bit.
    """
    return {
        key: perturb_pose_2d(
            poses[key], sigma, sigma * YAW_STD_PER_XY_STD,
            sweep_rng(seed, sigma, frame, agent),
        )
        for agent, key in enumerate(cav_keys)
    }


def draw_seeds(sigma: float, seeds: Sequence[int]) -> List[int]:
    """The seeds one sigma is actually drawn under; ``sigma = 0`` gets one."""
    return list(seeds[:1]) if sigma == 0.0 else list(seeds)


def sweep_draws(
    sigmas: Sequence[float], seeds: Sequence[int]
) -> List[Tuple[float, int]]:
    """Every ``(sigma, seed)`` the sweep actually evaluates, in a fixed order."""
    if not seeds:
        raise ValueError("a sweep needs at least one seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"sweep seeds must be distinct, got {list(seeds)}")
    if len(set(sigmas)) != len(sigmas):
        # draw_slots keys on (name, sigma, seed); a repeated sigma collapses
        # into one accumulator and the second pass overwrites the first.
        raise ValueError(f"sweep sigmas must be distinct, got {list(sigmas)}")
    return [(sigma, seed) for sigma in sigmas for seed in draw_seeds(sigma, seeds)]


def draw_slots(
    names: Sequence[str],
    sigmas: Sequence[float],
    seeds: Sequence[int],
    factory: Callable[[], T],
) -> Dict[Tuple[str, float, int], T]:
    """One accumulator per ``(name, sigma, seed)``, SHARED where there is no draw.

    A sigma that is drawn once gets one accumulator that every seed's entry
    points at, so the single run reaches all of them, nothing is recomputed,
    and no seed can silently disagree with another about a cell whose value
    cannot depend on the seed. Predictions and pose statistics are both laid
    out by this one function, so the two cannot come apart.
    """
    slots: Dict[Tuple[str, float, int], T] = {}
    for name in names:
        for sigma in sigmas:
            drawn = draw_seeds(sigma, seeds)
            for seed in seeds:
                slots[(name, sigma, seed)] = (
                    factory() if seed in drawn else slots[(name, sigma, drawn[0])]
                )
    return slots


@dataclass(frozen=True)
class DrawnCondition:
    """What one condition contributed to one draw: the estimate AND the boxes.

    Both come out of :func:`draw_aligned` together so that the pose statistics
    and the fused AP can never be computed from differently shrunk copies of
    the same estimate.
    """

    estimate: Any
    detections: AgentDetections


def draw_aligned(
    noisy_detections: AgentDetections,
    estimates: Mapping[str, Any],
    shrinkage: Optional[ShrinkageCalibration],
    *,
    unshrunk: Sequence[str],
) -> Dict[str, DrawnCondition]:
    """Each condition's scored estimate and corrected boxes, from ONE perturbed set.

    Taking ``noisy_detections`` as a single argument is the point: it is
    structurally impossible for one condition to be scored against a different
    perturbation from another's, which is the property every error bar in
    :mod:`alignformer.seedstats` rests on.

    ``unshrunk`` names the conditions the shrinkage calibration must NOT reach.
    It is a required argument rather than a default because the one condition
    that belongs in it -- the FreeAlign reimplementation -- would be scored
    through AlignFormer's own calibration if it were ever forgotten, and that
    is a fairness error the output would not reveal. ``noisy_detections`` is
    not mutated; ``correct_detections`` returns a new object.
    """
    aligned: Dict[str, DrawnCondition] = {}
    for name, estimate in estimates.items():
        if shrinkage is not None and name not in unshrunk:
            estimate = shrink(estimate, shrinkage)
        aligned[name] = DrawnCondition(
            estimate=estimate,
            detections=correct_detections(
                noisy_detections, estimate.psi[0], estimate.t[0]
            ),
        )
    return aligned
