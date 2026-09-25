"""Paired error bars over independent noise draws.

Every AP cell on this branch was **one** noise seed. That was never stated as a
problem while the differences being quoted were 0.02-0.04 wide, but the
head-to-head against :mod:`alignformer.freealign` at AP@0.7 turns on
differences of 0.0005-0.011, and at that size a single draw is not a
measurement. This module is the statistics that replace the orderings which
were read off it.

**Pair, then difference -- never the other way round.** Conditions in this
sweep share their noise draw: the same perturbed poses reach ``uncorrected``,
``alignformer``, ``alignformer_irls`` and ``freealign`` inside one seed. They
are therefore strongly correlated, and the variance of their *difference* is
much smaller than either one's own variance::

    Var(A - B) = Var(A) + Var(B) - 2 Cov(A, B)

Overlapping independent error bars would throw the covariance away and declare
a draw wherever the two conditions merely move together across draws, which is
everywhere here. :func:`paired_difference` subtracts inside each seed first and
reports the spread of the differences.

**A difference no larger than its own standard error is a draw.** Not "a small
win", not "a trend": a draw, and in the direction that flatters this project as
readily as in the one that does not. :attr:`PairedDifference.is_draw` and
:func:`verdict` are the single place that rule lives, so a report cannot quietly
apply it asymmetrically.

**One draw is not a spread of zero.** ``sigma = 0`` has no noise to draw --
every seed perturbs by exactly nothing -- so it is run once. Its difference is
resolved *exactly*, and it has no standard error at all. Reporting ``0.0``
there would be a fabricated error bar and would make every deterministic cell
look infinitely significant, so ``sd``/``sem`` are ``None`` and the verdict is
:data:`DETERMINISTIC` rather than a win or a draw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

# Verdicts. ``left``/``right`` name the argument of :func:`paired_difference`
# that is ahead, rather than "win"/"loss", because this module is applied to
# our own arms and to the competitor's with the same code.
LEFT = "left"
RIGHT = "right"
DRAW = "draw"
DETERMINISTIC = "deterministic"


@dataclass(frozen=True)
class Spread:
    """One cell's mean over seeds, with its sample standard deviation.

    ``sd`` is ``None``, not ``0.0``, for a single draw: see the module
    docstring. Frozen, so a number that has been reported cannot be edited
    afterwards.
    """

    mean: float
    sd: Optional[float]
    n: int
    values: Tuple[float, ...]

    @property
    def deterministic(self) -> bool:
        """True when there was only one draw, so there is no spread to report."""
        return self.n == 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean,
            "sd": self.sd,
            "n": self.n,
            "per_seed": list(self.values),
        }


@dataclass(frozen=True)
class PairedDifference:
    """``left - right``, differenced **within** each seed and then averaged.

    ``sem`` is the standard error of that mean: the sample standard deviation
    of the per-seed differences over ``sqrt(n)``. It is ``None`` for a single
    draw.
    """

    mean: float
    sem: Optional[float]
    n: int
    values: Tuple[float, ...]

    @property
    def deterministic(self) -> bool:
        return self.n == 1

    @property
    def is_draw(self) -> Optional[bool]:
        """``|mean| <= sem``; ``None`` when there is no error bar to compare to.

        The comparison is on the absolute value on purpose. A rule that only
        demoted differences pointing the wrong way would not be a rule.
        """
        if self.sem is None:
            return None
        return abs(self.mean) <= self.sem

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean,
            "sem": self.sem,
            "n": self.n,
            "verdict": verdict(self),
            "per_seed": list(self.values),
        }


def spread(values: Sequence[float]) -> Spread:
    """Mean and sample standard deviation (``ddof = 1``) of one cell's draws.

    ``ddof = 1`` because the seeds *estimate* the spread of the noise draw;
    they are not the population.
    """
    observations = tuple(float(value) for value in values)
    if not observations:
        raise ValueError("a spread needs at least one value")
    count = len(observations)
    mean = math.fsum(observations) / count
    sd = None if count == 1 else _sample_sd(observations, mean)
    return Spread(mean=mean, sd=sd, n=count, values=observations)


def paired_difference(
    left: Sequence[float], right: Sequence[float]
) -> PairedDifference:
    """``left - right`` per seed, then the mean and its standard error.

    Both sequences must be the same seeds in the same order -- that is what
    makes the difference paired, and nothing in the types can check it, so the
    caller has to build them from one draw list.
    """
    first = tuple(float(value) for value in left)
    second = tuple(float(value) for value in right)
    if len(first) != len(second):
        raise ValueError(
            "a paired difference needs the same number of seeds on both sides, "
            f"got {len(first)} and {len(second)}"
        )
    if not first:
        raise ValueError("a paired difference needs at least one seed")

    differences = tuple(a - b for a, b in zip(first, second))
    count = len(differences)
    mean = math.fsum(differences) / count
    sem = None if count == 1 else _sample_sd(differences, mean) / math.sqrt(count)
    return PairedDifference(mean=mean, sem=sem, n=count, values=differences)


def unpaired_sem(
    left: Sequence[float], right: Sequence[float]
) -> Optional[float]:
    """The standard error two INDEPENDENT error bars would imply.

    ``sqrt(Var(A)/n + Var(B)/n)`` -- the covariance term dropped. It is not
    what this module reports; it is reported *beside* what this module reports,
    so that "the conditions are strongly correlated, so pairing is tighter" is
    a number a reader can check rather than a claim they have to accept. Where
    the two agree, the pairing bought nothing on that cell, and saying so is
    part of not overclaiming.
    """
    first = spread(left)
    second = spread(right)
    if first.n != second.n:
        raise ValueError(
            "an unpaired standard error needs the same number of seeds on "
            f"both sides, got {first.n} and {second.n}"
        )
    if first.sd is None or second.sd is None:
        return None
    return math.sqrt((first.sd ** 2 + second.sd ** 2) / first.n)


def verdict(difference: PairedDifference) -> str:
    """Which side the error bars support: :data:`LEFT`, :data:`RIGHT`,
    :data:`DRAW`, or :data:`DETERMINISTIC` when there was only one draw."""
    if difference.is_draw is None:
        return DETERMINISTIC
    if difference.is_draw:
        return DRAW
    return LEFT if difference.mean > 0 else RIGHT


def _sample_sd(values: Sequence[float], mean: float) -> float:
    count = len(values)
    return math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (count - 1))
