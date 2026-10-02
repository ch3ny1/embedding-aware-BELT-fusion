"""A frame-level pose graph over all agents: a sparse pair solved through a third agent.

Why. The pairs sharing one or two objects are 23 % of V2X-Real test frames
and the pairs sharing none another 10 %; no pairwise estimator can do much
with them (FreeAlign abstains, the soft fit answers with wrong matches in
the clean case). But every validation frame and 58 % of test frames carry
four agents, and a CAV that shares one object with the ego often shares
several with another agent whose correction to the ego is well determined.
Pairwise corrections compose, so the graph fills the sparse pair in from the
dense ones.

The convention, derived once and pinned by a test. Each CAV ``j`` has a
localization error; the sweep expresses it as the ego-frame correction
``C_j`` that takes ``j``'s boxes, projected with the noisy pose, onto their
true places in the ego frame. Those are the unknowns. The ego's own estimate
of ``j`` measures ``C_j`` directly. A CAV ``i`` that projects ``j``'s boxes
into ITS frame with both noisy poses sees

    M_ij = T_i<-ego . C_j . C_i^-1 . T_ego<-i

where ``T_i<-ego`` is the (noisy, hence known) relative pose. The ego's pose
is the reference and carries no error. A weak prior at the identity keeps a
CAV nobody measured at "uncorrected", which is what the sweep would have
emitted for it anyway.

Solved by Gauss-Newton with autograd Jacobians: three unknowns per CAV, a
handful of measurements, so a dense normal-equation solve is the right tool.
Measurement weights are the fits' own precision matrices (``PoseEstimate.
offset_precision``, ordered ``(t_x, t_y, psi)`` as in ``alignformer.abstain``);
``None`` means unit weight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import torch
from torch import Tensor

PARAMS_PER_NODE = 3
DEFAULT_ITERATIONS = 10
# Precision of the prior pulling every CAV toward the identity. It exists to
# pin an unobserved node, and it has to be far weaker than it looks: a cross
# measurement through a 30 m lever arm determines the combination "heading
# plus 30 m times translation" at only ~1/900 of unit information, so a prior
# of 1e-6 already moved a measured node by 0.4 mm. At 1e-10 the pull is
# below a micrometre on any measured node and still well-posed in float64.
DEFAULT_PRIOR_PRECISION = 1e-10
CONVERGED_STEP = 1e-9


@dataclass(frozen=True)
class SE2:
    """One rigid motion of the plane: ``x -> R(psi) x + t``."""

    psi: Tensor  # scalar, radians
    t: Tensor  # (2,)


@dataclass(frozen=True)
class Measurement:
    """``observer`` estimated ``value`` for ``target``; ``precision`` is ``(3, 3)`` or ``None``."""

    observer: str
    target: str
    value: SE2
    precision: Optional[Tensor]


def wrap(angle: Tensor) -> Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _rotation(psi: Tensor) -> Tensor:
    cos, sin = torch.cos(psi), torch.sin(psi)
    return torch.stack([torch.stack([cos, -sin]), torch.stack([sin, cos])])


def compose(a: SE2, b: SE2) -> SE2:
    """``a . b``: apply ``b`` first, then ``a``."""
    return SE2(psi=wrap(a.psi + b.psi), t=_rotation(a.psi) @ b.t + a.t)


def inverse(a: SE2) -> SE2:
    return SE2(psi=-a.psi, t=-(_rotation(-a.psi) @ a.t))


def apply_se2(a: SE2, points: Tensor) -> Tensor:
    """``(N, 2)`` points moved by ``a``."""
    return points @ _rotation(a.psi).T + a.t


def _unpack(x: Tensor, index: int) -> SE2:
    base = PARAMS_PER_NODE * index
    return SE2(psi=x[base + 2], t=x[base : base + 2])


def _predict(x: Tensor, measurement: Measurement, ego: str, slot: Mapping[str, int], frames: Mapping[str, SE2]) -> SE2:
    target = _unpack(x, slot[measurement.target])
    if measurement.observer == ego:
        return target
    observer = _unpack(x, slot[measurement.observer])
    t_i_ego = frames[measurement.observer]
    return compose(compose(compose(t_i_ego, target), inverse(observer)), inverse(t_i_ego))


def _weighted_residual(predicted: SE2, measurement: Measurement) -> Tensor:
    raw = torch.cat([predicted.t - measurement.value.t, wrap(predicted.psi - measurement.value.psi).reshape(1)])
    if measurement.precision is None:
        return raw
    # L^T r with P = L L^T, so that |L^T r|^2 = r^T P r.
    return torch.linalg.cholesky(measurement.precision).T @ raw


def _residuals(x, measurements, ego, slot, frames, prior_precision) -> Tensor:
    rows = [_weighted_residual(_predict(x, m, ego, slot, frames), m) for m in measurements]
    rows.append(math.sqrt(prior_precision) * x)
    return torch.cat(rows)


def _initial(cavs: Sequence[str], measurements: Sequence[Measurement], ego: str) -> Tensor:
    x = torch.zeros(PARAMS_PER_NODE * len(cavs), dtype=torch.float64)
    for index, cav in enumerate(cavs):
        direct = [m for m in measurements if m.observer == ego and m.target == cav]
        if direct:
            x[PARAMS_PER_NODE * index : PARAMS_PER_NODE * index + 2] = direct[0].value.t.to(torch.float64)
            x[PARAMS_PER_NODE * index + 2] = direct[0].value.psi.to(torch.float64)
    return x


def _validate(ego: str, cavs: Sequence[str], frames: Mapping[str, SE2], measurements: Sequence[Measurement]) -> None:
    known = set(cavs)
    for m in measurements:
        if m.target == ego or m.target not in known:
            raise ValueError(f"measurement target {m.target!r} is not a CAV of this frame")
        if m.observer != ego and m.observer not in known:
            raise ValueError(f"measurement observer {m.observer!r} is neither the ego nor a CAV")
        if m.observer == m.target:
            raise ValueError(f"an agent cannot measure itself ({m.observer!r})")
        if m.observer != ego and m.observer not in frames:
            raise ValueError(f"no relative pose T_{m.observer}<-ego for a cross measurement")


def _to_double(m: Measurement) -> Measurement:
    value = SE2(psi=m.value.psi.to(torch.float64), t=m.value.t.to(torch.float64))
    precision = None if m.precision is None else m.precision.to(torch.float64)
    return Measurement(m.observer, m.target, value, precision)


def solve_pose_graph(
    ego: str,
    cavs: Sequence[str],
    frames: Mapping[str, SE2],
    measurements: Sequence[Measurement],
    *,
    iterations: int = DEFAULT_ITERATIONS,
    prior_precision: float = DEFAULT_PRIOR_PRECISION,
) -> Dict[str, SE2]:
    """Ego-frame correction per CAV that best explains every measurement.

    ``frames[i]`` is ``T_i<-ego`` for each CAV (the noisy relative pose the
    sweep projects with). Returns one :class:`SE2` per CAV, in the ego frame,
    in the dtype of the first measurement (float64 internally).
    """
    _validate(ego, cavs, frames, measurements)
    slot = {cav: index for index, cav in enumerate(cavs)}
    doubles = [_to_double(m) for m in measurements]
    frames64 = {k: SE2(v.psi.to(torch.float64), v.t.to(torch.float64)) for k, v in frames.items()}
    x = _initial(cavs, doubles, ego)
    residual = lambda params: _residuals(params, doubles, ego, slot, frames64, prior_precision)  # noqa: E731
    for _ in range(iterations):
        jacobian = torch.autograd.functional.jacobian(residual, x)
        r = residual(x)
        step = torch.linalg.lstsq(jacobian, -r.unsqueeze(-1)).solution.squeeze(-1)
        x = x + step
        if float(step.norm()) < CONVERGED_STEP:
            break
    dtype = measurements[0].value.t.dtype if measurements else torch.float32
    return {
        cav: SE2(psi=wrap(_unpack(x, index).psi).to(dtype), t=_unpack(x, index).t.to(dtype))
        for cav, index in slot.items()
    }


__all__ = [
    "DEFAULT_ITERATIONS",
    "DEFAULT_PRIOR_PRECISION",
    "Measurement",
    "SE2",
    "apply_se2",
    "compose",
    "inverse",
    "solve_pose_graph",
    "wrap",
]
