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
class GraphSolution:
    """Per-CAV ego-frame correction and its marginal precision ``(3, 3)``, ordered ``(t_x, t_y, psi)``."""

    corrections: Dict[str, SE2]
    precision: Dict[str, Tensor]
    measured: Dict[str, bool]  # whether any measurement touched the node


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
    """The solve runs on the CPU in float64: a few unknowns, no reason for a device."""
    value = SE2(psi=m.value.psi.detach().cpu().to(torch.float64), t=m.value.t.detach().cpu().to(torch.float64))
    precision = None if m.precision is None else m.precision.detach().cpu().to(torch.float64)
    return Measurement(m.observer, m.target, value, precision)


def _marginal_precisions(jacobian: Tensor, cavs: Sequence[str]) -> Dict[str, Tensor]:
    """Per-node ``(3, 3)`` precision: the inverse of the node's block of the posterior covariance."""
    information = jacobian.T @ jacobian
    covariance = torch.linalg.pinv(information)
    out = {}
    for index, cav in enumerate(cavs):
        block = covariance[PARAMS_PER_NODE * index : PARAMS_PER_NODE * (index + 1),
                           PARAMS_PER_NODE * index : PARAMS_PER_NODE * (index + 1)]
        out[cav] = torch.linalg.pinv(block)
    return out


def solve_pose_graph_with_precision(
    ego: str,
    cavs: Sequence[str],
    frames: Mapping[str, SE2],
    measurements: Sequence[Measurement],
    *,
    iterations: int = DEFAULT_ITERATIONS,
    prior_precision: float = DEFAULT_PRIOR_PRECISION,
) -> GraphSolution:
    """:func:`solve_pose_graph` plus each node's marginal precision and whether it was measured."""
    _validate(ego, cavs, frames, measurements)
    if not cavs:
        return GraphSolution({}, {}, {})
    slot = {cav: index for index, cav in enumerate(cavs)}
    doubles = [_to_double(m) for m in measurements]
    frames64 = {
        k: SE2(v.psi.detach().cpu().to(torch.float64), v.t.detach().cpu().to(torch.float64))
        for k, v in frames.items()
    }
    x = _initial(cavs, doubles, ego)
    residual = lambda params: _residuals(params, doubles, ego, slot, frames64, prior_precision)  # noqa: E731
    # The sweep runs under no_grad; the Jacobian needs the graph back on.
    with torch.enable_grad():
        for _ in range(iterations):
            jacobian = torch.autograd.functional.jacobian(residual, x)
            r = residual(x)
            step = torch.linalg.lstsq(jacobian, -r.unsqueeze(-1)).solution.squeeze(-1)
            x = x + step
            if float(step.norm()) < CONVERGED_STEP:
                break
        jacobian = torch.autograd.functional.jacobian(residual, x)
    dtype = measurements[0].value.t.dtype if measurements else torch.float32
    touched = {m.observer for m in measurements} | {m.target for m in measurements}
    return GraphSolution(
        corrections={
            cav: SE2(psi=wrap(_unpack(x, index).psi).to(dtype), t=_unpack(x, index).t.to(dtype))
            for cav, index in slot.items()
        },
        precision={cav: p.to(dtype) for cav, p in _marginal_precisions(jacobian, cavs).items()},
        measured={cav: cav in touched for cav in cavs},
    )


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
    return solve_pose_graph_with_precision(
        ego, cavs, frames, measurements, iterations=iterations, prior_precision=prior_precision
    ).corrections


# ---------------------------------------------------------------------------
# From a frame's estimates to graph-corrected estimates
# ---------------------------------------------------------------------------

GRAPH_SUFFIX = "_graph"
# ``fill``: the ego's answered estimates stand; the graph fills only the CAVs
# the ego abstained on, each gated by the decision rule on the graph's own
# statistic. ``joint``: every CAV takes the joint solution.
GRAPH_FILL = "fill"
GRAPH_JOINT = "joint"
GRAPH_MODES = (GRAPH_FILL, GRAPH_JOINT)
# The graph's covariance comes from the fits' own estimated precisions, so
# the Wald reference is the chi-square limit of the F table (its largest dof).
GRAPH_STATISTIC_DOF = 4000.0


def graph_name(arm: str) -> str:
    """The graph-corrected arm built on ``arm``."""
    return arm + GRAPH_SUFFIX


def se2_from_transform(matrix: Tensor) -> SE2:
    """The SE(2) a 4x4 OpenCOOD transform implies (yaw from the rotation block)."""
    return SE2(psi=torch.atan2(matrix[1, 0], matrix[0, 0]), t=matrix[:2, 3])


def _answered(estimate) -> bool:
    """An estimate that emitted a correction; a zero one is a decision to abstain, not a measurement."""
    return bool((estimate.t.abs().sum() + estimate.psi.abs().sum()).item() > 0.0)


def _measurement(observer: str, target: str, estimate) -> Measurement:
    precision = estimate.offset_precision
    return Measurement(
        observer,
        target,
        SE2(psi=estimate.psi[0], t=estimate.t[0]),
        None if precision is None else precision[0],
    )


def _graph_candidate(base, solution: GraphSolution, cav: str, statistic: bool):
    """The ego estimate for ``cav`` with the graph's ``(psi, t)``, and, if asked, the graph's own Wald inputs."""
    from dataclasses import replace

    correction = solution.corrections[cav]
    psi = correction.psi.reshape(1).to(base.psi)
    t = correction.t.reshape(1, 2).to(base.t)
    if not statistic:
        return replace(base, psi=psi, t=t)
    precision = solution.precision[cav].to(base.t)
    theta = torch.cat([t[0], psi])
    return replace(
        base,
        psi=psi,
        t=t,
        offset_precision=precision.unsqueeze(0),
        offset_statistic=(theta @ precision @ theta).reshape(1),
        offset_dof=torch.full((1,), GRAPH_STATISTIC_DOF, dtype=base.psi.dtype, device=base.psi.device),
    )


def graph_estimates(
    ego_estimates: Mapping[str, object],
    cross_estimates: Mapping[tuple, object],
    frames: Mapping[str, SE2],
    *,
    ego: str = "ego",
    mode: str = GRAPH_JOINT,
    decision=None,
) -> Dict[str, object]:
    """One graph-corrected :class:`PoseEstimate` per CAV, from the frame's pairwise estimates.

    ``ego_estimates[j]`` is the ego's estimate for CAV ``j`` (batch of one);
    ``cross_estimates[(i, j)]`` is CAV ``i``'s estimate for CAV ``j`` in
    ``i``'s frame; ``frames[i]`` is ``T_i<-ego``. Only estimates that emitted
    a correction become measurements.

    ``joint``: every CAV takes the joint solution, other fields ride along.
    ``fill``: an answered ego estimate is returned as the very same object;
    a CAV the ego abstained on takes the graph's solution when some
    measurement touched it, carrying the graph's own precision and Wald
    statistic, and ``decision`` (an ``AbstentionConfig``) is applied to it;
    a CAV nothing measured is returned as the ego left it.
    """
    if mode not in GRAPH_MODES:
        raise ValueError(f"mode must be one of {GRAPH_MODES}, got {mode!r}")
    cavs = list(ego_estimates)
    measurements: List[Measurement] = [
        _measurement(ego, cav, estimate) for cav, estimate in ego_estimates.items() if _answered(estimate)
    ]
    measurements.extend(
        _measurement(i, j, estimate) for (i, j), estimate in cross_estimates.items() if _answered(estimate)
    )
    solution = solve_pose_graph_with_precision(ego, cavs, frames, measurements)
    if mode == GRAPH_JOINT:
        return {cav: _graph_candidate(ego_estimates[cav], solution, cav, statistic=False) for cav in cavs}
    return {cav: _fill_one(ego_estimates[cav], solution, cav, decision) for cav in cavs}


def _fill_one(base, solution: GraphSolution, cav: str, decision):
    if _answered(base) or not solution.measured.get(cav, False):
        return base
    candidate = _graph_candidate(base, solution, cav, statistic=True)
    if decision is None:
        return candidate
    from embedding_aware_belt_fusion.alignformer.abstain import decide

    return decide(candidate, decision)


__all__ = [
    "GRAPH_FILL",
    "GRAPH_JOINT",
    "GRAPH_MODES",
    "GRAPH_SUFFIX",
    "GraphSolution",
    "solve_pose_graph_with_precision",
    "graph_estimates",
    "graph_name",
    "se2_from_transform",
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
