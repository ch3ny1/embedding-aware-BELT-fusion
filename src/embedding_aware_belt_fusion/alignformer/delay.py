"""Transmission delay as a sweep axis, beside localization error.

Delay is not more localization error
------------------------------------

Localization error displaces the CAV's whole box set rigidly, so a single
SE(2) undoes it exactly; that is the estimand head B was built for. Delay is a
different shape. The CAV transmits what it saw at ``t - d`` together with its
own pose at ``t - d``, so **static** objects arrive in the right world
position and want no correction at all, while **moving** objects arrive
displaced by their own velocity times the delay. No rigid transform repairs
that, and a solver that tries is dragged off by whichever objects moved.

Which is what makes it worth measuring here rather than assuming. Under delay
the correspondences split into a correct majority and a velocity-corrupted
minority -- exactly the shape robust weighting exists for. Huber IRLS
down-weights them and the per-pair rule declines when what remains cannot
carry a correction. FreeAlign's relative-distance graph has no equivalent: one
moving object changes its distance to every other node at once, so the
corruption is spread across the graph instead of isolated in a few residuals.

Why this module exists at all
-----------------------------

``baselines.load_baseline`` strips ``wild_setting`` wholesale before building
a dataset, so that OpenCOOD's own constant-seed, z-perturbing localization
noise cannot run beside this sweep's controlled perturbation. That is correct
and stays. It also threw out the asynchrony model, which lives in the same
block and is independently useful. This module is the narrow re-entry: delay
on, OpenCOOD's localization noise firmly off, and ``sim`` mode so the delay is
a constant rather than a ``np.random.uniform`` draw that would consume from
the same global stream this sweep seeds and make the paired-seed comparison
unpaired.

The quantization is load-bearing
--------------------------------

OPV2V is 10 Hz and OpenCOOD floors delay to whole frames
(``time_delay // 100`` in ``basedataset.time_delay_calculation``). So 150 ms
and 100 ms are the SAME experiment, and a sweep reporting them as two points
would be reporting rounding. The axis is specified in **frames** for that
reason; :func:`frames_for_ms` exists to tell a caller who thinks in
milliseconds what they will actually get.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# OPV2V and V2X-Set are 10 Hz, and OpenCOOD hard-codes that divisor.
DELAY_FRAME_MS = 100


def frames_for_ms(milliseconds: int) -> int:
    """Whole frames OpenCOOD would actually apply for a delay in ms.

    Floor, not round, because that is what ``time_delay // 100`` does. 150 ms
    is one frame and 99 ms is none.
    """
    if milliseconds < 0:
        raise ValueError(f"delay must be non-negative, got {milliseconds}")
    return int(milliseconds) // DELAY_FRAME_MS


def delay_wild_setting(frames: int) -> Optional[Dict[str, Any]]:
    """OpenCOOD ``wild_setting`` for a delay of ``frames``, or ``None`` for zero.

    ``None`` rather than a neutral block on purpose: zero delay has to mean the
    dataset is constructed exactly as it was before this axis existed. A
    present-but-neutral ``wild_setting`` would still take OpenCOOD's async
    branch and would still have to be trusted to do nothing.

    Every key ``basedataset`` reads without a default is present, so a missing
    one is an error here rather than a ``KeyError`` halfway through dataset
    construction. A fresh dict is returned each call because OpenCOOD keeps a
    reference to what it is handed.
    """
    if frames < 0:
        raise ValueError(f"delay must be non-negative, got {frames} frames")
    if frames == 0:
        return None
    return {
        "seed": 0,
        "async": True,
        # Constant, never a draw. See the module docstring.
        "async_mode": "sim",
        "async_overhead": int(frames) * DELAY_FRAME_MS,
        # The reason wild_setting is stripped everywhere else. This sweep
        # applies its own localization perturbation; OpenCOOD's must not also.
        "loc_err": False,
        "xyz_std": 0.0,
        "ryp_std": 0.0,
        # Only read in 'real' mode, which this never selects; present so the
        # block is complete rather than relying on OpenCOOD's defaults.
        "data_size": 0,
        "transmission_speed": 1,
        "backbone_delay": 0,
    }


__all__ = ["DELAY_FRAME_MS", "delay_wild_setting", "frames_for_ms"]
