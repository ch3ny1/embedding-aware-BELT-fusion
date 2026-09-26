"""Choose the robust solve's psi, iteration count and mass guard on VALIDATION.

Task 22 pre-registers three choices -- the influence function, ``n_irls``, and
the evidence threshold below which the loop is suppressed -- and requires all
three to be made on the scenario-disjoint 15% validation slice of ``train/``,
never on ``OPV2V/test``. A full ``--metric noisy_ap`` sweep costs ~25 minutes
per configuration on validation, so a thirty-point grid is run here on pose
error instead and only the selected configuration goes to the AP sweep.

Two things beyond pose error are reported, and they are the reason this script
exists rather than a one-line grid:

- **engagement.** A guard set too high disables the loop everywhere, which
  reproduces the deployed arm exactly and would show up as a clean null rather
  than as a broken experiment. ``engaged_fraction`` is the share of pairs the
  guard let the loop run on and ``moved_fraction`` the share whose pose
  actually changed by more than a micrometre. A grid row with
  ``moved_fraction`` near zero is measuring nothing, whatever its MAE says.
- **the sparse slice.** The loop's known risk is trimming on thin evidence, and
  the pairs sharing one or two objects are exactly where AlignFormer beats a
  distance-graph competitor. They are scored as their own slice so a regression
  there cannot be averaged away by the 90% dense majority.

The pairs come from ``calibrate_freealign._collect_pairs`` rather than from a
second copy of it, so this diagnostic and the FreeAlign one are measured on
literally the same detections, truncation and noise draws.

Usage::

    python scripts/calibrate_robust_solve.py \\
        --config configs/alignformer_detector_r140.yaml \\
        --alignformer-config configs/alignformer_r140.yaml \\
        --split /media/chenyi/basement2/cache/opv2v_splits/val \\
        --checkpoint outputs/alignformer/r140/stage2_B_ivw_scalar/best.pth \\
        --shrinkage outputs/alignformer/r140/shrinkage_ivw_scalar_calibration_result.json \\
        --stride 8 --sigma 0 1.0 \\
        --output outputs/alignformer/r140/robust_solve_calibration_result.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibrate_freealign import _collect_pairs

from embedding_aware_belt_fusion.alignformer.robust import (
    DISABLED,
    GEMAN_MCCLURE,
    HUBER,
    RobustSolveConfig,
)
from embedding_aware_belt_fusion.alignformer.splits import resolve_split
from embedding_aware_belt_fusion.alignformer.stage2 import is_fallback

# The three axes the brief pre-registers, and nothing else. ``min_evidence``
# spans "no guard at all" to "six effective correspondences" so that the guard's
# own effect is visible rather than assumed; 3.0 is the default and is where a
# relative-distance graph stops being degenerate.
_GRID = {
    "mode": (HUBER, GEMAN_MCCLURE),
    "iterations": (1, 2, 3),
    "min_evidence": (0.0, 2.0, 3.0, 4.0, 6.0),
}

# Below this the two arms' corrections are the same number to machine
# precision, so the loop did not move the pose.
_MOVED_EPSILON_M = 1e-6
# FreeAlign's own headline matching metric, carried here so the three
# diagnostics in this project report the same tail statistic.
_ERROR_RATE_THRESHOLD_M = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--alignformer-config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True,
                        help="the VALIDATION split directory, unless "
                             "--measure-only (see its help)")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shrinkage", type=Path, default=None)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--sigma", type=float, nargs="+", default=[0.0, 1.0],
                        help="localization-noise levels the grid is scored at")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--measure-only", action="store_true",
        help="skip the grid and score ONE frozen configuration against the "
             "deployed arm on the same pairs. This is a MEASUREMENT, not a "
             "selection, so it is the only mode that may point at the test "
             "split: the sweep reports means and the brief also asks for "
             "medians, paired.",
    )
    parser.add_argument("--mode", default=HUBER,
                        help="--measure-only: the frozen psi function")
    parser.add_argument("--iterations", type=int, default=2,
                        help="--measure-only: the frozen iteration count")
    parser.add_argument("--min-evidence", type=float, default=3.0,
                        help="--measure-only: the frozen evidence guard")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _errors(estimates, pairs):
    """``(translation, yaw, answered)`` arrays over the pairs, in m / deg."""
    translation, yaw, answered = [], [], []
    for estimate, pair in zip(estimates, pairs):
        psi, t = float(estimate.psi[0]), estimate.t[0]
        translation.append(float(np.hypot(
            float(t[0]) - pair["true_t"][0], float(t[1]) - pair["true_t"][1]
        )))
        difference = psi - pair["true_psi"]
        yaw.append(abs(float(np.degrees(
            np.arctan2(np.sin(difference), np.cos(difference))
        ))))
        answered.append(not bool(is_fallback(estimate)[0]))
    return np.array(translation), np.array(yaw), np.array(answered)


def _summarize(translation, yaw, answered, engaged, moved) -> dict:
    def mean(values, mask):
        return float(values[mask].mean()) if mask.any() else None

    def median(values, mask):
        return float(np.median(values[mask])) if mask.any() else None

    return {
        "pairs": int(translation.shape[0]),
        "translation_mae_m": float(translation.mean()),
        "translation_median_m": float(np.median(translation)),
        "yaw_mae_deg": float(yaw.mean()),
        "yaw_median_deg": float(np.median(yaw)),
        "coverage": float(answered.mean()),
        "error_rate_over_3m": float((translation > _ERROR_RATE_THRESHOLD_M).mean()),
        "answered_translation_mae_m": mean(translation, answered),
        "answered_translation_median_m": median(translation, answered),
        "answered_yaw_median_deg": median(yaw, answered),
        # The honesty columns: a row that never engages is not a null result,
        # it is an experiment that did not run.
        "engaged_fraction": float(engaged.mean()),
        "moved_fraction": float(moved.mean()),
    }


def _estimator(args, device):
    """The deployed estimator as a callable of ``(batch, robust_config)``.

    One pass of the frozen embedding head per call, the deployed shrinkage
    applied exactly as ``noisy_fusion`` applies it, so a grid row differs from
    the deployed arm in the solve and in nothing else.
    """
    from embedding_aware_belt_fusion.alignformer.evaluate import _load_shrinkage
    from embedding_aware_belt_fusion.alignformer.shrinkage import shrink
    from embedding_aware_belt_fusion.alignformer.stage2 import load_stage2
    from embedding_aware_belt_fusion.alignformer.train import embed_batch

    modules, checkpoint = load_stage2(args.checkpoint, device)
    ablate = checkpoint["message_content"] == "boxes_only"
    calibration = _load_shrinkage(args)

    def score(batch, robust: RobustSolveConfig):
        enriched = embed_batch(modules["embedding"], batch, ablate=ablate)
        estimate = (
            modules["pose"](enriched) if not robust.enabled
            else modules["pose"](enriched, robust=robust)
        )
        return estimate if calibration is None else shrink(estimate, calibration)

    return score, checkpoint


def _score_configs(pairs, estimator, configs) -> list:
    """Each configuration plus the deployed baseline, on one set of pairs."""
    from calibrate_freealign import _batch

    batches = [_batch(pair) for pair in pairs]
    with torch.no_grad():
        baseline = [estimator(batch, DISABLED) for batch in batches]
    base_translation, base_yaw, base_answered = _errors(baseline, pairs)
    never = np.zeros(len(pairs), dtype=bool)

    slices = {
        "shared_1_2": np.array([1 <= p["shared"] <= 2 for p in pairs]),
        "shared_3plus": np.array([p["shared"] >= 3 for p in pairs]),
    }

    def rows(translation, yaw, answered, engaged, moved):
        overall = _summarize(translation, yaw, answered, engaged, moved)
        for name, mask in slices.items():
            overall[name] = (
                _summarize(
                    translation[mask], yaw[mask], answered[mask],
                    engaged[mask], moved[mask],
                )
                if mask.any() else None
            )
        return overall

    results = [{
        "config": DISABLED.to_dict(),
        "deployed_baseline": True,
        **rows(base_translation, base_yaw, base_answered, never, never),
    }]

    for config in configs:
        with torch.no_grad():
            estimates = [estimator(batch, config) for batch in batches]
        translation, yaw, answered = _errors(estimates, pairs)
        engaged = np.array([
            float(estimate.confidence[0]) >= config.min_evidence
            for estimate in estimates
        ])
        moved = np.array([
            float(
                (estimate.t[0] - reference.t[0]).abs().max()
                + (estimate.psi[0] - reference.psi[0]).abs()
            ) > _MOVED_EPSILON_M
            for estimate, reference in zip(estimates, baseline)
        ])
        row = {
            "config": config.to_dict(),
            "deployed_baseline": False,
            "delta_translation_mae_m": float(
                translation.mean() - base_translation.mean()
            ),
            "delta_translation_median_m": float(
                np.median(translation) - np.median(base_translation)
            ),
            **rows(translation, yaw, answered, engaged, moved),
        }
        results.append(row)
        print(
            f"  {config.mode:<14} n={config.iterations} "
            f"guard={config.min_evidence:<4} -> "
            f"med {row['translation_median_m']:.4f} m "
            f"(base {np.median(base_translation):.4f})  "
            f"mean {row['translation_mae_m']:.4f} m  "
            f"engaged {row['engaged_fraction']:.3f}  "
            f"moved {row['moved_fraction']:.3f}",
            flush=True,
        )
    return results


def _grid_configs() -> list:
    """Every point of the pre-registered grid, in a fixed order."""
    names = list(_GRID)
    return [
        RobustSolveConfig(**dict(zip(names, values)))
        for values in itertools.product(*(_GRID[name] for name in names))
    ]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    estimator, checkpoint = _estimator(args, device)
    configs = (
        [RobustSolveConfig(
            mode=args.mode,
            iterations=args.iterations,
            min_evidence=args.min_evidence,
        )]
        if args.measure_only else _grid_configs()
    )

    by_sigma = {}
    for sigma in args.sigma:
        pairs = _collect_pairs(
            SimpleNamespace(
                config=args.config,
                alignformer_config=args.alignformer_config,
                split=resolve_split(args.split, allow_test=args.measure_only),
                stride=args.stride,
                sigma=sigma,
                seed=args.seed,
                device=args.device,
            ),
            device,
        )
        print(f"\nsigma = {sigma} m, {len(pairs)} pairs from {args.split}", flush=True)
        by_sigma[f"sigma_{sigma:g}m"] = {
            "pairs": len(pairs),
            "grid": _score_configs(pairs, estimator, configs),
        }

    payload = {
        "method": "alignformer_robust_solve",
        "metric": (
            "robust_solve_measurement" if args.measure_only
            else "robust_solve_calibration"
        ),
        "note": (
            "A MEASUREMENT of one frozen configuration against the deployed "
            "arm on the same pairs; nothing is selected here, which is why it "
            "may point at the test split."
            if args.measure_only else
            "Selection of the IRLS loop's psi, iteration count and evidence "
            "guard on the scenario-disjoint validation slice only. Never test."
        ),
        "measure_only": args.measure_only,
        "split": str(args.split),
        "stride": args.stride,
        "seed": args.seed,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_head": checkpoint.get("head"),
        "checkpoint_variance_weighting": checkpoint.get("variance_weighting"),
        "shrinkage": None if args.shrinkage is None else str(args.shrinkage),
        "grid_axes": (
            None if args.measure_only
            else {name: list(values) for name, values in _GRID.items()}
        ),
        "configurations": [config.to_dict() for config in configs],
        "by_sigma": by_sigma,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
