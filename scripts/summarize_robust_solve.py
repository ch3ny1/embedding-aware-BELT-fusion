"""Apply task 22's PRE-REGISTERED decision rule to the robust-solve sweeps.

The rule was fixed in ``task-22-robust-solve-brief.md`` before any code ran and
is evaluated here rather than re-derived in prose. On VALIDATION only, the IRLS
arm replaces the deployed arm if and only if all three hold:

1. mean AP@0.7 over the eight sigmas improves by **>= 0.005**;
2. AP@0.7 improves at **at least 6 of 8** sigmas;
3. the sparse slice (1-2 shared objects) AP@0.7 does **not** regress by more
   than 0.01.

The brief does not say whether (3) is read as a mean over sigmas or as a worst
case, so **both are computed**; the verdict uses the mean, to match the mean
form of (1), and the worst sigma is reported beside it and also verdicted, so a
configuration that buys its average by destroying one sigma of the sparse slice
cannot hide. Anything short of all three is a NULL and is reported as one.

Two honesty checks run alongside the rule, because the ways this experiment
could produce a spurious win are known in advance:

- ``deployed_agreement``: the ``uncorrected``, ``alignformer``, ``freealign``
  and ``oracle`` rows of these runs must reproduce the published FreeAlign run
  bit for bit. If they do not, the new arm changed something else and no
  comparison in this file means anything.
- ``engagement``: a guard set too high disables the loop everywhere, which
  would reproduce the deployed arm exactly and read as a clean null. The
  fraction of pairs the loop was allowed to run on comes from the calibration
  file and is printed with the verdict.

Usage::

    python scripts/summarize_robust_solve.py \\
        --validation outputs/alignformer/r140/robust_solve_val_result.json \\
        --test outputs/alignformer/r140/robust_solve_test_result.json \\
        --deployed outputs/alignformer/r140/freealign_test_result.json \\
        --deployed-validation outputs/alignformer/r140/freealign_val_result.json \\
        --calibration outputs/alignformer/r140/robust_solve_calibration_result.json \\
        --output outputs/alignformer/r140/robust_solve_result.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

# The pre-registered thresholds, in the brief's own words.
MEAN_IMPROVEMENT_THRESHOLD = 0.005
MIN_IMPROVING_SIGMAS = 6
SPARSE_REGRESSION_TOLERANCE = 0.01
DECISION_METRIC = "ap_70"

ALIGNFORMER = "alignformer"
ALIGNFORMER_IRLS = "alignformer_irls"
FREEALIGN = "freealign"
UNCORRECTED = "uncorrected"
ORACLE = "oracle"
AP_KEYS = ("ap_30", "ap_50", "ap_70")
SPARSE_BUCKET = "shared_1_2"
# Every arm that must be untouched by the new condition, i.e. everything the
# published FreeAlign run already carries.
UNCHANGED_CONDITIONS = (UNCORRECTED, ALIGNFORMER, FREEALIGN)


def _sigma_key(condition: str, sigma: float) -> str:
    return condition if condition == ORACLE else f"{condition}_sigma_{sigma:g}m"


def _ap(result: Dict, condition: str, sigma: float, metric: str = DECISION_METRIC):
    key = _sigma_key(condition, sigma)
    entry = result["ap"].get(key)
    return None if entry is None else entry[metric]["global_sorted"]


def _sparse_ap(result: Dict, bucket: str, condition: str, sigma: float):
    slices = result.get("ap_by_shared_objects", {})
    entry = slices.get(bucket, {}).get("conditions", {}).get(_sigma_key(condition, sigma))
    return None if entry is None else entry[DECISION_METRIC]["global_sorted"]


def _pose(result: Dict, condition: str, sigma: float) -> Dict:
    return result["pose_by_condition"][condition][f"sigma_{sigma:g}m"]


def _sigmas(result: Dict) -> List[float]:
    return [float(value) for value in result["sweep_sigmas_m"]]


def ap_rows(result: Dict) -> List[Dict]:
    """One row per sigma: every condition at all three IoU thresholds."""
    present = [
        condition
        for condition in (UNCORRECTED, ALIGNFORMER, ALIGNFORMER_IRLS, FREEALIGN)
        if _ap(result, condition, _sigmas(result)[0]) is not None
    ]
    rows = []
    for sigma in _sigmas(result):
        row: Dict[str, object] = {"sigma_m": sigma}
        for condition in present:
            row[condition] = {
                key: _ap(result, condition, sigma, key) for key in AP_KEYS
            }
        if ALIGNFORMER_IRLS in present:
            row["delta_ap_70_irls_minus_deployed"] = (
                row[ALIGNFORMER_IRLS]["ap_70"] - row[ALIGNFORMER]["ap_70"]
            )
            if FREEALIGN in present:
                row["delta_ap_70_irls_minus_freealign"] = (
                    row[ALIGNFORMER_IRLS]["ap_70"] - row[FREEALIGN]["ap_70"]
                )
        rows.append(row)
    return rows


def slice_rows(result: Dict) -> Dict[str, List[Dict]]:
    """AP@0.7 per sigma inside each shared-object slice, and the IRLS delta."""
    buckets = result.get("ap_by_shared_objects", {})
    out: Dict[str, List[Dict]] = {}
    for bucket in buckets:
        rows = []
        for sigma in _sigmas(result):
            row: Dict[str, object] = {"sigma_m": sigma}
            for condition in (UNCORRECTED, ALIGNFORMER, ALIGNFORMER_IRLS, FREEALIGN):
                value = _sparse_ap(result, bucket, condition, sigma)
                if value is not None:
                    row[condition] = value
            if ALIGNFORMER_IRLS in row and ALIGNFORMER in row:
                row["delta_ap_70_irls_minus_deployed"] = (
                    row[ALIGNFORMER_IRLS] - row[ALIGNFORMER]
                )
            rows.append(row)
        out[bucket] = rows
    return out


def pose_rows(result: Dict) -> List[Dict]:
    """Per sigma, each arm's translation/yaw MAE **and median-proxy** statistics.

    The sweep records means and coverage; the medians the brief also asks for
    come from the paired stride-8 diagnostic, not from here, and the keys that
    do exist are carried through unchanged.
    """
    conditions = [
        name for name in (ALIGNFORMER, ALIGNFORMER_IRLS, FREEALIGN)
        if name in result.get("pose_by_condition", {})
    ]
    rows = []
    for sigma in _sigmas(result):
        row: Dict[str, object] = {"sigma_m": sigma}
        for condition in conditions:
            pose = _pose(result, condition, sigma)
            row[condition] = {
                "translation_mae_m": pose["translation_mae_m"],
                "answered_translation_mae_m": pose["answered_translation_mae_m"],
                "yaw_mae_deg": pose["yaw_mae_deg"],
                "coverage": pose["coverage"],
                "shared_1_2_translation_mae_m": pose.get(
                    "shared_1_2_translation_mae_m"
                ),
                "shared_3plus_translation_mae_m": pose.get(
                    "shared_3plus_translation_mae_m"
                ),
            }
        rows.append(row)
    return rows


def decide(validation: Dict) -> Dict:
    """Evaluate the pre-registered rule. No judgement here, only arithmetic."""
    sigmas = _sigmas(validation)
    per_sigma = {
        f"sigma_{sigma:g}m": (
            _ap(validation, ALIGNFORMER_IRLS, sigma) - _ap(validation, ALIGNFORMER, sigma)
        )
        for sigma in sigmas
    }
    deltas = list(per_sigma.values())
    mean = sum(deltas) / len(deltas)
    improved = sum(1 for value in deltas if value > 0)

    sparse_per_sigma = {}
    for sigma in sigmas:
        irls = _sparse_ap(validation, SPARSE_BUCKET, ALIGNFORMER_IRLS, sigma)
        deployed = _sparse_ap(validation, SPARSE_BUCKET, ALIGNFORMER, sigma)
        if irls is not None and deployed is not None:
            sparse_per_sigma[f"sigma_{sigma:g}m"] = irls - deployed
    sparse_values = list(sparse_per_sigma.values())
    sparse_mean = sum(sparse_values) / len(sparse_values) if sparse_values else None
    sparse_worst = min(sparse_values) if sparse_values else None

    criteria = {
        "mean_improvement_at_least_0.005": mean >= MEAN_IMPROVEMENT_THRESHOLD,
        "improves_at_6_of_8_sigmas": improved >= MIN_IMPROVING_SIGMAS,
        "sparse_slice_mean_not_worse_than_0.01": (
            sparse_mean is None or sparse_mean >= -SPARSE_REGRESSION_TOLERANCE
        ),
    }
    # Reported beside the rule, never folded into it: the brief's third
    # criterion does not say whether it is a mean or a worst case.
    sparse_worst_ok = (
        sparse_worst is None or sparse_worst >= -SPARSE_REGRESSION_TOLERANCE
    )
    passed = all(criteria.values())

    return {
        "rule": (
            "on VALIDATION only: mean AP@0.7 (global-sorted) over the eight "
            "sigmas improves by >= 0.005 AND improves at >= 6 of 8 sigmas AND "
            "the 1-2 shared-object slice does not regress by more than 0.01"
        ),
        "split": validation["split"],
        "metric": DECISION_METRIC,
        "sigmas": sigmas,
        "delta_per_sigma": per_sigma,
        "mean_delta": mean,
        "min_delta": min(deltas),
        "max_delta": max(deltas),
        "sigmas_improved": improved,
        "sparse_slice_delta_per_sigma": sparse_per_sigma,
        "sparse_slice_mean_delta": sparse_mean,
        "sparse_slice_worst_delta": sparse_worst,
        "sparse_slice_worst_within_tolerance": sparse_worst_ok,
        "criteria": criteria,
        "verdict": "replace_deployed_arm" if passed else "null",
    }


def deployed_agreement(result: Dict, reference: Optional[Dict]) -> Optional[Dict]:
    """Every untouched arm, cell by cell, against its already-published run.

    Adding a condition must not move the deployed ``alignformer`` arm, the
    ``uncorrected`` floor, the ``oracle`` ceiling or the ``freealign``
    competitor by a digit: the noise draws key on ``(seed, sigma, frame,
    agent)`` alone, so the two runs saw identical perturbations and any
    difference is a regression rather than a re-roll. This is the check that
    makes "nothing else changed" a measurement.
    """
    if reference is None:
        return None
    differences: Dict[str, float] = {}
    for condition in UNCHANGED_CONDITIONS:
        for sigma in _sigmas(result):
            for metric in AP_KEYS:
                mine = _ap(result, condition, sigma, metric)
                theirs = _ap(reference, condition, sigma, metric)
                if mine is None or theirs is None:
                    continue
                differences[f"{_sigma_key(condition, sigma)}:{metric}"] = mine - theirs
    for metric in AP_KEYS:
        mine = _ap(result, ORACLE, 0.0, metric)
        theirs = _ap(reference, ORACLE, 0.0, metric)
        if mine is not None and theirs is not None:
            differences[f"oracle:{metric}"] = mine - theirs
    return {
        "cells_compared": len(differences),
        "max_abs_difference": max((abs(v) for v in differences.values()), default=None),
        "identical": all(value == 0.0 for value in differences.values()),
        "non_zero": {k: v for k, v in differences.items() if v != 0.0},
    }


def engagement(calibration: Optional[Dict], config: Optional[Dict]) -> Optional[Dict]:
    """The selected configuration's engagement rate on the validation grid.

    A row that never engages measures nothing, whatever its AP says, so this
    travels with the verdict rather than sitting in a separate file.
    """
    if calibration is None or config is None:
        return None
    found = {}
    for sigma_key, block in calibration["by_sigma"].items():
        for row in block["grid"]:
            if row["deployed_baseline"]:
                continue
            if all(
                row["config"][name] == config[name]
                for name in ("mode", "iterations", "min_evidence")
            ):
                found[sigma_key] = {
                    "engaged_fraction": row["engaged_fraction"],
                    "moved_fraction": row["moved_fraction"],
                    "pairs": row["pairs"],
                }
    return found or None


def _table(rows: List[Dict]) -> str:
    conditions = [
        name for name in (UNCORRECTED, ALIGNFORMER, ALIGNFORMER_IRLS, FREEALIGN)
        if name in rows[0]
    ]
    header = "| sigma (m) | " + " | ".join(conditions) + " | IRLS - deployed |"
    lines = [header, "|---" * (len(conditions) + 2) + "|"]
    for row in rows:
        cells = " | ".join(f"{row[name]['ap_70']:.4f}" for name in conditions)
        delta = row.get("delta_ap_70_irls_minus_deployed")
        lines.append(
            f"| {row['sigma_m']:g} | {cells} | "
            f"{'--' if delta is None else format(delta, '+.4f')} |"
        )
    return "\n".join(lines)


def _slice_table(rows: List[Dict]) -> str:
    conditions = [
        name for name in (UNCORRECTED, ALIGNFORMER, ALIGNFORMER_IRLS, FREEALIGN)
        if name in rows[0]
    ]
    lines = [
        "| sigma (m) | " + " | ".join(conditions) + " | IRLS - deployed |",
        "|---" * (len(conditions) + 2) + "|",
    ]
    for row in rows:
        cells = " | ".join(f"{row[name]:.4f}" for name in conditions)
        delta = row.get("delta_ap_70_irls_minus_deployed")
        lines.append(
            f"| {row['sigma_m']:g} | {cells} | "
            f"{'--' if delta is None else format(delta, '+.4f')} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--deployed", type=Path, default=None,
                        help="the published freealign TEST run, for bit-identity")
    parser.add_argument("--deployed-validation", type=Path, default=None,
                        help="the published freealign VALIDATION run, same check")
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    validation = json.loads(args.validation.read_text())
    test = json.loads(args.test.read_text())
    reference_test = json.loads(args.deployed.read_text()) if args.deployed else None
    reference_val = (
        json.loads(args.deployed_validation.read_text())
        if args.deployed_validation else None
    )
    calibration = (
        json.loads(args.calibration.read_text()) if args.calibration else None
    )

    config = test.get("robust_solve_config")
    summary = {
        "method": "alignformer_robust_solve",
        "metric": "noisy_ap",
        "task": 22,
        "robust_solve_config": config,
        "pose_checkpoint": validation["pose_checkpoint"],
        "pose_checkpoint_provenance": validation["pose_checkpoint_provenance"],
        "shrinkage": validation["shrinkage"],
        "sources": {
            "validation": str(args.validation),
            "test": str(args.test),
            "deployed_reference_test": str(args.deployed) if args.deployed else None,
            "deployed_reference_validation": (
                str(args.deployed_validation) if args.deployed_validation else None
            ),
            "calibration": str(args.calibration) if args.calibration else None,
        },
        "decision": decide(validation),
        "engagement_on_validation": engagement(calibration, config),
        "deployed_agreement_on_test": deployed_agreement(test, reference_test),
        "deployed_agreement_on_validation": deployed_agreement(
            validation, reference_val
        ),
        "validation": {
            "split": validation["split"],
            "frames": validation["frames"],
            "ap": ap_rows(validation),
            "ap_by_shared_objects": slice_rows(validation),
            "pairs_by_shared_object_bucket": validation.get(
                "pairs_by_shared_object_bucket"
            ),
            "pose": pose_rows(validation),
        },
        "test": {
            "split": test["split"],
            "frames": test["frames"],
            "ap": ap_rows(test),
            "ap_by_shared_objects": slice_rows(test),
            "pairs_by_shared_object_bucket": test.get("pairs_by_shared_object_bucket"),
            "pose": pose_rows(test),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"VALIDATION ({validation['frames']} frames)  AP@0.7 global-sorted")
    print(_table(summary["validation"]["ap"]))
    for bucket, rows in summary["validation"]["ap_by_shared_objects"].items():
        print(f"\nVALIDATION slice {bucket}")
        print(_slice_table(rows))
    print(f"\nTEST ({test['frames']} frames)  AP@0.7 global-sorted")
    print(_table(summary["test"]["ap"]))
    for bucket, rows in summary["test"]["ap_by_shared_objects"].items():
        print(f"\nTEST slice {bucket}")
        print(_slice_table(rows))

    decision = summary["decision"]
    print(f"\nDECISION (validation, pre-registered): {decision['verdict'].upper()}")
    print(
        f"  mean delta {decision['mean_delta']:+.4f} "
        f"(needs >= {MEAN_IMPROVEMENT_THRESHOLD})  "
        f"improved at {decision['sigmas_improved']}/8 "
        f"(needs >= {MIN_IMPROVING_SIGMAS})  "
        f"sparse mean {decision['sparse_slice_mean_delta']}  "
        f"sparse worst {decision['sparse_slice_worst_delta']}"
    )
    for name, value in decision["criteria"].items():
        print(f"    {name}: {value}")
    for label in ("deployed_agreement_on_validation", "deployed_agreement_on_test"):
        agreement = summary[label]
        if agreement is not None:
            print(
                f"  {label}: identical={agreement['identical']} "
                f"over {agreement['cells_compared']} cells, "
                f"max |diff| {agreement['max_abs_difference']}"
            )
    if summary["engagement_on_validation"]:
        print("  engagement of the selected configuration on validation:")
        for sigma_key, row in summary["engagement_on_validation"].items():
            print(
                f"    {sigma_key}: engaged {row['engaged_fraction']:.3f}, "
                f"moved {row['moved_fraction']:.3f} of {row['pairs']} pairs"
            )


if __name__ == "__main__":
    main()
