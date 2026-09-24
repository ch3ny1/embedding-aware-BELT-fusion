"""Apply task 21's PRE-REGISTERED decision rule to the oracle-correspondence sweeps.

The rule was fixed before the runs finished and is evaluated here rather than
re-derived in prose:

    delta = mean over sigma in [0.2, 2.0] of
            AP@0.7(oracle_match_ivw) - AP@0.7(alignformer),
    on VALIDATION only.

    delta > 0.02   -> headroom exists in matching
    delta < 0.01   -> matching is closed on OPV2V
    otherwise      -> ambiguous, report per-sigma and recommend nothing

Validation is the scenario-disjoint 15% slice of ``train/`` (split_seed 0),
materialized at ``/media/chenyi/basement2/cache/opv2v_splits/val``. The test
split is summarized too, for the paper's ceiling claim, and is never used to
decide anything.

The ``alignformer`` arm is taken from the SAME run as the oracle arms, so the
two share the frame's detections and its noise draw and the comparison is
paired. ``--deployed`` optionally cross-checks that run's test-split
``alignformer`` numbers against the already-published file, which must agree
exactly: nothing on that path changed.

Usage::

    python scripts/summarize_oracle_correspondence.py \\
        --validation outputs/alignformer/r140/oracle_correspondence_val_result.json \\
        --test outputs/alignformer/r140/oracle_correspondence_test_result.json \\
        --deployed outputs/alignformer/r140/p2_r140_ivw_scalar_noisy_ap_result.json \\
        --output outputs/alignformer/r140/oracle_correspondence_result.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

# The pre-registered rule. Sigma = 0 is excluded: there is no pose error to
# correct there, so it measures the estimator's noise floor rather than its
# matching (ruling R44).
DECISION_SIGMAS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0)
HEADROOM_THRESHOLD = 0.02
CLOSED_THRESHOLD = 0.01
DECISION_METRIC = "ap_70"

ALIGNFORMER = "alignformer"
ORACLE_MATCH_IVW = "oracle_match_ivw"
ORACLE_MATCH_UNIFORM = "oracle_match_uniform"
AP_KEYS = ("ap_30", "ap_50", "ap_70")


def _sigma_key(condition: str, sigma: float) -> str:
    return f"{condition}_sigma_{sigma:g}m"


def _ap(result: Dict, condition: str, sigma: float, metric: str) -> float:
    return result["ap"][_sigma_key(condition, sigma)][metric]["global_sorted"]


def _pose(result: Dict, condition: str, sigma: float) -> Dict:
    return result["pose_by_condition"][condition][f"sigma_{sigma:g}m"]


def _sigmas(result: Dict) -> List[float]:
    return [float(value) for value in result["sweep_sigmas_m"]]


def ap_rows(result: Dict) -> List[Dict]:
    """One row per sigma: every condition's AP at all three IoU thresholds."""
    conditions = [ALIGNFORMER, ORACLE_MATCH_UNIFORM, ORACLE_MATCH_IVW]
    rows = []
    for sigma in _sigmas(result):
        row: Dict[str, object] = {"sigma_m": sigma}
        row["uncorrected"] = {
            key: _ap(result, "uncorrected", sigma, key) for key in AP_KEYS
        }
        for condition in conditions:
            row[condition] = {key: _ap(result, condition, sigma, key) for key in AP_KEYS}
        row["delta_ap_70_ivw_minus_alignformer"] = (
            row[ORACLE_MATCH_IVW]["ap_70"] - row[ALIGNFORMER]["ap_70"]
        )
        row["delta_ap_70_uniform_minus_alignformer"] = (
            row[ORACLE_MATCH_UNIFORM]["ap_70"] - row[ALIGNFORMER]["ap_70"]
        )
        rows.append(row)
    return rows


def pose_rows(result: Dict) -> List[Dict]:
    """One row per sigma: each estimator's translation and yaw MAE.

    This is the decomposition the paper needs -- the oracle-correspondence row
    is the floor a perfect matcher would leave, and the difference is the part
    of the deployed error that better matching could reach at all.
    """
    rows = []
    for sigma in _sigmas(result):
        row: Dict[str, object] = {"sigma_m": sigma}
        for condition in (ALIGNFORMER, ORACLE_MATCH_UNIFORM, ORACLE_MATCH_IVW):
            pose = _pose(result, condition, sigma)
            row[condition] = {
                "translation_mae_m": pose["translation_mae_m"],
                "yaw_mae_deg": pose["yaw_mae_deg"],
                "fallback_fraction": pose["fallback_fraction"],
            }
        deployed, floor = row[ALIGNFORMER], row[ORACLE_MATCH_IVW]
        row["matching_residual"] = {
            "translation_m": deployed["translation_mae_m"] - floor["translation_mae_m"],
            "yaw_deg": deployed["yaw_mae_deg"] - floor["yaw_mae_deg"],
            "translation_fraction_of_deployed": (
                (deployed["translation_mae_m"] - floor["translation_mae_m"])
                / deployed["translation_mae_m"]
                if deployed["translation_mae_m"]
                else None
            ),
            "yaw_fraction_of_deployed": (
                (deployed["yaw_mae_deg"] - floor["yaw_mae_deg"]) / deployed["yaw_mae_deg"]
                if deployed["yaw_mae_deg"]
                else None
            ),
        }
        rows.append(row)
    return rows


def decide(validation: Dict) -> Dict:
    """Evaluate the pre-registered rule. No judgement here, only arithmetic."""
    per_sigma = {
        f"sigma_{sigma:g}m": (
            _ap(validation, ORACLE_MATCH_IVW, sigma, DECISION_METRIC)
            - _ap(validation, ALIGNFORMER, sigma, DECISION_METRIC)
        )
        for sigma in DECISION_SIGMAS
    }
    deltas = list(per_sigma.values())
    mean = sum(deltas) / len(deltas)

    if mean > HEADROOM_THRESHOLD:
        verdict = "headroom_exists"
    elif mean < CLOSED_THRESHOLD:
        verdict = "matching_is_closed"
    else:
        verdict = "ambiguous"

    return {
        "rule": (
            "mean over sigma in [0.2, 2.0] of AP@0.7(oracle_match_ivw) - "
            "AP@0.7(alignformer), global-sorted, on the scenario-disjoint "
            "validation slice; >0.02 headroom, <0.01 closed, else ambiguous"
        ),
        "split": validation["split"],
        "metric": DECISION_METRIC,
        "sigmas": list(DECISION_SIGMAS),
        "delta_per_sigma": per_sigma,
        "mean_delta": mean,
        "min_delta": min(deltas),
        "max_delta": max(deltas),
        "sign_flips_across_sigma": any(a > 0 for a in deltas) and any(a < 0 for a in deltas),
        "verdict": verdict,
    }


def deployed_agreement(test: Dict, deployed: Optional[Dict]) -> Optional[Dict]:
    """Cross-check the untouched ``alignformer`` arm against its published run.

    The refactor that made the oracle share head B's solver must not have moved
    the deployed numbers by so much as a digit; the noise draws are keyed on
    ``(seed, sigma, frame, agent)`` alone, so the two runs saw identical
    perturbations and any difference is a regression, not a re-roll.
    """
    if deployed is None:
        return None
    differences = {}
    for sigma in _sigmas(test):
        key = _sigma_key(ALIGNFORMER, sigma)
        if key not in deployed["ap"]:
            continue
        differences[key] = (
            _ap(test, ALIGNFORMER, sigma, DECISION_METRIC)
            - deployed["ap"][key][DECISION_METRIC]["global_sorted"]
        )
    return {
        "file": deployed.get("split"),
        "metric": DECISION_METRIC,
        "max_abs_difference": max((abs(v) for v in differences.values()), default=None),
        "difference_per_sigma": differences,
        "identical": all(value == 0.0 for value in differences.values()),
    }


def _table(rows: List[Dict]) -> str:
    lines = [
        "| sigma (m) | uncorrected | alignformer | oracle uniform | oracle IVW | "
        "delta (IVW - alignformer) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['sigma_m']:g} | {row['uncorrected']['ap_70']:.4f} "
            f"| {row[ALIGNFORMER]['ap_70']:.4f} "
            f"| {row[ORACLE_MATCH_UNIFORM]['ap_70']:.4f} "
            f"| {row[ORACLE_MATCH_IVW]['ap_70']:.4f} "
            f"| {row['delta_ap_70_ivw_minus_alignformer']:+.4f} |"
        )
    return "\n".join(lines)


def _pose_table(rows: List[Dict]) -> str:
    lines = [
        "| sigma (m) | deployed t MAE (m) | oracle-IVW t MAE (m) | reachable by matching "
        "| deployed yaw MAE (deg) | oracle-IVW yaw MAE (deg) | reachable by matching |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        residual = row["matching_residual"]
        lines.append(
            f"| {row['sigma_m']:g} | {row[ALIGNFORMER]['translation_mae_m']:.4f} "
            f"| {row[ORACLE_MATCH_IVW]['translation_mae_m']:.4f} "
            f"| {residual['translation_m']:+.4f} "
            f"| {row[ALIGNFORMER]['yaw_mae_deg']:.4f} "
            f"| {row[ORACLE_MATCH_IVW]['yaw_mae_deg']:.4f} "
            f"| {residual['yaw_deg']:+.4f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--deployed", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    validation = json.loads(args.validation.read_text())
    test = json.loads(args.test.read_text())
    deployed = json.loads(args.deployed.read_text()) if args.deployed else None

    summary = {
        "method": "alignformer_oracle_correspondence_ceiling",
        "metric": "noisy_ap",
        "task": 21,
        "pose_checkpoint": validation["pose_checkpoint"],
        "pose_checkpoint_provenance": validation["pose_checkpoint_provenance"],
        "shrinkage": validation["shrinkage"],
        "sources": {
            "validation": str(args.validation),
            "test": str(args.test),
            "deployed_reference": str(args.deployed) if args.deployed else None,
        },
        "decision": decide(validation),
        "deployed_agreement_on_test": deployed_agreement(test, deployed),
        "validation": {
            "split": validation["split"],
            "frames": validation["frames"],
            "ap": ap_rows(validation),
            "pose": pose_rows(validation),
        },
        "test": {
            "split": test["split"],
            "frames": test["frames"],
            "ap": ap_rows(test),
            "pose": pose_rows(test),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")

    decision = summary["decision"]
    print(f"VALIDATION ({validation['frames']} frames)  AP@0.7 global-sorted")
    print(_table(summary["validation"]["ap"]))
    print()
    print("VALIDATION pose error")
    print(_pose_table(summary["validation"]["pose"]))
    print()
    print(f"TEST ({test['frames']} frames)  AP@0.7 global-sorted")
    print(_table(summary["test"]["ap"]))
    print()
    print("TEST pose error")
    print(_pose_table(summary["test"]["pose"]))
    print()
    print(f"DECISION (validation, pre-registered): {decision['verdict'].upper()}")
    print(
        f"  mean delta {decision['mean_delta']:+.4f}  "
        f"range [{decision['min_delta']:+.4f}, {decision['max_delta']:+.4f}]  "
        f"sign flips: {decision['sign_flips_across_sigma']}"
    )
    agreement = summary["deployed_agreement_on_test"]
    if agreement is not None:
        print(
            f"  deployed alignformer arm reproduces the published run exactly: "
            f"{agreement['identical']} (max |diff| {agreement['max_abs_difference']})"
        )


if __name__ == "__main__":
    main()
