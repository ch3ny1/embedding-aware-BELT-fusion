"""Select AlignFormer's decision rule on validation AP.

The treatment FreeAlign got on V2X-Real, applied to our own side: the per-pair
decision-rule family (per-pair James-Stein shrinkage, hard Wald thresholds,
threshold-then-shrink) is swept on the official validation split through the
deployed pipeline, and the arm with the best AP@0.7 sweep mean goes to test
beside the deployed ``per_pair`` arm. FreeAlign's row is carried for context
and is never a candidate. Writes the selection file and the two arm lists the
test and delay chains read (``.arms_for_test``, ``.arms_for_delay``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

METRICS = ("ap_70", "ap_50", "ap_30")
SELECTION_METRIC = "ap_70"
DEPLOYED = "alignformer_per_pair"
GLOBAL_RULES = ("alignformer", "alignformer_irls")
TOP_ARMS_TO_TEST = 3
CRITERION = "AP@0.7 sweep mean, global_sorted, official val, 3 noise seeds, boxes-only trunk"


def _sigma_key(condition: str, sigma: float) -> str:
    return f"{condition}_sigma_{sigma:g}m"


def _pose_field(pose: dict, sigma_key: str, field: str):
    return pose.get(sigma_key, {}).get(field)


def sweep_rows(result: dict) -> dict:
    """AP sweep means per condition plus the sigma-0 / sigma-2 pose summary."""
    sigmas = result["sweep_sigmas_m"]
    ap = result["ap"]
    conditions = list(GLOBAL_RULES) + list(result["conditions"]["abstention"]) + ["freealign"]

    def row(condition: str) -> dict:
        means = {m: sum(ap[_sigma_key(condition, s)][m]["global_sorted"] for s in sigmas) / len(sigmas) for m in METRICS}
        pose = result.get("pose_by_condition", {}).get(condition, {})
        return {
            **means,
            "sigma0_ap_70": ap[_sigma_key(condition, 0.0)]["ap_70"]["global_sorted"],
            "sigma2_ap_70": ap.get(_sigma_key(condition, 2.0), {}).get("ap_70", {}).get("global_sorted"),
            "sigma0_mae_m": _pose_field(pose, "sigma_0m", "translation_mae_m"),
            "sigma0_coverage": _pose_field(pose, "sigma_0m", "coverage"),
            "sigma2_coverage": _pose_field(pose, "sigma_2m", "coverage"),
        }

    return {c: row(c) for c in conditions}


def select(result: dict) -> dict:
    rows = sweep_rows(result)
    candidates = [c for c in rows if c != "freealign"]
    ranked = sorted(candidates, key=lambda c: rows[c][SELECTION_METRIC], reverse=True)
    return {
        "method": "alignformer_decision_rule_selection",
        "criterion": CRITERION,
        "split": result["split"],
        "frames": result["frames"],
        "noise_seeds": result["noise_seeds"],
        "sweep_sigmas_m": result["sweep_sigmas_m"],
        "candidates": candidates,
        "ranked": ranked,
        "chosen": ranked[0],
        "deployed": DEPLOYED,
        "rows": rows,
        "freealign": {"calibration": result.get("freealign_calibration"), "row": rows["freealign"]},
    }


def arm_spec(condition: str) -> str:
    """``alignformer_abstain_0.2`` -> ``abstain:0.2``; ``alignformer_per_pair`` -> ``per_pair``."""
    tail = condition[len("alignformer_"):]
    mode, _, level = tail.rpartition("_")
    return f"{mode}:{level}" if mode and _is_number(level) else tail


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _is_per_pair_arm(condition: str) -> bool:
    return condition.startswith("alignformer_") and condition not in GLOBAL_RULES


def arms_for_test(ranked: Sequence[str]) -> list:
    top = [arm_spec(c) for c in ranked[:TOP_ARMS_TO_TEST] if _is_per_pair_arm(c) and c != DEPLOYED]
    return ["per_pair"] + top


def arms_for_delay(chosen: str) -> list:
    chosen_spec = arm_spec(chosen) if _is_per_pair_arm(chosen) else "per_pair"
    return list(dict.fromkeys(["per_pair", chosen_spec]))


def _print_table(selection: dict) -> None:
    fmt = lambda x: "  -  " if x is None else f"{x:.3f}"  # noqa: E731
    for c, r in selection["rows"].items():
        print(
            f"{c:<28} AP@0.7 {r['ap_70']:.4f} AP@0.5 {r['ap_50']:.4f} AP@0.3 {r['ap_30']:.4f}"
            f" | s0 {r['sigma0_ap_70']:.4f} s2 {fmt(r['sigma2_ap_70'])}"
            f" | s0 mae {fmt(r['sigma0_mae_m'])} cov {fmt(r['sigma0_coverage'])} s2 cov {fmt(r['sigma2_coverage'])}"
        )
    same = "SAME as" if selection["chosen"] == DEPLOYED else "DIFFERENT from"
    print("ranked:", selection["ranked"][:5])
    print(f"chosen: {selection['chosen']} ({same} deployed {DEPLOYED})")


def _arm_flags(specs: Sequence[str]) -> str:
    return " ".join(f"--abstain-arm {a}" for a in specs) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result", required=True, type=Path, help="val sweep result with every arm (B_*_arms_val_result.json)")
    parser.add_argument("--output", required=True, type=Path, help="selection file to write")
    parser.add_argument("--arms-dir", type=Path, default=None, help="where .arms_for_test / .arms_for_delay go (default: beside --output)")
    args = parser.parse_args(argv)

    selection = select(json.loads(args.result.read_text()))
    _print_table(selection)
    args.output.write_text(json.dumps(selection, indent=2) + "\n")
    arms_dir = args.arms_dir or args.output.parent
    (arms_dir / ".arms_for_test").write_text(_arm_flags(arms_for_test(selection["ranked"])))
    (arms_dir / ".arms_for_delay").write_text(_arm_flags(arms_for_delay(selection["chosen"])))
    print("test arms:", (arms_dir / ".arms_for_test").read_text().strip())


if __name__ == "__main__":
    main()
