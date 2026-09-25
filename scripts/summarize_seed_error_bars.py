"""Task 23 part A: error bars on the AP head-to-heads, and part B's decision rule.

Every AP cell on this branch before task 23 was **one** noise draw. Differences
of 0.02-0.04 clear that instrument; the AP@0.7 head-to-head against FreeAlign,
which turns on 0.0005-0.011, does not. This script reads multi-seed sweeps
(``evaluate.py --metric noisy_ap --ap-seeds N``) and restates the three
head-to-heads with paired standard errors, so an ordering is quoted only where
the error bars support it.

**Paired, always.** Conditions inside one draw see the same perturbed poses, so
the per-seed *difference* is the statistic (see ``alignformer.seedstats``).
Overlapping independent bars would throw the covariance away and call almost
everything a draw.

**A difference inside its own standard error is a draw**, in the direction that
flatters this project as readily as in the one that does not.

**sigma = 0 has no draw.** It perturbs nothing, is run once, and is reported as
``deterministic``: exact, and silent about its own spread. It is never given a
fabricated standard deviation of 0.0.

Part B's pre-registered rule (``task-23-seeds-and-shrinkage-brief.md``) is
evaluated here rather than re-derived in prose. On VALIDATION only, the
refitted tau ships if and only if all three hold, across seeds:

1. mean AP@0.7 over the eight sigmas improves by **more than one paired
   standard error**;
2. AP@0.7 at sigma 0 does **not** regress beyond one paired standard error;
3. the 1-2 shared slice does not regress beyond one paired standard error **on
   TEST's 290-frame slice**, not validation's 5-frame one.

Clause 3 reads test deliberately and only as a *guard against regression*.
Validation's sparse slice is 5 frames / 33 pairs and swings +-0.09 with no
structure -- task 22 measured that -- so the criterion is one validation cannot
answer. Test is never used to *select* anything here; it vetoes, it does not
choose. Clause 2 has an edge the brief could not have foreseen: sigma = 0 is
deterministic, so "one paired standard error" there is exactly zero and the
clause degenerates to "does not regress at all". Both the exact delta and that
reading are reported.

Usage::

    python scripts/summarize_seed_error_bars.py \\
        --validation outputs/alignformer/r140/seeds_val_result.json \\
        --test outputs/alignformer/r140/seeds_test_result.json \\
        --single-seed-validation outputs/alignformer/r140/robust_solve_val_result.json \\
        --single-seed-test outputs/alignformer/r140/robust_solve_test_result.json \\
        --output outputs/alignformer/r140/seed_error_bars_result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embedding_aware_belt_fusion.alignformer.seedstats import (  # noqa: E402
    DETERMINISTIC,
    PairedDifference,
    paired_difference,
    spread,
    unpaired_sem,
    verdict,
)

ALIGNFORMER = "alignformer"
ALIGNFORMER_IRLS = "alignformer_irls"
FREEALIGN = "freealign"
UNCORRECTED = "uncorrected"
ORACLE = "oracle"
AP_KEYS = ("ap_30", "ap_50", "ap_70")
DECISION_METRIC = "ap_70"
SPARSE_BUCKET = "shared_1_2"
SORT = "global_sorted"

# The three orderings task 23 exists to settle, left minus right.
HEAD_TO_HEADS = (
    (ALIGNFORMER_IRLS, ALIGNFORMER),
    (ALIGNFORMER_IRLS, FREEALIGN),
    (FREEALIGN, ALIGNFORMER),
)


def _key(condition: str, sigma: float) -> str:
    return condition if condition == ORACLE else f"{condition}_sigma_{sigma:g}m"


def _seeds(result: Dict) -> List[str]:
    return [str(seed) for seed in result["noise_seeds"]]


def _series(
    result: Dict, condition: str, sigma: float, metric: str, *, bucket: Optional[str] = None
) -> List[float]:
    """One cell's value under every seed, in seed order.

    A sigma with no draw in it (sigma = 0) is the same run under every seed, so
    the series is a constant -- which is the honest input to a paired
    difference: it contributes to the mean and nothing to the spread.
    """
    values = []
    for seed in _seeds(result):
        if bucket is None:
            cell = result["ap_by_seed"][seed][_key(condition, sigma)]
        else:
            cell = result["ap_by_shared_objects_by_seed"][seed][bucket]["conditions"][
                _key(condition, sigma)
            ]
        values.append(float(cell[metric][SORT]))
    return values


def _drawn_once(result: Dict, sigma: float) -> bool:
    return sigma in result.get("sigmas_drawn_once_m", [])


def _cell_series(
    result: Dict, condition: str, sigma: float, metric: str,
    *, bucket: Optional[str] = None,
) -> List[float]:
    """The DISTINCT draws of one cell -- one value where there is no draw.

    ``_series`` repeats a deterministic sigma once per seed, which is what a
    sweep mean needs. A per-sigma error bar must not: five copies of one number
    would report a standard deviation of 0.0 and make the cell look infinitely
    well resolved, which is precisely the fabricated error bar this task exists
    to remove.
    """
    values = _series(result, condition, sigma, metric, bucket=bucket)
    return values[:1] if _drawn_once(result, sigma) else values


def _sweep_mean_series(
    result: Dict, condition: str, sigmas: Sequence[float], metric: str
) -> List[float]:
    """Per seed, the mean of that seed's AP over the whole sigma grid.

    Deterministic sigmas enter every seed's mean as the same constant, which is
    correct: they carry signal and no sampling noise.
    """
    per_seed = [_series(result, condition, sigma, metric) for sigma in sigmas]
    return [sum(column) / len(column) for column in zip(*per_seed)]


def _cells(result: Dict, sigmas: Sequence[float], metric: str) -> Dict:
    """Mean/sd per condition per sigma, and the three paired head-to-heads."""
    conditions = [UNCORRECTED, ALIGNFORMER, ALIGNFORMER_IRLS, FREEALIGN]
    available = [
        name for name in conditions
        if _key(name, sigmas[0]) in result["ap_by_seed"][_seeds(result)[0]]
    ]
    table: Dict[str, Dict] = {}
    for sigma in sigmas:
        entry: Dict[str, Dict] = {
            "drawn_once": _drawn_once(result, sigma),
            "conditions": {
                name: spread(_cell_series(result, name, sigma, metric)).to_dict()
                for name in available
            },
            "head_to_head": {},
        }
        for left, right in HEAD_TO_HEADS:
            if left in available and right in available:
                first = _cell_series(result, left, sigma, metric)
                second = _cell_series(result, right, sigma, metric)
                cell = paired_difference(first, second).to_dict()
                # Beside the paired bar, the one independent bars would give.
                # Where they agree, the pairing bought nothing here.
                cell["unpaired_sem"] = unpaired_sem(first, second)
                entry["head_to_head"][f"{left}_minus_{right}"] = cell
        table[f"sigma_{sigma:g}m"] = entry
    return table


def _sweep_summary(result: Dict, sigmas: Sequence[float], metric: str) -> Dict:
    """The head-to-heads on the sweep mean, plus a per-sigma verdict tally."""
    summary: Dict[str, Dict] = {}
    for left, right in HEAD_TO_HEADS:
        first = _seeds(result)[0]
        if _key(left, sigmas[0]) not in result["ap_by_seed"][first]:
            continue
        difference = paired_difference(
            _sweep_mean_series(result, left, sigmas, metric),
            _sweep_mean_series(result, right, sigmas, metric),
        )
        tally = {"left": 0, "right": 0, "draw": 0, DETERMINISTIC: 0}
        for sigma in sigmas:
            call = verdict(
                paired_difference(
                    _cell_series(result, left, sigma, metric),
                    _cell_series(result, right, sigma, metric),
                )
            )
            tally[call] += 1
        summary[f"{left}_minus_{right}"] = {
            "sweep_mean": difference.to_dict(),
            "per_sigma_verdicts": tally,
        }
    return summary


def _source(path: Path, result: Dict) -> Dict:
    """Where one side of the report came from: the file, and the split inside it."""
    return {
        "path": str(path),
        "split": result.get("split"),
        "frames": result.get("frames"),
    }


def _reject_one_split_reported_as_two(parser, left: Path, right: Path,
                                      validation: Dict, test: Dict) -> None:
    """Refuse a report whose two splits are the same data.

    Agreement between validation and test is load-bearing for the AP@0.7
    question this script exists to settle, and a run that read one file twice
    would MANUFACTURE that agreement while looking exactly like a real result.
    Both the file and the split recorded inside it are checked, because two
    different files can still be two copies of one split.
    """
    if left.resolve() == right.resolve():
        parser.error(
            f"--validation and --test are the same file ({left}); a summary "
            "whose test rows are its validation rows is not a comparison"
        )
    if validation.get("split") is not None and validation["split"] == test.get("split"):
        parser.error(
            f"--validation and --test are two files from the same split "
            f"({validation['split']!r}); they cannot both be reported"
        )


def _identity_check(multi: Dict, single: Dict, sigmas: Sequence[float]) -> Dict:
    """Does the primary seed still reproduce the published single-seed file?

    Everything downstream is a difference of differences; if the seed that
    matches has drifted, none of it means anything. Compared over every
    condition the old file carries, at all three IoU thresholds and both sort
    orders.
    """
    first = _seeds(multi)[0]
    worst, cells, missing = 0.0, 0, []
    for name, old in single["ap"].items():
        new = multi["ap_by_seed"][first].get(name)
        if new is None:
            missing.append(name)
            continue
        for threshold in AP_KEYS:
            for sort in (SORT, "frame_order"):
                worst = max(worst, abs(float(old[threshold][sort]) - float(new[threshold][sort])))
                cells += 1
    return {
        "seed": int(first),
        "cells_compared": cells,
        "max_abs_difference": worst,
        "conditions_missing_from_multi_seed_run": missing,
        "bit_identical": worst == 0.0 and not missing,
    }


def _check_refit_calibration(calibration: Dict, runs: Sequence[Dict]) -> Dict:
    """Refuse a "refit" sweep that was not actually run through a refitted tau.

    A sweep applies ONE calibration to every arm of ours, so a tau fitted on
    the IRLS arm's residuals also reaches the plain ``alignformer`` arm in that
    run. That column is then a deployed arm scored through another estimator's
    calibration: not a measurement of anything, and nothing in the file says
    so. Part B reads only :data:`ALIGNFORMER_IRLS` from these runs, and this
    check makes that a verified precondition rather than a convention -- it
    confirms the calibration was fitted with the robust loop ENABLED and that
    the sweeps were actually run through that same tau.
    """
    config = calibration.get("robust_solve_config") or {}
    tau = calibration["calibration"]["tau_translation_m"]
    applied = [run.get("shrinkage", {}).get("tau_translation_m") for run in runs]
    problems = []
    if not config.get("enabled"):
        problems.append(
            "the calibration was fitted on the DEPLOYED solve's residuals "
            "(robust_solve_config.enabled is false), so it is not a refit"
        )
    if any(value != tau for value in applied):
        problems.append(
            f"the refit sweeps were run through tau {applied}, not the "
            f"refitted {tau}"
        )
    if problems:
        raise ValueError("; ".join(problems))
    return {
        "refit_tau_translation_m": tau,
        "fitted_on": ALIGNFORMER_IRLS,
        "robust_solve_config": config,
        # Said out loud because the file cannot say it for itself.
        "deployed_arm_column_in_these_runs_is_not_a_measurement": (
            "one calibration reaches every arm of ours, so the `alignformer` "
            "rows of the refit sweeps are the deployed solve scored through "
            "the IRLS arm's tau; only alignformer_irls is read from them"
        ),
    }


def _decision(
    validation: Dict,
    test: Dict,
    baseline: Dict,
    baseline_test: Dict,
    calibration: Dict,
    sigmas: Sequence[float],
) -> Dict:
    """Part B's pre-registered rule, evaluated on the refitted-tau sweeps.

    ``validation``/``test`` carry the refitted-tau arm; ``baseline``/
    ``baseline_test`` carry the deployed-tau arm. Both must be multi-seed runs
    over the same seeds, or the differences are not paired.
    """
    arm = ALIGNFORMER_IRLS
    provenance = _check_refit_calibration(calibration, [validation, test])
    pairing = {
        split: _same_draws(new, old, sigmas)
        for split, new, old in (
            ("validation", validation, baseline), ("test", test, baseline_test)
        )
    }

    sweep = paired_difference(
        _sweep_mean_series(validation, arm, sigmas, DECISION_METRIC),
        _sweep_mean_series(baseline, arm, sigmas, DECISION_METRIC),
    )
    clean = paired_difference(
        _cell_series(validation, arm, 0.0, DECISION_METRIC),
        _cell_series(baseline, arm, 0.0, DECISION_METRIC),
    )
    sparse = {
        f"sigma_{sigma:g}m": paired_difference(
            _cell_series(test, arm, sigma, DECISION_METRIC, bucket=SPARSE_BUCKET),
            _cell_series(baseline_test, arm, sigma, DECISION_METRIC, bucket=SPARSE_BUCKET),
        )
        for sigma in sigmas
    }
    worst_sigma = min(sparse, key=lambda k: sparse[k].mean)
    worst = sparse[worst_sigma]

    criteria = {
        "mean_ap70_improves_by_more_than_one_paired_sem": {
            "measured": sweep.to_dict(),
            "required": "mean > sem (validation, over the sigma grid)",
            "pass": _exceeds(sweep),
        },
        "sigma_0_does_not_regress_beyond_one_paired_sem": {
            "measured": clean.to_dict(),
            "required": "mean >= -sem (validation, sigma = 0)",
            # sigma = 0 is deterministic: its paired sem is not estimated, so
            # "one standard error" is zero and the clause reads "does not
            # regress". Stated rather than silently relaxed.
            "note": (
                "sigma = 0 is a single deterministic run; one paired standard "
                "error is exactly zero, so this clause reads 'does not regress'"
            ),
            "pass": _not_worse(clean),
        },
        "test_sparse_slice_does_not_regress_beyond_one_paired_sem": {
            "worst_sigma": worst_sigma,
            "measured": worst.to_dict(),
            "per_sigma": {key: value.to_dict() for key, value in sparse.items()},
            "required": "mean >= -sem at every sigma (TEST's 290-frame slice)",
            "note": (
                "read on test because validation's sparse slice is 5 frames / 33 "
                "pairs; used ONLY as a guard against regression, never to select"
            ),
            "pass": all(_not_worse(value) for value in sparse.values()),
        },
    }
    passed = all(entry["pass"] for entry in criteria.values())
    return {
        # The two runs are separate invocations, so "paired" is a claim about
        # them, not a property of one file. `uncorrected` depends on the noise
        # draw and on nothing else the refit touched, so its agreeing bit for
        # bit across the two runs IS the proof that seed k saw the same
        # perturbed poses in both -- without which every difference below is a
        # difference of two unrelated experiments.
        "same_draws_in_both_runs": pairing,
        "refit_calibration": provenance,
        "hypothesis": (
            "H2: refitting tau to the IRLS arm's own residual distribution "
            "recovers the low-sigma AP@0.7 cells"
        ),
        "criteria": criteria,
        "verdict": "SHIP_REFITTED_TAU" if passed else "NULL_KEEP_DEPLOYED_TAU",
    }


def _same_draws(new: Dict, old: Dict, sigmas: Sequence[float]) -> Dict:
    """Do two runs' ``uncorrected`` rows agree exactly, seed for seed?"""
    worst, cells = 0.0, 0
    for sigma in sigmas:
        for metric in AP_KEYS:
            for left, right in zip(
                _series(new, UNCORRECTED, sigma, metric),
                _series(old, UNCORRECTED, sigma, metric),
            ):
                worst = max(worst, abs(left - right))
                cells += 1
    return {
        "cells_compared": cells,
        "max_abs_difference": worst,
        "paired": worst == 0.0,
    }


def _exceeds(difference: PairedDifference) -> bool:
    """Improves by strictly more than one paired standard error."""
    if difference.sem is None:
        return difference.mean > 0.0
    return difference.mean > difference.sem


def _not_worse(difference: PairedDifference) -> bool:
    """Does not regress beyond one paired standard error."""
    if difference.sem is None:
        return difference.mean >= 0.0
    return difference.mean >= -difference.sem


_LABELS = {
    UNCORRECTED: "uncorr",
    ALIGNFORMER: "AF",
    ALIGNFORMER_IRLS: "+IRLS",
    FREEALIGN: "FA",
}


def _cell_text(cell: Optional[Dict]) -> str:
    if cell is None:
        return f"{'--':>17}"
    if cell["sd"] is None:
        return f"{cell['mean']:>10.4f}  det  "
    return f"{cell['mean']:>10.4f}+-{cell['sd']:<.4f}"


def _difference_text(difference: Optional[Dict]) -> str:
    if difference is None:
        return f"{'--':>23}"
    if difference["sem"] is None:
        return f"  {difference['mean']:+.4f}          det"
    return (
        f"  {difference['mean']:+.4f}+-{difference['sem']:.4f} "
        f"{difference['verdict']:>5}"
    )


def _print_table(name: str, table: Dict, metric: str) -> None:
    print(f"\n== {name} -- {metric.replace('ap_', 'AP@0.')} ==", flush=True)
    header = f"{'sigma':>5} "
    for condition in (UNCORRECTED, ALIGNFORMER, ALIGNFORMER_IRLS, FREEALIGN):
        header += f"{_LABELS[condition]:>17}"
    print(header + f"{'+IRLS - AF':>23}{'+IRLS - FA':>23}", flush=True)
    for key, entry in table.items():
        row = f"{key.replace('sigma_', '').replace('m', ''):>5} "
        for condition in (UNCORRECTED, ALIGNFORMER, ALIGNFORMER_IRLS, FREEALIGN):
            row += _cell_text(entry["conditions"].get(condition))
        for pair in (
            f"{ALIGNFORMER_IRLS}_minus_{ALIGNFORMER}",
            f"{ALIGNFORMER_IRLS}_minus_{FREEALIGN}",
        ):
            row += _difference_text(entry["head_to_head"].get(pair))
        print(row, flush=True)


def _print_head_to_heads(name: str, split: Dict) -> None:
    """All three orderings, per threshold: the sweep mean and the per-sigma tally.

    ``left``/``right`` count the sigmas where the paired difference exceeds its
    own standard error; ``draw`` counts the ones where it does not and which
    therefore support NO ordering, in either direction. ``det`` is sigma = 0,
    resolved exactly by one deterministic run and silent about its spread.
    """
    print(f"\n== {name} -- head-to-heads, paired over seeds ==", flush=True)
    for metric in AP_KEYS:
        for pair, entry in split[metric]["summary"].items():
            mean = entry["sweep_mean"]
            tally = entry["per_sigma_verdicts"]
            bar = (
                "  det" if mean["sem"] is None
                else f"+-{mean['sem']:.4f} {mean['verdict']:>5}"
            )
            print(
                f"  {metric.replace('ap_', 'AP@0.'):>7}  {pair:<38} "
                f"sweep mean {mean['mean']:+.4f} {bar}   per sigma: "
                f"left {tally['left']}  right {tally['right']}  "
                f"draw {tally['draw']}  det {tally[DETERMINISTIC]}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument(
        "--single-seed-validation", type=Path, default=None,
        help="the published single-seed run the primary seed must reproduce",
    )
    parser.add_argument("--single-seed-test", type=Path, default=None)
    parser.add_argument(
        "--refit-validation", type=Path, default=None,
        help="part B: the refitted-tau validation sweep, over the same seeds",
    )
    parser.add_argument("--refit-test", type=Path, default=None)
    parser.add_argument(
        "--refit-calibration", type=Path, default=None,
        help="part B: the shrinkage JSON the refit sweeps were run through, so "
             "that 'this is a refit' is verified rather than assumed",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    validation = json.loads(args.validation.read_text())
    test = json.loads(args.test.read_text())
    _reject_one_split_reported_as_two(
        parser, args.validation, args.test, validation, test
    )
    sigmas = list(validation["sweep_sigmas_m"])
    if list(test["sweep_sigmas_m"]) != sigmas:
        raise ValueError("the two splits were swept over different sigma grids")
    if validation["noise_seeds"] != test["noise_seeds"]:
        raise ValueError("the two splits were run under different seeds")

    payload: Dict[str, object] = {
        "task": "task-23 seeds and shrinkage",
        # WHICH file and WHICH split each side of every table came from. Every
        # other result file in this project records its split; a summary that
        # spans two of them has to record both, or a reader cannot tell an
        # agreement between the splits from the same split reported twice.
        "sources": {
            "validation": _source(args.validation, validation),
            "test": _source(args.test, test),
        },
        "noise_seeds": validation["noise_seeds"],
        "sweep_sigmas_m": sigmas,
        "sigmas_drawn_once_m": validation.get("sigmas_drawn_once_m", []),
        "frames": {"validation": validation["frames"], "test": test["frames"]},
        "shrinkage": {
            "validation": validation.get("shrinkage"),
            "test": test.get("shrinkage"),
        },
        "robust_solve_config": validation.get("robust_solve_config"),
    }

    for split_name, result in (("validation", validation), ("test", test)):
        payload[split_name] = {
            metric: {
                "per_sigma": _cells(result, sigmas, metric),
                "summary": _sweep_summary(result, sigmas, metric),
            }
            for metric in AP_KEYS
        }
        payload[f"{split_name}_sparse_slice_ap_70"] = {
            f"sigma_{sigma:g}m": {
                f"{left}_minus_{right}": paired_difference(
                    _cell_series(result, left, sigma, DECISION_METRIC, bucket=SPARSE_BUCKET),
                    _cell_series(result, right, sigma, DECISION_METRIC, bucket=SPARSE_BUCKET),
                ).to_dict()
                for left, right in HEAD_TO_HEADS
            }
            for sigma in sigmas
        }
        for metric in AP_KEYS:
            _print_table(split_name, payload[split_name][metric]["per_sigma"], metric)
        _print_head_to_heads(split_name, payload[split_name])

    identity = {}
    for split_name, multi, single in (
        ("validation", validation, args.single_seed_validation),
        ("test", test, args.single_seed_test),
    ):
        if single is not None:
            identity[split_name] = _identity_check(
                multi, json.loads(single.read_text()), sigmas
            )
    if identity:
        payload["single_seed_reproduction"] = identity
        for split_name, check in identity.items():
            print(
                f"\nbit-identity {split_name}: seed {check['seed']}, "
                f"{check['cells_compared']} cells, max |diff| = "
                f"{check['max_abs_difference']:.10g} -> "
                f"{'IDENTICAL' if check['bit_identical'] else 'DRIFTED'}",
                flush=True,
            )

    if args.refit_validation is not None and args.refit_test is not None:
        if args.refit_calibration is None:
            parser.error("--refit-validation/--refit-test need --refit-calibration")
        refit_val = json.loads(args.refit_validation.read_text())
        refit_test = json.loads(args.refit_test.read_text())
        if refit_val["noise_seeds"] != validation["noise_seeds"]:
            raise ValueError("the refit sweep used different seeds; it is not paired")
        # A refit sweep is compared against the baseline sweep of the SAME
        # split; crossing them would compare validation against test.
        for name, refit, baseline in (
            ("validation", refit_val, validation), ("test", refit_test, test)
        ):
            if refit.get("split") != baseline.get("split"):
                parser.error(
                    f"the {name} refit sweep is from split {refit.get('split')!r} "
                    f"but its baseline is from {baseline.get('split')!r}"
                )
        payload["refit_sources"] = {
            "validation": _source(args.refit_validation, refit_val),
            "test": _source(args.refit_test, refit_test),
            "calibration": str(args.refit_calibration),
        }
        decision = _decision(
            refit_val, refit_test, validation, test,
            json.loads(args.refit_calibration.read_text()), sigmas,
        )
        payload["part_b_decision"] = decision
        for split, check in decision["same_draws_in_both_runs"].items():
            print(
                f"\npaired draws {split}: {check['cells_compared']} uncorrected "
                f"cells, max |diff| = {check['max_abs_difference']:.10g} -> "
                f"{'SAME DRAWS' if check['paired'] else 'NOT PAIRED'}",
                flush=True,
            )
        print(f"part B: {decision['verdict']}", flush=True)
        for name, entry in decision["criteria"].items():
            print(f"  [{'pass' if entry['pass'] else 'FAIL'}] {name}", flush=True)

    args.output.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
