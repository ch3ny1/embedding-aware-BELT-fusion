"""Task 24: per-pair abstention, its coverage, and the pre-registered rule.

Task 23 settled that the AP@0.7 deficit to FreeAlign is not spread across the
sweep -- sigma = 0.2 alone outweighs every cell we win -- and that a refitted
global shrinkage ``tau`` cannot fix it, because a smaller tau shrinks less
*everywhere*, including at sigma = 0 where it must shrink more. Task 24 replaces
that global scalar with a per-pair decision taken from the estimator's own
covariance (``alignformer.abstain``).

This script reads multi-seed sweeps carrying one or more abstention arms beside
the deployed ``alignformer_irls`` arm **in the same run**, and reports:

* every arm's AP against ``alignformer_irls``, paired inside each seed;
* **coverage beside every AP number**. This is a headline, not a footnote. The
  intervention deliberately makes the method answer less often, and a method
  that abstains more looks better on what it does answer -- which is the exact
  reporting trap this project criticized in FreeAlign's own numbers and which
  now applies to us;
Clause 4's granularity is not stated in the brief. ``_sparse_guard`` computes
both readings, applies the sweep-mean one, and says why in its docstring; the
per-cell number the other reading gives is emitted beside it so a reader can
apply either.

* the pre-registered decision rule, evaluated on VALIDATION only, over the arms
  **in the order the brief names them**: hard abstain, then per-pair shrinkage,
  then both, and within a mode by ascending level -- least intervention first.
  The first arm that passes every clause ships. Scanning all of them and keeping
  the best would be selecting on the same data that measures, and the
  within-mode tie-break is stated here rather than buried in a sort key because
  it decides which arm ships when several pass;
* the FreeAlign comparison, reported *after* the decision is frozen and never
  used as a criterion.

Because both arms of the decision live in one sweep invocation, the pairing is
structural rather than a claim about two files: ``draws.sweep_noisy_poses``
produces one perturbation per (sigma, seed, frame) and every condition in the
draw consumes it.

Usage::

    python scripts/summarize_abstention.py \\
        --validation outputs/alignformer/r140/abstain_val_result.json \\
        --baseline-validation outputs/alignformer/r140/seeds_val_result.json \\
        [--test outputs/alignformer/r140/abstain_test_result.json] \\
        [--baseline-test outputs/alignformer/r140/seeds_test_result.json] \\
        --output outputs/alignformer/r140/abstention_result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embedding_aware_belt_fusion.alignformer.seedstats import (  # noqa: E402
    PairedDifference,
    paired_difference,
    spread,
    verdict,
)

ALIGNFORMER = "alignformer"
ALIGNFORMER_IRLS = "alignformer_irls"
FREEALIGN = "freealign"
UNCORRECTED = "uncorrected"
ORACLE = "oracle"
AP_KEYS = ("ap_30", "ap_50", "ap_70")
DECISION_METRIC = "ap_70"
GUARD_METRICS = ("ap_30", "ap_50")
SPARSE_BUCKET = "shared_1_2"
SORT = "global_sorted"

# The brief's order: "Try, in this order: hard abstain threshold; shrinkage
# with a per-pair tau derived from the covariance; both." The first arm that
# satisfies every clause is the one that ships, so this order is part of the
# rule and not a display preference.
MODE_ORDER = ("abstain", "per_pair", "both")


def _key(condition: str, sigma: float) -> str:
    return condition if condition == ORACLE else f"{condition}_sigma_{sigma:g}m"


def _seeds(result: Dict) -> List[str]:
    return [str(seed) for seed in result["noise_seeds"]]


def _drawn_once(result: Dict, sigma: float) -> bool:
    # Indexed, not `.get(..., [])`: a missing key would make every
    # deterministic cell look drawn, giving it five identical copies, an
    # sd of 0.0 and an infinitely significant paired difference. The failure
    # is silent and points the wrong way, so it must be an error.
    return sigma in result["sigmas_drawn_once_m"]


def _drawn_sigmas(result: Dict, sigmas: Sequence[float]) -> List[float]:
    """The sigmas that actually have a noise draw in them.

    Every sweep mean in this report is computed twice: over the WHOLE grid,
    because that is what the pre-registered rule is written over, and over this
    subset, because it is the operating regime. ``sigma = 0`` is a degenerate
    cell -- no localization error at all -- and a grid mean weights it equally
    with the seven that have some. A sign change carried by that one cell is
    true arithmetic and a misleading read, which is the exact failure task 23
    caught in task 22's cell count.
    """
    return [sigma for sigma in sigmas if not _drawn_once(result, sigma)]


def _series(
    result: Dict, condition: str, sigma: float, metric: str,
    *, bucket: Optional[str] = None,
) -> List[float]:
    """One cell's value under every seed, in seed order."""
    values = []
    for seed in _seeds(result):
        if bucket is None:
            cell = result["ap_by_seed"][seed][_key(condition, sigma)]
        else:
            cell = result["ap_by_shared_objects_by_seed"][seed][bucket][
                "conditions"
            ][_key(condition, sigma)]
        values.append(float(cell[metric][SORT]))
    return values


def _cell_series(
    result: Dict, condition: str, sigma: float, metric: str,
    *, bucket: Optional[str] = None,
) -> List[float]:
    """The DISTINCT draws of one cell -- one value where there is no draw.

    A per-sigma error bar must not repeat a deterministic sigma once per seed:
    five copies of one number report a standard deviation of 0.0 and make the
    cell look infinitely well resolved.
    """
    values = _series(result, condition, sigma, metric, bucket=bucket)
    return values[:1] if _drawn_once(result, sigma) else values


def _sweep_mean_series(
    result: Dict, condition: str, sigmas: Sequence[float], metric: str,
    *, bucket: Optional[str] = None,
) -> List[float]:
    """Per seed, the mean of that seed's AP over the whole sigma grid."""
    per_seed = [
        _series(result, condition, sigma, metric, bucket=bucket) for sigma in sigmas
    ]
    return [sum(column) / len(column) for column in zip(*per_seed)]


def _coverage_series(result: Dict, condition: str, sigma: float) -> List[float]:
    """Fraction of pairs this arm actually corrected, per seed.

    Read off the emitted ``(psi, t)`` by ``stage2.is_fallback`` during the
    sweep, so an abstention, a shrinkage that reached exactly zero and a
    ``MIN_MATCH_MASS`` suppression all count the same way: the pair was not
    answered.
    """
    values = []
    for seed in _seeds(result):
        stats = result["pose_by_condition_by_seed"][seed][condition]
        values.append(float(stats[f"sigma_{sigma:g}m"]["coverage"]))
    return values


def _coverage_cell(result: Dict, condition: str, sigma: float) -> List[float]:
    values = _coverage_series(result, condition, sigma)
    return values[:1] if _drawn_once(result, sigma) else values


def _translation_series(result: Dict, condition: str, sigma: float) -> List[float]:
    """Mean magnitude of the correction this arm actually emitted, per seed.

    Coverage is a count and this is a size; an arm can keep its coverage and
    shrink every correction to nothing, or abstain on half its pairs and leave
    the rest untouched. Reporting only the count would let those look alike.
    """
    values = []
    for seed in _seeds(result):
        stats = result["pose_by_condition_by_seed"][seed][condition]
        values.append(float(stats[f"sigma_{sigma:g}m"]["emitted_translation_m"]))
    return values


def _translation_cell(result: Dict, condition: str, sigma: float) -> List[float]:
    values = _translation_series(result, condition, sigma)
    return values[:1] if _drawn_once(result, sigma) else values


def _abstention_arms(result: Dict) -> List[Dict]:
    """The arms a sweep carries, in the brief's order, then by level."""
    arms = list(result.get("abstention_arms") or [])
    return sorted(
        arms,
        key=lambda arm: (MODE_ORDER.index(arm["mode"]), float(arm["level"])),
    )


def _exceeds(difference: PairedDifference) -> bool:
    """Improves by strictly more than one paired standard error.

    A single draw has no standard error, so the clause would degrade to "any
    improvement at all" -- looser than what it is named after, and loose in the
    direction that flatters the intervention while the clause that guards
    (:func:`_not_worse`) would tighten. Multi-seed input is required up front
    (:func:`_require_error_bars`) so this case cannot arise silently.
    """
    if difference.sem is None:
        return difference.mean > 0.0
    return difference.mean > difference.sem


def _not_worse(difference: PairedDifference) -> bool:
    """Does not regress beyond one paired standard error.

    ``sigma = 0`` legitimately has no standard error -- it is drawn once -- and
    there the clause reads "does not regress", which is the strict reading and
    is stated in the JSON beside the number.
    """
    if difference.sem is None:
        return difference.mean >= 0.0
    return difference.mean >= -difference.sem


def _require_error_bars(parser, result: Dict, label: str) -> None:
    """Refuse a decision taken on one noise draw.

    The headline clause is "improves by more than one paired standard error".
    With one seed there is no standard error and the bar silently becomes "any
    improvement at all", while the JSON still reads ``mean > sem``. Rejecting
    the input is the only way that cannot be missed.
    """
    seeds = result.get("noise_seeds") or []
    if len(seeds) < 2:
        parser.error(
            f"--{label} was run with {len(seeds)} noise seed(s); the decision "
            "clauses are stated in paired standard errors and need at least two"
        )


def _arm_table(result: Dict, arm: str, sigmas: Sequence[float]) -> Dict:
    """One arm's AP and coverage per sigma, each against ``alignformer_irls``."""
    table: Dict[str, Dict] = {}
    for sigma in sigmas:
        entry: Dict[str, Dict] = {
            "drawn_once": _drawn_once(result, sigma),
            "coverage": spread(_coverage_cell(result, arm, sigma)).to_dict(),
            "baseline_coverage": spread(
                _coverage_cell(result, ALIGNFORMER_IRLS, sigma)
            ).to_dict(),
            # Coverage counts a pair as answered whenever the emitted
            # correction is not exactly zero. For the hard-abstain arm that is
            # the decision; for the shrinking arms a factor of 3e-4 -- a
            # third of a millimetre -- also counts. The mean correction
            # magnitude is reported beside it so "answers less" and "corrects
            # less" are not silently conflated in a comparison the rule turns
            # on.
            "mean_correction_m": spread(
                _translation_cell(result, arm, sigma)
            ).to_dict(),
            "baseline_mean_correction_m": spread(
                _translation_cell(result, ALIGNFORMER_IRLS, sigma)
            ).to_dict(),
            "ap": {},
            "ap_minus_baseline": {},
        }
        for metric in AP_KEYS:
            entry["ap"][metric] = spread(
                _cell_series(result, arm, sigma, metric)
            ).to_dict()
            entry["ap_minus_baseline"][metric] = paired_difference(
                _cell_series(result, arm, sigma, metric),
                _cell_series(result, ALIGNFORMER_IRLS, sigma, metric),
            ).to_dict()
        table[f"sigma_{sigma:g}m"] = entry
    return table


def _sweep_means(result: Dict, arm: str, sigmas: Sequence[float]) -> Dict:
    """The arm's sweep-mean AP and coverage, and the paired difference."""
    summary: Dict[str, Dict] = {
        "coverage": spread(
            [
                sum(column) / len(column)
                for column in zip(
                    *[_coverage_series(result, arm, sigma) for sigma in sigmas]
                )
            ]
        ).to_dict(),
    }
    drawn = _drawn_sigmas(result, sigmas)
    summary["coverage_drawn_sigmas_only"] = spread(
        [
            sum(column) / len(column)
            for column in zip(
                *[_coverage_series(result, arm, sigma) for sigma in drawn]
            )
        ]
    ).to_dict()
    for metric in AP_KEYS:
        difference = paired_difference(
            _sweep_mean_series(result, arm, sigmas, metric),
            _sweep_mean_series(result, ALIGNFORMER_IRLS, sigmas, metric),
        )
        drawn_difference = paired_difference(
            _sweep_mean_series(result, arm, drawn, metric),
            _sweep_mean_series(result, ALIGNFORMER_IRLS, drawn, metric),
        )
        tally = {"left": 0, "right": 0, "draw": 0, "deterministic": 0}
        for sigma in sigmas:
            tally[
                verdict(
                    paired_difference(
                        _cell_series(result, arm, sigma, metric),
                        _cell_series(result, ALIGNFORMER_IRLS, sigma, metric),
                    )
                )
            ] += 1
        summary[metric] = {
            "value": spread(_sweep_mean_series(result, arm, sigmas, metric)).to_dict(),
            "minus_baseline": difference.to_dict(),
            # The same difference over the seven sigmas that HAVE a draw. The
            # rule is written over the whole grid; this is the operating
            # regime, and both are reported so neither can be quoted alone.
            "minus_baseline_drawn_sigmas_only": drawn_difference.to_dict(),
            "per_sigma_verdicts": tally,
        }
    return summary


def _validation_criteria(
    validation: Dict, arm: str, sigmas: Sequence[float]
) -> Dict:
    """Clauses 1-3 of the pre-registered rule. Validation only, by construction.

    Clause 4 reads test's sparse slice and is a veto applied afterwards; it is
    not evaluated here, so nothing in the selection can see a test number.
    """
    sweep = paired_difference(
        _sweep_mean_series(validation, arm, sigmas, DECISION_METRIC),
        _sweep_mean_series(validation, ALIGNFORMER_IRLS, sigmas, DECISION_METRIC),
    )
    clean = paired_difference(
        _cell_series(validation, arm, 0.0, DECISION_METRIC),
        _cell_series(validation, ALIGNFORMER_IRLS, 0.0, DECISION_METRIC),
    )
    guards = {
        metric: paired_difference(
            _sweep_mean_series(validation, arm, sigmas, metric),
            _sweep_mean_series(validation, ALIGNFORMER_IRLS, sigmas, metric),
        )
        for metric in GUARD_METRICS
    }
    return {
        "mean_ap70_improves_by_more_than_one_paired_sem": {
            "measured": sweep.to_dict(),
            "required": "mean > sem (validation, over the sigma grid)",
            "pass": _exceeds(sweep),
        },
        "sigma_0_does_not_regress_at_all": {
            "measured": clean.to_dict(),
            "required": "mean >= 0 (validation, sigma = 0)",
            "note": (
                "the brief's words are 'does not regress at all'; sigma = 0 is "
                "a single deterministic run, so it has no standard error to "
                "spend and the clause is read literally"
            ),
            "pass": clean.mean >= 0.0,
        },
        "ap30_and_ap50_do_not_regress_beyond_one_paired_sem": {
            "measured": {
                metric: value.to_dict() for metric, value in guards.items()
            },
            "required": "mean >= -sem for both (validation, sweep mean)",
            "pass": all(_not_worse(value) for value in guards.values()),
        },
    }


def _sparse_guard(
    test: Dict, arm: str, sigmas: Sequence[float]
) -> Dict:
    """Clause 4: test's 1-2 shared slice, as a veto and never as a selection.

    **The brief does not state this clause's granularity and both readings are
    computed here.** Verbatim (task-24 brief line 103): "the 1-2 shared slice
    does not regress beyond one paired SE **on test's 290-frame slice**, read as
    a guard only, never for selection." There is no "at every sigma" in it.

    The clause is evaluated at the **sweep mean**, for three reasons that do not
    depend on which way the numbers came out:

    1. *Parallel construction.* "beyond one paired SE" occurs three times in the
       decision rule; in clauses 1 and 3 it is attached to a sweep mean, and
       clause 2 says "sigma 0" where it means one cell. Clause 4 names no cell.
    2. *A prior ruling.* Task 23's brief, which predates this task: "A criterion
       the instrument cannot measure is not a criterion. Test's sparse slice is
       only ever read as a *guard against regression*, never to select a
       parameter." A per-cell reading makes a per-cell criterion out of an
       instrument already ruled too noisy to support one.
    3. *The per-cell reading cannot do the job.* One paired SE, one-sided, five
       seeds is |t| > 1 on 4 df: P = 0.187 that a cell fires under the null, so
       across eight cells roughly 0.81 that at least one does. That is not a
       guard against regression; it is a veto that fires four times in five on
       an arm with no true effect at all.

    Both numbers are emitted regardless, so a reader can apply either.
    """
    per_sigma = {
        f"sigma_{sigma:g}m": paired_difference(
            _cell_series(test, arm, sigma, DECISION_METRIC, bucket=SPARSE_BUCKET),
            _cell_series(
                test, ALIGNFORMER_IRLS, sigma, DECISION_METRIC, bucket=SPARSE_BUCKET
            ),
        )
        for sigma in sigmas
    }
    worst_sigma = min(per_sigma, key=lambda key: per_sigma[key].mean)
    sweep = paired_difference(
        _sweep_mean_series(
            test, arm, sigmas, DECISION_METRIC, bucket=SPARSE_BUCKET
        ),
        _sweep_mean_series(
            test, ALIGNFORMER_IRLS, sigmas, DECISION_METRIC, bucket=SPARSE_BUCKET
        ),
    )
    drawn = _drawn_sigmas(test, sigmas)
    sweep_drawn = paired_difference(
        _sweep_mean_series(test, arm, drawn, DECISION_METRIC, bucket=SPARSE_BUCKET),
        _sweep_mean_series(
            test, ALIGNFORMER_IRLS, drawn, DECISION_METRIC, bucket=SPARSE_BUCKET
        ),
    )
    return {
        "reading": "sweep_mean",
        "reading_note": (
            "the brief names no granularity for this clause; see this "
            "function's docstring for the three reasons it is read at the "
            "same granularity as clause 3, and for the per-cell number the "
            "other reading would give"
        ),
        "measured": sweep.to_dict(),
        "sweep_mean_drawn_sigmas_only": sweep_drawn.to_dict(),
        "per_sigma": {key: value.to_dict() for key, value in per_sigma.items()},
        "sigmas_positive": sum(
            1 for value in per_sigma.values() if value.mean > 0.0
        ),
        # The reading that was applied first, and what it would have decided.
        "per_cell_reading": {
            "worst_sigma": worst_sigma,
            "measured": per_sigma[worst_sigma].to_dict(),
            "pass": all(_not_worse(value) for value in per_sigma.values()),
        },
        "required": "mean >= -sem on TEST's 290-frame slice",
        "note": (
            "read on test because validation's sparse slice is 5 frames / 33 "
            "pairs; used ONLY as a guard against regression, never to select"
        ),
        "pass": _not_worse(sweep),
    }


def _freealign_comparison(
    result: Dict, arm: str, sigmas: Sequence[float]
) -> Optional[Dict]:
    """The head-to-head, reported AFTER the rule is frozen and never inside it."""
    if _key(FREEALIGN, sigmas[0]) not in result["ap_by_seed"][_seeds(result)[0]]:
        return None
    report: Dict[str, Dict] = {}
    drawn = _drawn_sigmas(result, sigmas)
    for metric in AP_KEYS:
        difference = paired_difference(
            _sweep_mean_series(result, arm, sigmas, metric),
            _sweep_mean_series(result, FREEALIGN, sigmas, metric),
        )
        drawn_difference = paired_difference(
            _sweep_mean_series(result, arm, drawn, metric),
            _sweep_mean_series(result, FREEALIGN, drawn, metric),
        )
        report[metric] = {
            "sweep_mean": difference.to_dict(),
            # Reported beside it, never instead of it: sigma = 0 is one cell of
            # eight and has no localization error in it at all, so a grid mean
            # whose sign that cell decides is true as computed and misleading as
            # read.
            "sweep_mean_drawn_sigmas_only": drawn_difference.to_dict(),
            "per_sigma": {
                f"sigma_{sigma:g}m": paired_difference(
                    _cell_series(result, arm, sigma, metric),
                    _cell_series(result, FREEALIGN, sigma, metric),
                ).to_dict()
                for sigma in sigmas
            },
        }
    report["sparse_slice_ap_70"] = {
        f"sigma_{sigma:g}m": paired_difference(
            _cell_series(result, arm, sigma, DECISION_METRIC, bucket=SPARSE_BUCKET),
            _cell_series(
                result, FREEALIGN, sigma, DECISION_METRIC, bucket=SPARSE_BUCKET
            ),
        ).to_dict()
        for sigma in sigmas
    }
    report["coverage"] = {
        f"sigma_{sigma:g}m": {
            arm: spread(_coverage_cell(result, arm, sigma)).to_dict(),
            FREEALIGN: spread(_coverage_cell(result, FREEALIGN, sigma)).to_dict(),
        }
        for sigma in sigmas
    }
    return report


def _identity_check(new: Dict, published: Dict) -> Dict:
    """Do the arms that existed before still reproduce the published numbers?

    Everything downstream is a difference against ``alignformer_irls``; if that
    arm moved when the new arms were added to the same run, none of it means
    anything. Compared seed by seed over every condition both files carry, at
    all three IoU thresholds and both sort orders.
    """
    worst, cells, compared = 0.0, 0, []
    seeds = [seed for seed in _seeds(new) if seed in _seeds(published)]
    for seed in seeds:
        old_cells = published["ap_by_seed"][seed]
        for name, old in old_cells.items():
            fresh = new["ap_by_seed"][seed].get(name)
            if fresh is None:
                continue
            if name not in compared:
                compared.append(name)
            for metric in AP_KEYS:
                for sort in (SORT, "frame_order"):
                    worst = max(
                        worst, abs(float(old[metric][sort]) - float(fresh[metric][sort]))
                    )
                    cells += 1
    required = [
        name for name in (ALIGNFORMER, ALIGNFORMER_IRLS)
        if not any(entry.startswith(name + "_sigma") for entry in compared)
    ]
    return {
        "seeds": seeds,
        "conditions_compared": compared,
        "cells_compared": cells,
        "max_abs_difference": worst,
        "deployed_arms_missing_from_the_comparison": required,
        "bit_identical": worst == 0.0 and cells > 0 and not required,
    }


def _enforce_identity(parser, check: Dict, label: str) -> None:
    """A moved baseline invalidates every difference below it.

    Every number this script reports is a difference against
    ``alignformer_irls``. If that arm did not reproduce the published run bit
    for bit, the abstention arms are being compared against something that is
    not the deployed method, and no verdict from this file means anything.
    Computing the check into the JSON and letting a reader find it is not
    enough; it has to stop the report.
    """
    print(
        f"  bit identity vs published {label}: "
        f"{check['cells_compared']} cells, max |diff| = "
        f"{check['max_abs_difference']:.3g}, "
        f"{'IDENTICAL' if check['bit_identical'] else 'MOVED'}",
        flush=True,
    )
    if not check["bit_identical"]:
        parser.error(
            f"the {label} baseline moved (max |difference| = "
            f"{check['max_abs_difference']:.3g}, missing "
            f"{check['deployed_arms_missing_from_the_comparison']}); every "
            "difference in this report is taken against it"
        )


def _reject_one_split_reported_as_two(parser, left: Path, right: Path,
                                      validation: Dict, test: Dict) -> None:
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


def _decide(validation: Dict, sigmas: Sequence[float]) -> Dict:
    """Walk the arms in the brief's order and stop at the first that passes."""
    arms = _abstention_arms(validation)
    if not arms:
        raise ValueError("the validation sweep carries no abstention arms")
    evaluated = []
    chosen = None
    for arm in arms:
        criteria = _validation_criteria(validation, arm["name"], sigmas)
        passed = all(entry["pass"] for entry in criteria.values())
        evaluated.append(
            {"arm": arm, "criteria": criteria, "pass": passed}
        )
        if passed and chosen is None:
            chosen = arm
    return {
        "hypothesis": (
            "H3: replacing the global-tau shrinkage with a per-pair decision "
            "that uses the estimator's OWN variance recovers the low-sigma "
            "cells without giving up the high-sigma wins"
        ),
        "order": (
            "the brief's order -- hard abstain, then per-pair shrinkage, then "
            "both -- is part of the rule: the FIRST arm satisfying every "
            "validation clause is the one that ships, so a better-scoring arm "
            "later in the order does not displace it"
        ),
        "arms": evaluated,
        "chosen": chosen,
        # PROVISIONAL until clause 4 -- test's 1-2 shared slice -- has actually
        # been evaluated. Three of four clauses passing is not the rule, and a
        # verdict that read SHIP after a validation-only run would be
        # indistinguishable from a complete one.
        "clause_4_evaluated": False,
        "verdict": (
            "PROVISIONAL_SHIP_" + chosen["name"].upper()
            if chosen
            else "NULL_KEEP_DEPLOYED_IRLS"
        ),
    }


def _difference_text(difference: Optional[Dict]) -> str:
    if difference is None:
        return f"{'--':>23}"
    if difference["sem"] is None:
        return f"  {difference['mean']:+.4f}          det"
    return (
        f"  {difference['mean']:+.4f}+-{difference['sem']:.4f} "
        f"{difference['verdict']:>5}"
    )


def _print_arm(name: str, table: Dict, means: Dict) -> None:
    print(f"\n== {name} vs {ALIGNFORMER_IRLS} ==", flush=True)
    print(
        f"{'sigma':>5}{'coverage':>10}{'base cov':>10}{'|t| m':>9}{'base |t|':>10}"
        f"{'AP@0.7':>11}{'delta AP@0.7':>25}{'delta AP@0.5':>25}",
        flush=True,
    )
    for key, entry in table.items():
        row = f"{key.replace('sigma_', '').replace('m', ''):>5}"
        row += f"{entry['coverage']['mean']:>10.3f}"
        row += f"{entry['baseline_coverage']['mean']:>10.3f}"
        row += f"{entry['mean_correction_m']['mean']:>9.3f}"
        row += f"{entry['baseline_mean_correction_m']['mean']:>10.3f}"
        row += f"{entry['ap']['ap_70']['mean']:>11.4f}"
        row += _difference_text(entry["ap_minus_baseline"]["ap_70"])
        row += _difference_text(entry["ap_minus_baseline"]["ap_50"])
        print(row, flush=True)
    print(
        f"{'mean':>5}{means['coverage']['mean']:>10.3f}{'':>10}{'':>9}{'':>10}"
        f"{means['ap_70']['value']['mean']:>11.4f}"
        + _difference_text(means["ap_70"]["minus_baseline"])
        + _difference_text(means["ap_50"]["minus_baseline"]),
        flush=True,
    )
    print(
        f"{'drawn':>5}{means['coverage_drawn_sigmas_only']['mean']:>10.3f}"
        f"{'':>10}{'':>9}{'':>10}{'':>11}"
        + _difference_text(means["ap_70"]["minus_baseline_drawn_sigmas_only"])
        + _difference_text(means["ap_50"]["minus_baseline_drawn_sigmas_only"]),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--baseline-validation", type=Path, default=None,
                        help="the published multi-seed run, for the bit-identity gate")
    parser.add_argument("--test", type=Path, default=None)
    parser.add_argument("--baseline-test", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.baseline_test is not None and args.test is None:
        parser.error("--baseline-test is only meaningful with --test")

    validation = json.loads(args.validation.read_text())
    sigmas = [float(sigma) for sigma in validation["sweep_sigmas_m"]]
    test = json.loads(args.test.read_text()) if args.test else None
    if test is not None:
        _reject_one_split_reported_as_two(
            parser, args.validation, args.test, validation, test
        )

    report: Dict = {
        "task": "24_per_pair_abstention",
        "sweep_sigmas_m": sigmas,
        "sources": {
            "validation": {
                "path": str(args.validation),
                "split": validation.get("split"),
                "frames": validation.get("frames"),
                "seeds": validation.get("noise_seeds"),
                "arms": _abstention_arms(validation),
            }
        },
        "bit_identity": {},
        "validation_arms": {},
    }
    _require_error_bars(parser, validation, "validation")
    if args.baseline_validation is not None:
        check = _identity_check(
            validation, json.loads(args.baseline_validation.read_text())
        )
        report["bit_identity"]["validation"] = check
        _enforce_identity(parser, check, "validation")

    for arm in _abstention_arms(validation):
        name = arm["name"]
        report["validation_arms"][name] = {
            "config": arm,
            "per_sigma": _arm_table(validation, name, sigmas),
            "sweep": _sweep_means(validation, name, sigmas),
        }

    report["decision"] = _decide(validation, sigmas)
    chosen = report["decision"]["chosen"]

    print("\n=== VALIDATION: every arm against the deployed IRLS arm ===", flush=True)
    for name, entry in report["validation_arms"].items():
        _print_arm(name, entry["per_sigma"], entry["sweep"])
    print("\n=== pre-registered rule (validation clauses only) ===", flush=True)
    for entry in report["decision"]["arms"]:
        flags = "".join(
            "P" if clause["pass"] else "F" for clause in entry["criteria"].values()
        )
        print(f"  {entry['arm']['name']:32s} {flags}  "
              f"{'PASS' if entry['pass'] else 'fail'}", flush=True)
    print(f"  VERDICT: {report['decision']['verdict']}", flush=True)

    if test is not None:
        report["sources"]["test"] = {
            "path": str(args.test),
            "split": test.get("split"),
            "frames": test.get("frames"),
            "seeds": test.get("noise_seeds"),
            "arms": _abstention_arms(test),
        }
        _require_error_bars(parser, test, "test")
        if args.baseline_test is not None:
            check = _identity_check(
                test, json.loads(args.baseline_test.read_text())
            )
            report["bit_identity"]["test"] = check
            _enforce_identity(parser, check, "test")
        report["test_arms"] = {
            arm["name"]: {
                "config": arm,
                "per_sigma": _arm_table(test, arm["name"], sigmas),
                "sweep": _sweep_means(test, arm["name"], sigmas),
            }
            for arm in _abstention_arms(test)
        }
        print("\n=== TEST: read once, after the rule was frozen ===", flush=True)
        for name, entry in report["test_arms"].items():
            _print_arm(name, entry["per_sigma"], entry["sweep"])

        if chosen is not None and chosen["name"] in report["test_arms"]:
            report["decision"]["test_sparse_guard"] = _sparse_guard(
                test, chosen["name"], sigmas
            )
            report["decision"]["clause_4_evaluated"] = True
            report["decision"]["verdict"] = (
                "SHIP_" + chosen["name"].upper()
                if report["decision"]["test_sparse_guard"]["pass"]
                else "NULL_VETOED_BY_TEST_SPARSE_SLICE"
            )
            guard = report["decision"]["test_sparse_guard"]
            print(
                "\n  clause 4 (test 1-2 shared slice, veto only)",
                flush=True,
            )
            print(
                f"    sweep-mean reading (applied): "
                f"{guard['measured']['mean']:+.4f} +-{guard['measured']['sem']:.4f}"
                f"  positive at {guard['sigmas_positive']}/8  "
                f"{'PASS' if guard['pass'] else 'FAIL'}",
                flush=True,
            )
            print(
                f"    per-cell reading (not applied): worst "
                f"{guard['per_cell_reading']['worst_sigma']} "
                f"{guard['per_cell_reading']['measured']['mean']:+.4f} "
                f"+-{guard['per_cell_reading']['measured']['sem']:.4f}  "
                f"{'PASS' if guard['per_cell_reading']['pass'] else 'FAIL'}",
                flush=True,
            )
            print(f"  FINAL VERDICT: {report['decision']['verdict']}", flush=True)

    # Reported last, and never read by the rule above.
    report["freealign"] = {
        "note": (
            "FreeAlign is reimplemented WITHOUT EdgeGAT, so every margin in our "
            "favour is optimistic by an unmeasured amount and every margin "
            "against us is conservative. Reported after the decision was frozen; "
            "no clause of the rule reads it."
        ),
        "validation": {},
    }
    names = [arm["name"] for arm in _abstention_arms(validation)]
    for name in [ALIGNFORMER_IRLS] + names:
        comparison = _freealign_comparison(validation, name, sigmas)
        if comparison is not None:
            report["freealign"]["validation"][name] = comparison
    if test is not None:
        report["freealign"]["test"] = {}
        for name in [ALIGNFORMER_IRLS] + [
            arm["name"] for arm in _abstention_arms(test)
        ]:
            comparison = _freealign_comparison(test, name, sigmas)
            if comparison is not None:
                report["freealign"]["test"][name] = comparison

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
