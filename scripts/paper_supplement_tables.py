"""Supplement tables of the CVPR draft, written as LaTeX fragments from the result files.

    python scripts/paper_supplement_tables.py --out-dir ../AlignFormer-2027/tables

Per-sigma AP at IoU 0.7 / 0.5 / 0.3 for every stage on both test sets, and the
per-seed paired differences to FreeAlign behind the main text's sweep means.
Rows name the result file and condition they come from; every number is a
mean over the paired noise seeds of that file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

AP_KEYS = (("ap_70", "0.7"), ("ap_50", "0.5"), ("ap_30", "0.3"))
V2XREAL = Path("outputs/v2xreal")
OPV2V = Path("outputs/alignformer/r140")
# (label, result file, condition); the reference row is FreeAlign of the first file
V2XREAL_ROWS = (
    ("uncorrected late fusion", V2XREAL / "B_boxes_only_icprw_test_result.json", "uncorrected"),
    ("oracle (true pose)", V2XREAL / "B_boxes_only_icprw_test_result.json", "oracle"),
    ("FreeAlign (AP-selected, 1.0 m)", V2XREAL / "B_boxes_only_icprw_test_result.json", "freealign"),
    ("propose (Wald 0.2)", V2XREAL / "B_boxes_only_icprw_test_result.json", "alignformer_abstain_0.2"),
    ("+ verify, least squares", V2XREAL / "B_boxes_only_icp_test_result.json", "alignformer_abstain_0.2_icp"),
    ("+ verify, two-point RANSAC", V2XREAL / "B_boxes_only_icpr_test_result.json", "alignformer_abstain_0.2_icpr"),
    ("+ agreement 1.0 m", V2XREAL / "B_boxes_only_icpr_test_result.json", "alignformer_abstain_0.2_agree_1"),
    ("+ consensus floor 4", V2XREAL / "B_boxes_only_icpr_test_result.json", "alignformer_abstain_0.2_agree_1_c4"),
    ("+ weighted refit (AlignFormer)", V2XREAL / "B_boxes_only_icprw_test_result.json", "alignformer_abstain_0.2_agree_1_c4"),
)
OPV2V_ROWS = (
    ("uncorrected late fusion", OPV2V / "icprw_test_result.json", "uncorrected"),
    ("oracle (true pose)", OPV2V / "icprw_test_result.json", "oracle"),
    ("FreeAlign (AP-selected, 1.5 m)", OPV2V / "icprw_test_result.json", "freealign"),
    ("per-pair rule (deployed before)", OPV2V / "icprw_test_result.json", "alignformer_per_pair"),
    ("propose (Wald 0.2)", OPV2V / "icprw_test_result.json", "alignformer_abstain_0.2"),
    ("+ verify, RANSAC, weighted refit", OPV2V / "icprw_test_result.json", "alignformer_abstain_0.2_icprw"),
    ("+ agreement 1.0 m", OPV2V / "icprw_test_result.json", "alignformer_abstain_0.2_agree_1"),
    ("+ consensus floor 4 (AlignFormer)", OPV2V / "icprw_test_result.json", "alignformer_abstain_0.2_agree_1_c4"),
    ("per-pair rule + weighted re-solve (not pre-registered)", OPV2V / "icprw_test_result.json", "alignformer_per_pair_icprw"),
)


def _value(seed: Dict, condition: str, sigma: float, ap_key: str) -> float:
    key = f"{condition}_sigma_{sigma:g}m"
    return float(seed[key if key in seed else condition][ap_key]["global_sorted"])


def seed_sweep_mean(result: Dict, condition: str, ap_key: str, seed: int) -> float:
    seed_block = result["ap_by_seed"][str(seed)]
    return float(np.mean([_value(seed_block, condition, s, ap_key) for s in result["sweep_sigmas_m"]]))


def per_sigma_values(result: Dict, condition: str, ap_key: str) -> np.ndarray:
    """Seed-mean AP per sweep sigma, then the sweep mean appended."""
    table = np.array([[_value(result["ap_by_seed"][str(seed)], condition, s, ap_key) for s in result["sweep_sigmas_m"]]
                      for seed in result["noise_seeds"]])
    per_sigma = table.mean(axis=0)
    return np.append(per_sigma, per_sigma.mean())


def paired_difference(result: Dict, condition: str, reference: Dict, reference_condition: str, ap_key: str
                      ) -> Tuple[np.ndarray, float, float]:
    """Per-seed difference of sweep means, paired by noise seed, with its mean and standard error."""
    seeds = list(result["noise_seeds"])
    if seeds != list(reference["noise_seeds"]):
        raise ValueError(f"seeds differ: {seeds} vs {reference['noise_seeds']}")
    diffs = np.array([seed_sweep_mean(result, condition, ap_key, s) - seed_sweep_mean(reference, reference_condition, ap_key, s)
                      for s in seeds])
    se = float(diffs.std(ddof=1) / np.sqrt(len(seeds))) if len(seeds) > 1 else 0.0
    return diffs, float(diffs.mean()), se


def fmt(value: float, signed: bool = False, digits: int = 4, math: bool = False) -> str:
    """``.4223`` style; ``signed`` adds the sign, ``math`` writes the minus for use inside $...$."""
    text = f"{abs(value):.{digits}f}"
    text = text[1:] if abs(value) < 1.0 else f"{abs(value):.3f}"
    minus = "-" if math else "$-$"
    if signed:
        return (minus if value < 0 else "+") + text
    return (minus if value < 0 else "") + text


def _load(rows: Sequence[Tuple[str, Path, str]]) -> Dict[Path, Dict]:
    return {path: json.loads(path.read_text()) for _, path, _ in rows if path.exists()}


def per_sigma_table(rows: Sequence[Tuple[str, Path, str]], results: Dict[Path, Dict], dataset: str, label: str) -> str:
    sigmas = next(iter(results.values()))["sweep_sigmas_m"]
    header = " & ".join([f"$\\sigma$ (m / deg)"] + [f"{s:g}" for s in sigmas] + ["mean"]) + " \\\\"
    lines = [f"\\begin{{table*}}[t]", "  \\centering\\small",
             f"  \\caption{{\\textbf{{{dataset} test, per-$\\sigma$ AP at IoU 0.7, 0.5 and 0.3}}, mean over the paired noise seeds "
             f"of each row's result file. Rows above the rule are references; below it the stages of AlignFormer in order.}}",
             f"  \\label{{{label}}}", "  \\begin{tabular}{l" + "c" * (len(sigmas) + 1) + "}", "    \\toprule"]
    for ap_key, name in AP_KEYS:
        lines += [f"    \\multicolumn{{{len(sigmas) + 2}}}{{l}}{{\\textbf{{AP@{name}}}}} \\\\", "    " + header, "    \\midrule"]
        for i, (row_label, path, condition) in enumerate(rows):
            if path not in results:
                continue
            values = per_sigma_values(results[path], condition, ap_key)
            cells = " & ".join(fmt(v) for v in values)
            lines.append(f"    {row_label} & {cells} \\\\")
            if i == 2:
                lines.append("    \\midrule")
        lines.append("    \\addlinespace")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table*}"]
    return "\n".join(lines) + "\n"


def per_seed_table(rows: Sequence[Tuple[str, Path, str]], results: Dict[Path, Dict], dataset: str, label: str) -> str:
    """Each row against the FreeAlign condition of its own result file, so device differences between runs cancel."""
    seeds = next(iter(results.values()))["noise_seeds"]
    lines = [f"\\begin{{table*}}[t]", "  \\centering\\small",
             f"  \\caption{{\\textbf{{{dataset} test, per-seed paired differences to FreeAlign}} of the sweep-mean AP: one column per noise "
             f"seed at IoU 0.7, then mean $\\pm$ standard error at the three thresholds. Every row is compared with the FreeAlign "
             f"condition of its own result file, on identical noise draws.}}",
             f"  \\label{{{label}}}", "  \\resizebox{\\textwidth}{!}{\\begin{tabular}{l" + "c" * len(seeds) + "ccc}", "    \\toprule",
             "    row & " + " & ".join(f"seed {s}" for s in seeds) + " & AP@0.7 & AP@0.5 & AP@0.3 \\\\", "    \\midrule"]
    for row_label, path, condition in rows:
        if path not in results or condition == "freealign":
            continue
        diffs, _, _ = paired_difference(results[path], condition, results[path], "freealign", "ap_70")
        stats = []
        for ap_key, _ in AP_KEYS:
            _, mean, se = paired_difference(results[path], condition, results[path], "freealign", ap_key)
            stats.append(f"${fmt(mean, signed=True, math=True)}\\pm{fmt(se)}$")
        lines.append(f"    {row_label} & " + " & ".join(fmt(d, signed=True) for d in diffs) + " & " + " & ".join(stats) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}}", "\\end{table*}"]
    return "\n".join(lines) + "\n"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for dataset, rows, tag in (("V2X-Real", V2XREAL_ROWS, "v2xreal"), ("OPV2V", OPV2V_ROWS, "opv2v")):
        results = _load(rows)
        missing = sorted({str(p) for _, p, _ in rows if p not in results})
        if missing:
            print(f"{dataset}: skipping rows whose file is missing: {missing}")
        (args.out_dir / f"supp_per_sigma_{tag}.tex").write_text(per_sigma_table(rows, results, dataset, f"tab:supp_sigma_{tag}"))
        (args.out_dir / f"supp_per_seed_{tag}.tex").write_text(per_seed_table(rows, results, dataset, f"tab:supp_seed_{tag}"))
        print(f"wrote {args.out_dir}/supp_per_sigma_{tag}.tex and supp_per_seed_{tag}.tex")


if __name__ == "__main__":
    main()
