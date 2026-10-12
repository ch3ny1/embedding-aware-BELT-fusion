"""Delay tables for the paper, from whichever delay result files exist.

Main-text table (tables/delay_main.tex): per dataset and delay, the oracle,
the propose+verify+agree stage, FreeAlign and the selected arm at the three
IoU thresholds, sweep means over sigma with the paired seeds. Supplement
table (tables/supp_delay.tex): AP@0.7 per sigma plus the AP@0.5 / AP@0.3
sweep means.

Each (dataset, delay) cell comes from the selected arm's result file when it
exists, else from the earlier fallback run (V2X-Real: the unweighted exact
refit; OPV2V: the deployed per-pair rule), marked with a dagger; cells with
neither file are written as \\pending so the table can be regenerated as the
runs land.

    python scripts/paper_delay_tables.py --out-dir ../AlignFormer-2027/tables
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_supplement_tables import AP_KEYS, fmt, per_sigma_values

REPO = Path(__file__).resolve().parents[1]
V2XREAL = REPO / "outputs" / "v2xreal"
OPV2V = REPO / "outputs" / "alignformer" / "r140"
DELAY_FRAMES = (1, 2, 4)
FRAME_MS = 100
SELECTED_ARM = "alignformer_abstain_0.2_agree_1_c4"
VERIFY_ARM = "alignformer_abstain_0.2_agree_1"
PENDING = "\\pending"


@dataclass(frozen=True)
class Dataset:
    macro: str
    selected: str
    fallback: str
    fallback_arm: str
    fallback_label: str


@dataclass(frozen=True)
class Source:
    result: dict
    arm: str
    verify: str | None
    is_selected: bool


DATASETS = (
    Dataset("\\vxreal", str(V2XREAL / "B_boxes_only_icprw_delay{d}_test_result.json"),
            str(V2XREAL / "B_boxes_only_icpr_delay{d}_test_result.json"), SELECTED_ARM, "unweighted refit"),
    Dataset("\\opvv", str(OPV2V / "icprw_delay{d}_test_result.json"),
            str(OPV2V / "boxesonly_delay{d}_test_result.json"), "alignformer_per_pair", "per-pair rule"),
)


def _conditions(result: dict) -> set:
    seed = result["ap_by_seed"][str(result["noise_seeds"][0])]
    return {key.split("_sigma_")[0] for key in seed}


def load_source(dataset: Dataset, delay_frames: int) -> Source | None:
    selected = Path(dataset.selected.format(d=delay_frames))
    if selected.exists():
        result = json.loads(selected.read_text())
        return Source(result, SELECTED_ARM, VERIFY_ARM if VERIFY_ARM in _conditions(result) else None, True)
    fallback = Path(dataset.fallback.format(d=delay_frames))
    if not fallback.exists():
        return None
    result = json.loads(fallback.read_text())
    verify = VERIFY_ARM if VERIFY_ARM in _conditions(result) else None
    return Source(result, dataset.fallback_arm, verify, False)


def sweep_mean(source: Source, condition: str, ap_key: str) -> float:
    return float(per_sigma_values(source.result, condition, ap_key)[-1])


def bold_pair(ours: float, theirs: float) -> tuple[str, str]:
    """Format two numbers, bolding the larger one (ties bold neither)."""
    if ours > theirs:
        return f"\\textbf{{{fmt(ours)}}}", fmt(theirs)
    if theirs > ours:
        return fmt(ours), f"\\textbf{{{fmt(theirs)}}}"
    return fmt(ours), fmt(theirs)


def _delay_cell(delay_ms: int) -> str:
    return f"\\multirow{{3}}{{*}}{{\\SI{{{delay_ms}}}{{ms}}}}"


def main_block(source: Source | None, delay_ms: int) -> list[str]:
    """Three lines (AP@0.7 / 0.5 / 0.3) of the main-text table for one dataset and delay."""
    lines = []
    for i, (ap_key, name) in enumerate(AP_KEYS):
        first = _delay_cell(delay_ms) if i == 0 else ""
        if source is None:
            cells = [PENDING] * 4
        else:
            ours, theirs = bold_pair(sweep_mean(source, source.arm, ap_key), sweep_mean(source, "freealign", ap_key))
            verify = fmt(sweep_mean(source, source.verify, ap_key)) if source.verify else PENDING
            dagger = "" if source.is_selected else "$^\\dagger$"
            cells = [fmt(sweep_mean(source, "oracle", ap_key)), verify, theirs, ours + dagger]
        lines.append(f"    {first} & \\ap{{{name}}} & " + " & ".join(cells) + " \\\\")
    return lines


def _per_sigma_cells(source: Source, condition: str) -> list[float]:
    return [float(v) for v in per_sigma_values(source.result, condition, "ap_70")]


def supplement_block(source: Source | None, dataset_macro: str, delay_ms: int, fallback_label: str) -> list[str]:
    """Reference row, FreeAlign row and arm row of the supplement table for one dataset and delay."""
    head = f"    {dataset_macro}, \\SI{{{delay_ms}}}{{ms}}"
    if source is None:
        return [f"{head} & uncorrected / oracle & " + " & ".join([PENDING] * 9) + " \\\\",
                "     & \\freealign & " + " & ".join([PENDING] * 7) + " & & \\\\",
                "     & \\paper & " + " & ".join([PENDING] * 5) + " & & & " + " & ".join([PENDING] * 2) + " \\\\"]
    uncorrected, oracle = _per_sigma_cells(source, "uncorrected"), _per_sigma_cells(source, "oracle")
    reference = [f"{fmt(uncorrected[0])} / {fmt(oracle[0])}"] + [fmt(v) for v in uncorrected[1:-1]]
    ours, theirs = _per_sigma_cells(source, source.arm), _per_sigma_cells(source, "freealign")
    pairs = [bold_pair(a, b) for a, b in zip(ours, theirs)]
    tails = {k: (sweep_mean(source, source.arm, k), sweep_mean(source, "freealign", k)) for k, _ in AP_KEYS[1:]}
    arm_label = "\\paper" if source.is_selected else f"\\paper$^\\dagger$ ({fallback_label})"
    return [f"{head} & uncorrected / oracle & " + " & ".join(reference) + " & & & & & \\\\",
            ("     & \\freealign & " + " & ".join(p[1] for p in pairs) + " & "
             + " & ".join(fmt(v[1]) for v in tails.values()) + " & & \\\\"),
            (f"     & {arm_label} & " + " & ".join(p[0] for p in pairs) + " & & & "
             + " & ".join(fmt(v[0]) for v in tails.values()) + " \\\\")]


def legend(sources: dict[tuple[str, int], Source | None]) -> str:
    """Caption legend for the dagger and pending marks, empty when every cell is the selected arm."""
    marks = []
    if any(source is not None and not source.is_selected for source in sources.values()):
        marks.append(" $^\\dagger$: the selected arm's run is pending and the cell shows the earlier run of the same "
                     "pipeline (\\vxreal: the consensus-floor arm with the unweighted exact refit; \\opvv: the deployed "
                     "per-pair rule, no verify stage).")
    if any(source is None for source in sources.values()):
        marks.append(" \\pending: not yet run.")
    return "".join(marks)



def main_table(sources: dict[tuple[str, int], Source | None]) -> str:
    lines = ["\\begin{table}[t]", "  \\centering\\small",
             ("  \\caption{\\textbf{Test under communication delay}, sweep means over $\\sigma\\in\\{0,0.4,1,2\\}$, "
              "3 paired seeds; propose+verify+agree is the full method before the consensus floor. Last column: "
              "the full method." + legend(sources) + "}"),
             "  \\label{tab:delay}", "  \\resizebox{\\linewidth}{!}{\\begin{tabular}{llcccc}", "    \\toprule",
             "    delay & metric & oracle & propose+verify+agree & \\freealign & \\paper \\\\"]
    for dataset in DATASETS:
        lines += ["    \\midrule", f"    \\multicolumn{{6}}{{l}}{{\\textit{{{dataset.macro}}}}} \\\\"]
        for i, d in enumerate(DELAY_FRAMES):
            lines += (["    \\addlinespace"] if i else []) + main_block(sources[(dataset.macro, d)], d * FRAME_MS)
    lines += ["    \\bottomrule", "  \\end{tabular}}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def supplement_table(sources: dict[tuple[str, int], Source | None]) -> str:
    lines = ["\\begin{table*}[t]", "  \\centering\\small",
             ("  \\caption{\\textbf{Delay, per $\\sigma$, \\ap{0.7}} on test, 3 paired seeds, with the \\ap{0.5} / "
              "\\ap{0.3} sweep means; the selected arm against \\freealign at its val-AP threshold, both from the same "
              "result file." + legend(sources) + "}"),
             "  \\label{tab:supp_delay}", "  \\resizebox{\\textwidth}{!}{\\begin{tabular}{llcccc|c|cc|cc}", "    \\toprule",
             ("    & & \\multicolumn{4}{c|}{\\ap{0.7} at $\\sigma$ =} & & \\multicolumn{2}{c|}{\\freealign} & "
             "\\multicolumn{2}{c}{\\paper} \\\\"),
             "    dataset, delay & method & 0 & 0.4 & 1.0 & 2.0 & mean & \\ap{0.5} & \\ap{0.3} & \\ap{0.5} & \\ap{0.3} \\\\"]
    for dataset in DATASETS:
        lines.append("    \\midrule")
        for d in DELAY_FRAMES:
            lines += supplement_block(sources[(dataset.macro, d)], dataset.macro, d * FRAME_MS, dataset.fallback_label)
    lines += ["    \\bottomrule", "  \\end{tabular}}", "\\end{table*}"]
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sources = {(ds.macro, d): load_source(ds, d) for ds in DATASETS for d in DELAY_FRAMES}
    for (macro, d), source in sources.items():
        state = "missing" if source is None else ("selected arm" if source.is_selected else "fallback run")
        print(f"{macro} delay {d}: {state}")
    (args.out_dir / "delay_main.tex").write_text(main_table(sources))
    (args.out_dir / "supp_delay.tex").write_text(supplement_table(sources))
    print(f"wrote {args.out_dir}/delay_main.tex and supp_delay.tex")


if __name__ == "__main__":
    main()
