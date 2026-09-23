"""Render the head-to-head comparison tables from the measured JSONs.

Every number in the comparison section of ``docs/alignformer_p2.md`` comes out
of this script, from ``outputs/alignformer/baselines/*_result.json`` (the
baselines, run locally under this project's own sweep),
``outputs/alignformer/r70/p2_r70_noisy_ap_result.json`` (AlignFormer) and
``outputs/alignformer/baselines/bandwidth_result.json`` (bytes). Transcribing
them by hand is how a table and its evidence drift apart.

Usage::

    python scripts/summarize_alignformer_baselines.py \\
        --alignformer outputs/alignformer/r70/p2_r70_noisy_ap_result.json \\
        --baselines outputs/alignformer/baselines/*_result.json \\
        --bandwidth outputs/alignformer/baselines/bandwidth_result.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# The order the comparison is argued in: the robustness SOTA first, then the
# method designed for pose error, then the rest.
ORDER = ["v2xvit", "coalign", "cobevt", "attfuse", "where2comm", "fcooper", "v2vam"]
SIGMAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]

# The baselines are scored under the LATE-FUSION ground-truth convention,
# because that is the convention AlignFormer's numbers are on and a comparison
# needs one ground truth. The other convention is reported separately.
PRIMARY_CONVENTION = "late_fusion"


def sigma_key(sigma: float) -> str:
    return "sigma_{:g}m".format(sigma)


def _cell(value: Optional[float], digits: int = 4) -> str:
    if value is None or value != value:
        return "--"
    return "{:.{d}f}".format(value, d=digits)


def baseline_ap(result: Dict, sigma: float, threshold: str, convention: str) -> Optional[float]:
    table = result["ap"].get(convention)
    if table is None:
        return None
    cell = table.get(sigma_key(sigma))
    return None if cell is None else cell[threshold]["global_sorted"]


def alignformer_ap(result: Dict, sigma: float, threshold: str, condition: str) -> Optional[float]:
    cell = result["ap"].get("{}_{}".format(condition, sigma_key(sigma)))
    return None if cell is None else cell[threshold]["global_sorted"]


def head_to_head(alignformer: Dict, baselines: Dict[str, Dict], threshold: str) -> str:
    """One row per sigma, one column per method, at a single IoU threshold."""
    names = [name for name in ORDER if name in baselines]
    header = ["sigma (m)", "**AlignFormer**", "late fusion (uncorrected)"]
    header += [baselines[name]["label"] for name in names]
    lines = [
        "| " + " | ".join(header) + " |",
        "|---" + "|---:" * (len(header) - 1) + "|",
    ]
    for sigma in SIGMAS:
        row = [
            "{:g}".format(sigma),
            "**" + _cell(alignformer_ap(alignformer, sigma, threshold, "alignformer")) + "**",
            _cell(alignformer_ap(alignformer, sigma, threshold, "uncorrected")),
        ]
        row += [
            _cell(baseline_ap(baselines[name], sigma, threshold, PRIMARY_CONVENTION))
            for name in names
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def retention(alignformer: Dict, baselines: Dict[str, Dict], threshold: str) -> str:
    """AP retained relative to each method's own clean (sigma = 0) score."""
    names = [name for name in ORDER if name in baselines]
    header = ["sigma (m)", "**AlignFormer**"] + [baselines[n]["label"] for n in names]
    lines = [
        "| " + " | ".join(header) + " |",
        "|---" + "|---:" * (len(header) - 1) + "|",
    ]
    clean = {"AlignFormer": alignformer_ap(alignformer, 0.0, threshold, "alignformer")}
    for name in names:
        clean[name] = baseline_ap(baselines[name], 0.0, threshold, PRIMARY_CONVENTION)
    for sigma in SIGMAS[1:]:
        row = ["{:g}".format(sigma)]
        value = alignformer_ap(alignformer, sigma, threshold, "alignformer")
        base = clean["AlignFormer"]
        row.append("**{}**".format(_percent(value, base)))
        for name in names:
            row.append(
                _percent(
                    baseline_ap(baselines[name], sigma, threshold, PRIMARY_CONVENTION),
                    clean[name],
                )
            )
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _percent(value: Optional[float], base: Optional[float]) -> str:
    if value is None or base in (None, 0):
        return "--"
    return "{:.0f}%".format(100.0 * value / base)


def all_thresholds(alignformer: Dict, baselines: Dict[str, Dict]) -> str:
    """Every method at every sigma and every IoU threshold."""
    lines = [
        "| Method | sigma (m) | AP@0.3 | AP@0.5 | AP@0.7 |",
        "|---|---:|---:|---:|---:|",
    ]
    for sigma in SIGMAS:
        lines.append(
            "| AlignFormer | {:g} | {} | {} | {} |".format(
                sigma,
                _cell(alignformer_ap(alignformer, sigma, "ap_30", "alignformer")),
                _cell(alignformer_ap(alignformer, sigma, "ap_50", "alignformer")),
                _cell(alignformer_ap(alignformer, sigma, "ap_70", "alignformer")),
            )
        )
    for name in ORDER:
        if name not in baselines:
            continue
        result = baselines[name]
        for sigma in SIGMAS:
            lines.append(
                "| {} | {:g} | {} | {} | {} |".format(
                    result["label"],
                    sigma,
                    _cell(baseline_ap(result, sigma, "ap_30", PRIMARY_CONVENTION)),
                    _cell(baseline_ap(result, sigma, "ap_50", PRIMARY_CONVENTION)),
                    _cell(baseline_ap(result, sigma, "ap_70", PRIMARY_CONVENTION)),
                )
            )
    return "\n".join(lines)


def convention_sensitivity(baselines: Dict[str, Dict]) -> str:
    """How much the choice of ground-truth convention is worth, per baseline.

    OpenCOOD builds late-fusion and intermediate-fusion ground truth against
    ranges anchored to different agents, so the two admit different object sets
    on turned or sloped frames. The comparison uses one convention; this table
    is how large a thumb that puts on the scale.
    """
    lines = [
        "| Baseline | frames where the two GT sets differ | AP@0.7 sigma 0 (late-fusion GT) "
        "| AP@0.7 sigma 0 (native GT) | AP@0.7 sigma 1 (late-fusion GT) | AP@0.7 sigma 1 (native GT) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ORDER:
        if name not in baselines:
            continue
        result = baselines[name]
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                result["label"],
                result["run"].get("frames_where_gt_conventions_differ", "--"),
                _cell(baseline_ap(result, 0.0, "ap_70", "late_fusion")),
                _cell(baseline_ap(result, 0.0, "ap_70", "native")),
                _cell(baseline_ap(result, 1.0, "ap_70", "late_fusion")),
                _cell(baseline_ap(result, 1.0, "ap_70", "native")),
            )
        )
    return "\n".join(lines)


def convention_crossover(conventions: Dict, baselines: Dict[str, Dict]) -> str:
    """The head-to-head at AP@0.7 under *both* ground-truth conventions.

    The crossover point is the number the whole comparison turns on, so it is
    reported under each convention rather than under whichever one was picked.
    ``conventions`` is the AlignFormer sweep re-run to emit both.
    """
    names = [name for name in ORDER if name in baselines]
    header = ["sigma (m)", "AlignFormer (late-fusion GT)", "AlignFormer (intermediate GT)"]
    header += ["{} (late-fusion GT)".format(baselines[n]["label"]) for n in names[:1]]
    header += ["{} (intermediate GT)".format(baselines[n]["label"]) for n in names[:1]]
    lines = [
        "| " + " | ".join(header) + " |",
        "|---" + "|---:" * (len(header) - 1) + "|",
    ]
    for sigma in SIGMAS:
        row = [
            "{:g}".format(sigma),
            _cell(alignformer_ap(conventions, sigma, "ap_70", "alignformer")),
            _cell(
                conventions["ap_intermediate_convention_gt"][
                    "alignformer_{}".format(sigma_key(sigma))
                ]["ap_70"]["global_sorted"]
            ),
        ]
        for name in names[:1]:
            row.append(_cell(baseline_ap(baselines[name], sigma, "ap_70", "late_fusion")))
            row.append(_cell(baseline_ap(baselines[name], sigma, "ap_70", "native")))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def bandwidth_table(bandwidth: Dict, baselines: Dict[str, Dict]) -> str:
    """Bytes per frame per agent, measured, with the ratio to AlignFormer."""
    align = bandwidth["alignformer"]
    align_bytes = align["bytes_per_frame_per_agent"]
    lines = [
        "| Method | What crosses the wire | Bytes / frame / agent | vs AlignFormer |",
        "|---|---|---:|---:|",
        "| **AlignFormer** | {:.2f} boxes+embeddings x {} B ({} box floats + {} embedding floats, float32) | "
        "**{:,.0f}** | 1x |".format(
            align["objects_per_agent_mean"],
            align["bytes_per_object"],
            align["box_floats"],
            align["embed_dim"],
            align_bytes,
        ),
        "| AlignFormer, boxes only | {:.2f} boxes x {} B | {:,.0f} | {:.2f}x |".format(
            align["objects_per_agent_mean"],
            align["box_floats"] * 4,
            align["bytes_per_frame_per_agent_boxes_only"],
            align["bytes_per_frame_per_agent_boxes_only"] / align_bytes,
        ),
    ]
    for name in ORDER:
        entry = bandwidth["baselines"].get(name)
        if entry is None:
            continue
        label = baselines[name]["label"] if name in baselines else entry["label"]
        shape = "x".join(str(d) for d in entry["tensor_shape"][1:])
        lines.append(
            "| {} | BEV feature {} float32 | {:,.0f} | {:,.0f}x |".format(
                label, shape, entry["bytes_per_frame_per_agent"],
                entry["bytes_per_frame_per_agent"] / align_bytes,
            )
        )
        if "bytes_per_frame_per_agent_after_selection" in entry:
            lines.append(
                "| {}, after its own communication mask | selected cells only "
                "(rate {:.3f}) | {:,.0f} | {:,.0f}x |".format(
                    label,
                    entry["communication_rate"],
                    entry["bytes_per_frame_per_agent_after_selection"],
                    entry["bytes_per_frame_per_agent_after_selection"] / align_bytes,
                )
            )
    return "\n".join(lines)


def crossover(alignformer: Dict, baselines: Dict[str, Dict], threshold: str = "ap_70") -> str:
    """Where AlignFormer overtakes each baseline, stated plainly in both directions."""
    lines = [
        "| Baseline | sigma where AlignFormer overtakes it | sigmas where the baseline wins |",
        "|---|---|---|",
    ]
    for name in ORDER:
        if name not in baselines:
            continue
        result = baselines[name]
        wins, first_overtake = [], None
        for sigma in SIGMAS:
            ours = alignformer_ap(alignformer, sigma, threshold, "alignformer")
            theirs = baseline_ap(result, sigma, threshold, PRIMARY_CONVENTION)
            if ours is None or theirs is None:
                continue
            if theirs >= ours:
                wins.append("{:g}".format(sigma))
            elif first_overtake is None:
                first_overtake = sigma
        lines.append(
            "| {} | {} | {} |".format(
                result["label"],
                "never" if first_overtake is None else "{:g} m".format(first_overtake),
                ", ".join(wins) if wins else "none",
            )
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignformer", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, nargs="+", required=True)
    parser.add_argument("--bandwidth", type=Path, default=None)
    parser.add_argument(
        "--alignformer-gt-conventions", type=Path, default=None,
        help="the AlignFormer sweep re-run to emit both ground-truth conventions",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alignformer = json.loads(args.alignformer.read_text())
    baselines: Dict[str, Dict] = {}
    for path in args.baselines:
        result = json.loads(path.read_text())
        baselines[result["method"]] = result

    print("## Head-to-head: AP@0.7 under the localization-noise sweep\n")
    print(head_to_head(alignformer, baselines, "ap_70"))
    print("\n## Where AlignFormer overtakes each baseline (AP@0.7)\n")
    print(crossover(alignformer, baselines))
    print("\n## Retention: AP@0.7 as a fraction of each method's own clean score\n")
    print(retention(alignformer, baselines, "ap_70"))
    if args.bandwidth is not None:
        print("\n## Bytes per frame per agent\n")
        print(bandwidth_table(json.loads(args.bandwidth.read_text()), baselines))
    print("\n## Every method, every threshold\n")
    print(all_thresholds(alignformer, baselines))
    print("\n## Ground-truth convention sensitivity\n")
    print(convention_sensitivity(baselines))
    if args.alignformer_gt_conventions is not None:
        print("\n## The crossover under each ground-truth convention\n")
        print(
            convention_crossover(
                json.loads(args.alignformer_gt_conventions.read_text()), baselines
            )
        )
    print("\n## Provenance\n")
    print("| Method | Checkpoint | Trained with pose noise? | Notes |")
    print("|---|---|---|---|")
    for name in ORDER:
        if name not in baselines:
            continue
        result = baselines[name]
        wild = result.get("disabled_wild_setting") or {}
        noise = "yes -- " + str({k: wild[k] for k in ("loc_err", "xyz_std", "ryp_std") if k in wild}) \
            if wild.get("loc_err") else "no"
        print(
            "| {} | `{}` | {} | {} |".format(
                result["label"], Path(result["checkpoint"]).name, noise, result["notes"]
            )
        )


if __name__ == "__main__":
    main()
