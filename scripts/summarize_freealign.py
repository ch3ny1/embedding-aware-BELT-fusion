"""Render the AlignFormer vs FreeAlign head-to-head from the sweep result files.

Both rows come out of ONE ``--metric noisy_ap`` invocation per split, so the
detections, the noise draws, the fusion and the AP are shared by construction
and the only thing that differs between them is the alignment algorithm. This
script does no measurement of its own; it reads those JSONs and prints the
tables the report and ``docs/alignformer_p2.md`` carry, plus a summary JSON.

Three things are printed that a plain AP table would hide:

- **AP@0.3 and AP@0.5**, not just AP@0.7. Those are the only thresholds
  FreeAlign's paper reports, so they are where a reader will look.
- **Coverage** -- the fraction of ego-CAV pairs each method corrected at all.
  FreeAlign discards a message whose common subgraph is too small and
  AlignFormer falls back below ``MIN_MATCH_MASS``; a method that abstains often
  can look good on the pairs it does answer, and pose MAE is therefore reported
  both over all pairs and over answered pairs only, the latter being the
  conditioning FreeAlign's published pose figures use.
- **The shared-object slices** (0, 1-2, 3+). A relative-distance graph is
  structurally degenerate below three shared objects and AlignFormer's heading
  virtual points are not; averaged over the whole split that slice is diluted
  and the difference vanishes into noise.

Usage::

    python scripts/summarize_freealign.py \\
        --test outputs/alignformer/r140/freealign_test_result.json \\
        --validation outputs/alignformer/r140/freealign_val_result.json \\
        --calibration outputs/alignformer/r140/freealign_calibration_result.json \\
        --output outputs/alignformer/r140/freealign_result.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

_CONDITIONS = ("uncorrected", "alignformer", "freealign")
_THRESHOLDS = ("ap_30", "ap_50", "ap_70")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--validation", type=Path, default=None)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _ap(result: Dict, condition: str, sigma: float, threshold: str) -> Optional[float]:
    key = condition if condition == "oracle" else f"{condition}_sigma_{sigma:g}m"
    entry = result["ap"].get(key)
    return None if entry is None else entry[threshold]["global_sorted"]


def _cell(value: Optional[float], width: int = 6) -> str:
    return "--" if value is None else f"{value:.{width - 2}f}"


def _ap_table(result: Dict, threshold: str) -> List[str]:
    sigmas = result["sweep_sigmas_m"]
    lines = [
        f"| sigma (m) | {' | '.join(_CONDITIONS)} | delta (AlignFormer - FreeAlign) |",
        "|---" * (len(_CONDITIONS) + 2) + "|",
    ]
    for sigma in sigmas:
        values = [_ap(result, name, sigma, threshold) for name in _CONDITIONS]
        delta = (
            None if values[1] is None or values[2] is None else values[1] - values[2]
        )
        cells = " | ".join(_cell(v) for v in values)
        sign = "" if delta is None or delta < 0 else "+"
        lines.append(
            f"| {sigma:g} | {cells} | "
            f"{'--' if delta is None else f'{sign}{delta:.4f}'} |"
        )
    return lines


def _pose_table(result: Dict) -> List[str]:
    lines = [
        "| sigma (m) | method | coverage | trans MAE all (m) | trans MAE answered (m) "
        "| yaw MAE all (deg) | yaw MAE answered (deg) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for sigma in result["sweep_sigmas_m"]:
        key = f"sigma_{sigma:g}m"
        for name in ("alignformer", "freealign"):
            stats = result["pose_by_condition"].get(name, {}).get(key)
            if stats is None:
                continue
            lines.append(
                f"| {sigma:g} | {name} | {_cell(stats['coverage'], 5)} "
                f"| {_cell(stats['translation_mae_m'], 6)} "
                f"| {_cell(stats['answered_translation_mae_m'], 6)} "
                f"| {_cell(stats['yaw_mae_deg'], 6)} "
                f"| {_cell(stats['answered_yaw_mae_deg'], 6)} |"
            )
    return lines


def _shared_tables(result: Dict, threshold: str, sigma: float) -> List[str]:
    """AP and coverage inside each shared-object slice -- the decisive row."""
    lines = [
        f"| slice | frames | pairs | {' | '.join(_CONDITIONS)} "
        "| AlignFormer coverage | FreeAlign coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    pairs = result.get("pairs_by_shared_object_bucket", {})
    for bucket in result["shared_object_buckets"]:
        entry = result.get("ap_by_shared_objects", {}).get(bucket)
        if entry is None:
            continue
        values = []
        for name in _CONDITIONS:
            key = name if name == "oracle" else f"{name}_sigma_{sigma:g}m"
            row = entry["conditions"].get(key)
            values.append(None if row is None else row[threshold]["global_sorted"])
        coverage = [
            result["pose_by_condition"]
            .get(name, {})
            .get(f"sigma_{sigma:g}m", {})
            .get(f"{bucket}_coverage")
            for name in ("alignformer", "freealign")
        ]
        lines.append(
            f"| {bucket} | {entry['frames']} | {pairs.get(bucket, 0)} | "
            + " | ".join(_cell(v) for v in values)
            + f" | {_cell(coverage[0], 5)} | {_cell(coverage[1], 5)} |"
        )
    return lines


def _section(result: Dict, label: str) -> List[str]:
    lines = [f"\n## {label} -- {result['frames']} frames, {result['split']}"]
    for threshold in _THRESHOLDS:
        lines.append(f"\n### AP@{threshold.split('_')[1]} (global-sorted)")
        lines += _ap_table(result, threshold)
    lines.append("\n### Pose error and coverage")
    lines += _pose_table(result)
    sigmas = result["sweep_sigmas_m"]
    for sigma in [s for s in (0.0, 1.0, max(sigmas)) if s in sigmas]:
        for threshold in ("ap_50", "ap_70"):
            lines.append(
                f"\n### AP@{threshold.split('_')[1]} by shared-object count, "
                f"at sigma = {sigma:g} m"
            )
            lines += _shared_tables(result, threshold, sigma)
    lines.append(
        f"\nframes with pairs in more than one slice (counted in none): "
        f"{result.get('frames_with_mixed_shared_object_buckets')}"
    )
    return lines


def main() -> None:
    args = parse_args()
    results = {"test": json.loads(args.test.read_text())}
    if args.validation is not None:
        results["validation"] = json.loads(args.validation.read_text())

    lines: List[str] = []
    for label in ("validation", "test"):
        if label in results:
            lines += _section(results[label], label)
    print("\n".join(lines))

    summary = {
        "method": "alignformer_vs_freealign_reimplementation",
        "metric": "noisy_ap_head_to_head",
        "note": (
            "The FreeAlign row is THIS PROJECT'S reimplementation of Lei et al., "
            "ICRA 2024 (arXiv 2405.02965), run on our detections, our noise "
            "sweep and our evaluator. It is not the authors' code and its "
            "numbers must never be quoted as theirs."
        ),
        "freealign_config": results["test"].get("freealign_config"),
        "calibration": (
            None if args.calibration is None
            else json.loads(args.calibration.read_text())["selected"]
        ),
        "sources": {
            label: str(path)
            for label, path in (("test", args.test), ("validation", args.validation))
            if path is not None
        },
        "tables_markdown": "\n".join(lines),
        "splits": {
            label: {
                "frames": result["frames"],
                "sweep_sigmas_m": result["sweep_sigmas_m"],
                "ap": {
                    threshold: {
                        name: [
                            _ap(result, name, sigma, threshold)
                            for sigma in result["sweep_sigmas_m"]
                        ]
                        for name in _CONDITIONS
                    }
                    for threshold in _THRESHOLDS
                },
                "pose_by_condition": result["pose_by_condition"],
                "ap_by_shared_objects": result.get("ap_by_shared_objects"),
                "pairs_by_shared_object_bucket": result.get(
                    "pairs_by_shared_object_bucket"
                ),
            }
            for label, result in results.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
