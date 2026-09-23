"""Render the P2 tables from the gate JSONs, so docs/alignformer_p2.md is derived.

Every number in ``docs/alignformer_p2.md`` comes out of this script and out of
``outputs/alignformer/p2_gate.json`` / ``p2_noisy_ap_result.json``. Transcribing them
by hand is how a table and its evidence drift apart.

Usage::

    python scripts/summarize_alignformer_p2.py \\
        --pose outputs/alignformer/p2_gate.json \\
        --noisy-ap outputs/alignformer/p2_noisy_ap_result.json \\
        --history outputs/alignformer/stage2_B_boxes+embeddings/history.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional


def _cell(value: Optional[float], digits: int = 4) -> str:
    """Format a metric, or ``--`` when its subset was empty (written as null)."""
    if value is None or value != value:  # None, or a NaN that escaped sanitizing
        return "--"
    return f"{value:.{digits}f}"


def pose_table(result: Dict) -> str:
    """One row per (sigma, configuration), with the predict-zero baseline beside it."""
    lines = [
        "| sigma (m) | Configuration | Translation MAE (m) | Predict-zero translation (m) "
        "| Yaw MAE (deg) | Predict-zero yaw (deg) | Analytic predict-zero yaw (deg) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for entry in result["results"].values():
        for name, values in entry["configurations"].items():
            mean = values["mean"]
            lines.append(
                f"| {entry['sigma_xy_m']:g} | {name} "
                f"| {_cell(mean['translation_mae_m'])} "
                f"| {_cell(mean['predict_zero_translation_mae_m'])} "
                f"| {_cell(mean['yaw_mae_deg'])} "
                f"| {_cell(mean['predict_zero_yaw_mae_deg'])} "
                f"| {_cell(entry['analytic_predict_zero_yaw_mae_deg'])} |"
            )
    return "\n".join(lines)


def gate_table(result: Dict) -> str:
    gate = result["gate"]
    lines = [
        f"Gate configuration: **{gate.get('configuration')}** -- "
        f"**{'PASS' if gate['passed'] else 'FAIL'}**",
        "",
        "| sigma | Yaw MAE (deg) | Predict-zero yaw (deg) | Below? |",
        "|---|---:|---:|---|",
    ]
    for cell in gate.get("cells", []):
        lines.append(
            f"| {cell['sigma']} | {_cell(cell['yaw_mae_deg'])} "
            f"| {_cell(cell['predict_zero_yaw_mae_deg'])} "
            f"| {'yes' if cell['below'] else 'NO'} |"
        )
    return "\n".join(lines)


def ap_table(result: Dict) -> str:
    """Vanilla / AlignFormer / oracle AP at each sigma, global-sorted."""
    oracle = result["ap"]["oracle"]
    lines = [
        "| sigma (m) | Condition | AP@0.3 | AP@0.5 | AP@0.7 |",
        "|---|---|---:|---:|---:|",
        f"| -- | oracle (true pose) | {oracle['ap_30']['global_sorted']:.4f} "
        f"| {oracle['ap_50']['global_sorted']:.4f} "
        f"| {oracle['ap_70']['global_sorted']:.4f} |",
    ]
    for sigma in result["sweep_sigmas_m"]:
        for label, prefix in (
            ("vanilla late fusion (uncorrected)", "uncorrected"),
            ("AlignFormer-corrected", "alignformer"),
        ):
            key = f"{prefix}_sigma_{sigma:g}m"
            report = result["ap"][key]
            lines.append(
                f"| {sigma:g} | {label} "
                f"| {report['ap_30']['global_sorted']:.4f} "
                f"| {report['ap_50']['global_sorted']:.4f} "
                f"| {report['ap_70']['global_sorted']:.4f} |"
            )
    return "\n".join(lines)


def ap_recovery_table(result: Dict) -> str:
    """How much of the oracle-vs-vanilla gap AlignFormer closes, at AP@0.7."""
    oracle = result["ap"]["oracle"]["ap_70"]["global_sorted"]
    lines = [
        "| sigma (m) | Vanilla AP@0.7 | AlignFormer AP@0.7 | Oracle AP@0.7 "
        "| Gain over vanilla | Gap recovered |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for sigma in result["sweep_sigmas_m"]:
        vanilla = result["ap"][f"uncorrected_sigma_{sigma:g}m"]["ap_70"]["global_sorted"]
        corrected = result["ap"][f"alignformer_sigma_{sigma:g}m"]["ap_70"]["global_sorted"]
        gap = oracle - vanilla
        recovered = (corrected - vanilla) / gap if abs(gap) > 1e-9 else None
        lines.append(
            f"| {sigma:g} | {vanilla:.4f} | {corrected:.4f} | {oracle:.4f} "
            f"| {corrected - vanilla:+.4f} "
            f"| {'--' if recovered is None else f'{recovered:.1%}'} |"
        )
    return "\n".join(lines)


def test_split_pose_table(result: Dict) -> str:
    """Pose error on the OPV2V test split, measured inside the AP sweep."""
    lines = [
        "| sigma (m) | Pairs | Translation MAE (m) | Predict-zero (m) | Yaw MAE (deg) "
        "| Predict-zero (deg) | Unalignable pairs | Fell back |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, entry in result["pose"].items():
        lines.append(
            f"| {key.replace('sigma_', '').replace('m', '')} | {int(entry['pairs'])} "
            f"| {_cell(entry['translation_mae_m'])} "
            f"| {_cell(entry['predict_zero_translation_mae_m'])} "
            f"| {_cell(entry['yaw_mae_deg'])} "
            f"| {_cell(entry['predict_zero_yaw_mae_deg'])} "
            f"| {int(entry['unalignable_pairs'])} "
            f"| {_cell(entry['fallback_fraction'], 3)} |"
        )
    return "\n".join(lines)


def history_table(history: List[Dict]) -> str:
    """The per-epoch stage-2 training curve."""
    lines = [
        "| Epoch | sigma (m) | Train loss | Train corner (m) | Val corner (m) "
        "| Val translation MAE (m) | Val yaw MAE (deg) | Match temperature |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in history:
        temperature = row.get("log_temperature")
        lines.append(
            f"| {row['epoch']} | {row['train_sigma_m']:.3f} | {row['train_loss']:.4f} "
            f"| {row['train_corner_loss_m']:.4f} | {row['corner_loss_m']:.4f} "
            f"| {row['translation_mae_m']:.4f} | {row['yaw_mae_deg']:.4f} "
            f"| {'--' if temperature is None else f'{math.exp(temperature):.5f}'} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose", type=Path, default=None)
    parser.add_argument("--noisy-ap", type=Path, default=None)
    parser.add_argument("--history", type=Path, nargs="*", default=())
    args = parser.parse_args()

    if args.pose is not None:
        result = json.loads(args.pose.read_text())
        print("## P2 gate\n")
        print(gate_table(result))
        print("\n## Pose sweep\n")
        print(pose_table(result))
    if args.noisy_ap is not None:
        result = json.loads(args.noisy_ap.read_text())
        print("\n## Fused AP under localization error\n")
        print(ap_table(result))
        print("\n## Gap recovered at AP@0.7\n")
        print(ap_recovery_table(result))
        print("\n## Pose error on the test split\n")
        print(test_split_pose_table(result))
    for path in args.history:
        print(f"\n## Training curve: {path.parent.name}\n")
        print(history_table(json.loads(path.read_text())))


if __name__ == "__main__":
    main()
