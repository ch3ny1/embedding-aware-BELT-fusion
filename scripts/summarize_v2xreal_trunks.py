"""The three-trunk comparison on V2X-Real, paired across result files.

Boxes-only, LiDAR embedding and LiDAR+camera are three stage-2 checkpoints,
each swept by its own ``evaluate.py --metric noisy_ap --ap-seeds N`` run. The
runs share their noise draws -- the same config seed, the same seed list, the
same detector and detections -- so a per-seed difference between two FILES is
paired exactly as a per-seed difference between two conditions inside one file
is (:mod:`alignformer.seedstats`), and it is the statistic reported here.

Two families of head-to-head, both ``left - right`` within each seed:

- each trunk against the reference trunk (the shipped boxes-only one), the
  question the comparison exists to answer;
- each trunk against the FreeAlign row of its own file, the question the
  project exists to answer -- or, with ``--freealign-from``, against the
  FreeAlign row of one separate file. FreeAlign depends on the detections
  and the noise draws only, never on a checkpoint, so a re-run with new
  parameters made under the same draws supplies the column for every trunk.

Whatever would break the cross-file pairing -- a different seed list, sigma
grid, split, frame count or detector -- is refused, not warned about.

Usage::

    python scripts/summarize_v2xreal_trunks.py \\
        --trunk boxes_only=outputs/v2xreal/B_boxes_only_test_result.json \\
        --trunk lidar=outputs/v2xreal/B_boxes+embeddings_test_result.json \\
        --trunk lidar_camera=outputs/v2xreal/B_camera_test_result.json \\
        --reference boxes_only --output outputs/v2xreal/trunk_comparison_test_result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embedding_aware_belt_fusion.alignformer.seedstats import (  # noqa: E402
    paired_difference,
    spread,
)
# The per-cell readers live with the first multi-seed summary; they carry the
# rule that a sigma drawn once contributes one value, not one per seed.
from scripts.summarize_seed_error_bars import (  # noqa: E402
    AP_KEYS,
    FREEALIGN,
    _cell_series,
    _seeds,
    _sweep_mean_series,
)

DEFAULT_CONDITION = "alignformer_per_pair"
# Every field two files must agree on before a per-seed difference between
# them is a paired one.
PAIRING_FIELDS = ("noise_seeds", "sweep_sigmas_m", "sigmas_drawn_once_m", "split", "frames", "detector_checkpoint")


def parse_trunks(specs: Sequence[str]) -> Dict[str, Path]:
    """``NAME=PATH`` specs, in order."""
    trunks: Dict[str, Path] = {}
    for spec in specs:
        name, separator, path = spec.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"--trunk takes NAME=PATH, got {spec!r}")
        trunks[name] = Path(path)
    return trunks


_PATH_FIELDS = ("split", "detector_checkpoint")


def _pairing_value(result: dict, field: str, name: str):
    if field not in result:
        raise ValueError(f"{name} has no {field!r}; cannot tell whether it pairs with the others")
    value = result[field]
    # The same split or detector spelled two ways is one split or detector.
    return str(Path(value).resolve()) if field in _PATH_FIELDS and isinstance(value, str) else value


def check_paired(results: Dict[str, dict]) -> None:
    """Refuse two files whose per-seed differences would not be paired."""
    names = list(results)
    first = results[names[0]]
    for name in names[1:]:
        for field in PAIRING_FIELDS:
            mine = _pairing_value(results[name], field, name)
            theirs = _pairing_value(first, field, names[0])
            if mine != theirs:
                raise ValueError(f"{name} and {names[0]} differ in {field}: {mine!r} vs {theirs!r}; not pairable")


def _require_condition(result: dict, condition: str, sigmas: Sequence[float], name: str) -> None:
    key = f"{condition}_sigma_{sigmas[0]:g}m"
    if key not in result["ap_by_seed"][_seeds(result)[0]]:
        raise ValueError(f"{name} has no {condition!r} row ({key} missing from ap_by_seed)")


def _sigma_key(sigma: float) -> str:
    return f"{sigma:g}"


def _versus(left: dict, left_condition: str, right: dict, right_condition: str,
            sigmas: Sequence[float], metric: str) -> dict:
    """``left_condition`` in ``left`` minus ``right_condition`` in ``right``, per sigma and on the sweep mean."""
    per_sigma = {
        _sigma_key(sigma): paired_difference(
            _cell_series(left, left_condition, sigma, metric),
            _cell_series(right, right_condition, sigma, metric),
        ).to_dict()
        for sigma in sigmas
    }
    sweep_mean = paired_difference(
        _sweep_mean_series(left, left_condition, sigmas, metric),
        _sweep_mean_series(right, right_condition, sigmas, metric),
    ).to_dict()
    return {"per_sigma": per_sigma, "sweep_mean": sweep_mean}


def _trunk_rows(result: dict, freealign: dict, condition: str, sigmas: Sequence[float], metric: str) -> dict:
    """One trunk's own cells, and its head-to-head with the FreeAlign row of ``freealign``."""
    per_sigma = {
        _sigma_key(sigma): spread(_cell_series(result, condition, sigma, metric)).to_dict()
        for sigma in sigmas
    }
    sweep_mean = spread(_sweep_mean_series(result, condition, sigmas, metric)).to_dict()
    return {
        "per_sigma": per_sigma,
        "sweep_mean": sweep_mean,
        "minus_freealign": _versus(result, condition, freealign, FREEALIGN, sigmas, metric),
    }


def _freealign_rows(results: Dict[str, dict], freealign: Optional[dict]) -> Dict[str, dict]:
    """The file each trunk's FreeAlign row is read from: its own, or the one separate file."""
    if freealign is None:
        return dict(results)
    check_paired({**results, FREEALIGN: freealign})
    return {name: freealign for name in results}


def summarize(results: Dict[str, dict], reference: str, condition: str, freealign: Optional[dict] = None) -> dict:
    if reference not in results:
        raise ValueError(f"reference {reference!r} is not one of the trunks {list(results)}")
    check_paired(results)
    freealign_rows = _freealign_rows(results, freealign)
    first = results[reference]
    sigmas = [float(sigma) for sigma in first["sweep_sigmas_m"]]
    for name, result in results.items():
        _require_condition(result, condition, sigmas, name)
        _require_condition(freealign_rows[name], FREEALIGN, sigmas, name if freealign is None else FREEALIGN)
    per_metric = {}
    for metric in AP_KEYS:
        per_metric[metric] = {
            "trunks": {
                name: _trunk_rows(result, freealign_rows[name], condition, sigmas, metric)
                for name, result in results.items()
            },
            "minus_reference": {
                name: _versus(result, condition, first, condition, sigmas, metric)
                for name, result in results.items()
                if name != reference
            },
        }
    return {
        "method": "v2xreal_trunk_comparison",
        "condition": condition,
        "reference": reference,
        "sigmas_m": sigmas,
        "seeds": [int(seed) for seed in _seeds(first)],
        "freealign_row": "own file" if freealign is None else "separate file",
        "per_metric": per_metric,
    }


def _sources(trunks: Dict[str, Path], results: Dict[str, dict]) -> dict:
    return {
        name: {
            "path": str(trunks[name]),
            "split": results[name].get("split"),
            "frames": results[name].get("frames"),
            "pose_checkpoint_provenance": results[name].get("pose_checkpoint_provenance"),
        }
        for name in trunks
    }


def _value(cell: dict) -> str:
    """``mean±sd`` of one cell, or the bare mean where there was one draw."""
    return f"{cell['mean']:.4f}" if cell["sd"] is None else f"{cell['mean']:.4f}±{cell['sd']:.4f}"


def _difference(cell: dict) -> str:
    """``mean±sem`` of one paired difference, signed; bare where deterministic."""
    return f"{cell['mean']:+.4f}" if cell["sem"] is None else f"{cell['mean']:+.4f}±{cell['sem']:.4f}"


def _row(label: str, cells: Sequence[str], tail: str) -> str:
    return label.ljust(26) + "".join(f"{cell:>16}" for cell in cells) + f"{tail:>26}"


def _print(summary: dict, metric: str) -> None:
    sigmas = summary["sigmas_m"]
    keys = [_sigma_key(sigma) for sigma in sigmas]
    block = summary["per_metric"][metric]
    print(f"\n{metric}, condition {summary['condition']}, seeds {summary['seeds']}")
    print(_row("trunk", [f"sigma {key}" for key in keys], "sweep mean"))
    for name, rows in block["trunks"].items():
        print(_row(name, [_value(rows["per_sigma"][key]) for key in keys], _value(rows["sweep_mean"])))
        versus = rows["minus_freealign"]
        print(_row("  minus freealign", [_difference(versus["per_sigma"][key]) for key in keys],
                   f"{_difference(versus['sweep_mean'])} {versus['sweep_mean']['verdict']}"))
    for name, versus in block["minus_reference"].items():
        print(_row(f"{name} minus {summary['reference']}",
                   [_difference(versus["per_sigma"][key]) for key in keys],
                   f"{_difference(versus['sweep_mean'])} {versus['sweep_mean']['verdict']}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trunk", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--reference", default=None, help="trunk every other is differenced against; the first if omitted")
    parser.add_argument("--condition", default=DEFAULT_CONDITION)
    parser.add_argument(
        "--freealign-from", type=Path, default=None, metavar="PATH",
        help="a result file whose FreeAlign row replaces the one in every --trunk file; "
             "it must pair with them (same seeds, sigmas, split, frames, detector)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trunks = parse_trunks(args.trunk)
    if len(trunks) < 2:
        parser.error("at least two --trunk entries are needed")
    sources = dict(trunks) if args.freealign_from is None else {**trunks, FREEALIGN: args.freealign_from}
    try:
        loaded = {name: json.loads(path.read_text()) for name, path in sources.items()}
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f"could not read a result file: {error}")
    results = {name: loaded[name] for name in trunks}
    summary = summarize(results, args.reference or next(iter(trunks)), args.condition, loaded.get(FREEALIGN))
    payload = {**summary, "sources": _sources(sources, loaded)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    for metric in AP_KEYS:
        _print(summary, metric)


if __name__ == "__main__":
    main()
