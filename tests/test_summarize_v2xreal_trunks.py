"""The three-trunk comparison on V2X-Real, paired across result files.

The three trunks (boxes-only, LiDAR embedding, LiDAR+camera) are swept in
three separate ``noisy_ap`` runs. Their noise draws are the same -- same
config seed, same seed list, same detections -- so a per-seed difference
between two files is as paired as one between two conditions inside a file,
and it is the statistic. Anything that would break that pairing (different
seeds, sigmas, split, frame count) has to be a refusal, not a warning.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_v2xreal_trunks import (  # noqa: E402
    DEFAULT_CONDITION,
    check_paired,
    parse_trunks,
    summarize,
)

SIGMAS = (0.0, 1.0)
SEEDS = (0, 1, 2)


def _result(condition_ap, freealign_ap, *, seeds=SEEDS, sigmas=SIGMAS, split="/d/test", frames=10) -> dict:
    """``condition_ap[seed_index]`` is the arm's AP@0.7 under that seed; sigma 0 is seed 0's value."""

    def cell(value):
        return {f"ap_{t}": {"global_sorted": value, "frame_order": value} for t in ("30", "50", "70")}

    by_seed = {}
    for index, seed in enumerate(seeds):
        conditions = {"oracle": cell(0.9)}
        for sigma in sigmas:
            drawn = 0 if sigma == 0.0 else index
            conditions[f"uncorrected_sigma_{sigma:g}m"] = cell(0.5)
            conditions[f"{DEFAULT_CONDITION}_sigma_{sigma:g}m"] = cell(condition_ap[drawn])
            conditions[f"freealign_sigma_{sigma:g}m"] = cell(freealign_ap[drawn])
        by_seed[str(seed)] = conditions
    return {
        "split": split,
        "frames": frames,
        "sweep_sigmas_m": list(sigmas),
        "noise_seeds": list(seeds),
        "sigmas_drawn_once_m": [0.0],
        "detector_checkpoint": "/d/net_epoch13.pth",
        "pose_checkpoint_provenance": {"message_content": "boxes_only"},
        "ap_by_seed": by_seed,
    }


def _three_trunks():
    return {
        "boxes_only": _result([0.80, 0.82, 0.84], [0.79, 0.80, 0.81]),
        "lidar": _result([0.81, 0.83, 0.85], [0.79, 0.80, 0.81]),
        "lidar_camera": _result([0.80, 0.80, 0.80], [0.79, 0.80, 0.81]),
    }


def test_trunk_specs_parse_as_name_equals_path():
    assert parse_trunks(["boxes_only=/a.json", "lidar=/b.json"]) == {
        "boxes_only": Path("/a.json"),
        "lidar": Path("/b.json"),
    }


def test_a_trunk_spec_without_a_name_is_refused():
    with pytest.raises(ValueError, match="NAME=PATH"):
        parse_trunks(["/a.json"])


def test_two_files_with_different_seeds_cannot_be_paired():
    results = {"a": _result([0.8, 0.8, 0.8], [0.8, 0.8, 0.8]), "b": _result([0.8, 0.8], [0.8, 0.8], seeds=(0, 1))}

    with pytest.raises(ValueError, match="noise_seeds"):
        check_paired(results)


def test_two_files_from_different_splits_cannot_be_paired():
    results = {"a": _result([0.8] * 3, [0.8] * 3), "b": _result([0.8] * 3, [0.8] * 3, split="/d/val")}

    with pytest.raises(ValueError, match="split"):
        check_paired(results)


def test_the_difference_to_the_reference_trunk_is_paired_by_seed():
    summary = summarize(_three_trunks(), reference="boxes_only", condition=DEFAULT_CONDITION)

    lidar = summary["per_metric"]["ap_70"]["minus_reference"]["lidar"]
    # +0.01 under every seed at sigma 1: a real, tightly resolved lead.
    assert lidar["per_sigma"]["1"]["mean"] == pytest.approx(0.01)
    assert lidar["per_sigma"]["1"]["sem"] == pytest.approx(0.0, abs=1e-12)
    assert lidar["per_sigma"]["1"]["verdict"] == "left"
    # sigma 0 is drawn once: a difference with no error bar, labelled so.
    assert lidar["per_sigma"]["0"]["mean"] == pytest.approx(0.01)
    assert lidar["per_sigma"]["0"]["verdict"] == "deterministic"
    # The sweep mean pairs seed by seed too: (0.01 + 0.01)/2 under every seed.
    assert lidar["sweep_mean"]["mean"] == pytest.approx(0.01)


def test_a_trunk_that_trades_seeds_with_the_reference_is_a_draw():
    summary = summarize(_three_trunks(), reference="boxes_only", condition=DEFAULT_CONDITION)

    camera = summary["per_metric"]["ap_70"]["minus_reference"]["lidar_camera"]
    # 0.80 - (0.80, 0.82, 0.84) at sigma 1: mean -0.02, sd 0.02, sem 0.0115: draw? No: |−0.02| > 0.0115.
    assert camera["per_sigma"]["1"]["mean"] == pytest.approx(-0.02)
    assert camera["per_sigma"]["1"]["verdict"] == "right"
    # On the sweep mean the deterministic sigma-0 cell (0.80 both) halves it.
    assert camera["sweep_mean"]["mean"] == pytest.approx(-0.01)


def test_the_reference_trunk_is_not_differenced_against_itself():
    summary = summarize(_three_trunks(), reference="boxes_only", condition=DEFAULT_CONDITION)

    assert "boxes_only" not in summary["per_metric"]["ap_70"]["minus_reference"]
    assert summary["reference"] == "boxes_only"


def test_every_trunk_is_differenced_against_freealign_inside_its_own_file():
    summary = summarize(_three_trunks(), reference="boxes_only", condition=DEFAULT_CONDITION)

    boxes = summary["per_metric"]["ap_70"]["trunks"]["boxes_only"]
    # (0.80, 0.82, 0.84) - (0.79, 0.80, 0.81) = (0.01, 0.02, 0.03) at sigma 1.
    assert boxes["minus_freealign"]["per_sigma"]["1"]["mean"] == pytest.approx(0.02)
    assert boxes["minus_freealign"]["per_sigma"]["1"]["per_seed"] == pytest.approx([0.01, 0.02, 0.03])
    assert boxes["per_sigma"]["1"]["mean"] == pytest.approx(0.82)
    assert boxes["per_sigma"]["1"]["n"] == 3
    assert boxes["per_sigma"]["0"]["n"] == 1


def test_an_unknown_reference_is_refused():
    with pytest.raises(ValueError, match="reference"):
        summarize(_three_trunks(), reference="nope", condition=DEFAULT_CONDITION)


def test_the_cli_writes_the_summary_and_records_its_sources(tmp_path):
    paths = {}
    for name, result in _three_trunks().items():
        paths[name] = tmp_path / f"{name}_test_result.json"
        paths[name].write_text(json.dumps(result))
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "summary.json"

    run = subprocess.run(
        [sys.executable, str(root / "scripts" / "summarize_v2xreal_trunks.py"),
         "--trunk", f"boxes_only={paths['boxes_only']}", "--trunk", f"lidar={paths['lidar']}",
         "--trunk", f"lidar_camera={paths['lidar_camera']}", "--reference", "boxes_only",
         "--output", str(out)],
        capture_output=True, text=True, cwd=str(root),
    )

    assert run.returncode == 0, run.stderr
    summary = json.loads(out.read_text())
    assert summary["sources"]["lidar"]["path"] == str(paths["lidar"])
    assert summary["sources"]["lidar"]["split"] == "/d/test"
    assert summary["seeds"] == [0, 1, 2]
    assert "lidar" in run.stdout and "boxes_only" in run.stdout


def test_a_pairing_field_missing_from_both_files_is_refused_not_passed():
    """None == None must not count as agreement: the guard exists to refuse."""
    results = {"a": _result([0.8] * 3, [0.8] * 3), "b": _result([0.8] * 3, [0.8] * 3)}
    for result in results.values():
        del result["sigmas_drawn_once_m"]

    with pytest.raises(ValueError, match="sigmas_drawn_once_m"):
        check_paired(results)


def test_the_same_split_spelled_relative_and_absolute_pairs(tmp_path):
    absolute = str(tmp_path / "test")
    results = {"a": _result([0.8] * 3, [0.8] * 3, split=absolute), "b": _result([0.8] * 3, [0.8] * 3, split=absolute)}
    results["b"]["split"] = str(Path(absolute).parent / "." / "test")

    check_paired(results)  # no raise


def test_a_missing_condition_row_names_the_condition_and_the_trunk():
    results = _three_trunks()

    with pytest.raises(ValueError, match=r"boxes_only has no 'alignformer_irls' row"):
        summarize(results, reference="boxes_only", condition="alignformer_irls")


def test_a_freealign_file_replaces_the_freealign_row_of_every_trunk():
    """FreeAlign depends on the detections and noise draws only, so a re-run
    with new parameters supplies the FreeAlign column for every trunk file."""
    trunks = _three_trunks()
    recalibrated = _result([0.0, 0.0, 0.0], [0.85, 0.85, 0.85])

    summary = summarize(trunks, reference="boxes_only", condition=DEFAULT_CONDITION, freealign=recalibrated)

    for name, expected in (("boxes_only", [-0.05, -0.03, -0.01]), ("lidar", [-0.04, -0.02, 0.0])):
        versus = summary["per_metric"]["ap_70"]["trunks"][name]["minus_freealign"]
        assert versus["per_sigma"]["1"]["per_seed"] == pytest.approx(expected)
    assert summary["freealign_row"] == "separate file"


def test_without_a_freealign_file_the_row_comes_from_each_trunks_own_file():
    summary = summarize(_three_trunks(), reference="boxes_only", condition=DEFAULT_CONDITION)

    assert summary["freealign_row"] == "own file"


def test_a_freealign_file_that_does_not_pair_is_refused():
    recalibrated = _result([0.0] * 2, [0.85] * 2, seeds=(0, 1))

    with pytest.raises(ValueError, match="freealign.*noise_seeds"):
        summarize(_three_trunks(), reference="boxes_only", condition=DEFAULT_CONDITION, freealign=recalibrated)


def test_the_cli_takes_the_freealign_file_and_records_it(tmp_path):
    paths = {}
    for name, result in _three_trunks().items():
        paths[name] = tmp_path / f"{name}_test_result.json"
        paths[name].write_text(json.dumps(result))
    freealign = tmp_path / "recalibrated_test_result.json"
    freealign.write_text(json.dumps(_result([0.0] * 3, [0.85] * 3)))
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "summary.json"

    run = subprocess.run(
        [sys.executable, str(root / "scripts" / "summarize_v2xreal_trunks.py"),
         "--trunk", f"boxes_only={paths['boxes_only']}", "--trunk", f"lidar={paths['lidar']}",
         "--reference", "boxes_only", "--freealign-from", str(freealign), "--output", str(out)],
        capture_output=True, text=True, cwd=str(root),
    )

    assert run.returncode == 0, run.stderr
    summary = json.loads(out.read_text())
    assert summary["sources"]["freealign"]["path"] == str(freealign)
    assert summary["per_metric"]["ap_70"]["trunks"]["lidar"]["minus_freealign"]["per_sigma"]["1"]["mean"] == pytest.approx(-0.02)
