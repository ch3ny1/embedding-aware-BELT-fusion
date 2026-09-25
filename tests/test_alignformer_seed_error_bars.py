"""Paired error bars over independent noise seeds.

Every AP cell on this branch was one noise draw. The head-to-head against
FreeAlign at AP@0.7 turns on differences of 0.0005-0.011, which one draw cannot
resolve, so the orderings quoted from it were not measurements. This module is
the statistics that replace them.

Two things are load-bearing and each is pinned here:

- the difference between two conditions is **paired** -- both see the same
  perturbed poses, so the per-seed difference is the statistic and its spread
  is far smaller than either condition's own spread. Independent error bars on
  strongly correlated conditions overstate the uncertainty and would hide a
  real ordering as readily as they would invent one;
- a difference no larger than its own standard error is a **draw**, and the
  code must say so in the direction that flatters us as readily as in the one
  that does not.
"""

import math

import pytest

from embedding_aware_belt_fusion.alignformer.seedstats import (
    DETERMINISTIC,
    DRAW,
    PairedDifference,
    Spread,
    paired_difference,
    spread,
    verdict,
)


def test_a_spread_reports_the_mean_and_the_sample_standard_deviation():
    result = spread([0.80, 0.82, 0.84])

    assert result.n == 3
    assert result.mean == pytest.approx(0.82)
    # ddof = 1: three seeds estimate the spread of the draw, they are not it.
    assert result.sd == pytest.approx(0.02)
    assert result.values == (0.80, 0.82, 0.84)


def test_a_single_draw_has_no_spread_rather_than_a_spread_of_zero():
    """sigma = 0 has no noise to draw; a reported 0.0 would be a fake error bar."""
    result = spread([0.8512])

    assert result.n == 1
    assert result.mean == pytest.approx(0.8512)
    assert result.sd is None
    assert result.deterministic


def test_a_spread_needs_at_least_one_value():
    with pytest.raises(ValueError, match="at least one"):
        spread([])


def test_the_paired_difference_is_taken_per_seed_not_between_the_means():
    """The whole point: subtract within a draw, then average the differences."""
    # Two conditions that move together seed for seed -- a large spread each,
    # a tiny spread in their difference. Unpaired bars would call this a draw.
    left = [0.70, 0.80, 0.90]
    right = [0.69, 0.79, 0.89]

    difference = paired_difference(left, right)

    assert difference.n == 3
    assert difference.mean == pytest.approx(0.01)
    assert difference.values == pytest.approx((0.01, 0.01, 0.01))
    assert difference.sem == pytest.approx(0.0, abs=1e-12)
    # And the unpaired reading, which is what this class exists to avoid.
    assert spread(left).sd == pytest.approx(0.1)


def test_the_standard_error_is_the_sample_sd_over_root_n():
    difference = paired_difference([0.5, 0.7, 0.9, 1.1], [0.0, 0.0, 0.0, 0.0])

    values = (0.5, 0.7, 0.9, 1.1)
    mean = sum(values) / 4
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / 3)

    assert difference.mean == pytest.approx(mean)
    assert difference.sem == pytest.approx(sd / math.sqrt(4))


def test_a_paired_difference_over_one_seed_has_no_standard_error():
    difference = paired_difference([0.8512], [0.8461])

    assert difference.n == 1
    assert difference.mean == pytest.approx(0.0051)
    assert difference.sem is None
    assert difference.deterministic


def test_paired_difference_refuses_unequal_or_misaligned_seed_counts():
    with pytest.raises(ValueError, match="same number of seeds"):
        paired_difference([0.1, 0.2], [0.1])


def test_a_difference_inside_its_own_standard_error_is_a_draw():
    # mean 0.002, per-seed values scattered by far more than that.
    difference = paired_difference([0.10, 0.20, 0.30], [0.11, 0.19, 0.29])

    assert difference.sem is not None
    assert abs(difference.mean) <= difference.sem
    assert difference.is_draw
    assert verdict(difference) == DRAW


def test_a_draw_is_called_a_draw_when_the_sign_favours_us_too():
    """The rule is symmetric; it has to be, or it is not a rule."""
    ours_ahead = paired_difference([0.30, 0.20, 0.10], [0.29, 0.19, 0.11])
    ours_behind = paired_difference([0.29, 0.19, 0.11], [0.30, 0.20, 0.10])

    assert ours_ahead.mean > 0 and ours_behind.mean < 0
    assert ours_ahead.is_draw and ours_behind.is_draw
    assert verdict(ours_ahead) == DRAW == verdict(ours_behind)


def test_a_difference_larger_than_its_standard_error_keeps_its_sign():
    ahead = paired_difference([0.30, 0.31, 0.32], [0.20, 0.21, 0.22])
    behind = paired_difference([0.20, 0.21, 0.22], [0.30, 0.31, 0.32])

    assert not ahead.is_draw and not behind.is_draw
    assert verdict(ahead) == "left"
    assert verdict(behind) == "right"


def test_an_exactly_zero_difference_is_a_draw_even_with_zero_spread():
    difference = paired_difference([0.5, 0.5], [0.5, 0.5])

    assert difference.mean == 0.0
    assert difference.sem == pytest.approx(0.0)
    assert difference.is_draw


def test_a_deterministic_difference_is_neither_a_win_nor_a_draw_but_labelled():
    """One run resolves the number exactly and says nothing about its spread."""
    difference = paired_difference([0.8512], [0.8461])

    assert difference.is_draw is None
    assert verdict(difference) == DETERMINISTIC


def test_the_dataclasses_are_frozen_so_a_reported_error_bar_cannot_be_edited():
    difference = paired_difference([0.1, 0.2], [0.0, 0.0])
    with pytest.raises(Exception):
        difference.mean = 0.0  # type: ignore[misc]
    with pytest.raises(Exception):
        spread([0.1, 0.2]).mean = 0.0  # type: ignore[misc]


def test_to_dict_round_trips_the_numbers_a_report_has_to_carry():
    difference = paired_difference([0.30, 0.31, 0.32], [0.20, 0.21, 0.22])
    payload = difference.to_dict()

    assert payload["mean"] == pytest.approx(difference.mean)
    assert payload["sem"] == pytest.approx(difference.sem)
    assert payload["n"] == 3
    assert payload["verdict"] == "left"
    assert payload["per_seed"] == pytest.approx(list(difference.values))

    single = spread([0.8, 0.9]).to_dict()
    assert single["mean"] == pytest.approx(0.85)
    assert single["sd"] == pytest.approx(0.0707106781, rel=1e-6)
    assert single["n"] == 2


def test_a_deterministic_cell_serializes_its_missing_error_bar_as_null():
    assert spread([0.8512]).to_dict()["sd"] is None
    assert paired_difference([0.8512], [0.8461]).to_dict()["sem"] is None


def test_types_are_what_the_report_writer_expects():
    assert isinstance(spread([1.0, 2.0]), Spread)
    assert isinstance(paired_difference([1.0, 2.0], [0.0, 0.0]), PairedDifference)


# --- The split a number came from, recorded and enforced ---------------------
#
# The summarizer takes one result file per split. Passing the same file twice
# produces a report whose "test" rows ARE its "validation" rows, and agreement
# between the two splits is exactly what the AP@0.7 question turns on -- a
# silently aliased run would manufacture that agreement. Every other result
# file in this project records its split; this one has to as well.


def _seed_result(split: str, tag: float) -> dict:
    cell = {
        f"ap_{t}": {"global_sorted": tag, "frame_order": tag}
        for t in ("30", "50", "70")
    }
    conditions = {
        "oracle": cell,
        **{
            f"{name}_sigma_{sigma:g}m": cell
            for name in ("uncorrected", "alignformer", "alignformer_irls", "freealign")
            for sigma in (0.0, 1.0)
        },
    }
    return {
        "split": split,
        "frames": 10,
        "sweep_sigmas_m": [0.0, 1.0],
        "noise_seeds": [0, 1],
        "sigmas_drawn_once_m": [0.0],
        "ap_by_seed": {"0": conditions, "1": conditions},
        "ap_by_shared_objects_by_seed": {
            seed: {"shared_1_2": {"frames": 3, "conditions": conditions}}
            for seed in ("0", "1")
        },
    }


def _run_summarizer(tmp_path, validation: str, test: str):
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [
            sys.executable, str(root / "scripts" / "summarize_seed_error_bars.py"),
            "--validation", validation, "--test", test,
            "--output", str(tmp_path / "out.json"),
        ],
        capture_output=True, text=True, cwd=str(root),
    )


def test_the_same_file_cannot_be_passed_as_both_splits(tmp_path):
    import json

    path = tmp_path / "val_result.json"
    path.write_text(json.dumps(_seed_result("/data/OPV2V/train :: validation", 0.5)))

    result = _run_summarizer(tmp_path, str(path), str(path))

    assert result.returncode != 0
    assert "same file" in (result.stderr + result.stdout)


def test_a_summary_records_the_split_each_side_came_from(tmp_path):
    import json

    validation = tmp_path / "val_result.json"
    test = tmp_path / "test_result.json"
    validation.write_text(json.dumps(_seed_result("/data/OPV2V/train :: val", 0.5)))
    test.write_text(json.dumps(_seed_result("/data/OPV2V/test", 0.4)))

    result = _run_summarizer(tmp_path, str(validation), str(test))
    assert result.returncode == 0, result.stderr

    payload = json.loads((tmp_path / "out.json").read_text())
    assert payload["sources"]["validation"]["split"] == "/data/OPV2V/train :: val"
    assert payload["sources"]["test"]["split"] == "/data/OPV2V/test"
    assert payload["sources"]["validation"]["path"].endswith("val_result.json")
    assert payload["sources"]["test"]["path"].endswith("test_result.json")


def test_a_validation_file_passed_as_the_test_side_is_refused(tmp_path):
    """Two different files can still be the same split; the name says so."""
    import json

    validation = tmp_path / "seeds_val_result.json"
    other = tmp_path / "seeds_val_copy_result.json"
    payload = _seed_result("/data/OPV2V/train :: validation scenarios", 0.5)
    validation.write_text(json.dumps(payload))
    other.write_text(json.dumps(payload))

    result = _run_summarizer(tmp_path, str(validation), str(other))

    assert result.returncode != 0
    assert "same split" in (result.stderr + result.stdout)


# --- How much the pairing actually bought, reported rather than asserted -----
#
# The brief's reason for pairing is that the conditions are strongly correlated
# through the shared draw, so independent bars overstate the uncertainty. That
# is a claim about the data, and on some cells it is worth a factor of two and
# on others almost nothing. Reporting the naive unpaired standard error beside
# the paired one lets a reader see which, instead of taking the claim on faith.


def test_the_unpaired_standard_error_ignores_the_correlation():
    from embedding_aware_belt_fusion.alignformer.seedstats import unpaired_sem

    # Two conditions that move together: a large spread each, none in the
    # difference. This is the case pairing exists for.
    left, right = [0.70, 0.80, 0.90], [0.69, 0.79, 0.89]

    naive = unpaired_sem(left, right)
    paired = paired_difference(left, right).sem

    assert paired == pytest.approx(0.0, abs=1e-12)
    assert naive > 0.05
    assert naive > paired


def test_the_two_agree_when_the_conditions_are_uncorrelated():
    """Pairing is not free significance; where there is no covariance to
    recover, the paired bar is the unpaired one up to the sample size."""
    from embedding_aware_belt_fusion.alignformer.seedstats import unpaired_sem

    # One side constant: Cov = 0, so Var(A - B) = Var(A) exactly.
    left, right = [0.10, 0.20, 0.30, 0.40], [0.5, 0.5, 0.5, 0.5]

    assert unpaired_sem(left, right) == pytest.approx(
        paired_difference(left, right).sem
    )


def test_a_single_draw_has_no_unpaired_standard_error_either():
    from embedding_aware_belt_fusion.alignformer.seedstats import unpaired_sem

    assert unpaired_sem([0.85], [0.84]) is None
