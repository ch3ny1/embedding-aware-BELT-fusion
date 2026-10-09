"""The pure parts of the supplement table generator: per-seed sweep means,
paired differences between two conditions (possibly from two result files that
share the noise seeds), and the number formatting the tables use."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from paper_supplement_tables import fmt, paired_difference, per_sigma_values, seed_sweep_mean  # noqa: E402


def _result(values_by_seed):
    def ap(value):
        return {"ap_70": {"global_sorted": value, "frame_order": 0.0}}

    seeds = {}
    for seed, (v0, v1) in values_by_seed.items():
        seeds[str(seed)] = {"oracle": ap(0.9), "freealign_sigma_0m": ap(v0), "freealign_sigma_2m": ap(v1),
                            "ours_sigma_0m": ap(v0 + 0.02), "ours_sigma_2m": ap(v1 + 0.04)}
    return {"sweep_sigmas_m": [0.0, 2.0], "noise_seeds": sorted(values_by_seed), "ap_by_seed": seeds}


def test_seed_sweep_mean_averages_the_sigmas_of_one_seed():
    result = _result({0: (0.8, 0.6), 1: (0.82, 0.6)})

    assert seed_sweep_mean(result, "freealign", "ap_70", 0) == 0.7
    assert seed_sweep_mean(result, "oracle", "ap_70", 1) == 0.9


def test_paired_difference_is_per_seed_with_mean_and_standard_error():
    result = _result({0: (0.8, 0.6), 1: (0.82, 0.6)})

    diffs, mean, se = paired_difference(result, "ours", result, "freealign", "ap_70")

    np.testing.assert_allclose(diffs, [0.03, 0.03])
    assert abs(mean - 0.03) < 1e-12
    assert se == 0.0


def test_paired_difference_pairs_seeds_across_two_files():
    ours = _result({0: (0.8, 0.6), 1: (0.9, 0.7)})
    reference = _result({0: (0.8, 0.6), 1: (0.8, 0.6)})

    diffs, mean, _ = paired_difference(ours, "freealign", reference, "freealign", "ap_70")

    np.testing.assert_allclose(diffs, [0.0, 0.1])
    assert abs(mean - 0.05) < 1e-12


def test_per_sigma_values_are_seed_means_with_the_sweep_mean_appended():
    result = _result({0: (0.8, 0.6), 1: (0.82, 0.6)})

    values = per_sigma_values(result, "freealign", "ap_70")

    np.testing.assert_allclose(values, [0.81, 0.6, 0.705])


def test_fmt_drops_the_leading_zero_and_signs_differences():
    assert fmt(0.4223) == ".4223"
    assert fmt(1.0) == "1.000"
    assert fmt(0.0141, signed=True) == "+.0141"
    assert fmt(-0.002, signed=True) == "$-$.0020"
