"""The pure parts of the paper's figure scripts: box corners for the BEV
teaser, the AP-against-sigma curve pulled out of a sweep result, and the ECDF
the residual figure draws. The figures themselves are looked at, not tested."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from dump_v2xreal_teaser_frame import box_corners  # noqa: E402
from plot_paper_figures import ap_curve, ecdf, residual_series  # noqa: E402


def test_box_corners_follow_the_h_w_l_yaw_layout():
    # Arrange: [x, y, z, h, w, l, yaw]; a 4 m long, 2 m wide box at (1, 2), then the same box turned 90 degrees
    boxes = np.array([[1.0, 2.0, 0.0, 1.5, 2.0, 4.0, 0.0], [1.0, 2.0, 0.0, 1.5, 2.0, 4.0, np.pi / 2]])

    corners = box_corners(boxes)

    assert corners.shape == (2, 4, 2)
    np.testing.assert_allclose(sorted(set(np.round(corners[0, :, 0], 6))), [-1.0, 3.0])
    np.testing.assert_allclose(sorted(set(np.round(corners[0, :, 1], 6))), [1.0, 3.0])
    np.testing.assert_allclose(sorted(set(np.round(corners[1, :, 0], 6))), [0.0, 2.0])
    np.testing.assert_allclose(sorted(set(np.round(corners[1, :, 1], 6))), [0.0, 4.0])


def _sweep_result():
    def ap(value):
        return {"ap_70": {"global_sorted": value, "frame_order": 0.0}, "ap_50": {"global_sorted": value + 0.1, "frame_order": 0.0}}

    return {
        "sweep_sigmas_m": [0.0, 1.0, 2.0],
        "noise_seeds": [0, 1],
        "ap_by_seed": {
            "0": {"oracle": ap(0.9), "freealign_sigma_0m": ap(0.8), "freealign_sigma_1m": ap(0.7), "freealign_sigma_2m": ap(0.6)},
            "1": {"oracle": ap(0.9), "freealign_sigma_0m": ap(0.82), "freealign_sigma_1m": ap(0.68), "freealign_sigma_2m": ap(0.6)},
        },
    }


def test_ap_curve_averages_the_seeds_per_sigma_and_reports_their_standard_error():
    sigmas, mean, se = ap_curve(_sweep_result(), "freealign", "ap_70")

    np.testing.assert_allclose(sigmas, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(mean, [0.81, 0.69, 0.6])
    np.testing.assert_allclose(se, [0.01, 0.01, 0.0])


def test_ap_curve_repeats_a_sigma_free_condition_across_the_sweep():
    _, mean, se = ap_curve(_sweep_result(), "oracle", "ap_50")

    np.testing.assert_allclose(mean, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(se, [0.0, 0.0, 0.0])


def test_ecdf_is_a_step_from_zero_to_one_over_the_sorted_values():
    x, y = ecdf(np.array([3.0, 1.0, 2.0]))

    np.testing.assert_allclose(x, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(y, [1 / 3, 2 / 3, 1.0])


def test_residual_series_collects_each_kind_over_the_pairs_that_have_it():
    diagnosis = {"pairs": {"2": [
        {"residual_irls": 0.5, "residual_icp": 0.4, "residual_freealign": 3.0},
        {"residual_irls": 0.6, "residual_icp": 0.2},
    ]}}

    series = residual_series(diagnosis, "2")

    np.testing.assert_allclose(series["residual_irls"], [0.5, 0.6])
    np.testing.assert_allclose(series["residual_freealign"], [3.0])
    assert "residual_oracle_pairs_exact" not in series
