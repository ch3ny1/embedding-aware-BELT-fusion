"""Selecting AlignFormer's decision rule on validation AP.

The same treatment FreeAlign got on V2X-Real: every arm of the per-pair
decision-rule family is swept on the official validation split through the
deployed pipeline, and the one with the best AP@0.7 sweep mean goes to test
beside the deployed per-pair rule. FreeAlign's row is reported for context
and is never a candidate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from select_alignformer_arm import (  # noqa: E402
    arm_spec,
    arms_for_delay,
    arms_for_test,
    main,
    select,
    sweep_rows,
)

SIGMAS = [0.0, 1.0]


def _ap(ap_70: float, ap_50: float = 0.5, ap_30: float = 0.6) -> dict:
    return {m: {"global_sorted": v} for m, v in (("ap_70", ap_70), ("ap_50", ap_50), ("ap_30", ap_30))}


def _result() -> dict:
    # per_pair averages 0.30, abstain 0.2 averages 0.31, freealign 0.35.
    ap = {
        "alignformer_sigma_0m": _ap(0.40),
        "alignformer_sigma_1m": _ap(0.10),
        "alignformer_irls_sigma_0m": _ap(0.40),
        "alignformer_irls_sigma_1m": _ap(0.10),
        "alignformer_per_pair_sigma_0m": _ap(0.38),
        "alignformer_per_pair_sigma_1m": _ap(0.22),
        "alignformer_abstain_0.2_sigma_0m": _ap(0.39),
        "alignformer_abstain_0.2_sigma_1m": _ap(0.23),
        "freealign_sigma_0m": _ap(0.37),
        "freealign_sigma_1m": _ap(0.33),
    }
    pose = {
        "alignformer_per_pair": {"sigma_0m": {"translation_mae_m": 1.0, "coverage": 0.5}, "sigma_1m": {"coverage": 0.7}},
        "alignformer_abstain_0.2": {"sigma_0m": {"translation_mae_m": 0.9, "coverage": 0.3}, "sigma_1m": {"coverage": 0.6}},
    }
    return {
        "split": "val",
        "frames": 7,
        "noise_seeds": [0, 1, 2],
        "sweep_sigmas_m": SIGMAS,
        "conditions": {"abstention": ["alignformer_per_pair", "alignformer_abstain_0.2"]},
        "freealign_calibration": "fa.json",
        "ap": ap,
        "pose_by_condition": pose,
    }


def test_refined_and_agreement_arms_are_candidates_when_the_result_carries_them():
    result = _result()
    result["conditions"]["refined"] = ["alignformer_icp", "alignformer_abstain_0.2_icp"]
    result["conditions"]["agreement"] = ["alignformer_abstain_0.2_agree_0.5"]
    for name, a0, a1 in (("alignformer_icp", 0.40, 0.30), ("alignformer_abstain_0.2_icp", 0.41, 0.31),
                         ("alignformer_abstain_0.2_agree_0.5", 0.42, 0.32)):
        result["ap"][f"{name}_sigma_0m"] = _ap(a0)
        result["ap"][f"{name}_sigma_1m"] = _ap(a1)

    selection = select(result)

    assert selection["chosen"] == "alignformer_abstain_0.2_agree_0.5"
    assert selection["ranked"][:3] == ["alignformer_abstain_0.2_agree_0.5", "alignformer_abstain_0.2_icp", "alignformer_icp"]
    assert arm_spec("alignformer_abstain_0.2_icp") == "abstain:0.2"  # the base decision arm's flag
    assert arm_spec("alignformer_abstain_0.2_agree_0.5") == "abstain:0.2"
    assert arm_spec("alignformer_icp") == "per_pair"


def test_sweep_rows_average_ap_over_the_sigmas_and_carry_the_pose_summary():
    rows = sweep_rows(_result())

    assert rows["alignformer_per_pair"]["ap_70"] == pytest.approx(0.30)
    assert rows["alignformer_per_pair"]["sigma0_ap_70"] == pytest.approx(0.38)
    assert rows["alignformer_per_pair"]["sigma0_mae_m"] == pytest.approx(1.0)
    assert rows["alignformer_per_pair"]["sigma2_coverage"] is None  # no sigma 2 in this sweep
    assert rows["alignformer"]["sigma0_mae_m"] is None  # no pose row recorded


def test_select_ranks_by_ap70_sweep_mean_and_never_picks_freealign():
    selection = select(_result())

    assert selection["ranked"][:2] == ["alignformer_abstain_0.2", "alignformer_per_pair"]
    assert selection["chosen"] == "alignformer_abstain_0.2"
    assert selection["deployed"] == "alignformer_per_pair"
    assert "freealign" not in selection["candidates"]
    assert selection["freealign"]["row"]["ap_70"] == pytest.approx(0.35)
    assert selection["freealign"]["calibration"] == "fa.json"


def test_arm_spec_turns_a_condition_name_back_into_the_cli_spec():
    assert arm_spec("alignformer_per_pair") == "per_pair"
    assert arm_spec("alignformer_abstain_0.2") == "abstain:0.2"
    assert arm_spec("alignformer_both_0.05") == "both:0.05"


def test_test_arms_are_per_pair_plus_the_top_three_and_delay_arms_per_pair_plus_the_chosen():
    ranked = ["alignformer_abstain_0.2", "alignformer_both_0.2", "alignformer_per_pair", "alignformer_abstain_0.1", "alignformer"]

    assert arms_for_test(ranked) == ["per_pair", "abstain:0.2", "both:0.2"]
    assert arms_for_delay("alignformer_abstain_0.2") == ["per_pair", "abstain:0.2"]
    assert arms_for_delay("alignformer_per_pair") == ["per_pair"]
    assert arms_for_delay("alignformer_irls") == ["per_pair"]  # the global rules are not per-pair arms


def test_main_writes_the_selection_file_and_the_two_arm_lists(tmp_path):
    result_path = tmp_path / "arms_val_result.json"
    result_path.write_text(json.dumps(_result()))
    out = tmp_path / "selection.json"

    main(["--result", str(result_path), "--output", str(out), "--arms-dir", str(tmp_path)])

    written = json.loads(out.read_text())
    assert written["chosen"] == "alignformer_abstain_0.2"
    assert (tmp_path / ".arms_for_test").read_text().split() == ["--abstain-arm", "per_pair", "--abstain-arm", "abstain:0.2"]
    assert (tmp_path / ".arms_for_delay").read_text().split() == ["--abstain-arm", "per_pair", "--abstain-arm", "abstain:0.2"]


def test_arm_spec_maps_the_ransac_refined_arms_back_to_their_decision_rule():
    from select_alignformer_arm import arm_spec

    assert arm_spec("alignformer_abstain_0.2_icpr") == arm_spec("alignformer_abstain_0.2")
    assert arm_spec("alignformer_abstain_0.2_icpr_agree_1") == arm_spec("alignformer_abstain_0.2")
