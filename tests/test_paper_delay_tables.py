"""The pure parts of the delay table generator: picking the selected-arm file
over the fallback file, the pending mark when neither exists, the dagger on a
fallback row, and bold on the better of FreeAlign and ours."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from paper_delay_tables import Source, bold_pair, main_block, supplement_block

SIGMAS = (0.0, 2.0)


def _result(ours: float, freealign: float, oracle: float = 0.9):
    def ap(value):
        return {k: {"global_sorted": value} for k in ("ap_70", "ap_50", "ap_30")}

    seed = {"oracle": ap(oracle), "uncorrected_sigma_0m": ap(oracle), "uncorrected_sigma_2m": ap(0.1),
            "freealign_sigma_0m": ap(freealign), "freealign_sigma_2m": ap(freealign),
            "arm_sigma_0m": ap(ours), "arm_sigma_2m": ap(ours), "verify_sigma_0m": ap(ours - 0.01),
            "verify_sigma_2m": ap(ours - 0.01)}
    return {"sweep_sigmas_m": list(SIGMAS), "noise_seeds": [0], "ap_by_seed": {"0": seed}}


def test_bold_pair_marks_the_larger_value_only():
    # Arrange / Act
    ours, theirs = bold_pair(0.30, 0.28)

    # Assert
    assert ours == "\\textbf{.3000}"
    assert theirs == ".2800"


def test_main_block_uses_the_selected_arm_without_a_dagger():
    # Arrange
    source = Source(_result(0.30, 0.28), arm="arm", verify="verify", is_selected=True)

    # Act
    lines = main_block(source, delay_ms=100)

    # Assert
    assert len(lines) == 3
    assert "\\textbf{.3000}" in lines[0] and "dagger" not in lines[0] and ".2900" in lines[0]


def test_main_block_marks_a_fallback_arm_with_a_dagger_and_missing_verify_as_pending():
    # Arrange
    source = Source(_result(0.30, 0.28), arm="arm", verify=None, is_selected=False)

    # Act
    lines = main_block(source, delay_ms=200)

    # Assert
    assert "$^\\dagger$" in lines[0] and "\\pending" in lines[0]


def test_main_block_without_a_source_is_all_pending():
    # Act
    lines = main_block(None, delay_ms=400)

    # Assert
    assert len(lines) == 3 and all(line.count("\\pending") == 4 for line in lines)


def test_supplement_block_has_reference_freealign_and_arm_rows():
    # Arrange
    source = Source(_result(0.30, 0.28), arm="arm", verify="verify", is_selected=True)

    # Act
    lines = supplement_block(source, "\\vxreal", delay_ms=100, fallback_label="unweighted refit")

    # Assert
    assert len(lines) == 3
    assert ".9000 / .9000" in lines[0]
    assert lines[1].lstrip().startswith("& \\freealign")
    assert "\\paper &" in lines[2] and "dagger" not in lines[2]


def test_supplement_block_without_a_source_is_pending():
    # Act
    lines = supplement_block(None, "\\opvv", delay_ms=400, fallback_label="per-pair rule")

    # Assert
    assert len(lines) == 3 and "\\pending" in lines[2]
