"""The split rule, enforced rather than asked.

Two constraints carry this project's headline claim, and until now both lived
only in argparse help text:

- ``OPV2V/validate/`` shares scenario names with ``test/`` on this machine, so
  using it as a held-out set leaks the test split into every constant chosen
  on it. The project's validation set is a scenario-disjoint 15% slice of
  ``train/`` (``split_seed 0``), cached as ``opv2v_splits/val``.
- Calibration and model selection must never read ``test/`` at all.

A rule a caller has to remember is the rule that eventually is not followed,
so these tests pin the refusal, not the documentation.
"""

from pathlib import Path

import pytest

from embedding_aware_belt_fusion.alignformer.splits import (
    LEAKY_SPLIT_NAME,
    resolve_split,
)


def test_the_overlapping_validate_directory_is_refused():
    # On this machine OPV2V/validate is a SYMLINK to OPV2V/test, so a guard
    # that only inspected the resolved path would see "test" and wave it
    # through under allow_test=True -- the exact call that leaks. The name as
    # given has to be refused too.
    with pytest.raises(ValueError, match="overlaps|same directory|validate"):
        resolve_split(
            Path("/media/chenyi/Elements1/Dataset/OPV2V/validate"), allow_test=True
        )


def test_the_refusal_names_the_split_to_use_instead():
    # A guard that only says "no" gets worked around; this one has to say what
    # the caller should have passed.
    with pytest.raises(ValueError) as caught:
        resolve_split(Path("OPV2V") / LEAKY_SPLIT_NAME, allow_test=True)

    message = str(caught.value)
    assert "train" in message and "split_seed" in message


def test_validate_is_refused_anywhere_in_the_path():
    # Not just as the final component: a cache mirror of the leaky split leaks
    # exactly as much as the original.
    with pytest.raises(ValueError, match="overlaps"):
        resolve_split(Path("/cache/opv2v_splits/validate/scenario_7"), allow_test=True)


def test_test_is_refused_when_the_caller_is_selecting():
    with pytest.raises(ValueError, match="never"):
        resolve_split(
            Path("/media/chenyi/Elements1/Dataset/OPV2V/test"), allow_test=False
        )


def test_test_is_allowed_when_the_caller_says_it_is_reporting():
    path = Path("/media/chenyi/Elements1/Dataset/OPV2V/test")

    assert resolve_split(path, allow_test=True) == path.resolve()


def test_the_real_validation_slice_passes_under_both_settings():
    path = Path("/media/chenyi/basement2/cache/opv2v_splits/val")

    assert resolve_split(path, allow_test=False) == path.resolve()
    assert resolve_split(path, allow_test=True) == path.resolve()


def test_the_train_root_passes():
    path = Path("/media/chenyi/Elements1/Dataset/OPV2V/train")

    assert resolve_split(path, allow_test=False) == path.resolve()


def test_allow_test_has_no_default():
    # Keyword-only AND required: a caller that does not say whether it is
    # selecting or reporting is the caller that gets this wrong, so the
    # omission must be a TypeError rather than a permissive default.
    with pytest.raises(TypeError):
        resolve_split(Path("/media/chenyi/Elements1/Dataset/OPV2V/train"))


def test_a_substring_match_is_not_a_component_match():
    # `validate_v2` and `pretest` are ordinary directory names; the rule is
    # about path COMPONENTS, not substrings, or it would refuse innocent paths
    # and teach callers to route around it.
    for name in ("validate_v2", "revalidate", "pretest", "test_mini"):
        path = Path("/cache") / name
        assert resolve_split(path, allow_test=False) == path.resolve()


# ----------------------------------------------------------------------------
# Naming the validation split in a result file
# ----------------------------------------------------------------------------


def test_the_validation_split_description_names_the_carved_slice_by_default():
    from embedding_aware_belt_fusion.alignformer.splits import validation_split_description

    config = {"data": {"train_root": "/d/train", "val_scenario_fraction": 0.15, "split_seed": 0}}

    description = validation_split_description(config)

    assert description == (
        "/d/train :: validation scenarios (scenario-disjoint, val_scenario_fraction=0.15, split_seed=0)"
    )


def test_the_validation_split_description_names_the_official_split_when_configured():
    from embedding_aware_belt_fusion.alignformer.splits import validation_split_description

    config = {"data": {"train_root": "/d/train", "val_root": "/d/val", "val_scenario_fraction": 0.15, "split_seed": 0}}

    description = validation_split_description(config)

    assert description == "/d/val :: the official validation split (data.val_root)"
    assert "scenario-disjoint" not in description
