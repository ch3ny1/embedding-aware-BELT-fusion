"""The per-agent detector's ``cav_lidar_range`` and everything that follows it.

Task 17 found that AlignFormer's late-fusion detector ran at OpenCOOD's stock
``point_pillar_late_fusion.yaml`` range of +/-70.4 m in x while every
intermediate-fusion baseline it was compared against ran at +/-140.8 m and the
evaluated ``GT_RANGE`` was +/-140 m for all of them. 10.0% of the evaluated
ground-truth boxes were therefore outside the detector's field of view and
counted as misses in every row of every table.

These tests pin the widened configuration and, more importantly, the two
invariants that make a second configuration necessary rather than optional:
the ROI cache is range-dependent (so the two ranges cannot share one), and the
two configs must differ ONLY in the range and the paths that follow from it.
"""

from pathlib import Path

import pytest
import torch
import yaml

from embedding_aware_belt_fusion.alignformer.embedding import rotated_roi_align

R70_DETECTOR = Path("configs/alignformer_detector.yaml")
R140_DETECTOR = Path("configs/alignformer_detector_r140.yaml")
R70_ALIGNFORMER = Path("configs/alignformer.yaml")
R140_ALIGNFORMER = Path("configs/alignformer_r140.yaml")

# opencood.data_utils.datasets.GT_RANGE -- what every method in the
# head-to-head comparison is scored against.
GT_RANGE = [-140, -40, -3, 140, 40, 1]

# opencood/hypes_yaml/point_pillar_intermediate_fusion.yaml and the six other
# intermediate-fusion baselines' own cav_lidar_range.
BASELINE_RANGE = [-140.8, -40, -3, 140.8, 40, 1]


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def test_roi_features_depend_on_the_lidar_range():
    """The ROI cache is not range-independent, so it cannot be reused.

    ``rotated_roi_align`` maps a box's world coordinates into normalized BEV
    grid coordinates using ``lidar_range``. Widening the range therefore moves
    every box to a different place in the feature map, and a cache built at
    one range is silently wrong at the other. This is the test that says the
    cache must be rebuilt rather than inherited.
    """
    torch.manual_seed(0)
    features = torch.randn(8, 200, 704)
    box = torch.tensor([[30.0, 5.0, 0.0, 1.56, 1.6, 3.9, 0.3]])

    roi_narrow = rotated_roi_align(features, box, [-70.4, -40, -3, 70.4, 40, 1], 4)
    roi_wide = rotated_roi_align(features, box, BASELINE_RANGE, 4)

    assert not torch.allclose(roi_narrow, roi_wide)


def test_the_r140_detector_sees_the_whole_evaluated_ground_truth_range():
    config = _load(R140_DETECTOR)
    lidar_range = config["preprocess"]["cav_lidar_range"]

    assert lidar_range == BASELINE_RANGE, (
        "the widened detector must run at the same range as the "
        "intermediate-fusion baselines it is compared against"
    )
    assert lidar_range[0] <= GT_RANGE[0] and lidar_range[3] >= GT_RANGE[3], (
        "no evaluated ground-truth box may lie outside the detector's field of view"
    )


def test_the_r70_detector_config_is_kept_and_still_narrow():
    """The old configuration is a labelled result, not something to overwrite."""
    assert _load(R70_DETECTOR)["preprocess"]["cav_lidar_range"] == [
        -70.4, -40, -3, 70.4, 40, 1
    ]


def test_the_two_detector_configs_differ_only_in_range_and_paths():
    """Nothing but the range (and what follows it) may drift between them.

    A silent change to the learning rate, the anchor sizes or the voxel
    budget would make the two ranges' results incomparable, which is exactly
    what this re-measurement exists to compare.
    """
    narrow, wide = _load(R70_DETECTOR), _load(R140_DETECTOR)

    # The range itself, and the three places the yaml anchors it to.
    assert narrow["preprocess"]["cav_lidar_range"] != wide["preprocess"]["cav_lidar_range"]
    for config in (narrow, wide):
        anchored = config["preprocess"]["cav_lidar_range"]
        assert config["postprocess"]["anchor_args"]["cav_lidar_range"] == anchored
        assert config["model"]["args"]["lidar_range"] == anchored

    # Everything else, including the voxel budget, must be identical.
    for section in ("train_params", "fusion", "data_augment", "loss", "optimizer", "lr_scheduler"):
        assert narrow[section] == wide[section], f"{section} drifted between the two ranges"
    assert narrow["preprocess"]["args"] == wide["preprocess"]["args"]
    assert narrow["postprocess"]["target_args"] == wide["postprocess"]["target_args"]
    assert narrow["postprocess"]["nms_thresh"] == wide["postprocess"]["nms_thresh"]
    assert narrow["postprocess"]["order"] == wide["postprocess"]["order"]
    # The two split roots are different directories (the +/-70.4 m one lives on
    # an NTFS volume that has stopped round-tripping symlinks; see
    # test_the_two_ranges_train_on_the_same_scenarios), but they must hold the
    # same scenarios -- that is the invariant, not the path.
    assert Path(narrow["root_dir"]).name == Path(wide["root_dir"]).name == "train"
    assert Path(narrow["validate_dir"]).name == Path(wide["validate_dir"]).name == "val"

    # ...but they must not write to, or read from, the same checkpoint.
    assert narrow["detector"]["checkpoint"] != wide["detector"]["checkpoint"]


def test_the_two_ranges_do_not_share_an_roi_cache():
    narrow, wide = _load(R70_ALIGNFORMER), _load(R140_ALIGNFORMER)
    assert narrow["data"]["cache_root"] != wide["data"]["cache_root"], (
        "rotated_roi_align is range-dependent (see "
        "test_roi_features_depend_on_the_lidar_range), so one cache root "
        "cannot serve both ranges"
    )


def test_the_two_ranges_share_the_split_the_comm_range_and_the_model():
    """Only the cache root may differ: the split and the model are the control."""
    narrow, wide = _load(R70_ALIGNFORMER), _load(R140_ALIGNFORMER)

    assert narrow["model"] == wide["model"]
    assert narrow["train"] == wide["train"]
    for key in ("train_root", "test_root", "pair_cache_dir", "val_scenario_fraction",
                "split_seed", "comm_range_m"):
        assert narrow["data"][key] == wide["data"][key], f"data.{key} drifted"


def test_the_pair_index_is_still_the_seed_0_scenario_disjoint_slice():
    data = _load(R140_ALIGNFORMER)["data"]
    assert data["split_seed"] == 0
    assert data["val_scenario_fraction"] == 0.15
    assert data["comm_range_m"] == 70.0


def test_train_late_fusion_writes_to_the_output_directory_it_is_given(tmp_path):
    """The trainer's output directory must be selectable, not hardcoded.

    Both ranges' checkpoints have to coexist: overwriting the +/-70.4 m run
    would destroy the very comparison this re-measurement produces.
    """
    pytest.importorskip("opencood")
    import scripts.train_late_fusion as train_late_fusion
    from opencood.tools import train_utils

    original = train_utils.setup_train
    try:
        destination = tmp_path / "point_pillar_late_fusion_r140"
        train_late_fusion.install_fixed_output_dir(destination)
        returned = train_utils.setup_train({"name": "unit-test"})

        assert Path(returned) == destination
        assert yaml.safe_load((destination / "config.yaml").read_text()) == {"name": "unit-test"}
    finally:
        train_utils.setup_train = original


def test_the_two_ranges_train_on_the_same_scenarios():
    """Same scenario-disjoint split, whichever volume it is materialized on.

    ``scripts/train_late_fusion.py`` builds the split as a directory of
    symlinks into the real OPV2V tree. The repository lives on an NTFS volume
    that has since stopped round-tripping symlinks (they read back as
    ``unsupported reparse tag``), so the +/-140.8 m run's split is
    materialized on the ext4 disk instead. Only the location changed: the
    scenarios are drawn by ``split_scenarios(..., 0.15, seed=0)`` in both
    cases, and this asserts the two directories agree name for name.
    """
    narrow, wide = _load(R70_DETECTOR), _load(R140_DETECTOR)
    for key in ("root_dir", "validate_dir"):
        left, right = Path(narrow[key]), Path(wide[key])
        if not (left.is_dir() and right.is_dir()):
            pytest.skip(f"{key} not materialized on this machine")
        assert {p.name for p in left.iterdir()} == {p.name for p in right.iterdir()}


def test_build_scenario_split_replaces_a_broken_symlink(tmp_path):
    """A dangling link must be replaced, not tripped over.

    ``Path.exists()`` follows symlinks, so a link whose target cannot be read
    -- which is exactly what the NTFS volume now produces -- looks absent and
    ``symlink_to`` then raises ``FileExistsError``. The name-set fast path
    this function used to take would not even look. Left unhandled, a stale
    split directory is unrepairable without deleting it by hand.
    """
    import scripts.train_late_fusion as train_late_fusion
    from embedding_aware_belt_fusion.coloca.train import split_scenarios

    names = [f"scenario_{index:02d}" for index in range(10)]
    train_root = tmp_path / "opv2v_train"
    for name in names:
        (train_root / name).mkdir(parents=True)

    train_names, _ = split_scenarios(
        names, train_late_fusion.VAL_FRACTION, train_late_fusion.SPLIT_SEED
    )
    train_dir, val_dir = tmp_path / "split" / "train", tmp_path / "split" / "val"
    train_dir.mkdir(parents=True)
    dangling = train_dir / train_names[0]
    dangling.symlink_to(tmp_path / "does_not_exist")

    train_late_fusion.build_scenario_split(train_root, train_dir, val_dir)

    assert dangling.is_symlink()
    assert dangling.resolve() == (train_root / train_names[0]).resolve()


def test_the_r140_detector_checkpoint_exists_and_has_detection_heads():
    """The widened checkpoint the re-measurement runs on, asserted not skipped.

    The narrow counterpart is asserted the same way in
    tests/test_alignformer_detector.py; a test that can skip forever is a test
    that never runs.
    """
    torch_checkpoint = Path(_load(R140_DETECTOR)["detector"]["checkpoint"])
    assert torch_checkpoint.exists(), (
        f"detector checkpoint missing: {torch_checkpoint} (see "
        "scripts/train_late_fusion.py --hypes_yaml configs/alignformer_detector_r140.yaml)"
    )

    state = torch.load(torch_checkpoint, map_location="cpu")
    state = state.get("model_state_dict", state)
    keys = set(state)
    assert any(key.startswith("cls_head.") for key in keys)
    assert any(key.startswith("reg_head.") for key in keys)
    assert any(key.startswith("pillar_vfe.") for key in keys)
