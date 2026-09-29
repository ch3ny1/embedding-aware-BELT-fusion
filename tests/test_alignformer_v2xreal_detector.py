"""The V2X-Real detector config and the training script's dataset switch.

The config is derived from the +/-140.8 m OPV2V detector, and these tests
say exactly which keys may differ and why, the way
``tests/test_alignformer_range.py`` pins the two OPV2V ranges against each
other. Anything else drifting between the two would make the OPV2V-vs-real
contrast a confound rather than a result.

The training script keeps its OPV2V path byte-for-byte: ``--dataset v2xreal``
must skip the OPV2V split materialization and the pcd cache shim (both would
fail loudly on a tree that has no ``.pcd`` files) and route OpenCOOD's trainer
through the V2X-Real builder instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from embedding_aware_belt_fusion.alignformer.v2xreal import (
    CORE_METHOD,
    V2XREAL_COM_RANGE,
    V2XREAL_GT_RANGE,
)

OPV2V = Path("configs/alignformer_detector_r140.yaml")
V2XREAL = Path("configs/v2xreal_detector.yaml")

# Half-extent medians over 918 sampled train frames: Car 2.29 / 1.02 / 0.80.
CENSUS_CAR_LWH = (4.5, 2.0, 1.6)

# Every key that may differ between the two configs, with the reason it does.
ALLOWED_DIFFERENCES = {
    "name",
    "root_dir",
    "validate_dir",
    "v2xreal.gt_range",
    "v2xreal.comm_range",
    "fusion.core_method",
    "preprocess.args.voxel_size",
    "preprocess.args.max_voxel_train",
    "preprocess.args.max_voxel_test",
    "preprocess.cav_lidar_range",
    "postprocess.anchor_args.cav_lidar_range",
    "postprocess.anchor_args.l",
    "postprocess.anchor_args.w",
    "postprocess.anchor_args.h",
    "model.args.voxel_size",
    "model.args.lidar_range",
    "detector.checkpoint",
}


def _load(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _flatten(tree: Any, prefix: str = "") -> Dict[str, Any]:
    if not isinstance(tree, dict):
        return {prefix: tree}
    flat: Dict[str, Any] = {}
    for key, value in tree.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        flat.update(_flatten(value, name))
    return flat


def test_the_two_detector_configs_differ_only_where_documented():
    ours, theirs = _flatten(_load(V2XREAL)), _flatten(_load(OPV2V))

    differing = {k for k in ours.keys() | theirs.keys() if ours.get(k) != theirs.get(k)}

    assert differing == ALLOWED_DIFFERENCES


def test_the_config_uses_the_adapter_and_the_benchmark_ranges():
    config = _load(V2XREAL)

    assert config["fusion"]["core_method"] == CORE_METHOD
    assert config["v2xreal"] == {"gt_range": V2XREAL_GT_RANGE, "comm_range": V2XREAL_COM_RANGE}
    assert config["preprocess"]["cav_lidar_range"] == V2XREAL_GT_RANGE
    assert config["postprocess"]["anchor_args"]["cav_lidar_range"] == V2XREAL_GT_RANGE
    assert config["model"]["args"]["lidar_range"] == V2XREAL_GT_RANGE


def test_one_pillar_layer_spans_the_whole_z_range():
    # z from -15 to 15 is one 30 m pillar; a shorter pillar would silently
    # stack layers the scatter step cannot represent.
    config = _load(V2XREAL)
    z_min, z_max = config["preprocess"]["cav_lidar_range"][2], config["preprocess"]["cav_lidar_range"][5]

    assert config["preprocess"]["args"]["voxel_size"][2] == z_max - z_min
    assert config["model"]["args"]["voxel_size"] == config["preprocess"]["args"]["voxel_size"]


def test_the_anchor_is_a_real_car_not_a_carla_car():
    anchor = _load(V2XREAL)["postprocess"]["anchor_args"]

    assert (anchor["l"], anchor["w"], anchor["h"]) == CENSUS_CAR_LWH


def test_the_voxel_budget_fits_a_128_beam_frame():
    # ~65k points per frame; the OPV2V budget of 16000 pillars truncates.
    args = _load(V2XREAL)["preprocess"]["args"]

    assert args["max_voxel_train"] >= 64000
    assert args["max_voxel_test"] >= args["max_voxel_train"]


def test_the_splits_are_the_official_ones_on_the_nvme_mirror():
    config = _load(V2XREAL)

    assert config["root_dir"] == "/media/chenyi/basement2/dataset/v2x-real/train"
    assert config["validate_dir"] == "/media/chenyi/basement2/dataset/v2x-real/val"


# ----------------------------------------------------------------------------
# The training script
# ----------------------------------------------------------------------------


def test_the_dataset_switch_defaults_to_opv2v(monkeypatch):
    pytest.importorskip("opencood")
    import scripts.train_late_fusion as train_late_fusion

    monkeypatch.setattr("sys.argv", ["train_late_fusion.py"])

    assert train_late_fusion.parse_args().dataset == "opv2v"


def test_the_dataset_switch_accepts_v2xreal_and_nothing_else(monkeypatch):
    pytest.importorskip("opencood")
    import scripts.train_late_fusion as train_late_fusion

    monkeypatch.setattr("sys.argv", ["train_late_fusion.py", "--dataset", "v2xreal"])
    assert train_late_fusion.parse_args().dataset == "v2xreal"

    monkeypatch.setattr("sys.argv", ["train_late_fusion.py", "--dataset", "dair"])
    with pytest.raises(SystemExit):
        train_late_fusion.parse_args()


def test_install_v2xreal_training_routes_opencoods_trainer_through_the_adapter():
    pytest.importorskip("opencood")
    import scripts.train_late_fusion as train_late_fusion
    from opencood.tools import train as opencood_train

    from embedding_aware_belt_fusion.alignformer import v2xreal

    original = opencood_train.build_dataset
    try:
        train_late_fusion.install_v2xreal_training()

        assert opencood_train.build_dataset is v2xreal.build_dataset
    finally:
        opencood_train.build_dataset = original


def test_v2xreal_training_skips_the_opv2v_split_and_pcd_shim(monkeypatch, tmp_path):
    """Neither OPV2V step can run on a tree without .pcd files; both would
    fail before training started. The switch has to bypass them, not
    survive them."""
    pytest.importorskip("opencood")
    import scripts.train_late_fusion as train_late_fusion
    from opencood.tools import train as opencood_train

    monkeypatch.setattr(train_late_fusion, "build_scenario_split", lambda *a, **k: pytest.fail("OPV2V split ran"))
    monkeypatch.setattr(train_late_fusion, "install_pcd_cache_shim", lambda *a, **k: pytest.fail("pcd shim ran"))
    ran = []
    monkeypatch.setattr(opencood_train, "main", lambda: ran.append(True))
    monkeypatch.setattr(
        "sys.argv",
        ["train_late_fusion.py", "--dataset", "v2xreal", "--hypes_yaml", str(V2XREAL), "--output-dir", str(tmp_path / "out")],
    )
    original = opencood_train.build_dataset
    try:
        train_late_fusion.main()
    finally:
        opencood_train.build_dataset = original

    assert ran == [True]
