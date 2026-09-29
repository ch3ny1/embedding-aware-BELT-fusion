"""The V2X-Real dataset adapter: OpenCOOD's late-fusion dataset on real traffic.

Why an adapter and not a fork
-----------------------------

V2X-Real was recorded by the lab that wrote OpenCOOD and its yaml is the
OpenCOOD yaml, so ``LateFusionDataset`` already does almost everything. The
three things it does not do are pinned here:

1. The LiDAR is ``<ts>.bin`` (float32 x, y, z, intensity), not ``<ts>.pcd``.
   ``basedataset`` builds the ``.pcd`` name unconditionally and reads it
   through ``opencood.utils.pcd_utils.pcd_to_np``, the one slot the training
   script already patches for its NVMe cache. The adapter patches the same
   slot. Occasional NaN rows exist in real files and must be dropped.
2. The ``vehicles`` block holds pedestrians, riders, trash cans and buses.
   OpenCOOD's ``project_world_objects`` filters by nothing but position, so
   without a class filter a pedestrian would be a ground-truth vehicle. The
   filter is the V2X-Real ``vehicle`` super-class exactly as GenComm defines
   it: Car, PoliceCar, LongVehicle. Trucks and buses are a different
   super-class with a different anchor and are excluded on purpose.
3. Two infrastructure agents per scenario. ``basedataset`` rotates ONE
   negative id to the end of the list, so with folders ``-1, -2, 1, 2`` its
   ego would be ``-2``, a roadside unit. The vehicle-centric protocol puts
   vehicles first and the lowest-numbered vehicle is ego.

Everything here builds new objects; nothing mutates the yaml dict OpenCOOD
hands over, because OpenCOOD keeps references to what it is given.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import yaml

from embedding_aware_belt_fusion.alignformer.v2xreal import (
    CORE_METHOD,
    CORRUPT_FRAMES,
    V2XREAL_COM_RANGE,
    V2XREAL_GT_RANGE,
    VEHICLE_TYPES,
    bin_pcd_to_np,
    filter_vehicles,
    load_lidar_bin,
    order_agents,
)

# ----------------------------------------------------------------------------
# Synthetic V2X-Real tree
# ----------------------------------------------------------------------------

SCENARIO = "2023-01-01-00-00-00_1_0"


def _vehicle(kind: str, x: float, y: float) -> dict:
    return {
        "angle": [0.0, 90.0, 0.0],
        "attribute": "",
        "center": [0, 0, 0],
        "extent": [2.3, 1.0, 0.8],
        "location": [x, y, 0.5],
        "obj_type": kind,
    }


def _frame_yaml(infra: bool, x: float) -> dict:
    return {
        "cam1": {"cords": [0] * 6, "extrinsic": np.eye(4).tolist(), "intrinsic": np.eye(3).tolist()},
        "cam2": {"cords": [0] * 6, "extrinsic": np.eye(4).tolist(), "intrinsic": np.eye(3).tolist()},
        "ego_speed": 0,
        "infra": infra,
        "lidar_pose": [x, 0.0, 4.0 if infra else 1.8, 0.0, 0.0, 0.0],
        "true_ego_pose": [x, 0.0, 4.0 if infra else 1.8, 0.0, 0.0, 0.0],
        "vehicles": {
            "1": _vehicle("Car", x + 10.0, 2.0),
            "2": _vehicle("Pedestrian", x + 5.0, -3.0),
            "3": _vehicle("Bus", x + 20.0, 0.0),
            "4": _vehicle("PoliceCar", x + 15.0, -2.0),
        },
    }


def _write_bin(path: Path, points: np.ndarray) -> None:
    points.astype(np.float32).tofile(path)


@pytest.fixture
def scenario_tree(tmp_path: Path) -> Path:
    """One scenario with agents -1, -2, 1, 2 and two frames each."""
    root = tmp_path / "split"
    rng = np.random.default_rng(0)
    for agent, infra, x in (("-1", True, 0.0), ("-2", True, 40.0), ("1", False, 5.0), ("2", False, 30.0)):
        folder = root / SCENARIO / agent
        folder.mkdir(parents=True)
        for stamp in ("000000", "000001"):
            (folder / f"{stamp}.yaml").write_text(yaml.safe_dump(_frame_yaml(infra, x)))
            pts = rng.normal(size=(50, 4)).astype(np.float32)
            _write_bin(folder / f"{stamp}.bin", pts)
    return root


# ----------------------------------------------------------------------------
# Constants: the protocol is written down, not assumed
# ----------------------------------------------------------------------------


def test_the_vehicle_class_is_genecomms_vehicle_super_class_exactly():
    # GenComm data_utils/__init__.py:1-7. Trucks/buses are a different
    # super-class with a different anchor; pedestrians are not vehicles.
    assert VEHICLE_TYPES == frozenset({"Car", "PoliceCar", "LongVehicle"})
    for excluded in ("Bus", "Truck", "Van", "TrashCan", "ConcreteTruck", "Pedestrian", "ScooterRider"):
        assert excluded not in VEHICLE_TYPES


def test_the_ranges_are_the_v2xreal_benchmark_ranges():
    # GenComm datasets/__init__.py:44 and every stage-1 yaml. The z extent is
    # wide because infrastructure LiDARs sit metres above the road.
    assert V2XREAL_GT_RANGE == [-102.4, -51.2, -15, 102.4, 51.2, 15]
    assert V2XREAL_COM_RANGE == 70


def test_the_deleted_lidar_frame_is_named():
    assert ("2023-04-04-15-58-18_30_0", "1", "000112") in CORRUPT_FRAMES


# ----------------------------------------------------------------------------
# LiDAR: .bin in place of .pcd
# ----------------------------------------------------------------------------


def test_load_lidar_bin_returns_n_by_4_float32(tmp_path):
    pts = np.arange(40, dtype=np.float32).reshape(10, 4)
    _write_bin(tmp_path / "f.bin", pts)

    loaded = load_lidar_bin(tmp_path / "f.bin")

    assert loaded.dtype == np.float32
    assert loaded.shape == (10, 4)
    np.testing.assert_array_equal(loaded, pts)


def test_load_lidar_bin_drops_non_finite_rows(tmp_path):
    # Real V2X-Real files carry occasional NaN rows; the corrupt frame that
    # was deleted had 294 of them plus infinities. A NaN reaching the
    # voxelizer is a silent garbage pillar, so rows are dropped, not passed.
    pts = np.ones((5, 4), dtype=np.float32)
    pts[1, 0] = np.nan
    pts[3, 2] = np.inf
    _write_bin(tmp_path / "f.bin", pts)

    loaded = load_lidar_bin(tmp_path / "f.bin")

    assert loaded.shape == (3, 4)
    assert np.isfinite(loaded).all()


def test_load_lidar_bin_refuses_a_truncated_file(tmp_path):
    (tmp_path / "f.bin").write_bytes(b"\x00" * 30)  # not a multiple of 16

    with pytest.raises(ValueError, match="multiple of 16|truncated"):
        load_lidar_bin(tmp_path / "f.bin")


def test_bin_pcd_to_np_reads_the_bin_sibling_of_the_pcd_name(tmp_path):
    # basedataset names the file <ts>.pcd; the file on disk is <ts>.bin.
    pts = np.arange(16, dtype=np.float32).reshape(4, 4)
    _write_bin(tmp_path / "000000.bin", pts)
    reader = bin_pcd_to_np(lambda _: pytest.fail("real reader must not run"))

    loaded = reader(str(tmp_path / "000000.pcd"))

    np.testing.assert_array_equal(loaded, pts)


def test_bin_pcd_to_np_defers_to_the_real_reader_for_a_real_pcd(tmp_path):
    # OPV2V paths keep working in the same process: an existing .pcd goes to
    # the reader that was there before the shim.
    (tmp_path / "000000.pcd").write_text("not really a pcd")
    sentinel = np.zeros((1, 4), dtype=np.float32)
    reader = bin_pcd_to_np(lambda path: sentinel)

    loaded = reader(str(tmp_path / "000000.pcd"))

    assert loaded is sentinel


def test_bin_pcd_to_np_fails_loudly_when_neither_file_exists(tmp_path):
    reader = bin_pcd_to_np(lambda path: pytest.fail("real reader must not run"))

    with pytest.raises(FileNotFoundError, match="000000"):
        reader(str(tmp_path / "000000.pcd"))


# ----------------------------------------------------------------------------
# Class filter
# ----------------------------------------------------------------------------


def test_filter_vehicles_keeps_only_the_vehicle_super_class():
    params = _frame_yaml(infra=False, x=0.0)

    filtered = filter_vehicles(params)

    assert set(filtered["vehicles"]) == {"1", "4"}
    assert {v["obj_type"] for v in filtered["vehicles"].values()} == {"Car", "PoliceCar"}


def test_filter_vehicles_never_mutates_its_input():
    params = _frame_yaml(infra=False, x=0.0)
    before = yaml.safe_dump(params)

    filtered = filter_vehicles(params)

    assert yaml.safe_dump(params) == before
    assert filtered is not params
    assert filtered["vehicles"] is not params["vehicles"]


def test_filter_vehicles_keeps_every_other_key_untouched():
    params = _frame_yaml(infra=True, x=3.0)

    filtered = filter_vehicles(params)

    for key in ("cam1", "cam2", "ego_speed", "infra", "lidar_pose", "true_ego_pose"):
        assert filtered[key] == params[key]


# ----------------------------------------------------------------------------
# Agent order: vehicle-centric
# ----------------------------------------------------------------------------


def test_order_agents_puts_vehicles_first_then_infrastructure():
    # The exact case OpenCOOD's single rotation gets wrong.
    assert order_agents(["-1", "-2", "1", "2"]) == ("1", "2", "-1", "-2")


def test_order_agents_makes_the_only_vehicle_ego_when_vehicle_1_is_absent():
    # 8 train + 1 val scenarios have agents -1, -2, 2 or -1, -2, 1.
    assert order_agents(["-1", "-2", "2"]) == ("2", "-1", "-2")


def test_order_agents_sorts_numerically_not_lexically():
    assert order_agents(["10", "2", "-10", "-2"]) == ("2", "10", "-2", "-10")


def test_order_agents_refuses_a_scenario_with_no_vehicle():
    with pytest.raises(ValueError, match="no vehicle"):
        order_agents(["-1", "-2"])


# ----------------------------------------------------------------------------
# End to end through OpenCOOD's own LateFusionDataset
# ----------------------------------------------------------------------------


def _hypes_for(tree: Path):
    pytest.importorskip("opencood")
    from opencood.hypes_yaml.yaml_utils import load_yaml

    hypes = load_yaml("configs/alignformer_detector_r140.yaml", None)
    hypes = dict(hypes)
    hypes["root_dir"] = str(tree)
    hypes["validate_dir"] = str(tree)
    hypes["fusion"] = {"core_method": CORE_METHOD, "args": []}
    hypes["v2xreal"] = {"gt_range": V2XREAL_GT_RANGE, "comm_range": V2XREAL_COM_RANGE}
    hypes.pop("wild_setting", None)
    return hypes


def test_the_builder_delegates_every_other_core_method_to_opencood(monkeypatch):
    pytest.importorskip("opencood")
    import opencood.data_utils.datasets as opencood_datasets

    from embedding_aware_belt_fusion.alignformer.v2xreal import build_dataset

    calls = []
    monkeypatch.setattr(opencood_datasets, "build_dataset", lambda cfg, visualize, train: calls.append((cfg, visualize, train)) or "opencood-dataset")

    built = build_dataset({"fusion": {"core_method": "LateFusionDataset"}}, visualize=False, train=True)

    assert built == "opencood-dataset"
    assert calls == [({"fusion": {"core_method": "LateFusionDataset"}}, False, True)]


def test_the_ego_is_the_lowest_numbered_vehicle_and_infrastructure_is_last(scenario_tree):
    from embedding_aware_belt_fusion.alignformer.v2xreal import build_dataset

    dataset = build_dataset(_hypes_for(scenario_tree), visualize=False, train=False)

    agents = dataset.scenario_database[0]
    assert list(agents) == ["1", "2", "-1", "-2"]
    assert [agents[a]["ego"] for a in agents] == [True, False, False, False]


def test_retrieve_base_data_reads_the_bin_and_filters_the_classes(scenario_tree):
    from embedding_aware_belt_fusion.alignformer.v2xreal import build_dataset

    dataset = build_dataset(_hypes_for(scenario_tree), visualize=False, train=False)

    data = dataset.retrieve_base_data(0)

    assert list(data) == ["1", "2", "-1", "-2"]
    for cav_id, content in data.items():
        assert content["lidar_np"].shape == (50, 4), cav_id
        assert set(content["params"]["vehicles"]) == {"1", "4"}, cav_id
    assert data["1"]["ego"] is True
    assert data["-1"]["ego"] is False


def test_building_the_dataset_sets_opencoods_ranges_from_the_config(scenario_tree):
    import opencood.data_utils.datasets as opencood_datasets

    from embedding_aware_belt_fusion.alignformer.v2xreal import build_dataset

    build_dataset(_hypes_for(scenario_tree), visualize=False, train=False)

    assert opencood_datasets.GT_RANGE == V2XREAL_GT_RANGE
    assert opencood_datasets.COM_RANGE == V2XREAL_COM_RANGE


def test_the_dataset_is_the_same_length_as_the_ego_frame_count(scenario_tree):
    from embedding_aware_belt_fusion.alignformer.v2xreal import build_dataset

    dataset = build_dataset(_hypes_for(scenario_tree), visualize=False, train=False)

    assert len(dataset) == 2


# ----------------------------------------------------------------------------
# Frame yaml: the C loader with a bounded cache
# ----------------------------------------------------------------------------
#
# Profiled on the NVMe copy: one training item spent 0.83 s of 0.93 s in
# pure-Python yaml parsing, 21 loads for 4 agents, most of them the same
# four files reloaded by basedataset's distance calculation and its
# per-agent parameter loading. libyaml parses the same text several times
# faster and a small per-process cache removes the repeats. What the cache
# returns must be a fresh object each time, because OpenCOOD is free to edit
# what it is handed.


def test_fast_load_yaml_parses_exactly_what_opencood_parses(scenario_tree):
    pytest.importorskip("opencood")
    from opencood.hypes_yaml.yaml_utils import load_yaml

    from embedding_aware_belt_fusion.alignformer.v2xreal import fast_load_yaml

    path = str(scenario_tree / SCENARIO / "1" / "000000.yaml")

    assert fast_load_yaml(path) == load_yaml(path)


def test_fast_load_yaml_reads_scientific_floats_as_floats(tmp_path):
    # OpenCOOD adds an implicit resolver so that `1e-10` is a float and not
    # the string YAML 1.1 would make of it; the fast loader must match.
    from embedding_aware_belt_fusion.alignformer.v2xreal import fast_load_yaml

    (tmp_path / "f.yaml").write_text("eps: 1e-10\nlr: 2e-3\nn: 3\n")

    loaded = fast_load_yaml(str(tmp_path / "f.yaml"))

    assert loaded == {"eps": 1e-10, "lr": 2e-3, "n": 3}
    assert isinstance(loaded["eps"], float)


def test_fast_load_yaml_returns_a_fresh_object_on_every_call(tmp_path):
    from embedding_aware_belt_fusion.alignformer.v2xreal import fast_load_yaml

    (tmp_path / "f.yaml").write_text("vehicles: {1: {obj_type: Car}}\n")

    first = fast_load_yaml(str(tmp_path / "f.yaml"))
    first["vehicles"]["1"] = "mutated"
    second = fast_load_yaml(str(tmp_path / "f.yaml"))

    assert second == {"vehicles": {1: {"obj_type": "Car"}}}


def test_fast_load_yaml_serves_repeats_from_the_cache(tmp_path):
    from embedding_aware_belt_fusion.alignformer.v2xreal import _parse_yaml, fast_load_yaml

    (tmp_path / "f.yaml").write_text("a: 1\n")
    _parse_yaml.cache_clear()

    for _ in range(3):
        fast_load_yaml(str(tmp_path / "f.yaml"))

    assert _parse_yaml.cache_info().hits == 2
    assert _parse_yaml.cache_info().misses == 1


def test_fast_load_yaml_defers_hypes_loading_to_opencood(tmp_path):
    # load_yaml(file, opt) with opt.model_dir reads config.yaml and runs the
    # yaml_parser; that is OpenCOOD's job, not the frame loader's.
    pytest.importorskip("opencood")
    from embedding_aware_belt_fusion.alignformer.v2xreal import fast_load_yaml

    class Opt:
        model_dir = str(tmp_path)

    (tmp_path / "config.yaml").write_text("name: from-config\n")

    assert fast_load_yaml("ignored.yaml", Opt())["name"] == "from-config"


def test_building_the_dataset_installs_the_fast_loader(scenario_tree):
    import opencood.data_utils.datasets.basedataset as basedataset
    import opencood.data_utils.datasets.late_fusion_dataset as late_fusion

    from embedding_aware_belt_fusion.alignformer.v2xreal import build_dataset, fast_load_yaml

    build_dataset(_hypes_for(scenario_tree), visualize=False, train=False)

    assert basedataset.load_yaml is fast_load_yaml
    assert late_fusion.load_yaml is fast_load_yaml


# ----------------------------------------------------------------------------
# Frame hooks for the cache builder
# ----------------------------------------------------------------------------
#
# The cache builder assembles one agent-frame at a time from a yaml path and
# a LiDAR path, outside retrieve_base_data. The V2X-Real dataset therefore
# exposes the same two differences as methods the builder can ask for.


def test_frame_params_filters_the_classes_the_way_retrieve_base_data_does(scenario_tree):
    from embedding_aware_belt_fusion.alignformer.v2xreal import build_dataset

    dataset = build_dataset(_hypes_for(scenario_tree), visualize=False, train=False)

    params = dataset.frame_params(scenario_tree / SCENARIO / "1" / "000000.yaml")

    assert set(params["vehicles"]) == {"1", "4"}
    assert params["lidar_pose"][0] == 5.0


def test_frame_points_reads_the_bin_named_by_its_pcd_path(scenario_tree):
    from embedding_aware_belt_fusion.alignformer.v2xreal import build_dataset

    dataset = build_dataset(_hypes_for(scenario_tree), visualize=False, train=False)

    points = dataset.frame_points(scenario_tree / SCENARIO / "1" / "000000.pcd")

    assert points.shape == (50, 4)


def test_frame_cameras_returns_both_images_with_their_calibrations(scenario_tree):
    from PIL import Image

    from embedding_aware_belt_fusion.alignformer.v2xreal import build_dataset

    folder = scenario_tree / SCENARIO / "1"
    for cam in ("cam1", "cam2"):
        Image.new("RGB", (64, 32), color=(10, 20, 30)).save(folder / f"000000_{cam}.jpeg")
    dataset = build_dataset(_hypes_for(scenario_tree), visualize=False, train=False)

    cameras = dataset.frame_cameras(folder / "000000.yaml")

    assert len(cameras) == 2
    image, calib = cameras[0]
    assert image.shape == (32, 64, 3)
    assert calib.intrinsic.shape == (3, 3) and calib.camera_to_lidar.shape == (4, 4)


# ----------------------------------------------------------------------------
# The corrupt frame is dropped from the index, for every agent
# ----------------------------------------------------------------------------
#
# Refusing it at read time crashed the detector's first epoch (2026-09-30):
# the yaml still exists, so basedataset indexes the frame, and the read-time
# refusal fires inside a DataLoader worker. And dropping it for one agent
# only would not do either, because every agent is looked up by the ego's
# timestamp key. The read-time refusal stays as the backstop.


CORRUPT_SCENARIO, CORRUPT_AGENT, CORRUPT_STAMP = "2023-04-04-15-58-18_30_0", "1", "000112"


@pytest.fixture
def tree_with_the_corrupt_frame(tmp_path: Path) -> Path:
    root = tmp_path / "split"
    rng = np.random.default_rng(0)
    for agent, infra, x in (("-1", True, 0.0), ("1", False, 5.0), ("2", False, 30.0)):
        folder = root / CORRUPT_SCENARIO / agent
        folder.mkdir(parents=True)
        for stamp in ("000111", CORRUPT_STAMP, "000113"):
            (folder / f"{stamp}.yaml").write_text(yaml.safe_dump(_frame_yaml(infra, x)))
            if not (agent == CORRUPT_AGENT and stamp == CORRUPT_STAMP):
                _write_bin(folder / f"{stamp}.bin", rng.normal(size=(50, 4)).astype(np.float32))
    return root


def test_the_corrupt_timestamp_is_absent_from_every_agent_and_the_length(tree_with_the_corrupt_frame):
    from embedding_aware_belt_fusion.alignformer.v2xreal import build_dataset

    dataset = build_dataset(_hypes_for(tree_with_the_corrupt_frame), visualize=False, train=False)

    for cav_id, content in dataset.scenario_database[0].items():
        stamps = [k for k in content if k != "ego"]
        assert stamps == ["000111", "000113"], cav_id
    assert len(dataset) == 2
    assert dataset.len_record == [2]


def test_every_remaining_frame_of_that_scenario_loads(tree_with_the_corrupt_frame):
    from embedding_aware_belt_fusion.alignformer.v2xreal import build_dataset

    dataset = build_dataset(_hypes_for(tree_with_the_corrupt_frame), visualize=False, train=False)

    for index in range(len(dataset)):
        data = dataset.retrieve_base_data(index)
        assert list(data) == ["1", "2", "-1"]


# ----------------------------------------------------------------------------
# Every AlignFormer entry point builds its OpenCOOD dataset through the adapter
# ----------------------------------------------------------------------------
#
# The adapter delegates every non-V2X-Real core_method to OpenCOOD unchanged,
# so routing everything through it costs OPV2V nothing and is what lets one
# --config select the dataset. A direct import of OpenCOOD's builder in one
# of these would silently make that entry point OPV2V-only.


@pytest.mark.parametrize(
    "module_path",
    [
        "src/embedding_aware_belt_fusion/alignformer/evaluate.py",
        "src/embedding_aware_belt_fusion/alignformer/cache.py",
        "src/embedding_aware_belt_fusion/alignformer/baselines.py",
        "scripts/calibrate_freealign.py",
        "scripts/correspondence_evidence_budget.py",
    ],
)
def test_alignformer_entry_points_build_datasets_through_the_adapter(module_path):
    source = Path(module_path).read_text()

    assert "from opencood.data_utils.datasets import build_dataset" not in source
    assert "from embedding_aware_belt_fusion.alignformer.v2xreal import build_dataset" in source


def test_frame_cameras_drops_an_unreadable_image_with_a_warning_naming_it(scenario_tree):
    """test/2023-04-03-18-28-32_22_0/-1/000031_cam1.jpeg is a zero-byte file
    in the released dataset. A dropped camera frame is 'no camera' for that
    agent-frame, not a crash of the whole cache -- but never silently."""
    import warnings

    from PIL import Image

    from embedding_aware_belt_fusion.alignformer.v2xreal import build_dataset

    folder = scenario_tree / SCENARIO / "1"
    (folder / "000000_cam1.jpeg").write_bytes(b"")
    Image.new("RGB", (64, 32), color=(10, 20, 30)).save(folder / "000000_cam2.jpeg")
    dataset = build_dataset(_hypes_for(scenario_tree), visualize=False, train=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cameras = dataset.frame_cameras(folder / "000000.yaml")

    assert len(cameras) == 1
    assert any("000000_cam1.jpeg" in str(w.message) for w in caught)
