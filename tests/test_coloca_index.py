"""Tests for the OPV2V agent-pair index and the binary point-cloud cache."""

from pathlib import Path

import numpy as np
import pytest

from embedding_aware_belt_fusion.coloca.index import (
    build_pairs,
    load_or_build_pairs,
    opencood_ego_id,
    pair_cache_path,
    parse_lidar_pose,
    scan_split,
)
from embedding_aware_belt_fusion.coloca.pcd_cache import cached_pcd_path

YAML_TEMPLATE = """\
ego_speed: 7.83
lidar_pose:
- {x}
- {y}
- 1.93
- -0.017
- {yaw}
- 0.028
vehicles:
  650:
    angle:
    - 0.0
"""


def write_frame(cav_dir: Path, timestamp: str, x: float, y: float, yaw: float) -> None:
    """Write a minimal OPV2V-shaped frame (yaml + a stand-in pcd)."""
    cav_dir.mkdir(parents=True, exist_ok=True)
    (cav_dir / f"{timestamp}.yaml").write_text(YAML_TEMPLATE.format(x=x, y=y, yaw=yaw))
    (cav_dir / f"{timestamp}.pcd").write_text("# placeholder\n")


@pytest.fixture
def opv2v_split(tmp_path: Path) -> Path:
    """Two scenarios: one with two nearby CAVs, one with a single CAV."""
    root = tmp_path / "train"
    write_frame(root / "scene_a" / "641", "000069", 0.0, 0.0, 10.0)
    write_frame(root / "scene_a" / "650", "000069", 12.0, 5.0, -20.0)
    write_frame(root / "scene_a" / "641", "000071", 1.0, 0.0, 11.0)
    write_frame(root / "scene_a" / "650", "000071", 13.0, 5.0, -21.0)
    write_frame(root / "scene_b" / "700", "000069", 0.0, 0.0, 0.0)
    (root / "scene_a" / "data_protocol.yaml").write_text("protocol: test\n")
    return root


def test_parse_lidar_pose_reads_the_six_pose_values(opv2v_split: Path):
    # Act
    pose = parse_lidar_pose(opv2v_split / "scene_a" / "641" / "000069.yaml")

    # Assert
    assert pose == [0.0, 0.0, 1.93, -0.017, 10.0, 0.028]


def test_parse_lidar_pose_rejects_a_yaml_without_the_key(tmp_path: Path):
    # Arrange
    path = tmp_path / "bad.yaml"
    path.write_text("ego_speed: 1.0\n")

    # Act / Assert
    with pytest.raises(ValueError, match="no lidar_pose"):
        parse_lidar_pose(path)


def test_scan_split_skips_non_directories_and_single_agent_scenarios(opv2v_split: Path):
    # Act
    poses = scan_split(opv2v_split)

    # Assert: data_protocol.yaml is not mistaken for a CAV folder
    assert sorted(poses) == ["scene_a", "scene_b"]
    assert sorted(poses["scene_a"]) == ["641", "650"]
    assert sorted(poses["scene_a"]["641"]) == ["000069", "000071"]


def test_scan_split_ignores_yaml_without_a_matching_pcd(opv2v_split: Path):
    # Arrange: a yaml whose point cloud is missing cannot be loaded
    (opv2v_split / "scene_a" / "641" / "000073.yaml").write_text(
        YAML_TEMPLATE.format(x=2.0, y=0.0, yaw=12.0)
    )

    # Act
    poses = scan_split(opv2v_split)

    # Assert
    assert "000073" not in poses["scene_a"]["641"]


def test_build_pairs_emits_both_orderings_and_skips_lone_agents(opv2v_split: Path):
    # Act
    pairs = build_pairs(scan_split(opv2v_split), comm_range_m=40.0)

    # Assert: 2 timestamps x 2 ordered pairs; scene_b has a single CAV
    assert len(pairs) == 4
    assert {p.scenario for p in pairs} == {"scene_a"}
    assert {(p.ego_id, p.cav_id) for p in pairs} == {("641", "650"), ("650", "641")}


def test_build_pairs_drops_agents_beyond_the_communication_range(opv2v_split: Path):
    # Arrange: the two CAVs are 13 m apart
    poses = scan_split(opv2v_split)

    # Act
    within = build_pairs(poses, comm_range_m=20.0)

    # Assert
    assert len(within) == 4
    with pytest.raises(ValueError, match="no agent pairs"):
        build_pairs(poses, comm_range_m=5.0)


def test_build_pairs_rejects_a_non_positive_range(opv2v_split: Path):
    # Act / Assert
    with pytest.raises(ValueError, match="comm_range_m must be positive"):
        build_pairs(scan_split(opv2v_split), comm_range_m=0.0)


def test_scan_split_reports_a_missing_directory(tmp_path: Path):
    # Act / Assert
    with pytest.raises(FileNotFoundError, match="not found"):
        scan_split(tmp_path / "nope")


def test_pair_cache_round_trips_and_is_reused(opv2v_split: Path, tmp_path: Path):
    # Arrange
    cache_path = tmp_path / "cache" / "pairs.npz"

    # Act
    first = load_or_build_pairs(opv2v_split, cache_path, comm_range_m=40.0)
    second = load_or_build_pairs(opv2v_split, cache_path, comm_range_m=40.0)

    # Assert
    assert cache_path.exists()
    assert first == second


def test_pair_cache_is_rebuilt_when_the_communication_range_changes(
    opv2v_split: Path, tmp_path: Path
):
    # Arrange
    cache_path = tmp_path / "pairs.npz"
    load_or_build_pairs(opv2v_split, cache_path, comm_range_m=40.0)

    # Act: a range that excludes every pair must not silently reuse the cache
    with pytest.raises(ValueError, match="no agent pairs"):
        load_or_build_pairs(opv2v_split, cache_path, comm_range_m=1.0)


def test_cached_pcd_path_mirrors_the_split_layout():
    # Arrange
    split_root = Path("/data/OPV2V/train")
    source = split_root / "scene_a" / "641" / "000069.pcd"

    # Act
    cached = cached_pcd_path(Path("/nvme/cache/train"), source, split_root)

    # Assert
    assert cached == Path("/nvme/cache/train/scene_a/641/000069.npy")


def test_load_points_prefers_the_cache_over_the_source_pcd(tmp_path: Path):
    # Arrange: a cached array whose source pcd is deliberately unreadable
    from embedding_aware_belt_fusion.coloca.pcd_cache import load_points

    split_root = tmp_path / "train"
    source = split_root / "scene_a" / "641" / "000069.pcd"
    source.parent.mkdir(parents=True)
    source.write_text("not a real pcd")

    cache_root = tmp_path / "cache"
    expected = np.arange(12, dtype=np.float32).reshape(3, 4)
    cached = cached_pcd_path(cache_root, source, split_root)
    cached.parent.mkdir(parents=True)
    np.save(cached, expected)

    # Act
    points = load_points(source, cache_root, split_root)

    # Assert
    np.testing.assert_array_equal(points, expected)


def test_training_rng_varies_per_epoch_but_is_stable_within_one():
    """Guards the persistent-workers bug: identical noise on every epoch."""
    # Arrange / Act
    from embedding_aware_belt_fusion.coloca.dataset import sample_rng

    epoch0 = sample_rng(seed=0, epoch=0, index=7, train=True).normal(size=3)
    epoch1 = sample_rng(seed=0, epoch=1, index=7, train=True).normal(size=3)
    epoch0_again = sample_rng(seed=0, epoch=0, index=7, train=True).normal(size=3)

    # Assert
    assert not np.allclose(epoch0, epoch1)
    np.testing.assert_array_equal(epoch0, epoch0_again)


def test_evaluation_rng_ignores_the_epoch_so_metrics_are_reproducible():
    # Arrange / Act
    from embedding_aware_belt_fusion.coloca.dataset import sample_rng

    first = sample_rng(seed=20, epoch=0, index=3, train=False).normal(size=3)
    second = sample_rng(seed=20, epoch=9, index=3, train=False).normal(size=3)

    # Assert
    np.testing.assert_array_equal(first, second)


def test_different_samples_get_different_noise():
    # Arrange / Act
    from embedding_aware_belt_fusion.coloca.dataset import sample_rng

    a = sample_rng(seed=0, epoch=0, index=1, train=True).normal(size=3)
    b = sample_rng(seed=0, epoch=0, index=2, train=True).normal(size=3)

    # Assert
    assert not np.allclose(a, b)


def test_pair_cache_path_keys_the_filename_on_the_communication_range():
    # Two ranges over the same split are two different indexes, and the cache
    # signature only tells you the file on disk is stale -- it cannot keep both.
    # Keying the NAME on the range lets the 40 m index (which the CoLoca-QuA
    # baseline still uses) and the 70 m one coexist instead of evicting each
    # other on every alternating run.
    # Arrange
    directory = Path("outputs/coloca/cache")

    # Act
    forty = pair_cache_path(directory, "train", 40.0)
    seventy = pair_cache_path(directory, "train", 70.0)

    # Assert
    assert forty == directory / "train_pairs_40.npz"
    assert seventy == directory / "train_pairs_70.npz"
    assert pair_cache_path(directory, "test", 70.0) == directory / "test_pairs_70.npz"


def test_the_opencood_ego_is_the_lowest_string_sorted_agent():
    # OpenCOOD's BaseDataset picks sorted(os.listdir(scenario))[0] as ego, and
    # its sort is over the directory names, i.e. lexicographic, not numeric.
    # A diagnostic that wants the same population the fused-AP sweep evaluates
    # has to reproduce that choice exactly rather than guess at it.
    assert opencood_ego_id(["650", "641", "1045"]) == "1045"
    assert opencood_ego_id(["641", "650"]) == "641"


def test_a_roadside_unit_is_never_the_opencood_ego():
    # Negative ids are roadside units; BaseDataset rotates one off the front of
    # the list precisely so it cannot become the ego.
    assert opencood_ego_id(["-1", "641", "650"]) == "641"


def test_the_opencood_ego_needs_at_least_one_agent():
    with pytest.raises(ValueError, match="at least one agent"):
        opencood_ego_id([])


# ----------------------------------------------------------------------------
# V2X-Real: the LiDAR is <ts>.bin, never <ts>.pcd
# ----------------------------------------------------------------------------


def test_scan_split_counts_a_frame_whose_lidar_is_a_bin_file(tmp_path):
    from embedding_aware_belt_fusion.coloca.index import scan_split

    cav = tmp_path / "2023-01-01-00-00-00_1_0" / "1"
    cav.mkdir(parents=True)
    (cav / "000000.yaml").write_text("lidar_pose:\n- 1.0\n- 2.0\n- 3.0\n- 0.0\n- 0.0\n- 0.0\n")
    (cav / "000000.bin").write_bytes(b"\x00" * 16)
    (cav / "000001.yaml").write_text("lidar_pose:\n- 1.0\n- 2.0\n- 3.0\n- 0.0\n- 0.0\n- 0.0\n")
    # no LiDAR at all for 000001: not a usable frame

    poses = scan_split(tmp_path)

    assert list(poses["2023-01-01-00-00-00_1_0"]["1"]) == ["000000"]
