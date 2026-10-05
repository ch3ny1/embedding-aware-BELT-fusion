"""Does camera colour carry cross-agent object identity on V2X-Real?

The OPV2V answer was no, for two reasons: a small CARLA asset library and a
cross-agent viewpoint change that took colour from AUC 0.71 (same agent) to
0.60. The first reason does not transfer to real traffic; the second might.
The pure parts of the V2X-Real probe are tested here: pair enumeration over
the on-disk layout, the LiDAR-frame corners, the shared-count bucket the
report is split by, and the per-frame view extraction on a synthetic image.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_v2xreal_colour_separability import (  # noqa: E402
    Pair,
    enumerate_pairs,
    refuse_test_split,
    shared_bucket,
    vehicle_lidar_corners,
    views_for_agent_frame,
)


def _vehicle(location, extent=(2.0, 1.0, 0.75), angle=(0.0, 0.0, 0.0), obj_type="Car"):
    return {
        "location": list(location),
        "center": [0.0, 0.0, 0.0],
        "extent": list(extent),
        "angle": list(angle),
        "obj_type": obj_type,
    }


def test_enumerate_pairs_lists_every_ordered_agent_pair_at_a_shared_timestamp(tmp_path):
    scenario = tmp_path / "2023-04-04-14-28-53_45_0"
    for agent, stamps in {"-1": ("000010", "000011"), "1": ("000010", "000011"), "2": ("000011",)}.items():
        (scenario / agent).mkdir(parents=True)
        for stamp in stamps:
            (scenario / agent / f"{stamp}.yaml").write_text("{}")
    (scenario / "1" / "._000010.yaml").write_text("")  # macOS resource fork, never a frame

    pairs = enumerate_pairs(tmp_path)

    assert Pair("2023-04-04-14-28-53_45_0", "000010", "-1", "1") in pairs
    assert Pair("2023-04-04-14-28-53_45_0", "000010", "1", "-1") in pairs
    assert Pair("2023-04-04-14-28-53_45_0", "000011", "2", "-1") in pairs
    assert len(pairs) == 2 + 6  # two agents at 000010, three at 000011


def test_the_test_split_is_refused():
    with pytest.raises(ValueError):
        refuse_test_split(Path("/data/v2x-real/test"))
    refuse_test_split(Path("/data/v2x-real/val"))


def test_shared_bucket_splits_at_three():
    assert shared_bucket(1) == "shared_1_2"
    assert shared_bucket(2) == "shared_1_2"
    assert shared_bucket(3) == "shared_3plus"


def test_vehicle_corners_land_in_the_lidar_frame_with_half_extents():
    lidar_pose = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # LiDAR at world x = 10, unrotated
    vehicle = _vehicle(location=(15.0, 2.0, 0.0))

    corners = vehicle_lidar_corners(vehicle, lidar_pose)

    assert corners.shape == (8, 3)
    np.testing.assert_allclose(corners.mean(axis=0), [5.0, 2.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(corners.max(axis=0) - corners.min(axis=0), [4.0, 2.0, 1.5], atol=1e-9)


def _identity_camera_block(focal=1000.0):
    intrinsic = [[focal, 0.0, 960.0], [0.0, focal, 540.0], [0.0, 0.0, 1.0]]
    return {"cords": [0.0] * 6, "extrinsic": np.eye(4).tolist(), "intrinsic": intrinsic}


def test_views_pick_up_a_vehicle_in_front_of_the_camera_and_skip_one_behind_it():
    # Identity extrinsic: the camera frame IS the LiDAR frame, looking down +z.
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(1080, 1920, 3), dtype=np.uint8)
    params = {
        "lidar_pose": [0.0] * 6,
        "cam1": _identity_camera_block(),
        "vehicles": {
            "7": _vehicle(location=(0.0, 0.0, 12.0)),
            "8": _vehicle(location=(0.0, 0.0, -12.0)),
            "9": {**_vehicle(location=(3.0, 0.0, 12.0)), "obj_type": "Pedestrian"},
        },
    }

    views = views_for_agent_frame(params, {"cam1": image})

    assert set(views) == {"7"}
    assert views["7"]["camera"] == "cam1"
    assert views["7"]["pixels"] > 50
    assert views["7"]["hue_saturation"].shape == (24 * 8,)
    assert views["7"]["mean_colour"].shape == (3,)


# ----------------------------------------------------------------------------
# Pluggable descriptors: the same probe, a foundation-model cue
# ----------------------------------------------------------------------------


def test_views_carry_whatever_descriptors_the_describer_returns():
    from analyze_v2xreal_colour_separability import Describer

    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(1080, 1920, 3), dtype=np.uint8)
    params = {"lidar_pose": [0.0] * 6, "cam1": _identity_camera_block(), "vehicles": {"7": _vehicle(location=(0.0, 0.0, 12.0))}}
    seen = []

    def constant(image_bgr, mask, view):
        seen.append((image_bgr.shape, int(mask.sum()), view.box))
        return {"toy": np.array([1.0, 0.0])}

    views = views_for_agent_frame(params, {"cam1": image}, describe=Describer(("toy",), constant))

    assert views["7"]["toy"].tolist() == [1.0, 0.0]
    assert "hue_saturation" not in views["7"]
    assert seen and seen[0][0] == image.shape and seen[0][1] > 50


def test_describer_for_colour_is_the_default_and_names_both_colour_descriptors():
    from analyze_v2xreal_colour_separability import colour_describer

    describer = colour_describer()

    assert describer.names == ("hue_saturation", "mean_colour")


def test_combined_describer_merges_names_and_descriptors():
    from analyze_v2xreal_colour_separability import Describer, combine

    first = Describer(("a",), lambda image, mask, view: {"a": np.ones(2)})
    second = Describer(("b",), lambda image, mask, view: {"b": np.zeros(3)})
    view = type("V", (), {"box": (0, 0, 1, 1)})()

    both = combine(first, second)
    out = both.describe(np.zeros((2, 2, 3), np.uint8), np.ones((2, 2), bool), view)

    assert both.names == ("a", "b")
    assert set(out) == {"a", "b"}


def test_combined_describer_returns_none_when_any_part_declines():
    from analyze_v2xreal_colour_separability import Describer, combine

    first = Describer(("a",), lambda image, mask, view: {"a": np.ones(2)})
    second = Describer(("b",), lambda image, mask, view: None)
    view = type("V", (), {"box": (0, 0, 1, 1)})()

    assert combine(first, second).describe(np.zeros((2, 2, 3), np.uint8), np.ones((2, 2), bool), view) is None


def test_verdict_picks_the_best_descriptor_among_the_names_given():
    from analyze_v2xreal_colour_separability import verdict

    report = {
        "ambiguous": {"x": {"auc_paired": 0.6}, "y": {"auc_paired": 0.9}},
        "shuffled_control": {"x": {"auc_paired": 0.5}, "y": {"auc_paired": 0.51}},
        "coverage": {"fraction": 0.5, "by_range": {"40-70m": {"fraction": 0.3}}},
    }

    out = verdict(report, names=("x", "y"))

    assert out["best_descriptor"] == "y"
    assert out["worth_building"] is True
