"""Per-object camera features for the LiDAR+camera trunk on V2X-Real.

What the branch is, and what it is not
--------------------------------------

Each detected box (from the LiDAR detector, in the agent's LiDAR frame) is
projected into each of the agent's cameras with ``K @ inv(extrinsic)`` --
the convention verified by eye on V2X-Real (memory:
``v2x-real-dataset-state``: the stored ``extrinsic`` is camera->LiDAR). The
camera in which the box has the larger visible area wins, and a frozen
ImageNet ResNet-18, cut at ``layer3`` (stride 16, 256 channels), is pooled
over the 2-D box with ``roi_align``. A box behind the camera or too small
on the image gets a zero vector and ``has_camera = False``; the embedding
head sees the flag, so absence is information, not a zero pretending to be
a feature.

Frozen on purpose: the feature is computed ONCE into the cache beside the
LiDAR ROI feature, so the three trunks train through an identical pipeline
and differ only in what the embedding head is fed. If it shows signal,
fine-tuning is the follow-up; if it shows none, the null is about frozen
ImageNet features on this data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from embedding_aware_belt_fusion.alignformer.camera_features import (
    FEATURE_DIM,
    MIN_BOX_SIDE_PX,
    MIN_DEPTH_M,
    STRIDE,
    Box2D,
    Calibration,
    CameraBackbone,
    box_2d,
    calibration_from_yaml,
    choose_camera,
    frame_camera_features,
    pooled_features,
    project_points,
)

IMAGE_SIZE = (1920, 1080)
FOCAL = 1000.0


def _calibration(camera_to_lidar: np.ndarray | None = None) -> Calibration:
    intrinsic = np.array([[FOCAL, 0.0, 960.0], [0.0, FOCAL, 540.0], [0.0, 0.0, 1.0]])
    return Calibration(
        intrinsic=intrinsic,
        camera_to_lidar=np.eye(4) if camera_to_lidar is None else camera_to_lidar,
    )


def _cube(center: np.ndarray, half: float = 1.0) -> np.ndarray:
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    return center[None, :] + half * signs


# ----------------------------------------------------------------------------
# Calibration and projection
# ----------------------------------------------------------------------------


def test_calibration_from_yaml_reads_the_two_matrices():
    block = {"cords": [0] * 6, "extrinsic": (2 * np.eye(4)).tolist(), "intrinsic": (3 * np.eye(3)).tolist()}

    calib = calibration_from_yaml(block)

    np.testing.assert_array_equal(calib.intrinsic, 3 * np.eye(3))
    np.testing.assert_array_equal(calib.camera_to_lidar, 2 * np.eye(4))


def test_a_point_on_the_optical_axis_projects_to_the_principal_point():
    # With an identity camera->LiDAR matrix the camera frame IS the LiDAR
    # frame, so the camera looks down LiDAR +z.
    uv, depth = project_points(np.array([[0.0, 0.0, 10.0]]), _calibration())

    np.testing.assert_allclose(uv, [[960.0, 540.0]])
    np.testing.assert_allclose(depth, [10.0])


def test_projection_uses_the_inverse_of_the_stored_extrinsic():
    # extrinsic is camera->LiDAR: a camera translated +5 in LiDAR x sees a
    # point at LiDAR x = 5 on its own axis.
    camera_to_lidar = np.eye(4)
    camera_to_lidar[0, 3] = 5.0

    uv, depth = project_points(np.array([[5.0, 0.0, 10.0]]), _calibration(camera_to_lidar))

    np.testing.assert_allclose(uv, [[960.0, 540.0]])
    np.testing.assert_allclose(depth, [10.0])


def test_box_2d_is_the_clipped_bounding_box_of_the_projected_corners():
    corners = _cube(np.array([0.0, 0.0, 10.0]))

    box = box_2d(corners, _calibration(), IMAGE_SIZE)

    # nearest corners (depth 9) spread the most: 960 +/- 1000/9.
    assert box is not None
    assert box.x0 == pytest.approx(960 - FOCAL / 9)
    assert box.x1 == pytest.approx(960 + FOCAL / 9)
    assert box.y0 == pytest.approx(540 - FOCAL / 9)
    assert box.y1 == pytest.approx(540 + FOCAL / 9)
    assert box.visible_fraction == pytest.approx(1.0)


def test_a_box_behind_the_camera_has_no_2d_box():
    assert box_2d(_cube(np.array([0.0, 0.0, -10.0])), _calibration(), IMAGE_SIZE) is None


def test_a_box_straddling_the_image_plane_has_no_2d_box():
    # One corner at depth -0.5, the rest ahead: the projection of the near
    # corners is meaningless, so the box is declared not visible.
    corners = _cube(np.array([0.0, 0.0, 0.5]))
    assert corners[:, 2].min() < MIN_DEPTH_M

    assert box_2d(corners, _calibration(), IMAGE_SIZE) is None


def test_a_box_entirely_outside_the_image_has_no_2d_box():
    # Far to the left: u = 960 - 1000 * 50 / 10 < 0 for every corner.
    assert box_2d(_cube(np.array([-50.0, 0.0, 10.0])), _calibration(), IMAGE_SIZE) is None


def test_a_box_partly_outside_is_clipped_and_reports_its_visible_fraction():
    # Centre at u = 960 - 1000 * 9 / 10 = 60, spreading +/- 111 at depth 9:
    # x0 would be negative and is clipped to 0.
    box = box_2d(_cube(np.array([-9.0, 0.0, 10.0])), _calibration(), IMAGE_SIZE)

    assert box is not None
    assert box.x0 == 0.0
    assert 0.0 < box.visible_fraction < 1.0


def test_a_box_thinner_than_the_minimum_side_has_no_2d_box():
    # A 2 m cube at 500 m is ~4 px wide at f = 1000: under the floor.
    corners = _cube(np.array([0.0, 0.0, 500.0]))
    projected_width = 2 * FOCAL / 499
    assert projected_width < MIN_BOX_SIDE_PX

    assert box_2d(corners, _calibration(), IMAGE_SIZE) is None


# ----------------------------------------------------------------------------
# Choosing a camera
# ----------------------------------------------------------------------------


def test_choose_camera_takes_the_larger_visible_area_and_minus_one_when_none():
    small = Box2D(0.0, 0.0, 10.0, 10.0, 1.0)
    large = Box2D(0.0, 0.0, 30.0, 30.0, 1.0)

    assert choose_camera([[small, large], [large, small], [None, None]]) == [1, 0, -1]


# ----------------------------------------------------------------------------
# The backbone and the pooling
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def backbone() -> CameraBackbone:
    return CameraBackbone().eval()


def test_the_backbone_is_frozen_stride_16_and_256_channels(backbone):
    assert all(not p.requires_grad for p in backbone.parameters())
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)

    feature_map = backbone.feature_map(image)

    assert feature_map.shape == (1, FEATURE_DIM, 1080 // STRIDE + 1, 1920 // STRIDE)
    assert STRIDE == 16 and FEATURE_DIM == 256


def test_pooled_features_are_deterministic_and_box_dependent(backbone):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(1080, 1920, 3), dtype=np.uint8)
    feature_map = backbone.feature_map(image)
    boxes = torch.tensor([[100.0, 100.0, 300.0, 250.0], [1200.0, 600.0, 1500.0, 900.0]])

    first = pooled_features(feature_map, boxes)
    second = pooled_features(feature_map, boxes)

    assert first.shape == (2, FEATURE_DIM)
    torch.testing.assert_close(first, second)
    assert not torch.allclose(first[0], first[1])


# ----------------------------------------------------------------------------
# One agent-frame
# ----------------------------------------------------------------------------


def test_frame_features_flag_the_invisible_object_and_zero_its_vector(backbone):
    rng = np.random.default_rng(1)
    image = rng.integers(0, 255, size=(1080, 1920, 3), dtype=np.uint8)
    corners = np.stack([_cube(np.array([0.0, 0.0, 10.0])), _cube(np.array([0.0, 0.0, -10.0]))])

    result = frame_camera_features(backbone, corners, [(image, _calibration())])

    assert result.has_camera.tolist() == [True, False]
    assert result.camera_index.tolist() == [0, -1]
    assert result.features.shape == (2, FEATURE_DIM)
    assert np.abs(result.features[1]).max() == 0.0
    assert np.abs(result.features[0]).max() > 0.0


def test_frame_features_with_no_objects_are_empty_not_an_error(backbone):
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)

    result = frame_camera_features(backbone, np.zeros((0, 8, 3)), [(image, _calibration())])

    assert result.features.shape == (0, FEATURE_DIM)
    assert result.has_camera.shape == (0,)


VERIFIED_FRAME = Path("/media/chenyi/Elements1/Dataset/v2x-real/train/2023-04-04-13-58-53_15_0/1")


@pytest.mark.skipif(not VERIFIED_FRAME.exists(), reason="V2X-Real not on this machine")
def test_the_verified_real_frame_puts_vehicles_in_the_image():
    """The frame whose projection was checked by eye on 2026-09-29: the bus
    and the car down the road land on themselves with K @ inv(extrinsic)."""
    import yaml

    from embedding_aware_belt_fusion.alignformer.camera_features import load_image

    params = yaml.safe_load((VERIFIED_FRAME / "000050.yaml").read_text())
    from opencood.utils.transformation_utils import x_to_world

    lidar_from_world = np.linalg.inv(x_to_world(params["lidar_pose"]))
    corners = []
    for entry in params["vehicles"].values():
        if entry["obj_type"] not in ("Car", "Bus", "Truck", "Van"):
            continue
        box_from_world = x_to_world(entry["location"] + entry["angle"])
        half = np.array(entry["extent"])
        local = _cube(np.zeros(3)) * half
        world = (box_from_world @ np.c_[local, np.ones(8)].T).T
        corners.append((lidar_from_world @ world.T).T[:, :3])
    image = load_image(VERIFIED_FRAME / "000050_cam1.jpeg")
    calib = calibration_from_yaml(params["cam1"])

    boxes = [box_2d(c, calib, (image.shape[1], image.shape[0])) for c in corners]
    visible = [b for b in boxes if b is not None]

    # Four were in view by eye; the floor on box size may drop the smallest.
    assert len(visible) >= 3
    for box in visible:
        assert 0 <= box.x0 < box.x1 <= image.shape[1]
        assert 0 <= box.y0 < box.y1 <= image.shape[0]
        assert box.y0 > 300  # on the road, not in the sky: the wrong convention put them there
