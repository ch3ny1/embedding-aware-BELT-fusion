"""Per-object camera appearance, and whether it carries identity.

Why this exists
---------------

The measured null stands: AlignFormer's 128-d rotated-ROI LiDAR embedding adds
``+0.0001 [-0.0013, +0.0014]`` to cross-agent association, and one of the three
reasons it cannot work is structural -- OPV2V is CARLA with a small vehicle
asset library, so competing vehicles differ by a median 0.071 m in width and
all 65,774 CAV box widths are ``2.014 +/- 0.081`` m. The competitors are the
same object *in shape*.

Colour is the one channel that claim does not cover, because LiDAR cannot see
it at all. A first probe on 118 vehicle views says the separation is there:
the mean-BGR spread **within** one vehicle identity across frames and cameras
is 3.2, and **between** identities it is 32.6. A ten-to-one ratio is not what
a small asset library looks like.

That probe is same-agent, though, which is the easy case. This module is the
machinery for the cross-agent question -- ego's cameras against the CAV's,
different viewpoint, different range, different incident light -- and these
tests pin the parts of it that could be silently wrong and still produce a
plausible AUC.

What is load-bearing here
-------------------------

1. **The crop is the vehicle, not the road.** An unmasked bounding-box crop of
   a car at 60 m is mostly asphalt, and asphalt matches asphalt: the
   background would manufacture the very separation the diagnostic is trying
   to measure. Every feature is computed over the cuboid silhouette only.
2. **The descriptor is invariant to how many pixels the vehicle happens to
   cover.** Ego sees an object at 15 m and the CAV at 55 m; a descriptor that
   moved with crop size would score that pair low for a reason that has
   nothing to do with identity.
3. **Occlusion is charged for.** A car that is 80% behind a van is a crop of
   the van, and a matcher fed that would be learning the occluder's colour.
"""

import math

import numpy as np
import pytest

from embedding_aware_belt_fusion.alignformer.camera import (
    MIN_CROP_PIXELS,
    appearance,
    cosine_similarity,
    occlusion_fraction,
    project_corners,
    silhouette_mask,
    vehicle_world_corners,
    world_from_pose,
)

IMAGE_SHAPE = (600, 800, 3)
# The intrinsics every OPV2V frame carries: 800x600 at a 100-degree horizontal
# field of view. Testing off a different camera would test a different problem.
INTRINSIC = [[335.639852470912, 0.0, 400.0], [0.0, 335.639852470912, 300.0], [0.0, 0.0, 1.0]]


def _camera(x=0.0, y=0.0, z=1.0, yaw=0.0):
    return {"cords": [x, y, z, 0.0, yaw, 0.0], "intrinsic": INTRINSIC}


def _vehicle(x, y=0.0, z=0.0, yaw=0.0, extent=(2.28, 1.0, 0.75)):
    return {
        "center": [0.0, 0.0, 0.0],
        "location": [x, y, z],
        "angle": [0.0, yaw, 0.0],
        "extent": list(extent),
    }


# --------------------------------------------------------------------------
# GEOMETRY
# --------------------------------------------------------------------------


def test_a_vehicle_straight_ahead_projects_to_the_principal_point():
    # CARLA sensor axes are (forward, right, up) and OpenCV's are
    # (right, down, forward). Getting that permutation wrong still produces
    # pixels, just the wrong ones, so it is pinned against a case whose answer
    # is known by symmetry rather than by running the code.
    corners = vehicle_world_corners(_vehicle(x=30.0, y=0.0, z=1.0))

    projected = project_corners(corners, _camera(), IMAGE_SHAPE)

    assert projected is not None
    centre = projected.pixels.mean(axis=0)
    assert centre[0] == pytest.approx(400.0, abs=1.0)
    assert centre[1] == pytest.approx(300.0, abs=1.0)
    assert projected.depth == pytest.approx(30.0, rel=0.02)


def test_a_vehicle_behind_the_camera_does_not_project():
    corners = vehicle_world_corners(_vehicle(x=-20.0, y=0.0))

    assert project_corners(corners, _camera(), IMAGE_SHAPE) is None


def test_a_vehicle_to_the_left_lands_left_of_centre():
    corners = vehicle_world_corners(_vehicle(x=30.0, y=-10.0, z=1.0))

    projected = project_corners(corners, _camera(), IMAGE_SHAPE)

    assert projected is not None
    assert projected.pixels[:, 0].mean() < 400.0


def test_apparent_width_falls_as_one_over_range():
    near = project_corners(
        vehicle_world_corners(_vehicle(x=20.0, z=1.0)), _camera(), IMAGE_SHAPE
    )
    far = project_corners(
        vehicle_world_corners(_vehicle(x=60.0, z=1.0)), _camera(), IMAGE_SHAPE
    )

    near_width = near.box[2] - near.box[0]
    far_width = far.box[2] - far.box[0]
    assert far_width == pytest.approx(near_width / 3.0, rel=0.15)


def test_the_pose_transform_round_trips():
    pose = [12.0, -3.0, 1.5, 4.0, 37.0, -2.0]
    point = np.array([1.0, 2.0, 3.0, 1.0])

    forward = world_from_pose(pose)
    recovered = np.linalg.inv(forward) @ (forward @ point)

    assert np.allclose(recovered, point, atol=1e-9)


# --------------------------------------------------------------------------
# THE CROP IS THE VEHICLE, NOT THE ROAD
# --------------------------------------------------------------------------


def test_the_silhouette_excludes_the_background_inside_the_bounding_box():
    # A rotated car fills well under its axis-aligned box. If the mask were
    # the box, the corners would be road, and road matches road.
    corners = vehicle_world_corners(_vehicle(x=18.0, z=1.0, yaw=35.0))
    projected = project_corners(corners, _camera(), IMAGE_SHAPE)

    mask = silhouette_mask(projected.pixels, IMAGE_SHAPE)

    x1, y1, x2, y2 = projected.box
    box_area = (x2 - x1) * (y2 - y1)
    assert 0 < int(mask.sum()) < box_area


def test_a_masked_crop_ignores_what_surrounds_the_vehicle():
    # Same vehicle pixels, wildly different surroundings: the descriptor must
    # not move. This is the test that fails if the mask is ever dropped.
    corners = vehicle_world_corners(_vehicle(x=18.0, z=1.0))
    projected = project_corners(corners, _camera(), IMAGE_SHAPE)
    mask = silhouette_mask(projected.pixels, IMAGE_SHAPE)

    red = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    red[mask] = (40, 40, 200)
    on_black = appearance(red, mask)
    red[~mask] = (10, 200, 10)
    on_green = appearance(red, mask)

    for channel in on_black:
        assert cosine_similarity(on_black[channel], on_green[channel]) == pytest.approx(
            1.0, abs=1e-6
        )


# --------------------------------------------------------------------------
# THE DESCRIPTOR IS ABOUT IDENTITY, NOT ABOUT RANGE
# --------------------------------------------------------------------------


def test_the_same_colour_at_two_ranges_still_matches():
    # Ego at 15 m, the CAV at 55 m, one vehicle. A descriptor that moved with
    # the pixel count would score this low for the wrong reason.
    scores = []
    for distance in (15.0, 55.0):
        corners = vehicle_world_corners(_vehicle(x=distance, z=1.0))
        projected = project_corners(corners, _camera(), IMAGE_SHAPE)
        mask = silhouette_mask(projected.pixels, IMAGE_SHAPE)
        image = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
        image[mask] = (40, 40, 200)
        scores.append(appearance(image, mask))

    assert cosine_similarity(scores[0]["hue_saturation"], scores[1]["hue_saturation"]) > 0.95


def test_two_different_colours_do_not_match():
    corners = vehicle_world_corners(_vehicle(x=18.0, z=1.0))
    projected = project_corners(corners, _camera(), IMAGE_SHAPE)
    mask = silhouette_mask(projected.pixels, IMAGE_SHAPE)

    descriptors = []
    for colour in ((40, 40, 200), (200, 60, 40)):
        image = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
        image[mask] = colour
        descriptors.append(appearance(image, mask))

    assert cosine_similarity(
        descriptors[0]["hue_saturation"], descriptors[1]["hue_saturation"]
    ) < 0.2


def test_a_crop_with_too_few_pixels_is_refused():
    mask = np.zeros(IMAGE_SHAPE[:2], dtype=bool)
    mask[:4, :4] = True
    image = np.zeros(IMAGE_SHAPE, dtype=np.uint8)

    assert mask.sum() < MIN_CROP_PIXELS
    assert appearance(image, mask) is None


def test_the_pixel_floor_does_not_silently_delete_the_far_field():
    # THE constraint on this whole approach, pinned so a future tightening of
    # MIN_CROP_PIXELS has to argue with a test. A vehicle at 55 m covers about
    # 110 masked pixels at the deployed intrinsics, so a floor of 200 -- which
    # is what "a 20x20 crop" sounds reasonable for -- would exclude everything
    # past roughly 45 m. That is precisely the range band where the pose fit
    # is worst and where colour would have to earn its place.
    corners = vehicle_world_corners(_vehicle(x=55.0, z=1.0))
    projected = project_corners(corners, _camera(), IMAGE_SHAPE)
    mask = silhouette_mask(projected.pixels, IMAGE_SHAPE)

    assert int(mask.sum()) < 200
    assert int(mask.sum()) >= MIN_CROP_PIXELS


def test_cosine_similarity_of_a_zero_descriptor_is_zero_not_a_nan():
    assert cosine_similarity(np.zeros(8), np.ones(8)) == 0.0


# --------------------------------------------------------------------------
# OCCLUSION IS CHARGED FOR
# --------------------------------------------------------------------------


def test_a_nearer_vehicle_occludes_a_farther_one():
    camera = _camera()
    far = project_corners(
        vehicle_world_corners(_vehicle(x=45.0, z=1.0)), camera, IMAGE_SHAPE
    )
    near = project_corners(
        vehicle_world_corners(_vehicle(x=12.0, z=1.0, extent=(2.28, 2.0, 1.6))),
        camera,
        IMAGE_SHAPE,
    )

    covered = occlusion_fraction(far, [near], IMAGE_SHAPE)

    assert covered > 0.9


def test_a_vehicle_with_nothing_in_front_of_it_is_unoccluded():
    camera = _camera()
    target = project_corners(
        vehicle_world_corners(_vehicle(x=25.0, z=1.0)), camera, IMAGE_SHAPE
    )
    beside = project_corners(
        vehicle_world_corners(_vehicle(x=25.0, y=12.0, z=1.0)), camera, IMAGE_SHAPE
    )

    assert occlusion_fraction(target, [beside], IMAGE_SHAPE) == pytest.approx(0.0, abs=1e-6)


def test_a_vehicle_behind_the_target_does_not_occlude_it():
    # Only NEARER silhouettes may occlude. Ordering by depth is the whole
    # content of the test: a painter's-algorithm bug reads as heavy occlusion
    # on exactly the near, well-resolved objects the probe most wants.
    camera = _camera()
    target = project_corners(
        vehicle_world_corners(_vehicle(x=12.0, z=1.0)), camera, IMAGE_SHAPE
    )
    behind = project_corners(
        vehicle_world_corners(_vehicle(x=40.0, z=1.0)), camera, IMAGE_SHAPE
    )

    assert occlusion_fraction(target, [behind], IMAGE_SHAPE) == pytest.approx(0.0, abs=1e-6)
