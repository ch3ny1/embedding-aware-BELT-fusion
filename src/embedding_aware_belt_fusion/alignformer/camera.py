"""Per-object appearance from OPV2V's four RGB cameras.

The measured null this exists to re-open
----------------------------------------

AlignFormer's LiDAR appearance embedding contributes nothing to cross-agent
association on OPV2V: ``+0.0001``, 95% CI ``[-0.0013, +0.0014]``. Three
independent reasons were recorded, and the first is about *shape* --
separability of a true partner from its nearest competitor is AUC 0.560 on raw
pooled ROI features, competing vehicles differ by a median 0.071 m in width,
and all 65,774 CAV box widths are ``2.014 +/- 0.081`` m. CARLA's asset library
is small and the competitors are the same object.

Colour is outside that argument, because LiDAR does not measure it. A first
probe over 118 vehicle views puts the mean-BGR spread *within* one identity
(across frames and cameras) at 3.2 and *between* identities at 32.6. This
module is the machinery to ask the cross-agent version of that question
properly, and the one reason it is not already answered is that nothing here
read the cameras before.

The third recorded reason is untouched by any of this and still binds: **no
ego object in the validation split has a competitor within 2 m**, so at low
localization error there is nothing to disambiguate and no feature can help.
Any gain colour can offer lives at ``sigma >= 1.5 m``, where displacement
overlaps neighbours -- which is also the one cell where FreeAlign still leads.

Three things here are easy to get wrong in a way that still yields a
plausible AUC, so each is a test rather than a comment.

*The crop must be the vehicle.* An axis-aligned box around a car at 60 m is
mostly asphalt, and asphalt matches asphalt: an unmasked crop manufactures the
separation the diagnostic is trying to measure. Everything is computed over
the projected cuboid silhouette.

*The descriptor must not move with apparent size.* Ego may see an object at
15 m and the CAV at 55 m, a three-fold difference in width. A normalized
hue-saturation histogram over the masked pixels is invariant to that; a raw
pixel vector is not.

*Occlusion must be charged for.* A car 80% behind a van is a crop of the van.

Duplication note: ``scripts/opv2v_camera_bbox_viewer.py`` carries its own copy
of the pose and projection arithmetic. It is an interactive viewer and it is
left alone deliberately -- this module is the tested one, and a measurement
should not depend on a tool whose job is to draw rectangles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

# Masked pixels a vehicle must cover before its colour is a statistic rather
# than a handful of samples. Chosen from the spreads rather than by feel: the
# per-pixel variation within one vehicle is about 30 grey levels (paint plus
# specularity), identities are separated by about 32.6, so the standard error
# of a mean colour over n pixels is 30/sqrt(n) and n = 36 already puts it at
# a sixth of the separation being measured. Rounded up to 50.
#
# This threshold has teeth and they point the wrong way. At the deployed
# intrinsics (f = 335.6 px) a 2.0 x 1.5 m vehicle covers about 12 x 9 px at
# 55 m, which is 110 pixels before the hull is taken, so a floor of 200 would
# have silently excluded everything past roughly 45 m -- the far field this
# whole line of work exists to improve. The diagnostic reports coverage BY
# RANGE for exactly this reason: the honest limit of the method is how far out
# a vehicle still covers enough pixels to have a colour.
MIN_CROP_PIXELS = 50

# Hue and saturation bins. Hue carries the identity and saturation separates a
# coloured car from a grey one; value is dropped on purpose, because it is
# mostly incident light and differs between two agents looking from different
# sides at the same vehicle.
HUE_BINS = 24
SATURATION_BINS = 8

_EPSILON = 1e-12


@dataclass(frozen=True)
class Projection:
    """One vehicle's cuboid as it lands in one camera."""

    pixels: np.ndarray
    box: Tuple[int, int, int, int]
    depth: float


def world_from_pose(pose: Sequence[float]) -> np.ndarray:
    """OPV2V/CARLA sensor-local to world transform for ``[x, y, z, r, y, p]``."""
    x, y, z, roll, yaw, pitch = (float(value) for value in pose)
    cy, sy = np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))
    cr, sr = np.cos(np.deg2rad(roll)), np.sin(np.deg2rad(roll))
    cp, sp = np.cos(np.deg2rad(pitch)), np.sin(np.deg2rad(pitch))
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = (x, y, z)
    matrix[:3, :3] = np.array(
        [
            [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
            [sp, -cp * sr, cp * cr],
        ]
    )
    return matrix


def vehicle_world_corners(vehicle: Dict) -> np.ndarray:
    """``(8, 3)`` world-coordinate corners of one OPV2V vehicle annotation."""
    extent = np.asarray(vehicle["extent"], dtype=np.float64)
    signs = np.array(
        [
            [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1],
            [1, -1, 1], [1, 1, 1], [-1, 1, 1], [-1, -1, 1],
        ],
        dtype=np.float64,
    )
    local = signs * extent
    centre = np.asarray(vehicle["center"], dtype=np.float64)
    location = np.asarray(vehicle["location"], dtype=np.float64)
    pose = [*(location + centre), *vehicle["angle"]]
    homogeneous = np.concatenate((local, np.ones((8, 1))), axis=1)
    return (world_from_pose(pose) @ homogeneous.T).T[:, :3]


def project_corners(
    corners_world: np.ndarray, camera: Dict, image_shape: Tuple[int, int, int]
) -> Optional[Projection]:
    """Project a fully front-facing cuboid, or ``None`` if it is not one.

    CARLA sensor coordinates are (forward, right, up) and OpenCV's pinhole
    coordinates are (right, down, forward), which is the ``[y, -z, x]``
    permutation below. A box with any corner at or behind the image plane is
    rejected outright rather than clipped: a partially projected cuboid has no
    silhouette worth cropping, and the alternative is a crop whose contents
    depend on where the clip happened to fall.
    """
    world_to_camera = np.linalg.inv(world_from_pose(camera["cords"]))
    homogeneous = np.concatenate((corners_world, np.ones((8, 1))), axis=1)
    sensor = (world_to_camera @ homogeneous.T).T[:, :3]
    camera_xyz = np.stack((sensor[:, 1], -sensor[:, 2], sensor[:, 0]), axis=1)
    depth = camera_xyz[:, 2]
    if np.any(depth <= 1e-3):
        return None

    intrinsic = np.asarray(camera["intrinsic"], dtype=np.float64)
    image_points = (intrinsic @ camera_xyz.T).T
    pixels = image_points[:, :2] / image_points[:, 2:3]

    height, width = image_shape[:2]
    x1 = max(0, int(np.floor(pixels[:, 0].min())))
    y1 = max(0, int(np.floor(pixels[:, 1].min())))
    x2 = min(width - 1, int(np.ceil(pixels[:, 0].max())))
    y2 = min(height - 1, int(np.ceil(pixels[:, 1].max())))
    if x1 >= x2 or y1 >= y2:
        return None
    return Projection(pixels=pixels, box=(x1, y1, x2, y2), depth=float(depth.mean()))


def silhouette_mask(
    pixels: np.ndarray, image_shape: Tuple[int, int, int]
) -> np.ndarray:
    """Boolean mask of the cuboid's convex hull, clipped to the image.

    The hull rather than the axis-aligned box: a car seen at an angle fills
    well under its bounding rectangle, and the remainder is road.
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    hull = cv2.convexHull(np.rint(pixels).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 1)
    return mask.astype(bool)


def occlusion_fraction(
    target: Projection,
    others: Sequence[Optional[Projection]],
    image_shape: Tuple[int, int, int],
) -> float:
    """Fraction of ``target``'s silhouette covered by NEARER silhouettes.

    Only nearer ones: a painter's-algorithm slip that let farther vehicles
    occlude would report heavy occlusion on precisely the near, well-resolved
    objects this probe most wants to keep.
    """
    mask = silhouette_mask(target.pixels, image_shape)
    total = int(mask.sum())
    if total == 0:
        return 1.0
    covered = np.zeros(image_shape[:2], dtype=bool)
    for other in others:
        if other is None or other.depth >= target.depth:
            continue
        covered |= silhouette_mask(other.pixels, image_shape)
    return float(np.count_nonzero(mask & covered) / total)


def appearance(
    image: np.ndarray, mask: np.ndarray
) -> Optional[Dict[str, np.ndarray]]:
    """Descriptors of the masked pixels, or ``None`` if there are too few.

    Two, because they fail differently. ``hue_saturation`` is the joint
    histogram, normalized, so it is invariant to how many pixels the vehicle
    covers -- the property that lets a 15 m view be compared with a 55 m one.
    ``mean_colour`` is the three-number version, kept because a histogram that
    beat it would be carrying texture and not merely paint, and that is worth
    knowing separately.

    Value is deliberately excluded from the histogram: between two agents
    viewing one vehicle from different sides it is mostly incident light.
    """
    if int(mask.sum()) < MIN_CROP_PIXELS:
        return None
    pixels = image[mask]
    if pixels.size == 0:
        return None

    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv], [0, 1], None, [HUE_BINS, SATURATION_BINS], [0, 180, 0, 256]
    ).flatten()
    total = histogram.sum()
    if total <= 0.0:
        return None
    return {
        "hue_saturation": histogram / total,
        "mean_colour": pixels.mean(axis=0).astype(np.float64),
    }


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine of two descriptors; zero, never a NaN, if either has no mass."""
    left = np.asarray(left, dtype=np.float64).ravel()
    right = np.asarray(right, dtype=np.float64).ravel()
    norms = np.linalg.norm(left) * np.linalg.norm(right)
    if norms < _EPSILON:
        return 0.0
    return float(np.dot(left, right) / norms)


__all__ = [
    "HUE_BINS",
    "MIN_CROP_PIXELS",
    "SATURATION_BINS",
    "Projection",
    "appearance",
    "cosine_similarity",
    "occlusion_fraction",
    "project_corners",
    "silhouette_mask",
    "vehicle_world_corners",
    "world_from_pose",
]
