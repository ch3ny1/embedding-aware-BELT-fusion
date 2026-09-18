"""SE(2) pose-error geometry for the CoLoca-QuA baseline.

CoLoca-QuA regresses the residual relative pose error between the ego vehicle
and a CAV whose self-reported pose is corrupted by localization noise.  Section
III-A of the paper restricts the problem to the ground plane, so all labels
here live in SE(2): translation along x and y plus a yaw correction.

Conventions
-----------
Poses use the OPV2V/CARLA layout ``[x, y, z, roll, yaw, pitch]`` with angles in
degrees; only ``x``, ``y`` and ``yaw`` participate in the SE(2) projection.

A relative transform ``T_j^i`` maps points from agent ``j``'s frame into ego
agent ``i``'s frame (paper Eq. 1).  The regression target is the correction
``C_j^i`` of Eq. (5), defined so that applying it to the noisy estimate
recovers the true transform, matching Eq. (7)::

    C @ T_noisy == T_true
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

POSE_LENGTH = 6
_X, _Y, _YAW = 0, 1, 4


def se2_matrix(x: float, y: float, yaw_deg: float) -> np.ndarray:
    """Build the 3x3 homogeneous SE(2) matrix for a planar pose."""
    yaw = np.radians(yaw_deg)
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cos_yaw, -sin_yaw, x],
            [sin_yaw, cos_yaw, y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def se2_from_pose(pose: Sequence[float]) -> np.ndarray:
    """Project a 6-DoF CARLA pose onto SE(2), dropping z, roll and pitch."""
    _validate_pose(pose)
    return se2_matrix(pose[_X], pose[_Y], pose[_YAW])


def pose_from_se2(matrix: np.ndarray) -> tuple[float, float, float]:
    """Decompose an SE(2) matrix into ``(x, y, yaw_deg)`` with yaw in (-180, 180]."""
    if matrix.shape != (3, 3):
        raise ValueError(f"expected a 3x3 SE(2) matrix, got shape {matrix.shape}")
    yaw_deg = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    return float(matrix[0, 2]), float(matrix[1, 2]), yaw_deg


def relative_pose_error(
    ego_pose: Sequence[float],
    cav_pose_true: Sequence[float],
    cav_pose_noisy: Sequence[float],
) -> tuple[float, float, float]:
    """Return the SE(2) correction ``(dx, dy, dpsi_deg)`` for a noisy CAV pose.

    The correction is expressed in the ego frame and satisfies
    ``se2_matrix(dx, dy, dpsi) @ T_noisy == T_true``, where ``T_*`` are the
    CAV-to-ego relative transforms built from the respective poses.
    """
    world_to_ego = np.linalg.inv(se2_from_pose(ego_pose))
    t_true = world_to_ego @ se2_from_pose(cav_pose_true)
    t_noisy = world_to_ego @ se2_from_pose(cav_pose_noisy)
    return pose_from_se2(t_true @ np.linalg.inv(t_noisy))


def perturb_pose_2d(
    pose: Sequence[float],
    xy_std: float,
    yaw_std_deg: float,
    rng: np.random.Generator,
) -> list[float]:
    """Return a copy of ``pose`` with Gaussian noise on x, y and yaw.

    Matches the paper's noise model (Section IV-B): zero-mean Gaussian
    position noise applied independently per axis, plus zero-mean Gaussian
    heading noise.  ``z``, ``roll`` and ``pitch`` are left untouched because the
    method only estimates the planar pose.
    """
    _validate_pose(pose)
    if xy_std < 0:
        raise ValueError(f"xy_std must be non-negative, got {xy_std}")
    if yaw_std_deg < 0:
        raise ValueError(f"yaw_std_deg must be non-negative, got {yaw_std_deg}")

    noisy = list(map(float, pose))
    noisy[_X] += float(rng.normal(0.0, xy_std))
    noisy[_Y] += float(rng.normal(0.0, xy_std))
    noisy[_YAW] += float(rng.normal(0.0, yaw_std_deg))
    return noisy


def _validate_pose(pose: Sequence[float]) -> None:
    if len(pose) < POSE_LENGTH:
        raise ValueError(f"pose must have {POSE_LENGTH} elements [x,y,z,roll,yaw,pitch]")
