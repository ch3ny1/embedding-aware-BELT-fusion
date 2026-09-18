"""CoLoca-QuA: cooperative relative pose estimation with query-based attention.

Reproduction of Gao et al., "CoLoca-QuA: Cooperative Relative Pose Estimation
With Query-Based Attention Using Neural Intermediate Features in V2X Network",
IEEE T-VT 75(7), 2026, used here as a localization baseline on OPV2V.
"""

from embedding_aware_belt_fusion.coloca.geometry import (
    perturb_pose_2d,
    pose_from_se2,
    relative_pose_error,
    se2_from_pose,
    se2_matrix,
)

__all__ = [
    "perturb_pose_2d",
    "pose_from_se2",
    "relative_pose_error",
    "se2_from_pose",
    "se2_matrix",
]
