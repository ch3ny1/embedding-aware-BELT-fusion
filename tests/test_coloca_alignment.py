"""End-to-end check that the regressed correction actually re-aligns the clouds.

The dataset projects the CAV point cloud with a full SE(3) transform built from
the noisy 6-DoF pose, but the label is an SE(2) correction.  That mismatch is
the one place a subtle sign or frame error would hide, so this asserts the
round trip the paper's Eq. (6)-(8) promises::

    C_{4x4} @ (T_noisy @ P) == T_true @ P
"""

import numpy as np
import pytest

from embedding_aware_belt_fusion.coloca.geometry import relative_pose_error, se2_matrix

pytest.importorskip("opencood", reason="requires OpenCOOD on PYTHONPATH")

from opencood.utils.transformation_utils import x1_to_x2


def embed_se2_in_se3(dx: float, dy: float, dpsi_deg: float) -> np.ndarray:
    """Paper Eq. (6): lift the planar correction into a 4x4 transform."""
    planar = se2_matrix(dx, dy, dpsi_deg)
    transform = np.eye(4)
    transform[:2, :2] = planar[:2, :2]
    transform[:2, 3] = planar[:2, 2]
    return transform


def project(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def make_points(rng: np.random.Generator, count: int = 2000) -> np.ndarray:
    return np.column_stack(
        [
            rng.uniform(-100, 100, count),
            rng.uniform(-38, 38, count),
            rng.uniform(-3, 1, count),
        ]
    )


def test_correction_realigns_the_projected_cloud_on_flat_ground():
    # Arrange: planar poses, i.e. roll = pitch = 0 as in most of OPV2V
    rng = np.random.default_rng(0)
    for _ in range(20):
        ego = [*rng.uniform(-200, 200, 2), 1.9, 0.0, rng.uniform(-180, 180), 0.0]
        cav_true = [*rng.uniform(-200, 200, 2), 1.9, 0.0, rng.uniform(-180, 180), 0.0]
        cav_noisy = list(cav_true)
        cav_noisy[0] += rng.normal(0, 2.0)
        cav_noisy[1] += rng.normal(0, 2.0)
        cav_noisy[4] += rng.normal(0, 1.0)
        points = make_points(rng)

        # Act: project with the noisy pose, then apply the regressed correction
        noisy_cloud = project(points, x1_to_x2(cav_noisy, ego))
        true_cloud = project(points, x1_to_x2(cav_true, ego))
        correction = embed_se2_in_se3(*relative_pose_error(ego, cav_true, cav_noisy))
        corrected = project(noisy_cloud, correction)

        # Assert
        np.testing.assert_allclose(corrected, true_cloud, atol=1e-6)


def test_uncorrected_misalignment_matches_the_injected_noise_magnitude():
    # Arrange: a pure 2 m lateral pose error with no rotation anywhere
    rng = np.random.default_rng(1)
    ego = [0.0, 0.0, 1.9, 0.0, 0.0, 0.0]
    cav_true = [20.0, 0.0, 1.9, 0.0, 0.0, 0.0]
    cav_noisy = [22.0, 0.0, 1.9, 0.0, 0.0, 0.0]
    points = make_points(rng)

    # Act
    noisy_cloud = project(points, x1_to_x2(cav_noisy, ego))
    true_cloud = project(points, x1_to_x2(cav_true, ego))
    displacement = np.linalg.norm(noisy_cloud - true_cloud, axis=1)

    # Assert: every point is displaced by exactly the 2 m pose error
    assert displacement.mean() == pytest.approx(2.0, abs=1e-5)


# Measured over the 22,672 OPV2V train pairs: ego roll is negligible
# (std 0.27 deg) but pitch is not (std 2.85 deg, p99 12.3 deg, max 17.1 deg),
# which exceeds the "within 5 degrees" assumption the paper states in Sec. III-A.
OPV2V_TRAIN_EGO_TILT = {"median": (0.0, 0.0), "p99": (0.6, 12.3)}


@pytest.mark.parametrize("tilt", sorted(OPV2V_TRAIN_EGO_TILT))
def test_residual_is_driven_by_ego_tilt_not_by_the_label(tilt: str):
    """The SE(2) label is exact; only 3-D re-alignment degrades with ego tilt.

    The correction is conjugated by the ego pose, so a world-frame yaw error
    stays a pure z-rotation only when the *ego* is level.  This quantifies what
    that costs rather than assuming it away.
    """
    # Arrange
    roll, pitch = OPV2V_TRAIN_EGO_TILT[tilt]
    rng = np.random.default_rng(2)
    residuals, misalignments = [], []

    for _ in range(30):
        ego = [*rng.uniform(-200, 200, 2), 1.9, roll, rng.uniform(-180, 180), pitch]
        cav_true = [
            *rng.uniform(-200, 200, 2), 1.9,
            rng.uniform(-1, 1), rng.uniform(-180, 180), rng.uniform(-5, 5),
        ]
        cav_noisy = list(cav_true)
        cav_noisy[0] += rng.normal(0, 2.0)
        cav_noisy[1] += rng.normal(0, 2.0)
        cav_noisy[4] += rng.normal(0, 1.0)
        points = make_points(rng)

        # Act
        noisy_cloud = project(points, x1_to_x2(cav_noisy, ego))
        true_cloud = project(points, x1_to_x2(cav_true, ego))
        correction = embed_se2_in_se3(*relative_pose_error(ego, cav_true, cav_noisy))
        residuals.append(np.linalg.norm(project(noisy_cloud, correction) - true_cloud, axis=1).mean())
        misalignments.append(np.linalg.norm(noisy_cloud - true_cloud, axis=1).mean())

    residual, misalignment = float(np.mean(residuals)), float(np.mean(misalignments))

    # Assert
    if tilt == "median":
        # A level ego makes the planar correction exact.
        assert residual < 1e-9
    else:
        # At p99 pitch it is ~0.34 m, i.e. the correction still removes ~85% of
        # the misalignment. Documented in docs/coloca_qua_baseline.md because it
        # is a floor on 3-D alignment, though not on the reported pose-error MAE.
        assert 0.2 < residual < 0.5
        assert residual < 0.2 * misalignment
