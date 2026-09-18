"""Tests for CoLoca-QuA SE(2) pose-error geometry.

The label definition follows CoLoca-QuA Eq. (5)-(7): the network regresses an
SE(2) correction ``C`` such that ``C @ T_noisy == T_true``, where both
transforms map the CAV frame into the ego frame.
"""

import numpy as np
import pytest

from embedding_aware_belt_fusion.coloca.geometry import (
    perturb_pose_2d,
    pose_from_se2,
    relative_pose_error,
    se2_from_pose,
    se2_matrix,
)


def test_se2_from_pose_uses_only_x_y_and_yaw():
    # Arrange: roll/pitch are non-zero but must be ignored by the SE(2) projection
    pose = [3.0, -4.0, 17.0, 5.0, 30.0, -7.0]  # x, y, z, roll, yaw, pitch

    # Act
    matrix = se2_from_pose(pose)

    # Assert
    np.testing.assert_allclose(matrix, se2_matrix(3.0, -4.0, 30.0), atol=1e-12)


def test_pose_from_se2_round_trips_se2_matrix():
    # Arrange
    x, y, yaw_deg = 12.5, -3.25, 123.0

    # Act
    recovered = pose_from_se2(se2_matrix(x, y, yaw_deg))

    # Assert
    np.testing.assert_allclose(recovered, (x, y, yaw_deg), atol=1e-10)


def test_pose_from_se2_wraps_yaw_into_180_degree_range():
    # Arrange: 190 degrees must come back as -170, not 190
    matrix = se2_matrix(0.0, 0.0, 190.0)

    # Act
    _, _, yaw_deg = pose_from_se2(matrix)

    # Assert
    assert yaw_deg == pytest.approx(-170.0, abs=1e-10)


def test_pure_translation_noise_yields_opposite_translation_correction():
    # Arrange: CAV truly 10 m ahead of ego, reported 2 m too far away
    ego_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cav_true = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cav_noisy = [12.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Act
    dx, dy, dpsi = relative_pose_error(ego_pose, cav_true, cav_noisy)

    # Assert
    assert (dx, dy, dpsi) == pytest.approx((-2.0, 0.0, 0.0), abs=1e-10)


def test_yaw_noise_at_a_lever_arm_induces_translation_correction():
    # Arrange: heading-only error on a CAV 10 m away still moves its frame origin
    ego_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cav_true = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cav_noisy = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0]

    # Act
    dx, dy, dpsi = relative_pose_error(ego_pose, cav_true, cav_noisy)

    # Assert
    assert dpsi == pytest.approx(-10.0, abs=1e-10)
    expected_dx = 10.0 - 10.0 * np.cos(np.radians(10.0))
    expected_dy = 10.0 * np.sin(np.radians(10.0))
    assert (dx, dy) == pytest.approx((expected_dx, expected_dy), abs=1e-10)


def test_correction_exactly_recovers_the_true_relative_transform():
    # Arrange: arbitrary ego/CAV poses and an arbitrary perturbation
    rng = np.random.default_rng(0)
    for _ in range(50):
        ego_pose = [*rng.uniform(-200, 200, 2), 1.9, 0.0, rng.uniform(-180, 180), 0.0]
        cav_true = [*rng.uniform(-200, 200, 2), 1.9, 0.0, rng.uniform(-180, 180), 0.0]
        cav_noisy = perturb_pose_2d(cav_true, xy_std=2.0, yaw_std_deg=1.0, rng=rng)

        # Act
        dx, dy, dpsi = relative_pose_error(ego_pose, cav_true, cav_noisy)
        correction = se2_matrix(dx, dy, dpsi)
        t_noisy = np.linalg.inv(se2_from_pose(ego_pose)) @ se2_from_pose(cav_noisy)
        t_true = np.linalg.inv(se2_from_pose(ego_pose)) @ se2_from_pose(cav_true)

        # Assert
        np.testing.assert_allclose(correction @ t_noisy, t_true, atol=1e-9)


def test_zero_noise_yields_zero_error():
    # Arrange
    ego_pose = [5.0, 6.0, 1.9, 0.0, 42.0, 0.0]
    cav_pose = [-3.0, 11.0, 1.9, 0.0, -17.0, 0.0]

    # Act
    error = relative_pose_error(ego_pose, cav_pose, cav_pose)

    # Assert
    assert error == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)


def test_perturb_pose_2d_does_not_mutate_the_input_pose():
    # Arrange
    original = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    snapshot = list(original)

    # Act
    perturb_pose_2d(original, xy_std=2.0, yaw_std_deg=1.0, rng=np.random.default_rng(1))

    # Assert
    assert original == snapshot


def test_perturb_pose_2d_only_touches_x_y_and_yaw():
    # Arrange
    original = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    # Act
    noisy = perturb_pose_2d(original, xy_std=2.0, yaw_std_deg=1.0, rng=np.random.default_rng(1))

    # Assert
    assert noisy[2] == original[2]  # z
    assert noisy[3] == original[3]  # roll
    assert noisy[5] == original[5]  # pitch
    assert noisy[0] != original[0] and noisy[1] != original[1] and noisy[4] != original[4]


def test_perturb_pose_2d_matches_the_requested_standard_deviations():
    # Arrange
    rng = np.random.default_rng(7)
    pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Act
    samples = np.array(
        [perturb_pose_2d(pose, xy_std=2.0, yaw_std_deg=1.0, rng=rng) for _ in range(20000)]
    )

    # Assert
    assert samples[:, 0].std() == pytest.approx(2.0, rel=0.05)
    assert samples[:, 1].std() == pytest.approx(2.0, rel=0.05)
    assert samples[:, 4].std() == pytest.approx(1.0, rel=0.05)


def test_perturb_pose_2d_rejects_negative_standard_deviations():
    # Arrange
    pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Act / Assert
    with pytest.raises(ValueError, match="xy_std"):
        perturb_pose_2d(pose, xy_std=-1.0, yaw_std_deg=1.0, rng=np.random.default_rng(0))


def test_se2_from_pose_rejects_short_poses():
    # Act / Assert
    with pytest.raises(ValueError, match="6 elements"):
        se2_from_pose([1.0, 2.0, 3.0])
