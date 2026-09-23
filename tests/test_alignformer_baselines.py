"""The baseline comparison's own invariants.

The head-to-head table is only a comparison if the baselines see the same
perturbation AlignFormer saw, on the same frames, and if nothing in the noise
path can reach the ground truth. These are the properties that, if they broke,
would leave the table looking fine and meaning nothing.
"""

from collections import OrderedDict

import numpy as np
import pytest
import torch

from embedding_aware_belt_fusion.alignformer.bandwidth import (
    BYTES_PER_FLOAT32,
    MESSAGE_SITES,
    MessageProbe,
    _resolve,
    alignformer_message_bytes,
)
from embedding_aware_belt_fusion.alignformer.baselines import (
    BASELINES,
    _apply_transforms,
    noisy_transforms,
)
from embedding_aware_belt_fusion.alignformer.dataset import YAW_STD_PER_XY_STD
from embedding_aware_belt_fusion.alignformer.noisy_fusion import _sweep_rng
from embedding_aware_belt_fusion.coloca.geometry import perturb_pose_2d

EGO_POSE = [10.0, -4.0, 1.0, 0.0, 35.0, 0.0]


def _frame(poses):
    """A minimal ``base_data_dict``: ego first, then the named CAVs."""
    frame = OrderedDict()
    frame["ego"] = {"ego": True, "params": {"lidar_pose": list(EGO_POSE)}}
    for cav_id, pose in poses.items():
        frame[cav_id] = {"ego": False, "params": {"lidar_pose": list(pose)}}
    return frame


def test_the_baseline_draws_the_same_displacement_alignformer_drew():
    # Arrange: the agent ordering noisy_fusion uses is sorted() over the
    # non-ego keys, and agent index 1 must get draw number 1, not number 0.
    frame = _frame({"641": [40.0, -4.0, 1.0, 0.0, 10.0, 0.0],
                    "310": [20.0, 6.0, 1.0, 0.0, -80.0, 0.0]})

    # Act
    transforms = noisy_transforms(
        frame, "ego", EGO_POSE, sigma=0.8, seed=0, frame=17
    )

    # Assert: reconstruct what noisy_fusion would have produced for agent 1.
    from opencood.utils.transformation_utils import x1_to_x2

    expected_pose = perturb_pose_2d(
        frame["641"]["params"]["lidar_pose"],
        0.8,
        0.8 * YAW_STD_PER_XY_STD,
        _sweep_rng(0, 0.8, 17, 1),  # "641" sorts after "310"
    )
    assert np.allclose(transforms["641"], x1_to_x2(expected_pose, EGO_POSE))


def test_an_out_of_range_cav_does_not_shift_the_other_agents_draws():
    # Arrange: AlignFormer drops CAVs beyond COM_RANGE before it enumerates
    # agents, so a far CAV that sorts first must not consume draw 0 here.
    import opencood.data_utils.datasets as opencood_datasets

    far = EGO_POSE[0] + opencood_datasets.COM_RANGE + 50.0
    near_only = _frame({"641": [40.0, -4.0, 1.0, 0.0, 10.0, 0.0]})
    with_far = _frame({"310": [far, -4.0, 1.0, 0.0, 10.0, 0.0],
                       "641": [40.0, -4.0, 1.0, 0.0, 10.0, 0.0]})

    # Act
    without = noisy_transforms(near_only, "ego", EGO_POSE, sigma=1.0, seed=0, frame=3)
    with_it = noisy_transforms(with_far, "ego", EGO_POSE, sigma=1.0, seed=0, frame=3)

    # Assert
    assert "310" not in with_it
    assert np.allclose(without["641"], with_it["641"])


def test_sigma_zero_leaves_the_transform_exact():
    # Arrange
    frame = _frame({"641": [40.0, -4.0, 1.0, 0.0, 10.0, 0.0]})
    from opencood.utils.transformation_utils import x1_to_x2

    # Act
    transforms = noisy_transforms(frame, "ego", EGO_POSE, sigma=0.0, seed=0, frame=0)

    # Assert
    assert np.allclose(
        transforms["641"], x1_to_x2(frame["641"]["params"]["lidar_pose"], EGO_POSE)
    )


def test_applying_a_transform_never_mutates_the_shared_frame():
    # Arrange: one base_data_dict is reused for every sigma of a frame, so an
    # in-place write would leak one noise level into the next.
    frame = _frame({"641": [40.0, -4.0, 1.0, 0.0, 10.0, 0.0]})
    frame["641"]["params"]["transformation_matrix"] = np.eye(4)
    frame["641"]["params"]["vehicles"] = {"7": {}}

    # Act
    perturbed = _apply_transforms(frame, {"641": np.full((4, 4), 9.0)})

    # Assert
    assert np.allclose(frame["641"]["params"]["transformation_matrix"], np.eye(4))
    assert np.allclose(perturbed["641"]["params"]["transformation_matrix"], 9.0)
    # the ground-truth-bearing fields are carried through untouched
    assert perturbed["641"]["params"]["vehicles"] is frame["641"]["params"]["vehicles"]
    assert perturbed["641"]["params"]["lidar_pose"] == frame["641"]["params"]["lidar_pose"]


def test_the_noise_never_reaches_the_pose_the_ground_truth_is_built_from():
    # Arrange
    frame = _frame({"641": [40.0, -4.0, 1.0, 0.0, 10.0, 0.0]})
    original = list(frame["641"]["params"]["lidar_pose"])

    # Act
    transforms = noisy_transforms(frame, "ego", EGO_POSE, sigma=2.0, seed=0, frame=1)
    perturbed = _apply_transforms(frame, transforms)

    # Assert: lidar_pose is what COM_RANGE selection and generate_object_center
    # both read, and it must be the true one at every sigma.
    assert perturbed["641"]["params"]["lidar_pose"] == original


class _Fusion(torch.nn.Module):
    def forward(self, x):
        return x.sum(0, keepdim=True)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shrink_conv = torch.nn.Conv2d(4, 8, 1)
        self.fusion_net = torch.nn.ModuleList([_Fusion()])

    def forward(self, x):
        return self.fusion_net[0](self.shrink_conv(x))


def test_the_message_probe_reports_per_agent_bytes_not_per_frame_bytes():
    # Arrange: three agents in the batch dimension, an 8 x 5 x 6 message each.
    model = _Model()
    probe = MessageProbe(model, "point_pillar_cobevt")

    # Act
    model(torch.zeros(3, 4, 5, 6))
    summary = probe.summary()
    probe.close()

    # Assert
    assert summary["tensor_shape"] == [3, 8, 5, 6]
    assert summary["elements_per_agent"] == 8 * 5 * 6
    assert summary["bytes_per_frame_per_agent"] == 8 * 5 * 6 * BYTES_PER_FLOAT32


def test_the_message_probe_can_read_a_fusion_modules_input():
    # Arrange
    model = _Model()
    probe = MessageProbe(model, "point_pillar_coalign")  # site fusion_net.0, input

    # Act
    model(torch.zeros(2, 4, 5, 6))
    summary = probe.summary()
    probe.close()

    # Assert: the shrink_conv output is what fusion_net[0] receives.
    assert summary["tensor_shape"] == [2, 8, 5, 6]
    assert summary["elements_per_agent"] == 8 * 5 * 6


def test_alignformer_bytes_count_the_box_and_the_embedding(tmp_path):
    # Arrange: two cached agent-frames, 3 and 5 detections.
    for index, count in enumerate((3, 5)):
        np.savez(
            tmp_path / f"{index}.npz",
            boxes=np.zeros((count, 7), dtype=np.float32),
            scores=np.zeros(count, dtype=np.float32),
            gt_ids=np.array(["a"] * count, dtype=np.str_),
            roi=np.zeros((count, 2, 2, 2), dtype=np.float16),
        )

    # Act
    result = alignformer_message_bytes(tmp_path, embed_dim=128, max_objects=64)

    # Assert
    assert result["objects_per_agent_mean"] == 4.0
    assert result["bytes_per_object"] == (7 + 128) * BYTES_PER_FLOAT32
    assert result["bytes_per_frame_per_agent"] == 4.0 * 135 * BYTES_PER_FLOAT32


def test_truncation_is_applied_before_the_byte_count(tmp_path):
    # Arrange: a frame with more detections than the trunk's token budget.
    np.savez(
        tmp_path / "wide.npz",
        boxes=np.zeros((200, 7), dtype=np.float32),
        scores=np.zeros(200, dtype=np.float32),
        gt_ids=np.array(["a"] * 200, dtype=np.str_),
        roi=np.zeros((200, 2, 2, 2), dtype=np.float16),
    )

    # Act
    result = alignformer_message_bytes(tmp_path, embed_dim=128, max_objects=64)

    # Assert: the message is capped, so the byte count must be too.
    assert result["objects_per_agent_mean"] == 64
    assert result["truncated_at_max_objects"] == 1


def test_every_registered_baseline_has_a_message_site_or_is_named_as_lacking_one():
    # Arrange / Act
    missing = {
        name
        for name, spec in BASELINES.items()
        if spec.get("message") and name not in {"ermvp"}
    }

    # Assert: a baseline whose model core method has no probe site would come
    # out of the bandwidth run as a silent blank, so the mapping is pinned.
    expected_core_methods = {
        "v2xvit": "point_pillar_transformer",
        "coalign": "point_pillar_coalign",
        "cobevt": "point_pillar_cobevt",
        "attfuse": "point_pillar_intermediate",
        "where2comm": "point_pillar_where2comm",
        "fcooper": "point_pillar_fcooper",
        "v2vam": "point_pillar_intermediate_V2VAM",
    }
    for name in sorted(missing - {"ermvp"}):
        assert expected_core_methods[name] in MESSAGE_SITES


def test_resolve_walks_module_lists_by_index():
    # Arrange
    model = _Model()

    # Act / Assert
    assert _resolve(model, "fusion_net.0") is model.fusion_net[0]
    assert _resolve(model, "shrink_conv") is model.shrink_conv
