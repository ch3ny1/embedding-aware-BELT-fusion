"""Environment preconditions for AlignFormer.

These are not unit tests of our code; they assert that the compiled OpenCOOD
extensions AlignFormer depends on are actually importable in this env.
"""

import torch


def test_iou3d_nms_cuda_extension_is_built():
    from opencood.pcdet_utils.iou3d_nms import iou3d_nms_utils

    assert hasattr(iou3d_nms_utils, "nms_gpu")


def _box_corners(x, y, length=4.0, width=2.0):
    """Axis-aligned 8-corner box centred at (x, y)."""
    half_l, half_w = length / 2, width / 2
    base = [
        (x - half_l, y - half_w), (x + half_l, y - half_w),
        (x + half_l, y + half_w), (x - half_l, y + half_w),
    ]
    return [[cx, cy, z] for z in (0.0, 1.5) for cx, cy in base]


def test_nms_rotated_actually_suppresses_overlapping_boxes():
    # Deliberately NOT a single box: with one input, nms_rotated returns
    # keep=[0] through a CPU/Shapely path without ever invoking the CUDA
    # extension, so a single-box test passes even when the build is broken.
    # Two near-identical boxes plus one far away force real IoU work.
    from opencood.utils import box_utils

    corners = torch.tensor(
        [_box_corners(0.0, 0.0), _box_corners(0.3, 0.1), _box_corners(60.0, 20.0)]
    ).cuda()
    scores = torch.tensor([0.9, 0.8, 0.7]).cuda()

    keep = sorted(int(i) for i in box_utils.nms_rotated(corners, scores, 0.15))

    assert keep == [0, 2]


def test_late_fusion_dataset_is_importable():
    from opencood.data_utils.datasets.late_fusion_dataset import LateFusionDataset
    from torch.utils.data import Dataset

    assert issubclass(LateFusionDataset, Dataset)
    assert hasattr(LateFusionDataset, "get_item_single_car")


def test_voxel_postprocessor_is_importable():
    from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
    from opencood.data_utils.post_processor.base_postprocessor import BasePostprocessor

    assert issubclass(VoxelPostprocessor, BasePostprocessor)
    assert hasattr(VoxelPostprocessor, "generate_anchor_box")
