"""Environment preconditions for AlignFormer.

These are not unit tests of our code; they assert that the compiled OpenCOOD
extensions AlignFormer depends on are actually importable in this env.
"""

import torch


def test_iou3d_nms_cuda_extension_is_built():
    from opencood.pcdet_utils.iou3d_nms import iou3d_nms_utils

    assert hasattr(iou3d_nms_utils, "nms_gpu")


def test_nms_rotated_is_callable():
    from opencood.utils import box_utils

    corners = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0], [2.0, 0.0, 1.0], [2.0, 1.0, 1.0], [0.0, 1.0, 1.0]],
        ]
    ).cuda()
    scores = torch.tensor([0.9]).cuda()

    keep = box_utils.nms_rotated(corners, scores, 0.15)

    assert len(keep) == 1


def test_late_fusion_dataset_is_importable():
    from opencood.data_utils.datasets.late_fusion_dataset import LateFusionDataset

    assert LateFusionDataset is not None


def test_voxel_postprocessor_is_importable():
    from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor

    assert VoxelPostprocessor is not None
