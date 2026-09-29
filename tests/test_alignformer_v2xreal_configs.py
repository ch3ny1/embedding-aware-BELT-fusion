"""The two V2X-Real AlignFormer configs, pinned against each other and OPV2V's.

The three trunks are ONE configuration with one switch each: boxes-only is
``--message-content boxes_only`` on the LiDAR config, and the camera trunk is
the LiDAR config with ``model.camera_dim`` set. Anything else drifting
between the two files would turn the camera question into a
hyper-parameter question.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from embedding_aware_belt_fusion.alignformer.camera_features import FEATURE_DIM

OPV2V = Path("configs/alignformer_r140.yaml")
LIDAR = Path("configs/alignformer_v2xreal.yaml")
CAMERA = Path("configs/alignformer_v2xreal_camera.yaml")


def _flat(tree: Any, prefix: str = "") -> Dict[str, Any]:
    if not isinstance(tree, dict):
        return {prefix: tree}
    out: Dict[str, Any] = {}
    for k, v in tree.items():
        out.update(_flat(v, f"{prefix}.{k}" if prefix else str(k)))
    return out


def _load(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text())


def test_the_camera_config_differs_from_the_lidar_config_only_in_camera_dim():
    lidar, camera = _flat(_load(LIDAR)), _flat(_load(CAMERA))

    differing = {k for k in lidar.keys() | camera.keys() if lidar.get(k) != camera.get(k)}

    assert differing == {"model.camera_dim"}
    assert lidar["model.camera_dim"] == 0
    assert camera["model.camera_dim"] == FEATURE_DIM


def test_the_v2xreal_config_differs_from_opv2v_only_in_data_paths_and_the_switch():
    opv2v, lidar = _flat(_load(OPV2V)), _flat(_load(LIDAR))

    differing = {k for k in opv2v.keys() | lidar.keys() if opv2v.get(k) != lidar.get(k)}

    assert differing == {
        "model.camera_dim",
        "data.train_root",
        "data.val_root",
        "data.test_root",
        "data.cache_root",
    }


def test_the_splits_are_the_official_ones_on_the_nvme_mirror():
    data = _load(LIDAR)["data"]

    assert data["train_root"] == "/media/chenyi/basement2/dataset/v2x-real/train"
    assert data["val_root"] == "/media/chenyi/basement2/dataset/v2x-real/val"
    assert data["test_root"] == "/media/chenyi/basement2/dataset/v2x-real/test"
    assert data["cache_root"] != _load(OPV2V)["data"]["cache_root"]


def test_the_variance_model_is_off_until_refit_on_v2xreal():
    # The OPV2V numbers are placeholders; mode none means they are unused.
    assert _load(LIDAR)["model"]["correspondence_variance"]["mode"] == "none"
