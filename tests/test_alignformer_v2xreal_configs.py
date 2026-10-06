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

import pytest
import yaml

from embedding_aware_belt_fusion.alignformer.camera_features import FEATURE_DIM

OPV2V = Path("configs/alignformer_r140.yaml")
LIDAR = Path("configs/alignformer_v2xreal.yaml")
CAMERA = Path("configs/alignformer_v2xreal_camera.yaml")
DINO = Path("configs/alignformer_v2xreal_dino.yaml")
DINO_DET = Path("configs/alignformer_v2xreal_dino_det.yaml")


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

    # The four variance numbers are DATA, refit on V2X-Real val by
    # scripts/fit_correspondence_variance.py (outputs/v2xreal/variance_fit_result.json);
    # everything else that could differ is a hyper-parameter and must not.
    assert differing == {
        "model.camera_dim",
        "model.correspondence_variance.sigma_translation_m",
        "model.correspondence_variance.translation_exponent",
        "model.correspondence_variance.sigma_yaw_deg",
        "model.correspondence_variance.yaw_exponent",
        "data.train_root",
        "data.val_root",
        "data.test_root",
        "data.cache_root",
    }


def test_the_variance_numbers_are_the_ones_the_fit_wrote():
    import json

    fitted = json.loads(Path("outputs/v2xreal/variance_fit_result.json").read_text())["correspondence_variance"]
    block = _load(LIDAR)["model"]["correspondence_variance"]

    for key in ("sigma_translation_m", "translation_exponent", "sigma_yaw_deg", "yaw_exponent"):
        assert block[key] == pytest.approx(fitted[key], abs=5e-5), key
    assert block["score_reference"] == fitted["score_reference"]


def test_the_splits_are_the_official_ones_on_the_nvme_mirror():
    data = _load(LIDAR)["data"]

    assert data["train_root"] == "/media/chenyi/basement2/dataset/v2x-real/train"
    assert data["val_root"] == "/media/chenyi/basement2/dataset/v2x-real/val"
    assert data["test_root"] == "/media/chenyi/basement2/dataset/v2x-real/test"
    assert data["cache_root"] != _load(OPV2V)["data"]["cache_root"]


def test_the_variance_model_is_off_until_refit_on_v2xreal():
    # The OPV2V numbers are placeholders; mode none means they are unused.
    assert _load(LIDAR)["model"]["correspondence_variance"]["mode"] == "none"


def test_the_dino_config_differs_from_the_lidar_config_only_in_the_camera_source_and_its_cache():
    lidar, dino = _flat(_load(LIDAR)), _flat(_load(DINO))

    differing = {k for k in lidar.keys() | dino.keys() if lidar.get(k) != dino.get(k)}

    assert differing == {
        "model.camera_dim",
        "model.camera_source.backbone",
        "model.camera_source.head_checkpoint",
        "model.camera_source.dino",
        "model.camera_source.track_window",
        "model.camera_source.track_gate_m",
        "model.camera_source.track_gate_per_frame_m",
        "data.cache_root",
    }
    assert dino["model.camera_dim"] == 128
    assert dino["model.camera_source.backbone"] == "dino_head"
    assert dino["model.camera_source.track_window"] == 4


def test_the_detection_trained_dino_config_differs_only_in_the_head_and_its_cache():
    dino, det = _flat(_load(DINO)), _flat(_load(DINO_DET))

    differing = {k for k in dino.keys() | det.keys() if dino.get(k) != det.get(k)}

    assert differing == {"model.camera_source.head_checkpoint", "data.cache_root"}

