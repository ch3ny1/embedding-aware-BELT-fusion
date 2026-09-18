"""Late-fusion detector checkpoint used by AlignFormer.

No pretrained late-fusion checkpoint could be obtained (OpenCOOD's release is
unreachable on both mirrors it ships -- see
.superpowers/sdd/2026-09-17-alignformer-p0-p2/task-2-report.md), so this
checkpoint is trained from scratch by scripts/train_late_fusion.py. Until
that finishes, the checkpoint file this config points at does not exist yet;
this test skips (rather than fails) in that case, and becomes a real
assertion once training lands the weights.
"""

from pathlib import Path

import pytest
import torch
import yaml


def _load_config() -> dict:
    return yaml.safe_load(Path("configs/alignformer_detector.yaml").read_text())


def test_detector_checkpoint_exists_and_has_detection_heads():
    config = _load_config()
    checkpoint = Path(config["detector"]["checkpoint"])
    if not checkpoint.exists():
        pytest.skip(
            f"detector checkpoint not trained yet: {checkpoint} "
            "(see scripts/train_late_fusion.py and outputs/alignformer/detector_train.log)"
        )

    state = torch.load(checkpoint, map_location="cpu")
    state = state.get("model_state_dict", state)
    keys = set(state)

    # Late fusion needs per-agent detection heads, unlike the CoLoca backbone
    # which deliberately discards them.
    assert any(k.startswith("cls_head.") for k in keys)
    assert any(k.startswith("reg_head.") for k in keys)
    assert any(k.startswith("pillar_vfe.") for k in keys)
