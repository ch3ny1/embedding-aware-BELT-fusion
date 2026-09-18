# AlignFormer P0-P2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an embedding-aware late-fusion pipeline on OPV2V that estimates each CAV's SE(2) localization error from transmitted boxes plus light per-object embeddings, through the P2 go/no-go gate.

**Architecture:** A frozen PointPillars detector produces boxes and a BEV feature map per agent. A rotated ROI-align in each box's canonical frame yields a pose-invariant per-object embedding. At the ego, a shared transformer over the two object sets feeds two interchangeable heads: direct regression (A) and Sinkhorn soft-correspondence followed by closed-form weighted SE(2) Kabsch (B).

**Tech Stack:** Python 3.8, PyTorch 2.3.1+cu121, OpenCOOD (submodule, via PYTHONPATH), NumPy, pytest. Conda env `opencood`.

**Spec:** `docs/superpowers/specs/2026-09-17-alignformer-design.md`

## Scope

This plan covers **P0-P2 only**. P2 ends at an explicit go/no-go gate whose outcome
determines the shape of P3-P5, so those are planned separately afterwards rather
than speculatively now.

Two modules from the spec's §9 layout are therefore **not** built here:
`message.py` (message serialization and byte accounting) and `head_direct.py`,
whose contents are folded into `model.py` as `AlignFormerA`. Byte accounting is a
P5 deliverable; nothing in P0-P2 depends on it.

## Global Constraints

- Env: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate opencood`
- Always: `export PYTHONPATH=src:external/OpenCOOD`
- **Not** the `belt-fusion` env - it has no spconv and no OpenCOOD.
- OPV2V splits: `/media/chenyi/Elements1/Dataset/OPV2V/{train,test}`. `validate/` is a
  broken symlink; validation is 15% of **train scenarios**, split by scenario, never
  by frame (consecutive OPV2V frames are near-duplicates).
- Point clouds are read through the NVMe mirror at
  `/media/chenyi/basement2/cache/opv2v_coloca/{train,test}`, never the raw pcds.
  Reuse `coloca/pcd_cache.py:load_points`.
- New caches go under `/media/chenyi/basement2/cache/alignformer/`. Never under
  `/media/chenyi/basement` or `basement1` - stale empty mountpoints on a full disk.
- **Box order is OpenCOOD `hwl`**: `box = [x, y, z, h, w, l, yaw]`, so index 4 is
  width (across-track) and index 5 is length (along-track). Yaw is radians.
- **OPV2V pose is `[x, y, z, roll, yaw, pitch]` in degrees, index 4 is yaw.**
- Noise convention: `sigma_yaw (deg) = sigma_xy (m)` numerically. Applied to the CAV
  pose only, on x, y and yaw. z, roll, pitch untouched.
- Coding rules: files 200-400 lines (800 max), functions under 50 lines, no mutation
  of inputs, explicit error handling, named constants not magic numbers.
- Every task ends with a commit. Conventional commit format `<type>: <description>`.
  **No attribution trailers** (disabled globally in this project).
- GPU: RTX 4090, 24 GB.

---

### Task 1: Build the `iou3d_nms` CUDA extension

OpenCOOD's rotated NMS is a CUDA extension that is **not built** in this env.
CoLoca-QuA never needed it (it discards the detection heads); AlignFormer needs it
for every detection decode and every fusion step. Nothing else in this plan can run
until this works.

**Files:**
- Modify: none (builds in `external/OpenCOOD/`)
- Test: `tests/test_alignformer_env.py`

**Interfaces:**
- Consumes: nothing
- Produces: a working `opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils`, and
  therefore working `opencood.utils.box_utils.nms_rotated`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_alignformer_env.py
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate opencood
export PYTHONPATH=src:external/OpenCOOD
python -m pytest tests/test_alignformer_env.py -v
```

Expected: FAIL with `ImportError: cannot import name 'iou3d_nms_cuda'`.

- [ ] **Step 3: Build the extension**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate opencood
cd external/OpenCOOD
python setup.py build_ext --inplace 2>&1 | tail -30
```

If the build targets the wrong CUDA arch, the RTX 4090 is compute capability 8.9:

```bash
TORCH_CUDA_ARCH_LIST="8.9" python setup.py build_ext --inplace 2>&1 | tail -30
```

`nvcc` 12.2 is on PATH and torch is 2.3.1+cu121; a minor-version skew between them
is normal and does not need fixing.

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /media/chenyi/Elements1/Repos/embedding-aware-BELT-fusion
python -m pytest tests/test_alignformer_env.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

The built `.so` lives inside the submodule and must not be committed to this repo;
only the test is ours.

```bash
git add tests/test_alignformer_env.py
git commit -m "test: assert iou3d_nms CUDA extension is built

AlignFormer needs rotated NMS for every detection decode and fusion step.
The extension ships unbuilt in this env; this test fails loudly rather than
letting a later task fail with a confusing ImportError."
```

---

### Task 2: Obtain a PointPillars late-fusion detector

**There is no late-fusion checkpoint on this machine.** Every checkpoint under
`/media/chenyi/Elements1/models/opv2v/` is an *intermediate*-fusion model (f_cooper,
attentive_fusion, v2xvit, cobevt, coalign, where2comm, v2vam, mamba, ermvp). The
checkpoint that produced the README's 0.856 AP@0.7 lived on the original `dhia`
host and is gone, like `data/`.

F-Cooper is **not** a valid substitute. Its heads were trained on features maxpooled
across 2-5 agents, so running it per-agent is a distribution shift of unknown size.

**Files:**
- Create: `configs/alignformer_detector.yaml` (a copy of OpenCOOD's late-fusion
  hypes with this machine's OPV2V paths)
- Test: `tests/test_alignformer_detector.py`

**Interfaces:**
- Consumes: Task 1's working NMS.
- Produces: a checkpoint path recorded in `configs/alignformer_detector.yaml` under
  `detector.checkpoint`, loadable by Task 3.

- [ ] **Step 1: Try the released checkpoint first (cheap path)**

OpenCOOD publishes trained OPV2V checkpoints. Check its README for the
`point_pillar_late_fusion` download link:

```bash
grep -ri 'drive.google\|checkpoint\|model zoo' external/OpenCOOD/README.md | head -20
```

If a `pointpillar_late_fusion` checkpoint is obtainable, download it to
`/media/chenyi/Elements1/models/opv2v/pointpillar_late_fusion/` and skip to Step 3.

- [ ] **Step 2: Otherwise train it (fallback, about one day on the 4090)**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate opencood
export PYTHONPATH=src:external/OpenCOOD
python -u external/OpenCOOD/opencood/tools/train.py \
  --hypes_yaml external/OpenCOOD/opencood/hypes_yaml/point_pillar_late_fusion.yaml \
  2>&1 | tee outputs/alignformer/detector_train.log
```

Before launching, edit the yaml's `root_dir` / `validate_dir` to
`/media/chenyi/Elements1/Dataset/OPV2V/train` and the 15% scenario holdout. Run this
in the background and monitor; do not block the rest of the plan on it - Tasks 4-9
are pure-tensor modules with synthetic tests and need no checkpoint.

- [ ] **Step 3: Write the test**

```python
# tests/test_alignformer_detector.py
from pathlib import Path

import torch
import yaml


def test_detector_checkpoint_exists_and_has_detection_heads():
    config = yaml.safe_load(Path("configs/alignformer_detector.yaml").read_text())
    checkpoint = Path(config["detector"]["checkpoint"])
    assert checkpoint.exists(), f"detector checkpoint missing: {checkpoint}"

    state = torch.load(checkpoint, map_location="cpu")
    state = state.get("model_state_dict", state)
    keys = set(state)

    # Late fusion needs per-agent detection heads, unlike the CoLoca backbone
    # which deliberately discards them.
    assert any(k.startswith("cls_head.") for k in keys)
    assert any(k.startswith("reg_head.") for k in keys)
    assert any(k.startswith("pillar_vfe.") for k in keys)
```

- [ ] **Step 4: Run it**

```bash
python -m pytest tests/test_alignformer_detector.py -v
```

Expected: PASS once a checkpoint is in place.

- [ ] **Step 5: Commit**

```bash
git add configs/alignformer_detector.yaml tests/test_alignformer_detector.py
git commit -m "feat: late-fusion detector config for AlignFormer

No late-fusion checkpoint survived on this machine; every available OPV2V
checkpoint is an intermediate-fusion model whose heads were trained on
features maxpooled across agents. Records the checkpoint this project uses
and asserts it carries per-agent detection heads."
```

---

### Task 3: `alignformer/boxes.py` - per-agent detection behind the backbone interface

The single place that knows about the detector. A later SWFormer swap changes only
this file.

**Files:**
- Create: `src/embedding_aware_belt_fusion/alignformer/__init__.py`
- Create: `src/embedding_aware_belt_fusion/alignformer/boxes.py`
- Test: `tests/test_alignformer_boxes.py`

**Interfaces:**
- Consumes: `features/opencood_proposals.py:pointpillar_forward_with_features`,
  `decode_local_proposals`, `assign_proposals_to_ground_truth`.
- Produces:
  - `AgentDetections` frozen dataclass with fields `boxes: Tensor (M,7)`,
    `scores: Tensor (M,)`, `corners: Tensor (M,8,3)`, `gt_ids: list[str | None]`,
    `features: Tensor (C,H,W)` (the full BEV map, not per-proposal features).
  - `detect_agent(detector, cav_content, postprocessor, *, minimum_iou=0.3) -> AgentDetections`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_alignformer_boxes.py
import pytest
import torch

from embedding_aware_belt_fusion.alignformer.boxes import AgentDetections


def test_agent_detections_is_immutable_and_reports_count():
    detections = AgentDetections(
        boxes=torch.zeros(3, 7),
        scores=torch.zeros(3),
        corners=torch.zeros(3, 8, 3),
        gt_ids=["a", None, "c"],
        features=torch.zeros(256, 96, 256),
    )

    assert len(detections) == 3
    with pytest.raises(AttributeError):
        detections.boxes = torch.zeros(1, 7)


def test_agent_detections_rejects_inconsistent_lengths():
    with pytest.raises(ValueError, match="length"):
        AgentDetections(
            boxes=torch.zeros(3, 7),
            scores=torch.zeros(2),
            corners=torch.zeros(3, 8, 3),
            gt_ids=["a", None, "c"],
            features=torch.zeros(256, 96, 256),
        )
```

- [ ] **Step 2: Run to verify it fails**

```bash
python -m pytest tests/test_alignformer_boxes.py -v
```

Expected: FAIL with `ModuleNotFoundError: ...alignformer`.

- [ ] **Step 3: Implement**

```python
# src/embedding_aware_belt_fusion/alignformer/__init__.py
"""AlignFormer: embedding-aware late fusion robust to localization error."""
```

```python
# src/embedding_aware_belt_fusion/alignformer/boxes.py
"""Per-agent detection, and the one place that knows about the detector.

A backbone swap (SWFormer, or a stronger encoder) changes this file only. Every
downstream module consumes :class:`AgentDetections` and never touches OpenCOOD.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from torch import Tensor

from embedding_aware_belt_fusion.features.opencood_proposals import (
    assign_proposals_to_ground_truth,
    decode_local_proposals,
    pointpillar_forward_with_features,
)

# IoU below which a proposal is treated as having no ground-truth counterpart.
DEFAULT_MINIMUM_IOU = 0.3


@dataclass(frozen=True)
class AgentDetections:
    """One agent's detections plus the BEV map they were decoded from.

    ``boxes`` are in the agent's own LiDAR frame, OpenCOOD ``hwl`` order
    ``[x, y, z, h, w, l, yaw]`` with yaw in radians. ``gt_ids`` holds the OPV2V
    physical object id each detection was matched to, or ``None`` when it matched
    nothing - these are supervision only and are never transmitted.
    """

    boxes: Tensor
    scores: Tensor
    corners: Tensor
    gt_ids: Sequence[Optional[str]]
    features: Tensor

    def __post_init__(self) -> None:
        count = self.boxes.shape[0]
        lengths = {
            "scores": self.scores.shape[0],
            "corners": self.corners.shape[0],
            "gt_ids": len(self.gt_ids),
        }
        mismatched = {name: n for name, n in lengths.items() if n != count}
        if mismatched:
            raise ValueError(
                f"inconsistent length against {count} boxes: {mismatched}"
            )
        if self.features.dim() != 3:
            raise ValueError(
                f"features must be (C, H, W), got shape {tuple(self.features.shape)}"
            )

    def __len__(self) -> int:
        return self.boxes.shape[0]


def detect_agent(
    detector,
    cav_content: Mapping,
    postprocessor,
    *,
    minimum_iou: float = DEFAULT_MINIMUM_IOU,
) -> AgentDetections:
    """Run the detector on one agent and match its proposals to local ground truth."""
    output = pointpillar_forward_with_features(detector, cav_content)
    decoded = decode_local_proposals(output, cav_content, postprocessor)

    assignment = assign_proposals_to_ground_truth(
        decoded["corners"],
        cav_content["object_bbx_center"][cav_content["object_bbx_mask"].bool()],
        cav_content["object_ids"],
        order=postprocessor.params["order"],
        minimum_iou=minimum_iou,
    )

    return AgentDetections(
        boxes=decoded["boxes"],
        scores=decoded["scores"],
        corners=decoded["corners"],
        gt_ids=assignment["gt_ids"],
        features=output["spatial_features_2d"][0],
    )
```

- [ ] **Step 4: Run to verify it passes**

```bash
python -m pytest tests/test_alignformer_boxes.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/embedding_aware_belt_fusion/alignformer tests/test_alignformer_boxes.py
git commit -m "feat: per-agent detection interface for AlignFormer

AgentDetections is the single boundary between the detector and everything
downstream, so a backbone swap touches only boxes.py."
```

---

### Task 4: `alignformer/procrustes.py` - differentiable weighted SE(2) Kabsch

The mathematical core of Head B, and the module that makes yaw structural rather
than learned. Pure tensor code with no data dependencies, so it can be built and
tested while Task 2's detector trains.

**Files:**
- Create: `src/embedding_aware_belt_fusion/alignformer/procrustes.py`
- Test: `tests/test_alignformer_procrustes.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `weighted_se2_kabsch(p, q, w, *, eps=1e-6) -> tuple[Tensor, Tensor]` where
    `p, q` are `(B, N, 2)`, `w` is `(B, N)`, returning `psi (B,)` in radians and
    `t (B, 2)`, minimizing `sum_n w_n |R(psi) q_n + t - p_n|^2`.
  - `augment_with_heading(centres, yaws, lam) -> Tensor (B, 2N, 2)`
  - `MIN_MATCH_MASS = 1.0`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_alignformer_procrustes.py
import math

import pytest
import torch

from embedding_aware_belt_fusion.alignformer.procrustes import (
    MIN_MATCH_MASS,
    augment_with_heading,
    weighted_se2_kabsch,
)


def _apply(psi, t, q):
    cos, sin = math.cos(psi), math.sin(psi)
    rotation = torch.tensor([[cos, -sin], [sin, cos]])
    return q @ rotation.T + torch.tensor(t)


def test_recovers_a_known_transform_exactly():
    # Arrange
    q = torch.tensor([[[0.0, 0.0], [5.0, 1.0], [2.0, -3.0], [-4.0, 2.0]]])
    true_psi, true_t = 0.3, (1.5, -2.0)
    p = _apply(true_psi, true_t, q[0]).unsqueeze(0)
    w = torch.ones(1, 4)

    # Act
    psi, t = weighted_se2_kabsch(p, q, w)

    # Assert
    assert psi.item() == pytest.approx(true_psi, abs=1e-5)
    assert t[0, 0].item() == pytest.approx(true_t[0], abs=1e-5)
    assert t[0, 1].item() == pytest.approx(true_t[1], abs=1e-5)


def test_zero_weight_points_are_ignored():
    q = torch.tensor([[[0.0, 0.0], [4.0, 0.0], [999.0, 999.0]]])
    true_psi, true_t = -0.2, (0.5, 0.25)
    p = _apply(true_psi, true_t, q[0]).unsqueeze(0)
    p[0, 2] = torch.tensor([-50.0, 70.0])  # an outlier, down-weighted to zero
    w = torch.tensor([[1.0, 1.0, 0.0]])

    psi, t = weighted_se2_kabsch(p, q, w)

    assert psi.item() == pytest.approx(true_psi, abs=1e-5)


def test_a_single_object_with_heading_determines_full_se2():
    # A lone centre is rank-deficient for yaw. Adding the heading virtual point
    # makes one matched object sufficient - the spec's low-overlap claim.
    centres_q = torch.tensor([[[3.0, -1.0]]])
    yaws_q = torch.tensor([[0.4]])
    true_psi, true_t = 0.25, (-1.0, 2.0)

    centres_p = _apply(true_psi, true_t, centres_q[0]).unsqueeze(0)
    yaws_p = yaws_q + true_psi

    q = augment_with_heading(centres_q, yaws_q, lam=2.0)
    p = augment_with_heading(centres_p, yaws_p, lam=2.0)
    w = torch.ones(1, q.shape[1])

    psi, t = weighted_se2_kabsch(p, q, w)

    assert psi.item() == pytest.approx(true_psi, abs=1e-5)
    assert t[0, 0].item() == pytest.approx(true_t[0], abs=1e-5)


def test_degenerate_input_returns_identity_without_nan():
    p = torch.zeros(1, 3, 2)
    q = torch.zeros(1, 3, 2)
    w = torch.zeros(1, 3)

    psi, t = weighted_se2_kabsch(p, q, w)

    assert torch.isfinite(psi).all() and torch.isfinite(t).all()
    assert psi.item() == pytest.approx(0.0)
    assert torch.allclose(t, torch.zeros_like(t))


def test_gradients_flow_to_weights_and_are_finite():
    q = torch.tensor([[[0.0, 0.0], [5.0, 1.0], [2.0, -3.0]]])
    p = _apply(0.3, (1.0, 1.0), q[0]).unsqueeze(0)
    w = torch.full((1, 3), 0.5, requires_grad=True)

    psi, t = weighted_se2_kabsch(p, q, w)
    (psi.sum() + t.sum()).backward()

    assert w.grad is not None
    assert torch.isfinite(w.grad).all()


def test_degenerate_gradients_are_finite_not_nan():
    # atan2(0, 0) has undefined gradient; the guard must prevent NaN reaching
    # the optimizer, which would silently poison a whole training run.
    p = torch.zeros(1, 2, 2)
    q = torch.zeros(1, 2, 2)
    w = torch.zeros(1, 2, requires_grad=True)

    psi, t = weighted_se2_kabsch(p, q, w)
    (psi.sum() + t.sum()).backward()

    assert torch.isfinite(w.grad).all()


def test_min_match_mass_is_one_effective_object():
    assert MIN_MATCH_MASS == 1.0
```

- [ ] **Step 2: Run to verify they fail**

```bash
python -m pytest tests/test_alignformer_procrustes.py -v
```

Expected: FAIL, module not found.

- [ ] **Step 3: Implement**

```python
# src/embedding_aware_belt_fusion/alignformer/procrustes.py
"""Differentiable weighted SE(2) Procrustes, and the heading augmentation.

Head B never regresses the pose. Given soft correspondences it recovers the
SE(2) transform in closed form, which is why yaw is solved analytically here
rather than learned - the CoLoca-QuA reproduction showed a regressed yaw never
leaving the conditional mean on this data.

Augmenting each object with a second virtual point at ``centre + lam * heading``
folds box headings into the same least-squares problem as centres. A useful
consequence is that a *single* matched object then determines the full SE(2),
removing the collinearity degeneracy of centre-only Procrustes.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

# Total soft-match mass below which the correction is suppressed: less than one
# effective matched object carries no usable registration evidence.
MIN_MATCH_MASS = 1.0


def augment_with_heading(centres: Tensor, yaws: Tensor, lam: float) -> Tensor:
    """Append a heading virtual point per object.

    Parameters
    ----------
    centres: ``(B, N, 2)`` box centres.
    yaws: ``(B, N)`` box headings in radians.
    lam: offset in metres; about half a vehicle length makes headings and
        centres contribute comparably to the fit.

    Returns
    -------
    Tensor
        ``(B, 2N, 2)``, centres first then the corresponding heading tips, so a
        weight vector is extended by ``torch.cat([w, w], dim=1)``.
    """
    if centres.shape[:2] != yaws.shape:
        raise ValueError(
            f"centres {tuple(centres.shape)} and yaws {tuple(yaws.shape)} disagree"
        )
    direction = torch.stack([torch.cos(yaws), torch.sin(yaws)], dim=-1)
    return torch.cat([centres, centres + lam * direction], dim=1)


def weighted_se2_kabsch(
    p: Tensor, q: Tensor, w: Tensor, *, eps: float = 1e-6
) -> Tuple[Tensor, Tensor]:
    """Closed-form weighted SE(2) fit taking ``q`` onto ``p``.

    Minimizes ``sum_n w_n |R(psi) q_n + t - p_n|^2``.

    Parameters
    ----------
    p: ``(B, N, 2)`` target points (the ego's own objects).
    q: ``(B, N, 2)`` source points (CAV objects projected with the noisy pose).
    w: ``(B, N)`` non-negative correspondence weights.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``psi`` of shape ``(B,)`` in radians, and ``t`` of shape ``(B, 2)``.
        Degenerate inputs yield the identity transform with finite gradients.
    """
    if p.shape != q.shape:
        raise ValueError(f"p {tuple(p.shape)} and q {tuple(q.shape)} must match")
    if w.shape != p.shape[:2]:
        raise ValueError(f"w {tuple(w.shape)} must be {tuple(p.shape[:2])}")

    w = w.clamp_min(0.0)
    mass = w.sum(dim=1, keepdim=True)
    safe_mass = mass.clamp_min(eps)

    weights = w.unsqueeze(-1)
    p_bar = (weights * p).sum(dim=1) / safe_mass
    q_bar = (weights * q).sum(dim=1) / safe_mass

    dp = p - p_bar.unsqueeze(1)
    dq = q - q_bar.unsqueeze(1)
    cross = (w * (dq[..., 0] * dp[..., 1] - dq[..., 1] * dp[..., 0])).sum(dim=1)
    dot = (w * (dq[..., 0] * dp[..., 0] + dq[..., 1] * dp[..., 1])).sum(dim=1)

    # atan2(0, 0) is finite but its gradient is not. Route degenerate entries
    # through a constant branch so no NaN can reach the optimizer.
    magnitude = torch.sqrt(cross * cross + dot * dot)
    resolvable = magnitude > eps
    cross_safe = torch.where(resolvable, cross, torch.zeros_like(cross))
    dot_safe = torch.where(resolvable, dot, torch.ones_like(dot))
    psi = torch.atan2(cross_safe, dot_safe)

    cos, sin = torch.cos(psi), torch.sin(psi)
    rotated_q_bar = torch.stack(
        [cos * q_bar[:, 0] - sin * q_bar[:, 1], sin * q_bar[:, 0] + cos * q_bar[:, 1]],
        dim=-1,
    )
    t = p_bar - rotated_q_bar

    # Suppress the correction entirely when there is too little match evidence.
    usable = (mass.squeeze(1) >= MIN_MATCH_MASS) & resolvable
    psi = torch.where(usable, psi, torch.zeros_like(psi))
    t = torch.where(usable.unsqueeze(-1), t, torch.zeros_like(t))
    return psi, t
```

- [ ] **Step 4: Run to verify they pass**

```bash
python -m pytest tests/test_alignformer_procrustes.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/embedding_aware_belt_fusion/alignformer/procrustes.py \
        tests/test_alignformer_procrustes.py
git commit -m "feat: differentiable weighted SE(2) Procrustes with heading augmentation

Recovers the pose in closed form from soft correspondences, so yaw is solved
analytically rather than regressed - the structural answer to the yaw failure
documented in the CoLoca-QuA reproduction. Heading virtual points make a single
matched object sufficient to determine full SE(2)."
```

---

### Task 5: `alignformer/embedding.py` - rotated ROI-align and the embedding head

The transmit-side feature. Sampling in each box's **canonical frame** is what makes
the embedding pose-invariant, which is the spec's separation of concerns: the
embedding carries appearance, the box carries geometry.

**Files:**
- Create: `src/embedding_aware_belt_fusion/alignformer/embedding.py`
- Test: `tests/test_alignformer_embedding.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `rotated_roi_align(features, boxes, lidar_range, output_size) -> Tensor (M, C, k, k)`
    where `features` is `(C, H, W)` with H along y and W along x, and `boxes` is
    `(M, 7)` in `hwl` order in the same frame as `lidar_range`.
  - `ObjectEmbedding(in_channels, output_size, dim)` with
    `forward(roi: Tensor (M, C, k, k)) -> Tensor (M, dim)`, L2-normalized.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_alignformer_embedding.py
import math

import torch

from embedding_aware_belt_fusion.alignformer.embedding import (
    ObjectEmbedding,
    rotated_roi_align,
)

LIDAR_RANGE = [-102.4, -38.4, -3.0, 102.4, 38.4, 1.0]


def _stamp(features, x, y, value, lidar_range=LIDAR_RANGE):
    """Write `value` into the BEV cell containing world point (x, y)."""
    _, height, width = features.shape
    x_min, y_min, _, x_max, y_max, _ = lidar_range
    column = int((x - x_min) / (x_max - x_min) * width)
    row = int((y - y_min) / (y_max - y_min) * height)
    features[:, row, column] = value


def test_output_shape_is_per_box_grid():
    features = torch.randn(8, 96, 256)
    boxes = torch.tensor([[10.0, 5.0, 0.0, 1.6, 2.0, 4.5, 0.3]])

    roi = rotated_roi_align(features, boxes, LIDAR_RANGE, output_size=4)

    assert roi.shape == (1, 8, 4, 4)


def test_empty_box_set_returns_empty_without_error():
    features = torch.randn(8, 96, 256)

    roi = rotated_roi_align(features, torch.zeros(0, 7), LIDAR_RANGE, output_size=4)

    assert roi.shape == (0, 8, 4, 4)


def test_sampling_is_canonical_so_placement_does_not_change_the_patch():
    # The same synthetic object placed at two different positions AND
    # orientations must yield the same canonical ROI patch. This is the
    # pose-invariance property the whole matching design rests on.
    length = 8.0
    box_a = torch.tensor([[0.0, 0.0, 0.0, 1.6, 4.0, length, 0.0]])
    box_b = torch.tensor([[40.0, 12.0, 0.0, 1.6, 4.0, length, math.pi / 2]])

    # An asymmetric marker 3 m "ahead" of each box centre along its own heading,
    # so a wrong rotation convention shows up as a flipped patch.
    features_a = torch.zeros(1, 96, 256)
    _stamp(features_a, x=3.0, y=0.0, value=1.0)
    features_b = torch.zeros(1, 96, 256)
    _stamp(features_b, x=40.0, y=15.0, value=1.0)

    roi_a = rotated_roi_align(features_a, box_a, LIDAR_RANGE, output_size=4)
    roi_b = rotated_roi_align(features_b, box_b, LIDAR_RANGE, output_size=4)

    assert torch.argmax(roi_a.flatten()) == torch.argmax(roi_b.flatten())


def test_roi_align_is_differentiable_wrt_features():
    features = torch.randn(4, 96, 256, requires_grad=True)
    boxes = torch.tensor([[10.0, 5.0, 0.0, 1.6, 2.0, 4.5, 0.3]])

    rotated_roi_align(features, boxes, LIDAR_RANGE, output_size=4).sum().backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_embedding_is_unit_norm():
    head = ObjectEmbedding(in_channels=8, output_size=4, dim=128)
    roi = torch.randn(5, 8, 4, 4)

    embeddings = head(roi)

    assert embeddings.shape == (5, 128)
    assert torch.allclose(embeddings.norm(dim=1), torch.ones(5), atol=1e-5)


def test_embedding_handles_empty_object_set():
    head = ObjectEmbedding(in_channels=8, output_size=4, dim=128)

    embeddings = head(torch.zeros(0, 8, 4, 4))

    assert embeddings.shape == (0, 128)
```

- [ ] **Step 2: Run to verify they fail**

```bash
python -m pytest tests/test_alignformer_embedding.py -v
```

Expected: FAIL, module not found.

- [ ] **Step 3: Implement**

```python
# src/embedding_aware_belt_fusion/alignformer/embedding.py
"""The transmit-side per-object embedding.

Features are sampled over each box's footprint **in that box's own canonical
frame**, so the resulting descriptor is invariant to where the sender believes
the object is. That is the separation the method depends on: the embedding
carries appearance for matching, the box carries geometry for solving.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Box layout is OpenCOOD 'hwl': [x, y, z, h, w, l, yaw].
_BOX_WIDTH = 4
_BOX_LENGTH = 5
_BOX_YAW = 6


def rotated_roi_align(
    features: Tensor,
    boxes: Tensor,
    lidar_range: Sequence[float],
    output_size: int,
) -> Tensor:
    """Sample a ``k x k`` grid over each rotated box footprint.

    Parameters
    ----------
    features: ``(C, H, W)`` BEV map, H indexing y and W indexing x.
    boxes: ``(M, 7)`` in ``hwl`` order, in the same frame as ``lidar_range``.
    lidar_range: ``[x_min, y_min, z_min, x_max, y_max, z_max]``.
    output_size: the ``k`` of the ``k x k`` sampling grid.

    Returns
    -------
    Tensor
        ``(M, C, k, k)``. Row index runs along the box's length (forward) axis
        and column index across its width, both in the box frame.
    """
    if features.dim() != 3:
        raise ValueError(f"features must be (C, H, W), got {tuple(features.shape)}")
    if boxes.dim() != 2 or boxes.shape[1] != 7:
        raise ValueError(f"boxes must be (M, 7), got {tuple(boxes.shape)}")

    channels = features.shape[0]
    count = boxes.shape[0]
    if count == 0:
        return features.new_zeros((0, channels, output_size, output_size))

    x_min, y_min, _, x_max, y_max, _ = lidar_range

    # Canonical grid in box-local units, spanning [-0.5, 0.5] of each extent.
    axis = torch.linspace(-0.5, 0.5, output_size, device=boxes.device, dtype=boxes.dtype)
    along, across = torch.meshgrid(axis, axis, indexing="ij")
    along = along.reshape(1, -1) * boxes[:, _BOX_LENGTH : _BOX_LENGTH + 1]
    across = across.reshape(1, -1) * boxes[:, _BOX_WIDTH : _BOX_WIDTH + 1]

    cosine = torch.cos(boxes[:, _BOX_YAW]).unsqueeze(1)
    sine = torch.sin(boxes[:, _BOX_YAW]).unsqueeze(1)
    world_x = boxes[:, 0:1] + along * cosine - across * sine
    world_y = boxes[:, 1:2] + along * sine + across * cosine

    # grid_sample expects normalized coordinates in [-1, 1], last dim (x, y).
    grid_x = 2.0 * (world_x - x_min) / (x_max - x_min) - 1.0
    grid_y = 2.0 * (world_y - y_min) / (y_max - y_min) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).reshape(
        count, output_size, output_size, 2
    )

    batched = features.unsqueeze(0).expand(count, -1, -1, -1)
    return F.grid_sample(batched, grid, align_corners=False, padding_mode="zeros")


class ObjectEmbedding(nn.Module):
    """Map pooled ROI features to an L2-normalized per-object descriptor."""

    def __init__(self, in_channels: int, output_size: int, dim: int) -> None:
        super().__init__()
        flat = in_channels * output_size * output_size
        self.net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(flat, 2 * dim),
            nn.LayerNorm(2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        self.dim = dim

    def forward(self, roi: Tensor) -> Tensor:
        """Return ``(M, dim)`` unit-norm embeddings for ``(M, C, k, k)`` ROI features."""
        if roi.shape[0] == 0:
            return roi.new_zeros((0, self.dim))
        return F.normalize(self.net(roi), p=2.0, dim=1)
```

- [ ] **Step 4: Run to verify they pass**

```bash
python -m pytest tests/test_alignformer_embedding.py -v
```

Expected: 6 passed. If `test_sampling_is_canonical_...` fails, the rotation
convention is wrong - check that `along` uses index 5 (length) and `across` index 4
(width), matching `features/opencood_proposals.py:proposal_roi_cell_indices`.

- [ ] **Step 5: Commit**

```bash
git add src/embedding_aware_belt_fusion/alignformer/embedding.py \
        tests/test_alignformer_embedding.py
git commit -m "feat: rotated ROI-align and per-object embedding head

Sampling in each box's canonical frame makes the embedding pose-invariant, so
it describes the object rather than where the sender thinks it is."
```

---

### Task 6: `alignformer/head_match.py` - Sinkhorn soft assignment with dustbins

**Files:**
- Create: `src/embedding_aware_belt_fusion/alignformer/head_match.py`
- Test: `tests/test_alignformer_head_match.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `log_sinkhorn(scores, alpha, iterations) -> Tensor (B, M+1, N+1)` returning a
    log-domain assignment whose exponential has row sums 1 over the first M rows
    (including the dustbin column) and column sums 1 over the first N columns.
  - `DEFAULT_SINKHORN_ITERATIONS = 20`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_alignformer_head_match.py
import torch

from embedding_aware_belt_fusion.alignformer.head_match import (
    DEFAULT_SINKHORN_ITERATIONS,
    log_sinkhorn,
)


def test_assignment_rows_and_columns_are_normalized():
    scores = torch.randn(2, 4, 6)
    alpha = torch.tensor(0.5)

    assignment = log_sinkhorn(scores, alpha, DEFAULT_SINKHORN_ITERATIONS).exp()

    # Each real row distributes unit mass across real columns plus its dustbin.
    assert torch.allclose(assignment[:, :-1, :].sum(dim=2), torch.ones(2, 4), atol=1e-4)
    assert torch.allclose(assignment[:, :, :-1].sum(dim=1), torch.ones(2, 6), atol=1e-4)


def test_shape_includes_dustbin_row_and_column():
    assignment = log_sinkhorn(torch.randn(1, 3, 5), torch.tensor(0.0), 10)

    assert assignment.shape == (1, 4, 6)


def test_a_dominant_score_wins_its_row():
    scores = torch.full((1, 2, 2), -5.0)
    scores[0, 0, 1] = 10.0

    assignment = log_sinkhorn(scores, torch.tensor(0.0), 50).exp()

    assert assignment[0, 0, 1] > 0.9


def test_handles_asymmetric_and_single_object_sets():
    assignment = log_sinkhorn(torch.randn(1, 1, 7), torch.tensor(0.0), 20)

    assert assignment.shape == (1, 2, 8)
    assert torch.isfinite(assignment).all()


def test_an_empty_object_set_does_not_produce_inf_or_nan():
    # An agent can legitimately detect nothing. log(0) in the marginals would
    # put -inf into the iteration and NaN into the gradients.
    assignment = log_sinkhorn(torch.randn(1, 0, 4), torch.tensor(0.0), 20)

    assert assignment.shape == (1, 1, 5)
    assert torch.isfinite(assignment).all()


def test_is_differentiable_with_finite_gradients():
    scores = torch.randn(1, 3, 3, requires_grad=True)
    alpha = torch.tensor(0.3, requires_grad=True)

    log_sinkhorn(scores, alpha, 20).exp().sum().backward()

    assert torch.isfinite(scores.grad).all()
    assert torch.isfinite(alpha.grad).all()
```

- [ ] **Step 2: Run to verify they fail**

```bash
python -m pytest tests/test_alignformer_head_match.py -v
```

Expected: FAIL, module not found.

- [ ] **Step 3: Implement**

```python
# src/embedding_aware_belt_fusion/alignformer/head_match.py
"""Log-domain Sinkhorn with dustbins, for soft cross-agent correspondence.

Objects seen by only one agent have no counterpart, so the score matrix carries
a dustbin row and column that absorb their mass. Running Sinkhorn in the log
domain keeps it stable for the peaked score matrices a trained model produces.

Follows Sarlin et al., "SuperGlue: Learning Feature Matching with Graph Neural
Networks", CVPR 2020.
"""

from __future__ import annotations

import torch
from torch import Tensor

DEFAULT_SINKHORN_ITERATIONS = 20


def log_sinkhorn(scores: Tensor, alpha: Tensor, iterations: int) -> Tensor:
    """Normalize ``scores`` into a log-domain soft assignment with dustbins.

    Parameters
    ----------
    scores: ``(B, M, N)`` match scores between ego and CAV objects.
    alpha: scalar tensor, the learnable dustbin score.
    iterations: number of Sinkhorn normalization steps.

    Returns
    -------
    Tensor
        ``(B, M + 1, N + 1)`` log-assignment. The last row and column are the
        dustbins.
    """
    if scores.dim() != 3:
        raise ValueError(f"scores must be (B, M, N), got {tuple(scores.shape)}")
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")

    batch, rows, columns = scores.shape
    bin_score = alpha.to(scores)

    # An agent can legitimately detect nothing. With rows or columns at zero the
    # marginals would contain log(0) = -inf, which turns into NaN gradients.
    # There is nothing to normalize in that case, so return the couplings as-is.
    if rows == 0 or columns == 0:
        return torch.cat(
            [
                torch.cat([scores, bin_score.expand(batch, rows, 1)], dim=2),
                torch.cat(
                    [
                        bin_score.expand(batch, 1, columns),
                        bin_score.expand(batch, 1, 1),
                    ],
                    dim=2,
                ),
            ],
            dim=1,
        )

    row_count = scores.new_tensor(float(rows))
    column_count = scores.new_tensor(float(columns))

    couplings = torch.cat(
        [
            torch.cat([scores, bin_score.expand(batch, rows, 1)], dim=2),
            torch.cat(
                [bin_score.expand(batch, 1, columns), bin_score.expand(batch, 1, 1)],
                dim=2,
            ),
        ],
        dim=1,
    )

    normalizer = -(row_count + column_count).log()
    log_mu = torch.cat(
        [normalizer.expand(rows), column_count.log().reshape(1) + normalizer]
    ).expand(batch, -1)
    log_nu = torch.cat(
        [normalizer.expand(columns), row_count.log().reshape(1) + normalizer]
    ).expand(batch, -1)

    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(iterations):
        u = log_mu - torch.logsumexp(couplings + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(couplings + u.unsqueeze(2), dim=1)

    return couplings + u.unsqueeze(2) + v.unsqueeze(1) - normalizer
```

- [ ] **Step 4: Run to verify they pass**

```bash
python -m pytest tests/test_alignformer_head_match.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/embedding_aware_belt_fusion/alignformer/head_match.py \
        tests/test_alignformer_head_match.py
git commit -m "feat: log-domain Sinkhorn soft assignment with dustbins

Dustbins absorb objects that only one agent sees, which is the common case for
partially overlapping fields of view."
```

---

### Task 7: `alignformer/trunk.py` - shared transformer over the two object sets

**Files:**
- Create: `src/embedding_aware_belt_fusion/alignformer/trunk.py`
- Test: `tests/test_alignformer_trunk.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `tokenize(boxes, scores, embeddings) -> Tensor (B, N, 9 + d)` raw token features.
  - `AlignFormerTrunk(embed_dim, model_dim, layers, heads)` with
    `forward(ego_tokens, cav_tokens, ego_mask, cav_mask) -> tuple[Tensor, Tensor]`
    returning `(B, M, model_dim)` and `(B, N, model_dim)`. Masks are `(B, N)` bool,
    True for real objects.
  - `DEFAULT_MODEL_DIM = 256`, `DEFAULT_LAYERS = 4`, `DEFAULT_HEADS = 4`,
    `MAX_OBJECTS = 64`, `GEOMETRY_FEATURES = 9`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_alignformer_trunk.py
import torch

from embedding_aware_belt_fusion.alignformer.trunk import (
    DEFAULT_HEADS,
    DEFAULT_LAYERS,
    DEFAULT_MODEL_DIM,
    AlignFormerTrunk,
    tokenize,
)


def _trunk(embed_dim=16):
    return AlignFormerTrunk(
        embed_dim=embed_dim,
        model_dim=DEFAULT_MODEL_DIM,
        layers=DEFAULT_LAYERS,
        heads=DEFAULT_HEADS,
    ).eval()


def test_tokenize_concatenates_geometry_score_and_embedding():
    boxes = torch.randn(2, 5, 7)
    scores = torch.rand(2, 5)
    embeddings = torch.randn(2, 5, 16)

    tokens = tokenize(boxes, scores, embeddings)

    # 8 geometry features (x, y, z, h, w, l, cos yaw, sin yaw) + score + embedding
    assert tokens.shape == (2, 5, 9 + 16)


def test_forward_preserves_per_set_shapes():
    trunk = _trunk()
    ego = torch.randn(2, 5, 25)
    cav = torch.randn(2, 7, 25)
    ego_mask = torch.ones(2, 5, dtype=torch.bool)
    cav_mask = torch.ones(2, 7, dtype=torch.bool)

    ego_out, cav_out = trunk(ego, cav, ego_mask, cav_mask)

    assert ego_out.shape == (2, 5, DEFAULT_MODEL_DIM)
    assert cav_out.shape == (2, 7, DEFAULT_MODEL_DIM)


def test_output_is_permutation_equivariant():
    # Object sets have no intrinsic order, so permuting the input must permute
    # the output identically. A positional leak would break matching.
    torch.manual_seed(0)
    trunk = _trunk()
    ego = torch.randn(1, 4, 25)
    cav = torch.randn(1, 3, 25)
    ego_mask = torch.ones(1, 4, dtype=torch.bool)
    cav_mask = torch.ones(1, 3, dtype=torch.bool)

    with torch.no_grad():
        base, _ = trunk(ego, cav, ego_mask, cav_mask)
        order = torch.tensor([2, 0, 3, 1])
        permuted, _ = trunk(ego[:, order], cav, ego_mask[:, order], cav_mask)

    assert torch.allclose(base[:, order], permuted, atol=1e-5)


def test_padded_objects_do_not_affect_real_ones():
    torch.manual_seed(0)
    trunk = _trunk()
    ego = torch.randn(1, 3, 25)
    cav = torch.randn(1, 2, 25)
    ego_mask = torch.ones(1, 3, dtype=torch.bool)
    cav_mask = torch.ones(1, 2, dtype=torch.bool)

    padded_ego = torch.cat([ego, torch.randn(1, 4, 25)], dim=1)
    padded_ego_mask = torch.cat([ego_mask, torch.zeros(1, 4, dtype=torch.bool)], dim=1)

    with torch.no_grad():
        base, _ = trunk(ego, cav, ego_mask, cav_mask)
        padded, _ = trunk(padded_ego, cav, padded_ego_mask, cav_mask)

    assert torch.allclose(base, padded[:, :3], atol=1e-5)
```

- [ ] **Step 2: Run to verify they fail**

```bash
python -m pytest tests/test_alignformer_trunk.py -v
```

Expected: FAIL, module not found.

- [ ] **Step 3: Implement**

```python
# src/embedding_aware_belt_fusion/alignformer/trunk.py
"""Shared transformer over the ego and CAV object sets.

Both heads consume this trunk unchanged, so an A-vs-B comparison isolates the
head. Cross-attention between the two sets supplies the multiplicative ego x CAV
interaction whose absence made CoLoca-QuA's linear tokenization unable to
register; see docs/coloca_qua_baseline.md.

Object sets are unordered, so the trunk carries no positional encoding - spatial
information enters only through each token's own box geometry.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

DEFAULT_MODEL_DIM = 256
DEFAULT_LAYERS = 4
DEFAULT_HEADS = 4
# Token budget per agent; objects beyond this are dropped by descending score.
MAX_OBJECTS = 64

# 8 geometry features plus the detector score.
GEOMETRY_FEATURES = 9
_BOX_YAW = 6


def tokenize(boxes: Tensor, scores: Tensor, embeddings: Tensor) -> Tensor:
    """Build raw token features from boxes, scores and embeddings.

    Yaw is encoded as ``(cos, sin)`` so the representation is continuous across
    the +/-pi wrap.

    Parameters
    ----------
    boxes: ``(B, N, 7)`` in ``hwl`` order, already in the ego frame.
    scores: ``(B, N)`` detector confidences.
    embeddings: ``(B, N, d)`` unit-norm per-object descriptors.

    Returns
    -------
    Tensor
        ``(B, N, GEOMETRY_FEATURES + d)``.
    """
    yaw = boxes[..., _BOX_YAW]
    geometry = torch.cat(
        [
            boxes[..., :6],
            torch.cos(yaw).unsqueeze(-1),
            torch.sin(yaw).unsqueeze(-1),
            scores.unsqueeze(-1),
        ],
        dim=-1,
    )
    return torch.cat([geometry, embeddings], dim=-1)


class _Layer(nn.Module):
    """One self-attention pass within each set, then cross-attention across."""

    def __init__(self, model_dim: int, heads: int) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(model_dim, heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(model_dim, heads, batch_first=True)
        self.norm_self = nn.LayerNorm(model_dim)
        self.norm_cross = nn.LayerNorm(model_dim)
        self.norm_ff = nn.LayerNorm(model_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim),
            nn.GELU(),
            nn.Linear(4 * model_dim, model_dim),
        )

    def forward(self, x: Tensor, y: Tensor, x_mask: Tensor, y_mask: Tensor) -> Tensor:
        # key_padding_mask marks positions to IGNORE, so invert the validity mask.
        normed = self.norm_self(x)
        attended, _ = self.self_attention(
            normed, normed, normed, key_padding_mask=~x_mask, need_weights=False
        )
        x = x + attended

        normed_x, normed_y = self.norm_cross(x), self.norm_cross(y)
        attended, _ = self.cross_attention(
            normed_x, normed_y, normed_y, key_padding_mask=~y_mask, need_weights=False
        )
        x = x + attended

        return x + self.feed_forward(self.norm_ff(x))


class AlignFormerTrunk(nn.Module):
    """Interleaved self- and cross-attention over the two object sets."""

    def __init__(
        self,
        embed_dim: int,
        model_dim: int = DEFAULT_MODEL_DIM,
        layers: int = DEFAULT_LAYERS,
        heads: int = DEFAULT_HEADS,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(GEOMETRY_FEATURES + embed_dim, model_dim)
        self.set_embedding = nn.Parameter(torch.zeros(2, model_dim))
        self.ego_layers = nn.ModuleList(_Layer(model_dim, heads) for _ in range(layers))
        self.cav_layers = nn.ModuleList(_Layer(model_dim, heads) for _ in range(layers))
        self.output_norm = nn.LayerNorm(model_dim)
        self.model_dim = model_dim

    def forward(
        self, ego_tokens: Tensor, cav_tokens: Tensor, ego_mask: Tensor, cav_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Encode both sets, each attending to itself and to the other."""
        ego = self.input_projection(ego_tokens) + self.set_embedding[0]
        cav = self.input_projection(cav_tokens) + self.set_embedding[1]

        for ego_layer, cav_layer in zip(self.ego_layers, self.cav_layers):
            next_ego = ego_layer(ego, cav, ego_mask, cav_mask)
            next_cav = cav_layer(cav, ego, cav_mask, ego_mask)
            ego, cav = next_ego, next_cav

        ego = self.output_norm(ego) * ego_mask.unsqueeze(-1)
        cav = self.output_norm(cav) * cav_mask.unsqueeze(-1)
        return ego, cav
```

- [ ] **Step 4: Run to verify they pass**

```bash
python -m pytest tests/test_alignformer_trunk.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/embedding_aware_belt_fusion/alignformer/trunk.py \
        tests/test_alignformer_trunk.py
git commit -m "feat: shared ego/CAV transformer trunk

Cross-attention supplies the multiplicative ego x CAV interaction that
CoLoca-QuA's linear tokenization could not express. No positional encoding:
object sets are unordered, and the tests assert permutation equivariance."
```

---

### Task 8: `alignformer/losses.py` - corner loss and match NLL

**Files:**
- Create: `src/embedding_aware_belt_fusion/alignformer/losses.py`
- Test: `tests/test_alignformer_losses.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `bev_corners(boxes) -> Tensor (B, N, 4, 2)`
  - `apply_se2(points, psi, t) -> Tensor` for `points (B, N, 2)` or `(B, N, 4, 2)`
  - `corner_loss(boxes, psi_pred, t_pred, psi_true, t_true, mask) -> Tensor` scalar
  - `match_nll(log_assignment, ego_match, cav_match) -> Tensor` scalar, where
    `ego_match (B, M)` holds the matched CAV index per ego object or `-1`, and
    `cav_match (B, N)` the converse.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_alignformer_losses.py
import math

import pytest
import torch

from embedding_aware_belt_fusion.alignformer.losses import (
    apply_se2,
    bev_corners,
    corner_loss,
    match_nll,
)


def test_bev_corners_of_an_axis_aligned_box():
    # hwl order: [x, y, z, h, w, l, yaw], so width = 2, length = 4.
    boxes = torch.tensor([[[0.0, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0]]])

    corners = bev_corners(boxes)

    assert corners.shape == (1, 1, 4, 2)
    assert corners[0, 0, :, 0].abs().max().item() == pytest.approx(2.0)
    assert corners[0, 0, :, 1].abs().max().item() == pytest.approx(1.0)


def test_corner_loss_is_zero_for_a_perfect_prediction():
    boxes = torch.randn(2, 5, 7)
    psi = torch.tensor([0.1, -0.3])
    t = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
    mask = torch.ones(2, 5, dtype=torch.bool)

    loss = corner_loss(boxes, psi, t, psi, t, mask)

    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_corner_loss_grows_with_yaw_error():
    boxes = torch.tensor([[[30.0, 10.0, 0.0, 1.5, 2.0, 4.0, 0.0]]])
    mask = torch.ones(1, 1, dtype=torch.bool)
    zero_t = torch.zeros(1, 2)
    truth = torch.zeros(1)

    small = corner_loss(boxes, torch.tensor([0.01]), zero_t, truth, zero_t, mask)
    large = corner_loss(boxes, torch.tensor([0.10]), zero_t, truth, zero_t, mask)

    assert large.item() > small.item()


def test_corner_loss_ignores_padded_objects():
    boxes = torch.zeros(1, 3, 7)
    boxes[0, 2] = torch.tensor([500.0, 500.0, 0.0, 1.5, 2.0, 4.0, 0.0])
    mask = torch.tensor([[True, True, False]])

    loss = corner_loss(
        boxes, torch.tensor([0.2]), torch.zeros(1, 2),
        torch.zeros(1), torch.zeros(1, 2), mask,
    )

    # The padded far-away object would dominate if it were counted.
    assert loss.item() < 1.0


def test_apply_se2_rotates_and_translates():
    points = torch.tensor([[[1.0, 0.0]]])

    moved = apply_se2(points, torch.tensor([math.pi / 2]), torch.tensor([[0.0, 1.0]]))

    assert moved[0, 0, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert moved[0, 0, 1].item() == pytest.approx(2.0, abs=1e-6)


def test_match_nll_is_low_for_a_confident_correct_assignment():
    log_assignment = torch.full((1, 3, 3), -20.0)
    log_assignment[0, 0, 1] = 0.0
    log_assignment[0, 1, 0] = 0.0
    ego_match = torch.tensor([[1, 0]])
    cav_match = torch.tensor([[1, 0]])

    loss = match_nll(log_assignment, ego_match, cav_match)

    assert loss.item() < 0.1


def test_match_nll_charges_unmatched_objects_to_the_dustbin():
    log_assignment = torch.full((1, 3, 3), -20.0)
    log_assignment[0, 0, 2] = 0.0  # ego object 0 -> dustbin
    ego_match = torch.tensor([[-1, -1]])
    cav_match = torch.tensor([[-1, -1]])

    loss = match_nll(log_assignment, ego_match, cav_match)

    assert torch.isfinite(loss)
```

- [ ] **Step 2: Run to verify they fail**

```bash
python -m pytest tests/test_alignformer_losses.py -v
```

Expected: FAIL, module not found.

- [ ] **Step 3: Implement**

```python
# src/embedding_aware_belt_fusion/alignformer/losses.py
"""Training objectives for AlignFormer.

The pose objective is a **corner loss** rather than CoLoca-QuA's weighted
``(2, 2, 1)`` MSE on ``(dx, dy, dpsi)``. That formulation mixes metres and
radians under arbitrary weights, and the reproduction showed yaw dominating the
loss while still not being learned. Measuring displacement of the box corners is
in metres throughout, weights a yaw error by how far away the object is, and
optimizes exactly what IoU and AP measure.
"""

from __future__ import annotations

import torch
from torch import Tensor

# Box layout is OpenCOOD 'hwl': [x, y, z, h, w, l, yaw].
_BOX_WIDTH = 4
_BOX_LENGTH = 5
_BOX_YAW = 6


def bev_corners(boxes: Tensor) -> Tensor:
    """Return the 4 BEV corners of each box as ``(B, N, 4, 2)``."""
    if boxes.shape[-1] != 7:
        raise ValueError(f"boxes must end in 7 features, got {tuple(boxes.shape)}")

    half_length = boxes[..., _BOX_LENGTH] / 2
    half_width = boxes[..., _BOX_WIDTH] / 2
    signs = boxes.new_tensor([[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]])

    along = half_length.unsqueeze(-1) * signs[:, 0]
    across = half_width.unsqueeze(-1) * signs[:, 1]

    cosine = torch.cos(boxes[..., _BOX_YAW]).unsqueeze(-1)
    sine = torch.sin(boxes[..., _BOX_YAW]).unsqueeze(-1)
    x = boxes[..., 0:1] + along * cosine - across * sine
    y = boxes[..., 1:2] + along * sine + across * cosine
    return torch.stack([x, y], dim=-1)


def apply_se2(points: Tensor, psi: Tensor, t: Tensor) -> Tensor:
    """Apply a per-batch SE(2) transform to ``(B, ..., 2)`` points."""
    extra_dims = points.dim() - 2
    shape = (psi.shape[0],) + (1,) * extra_dims
    cosine = torch.cos(psi).reshape(shape)
    sine = torch.sin(psi).reshape(shape)

    x, y = points[..., 0], points[..., 1]
    rotated = torch.stack([cosine * x - sine * y, sine * x + cosine * y], dim=-1)
    return rotated + t.reshape(shape + (2,))


def corner_loss(
    boxes: Tensor,
    psi_pred: Tensor,
    t_pred: Tensor,
    psi_true: Tensor,
    t_true: Tensor,
    mask: Tensor,
) -> Tensor:
    """Mean L1 displacement of BEV corners under the predicted vs true correction.

    Parameters
    ----------
    boxes: ``(B, N, 7)`` CAV boxes in the ego frame, before correction.
    psi_pred, psi_true: ``(B,)`` yaw corrections in radians.
    t_pred, t_true: ``(B, 2)`` translation corrections in metres.
    mask: ``(B, N)`` bool, True for real objects.

    Returns
    -------
    Tensor
        Scalar loss in metres, averaged over real corners only.
    """
    corners = bev_corners(boxes)
    predicted = apply_se2(corners, psi_pred, t_pred)
    target = apply_se2(corners, psi_true, t_true)

    per_corner = (predicted - target).abs().sum(dim=-1)
    weights = mask.unsqueeze(-1).to(per_corner.dtype)
    return (per_corner * weights).sum() / weights.sum().clamp_min(1.0)


def match_nll(log_assignment: Tensor, ego_match: Tensor, cav_match: Tensor) -> Tensor:
    """Negative log-likelihood of the ground-truth assignment, dustbins included.

    Parameters
    ----------
    log_assignment: ``(B, M + 1, N + 1)`` from :func:`log_sinkhorn`.
    ego_match: ``(B, M)`` matched CAV index per ego object, ``-1`` if unmatched.
    cav_match: ``(B, N)`` matched ego index per CAV object, ``-1`` if unmatched.

    Unmatched objects are supervised onto their dustbin, which is what teaches
    the model that a detection seen by only one agent must not be matched.
    """
    _, rows, columns = log_assignment.shape
    ego_count, cav_count = rows - 1, columns - 1

    ego_target = torch.where(
        ego_match < 0, torch.full_like(ego_match, cav_count), ego_match
    )
    cav_target = torch.where(
        cav_match < 0, torch.full_like(cav_match, ego_count), cav_match
    )

    ego_terms = (
        log_assignment[:, :ego_count, :].gather(2, ego_target.unsqueeze(-1)).squeeze(-1)
    )
    cav_terms = (
        log_assignment[:, :, :cav_count].gather(1, cav_target.unsqueeze(1)).squeeze(1)
    )

    return -(ego_terms.mean() + cav_terms.mean()) / 2
```

- [ ] **Step 4: Run to verify they pass**

```bash
python -m pytest tests/test_alignformer_losses.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/embedding_aware_belt_fusion/alignformer/losses.py \
        tests/test_alignformer_losses.py
git commit -m "feat: corner loss and match NLL

Corner loss replaces CoLoca's weighted (2,2,1) MSE: it is in metres throughout,
weights yaw error by object distance, and optimizes what AP actually measures."
```

---

### Task 9: `alignformer/model.py` - assemble Heads A and B

**Files:**
- Create: `src/embedding_aware_belt_fusion/alignformer/model.py`
- Test: `tests/test_alignformer_model.py`

**Interfaces:**
- Consumes: `trunk.AlignFormerTrunk`, `trunk.tokenize`, `head_match.log_sinkhorn`,
  `procrustes.weighted_se2_kabsch`, `procrustes.augment_with_heading`.
- Produces:
  - `PoseEstimate` frozen dataclass: `psi (B,)`, `t (B,2)`, `confidence (B,)`,
    `log_assignment: Tensor | None`.
  - `AlignFormerB(embed_dim, **trunk_kwargs).forward(batch) -> PoseEstimate`
  - `AlignFormerA(embed_dim, **trunk_kwargs).forward(batch) -> PoseEstimate`
  - Both take `batch` as a dict with keys `ego_boxes (B,M,7)`, `ego_scores (B,M)`,
    `ego_embeddings (B,M,d)`, `ego_mask (B,M)` and the `cav_*` equivalents.
  - `DEFAULT_HEADING_LAMBDA = 2.0`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_alignformer_model.py
import math

import pytest
import torch

from embedding_aware_belt_fusion.alignformer.model import AlignFormerA, AlignFormerB


def _batch(ego_boxes, cav_boxes, embed_dim=8, ego_embeddings=None, cav_embeddings=None):
    ego_count, cav_count = ego_boxes.shape[1], cav_boxes.shape[1]
    return {
        "ego_boxes": ego_boxes,
        "ego_scores": torch.ones(1, ego_count),
        "ego_embeddings": (
            ego_embeddings if ego_embeddings is not None
            else torch.zeros(1, ego_count, embed_dim)
        ),
        "ego_mask": torch.ones(1, ego_count, dtype=torch.bool),
        "cav_boxes": cav_boxes,
        "cav_scores": torch.ones(1, cav_count),
        "cav_embeddings": (
            cav_embeddings if cav_embeddings is not None
            else torch.zeros(1, cav_count, embed_dim)
        ),
        "cav_mask": torch.ones(1, cav_count, dtype=torch.bool),
    }


def test_head_b_returns_a_pose_estimate_with_an_assignment():
    model = AlignFormerB(embed_dim=8).eval()
    batch = _batch(torch.randn(1, 4, 7), torch.randn(1, 5, 7))

    estimate = model(batch)

    assert estimate.psi.shape == (1,)
    assert estimate.t.shape == (1, 2)
    assert estimate.confidence.shape == (1,)
    assert estimate.log_assignment.shape == (1, 5, 6)


def test_head_a_returns_a_pose_estimate_without_an_assignment():
    model = AlignFormerA(embed_dim=8).eval()
    batch = _batch(torch.randn(1, 4, 7), torch.randn(1, 5, 7))

    estimate = model(batch)

    assert estimate.psi.shape == (1,)
    assert estimate.t.shape == (1, 2)
    assert estimate.log_assignment is None


def test_head_b_recovers_the_true_transform_given_oracle_embeddings():
    # Orthogonal one-hot embeddings make the correspondence unambiguous, so the
    # closed-form solver must recover the transform almost exactly. This is the
    # end-to-end sanity check that the wiring - not just each part - is correct.
    torch.manual_seed(0)
    count = 6
    identity = torch.eye(count).unsqueeze(0)

    centres = torch.tensor([[[0.0, 0.0], [12.0, 3.0], [-8.0, 5.0],
                             [20.0, -7.0], [4.0, 9.0], [-15.0, -2.0]]])
    yaws = torch.rand(1, count) * 2 * math.pi
    cav_boxes = torch.cat(
        [
            centres,
            torch.zeros(1, count, 1),
            torch.full((1, count, 1), 1.5),
            torch.full((1, count, 1), 2.0),
            torch.full((1, count, 1), 4.0),
            yaws.unsqueeze(-1),
        ],
        dim=-1,
    )

    true_psi, true_t = 0.15, torch.tensor([[1.2, -0.8]])
    cos, sin = math.cos(true_psi), math.sin(true_psi)
    rotation = torch.tensor([[cos, -sin], [sin, cos]])
    ego_boxes = cav_boxes.clone()
    ego_boxes[..., :2] = centres @ rotation.T + true_t
    ego_boxes[..., 6] = yaws + true_psi

    model = AlignFormerB(embed_dim=count).eval()
    batch = _batch(
        ego_boxes, cav_boxes,
        ego_embeddings=identity.clone(), cav_embeddings=identity.clone(),
    )
    # Bypass the untrained trunk: score directly on the oracle embeddings.
    model.use_raw_embedding_scores = True

    with torch.no_grad():
        estimate = model(batch)

    assert estimate.psi.item() == pytest.approx(true_psi, abs=1e-3)
    assert estimate.t[0, 0].item() == pytest.approx(1.2, abs=1e-2)


def test_confidence_is_zero_when_the_ego_set_is_empty():
    model = AlignFormerB(embed_dim=8).eval()
    batch = _batch(torch.zeros(1, 0, 7), torch.randn(1, 3, 7))

    estimate = model(batch)

    assert estimate.confidence.item() == pytest.approx(0.0)
    assert estimate.psi.item() == pytest.approx(0.0)
    assert torch.allclose(estimate.t, torch.zeros(1, 2))


def test_gradients_reach_the_trunk_through_the_closed_form_solver():
    model = AlignFormerB(embed_dim=8)
    batch = _batch(torch.randn(1, 4, 7), torch.randn(1, 4, 7))

    estimate = model(batch)
    (estimate.psi.sum() + estimate.t.sum()).backward()

    gradients = [p.grad for p in model.trunk.parameters() if p.grad is not None]
    assert gradients, "no gradient reached the trunk"
    assert all(torch.isfinite(g).all() for g in gradients)
```

- [ ] **Step 2: Run to verify they fail**

```bash
python -m pytest tests/test_alignformer_model.py -v
```

Expected: FAIL, module not found.

- [ ] **Step 3: Implement**

```python
# src/embedding_aware_belt_fusion/alignformer/model.py
"""The two interchangeable pose heads, over a shared trunk.

Head A regresses the pose from a mean-pooled descriptor. Head B builds a soft
correspondence and solves for the pose in closed form. They share tokenization
and trunk exactly, so a comparison between them isolates the head - the same
methodology that isolated CoLoca-QuA's two deficiencies.

Head A deliberately mean-pools rather than reading out a learned query token:
the reproduction showed a content-free token entering the residual stream makes
the prediction nearly input-independent and collapses it to the conditional mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from embedding_aware_belt_fusion.alignformer.head_match import (
    DEFAULT_SINKHORN_ITERATIONS,
    log_sinkhorn,
)
from embedding_aware_belt_fusion.alignformer.procrustes import (
    augment_with_heading,
    weighted_se2_kabsch,
)
from embedding_aware_belt_fusion.alignformer.trunk import AlignFormerTrunk, tokenize

# Heading virtual-point offset in metres, about half a vehicle length so that
# headings and centres contribute comparably to the Kabsch fit.
DEFAULT_HEADING_LAMBDA = 2.0
# Score assigned to padded positions so they can never win a match.
_MASKED_SCORE = -1e4
_BOX_YAW = 6


@dataclass(frozen=True)
class PoseEstimate:
    """A CAV's estimated SE(2) correction, with the evidence behind it."""

    psi: Tensor
    t: Tensor
    confidence: Tensor
    log_assignment: Optional[Tensor] = None


def _tokens(batch: Mapping[str, Tensor], prefix: str) -> Tensor:
    return tokenize(
        batch[f"{prefix}_boxes"], batch[f"{prefix}_scores"], batch[f"{prefix}_embeddings"]
    )


def _masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    weights = mask.unsqueeze(-1).to(x.dtype)
    return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class _Base(nn.Module):
    def __init__(self, embed_dim: int, **trunk_kwargs) -> None:
        super().__init__()
        self.trunk = AlignFormerTrunk(embed_dim=embed_dim, **trunk_kwargs)

    def _encode(self, batch: Mapping[str, Tensor]) -> Tuple[Tensor, Tensor]:
        return self.trunk(
            _tokens(batch, "ego"),
            _tokens(batch, "cav"),
            batch["ego_mask"],
            batch["cav_mask"],
        )


class AlignFormerA(_Base):
    """Direct regression from a mean-pooled descriptor of both sets."""

    def __init__(self, embed_dim: int, **trunk_kwargs) -> None:
        super().__init__(embed_dim, **trunk_kwargs)
        model_dim = self.trunk.model_dim
        self.head = nn.Sequential(
            nn.Linear(2 * model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 3),
        )

    def forward(self, batch: Mapping[str, Tensor]) -> PoseEstimate:
        ego, cav = self._encode(batch)
        pooled = torch.cat(
            [_masked_mean(ego, batch["ego_mask"]), _masked_mean(cav, batch["cav_mask"])],
            dim=-1,
        )
        output = self.head(pooled)

        present = batch["ego_mask"].any(dim=1) & batch["cav_mask"].any(dim=1)
        psi = torch.where(present, output[:, 2], torch.zeros_like(output[:, 2]))
        t = torch.where(
            present.unsqueeze(-1), output[:, :2], torch.zeros_like(output[:, :2])
        )
        return PoseEstimate(psi=psi, t=t, confidence=present.to(psi.dtype))


class AlignFormerB(_Base):
    """Soft correspondence, then a closed-form weighted SE(2) Kabsch solve."""

    def __init__(
        self,
        embed_dim: int,
        heading_lambda: float = DEFAULT_HEADING_LAMBDA,
        sinkhorn_iterations: int = DEFAULT_SINKHORN_ITERATIONS,
        **trunk_kwargs,
    ) -> None:
        super().__init__(embed_dim, **trunk_kwargs)
        model_dim = self.trunk.model_dim
        self.match_projection = nn.Linear(model_dim, model_dim)
        self.dustbin = nn.Parameter(torch.tensor(1.0))
        self.log_temperature = nn.Parameter(torch.tensor(0.1).log())
        self.heading_lambda = heading_lambda
        self.sinkhorn_iterations = sinkhorn_iterations
        # Test hook: score on the raw embeddings, bypassing an untrained trunk.
        self.use_raw_embedding_scores = False

    def forward(self, batch: Mapping[str, Tensor]) -> PoseEstimate:
        ego_mask, cav_mask = batch["ego_mask"], batch["cav_mask"]
        ego, cav = self._encode(batch)

        if self.use_raw_embedding_scores:
            ego_features = batch["ego_embeddings"]
            cav_features = batch["cav_embeddings"]
        else:
            ego_features = F.normalize(self.match_projection(ego), dim=-1)
            cav_features = F.normalize(self.match_projection(cav), dim=-1)

        scores = ego_features @ cav_features.transpose(1, 2) / self.log_temperature.exp()
        valid = ego_mask.unsqueeze(2) & cav_mask.unsqueeze(1)
        scores = scores.masked_fill(~valid, _MASKED_SCORE)

        log_assignment = log_sinkhorn(scores, self.dustbin, self.sinkhorn_iterations)
        weights = log_assignment[:, :-1, :-1].exp() * valid

        mass = weights.sum(dim=2)
        safe_mass = mass.clamp_min(1e-6).unsqueeze(-1)
        cav_centres = batch["cav_boxes"][..., :2]
        cav_yaws = batch["cav_boxes"][..., _BOX_YAW]

        virtual_centres = weights @ cav_centres / safe_mass
        virtual_direction = (
            weights @ torch.stack([torch.cos(cav_yaws), torch.sin(cav_yaws)], dim=-1)
            / safe_mass
        )
        virtual_yaws = torch.atan2(virtual_direction[..., 1], virtual_direction[..., 0])

        source = augment_with_heading(virtual_centres, virtual_yaws, self.heading_lambda)
        target = augment_with_heading(
            batch["ego_boxes"][..., :2],
            batch["ego_boxes"][..., _BOX_YAW],
            self.heading_lambda,
        )
        augmented_mass = torch.cat([mass, mass], dim=1)

        psi, t = weighted_se2_kabsch(target, source, augmented_mass)
        return PoseEstimate(
            psi=psi, t=t, confidence=mass.sum(dim=1), log_assignment=log_assignment
        )
```

Note: with an empty ego set, `mass` has shape `(B, 0)`, its sum is 0, so
`weighted_se2_kabsch` suppresses the correction via `MIN_MATCH_MASS` and
`confidence` is 0 - which is what the empty-set test asserts.

- [ ] **Step 4: Run to verify they pass**

```bash
python -m pytest tests/test_alignformer_model.py -v
```

Expected: 5 passed. If `test_head_b_recovers_the_true_transform...` fails, check the
Kabsch direction: the solver maps CAV points **onto** ego points, so `target` is the
ego set and `source` the CAV set.

- [ ] **Step 5: Commit**

```bash
git add src/embedding_aware_belt_fusion/alignformer/model.py \
        tests/test_alignformer_model.py
git commit -m "feat: AlignFormer heads A and B over a shared trunk

Head A mean-pools rather than reading a learned query token, per the CoLoca
finding that a content-free token collapses to the conditional mean. Head B
solves the pose in closed form from Sinkhorn correspondences."
```

---

### Task 10: `alignformer/cache.py` - ROI feature and detection cache

**Files:**
- Create: `src/embedding_aware_belt_fusion/alignformer/cache.py`
- Test: `tests/test_alignformer_cache.py`

**Interfaces:**
- Consumes: `boxes.detect_agent`, `embedding.rotated_roi_align`, and
  `coloca/index.py:scan_split` for enumeration.
- Produces:
  - `FrameRecord` frozen dataclass: `boxes np.ndarray (M,7)`, `scores (M,)`,
    `gt_ids list[str|None]`, `roi (M,C,k,k) float16`.
  - `write_frame(path, record) -> None` and `read_frame(path) -> FrameRecord`
  - `cache_path(cache_root, split, scenario, cav_id, timestamp) -> Path`
  - a `main()` CLI, mirroring `coloca/pcd_cache.py`.

Caching is load-bearing: `docs/coloca_qua_baseline.md` records the loader being
100% of wall clock until the NVMe mirror fixed it. ROI features are cached rather
than final embeddings so the embedding head stays trainable.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_alignformer_cache.py
import numpy as np
import pytest

from embedding_aware_belt_fusion.alignformer.cache import (
    FrameRecord,
    cache_path,
    read_frame,
    write_frame,
)


def _record(count=3, channels=4, grid=4):
    return FrameRecord(
        boxes=np.random.randn(count, 7).astype(np.float32),
        scores=np.random.rand(count).astype(np.float32),
        gt_ids=["12", None, "45"][:count],
        roi=np.random.randn(count, channels, grid, grid).astype(np.float16),
    )


def test_frame_round_trips_through_disk(tmp_path):
    record = _record()
    path = tmp_path / "000069.npz"

    write_frame(path, record)
    loaded = read_frame(path)

    assert np.allclose(loaded.boxes, record.boxes)
    assert np.allclose(loaded.scores, record.scores)
    assert loaded.gt_ids == record.gt_ids
    assert np.allclose(loaded.roi, record.roi)


def test_none_gt_ids_survive_the_round_trip(tmp_path):
    # None means "matched no ground-truth object" and must not become the
    # string "None", which would silently create a bogus shared identity that
    # the matching loss would then train towards.
    record = _record()
    path = tmp_path / "000069.npz"

    write_frame(path, record)

    assert read_frame(path).gt_ids[1] is None


def test_empty_frame_round_trips(tmp_path):
    record = FrameRecord(
        boxes=np.zeros((0, 7), np.float32),
        scores=np.zeros((0,), np.float32),
        gt_ids=[],
        roi=np.zeros((0, 4, 4, 4), np.float16),
    )
    path = tmp_path / "000070.npz"

    write_frame(path, record)
    loaded = read_frame(path)

    assert loaded.boxes.shape == (0, 7)
    assert loaded.gt_ids == []


def test_cache_path_is_split_scenario_cav_timestamp(tmp_path):
    path = cache_path(tmp_path, "train", "2021_08_18", "1045", "000069")

    assert path == tmp_path / "train" / "2021_08_18" / "1045" / "000069.npz"


def test_frame_record_rejects_inconsistent_lengths():
    with pytest.raises(ValueError, match="length"):
        FrameRecord(
            boxes=np.zeros((3, 7), np.float32),
            scores=np.zeros((2,), np.float32),
            gt_ids=["a", "b", "c"],
            roi=np.zeros((3, 4, 4, 4), np.float16),
        )
```

- [ ] **Step 2: Run to verify they fail**

```bash
python -m pytest tests/test_alignformer_cache.py -v
```

Expected: FAIL, module not found.

- [ ] **Step 3: Implement**

```python
# src/embedding_aware_belt_fusion/alignformer/cache.py
"""Per-frame detection and ROI-feature cache.

Running the detector inside the training loop would make it the bottleneck, the
same failure the point-cloud cache fixed for CoLoca-QuA (0.62 -> 11.2 it/s).
Detections and their ROI features are computed once and stored on the NVMe.

ROI features are cached rather than finished embeddings so that the embedding
head remains trainable downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

# Sentinel for "this detection matched no ground-truth object". npz has no null
# for a string array, and writing "None" would fabricate a shared identity that
# the matching loss would then train towards.
_NO_GT_ID = ""


@dataclass(frozen=True)
class FrameRecord:
    """One agent's cached detections at one timestamp."""

    boxes: np.ndarray
    scores: np.ndarray
    gt_ids: Sequence[Optional[str]]
    roi: np.ndarray

    def __post_init__(self) -> None:
        count = self.boxes.shape[0]
        lengths = {
            "scores": self.scores.shape[0],
            "gt_ids": len(self.gt_ids),
            "roi": self.roi.shape[0],
        }
        mismatched = {name: n for name, n in lengths.items() if n != count}
        if mismatched:
            raise ValueError(f"inconsistent length against {count} boxes: {mismatched}")


def cache_path(
    cache_root: Union[Path, str],
    split: str,
    scenario: str,
    cav_id: str,
    timestamp: str,
) -> Path:
    """Return the npz path for one agent-frame."""
    return Path(cache_root) / split / scenario / cav_id / f"{timestamp}.npz"


def write_frame(path: Union[Path, str], record: FrameRecord) -> None:
    """Persist one frame, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gt_ids = np.array(
        [_NO_GT_ID if value is None else str(value) for value in record.gt_ids],
        dtype=np.str_,
    )
    np.savez(
        path,
        boxes=record.boxes.astype(np.float32),
        scores=record.scores.astype(np.float32),
        gt_ids=gt_ids,
        roi=record.roi.astype(np.float16),
    )


def read_frame(path: Union[Path, str]) -> FrameRecord:
    """Load one cached frame."""
    with np.load(path, allow_pickle=False) as data:
        return FrameRecord(
            boxes=data["boxes"],
            scores=data["scores"],
            gt_ids=[
                None if value == _NO_GT_ID else str(value) for value in data["gt_ids"]
            ],
            roi=data["roi"],
        )
```

- [ ] **Step 4: Run to verify they pass**

```bash
python -m pytest tests/test_alignformer_cache.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Add the cache-building CLI and run it**

Append a `main()` that walks a split with `coloca.index.scan_split`, runs
`boxes.detect_agent` plus `embedding.rotated_roi_align` per agent-frame, and writes
each record. Then build both splits:

```bash
python -m embedding_aware_belt_fusion.alignformer.cache \
  --config configs/alignformer_detector.yaml \
  --splits /media/chenyi/Elements1/Dataset/OPV2V/train \
           /media/chenyi/Elements1/Dataset/OPV2V/test \
  --cache-root /media/chenyi/basement2/cache/alignformer \
  --output-size 4
```

Expect roughly 5 GB and about half an hour. Verify:

```bash
du -sh /media/chenyi/basement2/cache/alignformer
find /media/chenyi/basement2/cache/alignformer -name '*.npz' | wc -l
```

The file count should match the 27,125 clouds in the point cache.

- [ ] **Step 6: Commit**

```bash
git add src/embedding_aware_belt_fusion/alignformer/cache.py \
        tests/test_alignformer_cache.py
git commit -m "feat: per-frame detection and ROI feature cache

Caches ROI features rather than finished embeddings so the embedding head stays
trainable, and keeps the detector out of the training loop."
```

---

### Task 11: `alignformer/dataset.py` - pairwise object-set dataset

**Files:**
- Create: `src/embedding_aware_belt_fusion/alignformer/dataset.py`
- Test: `tests/test_alignformer_dataset.py`

**Interfaces:**
- Consumes: `coloca/index.py:load_or_build_pairs`, `coloca/geometry.py:perturb_pose_2d`
  and `relative_pose_error`, `cache.read_frame`.
- Produces:
  - `NoiseSchedule(max_xy_std)` with `sigma_for_epoch(epoch, total_epochs) -> float`,
    ramping linearly from 0 to `max_xy_std`.
  - `correspondence_indices(ego_ids, cav_ids) -> tuple[Tensor, Tensor]`
  - `collate(samples) -> dict` padding to the batch maximum and emitting
    `ego_mask` / `cav_mask`.
  - `OPV2VObjectSetDataset(pairs, cache_root, split, *, noise_schedule, train)`.
  - `YAW_STD_PER_XY_STD = 1.0`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_alignformer_dataset.py
import pytest
import torch

from embedding_aware_belt_fusion.alignformer.dataset import (
    NoiseSchedule,
    collate,
    correspondence_indices,
)


def test_noise_ramps_from_zero_to_the_maximum():
    schedule = NoiseSchedule(max_xy_std=2.0)

    assert schedule.sigma_for_epoch(0, 10) == pytest.approx(0.0)
    assert schedule.sigma_for_epoch(9, 10) == pytest.approx(2.0)
    assert schedule.sigma_for_epoch(5, 10) < 2.0


def test_correspondences_pair_shared_object_ids():
    ego_ids = ["7", "9", None, "3"]
    cav_ids = ["3", "7", None]

    ego_match, cav_match = correspondence_indices(ego_ids, cav_ids)

    assert ego_match.tolist() == [1, -1, -1, 0]
    assert cav_match.tolist() == [3, 0, -1]


def test_none_ids_never_match_each_other():
    # Two unmatched detections both carry None; treating that as a shared
    # identity would train the model towards a false correspondence.
    ego_match, cav_match = correspondence_indices([None, None], [None])

    assert ego_match.tolist() == [-1, -1]
    assert cav_match.tolist() == [-1]


def test_duplicate_ids_take_the_first_occurrence_only():
    ego_match, cav_match = correspondence_indices(["5", "5"], ["5"])

    assert (ego_match >= 0).sum() == 1
    assert (cav_match >= 0).sum() == 1


def _sample(ego_count, cav_count, ego_match, cav_match):
    return {
        "ego_boxes": torch.randn(ego_count, 7),
        "ego_scores": torch.rand(ego_count),
        "ego_roi": torch.randn(ego_count, 4, 4, 4),
        "cav_boxes": torch.randn(cav_count, 7),
        "cav_scores": torch.rand(cav_count),
        "cav_roi": torch.randn(cav_count, 4, 4, 4),
        "ego_match": torch.tensor(ego_match),
        "cav_match": torch.tensor(cav_match),
        "psi_true": torch.tensor(0.1),
        "t_true": torch.tensor([1.0, 2.0]),
    }


def test_collate_pads_to_the_batch_maximum_and_masks():
    batch = collate([_sample(2, 3, [-1, 0], [1, -1, -1]),
                     _sample(5, 1, [-1] * 5, [-1])])

    assert batch["ego_boxes"].shape == (2, 5, 7)
    assert batch["cav_boxes"].shape == (2, 3, 7)
    assert batch["ego_mask"][0].tolist() == [True, True, False, False, False]
    assert batch["cav_mask"][1].tolist() == [True, False, False]
    assert batch["psi_true"].shape == (2,)


def test_padded_match_targets_are_negative_one():
    batch = collate([_sample(1, 1, [0], [0]), _sample(3, 2, [-1, 1, -1], [-1, 1])])

    assert batch["ego_match"][0, 1:].tolist() == [-1, -1]
```

- [ ] **Step 2: Run to verify they fail**

```bash
python -m pytest tests/test_alignformer_dataset.py -v
```

Expected: FAIL, module not found.

- [ ] **Step 3: Implement**

The two pieces that carry real logic:

```python
# src/embedding_aware_belt_fusion/alignformer/dataset.py (key excerpts)
"""Pairwise object-set dataset with a localization-noise curriculum.

Splits are by scenario, never by frame: consecutive OPV2V frames are
near-duplicates and a frame-level split leaks. Reuses coloca/index.py for the
pair index and coloca/geometry.py for the exact SE(2) label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

# Yaw noise in degrees equals xy noise in metres, the convention this
# repository's existing sweeps already use (0.2 m / 0.2 deg, ...).
YAW_STD_PER_XY_STD = 1.0


@dataclass(frozen=True)
class NoiseSchedule:
    """Linear ramp of localization noise across training epochs."""

    max_xy_std: float

    def sigma_for_epoch(self, epoch: int, total_epochs: int) -> float:
        """Return the xy noise std for ``epoch``, ramping 0 -> ``max_xy_std``."""
        if total_epochs < 1:
            raise ValueError(f"total_epochs must be >= 1, got {total_epochs}")
        if total_epochs == 1:
            return self.max_xy_std
        fraction = min(max(epoch, 0), total_epochs - 1) / (total_epochs - 1)
        return self.max_xy_std * fraction


def correspondence_indices(
    ego_ids: List[Optional[str]], cav_ids: List[Optional[str]]
) -> Tuple[Tensor, Tensor]:
    """Match detections across agents by OPV2V physical object id.

    ``None`` means a detection matched no ground-truth object, so it can never
    correspond to anything - two ``None`` detections are different objects, not
    the same one. Duplicate ids keep only the first occurrence, so the result is
    a genuine one-to-one assignment.
    """
    ego_match = torch.full((len(ego_ids),), -1, dtype=torch.long)
    cav_match = torch.full((len(cav_ids),), -1, dtype=torch.long)

    first_cav: Dict[str, int] = {}
    for index, identifier in enumerate(cav_ids):
        if identifier is not None and identifier not in first_cav:
            first_cav[identifier] = index

    claimed = set()
    for ego_index, identifier in enumerate(ego_ids):
        if identifier is None:
            continue
        cav_index = first_cav.get(identifier)
        if cav_index is None or cav_index in claimed:
            continue
        ego_match[ego_index] = cav_index
        cav_match[cav_index] = ego_index
        claimed.add(cav_index)

    return ego_match, cav_match
```

And the collate function, which every test above exercises:

```python
# Per-object fields are padded to the batch maximum; the rest are stacked.
_EGO_FIELDS = ("ego_boxes", "ego_scores", "ego_roi", "ego_match")
_CAV_FIELDS = ("cav_boxes", "cav_scores", "cav_roi", "cav_match")
# Match targets pad with -1 ("no counterpart"), everything else with 0.
_PAD_VALUES = {"ego_match": -1, "cav_match": -1}


def _pad(tensors: List[Tensor], size: int, value: float) -> Tensor:
    padded = []
    for tensor in tensors:
        deficit = size - tensor.shape[0]
        if deficit:
            shape = (deficit,) + tuple(tensor.shape[1:])
            tensor = torch.cat([tensor, tensor.new_full(shape, value)], dim=0)
        padded.append(tensor)
    return torch.stack(padded)


def collate(samples: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """Pad both object sets to the batch maximum and emit validity masks."""
    if not samples:
        raise ValueError("cannot collate an empty batch")

    ego_counts = [int(s["ego_boxes"].shape[0]) for s in samples]
    cav_counts = [int(s["cav_boxes"].shape[0]) for s in samples]
    ego_size, cav_size = max(ego_counts), max(cav_counts)

    batch: Dict[str, Tensor] = {}
    for field in _EGO_FIELDS:
        batch[field] = _pad([s[field] for s in samples], ego_size, _PAD_VALUES.get(field, 0))
    for field in _CAV_FIELDS:
        batch[field] = _pad([s[field] for s in samples], cav_size, _PAD_VALUES.get(field, 0))

    indices = torch.arange(ego_size)
    batch["ego_mask"] = indices.unsqueeze(0) < torch.tensor(ego_counts).unsqueeze(1)
    indices = torch.arange(cav_size)
    batch["cav_mask"] = indices.unsqueeze(0) < torch.tensor(cav_counts).unsqueeze(1)

    batch["psi_true"] = torch.stack([s["psi_true"] for s in samples])
    batch["t_true"] = torch.stack([s["t_true"] for s in samples])
    return batch
```

`OPV2VObjectSetDataset.__getitem__` must:
1. Read both agents' `FrameRecord`s from the cache.
2. Perturb the CAV pose with `coloca.geometry.perturb_pose_2d` at the current
   epoch's sigma (yaw std in degrees = xy std in metres), using the same per-sample
   RNG discipline as `coloca/dataset.py:sample_rng` - fresh noise each epoch when
   training, fixed otherwise.
3. Project the CAV boxes into the ego frame using the **noisy** pose.
4. Compute the label with `coloca.geometry.relative_pose_error`, converting its
   degrees to radians for `psi_true`.
5. Build correspondences with `correspondence_indices`.
6. Truncate each set to `trunk.MAX_OBJECTS` by descending score.

- [ ] **Step 4: Run to verify they pass**

```bash
python -m pytest tests/test_alignformer_dataset.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Add the label-exactness integration test**

```python
def test_applying_the_label_realigns_the_cav_boxes(tmp_path):
    """The SE(2) label must exactly undo the injected pose noise.

    coloca/geometry.py guarantees C @ T_noisy == T_true to 1e-9, so a box
    projected with the noisy pose and then corrected must land where the
    true-pose projection puts it.
    """
```

Implement it against a small synthetic cache built in `tmp_path`: write two
`FrameRecord`s with known boxes, construct the dataset over a single hand-made
`AgentPair`, and assert the corrected CAV centres match the true-pose projection to
within 1e-4 m.

**This is the single most important test in the plan.** If the label sign or frame
convention is wrong, every downstream number is silently meaningless while still
looking plausible.

- [ ] **Step 6: Commit**

```bash
git add src/embedding_aware_belt_fusion/alignformer/dataset.py \
        tests/test_alignformer_dataset.py
git commit -m "feat: pairwise object-set dataset with noise curriculum

Reuses coloca's exact SE(2) label machinery and scenario-disjoint splits.
Correspondences come from OPV2V physical object ids; unmatched detections
carry None and can never match each other."
```

---

### Task 12: `alignformer/fusion.py`, `evaluate.py`, and the P0 gate

**GATE: clean late-fusion AP@0.7 = 0.856 +/- 0.01.** If this does not reproduce,
stop and diagnose before building anything on top - every later number inherits it.

**Files:**
- Create: `src/embedding_aware_belt_fusion/alignformer/fusion.py`
- Create: `src/embedding_aware_belt_fusion/alignformer/evaluate.py`
- Test: `tests/test_alignformer_fusion.py`

**Interfaces:**
- Consumes: `boxes.AgentDetections`, `losses.apply_se2`, `losses.bev_corners`.
- Produces:
  - `correct_detections(detections, psi, t) -> AgentDetections` returning a **new**
    object with corrected boxes; never mutates its input.
  - `late_fuse(detections_by_agent, nms_threshold) -> tuple[Tensor, Tensor]` giving
    fused boxes and scores after rotated NMS.
  - `average_precision(predictions, ground_truth, iou_threshold, *, global_sort=True) -> float`
    where `predictions` is a list of `(boxes, scores)` per frame and `ground_truth`
    a list of `boxes` per frame.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_alignformer_fusion.py
import math

import pytest
import torch

from embedding_aware_belt_fusion.alignformer.boxes import AgentDetections
from embedding_aware_belt_fusion.alignformer.fusion import (
    average_precision,
    correct_detections,
)


def _detections(boxes):
    count = boxes.shape[0]
    return AgentDetections(
        boxes=boxes,
        scores=torch.ones(count),
        corners=torch.zeros(count, 8, 3),
        gt_ids=[None] * count,
        features=torch.zeros(1, 4, 4),
    )


def test_correction_translates_and_rotates_boxes():
    detections = _detections(torch.tensor([[1.0, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0]]))

    corrected = correct_detections(
        detections, torch.tensor(math.pi / 2), torch.tensor([0.0, 1.0])
    )

    assert corrected.boxes[0, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert corrected.boxes[0, 1].item() == pytest.approx(2.0, abs=1e-6)
    assert corrected.boxes[0, 6].item() == pytest.approx(math.pi / 2, abs=1e-6)


def test_correction_does_not_mutate_its_input():
    original = torch.tensor([[1.0, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0]])
    detections = _detections(original.clone())

    correct_detections(detections, torch.tensor(0.5), torch.tensor([3.0, 3.0]))

    assert torch.allclose(detections.boxes, original)


def test_identity_correction_is_a_no_op():
    detections = _detections(torch.randn(4, 7))

    corrected = correct_detections(detections, torch.tensor(0.0), torch.zeros(2))

    assert torch.allclose(corrected.boxes, detections.boxes, atol=1e-6)


def test_average_precision_is_one_for_perfect_predictions():
    boxes = torch.tensor([
        [0.0, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0],
        [20.0, 5.0, 0.0, 1.5, 2.0, 4.0, 0.3],
    ])

    result = average_precision(
        [(boxes, torch.ones(2))], [boxes], iou_threshold=0.7, global_sort=True
    )

    assert result == pytest.approx(1.0, abs=1e-6)


def test_average_precision_is_zero_when_nothing_overlaps():
    predicted = torch.tensor([[0.0, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0]])
    truth = torch.tensor([[80.0, 30.0, 0.0, 1.5, 2.0, 4.0, 0.0]])

    result = average_precision(
        [(predicted, torch.ones(1))], [truth], iou_threshold=0.7, global_sort=True
    )

    assert result == pytest.approx(0.0, abs=1e-6)
```

- [ ] **Step 2: Run to verify they fail**

```bash
python -m pytest tests/test_alignformer_fusion.py -v
```

Expected: FAIL, module not found.

- [ ] **Step 3: Implement `fusion.py` and `evaluate.py`**

`correct_detections` builds a new `AgentDetections` via `dataclasses.replace`, with
centres transformed by `losses.apply_se2` and yaw incremented by `psi`.

`late_fuse` concatenates all agents' corrected boxes and applies
`opencood.utils.box_utils.nms_rotated`.

`average_precision` is the one piece the P0 gate depends on, so it is given in full.
Note the global sort: the README states that frame-order accumulation is **not** a
valid dataset-level PR curve.

```python
def average_precision(
    predictions: List[Tuple[Tensor, Tensor]],
    ground_truth: List[Tensor],
    iou_threshold: float,
    *,
    global_sort: bool = True,
) -> float:
    """VOC-style AP over rotated BEV IoU.

    Parameters
    ----------
    predictions: one ``(boxes, scores)`` pair per frame.
    ground_truth: one ``boxes`` tensor per frame, same ordering.
    global_sort: sort every detection in the dataset by confidence before
        building the precision-recall curve. Required for a valid dataset-level
        comparison; per-frame accumulation silently inflates AP.
    """
    from opencood.utils import box_utils, common_utils

    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"{len(predictions)} predicted frames vs {len(ground_truth)} truth frames"
        )

    records: List[Tuple[float, int]] = []  # (score, is_true_positive)
    total_truth = 0

    for (boxes, scores), truth in zip(predictions, ground_truth):
        total_truth += int(truth.shape[0])
        if boxes.shape[0] == 0:
            continue
        if truth.shape[0] == 0:
            records.extend((float(s), 0) for s in scores)
            continue

        predicted_polygons = common_utils.convert_format(
            box_utils.boxes_to_corners_3d(boxes, order="hwl")[:, :4, :2]
            .detach().cpu().numpy()
        )
        truth_polygons = common_utils.convert_format(
            box_utils.boxes_to_corners_3d(truth, order="hwl")[:, :4, :2]
            .detach().cpu().numpy()
        )

        # Greedy highest-confidence-first matching, one truth box per detection.
        claimed = set()
        for index in torch.argsort(scores, descending=True).tolist():
            iou = common_utils.compute_iou(predicted_polygons[index], truth_polygons)
            best = int(iou.argmax())
            hit = iou[best] >= iou_threshold and best not in claimed
            if hit:
                claimed.add(best)
            records.append((float(scores[index]), int(hit)))

    if total_truth == 0 or not records:
        return 0.0

    if global_sort:
        records.sort(key=lambda row: row[0], reverse=True)

    hits = torch.tensor([row[1] for row in records], dtype=torch.float64)
    true_positives = torch.cumsum(hits, dim=0)
    ranks = torch.arange(1, len(records) + 1, dtype=torch.float64)
    precision = true_positives / ranks
    recall = true_positives / total_truth

    # VOC-style: integrate the monotonically decreasing precision envelope.
    precision = torch.flip(torch.cummax(torch.flip(precision, [0]), dim=0).values, [0])
    recall = torch.cat([torch.zeros(1, dtype=torch.float64), recall])
    return float(((recall[1:] - recall[:-1]) * precision).sum())
```

`evaluate.py` is a CLI dispatching on `--method`, writing a JSON result file.

- [ ] **Step 4: Run to verify they pass**

```bash
python -m pytest tests/test_alignformer_fusion.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Run the P0 gate**

```bash
python -m embedding_aware_belt_fusion.alignformer.evaluate \
  --config configs/alignformer_detector.yaml \
  --split /media/chenyi/Elements1/Dataset/OPV2V/test \
  --method late_fusion_clean \
  --output outputs/alignformer/p0_gate.json
```

Expected: AP@0.7 within 0.01 of **0.856**, AP@0.5 about 0.905, AP@0.3 about 0.910.

If it is far below, check in this order: (1) rotated NMS actually running rather
than silently returning everything, (2) global confidence sorting enabled, (3) the
detector checkpoint being a genuine late-fusion model, (4) `cav_lidar_range`
matching the checkpoint's training range.

- [ ] **Step 6: Commit**

```bash
git add src/embedding_aware_belt_fusion/alignformer/fusion.py \
        src/embedding_aware_belt_fusion/alignformer/evaluate.py \
        tests/test_alignformer_fusion.py outputs/alignformer/p0_gate.json
git commit -m "feat: SE(2) correction, late fusion, and global-sorted AP

Passes the P0 gate: clean late-fusion AP@0.7 reproduces the 0.856 reference."
```

---

### Task 13: `alignformer/metrics.py`, stage-1 training, and the P1 gate

**GATE: cross-agent Top-1 >= 0.85** on the held-out validation scenarios.

**Files:**
- Create: `src/embedding_aware_belt_fusion/alignformer/metrics.py`
- Create: `src/embedding_aware_belt_fusion/alignformer/train.py`
- Create: `configs/alignformer.yaml`
- Test: `tests/test_alignformer_metrics.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `top1_accuracy(log_assignment, ego_match, ego_mask) -> float` - the fraction of
    ego objects that have a true counterpart whose highest-scoring CAV column is the
    correct one.
  - `translation_mae(psi_pred, t_pred, psi_true, t_true) -> float` in metres,
    defined as the mean Euclidean norm of the residual, matching
    `docs/coloca_qua_baseline.md`.
  - `yaw_mae_deg(psi_pred, psi_true) -> float`, wrapping across +/-pi.
  - `train_stage1(config) -> Path` returning the best checkpoint path.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_alignformer_metrics.py
import math

import pytest
import torch

from embedding_aware_belt_fusion.alignformer.metrics import (
    top1_accuracy,
    translation_mae,
    yaw_mae_deg,
)


def test_top1_counts_only_objects_that_have_a_counterpart():
    log_assignment = torch.full((1, 3, 3), -10.0)
    log_assignment[0, 0, 0] = 0.0  # correct
    log_assignment[0, 1, 1] = 0.0  # this ego object has no true match
    ego_match = torch.tensor([[0, -1]])
    ego_mask = torch.ones(1, 2, dtype=torch.bool)

    assert top1_accuracy(log_assignment, ego_match, ego_mask) == pytest.approx(1.0)


def test_top1_is_zero_when_every_match_is_wrong():
    log_assignment = torch.full((1, 3, 3), -10.0)
    log_assignment[0, 0, 1] = 0.0
    ego_match = torch.tensor([[0, -1]])
    ego_mask = torch.ones(1, 2, dtype=torch.bool)

    assert top1_accuracy(log_assignment, ego_match, ego_mask) == pytest.approx(0.0)


def test_top1_ignores_padded_objects():
    log_assignment = torch.full((1, 3, 3), -10.0)
    log_assignment[0, 0, 0] = 0.0
    ego_match = torch.tensor([[0, 1]])
    ego_mask = torch.tensor([[True, False]])

    assert top1_accuracy(log_assignment, ego_match, ego_mask) == pytest.approx(1.0)


def test_translation_mae_is_the_mean_euclidean_residual():
    result = translation_mae(
        torch.zeros(2), torch.tensor([[3.0, 4.0], [0.0, 0.0]]),
        torch.zeros(2), torch.zeros(2, 2),
    )

    assert result == pytest.approx(2.5)


def test_yaw_mae_wraps_across_pi():
    result = yaw_mae_deg(torch.tensor([math.pi - 0.01]), torch.tensor([-math.pi + 0.01]))

    assert result == pytest.approx(math.degrees(0.02), abs=1e-4)
```

- [ ] **Step 2: Run to verify they fail**

```bash
python -m pytest tests/test_alignformer_metrics.py -v
```

Expected: FAIL, module not found.

- [ ] **Step 3: Implement `metrics.py`, then run to verify they pass**

```bash
python -m pytest tests/test_alignformer_metrics.py -v
```

Expected: 5 passed.

- [ ] **Step 4: Write `configs/alignformer.yaml`**

Mirror `configs/coloca_qua_opv2v.yaml`'s structure and commenting style. Carry the
spec's §3.5 defaults verbatim:

```yaml
model:
  output_size: 4          # rotated ROI-align grid, k x k
  embed_dim: 128          # ablated over 32 / 64 / 128
  model_dim: 256
  layers: 4
  heads: 4
  heading_lambda: 2.0     # metres, about half a vehicle length
  sinkhorn_iterations: 20
  max_objects: 64

train:
  stage1_epochs: 20
  stage1_max_xy_std: 0.5  # matching is learned at low noise first
  stage2_epochs: 30
  max_xy_std: 2.0         # stage-2 curriculum ramps 0 -> this
  batch_size: 32
  lr: 1.0e-3
  match_weight: 1.0
  seed: 0

data:
  cache_root: /media/chenyi/basement2/cache/alignformer
  val_scenario_fraction: 0.15
  split_seed: 0
  comm_range_m: 40.0
```

- [ ] **Step 5: Implement and run stage 1**

Train the embedding head and trunk on `match_nll` alone, with sigma sampled in
`[0, 0.5]` m.

```bash
python -m embedding_aware_belt_fusion.alignformer.train \
  --config configs/alignformer.yaml --stage 1 \
  2>&1 | tee outputs/alignformer/stage1.log
```

- [ ] **Step 6: Check the P1 gate**

```bash
python -m embedding_aware_belt_fusion.alignformer.evaluate \
  --config configs/alignformer.yaml \
  --checkpoint outputs/alignformer/stage1/best.pth \
  --metric top1 --output outputs/alignformer/p1_gate.json
```

Expected: Top-1 >= 0.85. The prior Spatial TrackFormer reached 0.891 with a far
heavier model, so this is a reasonable bar.

If it stalls near chance, check that `correspondence_indices` is not matching `None`
to `None` (Task 11 has a test for exactly this), and that padded objects are masked
out of the score matrix.

- [ ] **Step 7: Commit**

```bash
git add src/embedding_aware_belt_fusion/alignformer/train.py \
        src/embedding_aware_belt_fusion/alignformer/metrics.py \
        configs/alignformer.yaml tests/test_alignformer_metrics.py \
        outputs/alignformer/p1_gate.json
git commit -m "feat: stage-1 matching training and association metrics

Passes the P1 gate: cross-agent Top-1 above 0.85 on held-out scenarios."
```

---

### Task 14: Stage-2 pose training, the boxes-only ablation, and the P2 gate

**GATE: yaw MAE strictly below the predict-zero value at every sigma.** Predicting
zero gives `E|dpsi| = sigma_yaw * sqrt(2/pi)`. This is precisely what CoLoca-QuA
failed; if Head B cannot clear it, the closed-form premise is wrong.

**The boxes-only ablation runs here, not at the end.** It answers the spec's main
scientific risk: if geometry alone matches the full model, the embedding
contributes nothing and the premise needs rethinking. Knowing that now is worth far
more than knowing it after P3-P5 are built on top.

**Files:**
- Modify: `src/embedding_aware_belt_fusion/alignformer/train.py` (add stage 2)
- Modify: `src/embedding_aware_belt_fusion/alignformer/evaluate.py` (pose sweep)
- Create: `docs/alignformer_p2.md`
- Test: `tests/test_alignformer_train.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `zero_embeddings(batch) -> dict` returning a **new** dict with embeddings
    replaced by zeros, leaving geometry untouched.
  - `train_stage2(config, head, message_content, match_weight) -> Path`, where
    `head` is `"A"` or `"B"` and `message_content` is `"boxes+embeddings"` or
    `"boxes_only"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_alignformer_train.py
import torch

from embedding_aware_belt_fusion.alignformer.model import AlignFormerB
from embedding_aware_belt_fusion.alignformer.train import zero_embeddings


def test_boxes_only_mode_removes_all_embedding_information():
    batch = {
        "ego_embeddings": torch.randn(2, 4, 16),
        "cav_embeddings": torch.randn(2, 5, 16),
        "ego_boxes": torch.randn(2, 4, 7),
    }

    ablated = zero_embeddings(batch)

    assert torch.count_nonzero(ablated["ego_embeddings"]) == 0
    assert torch.count_nonzero(ablated["cav_embeddings"]) == 0
    # Geometry must survive, or the ablation tests the wrong thing.
    assert torch.allclose(ablated["ego_boxes"], batch["ego_boxes"])


def test_boxes_only_mode_does_not_mutate_the_original_batch():
    batch = {
        "ego_embeddings": torch.randn(1, 2, 8),
        "cav_embeddings": torch.randn(1, 2, 8),
    }
    before = batch["ego_embeddings"].clone()

    zero_embeddings(batch)

    assert torch.allclose(batch["ego_embeddings"], before)


def test_a_model_step_reduces_the_loss_on_a_single_repeated_batch():
    # Overfitting one batch is the cheapest possible check that gradients flow
    # end to end through Sinkhorn and the closed-form solver.
    torch.manual_seed(0)
    model = AlignFormerB(embed_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = {
        "ego_boxes": torch.randn(4, 6, 7), "ego_scores": torch.rand(4, 6),
        "ego_embeddings": torch.randn(4, 6, 8),
        "ego_mask": torch.ones(4, 6, dtype=torch.bool),
        "cav_boxes": torch.randn(4, 6, 7), "cav_scores": torch.rand(4, 6),
        "cav_embeddings": torch.randn(4, 6, 8),
        "cav_mask": torch.ones(4, 6, dtype=torch.bool),
    }
    target_psi = torch.full((4,), 0.2)
    target_t = torch.full((4, 2), 1.0)

    def step():
        optimizer.zero_grad()
        estimate = model(batch)
        loss = (
            (estimate.psi - target_psi).abs().mean()
            + (estimate.t - target_t).abs().mean()
        )
        loss.backward()
        optimizer.step()
        return loss.item()

    first = step()
    for _ in range(50):
        last = step()

    assert last < first
```

- [ ] **Step 2: Run to verify they fail**

```bash
python -m pytest tests/test_alignformer_train.py -v
```

Expected: FAIL, `zero_embeddings` not defined.

- [ ] **Step 3: Implement stage 2**

`zero_embeddings` returns a new dict (never mutating its input). `train_stage2`
freezes the embedding head, trains the selected pose head on
`corner_loss + match_weight * match_nll`, and ramps sigma via
`NoiseSchedule.sigma_for_epoch`. For Head A, `match_weight` is 0 since it exposes no
assignment.

- [ ] **Step 4: Run to verify they pass**

```bash
python -m pytest tests/test_alignformer_train.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Run the five training configurations**

```bash
for head in B A; do
  for content in boxes+embeddings boxes_only; do
    python -m embedding_aware_belt_fusion.alignformer.train \
      --config configs/alignformer.yaml --stage 2 \
      --head "$head" --message-content "$content" \
      2>&1 | tee "outputs/alignformer/stage2_${head}_${content}.log"
  done
done

# Fairness ablation: Head B's architecture, without its extra supervision.
python -m embedding_aware_belt_fusion.alignformer.train \
  --config configs/alignformer.yaml --stage 2 \
  --head B --message-content boxes+embeddings --match-weight 0 \
  2>&1 | tee outputs/alignformer/stage2_B_nomatch.log
```

- [ ] **Step 6: Evaluate the P2 gate and write up the ablation**

```bash
python -m embedding_aware_belt_fusion.alignformer.evaluate \
  --config configs/alignformer.yaml \
  --checkpoint outputs/alignformer/stage2_B_boxes+embeddings/best.pth \
  --metric pose --sweep 0 0.2 0.4 0.6 0.8 1.0 1.5 2.0 --seeds 3 \
  --output outputs/alignformer/p2_gate.json
```

Record in `docs/alignformer_p2.md`, following the evidence style of
`docs/coloca_qua_baseline.md`:

| sigma | Method | Translation MAE (m) | Yaw MAE (deg) | Predict-zero yaw (deg) |
|---|---|---:|---:|---:|

with one row per (head, message content) configuration at each sigma, mean over 3
seeds.

The two decisions this table drives:

1. **Yaw gate.** Head B's yaw MAE below `sigma_yaw * sqrt(2/pi)` at every sigma -
   proceed. Flat at the predict-zero value - the closed-form premise has failed;
   stop and diagnose before P3.
2. **Embedding value.** Compare `boxes+embeddings` against `boxes_only`. If the gap
   is negligible, geometry suffices: record it as a finding and take the spec's
   camera-augmentation contingency (§8) as the next step rather than continuing to
   P3 as written.

- [ ] **Step 7: Commit**

```bash
git add src/embedding_aware_belt_fusion/alignformer/train.py \
        src/embedding_aware_belt_fusion/alignformer/evaluate.py \
        tests/test_alignformer_train.py docs/alignformer_p2.md \
        outputs/alignformer/p2_gate.json
git commit -m "feat: stage-2 pose training with noise curriculum and boxes-only ablation

Reports the P2 gate: Head B yaw MAE against the predict-zero baseline, and
whether the embedding adds anything over box geometry alone."
```

---

## After P2

Report both gate outcomes before starting P3-P5, which are planned separately once
the results are known:

- **Yaw gate passed, embedding helps** - proceed to P3-P5 as the spec describes.
- **Yaw gate passed, embedding does not help** - the contribution shifts to the
  closed-form solver plus the efficiency argument; take the camera-augmentation
  contingency (spec §8) before P3.
- **Yaw gate failed** - stop. Diagnose with the same discipline as the CoLoca
  ablation: hold the trunk fixed and vary one factor at a time (heading virtual
  points on/off, oracle correspondences substituted for Sinkhorn, corner loss vs
  component MSE). The oracle-correspondence run separates a matching failure from a
  solver failure and should be the first probe.
