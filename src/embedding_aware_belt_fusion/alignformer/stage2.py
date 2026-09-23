"""Stage-2 trainer: a pose head on the full localization-noise curriculum.

Stage 1 asked only whether the two object sets can be put into correspondence.
Stage 2 asks the question the method exists for: given that correspondence, can
the sender's SE(2) localization error be recovered well enough to be worth
correcting before fusion? The P2 gate is head B's yaw MAE strictly below the
predict-zero value ``sigma_yaw * sqrt(2/pi)`` at every non-zero sigma --
precisely what CoLoca-QuA's regressed yaw never achieved on this data
(``docs/coloca_qua_baseline.md``). ``evaluate.py::_p2_gate`` is where the
verdict is computed; ``sigma = 0`` is excluded there because the baseline is
exactly 0 and nothing can be strictly below it.

Three things here are not cosmetic:

- **Gradients are checked for finiteness every step, not just the loss.**
  Stage 1 drove head B's match temperature to ~0.038, below the ~0.043 at which
  design spec 3.4 places the ``atan2(0, 0)`` singularity in
  ``AlignFormerB._soft_correspondence``. ``match_nll`` never backpropagated
  through that branch; ``corner_loss`` does, and stage 2 starts from exactly
  that checkpoint. An earlier bug in this project produced finite losses while
  poisoning every parameter with NaN gradients, so a finite loss is not
  evidence of a healthy step.
- **The fraction of samples that actually FELL BACK is logged, and the
  structurally unalignable pairs are scored separately.** A pair that shares no
  object at all has no recoverable pose, so the only safe answer on it is the
  identity correction; ``procrustes.MIN_MATCH_MASS`` is the guard meant to
  produce that. Measured at the DETECTION level on this cache, such pairs are
  4.55% of the stage-2 training pairs and 0 of the 4176 scenario-disjoint
  validation pairs -- not the 18.5% that ``metrics.py``'s docstring asserted,
  which does not reproduce. The fallback is measured as the *observed* identity
  output (:func:`is_fallback`), never by comparing ``confidence`` against
  ``MIN_MATCH_MASS``: those two thresholds differ by a factor of two, for the
  reason :func:`is_fallback` documents.
- **The embedding head is frozen and in eval mode.** Stage 2 trains the pose
  head and trunk only, so a comparison against stage 1's association result is
  about the pose, not about a further-tuned descriptor.

The boxes-only ablation is a *message content*, not a second model: it runs the
identical forward pass with :func:`~embedding_aware_belt_fusion.alignformer.train.zero_embeddings`
applied, so nothing but the embedding can differ between the two conditions.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader

from embedding_aware_belt_fusion.alignformer.dataset import (
    NoiseSchedule,
    OPV2VObjectSetDataset,
)
from embedding_aware_belt_fusion.alignformer.embedding import ObjectEmbedding
from embedding_aware_belt_fusion.alignformer.losses import (
    corner_displacements,
    corner_loss,
    match_nll,
)
from embedding_aware_belt_fusion.alignformer.model import AlignFormerA, AlignFormerB
from embedding_aware_belt_fusion.alignformer.train import (
    DEFAULT_NUM_WORKERS,
    GRAD_CLIP,
    _checkpoint,
    _loaders,
    _schedule,
    _split_name,
    _to_device,
    build_eval_dataset,
    build_pair_split,
    embed_batch,
    roi_channels,
)
from embedding_aware_belt_fusion.alignformer.trunk import MAX_OBJECTS

# Stage 2 writes one directory per (head, message content) configuration under here.
STAGE2_ROOT = Path("outputs/alignformer")
# The stage-1 checkpoint each message content warm-starts from. boxes_only gets
# the boxes-only stage-1 run, not the full one: warm-starting the ablation from
# a trunk that was trained WITH embeddings would leak the embedding into the
# condition that is supposed to be without it.
STAGE1_CHECKPOINTS = {
    "boxes+embeddings": Path("outputs/alignformer/stage1/best.pth"),
    "boxes_only": Path("outputs/alignformer/stage1_zero_embeddings/best.pth"),
}
HEADS = ("A", "B")
MESSAGE_CONTENTS = tuple(STAGE1_CHECKPOINTS)
_PROGRESS_INTERVAL = 100


def build_stage2_modules(
    config: Mapping[str, Any],
    channels: int,
    device: torch.device,
    *,
    head: str,
    stage1_checkpoint: Optional[Path] = None,
) -> nn.ModuleDict:
    """The stage-2 pair: a frozen embedding head plus the selected pose head.

    The pose head is warm-started from ``stage1_checkpoint`` wherever the
    parameter names line up: head B is the same module stage 1 trained, so it
    loads whole; head A shares only the trunk, so it inherits the trunk and
    starts its regression MLP fresh. Warm-starting is what makes an A-vs-B
    comparison a comparison of *heads* rather than of how much trunk training
    each one happened to get.
    """
    if head not in HEADS:
        raise ValueError(f"head must be one of {HEADS}, got {head!r}")

    model_cfg = config["model"]
    if int(model_cfg["max_objects"]) != MAX_OBJECTS:
        raise ValueError(
            f"config model.max_objects={model_cfg['max_objects']} disagrees with "
            f"trunk.MAX_OBJECTS={MAX_OBJECTS}, which is what the dataset truncates to"
        )
    embed_dim = int(model_cfg["embed_dim"])
    trunk_kwargs = {
        "model_dim": int(model_cfg["model_dim"]),
        "layers": int(model_cfg["layers"]),
        "heads": int(model_cfg["heads"]),
    }
    if head == "B":
        pose: nn.Module = AlignFormerB(
            embed_dim=embed_dim,
            heading_lambda=float(model_cfg["heading_lambda"]),
            sinkhorn_iterations=int(model_cfg["sinkhorn_iterations"]),
            **trunk_kwargs,
        )
    else:
        pose = AlignFormerA(embed_dim=embed_dim, **trunk_kwargs)

    modules = nn.ModuleDict(
        {
            "embedding": ObjectEmbedding(
                in_channels=channels,
                output_size=int(model_cfg["output_size"]),
                dim=embed_dim,
            ),
            "pose": pose,
        }
    )

    if stage1_checkpoint is not None:
        loaded = _warm_start_from_stage1(modules, Path(stage1_checkpoint))
        print(f"stage 2: warm-started {loaded} tensors from {stage1_checkpoint}", flush=True)

    # Frozen, and kept in eval mode for the whole run (see _train_stage2_epoch).
    modules["embedding"].requires_grad_(False)
    return modules.to(device)


def _warm_start_from_stage1(modules: nn.ModuleDict, checkpoint_path: Path) -> int:
    """Copy stage-1 weights into the stage-2 modules by name; return the count.

    The embedding head loads whole (it is the same module). The pose head takes
    whatever of ``matcher.*`` it has a same-shaped parameter for -- everything
    for head B, the trunk only for head A.
    """
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    source = checkpoint["state_dict"]
    target = modules.state_dict()

    renamed: Dict[str, Tensor] = {}
    for name, tensor in source.items():
        if name.startswith("embedding."):
            renamed[name] = tensor
        elif name.startswith("matcher."):
            renamed["pose." + name[len("matcher.") :]] = tensor

    usable = {
        name: tensor
        for name, tensor in renamed.items()
        if name in target and target[name].shape == tensor.shape
    }
    if not any(name.startswith("embedding.") for name in usable):
        raise RuntimeError(
            f"{checkpoint_path} contributed no embedding-head weights; stage 2 must "
            "start from a stage-1 checkpoint, not from scratch"
        )
    modules.load_state_dict(usable, strict=False)
    return len(usable)


def is_fallback(estimate) -> Tensor:
    """``(B,)`` bool: which samples actually received the identity correction.

    Measured from the emitted ``(psi, t)``, not by comparing
    ``PoseEstimate.confidence`` against ``procrustes.MIN_MATCH_MASS`` -- those
    two do **not** describe the same threshold, and inferring the fallback from
    the constant gives an answer that is wrong for a band of real pairs.

    ``AlignFormerB`` hands ``weighted_se2_kabsch`` the *heading-augmented*
    weight vector ``cat([mass, mass])``, whose total is twice
    ``confidence = mass.sum()``. The solver gates on that augmented total, so
    the effective floor on ``confidence`` is ``MIN_MATCH_MASS / 2`` -- half an
    effective matched object, not the one its docstring claims. Pairs with
    confidence in ``[0.5, 1.0)`` therefore look suppressed by the constant and
    are nevertheless corrected. Verified in
    ``tests/test_alignformer_procrustes.py``. Reading the output settles it
    regardless of which threshold is in force.
    """
    return (estimate.psi == 0) & (estimate.t == 0).all(dim=-1)


class _PoseGroup:
    """Per-sample sums for one subset of the split.

    Sums rather than running means: the last batch is usually short, and
    averaging per-batch means would weight it as heavily as a full one.
    """

    def __init__(self) -> None:
        self.translation = 0.0
        self.yaw = 0.0
        self.zero_translation = 0.0
        self.zero_yaw = 0.0
        self.corner = 0.0
        self.zero_corner = 0.0
        self.objects = 0.0
        self.confidence = 0.0
        self.fallback = 0.0
        self.samples = 0

    def add(self, terms: Mapping[str, Tensor], selection: Tensor) -> None:
        picked = selection.to(terms["translation"].dtype)
        for name in ("translation", "yaw", "zero_translation", "zero_yaw",
                     "corner", "zero_corner", "objects", "confidence", "fallback"):
            setattr(self, name, getattr(self, name) + float((terms[name] * picked).sum().item()))
        self.samples += int(selection.sum().item())

    def compute(self) -> Dict[str, float]:
        def per_sample(total: float) -> float:
            return total / self.samples if self.samples else math.nan

        def per_object(total: float) -> float:
            return total / self.objects if self.objects else math.nan

        return {
            "translation_mae_m": per_sample(self.translation),
            "yaw_mae_deg": per_sample(self.yaw),
            "predict_zero_translation_mae_m": per_sample(self.zero_translation),
            "predict_zero_yaw_mae_deg": per_sample(self.zero_yaw),
            "corner_loss_m": per_object(self.corner),
            "predict_zero_corner_loss_m": per_object(self.zero_corner),
            "mean_confidence": per_sample(self.confidence),
            "fallback_fraction": per_sample(self.fallback),
            "samples": float(self.samples),
        }


def _yaw_error_deg(psi_pred: Tensor, psi_true: Tensor) -> Tensor:
    """Per-sample absolute yaw residual in degrees, wrapped onto ``(-pi, pi]``."""
    residual = psi_pred - psi_true
    return torch.rad2deg(torch.atan2(torch.sin(residual), torch.cos(residual)).abs())


class _PoseTally:
    """Accumulates pose error over a split, with two subsets reported separately.

    Every quantity is reported next to its **predict-zero** counterpart,
    measured on exactly the same samples: predicting the identity correction is
    what CoLoca-QuA's head never improved on (``docs/coloca_qua_baseline.md``),
    so a pose number without it beside it says nothing.

    The split into ``shared`` and ``unalignable`` is the second thing that has
    to be visible. A pair that shares no object has no recoverable pose, so the
    *only* safe answer on it is the identity correction -- and an aggregate
    number hides whether that is what it gets: a model that is excellent on the
    alignable majority and actively harmful on the rest can post a good mean
    while making fused mAP worse than doing nothing.
    """

    def __init__(self) -> None:
        self.all = _PoseGroup()
        self.shared = _PoseGroup()
        self.unalignable = _PoseGroup()

    def update(self, batch: Mapping[str, Tensor], estimate) -> None:
        psi_true, t_true = batch["psi_true"], batch["t_true"]
        zero_psi, zero_t = torch.zeros_like(psi_true), torch.zeros_like(t_true)
        mask = batch["cav_mask"].to(psi_true.dtype)

        def corner_sum(psi: Tensor, t: Tensor) -> Tensor:
            per_object = corner_displacements(batch["cav_boxes"], psi, t, psi_true, t_true)
            return (per_object * mask).sum(dim=1)

        terms = {
            "translation": torch.linalg.norm(estimate.t - t_true, dim=-1),
            "zero_translation": torch.linalg.norm(t_true, dim=-1),
            "yaw": _yaw_error_deg(estimate.psi, psi_true),
            "zero_yaw": _yaw_error_deg(zero_psi, psi_true),
            "corner": corner_sum(estimate.psi, estimate.t),
            "zero_corner": corner_sum(zero_psi, zero_t),
            "objects": mask.sum(dim=1),
            "confidence": estimate.confidence.to(psi_true.dtype),
            "fallback": is_fallback(estimate).to(psi_true.dtype),
        }

        # "Alignable" means the pair shares at least one physical object, which
        # is the ground truth for whether any SE(2) is recoverable at all.
        alignable = (batch["ego_match"] >= 0).any(dim=1)
        everything = torch.ones_like(alignable)
        self.all.add(terms, everything)
        self.shared.add(terms, alignable)
        self.unalignable.add(terms, ~alignable)

    def compute(self) -> Dict[str, float]:
        metrics = dict(self.all.compute())
        for name, group in (("shared", self.shared), ("unalignable", self.unalignable)):
            for key, value in group.compute().items():
                metrics[f"{name}_{key}"] = value
        return metrics


@torch.no_grad()
def evaluate_pose(
    modules: nn.ModuleDict,
    loader: DataLoader,
    device: torch.device,
    *,
    ablate_embeddings: bool = False,
) -> Dict[str, float]:
    """Measure pose error, and its predict-zero baseline, over a whole loader."""
    modules.eval()
    tally = _PoseTally()
    for batch in loader:
        batch = _to_device(batch, device)
        enriched = embed_batch(modules["embedding"], batch, ablate=ablate_embeddings)
        tally.update(batch, modules["pose"](enriched))
    return tally.compute()


def _format_pose(metrics: Mapping[str, float]) -> str:
    return (
        f"t_mae={metrics['translation_mae_m']:.4f} m "
        f"(zero {metrics['predict_zero_translation_mae_m']:.4f}) "
        f"yaw_mae={metrics['yaw_mae_deg']:.4f} deg "
        f"(zero {metrics['predict_zero_yaw_mae_deg']:.4f}) "
        f"corner={metrics['corner_loss_m']:.4f} m "
        f"(zero {metrics['predict_zero_corner_loss_m']:.4f}) "
        f"| unalignable {int(metrics['unalignable_samples'])}: "
        f"corner={metrics['unalignable_corner_loss_m']:.4f} "
        f"(zero {metrics['unalignable_predict_zero_corner_loss_m']:.4f}) "
        f"fallback={metrics['unalignable_fallback_fraction']:.3f} "
        f"n={int(metrics['samples'])}"
    )


def _assert_finite_gradients(
    modules: nn.ModuleDict, parameters: List[Tensor], epoch: int, step: int
) -> float:
    """Clip, and raise if ANY gradient was non-finite; returns the pre-clip norm.

    ``clip_grad_norm_`` returns the total L2 norm over every gradient, and a
    NaN or inf in any one of them makes that total non-finite -- so one
    device synchronization checks every parameter, which is why the check can
    afford to run on every step rather than being sampled. The raise happens
    BEFORE ``optimizer.step``, so the parameters themselves are never touched
    by a poisoned gradient, and the message names the offending tensors.
    """
    total = torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP)
    if torch.isfinite(total):
        return float(total.item())

    offenders = [
        name
        for name, parameter in modules.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    temperature = getattr(modules["pose"], "log_temperature", None)
    reading = (
        "" if temperature is None
        else f"; match temperature {float(temperature.exp().item()):.5f}"
    )
    raise RuntimeError(
        f"non-finite GRADIENT at epoch {epoch + 1} step {step} (total norm {total}); "
        f"offending parameters: {offenders[:12]}{reading}. Diagnose before "
        "clamping: the atan2 guards in model.py::_soft_correspondence and "
        "procrustes.py::weighted_se2_kabsch are the two places a degenerate "
        "correspondence can reach an undefined gradient."
    )


def _train_stage2_epoch(
    modules: nn.ModuleDict,
    loader: DataLoader,
    optimizer,
    scheduler,
    parameters: List[Tensor],
    device: torch.device,
    *,
    match_weight: float,
    epoch: int,
    epochs: int,
    sigma: float,
    ablate_embeddings: bool,
) -> Dict[str, float]:
    """One pass over the training split; returns the mean losses and diagnostics."""
    modules.train()
    # The embedding head is frozen: keep it in eval mode so nothing about it
    # (its LayerNorm statistics today, anything stateful added later) can drift.
    modules["embedding"].eval()

    running, running_corner, running_match = 0.0, 0.0, 0.0
    fallbacks, samples, steps = 0, 0, 0
    grad_norm = 0.0
    started = time.time()

    for step, batch in enumerate(loader):
        batch = _to_device(batch, device)
        enriched = embed_batch(modules["embedding"], batch, ablate=ablate_embeddings)
        estimate = modules["pose"](enriched)

        pose_loss = corner_loss(
            batch["cav_boxes"], estimate.psi, estimate.t,
            batch["psi_true"], batch["t_true"], batch["cav_mask"],
        )
        match_loss = pose_loss.new_zeros(())
        if match_weight and estimate.log_assignment is not None:
            match_loss = match_nll(
                estimate.log_assignment, batch["ego_match"], batch["cav_match"]
            )
        loss = pose_loss + match_weight * match_loss
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite stage-2 loss at epoch {epoch + 1} step {step} "
                f"(corner {float(pose_loss)}, match {float(match_loss)})"
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = _assert_finite_gradients(modules, parameters, epoch, step)
        optimizer.step()
        scheduler.step()

        running += float(loss.item())
        running_corner += float(pose_loss.item())
        running_match += float(match_loss.item())
        fallbacks += int(is_fallback(estimate).sum().item())
        samples += int(batch["psi_true"].shape[0])
        steps += 1
        if step % _PROGRESS_INTERVAL == 0:
            rate = (step + 1) / (time.time() - started)
            print(
                f"  [ep{epoch + 1}/{epochs} sigma={sigma:.3f}] step {step}/{len(loader)} "
                f"loss={running / max(steps, 1):.4f} corner={running_corner / max(steps, 1):.4f} "
                f"|g|={grad_norm:.3f} {rate:.1f} it/s",
                flush=True,
            )

    divisor = max(steps, 1)
    return {
        "train_loss": running / divisor,
        "train_corner_loss_m": running_corner / divisor,
        "train_match_nll": running_match / divisor,
        "train_fallback_fraction": fallbacks / samples if samples else math.nan,
        "last_grad_norm": grad_norm,
    }


def build_stage2_datasets(
    config: Mapping[str, Any], *, epochs: int
) -> Tuple[OPV2VObjectSetDataset, OPV2VObjectSetDataset]:
    """Train set ramping 0 -> ``train.max_xy_std``; validation pinned at the maximum."""
    data, train_cfg = config["data"], config["train"]
    train_pairs, val_pairs, _, _ = build_pair_split(config)
    sigma = float(train_cfg["max_xy_std"])
    train_set = OPV2VObjectSetDataset(
        train_pairs,
        cache_root=data["cache_root"],
        split=_split_name(Path(data["train_root"])),
        noise_schedule=NoiseSchedule(max_xy_std=sigma),
        train=True,
        total_epochs=epochs,
        seed=int(train_cfg["seed"]),
    )
    return train_set, build_eval_dataset(config, val_pairs, sigma)


def stage2_output_dir(head: str, message_content: str, match_weight: float) -> Path:
    """The conventional directory name for one stage-2 configuration.

    The ``_nomatch`` suffix marks head B's *fairness control* -- B's
    architecture deprived of B's extra supervision -- so it applies only to
    head B. Head A exposes no assignment at all, so ``match_weight`` is 0 for it
    by necessity rather than by choice; suffixing head A would name every head-A
    run after a control it cannot be the counterpart of.
    """
    suffix = "_nomatch" if head == "B" and not match_weight else ""
    return STAGE2_ROOT / f"stage2_{head}_{message_content}{suffix}"


def train_stage2(
    config: Mapping[str, Any],
    head: str = "B",
    message_content: str = "boxes+embeddings",
    match_weight: Optional[float] = None,
    *,
    output_dir: Optional[Path] = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
    device: Optional[torch.device] = None,
    stage1_checkpoint: Optional[Path] = None,
    epochs: Optional[int] = None,
) -> Path:
    """Train a pose head on ``corner_loss + match_weight * match_nll``.

    Parameters
    ----------
    head: ``"A"`` (direct regression) or ``"B"`` (soft correspondence plus the
        closed-form SE(2) solve).
    message_content: ``"boxes+embeddings"`` or ``"boxes_only"``; the latter is
        the ablation, run through :func:`zero_embeddings`.
    match_weight: weight on the auxiliary matching NLL. ``None`` means the
        config's value for head B and 0 for head A, which exposes no assignment
        to supervise. Passing 0 explicitly for head B is the fairness control:
        B's architecture without B's extra supervision.

    Returns
    -------
    Path
        The best checkpoint, selected by **validation corner loss** -- the
        training objective measured on held-out scenarios. Deliberately not by
        yaw MAE: yaw MAE is the P2 gate, and selecting on the gate would make
        the gate a statement about checkpoint selection rather than about the
        method.
    """
    if head not in HEADS:
        raise ValueError(f"head must be one of {HEADS}, got {head!r}")
    if message_content not in MESSAGE_CONTENTS:
        raise ValueError(
            f"message_content must be one of {MESSAGE_CONTENTS}, got {message_content!r}"
        )

    train_cfg = config["train"]
    if match_weight is None:
        match_weight = float(train_cfg["match_weight"]) if head == "B" else 0.0
    match_weight = float(match_weight)
    if head == "A" and match_weight:
        raise ValueError(
            "head A exposes no assignment, so match_weight must be 0; got "
            f"{match_weight}"
        )

    epochs = int(epochs if epochs is not None else train_cfg["stage2_epochs"])
    ablate = message_content == "boxes_only"
    stage1_checkpoint = Path(
        stage1_checkpoint
        if stage1_checkpoint is not None
        else STAGE1_CHECKPOINTS[message_content]
    )
    output_dir = Path(
        output_dir
        if output_dir is not None
        else stage2_output_dir(head, message_content, match_weight)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(int(train_cfg["seed"]))
    np.random.seed(int(train_cfg["seed"]))

    train_set, val_set = build_stage2_datasets(config, epochs=epochs)
    channels = roi_channels(train_set)
    print(
        f"stage 2 [head {head} / {message_content} / match_weight {match_weight:g}]: "
        f"{len(train_set)} train pairs / {len(val_set)} val pairs, {channels}-channel ROI, "
        f"sigma 0 -> {float(train_cfg['max_xy_std']):g} m over {epochs} epochs",
        flush=True,
    )

    train_loader, val_loader = _loaders(
        train_set, val_set, int(train_cfg["batch_size"]), num_workers
    )
    modules = build_stage2_modules(
        config, channels, device, head=head, stage1_checkpoint=stage1_checkpoint
    )
    parameters = [p for p in modules.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=float(train_cfg["lr"]))
    scheduler = _schedule(optimizer, max(epochs * len(train_loader), 1))

    (output_dir / "config.yaml").write_text(yaml.safe_dump(dict(config), sort_keys=False))
    provenance = {
        "head": head,
        "message_content": message_content,
        "match_weight": match_weight,
        "stage1_checkpoint": str(stage1_checkpoint.resolve()),
        "stage2_epochs": epochs,
        "max_xy_std_m": float(train_cfg["max_xy_std"]),
    }
    history: List[Dict[str, Any]] = []
    best = {"corner_loss_m": math.inf, "epoch": 0}
    best_path = output_dir / "best.pth"

    for epoch in range(epochs):
        train_set.set_epoch(epoch)
        sigma = train_set.noise_schedule.sigma_for_epoch(epoch, epochs)
        started = time.time()
        train_metrics = _train_stage2_epoch(
            modules, train_loader, optimizer, scheduler, parameters, device,
            match_weight=match_weight, epoch=epoch, epochs=epochs, sigma=sigma,
            ablate_embeddings=ablate,
        )
        metrics = evaluate_pose(modules, val_loader, device, ablate_embeddings=ablate)
        print(
            f"  [ep{epoch + 1}/{epochs}] train_loss={train_metrics['train_loss']:.4f} "
            f"sigma={sigma:.3f} | val {_format_pose(metrics)} "
            f"({time.time() - started:.0f}s)",
            flush=True,
        )

        temperature = getattr(modules["pose"], "log_temperature", None)
        history.append(
            {
                "epoch": epoch + 1,
                "train_sigma_m": sigma,
                "lr": scheduler.get_last_lr()[0],
                "log_temperature": float(temperature.item()) if temperature is not None else None,
                **train_metrics,
                **metrics,
            }
        )
        (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        checkpoint = {
            **_checkpoint(modules, config, epoch + 1, metrics, channels),
            **provenance,
        }
        torch.save(checkpoint, output_dir / "latest.pth")
        if metrics["corner_loss_m"] < best["corner_loss_m"]:
            best = {"corner_loss_m": metrics["corner_loss_m"], "epoch": epoch + 1}
            torch.save(checkpoint, best_path)
            print(f"  -> new best val corner loss {metrics['corner_loss_m']:.4f} m", flush=True)

    print(
        f"\nbest val corner loss {best['corner_loss_m']:.4f} m at epoch "
        f"{best['epoch']}; checkpoint {best_path}",
        flush=True,
    )
    return best_path


def load_stage2(
    checkpoint_path: Path, device: torch.device
) -> Tuple[nn.ModuleDict, Dict[str, Any]]:
    """Rebuild the stage-2 modules from a checkpoint and load their weights."""
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    modules = build_stage2_modules(
        checkpoint["config"],
        int(checkpoint["roi_channels"]),
        device,
        head=checkpoint["head"],
    )
    modules.load_state_dict(checkpoint["state_dict"])
    modules.eval()
    return modules, checkpoint
