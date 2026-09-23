"""Stage-1 trainer: the embedding head and trunk, on the matching loss alone.

Spec section 4's stage 1. Nothing about the pose is trained or even evaluated
here: the only question is whether a transmitted per-object embedding, plus the
box geometry, lets the ego agent pick out each of its own objects' counterpart
in the CAV's object set. The gate is cross-agent Top-1 >= 0.85 on
scenario-disjoint validation scenarios.

Three deliberate choices, because each one could otherwise flatter the number:

- **The validation noise is fixed at the curriculum's maximum**, not at the
  epoch's own sigma. A moving evaluation target would make the per-epoch curve
  uninterpretable and would report the easiest condition at epoch 0.
- **Two controls are measured alongside the gate, every epoch.** ``chance`` is
  uniform random matching (1/n for n CAV objects). ``nearest`` matches each ego
  object to the nearest CAV box centre, needing neither embedding nor training.
  At sub-metre localization error the second one is expected to be strong, and
  a learned Top-1 that does not clear it has not demonstrated that the
  embedding carries anything.
- **The dustbin mass is logged**, since a matcher that collapses onto "nothing
  matches" also drives ``match_nll`` down -- most objects really are unmatched.

No AMP: the model is a few million parameters over object sets of at most 64
tokens, so the run is data-loader bound and fp16 would buy nothing while
putting Sinkhorn's log-domain normalization at risk.

Stage 2 lives in :mod:`embedding_aware_belt_fusion.alignformer.stage2`, which
imports this module's shared pieces (the scenario split, the loaders, the LR
schedule, ``embed_batch``). ``--stage 2`` on this module's CLI imports it
lazily, inside the functions, so the dependency stays one-directional.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader

from embedding_aware_belt_fusion.alignformer.cache import cache_path, read_frame
from embedding_aware_belt_fusion.alignformer.dataset import (
    NoiseSchedule,
    OPV2VObjectSetDataset,
    collate,
)
from embedding_aware_belt_fusion.alignformer.embedding import ObjectEmbedding
from embedding_aware_belt_fusion.alignformer.losses import match_nll
from embedding_aware_belt_fusion.alignformer.metrics import (
    chance_top1_sums,
    dustbin_mass_sums,
    nearest_centre_top1_counts,
    top1_counts,
)
from embedding_aware_belt_fusion.alignformer.model import AlignFormerB
from embedding_aware_belt_fusion.alignformer.trunk import MAX_OBJECTS
from embedding_aware_belt_fusion.alignformer.variance import (
    VARIANCE_MODES,
    variance_model_from_config,
)
from embedding_aware_belt_fusion.coloca.index import (
    AgentPair,
    load_or_build_pairs,
    pair_cache_path,
)
from embedding_aware_belt_fusion.coloca.train import split_scenarios

# Where stage 1 writes best.pth / latest.pth / history.json unless overridden.
DEFAULT_OUTPUT_DIR = Path("outputs/alignformer/stage1")
# Two npz reads per sample, ~0.5 MB each, so the loop is I/O bound without workers.
DEFAULT_NUM_WORKERS = 8
# Fraction of the run spent linearly warming the learning rate up to its peak,
# then a cosine decay -- the same schedule coloca/train.py converged under.
WARMUP_FRACTION = 0.05
GRAD_CLIP = 5.0
_PROGRESS_INTERVAL = 100


def _split_name(root: Path) -> str:
    """Cache subdirectory for an OPV2V split root (``.../OPV2V/train`` -> ``train``)."""
    return root.name


def build_pair_split(
    config: Mapping[str, Any]
) -> Tuple[List[AgentPair], List[AgentPair], List[str], List[str]]:
    """Split the train split's ego-CAV pairs into scenario-disjoint halves.

    By scenario, never by frame: consecutive OPV2V frames are near-duplicates,
    so a frame split leaks validation into training. The fraction and seed are
    the ones ``configs/alignformer_detector.yaml`` used for the detector's own
    split, so the validation scenarios here are unseen by the detector too.
    """
    data = config["data"]
    root = Path(data["train_root"])
    comm_range_m = float(data["comm_range_m"])
    pairs = load_or_build_pairs(
        root,
        pair_cache_path(data["pair_cache_dir"], _split_name(root), comm_range_m),
        comm_range_m,
    )
    train_scenarios, val_scenarios = split_scenarios(
        sorted({pair.scenario for pair in pairs}),
        float(data["val_scenario_fraction"]),
        int(data["split_seed"]),
    )
    train_set, val_set = set(train_scenarios), set(val_scenarios)
    return (
        [pair for pair in pairs if pair.scenario in train_set],
        [pair for pair in pairs if pair.scenario in val_set],
        train_scenarios,
        val_scenarios,
    )


def build_datasets(
    config: Mapping[str, Any], *, epochs: int
) -> Tuple[OPV2VObjectSetDataset, OPV2VObjectSetDataset]:
    """Build the stage-1 train and validation datasets.

    The training set ramps sigma 0 -> ``stage1_max_xy_std`` across ``epochs``
    and redraws its noise every epoch. The validation set is pinned at
    ``stage1_max_xy_std`` with ``total_epochs=1`` (the schedule then returns the
    maximum unconditionally) and deterministic per-sample noise.
    """
    data, train_cfg = config["data"], config["train"]
    train_pairs, val_pairs, _, _ = build_pair_split(config)

    common = {
        "cache_root": data["cache_root"],
        "split": _split_name(Path(data["train_root"])),
    }
    sigma = float(train_cfg["stage1_max_xy_std"])
    train_set = OPV2VObjectSetDataset(
        train_pairs,
        noise_schedule=NoiseSchedule(max_xy_std=sigma),
        train=True,
        total_epochs=epochs,
        seed=int(train_cfg["seed"]),
        **common,
    )
    val_set = build_eval_dataset(config, val_pairs, sigma)
    return train_set, val_set


def build_eval_dataset(
    config: Mapping[str, Any],
    pairs: Sequence[AgentPair],
    sigma: float,
    *,
    seed: Optional[int] = None,
) -> OPV2VObjectSetDataset:
    """A fixed-noise, deterministic dataset at exactly ``sigma`` metres.

    ``seed`` selects WHICH deterministic draw: the noise stays reproducible
    either way, but a sweep that wants several independent draws of the same
    pairs at the same sigma varies it. Left ``None``, the dataset's own default
    is used, which is what every earlier gate measured under.
    """
    data = config["data"]
    extra = {} if seed is None else {"seed": int(seed)}
    return OPV2VObjectSetDataset(
        pairs,
        cache_root=data["cache_root"],
        split=_split_name(Path(data["train_root"])),
        noise_schedule=NoiseSchedule(max_xy_std=sigma),
        train=False,
        total_epochs=1,
        **extra,
    )


def roi_channels(dataset: OPV2VObjectSetDataset) -> int:
    """Read the cached ROI feature channel count off the first usable record.

    The detector's BEV channel count is a property of the cache, not something
    this config should restate and then drift from.
    """
    for pair in dataset.pairs:
        record = read_frame(
            cache_path(
                dataset.cache_root, dataset.split, pair.scenario, pair.ego_id, pair.timestamp
            )
        )
        if record.roi.shape[0]:
            return int(record.roi.shape[1])
    raise RuntimeError("no cached frame in this split has any detection to size the ROI from")


def build_modules(
    config: Mapping[str, Any],
    channels: int,
    device: torch.device,
    *,
    variance_weighting: Optional[str] = None,
) -> nn.ModuleDict:
    """The trainable stage-1 pair: the embedding head plus AlignFormer head B.

    A ``ModuleDict`` rather than a bespoke wrapper class, so one ``state_dict``
    round-trips both and stage 2 can load the embedding head alone by prefix.

    ``variance_weighting`` reaches head B's Procrustes weights only. Stage 1's
    loss is ``match_nll`` on the Sinkhorn assignment and never touches the
    Kabsch solve, so stage-1 results are identical under every mode -- which is
    why the stage-1 checkpoint is shared across the stage-2 variants.
    """
    model_cfg = config["model"]
    if int(model_cfg["max_objects"]) != MAX_OBJECTS:
        raise ValueError(
            f"config model.max_objects={model_cfg['max_objects']} disagrees with "
            f"trunk.MAX_OBJECTS={MAX_OBJECTS}, which is what the dataset truncates to"
        )
    embed_dim = int(model_cfg["embed_dim"])
    modules = nn.ModuleDict(
        {
            "embedding": ObjectEmbedding(
                in_channels=channels,
                output_size=int(model_cfg["output_size"]),
                dim=embed_dim,
            ),
            "matcher": AlignFormerB(
                embed_dim=embed_dim,
                heading_lambda=float(model_cfg["heading_lambda"]),
                sinkhorn_iterations=int(model_cfg["sinkhorn_iterations"]),
                variance_model=variance_model_from_config(model_cfg, variance_weighting),
                model_dim=int(model_cfg["model_dim"]),
                layers=int(model_cfg["layers"]),
                heads=int(model_cfg["heads"]),
            ),
        }
    )
    return modules.to(device)


# The two fields ``embed_batch`` writes and ``zero_embeddings`` ablates.
_EMBEDDING_FIELDS = ("ego_embeddings", "cav_embeddings")


def zero_embeddings(batch: Mapping[str, Tensor]) -> Dict[str, Tensor]:
    """Return a **new** batch dict with every embedding replaced by zeros.

    This is the ``boxes_only`` message content: the trunk then sees box
    geometry and detector scores only, so the gap against the full model is the
    embedding's actual contribution. Geometry is passed through untouched --
    zeroing it too would ablate the wrong thing -- and the input dict is never
    mutated, so a caller can run both conditions on the same batch.

    A batch that does not carry the embedding fields yet (a raw collated batch,
    before :func:`embed_batch`) is returned unchanged apart from being copied:
    there is nothing to zero, which is not an error.
    """
    ablated = dict(batch)
    for field in _EMBEDDING_FIELDS:
        value = batch.get(field)
        if value is not None:
            ablated[field] = torch.zeros_like(value)
    return ablated


def _embed(head: ObjectEmbedding, roi: Tensor) -> Tensor:
    """Run the embedding head over a padded ``(B, N, C, k, k)`` ROI stack.

    Padded rows are embedded too and then masked out downstream by
    ``ego_mask``/``cav_mask``; the alternative, gathering only the real rows,
    saves nothing measurable on object sets this small.
    """
    batch, count = roi.shape[0], roi.shape[1]
    flat = head(roi.reshape((batch * count,) + tuple(roi.shape[2:])).float())
    return flat.reshape(batch, count, head.dim)


def embed_batch(
    head: ObjectEmbedding, batch: Mapping[str, Tensor], *, ablate: bool = False
) -> Dict[str, Tensor]:
    """Return a new batch carrying per-object embeddings for both sets.

    ``ablate=True`` routes the result through :func:`zero_embeddings`, so the
    boxes-only condition is *literally* the full model with its embeddings set
    to zero rather than a second code path that could drift from it. The head
    still runs, which costs one small MLP over at most 64 tokens per agent and
    buys a single definition of the ablation.
    """
    enriched = dict(batch)
    enriched["ego_embeddings"] = _embed(head, batch["ego_roi"])
    enriched["cav_embeddings"] = _embed(head, batch["cav_roi"])
    return zero_embeddings(enriched) if ablate else enriched


def forward_batch(
    modules: nn.ModuleDict, batch: Dict[str, Tensor], *, zero_embeddings: bool = False
):
    """Embed both object sets and run head B over them.

    The keyword deliberately keeps its stage-1 name (``scripts/
    analyze_association.py`` calls it) even though it shadows the module-level
    :func:`zero_embeddings` inside this function; ``embed_batch`` above is the
    one that calls it.
    """
    enriched = embed_batch(modules["embedding"], batch, ablate=zero_embeddings)
    return modules["matcher"](enriched)


def _to_device(batch: Mapping[str, Tensor], device: torch.device) -> Dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


class _MatchingTally:
    """Accumulates the association diagnostics over a whole split.

    Counts, not per-batch means: the last batch is usually short and averaging
    accuracies would weight it as heavily as a full one.
    """

    def __init__(self) -> None:
        self.correct = 0
        self.strict_correct = 0
        self.countable = 0
        self.nearest_correct = 0
        self.chance = 0.0
        self.dustbin = 0.0
        self.rows = 0
        self.loss = 0.0
        self.batches = 0
        self.skipped = 0

    def update(self, batch: Mapping[str, Tensor], log_assignment: Tensor, loss: float) -> None:
        ego_match, ego_mask, cav_mask = batch["ego_match"], batch["ego_mask"], batch["cav_mask"]
        correct, countable = top1_counts(log_assignment, ego_match, ego_mask)
        strict, _ = top1_counts(log_assignment, ego_match, ego_mask, include_dustbin=True)
        nearest, _ = nearest_centre_top1_counts(
            batch["ego_boxes"], batch["cav_boxes"], ego_match, ego_mask, cav_mask
        )
        chance, _ = chance_top1_sums(ego_match, ego_mask, cav_mask)
        dustbin, rows = dustbin_mass_sums(log_assignment, ego_mask)

        self.correct += correct
        self.strict_correct += strict
        self.countable += countable
        self.nearest_correct += nearest
        self.chance += chance
        self.dustbin += dustbin
        self.rows += rows
        self.loss += loss
        self.batches += 1

    def compute(self) -> Dict[str, float]:
        # NaN, not 0.0, on an empty tally: nothing was scored, which is not the
        # same as scoring nothing correctly. Matches metrics.top1_accuracy.
        def ratio(numerator: float, denominator: int) -> float:
            return numerator / denominator if denominator else math.nan

        return {
            "top1": ratio(self.correct, self.countable),
            "top1_including_dustbin": ratio(self.strict_correct, self.countable),
            "top1_nearest_centre": ratio(self.nearest_correct, self.countable),
            "top1_chance": ratio(self.chance, self.countable),
            "dustbin_mass": ratio(self.dustbin, self.rows),
            "match_nll": ratio(self.loss, self.batches),
            "countable_objects": float(self.countable),
            "skipped_batches": float(self.skipped),
        }


@torch.no_grad()
def evaluate_matching(
    modules: nn.ModuleDict,
    loader: DataLoader,
    device: torch.device,
    *,
    zero_embeddings: bool = False,
) -> Dict[str, float]:
    """Measure Top-1 and its controls over a whole loader."""
    modules.eval()
    tally = _MatchingTally()
    for batch in loader:
        batch = _to_device(batch, device)
        estimate = forward_batch(modules, batch, zero_embeddings=zero_embeddings)
        if estimate.log_assignment is None:
            # Every sample in this batch had an empty object set on one side,
            # so head B short-circuits before building a correspondence. There
            # is nothing to score; counting it as zero would be wrong.
            tally.skipped += 1
            continue
        loss = match_nll(estimate.log_assignment, batch["ego_match"], batch["cav_match"])
        tally.update(batch, estimate.log_assignment, float(loss.item()))
    return tally.compute()


def _format(metrics: Mapping[str, float]) -> str:
    return (
        f"top1={metrics['top1']:.4f} (+dustbin {metrics['top1_including_dustbin']:.4f}) "
        f"nearest={metrics['top1_nearest_centre']:.4f} chance={metrics['top1_chance']:.4f} "
        f"dustbin_mass={metrics['dustbin_mass']:.4f} nll={metrics['match_nll']:.4f} "
        f"n={int(metrics['countable_objects'])}"
    )


def _loaders(
    train_set: OPV2VObjectSetDataset,
    val_set: OPV2VObjectSetDataset,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    kwargs = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": True}
    # The train loader must NOT use persistent workers: each worker holds its
    # own copy of the dataset, so `set_epoch` on the main-process copy would
    # never reach them and every epoch would silently reuse epoch 0's noise
    # draw -- and, here, epoch 0's noise is sigma = 0 exactly.
    train_loader = DataLoader(
        train_set, shuffle=True, drop_last=True, collate_fn=collate,
        persistent_workers=False, **kwargs,
    )
    val_loader = DataLoader(
        val_set, shuffle=False, drop_last=False, collate_fn=collate,
        persistent_workers=num_workers > 0, **kwargs,
    )
    return train_loader, val_loader


def _schedule(optimizer, total_steps: int):
    warmup = max(int(WARMUP_FRACTION * total_steps), 1)

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@dataclass(frozen=True)
class _Stage1Setup:
    """Everything stage 1 assembles once, before the first epoch."""

    train_set: OPV2VObjectSetDataset
    train_loader: DataLoader
    val_loader: DataLoader
    modules: nn.ModuleDict
    parameters: List[Tensor]
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LambdaLR
    channels: int


def _build_stage1(
    config: Mapping[str, Any],
    *,
    epochs: int,
    num_workers: int,
    device: torch.device,
    zero_embeddings: bool,
) -> _Stage1Setup:
    """Assemble the datasets, loaders, modules and optimizer for stage 1."""
    train_cfg = config["train"]
    train_set, val_set = build_datasets(config, epochs=epochs)
    channels = roi_channels(train_set)
    print(
        f"stage 1: {len(train_set)} train pairs / {len(val_set)} val pairs, "
        f"{channels}-channel ROI, embeddings "
        f"{'ZEROED (control)' if zero_embeddings else 'trained'}",
        flush=True,
    )

    train_loader, val_loader = _loaders(
        train_set, val_set, int(train_cfg["batch_size"]), num_workers
    )
    modules = build_modules(config, channels, device)
    parameters = [p for p in modules.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=float(train_cfg["lr"]))
    return _Stage1Setup(
        train_set=train_set,
        train_loader=train_loader,
        val_loader=val_loader,
        modules=modules,
        parameters=parameters,
        optimizer=optimizer,
        scheduler=_schedule(optimizer, max(epochs * len(train_loader), 1)),
        channels=channels,
    )


def _train_one_epoch(
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
    zero_embeddings: bool,
) -> float:
    """One pass over the training split; returns the mean match NLL."""
    modules.train()
    running, steps, started = 0.0, 0, time.time()

    for step, batch in enumerate(loader):
        batch = _to_device(batch, device)
        estimate = forward_batch(modules, batch, zero_embeddings=zero_embeddings)
        if estimate.log_assignment is None:
            # Head B short-circuits when an object set is empty on one side;
            # there is no correspondence to supervise.
            continue
        loss = match_weight * match_nll(
            estimate.log_assignment, batch["ego_match"], batch["cav_match"]
        )
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite match loss at epoch {epoch + 1} step {step}; see the "
                "NaN guards in trunk.py::_attend_without_degenerate_rows and "
                "model.py's atan2 has_weight guard before weakening anything"
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP)
        optimizer.step()
        scheduler.step()  # per step, not per epoch

        running += float(loss.item())
        steps += 1
        if step % _PROGRESS_INTERVAL == 0:
            rate = (step + 1) / (time.time() - started)
            print(
                f"  [ep{epoch + 1}/{epochs} sigma={sigma:.3f}] step {step}/"
                f"{len(loader)} loss={running / max(steps, 1):.4f} {rate:.1f} it/s",
                flush=True,
            )

    return running / max(steps, 1)


def train_stage1(
    config: Mapping[str, Any],
    *,
    output_dir: Optional[Path] = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
    device: Optional[torch.device] = None,
    zero_embeddings: bool = False,
) -> Path:
    """Train the embedding head and trunk on ``match_nll``; return the best checkpoint.

    "Best" is by validation Top-1, measured at the curriculum's maximum sigma.
    """
    train_cfg = config["train"]
    epochs = int(train_cfg["stage1_epochs"])
    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(int(train_cfg["seed"]))
    np.random.seed(int(train_cfg["seed"]))

    setup = _build_stage1(
        config, epochs=epochs, num_workers=num_workers, device=device,
        zero_embeddings=zero_embeddings,
    )
    train_set, train_loader, val_loader = setup.train_set, setup.train_loader, setup.val_loader
    modules, channels = setup.modules, setup.channels
    match_weight = float(train_cfg["match_weight"])

    (output_dir / "config.yaml").write_text(yaml.safe_dump(dict(config), sort_keys=False))
    history: List[Dict[str, Any]] = []
    best = {"top1": -math.inf, "epoch": 0}
    best_path = output_dir / "best.pth"

    for epoch in range(epochs):
        train_set.set_epoch(epoch)
        sigma = train_set.noise_schedule.sigma_for_epoch(epoch, epochs)
        started = time.time()
        train_loss = _train_one_epoch(
            modules, train_loader, setup.optimizer, setup.scheduler, setup.parameters,
            device, match_weight=match_weight, epoch=epoch, epochs=epochs, sigma=sigma,
            zero_embeddings=zero_embeddings,
        )
        metrics = evaluate_matching(
            modules, val_loader, device, zero_embeddings=zero_embeddings
        )
        print(
            f"  [ep{epoch + 1}/{epochs}] train_nll={train_loss:.4f} sigma={sigma:.3f} | "
            f"val {_format(metrics)} ({time.time() - started:.0f}s)",
            flush=True,
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_sigma_m": sigma,
                "train_match_nll": train_loss,
                "lr": setup.scheduler.get_last_lr()[0],
                "log_temperature": float(modules["matcher"].log_temperature.item()),
                "dustbin_score": float(modules["matcher"].dustbin.item()),
                **metrics,
            }
        )
        # Written every epoch, not at the end, so a run killed mid-way still
        # leaves a readable curve and a usable checkpoint behind.
        (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        checkpoint = _checkpoint(modules, config, epoch + 1, metrics, channels)
        torch.save(checkpoint, output_dir / "latest.pth")
        if metrics["top1"] > best["top1"]:
            best = {"top1": metrics["top1"], "epoch": epoch + 1}
            torch.save(checkpoint, best_path)
            print(f"  -> new best val Top-1 {metrics['top1']:.4f}", flush=True)

    print(
        f"\nbest val Top-1 {best['top1']:.4f} at epoch {best['epoch']}; "
        f"checkpoint {best_path}",
        flush=True,
    )
    return best_path


def _checkpoint(
    modules: nn.ModuleDict,
    config: Mapping[str, Any],
    epoch: int,
    metrics: Mapping[str, float],
    channels: int,
) -> Dict[str, Any]:
    return {
        "state_dict": modules.state_dict(),
        "config": dict(config),
        "epoch": epoch,
        "roi_channels": channels,
        "val_metrics": dict(metrics),
    }


def load_stage1(
    checkpoint_path: Path, device: torch.device
) -> Tuple[nn.ModuleDict, Dict[str, Any]]:
    """Rebuild the stage-1 modules from a checkpoint and load their weights."""
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    modules = build_modules(checkpoint["config"], int(checkpoint["roi_channels"]), device)
    modules.load_state_dict(checkpoint["state_dict"])
    modules.eval()
    return modules, checkpoint


def parse_args() -> argparse.Namespace:
    from embedding_aware_belt_fusion.alignformer import stage2

    parser = argparse.ArgumentParser(
        description="Train AlignFormer stage 1 (matching) or stage 2 (pose)"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/alignformer.yaml"))
    parser.add_argument("--stage", type=int, choices=(1, 2), default=1)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="overrides train.stage1_epochs / train.stage2_epochs",
    )
    parser.add_argument(
        "--zero-embeddings",
        action="store_true",
        help="stage 1 ablation control: feed the trunk zero embeddings, leaving box "
             "geometry only. Stage 2 uses --message-content boxes_only for the same thing.",
    )
    parser.add_argument("--head", choices=stage2.HEADS, default="B", help="stage 2 only")
    parser.add_argument(
        "--message-content", choices=stage2.MESSAGE_CONTENTS, default="boxes+embeddings",
        help="stage 2 only: what each agent transmits",
    )
    parser.add_argument(
        "--match-weight", type=float, default=None,
        help="stage 2 only: weight on the auxiliary matching NLL. Defaults to "
             "train.match_weight for head B and 0 for head A.",
    )
    parser.add_argument(
        "--stage1-checkpoint", type=Path, default=None,
        help="stage 2 only: overrides the warm-start checkpoint chosen by --message-content",
    )
    parser.add_argument(
        "--variance-weighting", choices=VARIANCE_MODES, default=None,
        help="stage 2 only: how to weight each correspondence in the Procrustes fit by "
             "its inverse disagreement variance. Overrides the config's "
             "model.correspondence_variance.mode. See alignformer/variance.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.stage == 1:
        if args.epochs is not None:
            config["train"]["stage1_epochs"] = args.epochs
        train_stage1(
            config,
            output_dir=args.output_dir,
            num_workers=args.num_workers,
            zero_embeddings=args.zero_embeddings,
        )
        return
    from embedding_aware_belt_fusion.alignformer.stage2 import train_stage2

    train_stage2(
        config,
        args.head,
        args.message_content,
        args.match_weight,
        variance_weighting=args.variance_weighting,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        stage1_checkpoint=args.stage1_checkpoint,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
