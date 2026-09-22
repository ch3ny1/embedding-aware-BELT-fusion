"""Re-analysis of stage-1 cross-agent association on the geometrically hard subset.

Task 13 measured Top-1 over *every* ego object that has a counterpart and found
the full model and the boxes-only control within 0.0006 of each other. That
number is dominated by ego objects with exactly one plausible CAV candidate,
which geometry resolves trivially. This module re-scores the same cached
detections and the same two trained checkpoints on the subset where geometry is
genuinely ambiguous, and probes why the embedding does or does not help there.

Nothing here retrains, and nothing here imports a modified copy of the model,
loss or metric code: it reuses ``alignformer.train``'s own loaders and
``alignformer.model``'s own forward passes so the numbers are the same
quantities the gate reported, restricted to a subset.

Definitions fixed in advance (see the report for the pre-registration note):

- An ego object is **ambiguous at radius r** when, after the CAV boxes are
  moved by the *ground-truth* SE(2) correction, two or more CAV box centres lie
  within ``r`` metres of it. The criterion uses the GT correction, so the
  subset is a property of the detections alone and is identical at every noise
  level; only the model's input changes with sigma.
- ``r`` is swept over ``AMBIGUITY_RADII`` and sigma over ``SIGMAS``; every cell
  is reported, so no threshold is chosen after the fact.
- Uncertainty is a cluster bootstrap over ego-CAV **pairs** (not objects):
  objects inside one frame are strongly correlated, so an object-level
  resample would understate the interval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader

from embedding_aware_belt_fusion.alignformer.dataset import collate
from embedding_aware_belt_fusion.alignformer.embedding import ObjectEmbedding
from embedding_aware_belt_fusion.alignformer.losses import apply_se2, match_nll
from embedding_aware_belt_fusion.alignformer.train import (
    build_eval_dataset,
    build_pair_split,
    forward_batch,
    load_stage1,
)
from embedding_aware_belt_fusion.alignformer.trunk import tokenize

# Pre-registered sweep. 4 radii x 4 noise levels = 16 cells, all reported.
AMBIGUITY_RADII: Tuple[float, ...] = (2.0, 3.0, 5.0, 8.0)
SIGMAS: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)
# Radius used to pick the "different nearby CAV object" of the discriminability
# probe: the widest ambiguity radius, so the distractor is always one of the
# candidates some cell of the sweep counts as competing.
PROBE_RADIUS = 8.0

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 12345
CONFIDENCE = 0.95

BATCH_SIZE = 64
# Box layout is OpenCOOD 'hwl' [x, y, z, h, w, l, yaw]; width/length columns.
_BOX_WIDTH, _BOX_LENGTH = 4, 5
_GRADIENT_BATCHES = 20


# --------------------------------------------------------------------------
# Per-object record collection
# --------------------------------------------------------------------------


class ObjectRecords:
    """Flat per-ego-object arrays accumulated over a whole split."""

    def __init__(self) -> None:
        self.pair: List[np.ndarray] = []
        self.candidates: List[np.ndarray] = []
        self.correct_full: List[np.ndarray] = []
        self.correct_zero: List[np.ndarray] = []
        self.correct_nearest: List[np.ndarray] = []
        self.correct_shuffled: List[np.ndarray] = []

    def add(self, **columns: np.ndarray) -> None:
        for name, value in columns.items():
            getattr(self, name).append(value)

    def finish(self) -> Dict[str, np.ndarray]:
        return {
            name: np.concatenate(getattr(self, name)) if getattr(self, name) else np.empty(0)
            for name in (
                "pair", "candidates", "correct_full", "correct_zero", "correct_nearest",
                "correct_shuffled",
            )
        }


def _aligned_cav_centres(batch: Mapping[str, Tensor]) -> Tensor:
    """CAV box centres moved by the ground-truth SE(2) correction.

    ``dataset.__getitem__`` projects the CAV boxes with the *noisy* pose and
    returns ``(psi_true, t_true)`` such that applying it recovers the true
    projection, so this is where the CAV objects actually are.
    """
    return apply_se2(batch["cav_boxes"][..., :2], batch["psi_true"], batch["t_true"])


def _candidate_counts(batch: Mapping[str, Tensor], radius: float) -> Tensor:
    """``(B, M)`` count of CAV boxes within ``radius`` of each ego box, GT-aligned."""
    distance = torch.cdist(batch["ego_boxes"][..., :2].float(), _aligned_cav_centres(batch).float())
    distance = distance.masked_fill(~batch["cav_mask"].unsqueeze(1), float("inf"))
    return (distance <= radius).sum(dim=2)


def _nearest_centre_prediction(batch: Mapping[str, Tensor]) -> Tensor:
    """``(B, M)`` argmin over CAV centres as the model sees them (noisy projection).

    Same quantity ``metrics.nearest_centre_top1_counts`` scores, but kept as
    per-object predictions instead of a summed count.
    """
    distance = torch.cdist(
        batch["ego_boxes"][..., :2].float(), batch["cav_boxes"][..., :2].float()
    )
    distance = distance.masked_fill(~batch["cav_mask"].unsqueeze(1), float("inf"))
    return distance.argmin(dim=2)


def _shuffle_cav_rois(batch: Mapping[str, Tensor], generator: torch.Generator) -> Dict[str, Tensor]:
    """Return a copy of ``batch`` with each sample's real CAV ROI features permuted.

    ``cav_roi`` feeds the embedding head and nothing else, so this detaches
    every CAV object's *appearance* from its box while leaving the geometry, the
    scores and the ground-truth correspondence untouched. If the full model's
    Top-1 is unchanged by it, the model's decision does not depend on the
    embedding -- a causal statement the frozen boxes-only control cannot make,
    since that control was also trained differently.
    """
    keys = torch.rand(batch["cav_mask"].shape, device=batch["cav_mask"].device,
                      generator=generator)
    # Padding always sits at the end of a row, so sending invalid entries to the
    # back keeps the permutation inside the real objects.
    keys = keys.masked_fill(~batch["cav_mask"], float("inf"))
    permutation = keys.argsort(dim=1)
    roi = batch["cav_roi"]
    index = permutation.reshape(permutation.shape + (1,) * (roi.dim() - 2)).expand_as(roi)
    shuffled = dict(batch)
    shuffled["cav_roi"] = roi.gather(1, index)
    return shuffled


def _top1_prediction(log_assignment: Tensor, ego_count: int) -> Tensor:
    """Argmax over the real CAV columns only -- the gate's own reading."""
    return log_assignment[:, :ego_count, :-1].argmax(dim=2)


@torch.no_grad()
def collect_records(
    full: nn.ModuleDict,
    zero: nn.ModuleDict,
    loader: DataLoader,
    device: torch.device,
    radii: Sequence[float],
) -> Dict[str, np.ndarray]:
    """Score every countable ego object under all three matchers, once per split."""
    records = ObjectRecords()
    offset = 0
    generator = torch.Generator(device=device).manual_seed(BOOTSTRAP_SEED)
    for batch in loader:
        batch_size = batch["ego_boxes"].shape[0]
        batch = {key: value.to(device) for key, value in batch.items()}
        ego_count = batch["ego_boxes"].shape[1]
        countable = batch["ego_mask"] & (batch["ego_match"] >= 0)

        estimate_full = forward_batch(full, batch, zero_embeddings=False)
        estimate_zero = forward_batch(zero, batch, zero_embeddings=True)
        estimate_shuffled = forward_batch(
            full, _shuffle_cav_rois(batch, generator), zero_embeddings=False
        )
        if estimate_full.log_assignment is None or estimate_zero.log_assignment is None:
            offset += batch_size
            continue

        predicted_full = _top1_prediction(estimate_full.log_assignment, ego_count)
        predicted_zero = _top1_prediction(estimate_zero.log_assignment, ego_count)
        predicted_shuffled = _top1_prediction(estimate_shuffled.log_assignment, ego_count)
        predicted_near = _nearest_centre_prediction(batch)
        counts = torch.stack([_candidate_counts(batch, r) for r in radii], dim=-1)

        selected = countable.nonzero(as_tuple=False)
        rows, columns = selected[:, 0], selected[:, 1]
        truth = batch["ego_match"][rows, columns]
        records.add(
            pair=(rows + offset).cpu().numpy().astype(np.int32),
            candidates=counts[rows, columns].cpu().numpy().astype(np.int16),
            correct_full=(predicted_full[rows, columns] == truth).cpu().numpy(),
            correct_zero=(predicted_zero[rows, columns] == truth).cpu().numpy(),
            correct_nearest=(predicted_near[rows, columns] == truth).cpu().numpy(),
            correct_shuffled=(predicted_shuffled[rows, columns] == truth).cpu().numpy(),
        )
        offset += batch_size
    return records.finish()


# --------------------------------------------------------------------------
# Subset accuracies and the cluster bootstrap
# --------------------------------------------------------------------------


def _per_pair_sums(
    pair: np.ndarray, mask: np.ndarray, columns: Mapping[str, np.ndarray], pairs: int
) -> Dict[str, np.ndarray]:
    """Sum each 0/1 column, and the subset size, per ego-CAV pair."""
    selected = pair[mask]
    sums = {"n": np.bincount(selected, minlength=pairs).astype(np.float64)}
    for name, values in columns.items():
        sums[name] = np.bincount(selected, weights=values[mask].astype(np.float64),
                                 minlength=pairs)
    return sums


def bootstrap_delta(
    sums: Mapping[str, np.ndarray], first: str, second: str, rng: np.random.Generator
) -> Dict[str, float]:
    """Percentile CI for ``acc(first) - acc(second)`` under a cluster bootstrap.

    Pairs (frames), not objects, are the resampling unit: two objects in the
    same frame share a pose error, a scene and a detector state, so treating
    them as independent would shrink the interval by roughly the square root of
    the objects-per-frame count.
    """
    pairs = sums["n"].shape[0]
    draws = rng.integers(0, pairs, size=(BOOTSTRAP_RESAMPLES, pairs))
    counts = sums["n"][draws].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        deltas = (sums[first][draws].sum(axis=1) - sums[second][draws].sum(axis=1)) / counts
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size == 0:
        return {"low": float("nan"), "high": float("nan"), "std_error": float("nan")}
    tail = (1.0 - CONFIDENCE) / 2.0
    return {
        "low": float(np.quantile(deltas, tail)),
        "high": float(np.quantile(deltas, 1.0 - tail)),
        "std_error": float(deltas.std(ddof=1)),
    }


def subset_report(
    records: Mapping[str, np.ndarray], radius_index: int, pairs: int, rng: np.random.Generator
) -> Dict[str, object]:
    """Accuracies, deltas and CIs on the objects with >= 2 candidates at one radius."""
    mask = records["candidates"][:, radius_index] >= 2
    size = int(mask.sum())
    columns = {
        "full": records["correct_full"],
        "zero": records["correct_zero"],
        "nearest": records["correct_nearest"],
        "shuffled": records["correct_shuffled"],
    }
    report: Dict[str, object] = {"n_objects": size, "n_pairs": int(
        np.unique(records["pair"][mask]).size if size else 0
    )}
    if size == 0:
        return report

    for name, values in columns.items():
        report[f"top1_{name}"] = float(values[mask].mean())
    sums = _per_pair_sums(records["pair"], mask, columns, pairs)
    for first, second in (("full", "zero"), ("full", "nearest"), ("full", "shuffled")):
        interval = bootstrap_delta(sums, first, second, rng)
        report[f"delta_{first}_minus_{second}"] = {
            "value": report[f"top1_{first}"] - report[f"top1_{second}"],
            **interval,
        }
    full, zero = columns["full"][mask], columns["zero"][mask]
    report["discordant_full_only"] = int((full & ~zero).sum())
    report["discordant_zero_only"] = int((~full & zero).sum())
    return report


# --------------------------------------------------------------------------
# Probe 1/2: embedding discriminability and the extents of the distractors
# --------------------------------------------------------------------------


def _trunk_match_features(
    matcher: nn.Module, batch: Mapping[str, Tensor], embeddings: Mapping[str, Tensor]
) -> Tuple[Tensor, Tensor]:
    """The unit-norm features whose inner product *is* the Sinkhorn score."""
    ego, cav = matcher.trunk(
        tokenize(batch["ego_boxes"], batch["ego_scores"], embeddings["ego"]),
        tokenize(batch["cav_boxes"], batch["cav_scores"], embeddings["cav"]),
        batch["ego_mask"],
        batch["cav_mask"],
    )
    return (
        F.normalize(matcher.match_projection(ego), dim=-1),
        F.normalize(matcher.match_projection(cav), dim=-1),
    )


def _roi_descriptor(roi: Tensor) -> Tensor:
    """Flatten and L2-normalize the raw pooled ROI features, (B, N, C*k*k)."""
    flat = roi.reshape(roi.shape[0], roi.shape[1], -1).float()
    return F.normalize(flat, dim=-1)


@torch.no_grad()
def probe_similarity(
    full: nn.ModuleDict, loader: DataLoader, device: torch.device
) -> Dict[str, object]:
    """Cosine similarity to the true partner vs to the nearest different CAV object.

    Three representations are scored at once, because they fail differently:
    the raw pooled ROI features (is identity present in the detector's BEV map
    at all?), the learned 128-d embedding (did the head keep it?), and the
    trunk's match features (is it in what Sinkhorn actually scores?).
    """
    channels: Dict[str, Dict[str, List[np.ndarray]]] = {
        name: {"partner": [], "distractor": []} for name in ("roi", "embedding", "match")
    }
    extents: Dict[str, List[np.ndarray]] = {"d_width": [], "d_length": [], "distance": []}
    widths: List[np.ndarray] = []
    lengths: List[np.ndarray] = []

    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        if batch["ego_boxes"].shape[1] == 0 or batch["cav_boxes"].shape[1] == 0:
            continue
        head = full["embedding"]
        embeddings = {
            "ego": _run_embedding(head, batch["ego_roi"]),
            "cav": _run_embedding(head, batch["cav_roi"]),
        }
        match_ego, match_cav = _trunk_match_features(full["matcher"], batch, embeddings)
        features = {
            "roi": (_roi_descriptor(batch["ego_roi"]), _roi_descriptor(batch["cav_roi"])),
            "embedding": (embeddings["ego"], embeddings["cav"]),
            "match": (match_ego, match_cav),
        }

        distance = torch.cdist(
            batch["ego_boxes"][..., :2].float(), _aligned_cav_centres(batch).float()
        ).masked_fill(~batch["cav_mask"].unsqueeze(1), float("inf"))
        countable = batch["ego_mask"] & (batch["ego_match"] >= 0)
        selected = countable.nonzero(as_tuple=False)
        if selected.numel() == 0:
            continue
        rows, columns = selected[:, 0], selected[:, 1]
        partner = batch["ego_match"][rows, columns]

        # Nearest CAV object that is NOT the true partner, within PROBE_RADIUS.
        own = distance[rows, columns].clone()
        own[torch.arange(own.shape[0], device=device), partner] = float("inf")
        distractor_distance, distractor = own.min(dim=1)
        keep = distractor_distance <= PROBE_RADIUS
        if not bool(keep.any()):
            continue
        rows, columns = rows[keep], columns[keep]
        partner, distractor = partner[keep], distractor[keep]

        for name, (ego_feature, cav_feature) in features.items():
            ego_vectors = ego_feature[rows, columns]
            channels[name]["partner"].append(
                (ego_vectors * cav_feature[rows, partner]).sum(-1).cpu().numpy()
            )
            channels[name]["distractor"].append(
                (ego_vectors * cav_feature[rows, distractor]).sum(-1).cpu().numpy()
            )

        cav_boxes = batch["cav_boxes"]
        partner_box = cav_boxes[rows, partner]
        distractor_box = cav_boxes[rows, distractor]
        extents["d_width"].append(
            (partner_box[:, _BOX_WIDTH] - distractor_box[:, _BOX_WIDTH]).abs().cpu().numpy()
        )
        extents["d_length"].append(
            (partner_box[:, _BOX_LENGTH] - distractor_box[:, _BOX_LENGTH]).abs().cpu().numpy()
        )
        extents["distance"].append(distractor_distance[keep].cpu().numpy())
        real = batch["cav_mask"]
        widths.append(cav_boxes[..., _BOX_WIDTH][real].cpu().numpy())
        lengths.append(cav_boxes[..., _BOX_LENGTH][real].cpu().numpy())

    report: Dict[str, object] = {}
    for name, sides in channels.items():
        partner_values = np.concatenate(sides["partner"])
        distractor_values = np.concatenate(sides["distractor"])
        report[name] = _similarity_summary(partner_values, distractor_values)
    report["distractor_extents"] = {
        key: _distribution(np.concatenate(values)) for key, values in extents.items()
    }
    report["all_box_extents"] = {
        "width": _distribution(np.concatenate(widths)),
        "length": _distribution(np.concatenate(lengths)),
    }
    return report


def _distribution(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "n": int(values.size),
    }


def _similarity_summary(partner: np.ndarray, distractor: np.ndarray) -> Dict[str, object]:
    """Separation between the two cosine distributions, plus their overlap.

    ``auc`` is the probability that a random true partner scores above a random
    distractor (the Mann-Whitney statistic): 0.5 means the representation
    carries no per-object identity at all, 1.0 means it separates perfectly.
    ``pairwise_accuracy`` is the same quantity computed *within* each ego
    object, which is the decision the matcher actually faces.
    """
    combined = np.concatenate([partner, distractor])
    ranks = combined.argsort().argsort().astype(np.float64) + 1.0
    n_partner, n_distractor = partner.size, distractor.size
    rank_sum = ranks[:n_partner].sum()
    auc = (rank_sum - n_partner * (n_partner + 1) / 2.0) / (n_partner * n_distractor)
    return {
        "partner": _distribution(partner),
        "distractor": _distribution(distractor),
        "auc": float(auc),
        "pairwise_accuracy": float((partner > distractor).mean()),
        "mean_gap": float(partner.mean() - distractor.mean()),
        "cohens_d": float(
            (partner.mean() - distractor.mean())
            / np.sqrt(0.5 * (partner.var() + distractor.var()) + 1e-12)
        ),
    }


def _run_embedding(head: ObjectEmbedding, roi: Tensor) -> Tensor:
    """``_embed`` without the zero branch: (B, N, C, k, k) -> (B, N, d)."""
    batch, count = roi.shape[0], roi.shape[1]
    flat = head(roi.reshape((batch * count,) + tuple(roi.shape[2:])).float())
    return flat.reshape(batch, count, head.dim)


# --------------------------------------------------------------------------
# Probe 3: is any gradient reaching the embedding branch?
# --------------------------------------------------------------------------


def probe_gradients(
    full: nn.ModuleDict, loader: DataLoader, device: torch.device, config: Mapping
) -> Dict[str, object]:
    """Compare the loss gradient on the geometry and embedding input columns.

    ``trunk.input_projection`` is a single ``Linear`` whose input is
    ``[9 geometry features | 128 embedding dims]``, so the two column blocks of
    its weight gradient are directly comparable: same layer, same output, same
    loss. Per-column RMS removes the 9-vs-128 width difference.

    Also reports how far the trained embedding head moved from its
    initialization, rebuilt with the training seed, and how distinguishable its
    outputs are (mean pairwise cosine between different objects' embeddings): a
    head that collapsed to one vector cannot carry identity whatever the
    gradient did.
    """
    full.train()
    projection = full["matcher"].trunk.input_projection
    embed_dim = int(config["model"]["embed_dim"])
    geometry_rms: List[float] = []
    embedding_rms: List[float] = []
    head_norm: List[float] = []

    for index, batch in enumerate(loader):
        if index >= _GRADIENT_BATCHES:
            break
        batch = {key: value.to(device) for key, value in batch.items()}
        estimate = forward_batch(full, batch, zero_embeddings=False)
        if estimate.log_assignment is None:
            continue
        loss = match_nll(estimate.log_assignment, batch["ego_match"], batch["cav_match"])
        full.zero_grad(set_to_none=True)
        loss.backward()

        gradient = projection.weight.grad
        geometry_rms.append(float(gradient[:, :-embed_dim].pow(2).mean().sqrt().item()))
        embedding_rms.append(float(gradient[:, -embed_dim:].pow(2).mean().sqrt().item()))
        head_norm.append(
            float(
                torch.sqrt(
                    sum(
                        p.grad.pow(2).sum()
                        for p in full["embedding"].parameters()
                        if p.grad is not None
                    )
                ).item()
            )
        )
    full.eval()
    full.zero_grad(set_to_none=True)

    return {
        "batches": len(geometry_rms),
        "input_projection_geometry_rms": float(np.mean(geometry_rms)),
        "input_projection_embedding_rms": float(np.mean(embedding_rms)),
        "geometry_over_embedding_ratio": float(
            np.mean(geometry_rms) / max(np.mean(embedding_rms), 1e-30)
        ),
        "embedding_head_grad_norm": float(np.mean(head_norm)),
        **_head_movement(full["embedding"], config),
    }


def _head_movement(head: ObjectEmbedding, config: Mapping) -> Dict[str, float]:
    """Relative weight change of the embedding head against a fresh init."""
    torch.manual_seed(int(config["train"]["seed"]))
    fresh = ObjectEmbedding(
        in_channels=head.net[1].in_features // (int(config["model"]["output_size"]) ** 2),
        output_size=int(config["model"]["output_size"]),
        dim=head.dim,
    )
    trained = torch.nn.utils.parameters_to_vector(head.parameters()).detach().cpu()
    initial = torch.nn.utils.parameters_to_vector(fresh.parameters()).detach().cpu()
    return {
        "head_weight_relative_change": float(
            (trained - initial).norm().item() / max(initial.norm().item(), 1e-12)
        ),
        "head_weight_norm_trained": float(trained.norm().item()),
        "head_weight_norm_init": float(initial.norm().item()),
    }


@torch.no_grad()
def probe_embedding_spread(
    full: nn.ModuleDict, loader: DataLoader, device: torch.device, config: Mapping
) -> Dict[str, float]:
    """Mean cosine between embeddings of *different* objects, trained vs fresh init.

    Near 1.0 means every object gets the same descriptor, i.e. the head emits no
    identity regardless of what its input contained.
    """
    torch.manual_seed(int(config["train"]["seed"]))
    head = full["embedding"]
    fresh = ObjectEmbedding(
        in_channels=head.net[1].in_features // (int(config["model"]["output_size"]) ** 2),
        output_size=int(config["model"]["output_size"]),
        dim=head.dim,
    ).to(device).eval()

    sums = {"trained": [], "init": []}
    for index, batch in enumerate(loader):
        if index >= _GRADIENT_BATCHES:
            break
        roi = batch["ego_roi"].to(device)
        mask = batch["ego_mask"].to(device)
        if not bool(mask.any()):
            continue
        flat = roi[mask].float()
        for name, module in (("trained", head), ("init", fresh)):
            vectors = module(flat)
            similarity = vectors @ vectors.t()
            off_diagonal = ~torch.eye(
                similarity.shape[0], dtype=torch.bool, device=similarity.device
            )
            sums[name].append(float(similarity[off_diagonal].mean().item()))
    return {f"mean_cosine_between_objects_{name}": float(np.mean(values))
            for name, values in sums.items()}


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def _format_cell(sigma: float, radius: float, report: Mapping[str, object]) -> str:
    head = f"  sigma={sigma:g} r={radius:g}: n={report['n_objects']}"
    delta = report.get("delta_full_minus_zero")
    if delta is None:
        return f"{head} (empty subset)"
    return (
        f"{head} full={report['top1_full']:.4f} zero={report['top1_zero']:.4f} "
        f"near={report['top1_nearest']:.4f} shuf={report['top1_shuffled']:.4f} "
        f"delta={delta['value']:+.4f} [{delta['low']:+.4f}, {delta['high']:+.4f}]"
    )


def _loader(config: Mapping, pairs: Sequence, sigma: float, workers: int) -> DataLoader:
    dataset = build_eval_dataset(config, pairs, sigma)
    return DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=workers, collate_fn=collate,
        shuffle=False, pin_memory=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/alignformer.yaml"))
    parser.add_argument(
        "--full-checkpoint", type=Path, default=Path("outputs/alignformer/stage1/best.pth")
    )
    parser.add_argument(
        "--zero-checkpoint",
        type=Path,
        default=Path("outputs/alignformer/stage1_zero_embeddings/best.pth"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/alignformer/association_diagnostic.json")
    )
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--max-pairs", type=int, default=None, help="limit validation pairs, for smoke tests"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = yaml.safe_load(args.config.read_text())
    _, val_pairs, _, val_scenarios = build_pair_split(config)
    if args.max_pairs is not None:
        val_pairs = val_pairs[: args.max_pairs]
    pairs = len(val_pairs)
    print(f"{pairs} validation pairs over {len(val_scenarios)} scenarios", flush=True)

    full, full_checkpoint = load_stage1(args.full_checkpoint, device)
    zero, zero_checkpoint = load_stage1(args.zero_checkpoint, device)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    result: Dict[str, object] = {
        "config": str(args.config),
        "full_checkpoint": {"path": str(args.full_checkpoint), "epoch": full_checkpoint["epoch"]},
        "zero_checkpoint": {"path": str(args.zero_checkpoint), "epoch": zero_checkpoint["epoch"]},
        "val_pairs": pairs,
        "val_scenarios": val_scenarios,
        "ambiguity_radii_m": list(AMBIGUITY_RADII),
        "sigmas_m": list(SIGMAS),
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "unit": "ego-CAV pair",
                      "confidence": CONFIDENCE},
        "cells": {},
    }

    for sigma in SIGMAS:
        loader = _loader(config, val_pairs, sigma, args.num_workers)
        records = collect_records(full, zero, loader, device, AMBIGUITY_RADII)
        overall = {
            "n_objects": int(records["pair"].size),
            "top1_full": float(records["correct_full"].mean()),
            "top1_zero": float(records["correct_zero"].mean()),
            "top1_nearest": float(records["correct_nearest"].mean()),
            "top1_shuffled": float(records["correct_shuffled"].mean()),
        }
        cell: Dict[str, object] = {"all_objects": overall, "by_radius": {}}
        for index, radius in enumerate(AMBIGUITY_RADII):
            report = subset_report(records, index, pairs, rng)
            cell["by_radius"][f"r_{radius:g}m"] = report
            print(_format_cell(sigma, radius, report), flush=True)
        result["cells"][f"sigma_{sigma:g}m"] = cell

    probe_loader = _loader(config, val_pairs, 0.5, args.num_workers)
    print("probing embedding discriminability ...", flush=True)
    result["probe_similarity"] = probe_similarity(full, probe_loader, device)
    print("probing gradients ...", flush=True)
    result["probe_gradients"] = probe_gradients(full, probe_loader, device, config)
    result["probe_embedding_spread"] = probe_embedding_spread(full, probe_loader, device, config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
