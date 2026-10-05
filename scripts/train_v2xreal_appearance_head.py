"""Train a projection head for cross-agent viewpoint invariance on frozen
DINOv2 features, and judge it with the separability probe on V2X-Real val.

THE DECISION RULE, FIXED BEFORE THE FIRST NUMBER WAS READ
---------------------------------------------------------
Zero-shot DINOv2-base scored AUC 0.681 on the probe's ambiguous subset
(same vehicle from two agents against its nearest neighbour within 8 m),
against the pre-registered bar of 0.80; colour scored 0.632. The head is
worth carrying into the matcher (stage 1, as the per-object embedding) if
and only if, on the official validation split, on the probe's own 400
pairs and candidates:

1. **Signal.** The trained head's ambiguous-subset AUC is at least **0.80**.
2. **Control.** A head trained identically on shuffled identities (positive
   = another shared vehicle of the partner frame) stays within **0.05 of
   0.5**: otherwise the pipeline leaks and the run is void.

Fixed epoch count, no early stopping on val: val is read once, at the end.
The per-epoch val curve is recorded for the write-up, not for selection.
Coverage is unchanged from the zero-shot probe (same candidates), so the
coverage bars are inherited, not re-tested.

Usage::

    python -u scripts/train_v2xreal_appearance_head.py \\
        --train-dir /media/chenyi/basement2/cache/alignformer_v2xreal_appearance/train \\
        --val-dir /media/chenyi/basement2/cache/alignformer_v2xreal_appearance/val \\
        --val-root /media/chenyi/basement2/dataset/v2x-real/val \\
        --checkpoint outputs/v2xreal/appearance_head_base.pth \\
        --output outputs/v2xreal/appearance_head_base_val_result.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_v2xreal_colour_separability import (  # noqa: E402
    BAR_AUC,
    SHARED_BUCKETS,
    Pair,
    _new_scores,
    _range_bucket,
    _score_object,
    _summaries,
    enumerate_pairs,
    refuse_test_split,
    shared_bucket,
)
from embedding_aware_belt_fusion.alignformer.appearance_head import (  # noqa: E402
    MAX_HARD_NEGATIVES,
    TEMPERATURE,
    AgentFrame,
    AppearanceHead,
    PairSet,
    build_pairs,
    frame_key,
    info_nce,
    load_frame,
)

CONTROL_TOLERANCE = 0.05
EPOCHS = 20
BATCH = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
VAL_EVERY = 5
TRAIN_PROBE_PAIRS = 400
ZERO_SHOT_NAMES = ("zero_shot_cls", "zero_shot_patch_mean")

Embeddings = Dict[str, np.ndarray]  # frame key -> (N, D)


def load_frames(directory: Path) -> List[AgentFrame]:
    return [load_frame(p) for p in sorted(directory.glob("*.npz"))]


def sampled_pairs(root: Path, count: int, seed: int) -> List[Pair]:
    """The probe's own sample: same enumeration, same seed, same draw."""
    pairs = enumerate_pairs(root)
    rng = random.Random(seed)
    return pairs if len(pairs) <= count else rng.sample(pairs, count)


def pairs_from_frames(frames: Sequence[AgentFrame], count: int, seed: int) -> List[Pair]:
    """Ordered agent pairs present in a cache (for the train-split fit check)."""
    by_stamp: Dict[Tuple[str, str], List[str]] = {}
    for frame in frames:
        by_stamp.setdefault((frame.scenario, frame.timestamp), []).append(frame.agent)
    pairs = [Pair(s, t, e, c) for (s, t), agents in sorted(by_stamp.items())
             for e in sorted(agents) for c in sorted(agents) if e != c]
    rng = random.Random(seed)
    return pairs if len(pairs) <= count else rng.sample(pairs, count)


def zero_shot_embeddings(frames: Sequence[AgentFrame], split_at: int) -> Dict[str, Embeddings]:
    """The two cached descriptors, un-projected: the probe's numbers must reappear."""
    keys = [frame_key(f.scenario, f.agent, f.timestamp) for f in frames]
    return {
        ZERO_SHOT_NAMES[0]: {k: f.features[:, :split_at] for k, f in zip(keys, frames)},
        ZERO_SHOT_NAMES[1]: {k: f.features[:, split_at:] for k, f in zip(keys, frames)},
    }


def head_embeddings(frames: Sequence[AgentFrame], head: AppearanceHead, device: str) -> Embeddings:
    head.eval()
    out: Embeddings = {}
    with torch.no_grad():
        for frame in frames:
            if len(frame.vids) == 0:
                out[frame_key(frame.scenario, frame.agent, frame.timestamp)] = np.zeros((0, 1), np.float32)
                continue
            emb = head(torch.from_numpy(frame.features).to(device)).cpu().numpy()
            out[frame_key(frame.scenario, frame.agent, frame.timestamp)] = emb
    return out


def _views(frame: AgentFrame, embeddings: Dict[str, Embeddings]) -> Dict[str, Dict]:
    key = frame_key(frame.scenario, frame.agent, frame.timestamp)
    return {vid: {name: table[key][i] for name, table in embeddings.items()} for i, vid in enumerate(frame.vids)}


def score_pairs(frames: Sequence[AgentFrame], pairs: Sequence[Pair], embeddings: Dict[str, Embeddings],
                rng: random.Random) -> Dict:
    """The probe's scoring on cached frames: partner vs nearest distractor, per name."""
    names = tuple(embeddings)
    by_key = {frame_key(f.scenario, f.agent, f.timestamp): f for f in frames}
    acc = {
        "scores": _new_scores(names), "ambiguous": _new_scores(names), "same_agent": _new_scores(names),
        "shuffled": _new_scores(names), "by_shared": {b: _new_scores(names) for b in SHARED_BUCKETS},
        "ambiguous_by_shared": {b: _new_scores(names) for b in SHARED_BUCKETS}, "ambiguous_by_range": {},
    }
    scored_pairs = 0
    for pair in pairs:
        ego = by_key.get(frame_key(pair.scenario, pair.ego, pair.timestamp))
        cav = by_key.get(frame_key(pair.scenario, pair.cav, pair.timestamp))
        if ego is None or cav is None:
            continue
        common = sorted(set(ego.vids) & set(cav.vids))
        if not common:
            continue
        scored_pairs += 1
        bucket = shared_bucket(len(set(ego.gt_vids) & set(cav.gt_vids)))
        ego_views, cav_views = _views(ego, embeddings), _views(cav, embeddings)
        centres = {vid: cav.centre_xy[i] for i, vid in enumerate(cav.vids)}
        for vid in common:
            range_bucket = _range_bucket(float(ego.range_m[ego.vids.index(vid)]))
            _score_object(vid, bucket, range_bucket, ego_views, cav_views, centres, acc, rng, names)
    return {
        "pairs_scored": scored_pairs,
        "all_objects": _summaries(acc["scores"], names),
        "ambiguous": _summaries(acc["ambiguous"], names),
        "ambiguous_by_shared": {b: _summaries(acc["ambiguous_by_shared"][b], names) for b in SHARED_BUCKETS},
        "ambiguous_by_range": {k: _summaries(v, names) for k, v in sorted(acc["ambiguous_by_range"].items())},
        "shuffled_control": _summaries(acc["shuffled"], names),
    }


def train_head(pairs: PairSet, *, epochs: int, batch_size: int, lr: float, temperature: float, seed: int,
               device: str, on_epoch: Optional[Callable[[int, AppearanceHead, float], None]] = None) -> Tuple[AppearanceHead, List[float]]:
    """Fixed-epoch InfoNCE training; ``on_epoch(epoch, head, mean_loss)`` after each epoch."""
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    head = AppearanceHead(in_dim=pairs.anchors.shape[1]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    anchors, positives = torch.from_numpy(pairs.anchors), torch.from_numpy(pairs.positives)
    hard, mask = torch.from_numpy(pairs.hard_negatives), torch.from_numpy(pairs.hard_mask)
    count, losses = anchors.shape[0], []
    for epoch in range(1, epochs + 1):
        head.train()
        order, total = generator.permutation(count), 0.0
        for start in range(0, count, batch_size):
            idx = torch.from_numpy(order[start : start + batch_size])
            a, p = head(anchors[idx].to(device)), head(positives[idx].to(device))
            h = hard[idx].to(device)
            hn = head(h.reshape(-1, h.shape[-1])).reshape(h.shape[0], h.shape[1], -1)
            loss = info_nce(a, p, hn, mask[idx].to(device), temperature)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss) * len(idx)
        losses.append(total / max(count, 1))
        if on_epoch is not None:
            on_epoch(epoch, head, losses[-1])
    return head, losses


def verdict(head_report: Dict, control_report: Dict) -> Dict:
    signal = head_report["ambiguous"]["head"]["auc_paired"] or 0.0
    control = control_report["ambiguous"]["head"]["auc_paired"]
    control_valid = control is not None and abs(control - 0.5) <= CONTROL_TOLERANCE
    return {"ambiguous_auc": signal, "control_ambiguous_auc": control, "bars": {"signal": signal >= BAR_AUC},
            "control_valid": control_valid, "worth_building": control_valid and signal >= BAR_AUC}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True, help="dataset val split, for the probe's pair sample")
    parser.add_argument("--val-pairs", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--hard-negatives", type=int, default=MAX_HARD_NEGATIVES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _fit(label: str, pairs: PairSet, args, val_frames, val_pairs, device) -> Tuple[AppearanceHead, Dict]:
    curve: List[Dict] = []

    def on_epoch(epoch: int, head: AppearanceHead, loss: float) -> None:
        entry = {"epoch": epoch, "loss": loss}
        if epoch % VAL_EVERY == 0 or epoch == args.epochs:
            report = score_pairs(val_frames, val_pairs, {"head": head_embeddings(val_frames, head, device)}, random.Random(args.seed))
            entry["val_ambiguous_auc"] = report["ambiguous"]["head"]["auc_paired"]
        print(f"  [{label}] epoch {epoch}: loss {loss:.4f}" + (f"  val ambiguous AUC {entry['val_ambiguous_auc']:.3f}" if "val_ambiguous_auc" in entry else ""), flush=True)
        curve.append(entry)

    head, _ = train_head(pairs, epochs=args.epochs, batch_size=args.batch, lr=args.lr, temperature=args.temperature,
                         seed=args.seed, device=device, on_epoch=on_epoch)
    return head, {"pairs": len(pairs.rows), "curve": curve}


def main(argv=None) -> None:
    args = parse_args(argv)
    refuse_test_split(args.val_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.time()
    train_frames, val_frames = load_frames(args.train_dir), load_frames(args.val_dir)
    val_pairs = sampled_pairs(args.val_root, args.val_pairs, args.seed)
    print(f"{len(train_frames)} train agent-frames, {len(val_frames)} val agent-frames, {len(val_pairs)} val pairs", flush=True)
    honest = build_pairs(train_frames, args.hard_negatives)
    control = build_pairs(train_frames, args.hard_negatives, shuffle_identities=np.random.default_rng(args.seed))
    print(f"{len(honest.rows)} training pairs ({honest.hard_mask.mean():.2f} of hard-negative slots filled)", flush=True)

    half = train_frames[0].features.shape[1] // 2
    zero_shot = score_pairs(val_frames, val_pairs, zero_shot_embeddings(val_frames, half), random.Random(args.seed))
    print("zero-shot check (val ambiguous AUC):", {n: round(zero_shot["ambiguous"][n]["auc_paired"], 3) for n in ZERO_SHOT_NAMES}, flush=True)

    head, fit = _fit("head", honest, args, val_frames, val_pairs, device)
    control_head, control_fit = _fit("control", control, args, val_frames, val_pairs, device)
    rng = random.Random(args.seed)
    head_report = score_pairs(val_frames, val_pairs, {"head": head_embeddings(val_frames, head, device)}, rng)
    control_report = score_pairs(val_frames, val_pairs, {"head": head_embeddings(val_frames, control_head, device)}, rng)
    train_probe = pairs_from_frames(train_frames, TRAIN_PROBE_PAIRS, args.seed)
    train_report = score_pairs(train_frames, train_probe, {"head": head_embeddings(train_frames, head, device)}, rng)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "in_dim": honest.anchors.shape[1]}, args.checkpoint)
    report = {
        "method": "v2xreal_appearance_head_on_frozen_dinov2",
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "pre_registered_bars": {"signal_auc": BAR_AUC, "control_tolerance": CONTROL_TOLERANCE},
        "train_agent_frames": len(train_frames), "val_agent_frames": len(val_frames),
        "zero_shot_val": zero_shot, "fit": fit, "control_fit": control_fit,
        "val": head_report, "control_val": control_report, "train_probe": train_report,
        "minutes": (time.time() - started) / 60.0,
    }
    report["verdict"] = verdict(head_report, control_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["verdict"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
