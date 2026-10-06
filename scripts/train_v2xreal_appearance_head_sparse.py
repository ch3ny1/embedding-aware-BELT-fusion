"""Train the cross-agent appearance head TOWARD the sparse pairs, select on
a held-out slice of train, and judge once on val.

Why
---
The first head (``train_v2xreal_appearance_head.py``) cleared the signal bar
on the probe (0.83-0.85) but was at chance on pairs sharing one or two
annotated objects (0.537, n 374 over all of val), the bucket where the
matcher loses. The cache diagnostic (2026-10-05) says why: those anchors
are far (median 51 m train / 61 m val, 41-49 % beyond 70 m), zero-shot is
at chance there in train too, and the train split holds only 280 such
ambiguous objects. Range, not the bucket, is the difficulty, and dense
pairs carry 17,000 far anchors. So: weight the draws toward far anchors
and sparse pairs, multiply the positives with the partner agent's
neighbouring cached frames, regularize, and pick the configuration and
epoch on held-out TRAIN scenarios by their far-range AUC.

THE DECISION RULE, FIXED BEFORE THE FIRST NUMBER WAS READ
---------------------------------------------------------
1. **Selection** is on the last fifth of train scenarios (held out from
   training), metric = mean ambiguous AUC over the 40-70 m and 70 m+ range
   bands. Val is never used to select.
2. **Signal, where it pays.** The selected head's ambiguous AUC on val
   pairs sharing one or two annotated objects, over EVERY val pair
   (n ~374), is at least **0.80**.
3. **Guard.** Its AUC on pairs sharing three or more stays at least
   **0.80** (the previous head's 0.857 is not traded away).

Usage::

    python -u scripts/train_v2xreal_appearance_head_sparse.py \\
        --train-dir .../alignformer_v2xreal_appearance/train --val-dir .../val \\
        --val-root /media/chenyi/basement2/dataset/v2x-real/val \\
        --checkpoint outputs/v2xreal/appearance_head_sparse.pth \\
        --output outputs/v2xreal/appearance_head_sparse_val_result.json
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_v2xreal_colour_separability import BAR_AUC, enumerate_pairs, refuse_test_split  # noqa: E402
from train_v2xreal_appearance_head import (  # noqa: E402
    ZERO_SHOT_NAMES,
    head_embeddings,
    load_frames,
    pairs_from_frames,
    sampled_pairs,
    score_pairs,
    train_head,
    zero_shot_embeddings,
)
from embedding_aware_belt_fusion.alignformer.appearance_head import (  # noqa: E402
    MAX_HARD_NEGATIVES,
    TEMPERATURE,
    AgentFrame,
    AppearanceHead,
    build_pairs,
    feature_dim,
    row_weights,
)

HOLDOUT_FRACTION = 0.2
FAR_FROM_M = 40.0
SELECTION_BANDS = ("40-70m", "70m+")
EPOCHS = 20
BATCH = 512
LEARNING_RATE = 1e-3


class Config(NamedTuple):
    name: str
    far_weight: float
    sparse_weight: float
    offsets: Tuple[int, ...]
    dropout: float


CONFIGS: Tuple[Config, ...] = (
    Config("baseline", 0.0, 0.0, (0,), 0.0),
    Config("far4", 4.0, 0.0, (0,), 0.0),
    Config("far16", 16.0, 0.0, (0,), 0.0),
    Config("far16_sparse10", 16.0, 10.0, (0,), 0.0),
    Config("far16_sparse10_offsets", 16.0, 10.0, (-1, 0, 1), 0.0),
    Config("far16_sparse10_offsets_drop", 16.0, 10.0, (-1, 0, 1), 0.2),
)


def split_scenarios(frames: Sequence[AgentFrame], fraction: float = HOLDOUT_FRACTION) -> Tuple[List[AgentFrame], List[AgentFrame]]:
    """Deterministic: the last ``fraction`` of sorted scenario names is held out whole."""
    scenarios = sorted({f.scenario for f in frames})
    held = set(scenarios[len(scenarios) - max(1, int(round(len(scenarios) * fraction))) :])
    return [f for f in frames if f.scenario not in held], [f for f in frames if f.scenario in held]


def selection_metric(report: Dict, name: str = "head") -> Optional[float]:
    """Mean ambiguous AUC over the far range bands; None if a band is empty."""
    values = [report["ambiguous_by_range"].get(band, {}).get(name, {}).get("auc_paired") for band in SELECTION_BANDS]
    return None if any(v is None for v in values) else float(np.mean(values))


def pick(results: Sequence[Dict]) -> Dict:
    """The configuration with the best held-out selection metric (ties: first)."""
    scored = [r for r in results if r["best_holdout_metric"] is not None]
    return max(scored, key=lambda r: r["best_holdout_metric"]) if scored else results[0]


def verdict(val_report: Dict, name: str = "head") -> Dict:
    sparse = val_report["ambiguous_by_shared"]["shared_1_2"][name]["auc_paired"] or 0.0
    dense = val_report["ambiguous_by_shared"]["shared_3plus"][name]["auc_paired"] or 0.0
    return {"sparse_ambiguous_auc": sparse, "dense_ambiguous_auc": dense,
            "bars": {"sparse_signal": sparse >= BAR_AUC, "dense_guard": dense >= BAR_AUC},
            "worth_building": sparse >= BAR_AUC and dense >= BAR_AUC}


def fit_config(config: Config, train_frames, holdout_frames, holdout_pairs, args, device) -> Dict:
    """Train one configuration; keep the epoch with the best held-out far-range AUC."""
    pairs = build_pairs(train_frames, args.hard_negatives, offsets=config.offsets)
    weights = row_weights(pairs.shared_counts, pairs.anchor_ranges, config.far_weight, config.sparse_weight, FAR_FROM_M)
    state = {"best": None, "best_epoch": None, "best_state": None, "curve": []}

    def on_epoch(epoch: int, head: AppearanceHead, loss: float) -> None:
        report = score_pairs(holdout_frames, holdout_pairs, {"head": head_embeddings(holdout_frames, head, device)}, random.Random(args.seed))
        metric = selection_metric(report)
        sparse = report["ambiguous_by_shared"]["shared_1_2"]["head"]["auc_paired"]
        state["curve"].append({"epoch": epoch, "loss": loss, "holdout_far_auc": metric, "holdout_sparse_auc": sparse})
        print(f"  [{config.name}] epoch {epoch}: loss {loss:.4f}  holdout far {metric if metric is None else round(metric, 3)}  sparse {sparse if sparse is None else round(sparse, 3)}", flush=True)
        if metric is not None and (state["best"] is None or metric > state["best"]):
            state.update(best=metric, best_epoch=epoch, best_state=copy.deepcopy(head.state_dict()))

    head, _ = train_head(pairs, epochs=args.epochs, batch_size=args.batch, lr=args.lr, temperature=args.temperature,
                         seed=args.seed, device=device, on_epoch=on_epoch, sample_weights=weights, dropout=config.dropout)
    if state["best_state"] is not None:
        head.load_state_dict(state["best_state"])
    return {"config": config._asdict(), "rows": len(pairs.rows), "sparse_rows": int((pairs.shared_counts <= 2).sum()),
            "far_rows": int((pairs.anchor_ranges > FAR_FROM_M).sum()), "best_holdout_metric": state["best"],
            "best_epoch": state["best_epoch"], "curve": state["curve"], "head": head}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--val-pairs", type=int, default=400, help="the probe's sample, reported beside every-pair")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--hard-negatives", type=int, default=MAX_HARD_NEGATIVES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--configs", nargs="*", default=[c.name for c in CONFIGS])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    refuse_test_split(args.val_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.time()
    all_train, val_frames = load_frames(args.train_dir), load_frames(args.val_dir)
    train_frames, holdout_frames = split_scenarios(all_train)
    holdout_pairs = pairs_from_frames(holdout_frames, 10**9, args.seed)
    print(f"{len(train_frames)} train / {len(holdout_frames)} held-out agent-frames ({len(holdout_pairs)} held-out pairs), {len(val_frames)} val", flush=True)
    configs = [c for c in CONFIGS if c.name in args.configs]
    results = [fit_config(c, train_frames, holdout_frames, holdout_pairs, args, device) for c in configs]
    chosen = pick(results)
    print(f"selected {chosen['config']['name']} at epoch {chosen['best_epoch']} (held-out far AUC {chosen['best_holdout_metric']:.3f})", flush=True)

    every_val = enumerate_pairs(args.val_root)
    probe_val = sampled_pairs(args.val_root, args.val_pairs, args.seed)
    tables = {"head": head_embeddings(val_frames, chosen["head"], device), **zero_shot_embeddings(val_frames, feature_dim(val_frames) // 2)}
    val_all = score_pairs(val_frames, every_val, tables, random.Random(args.seed))
    val_probe = score_pairs(val_frames, probe_val, tables, random.Random(args.seed))
    torch.save({"state_dict": chosen["head"].state_dict(), "in_dim": feature_dim(val_frames), "config": chosen["config"]}, args.checkpoint)
    report = {
        "method": "v2xreal_appearance_head_trained_toward_sparse_pairs",
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "pre_registered": {"selection": "held-out train scenarios, mean ambiguous AUC over " + " and ".join(SELECTION_BANDS),
                           "sparse_signal_auc": BAR_AUC, "dense_guard_auc": BAR_AUC, "holdout_fraction": HOLDOUT_FRACTION},
        "candidates": [{k: v for k, v in r.items() if k != "head"} for r in results],
        "selected": chosen["config"]["name"], "selected_epoch": chosen["best_epoch"],
        "val_every_pair": val_all, "val_probe_pairs": val_probe, "minutes": (time.time() - started) / 60.0,
    }
    report["verdict"] = verdict(val_all)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["verdict"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
