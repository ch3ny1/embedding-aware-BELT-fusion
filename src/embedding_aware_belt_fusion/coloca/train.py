"""Two-stage trainer for the CoLoca-QuA baseline on OPV2V.

Follows the paper's Section IV-B protocol, minus stage 1 (the cooperative
detection backbone), which is supplied as a pretrained F-Cooper checkpoint:

    stage 2  fix the backbone, train the localization module
    stage 3  unfix the backbone and fine-tune the whole model end to end
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from embedding_aware_belt_fusion.coloca.backbone import PointPillarsBEVEncoder, bev_feature_size
from embedding_aware_belt_fusion.coloca.dataset import OPV2VPoseErrorDataset
from embedding_aware_belt_fusion.coloca.index import pair_cache_path
from embedding_aware_belt_fusion.coloca.metrics import (
    MetricAccumulator,
    format_metrics,
    pose_error_loss,
)
from embedding_aware_belt_fusion.coloca.model import ColocaQuANet

# Fraction of each stage spent linearly warming the learning rate up to its peak.
WARMUP_FRACTION = 0.05


def split_scenarios(
    scenarios: list[str], val_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Split scenario names into train/val.

    Splitting by scenario rather than by frame matters: consecutive OPV2V
    frames within a scenario are near-duplicates, so a frame-level split would
    leak the validation set into training.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    shuffled = list(scenarios)
    np.random.default_rng(seed).shuffle(shuffled)
    num_val = max(1, round(len(shuffled) * val_fraction)) if val_fraction else 0
    return sorted(shuffled[num_val:]), sorted(shuffled[:num_val])


def build_datasets(config: Mapping[str, Any]) -> tuple[OPV2VPoseErrorDataset, OPV2VPoseErrorDataset]:
    data_cfg, noise_cfg = config["data"], config["noise"]
    cache_dir = Path(data_cfg["cache_dir"])

    common = {
        "root_dir": data_cfg["train_root"],
        "preprocess_params": config["preprocess"],
        "cache_path": pair_cache_path(cache_dir, "train", data_cfg["comm_range_m"]),
        "yaw_std_deg": noise_cfg["yaw_std_deg"],
        "comm_range_m": data_cfg["comm_range_m"],
        "pcd_cache_root": data_cfg.get("train_pcd_cache"),
    }

    probe = OPV2VPoseErrorDataset(**common, xy_std=noise_cfg["train_xy_std"], train=False)
    train_scenarios, val_scenarios = split_scenarios(
        probe.scenario_names(), data_cfg["val_scenario_fraction"], data_cfg["split_seed"]
    )

    train_set = OPV2VPoseErrorDataset(
        **common,
        xy_std=noise_cfg["train_xy_std"],
        train=True,
        seed=config["train"]["seed"],
        scenarios=train_scenarios,
    )
    val_set = OPV2VPoseErrorDataset(
        **common,
        xy_std=noise_cfg["train_xy_std"],
        train=False,
        seed=noise_cfg["eval_seed"],
        scenarios=val_scenarios,
    )
    return train_set, val_set


def build_model(config: Mapping[str, Any], device: torch.device) -> ColocaQuANet:
    backbone_cfg = config["backbone"]
    encoder = PointPillarsBEVEncoder(backbone_cfg["args"])
    encoder.load_fcooper_checkpoint(backbone_cfg["checkpoint"])
    feature_size = bev_feature_size(
        config["preprocess"]["cav_lidar_range"], config["preprocess"]["args"]["voxel_size"]
    )
    return ColocaQuANet(encoder, feature_size=feature_size, **config["model"]).to(device)


def move_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: (
            {name: tensor.to(device, non_blocking=True) for name, tensor in value.items()}
            if isinstance(value, dict)
            else value.to(device, non_blocking=True)
            if torch.is_tensor(value)
            else value
        )
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate(
    model: ColocaQuANet,
    loader: DataLoader,
    device: torch.device,
    thresholds: list[float],
    amp: bool,
) -> dict[str, float]:
    model.eval()
    accumulator = MetricAccumulator(thresholds)
    for batch in loader:
        batch = move_to_device(batch, device)
        with torch.autocast("cuda", enabled=amp):
            prediction = model(batch)
        accumulator.update(prediction.float(), batch["pose_error"])
    return accumulator.compute()


def run_stage(
    stage: str,
    model: ColocaQuANet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Mapping[str, Any],
    device: torch.device,
    epochs: int,
    lr: float,
    output_dir: Path,
    history: list[dict[str, Any]],
    best: dict[str, float],
) -> None:
    train_cfg = config["train"]
    loss_weights = config["loss"]["weights"]
    thresholds = config["eval"]["thresholds"]
    amp = bool(train_cfg["amp"])

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        parameters, lr=lr, weight_decay=train_cfg["weight_decay"], eps=1e-10
    )
    # Per-step cosine schedule with a short linear warm-up. A ViT trained from
    # scratch is unstable in its first few hundred steps without one.
    total_steps = max(epochs * len(train_loader), 1)
    warmup_steps = max(int(WARMUP_FRACTION * total_steps), 1)

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    trainable = sum(p.numel() for p in parameters)
    print(f"\n=== stage: {stage} | epochs={epochs} lr={lr:g} trainable={trainable / 1e6:.2f}M ===")

    for epoch in range(epochs):
        model.train()
        train_loader.dataset.set_epoch(epoch)
        running_loss, seen, started = 0.0, 0, time.time()

        for step, batch in enumerate(train_loader):
            batch = move_to_device(batch, device)
            with torch.autocast("cuda", enabled=amp):
                prediction = model(batch)
                loss = pose_error_loss(prediction.float(), batch["pose_error"], loss_weights)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, train_cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # per step, not per epoch

            running_loss += loss.item() * prediction.shape[0]
            seen += prediction.shape[0]
            if step % 100 == 0:
                rate = (step + 1) / (time.time() - started)
                print(
                    f"  [{stage} ep{epoch + 1}/{epochs}] step {step}/{len(train_loader)} "
                    f"loss={running_loss / max(seen, 1):.4f} {rate:.2f} it/s",
                    flush=True,
                )

        metrics = evaluate(model, val_loader, device, thresholds, amp)
        epoch_loss = running_loss / max(seen, 1)
        print(
            f"  [{stage} ep{epoch + 1}/{epochs}] train_loss={epoch_loss:.4f} | "
            f"val {format_metrics(metrics)} ({time.time() - started:.0f}s)",
            flush=True,
        )

        history.append({"stage": stage, "epoch": epoch + 1, "train_loss": epoch_loss, **metrics})
        (output_dir / "history.json").write_text(json.dumps(history, indent=2))
        torch.save(model.state_dict(), output_dir / "latest.pth")
        if metrics["mae_m"] < best["mae_m"]:
            best.update({"mae_m": metrics["mae_m"], "stage": stage, "epoch": epoch + 1})
            torch.save(model.state_dict(), output_dir / "best.pth")
            print(f"  -> new best val MAE {metrics['mae_m']:.4f} m", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CoLoca-QuA on OPV2V")
    parser.add_argument("--config", default="configs/coloca_qua_opv2v.yaml")
    parser.add_argument("--output-dir", default=None, help="overrides train.output_dir")
    parser.add_argument("--frozen-epochs", type=int, default=None)
    parser.add_argument("--finetune-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    train_cfg = config["train"]
    for name, value in (
        ("frozen_epochs", args.frozen_epochs),
        ("finetune_epochs", args.finetune_epochs),
        ("batch_size", args.batch_size),
        ("output_dir", args.output_dir),
    ):
        if value is not None:
            train_cfg[name] = value

    torch.manual_seed(train_cfg["seed"])
    np.random.seed(train_cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("a CUDA device is required to train CoLoca-QuA")

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    train_set, val_set = build_datasets(config)
    print(
        f"train pairs: {len(train_set)} over {len(train_set.scenario_names())} scenarios | "
        f"val pairs: {len(val_set)} over {len(val_set.scenario_names())} scenarios"
    )

    loader_kwargs = {
        "batch_size": train_cfg["batch_size"],
        "num_workers": train_cfg["num_workers"],
        "pin_memory": True,
    }
    # The train loader must NOT use persistent workers: each worker holds its own
    # copy of the dataset, so `set_epoch` on the main-process copy would never
    # reach them and every epoch would silently reuse epoch 0's noise draw.
    # Re-forking per epoch propagates it and costs ~2 s against a ~100 s epoch.
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        drop_last=True,
        collate_fn=train_set.collate,
        persistent_workers=False,
        **loader_kwargs,
    )
    # The val loader draws fixed per-sample noise, so persistence is safe there.
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        drop_last=False,
        collate_fn=val_set.collate,
        persistent_workers=train_cfg["num_workers"] > 0,
        **loader_kwargs,
    )

    model = build_model(config, device)
    history: list[dict[str, Any]] = []
    best = {"mae_m": float("inf")}

    # Stage 2: backbone fixed, localization module only.
    model.encoder.freeze()
    run_stage(
        "frozen", model, train_loader, val_loader, config, device,
        train_cfg["frozen_epochs"], train_cfg["lr"], output_dir, history, best,
    )

    # Stage 3: unfix the backbone and fine-tune the entire model.
    if train_cfg["finetune_epochs"] > 0:
        best_state = output_dir / "best.pth"
        if best_state.exists():
            model.load_state_dict(torch.load(best_state, map_location=device))
        model.encoder.unfreeze()
        run_stage(
            "finetune", model, train_loader, val_loader, config, device,
            train_cfg["finetune_epochs"], train_cfg["finetune_lr"], output_dir, history, best,
        )

    print(f"\nbest val MAE {best['mae_m']:.4f} m at {best.get('stage')} epoch {best.get('epoch')}")
    print(f"checkpoints written to {output_dir}")


if __name__ == "__main__":
    main()
