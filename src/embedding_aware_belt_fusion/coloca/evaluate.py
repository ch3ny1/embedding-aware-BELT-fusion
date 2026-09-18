"""Evaluate a trained CoLoca-QuA model on the official OPV2V test split.

Reports the paper's Table II metrics (Section IV-C) at each evaluated noise
level, alongside the uncorrected input error so the improvement attributable to
the model is unambiguous.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from embedding_aware_belt_fusion.coloca.dataset import OPV2VPoseErrorDataset
from embedding_aware_belt_fusion.coloca.metrics import MetricAccumulator
from embedding_aware_belt_fusion.coloca.train import build_model, move_to_device


def evaluate_noise_level(
    model: torch.nn.Module,
    config: Mapping[str, Any],
    xy_std: float,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Evaluate at one position-noise sigma, plus the no-correction control."""
    data_cfg, noise_cfg, train_cfg = config["data"], config["noise"], config["train"]
    thresholds = config["eval"]["thresholds"]

    dataset = OPV2VPoseErrorDataset(
        root_dir=data_cfg["test_root"],
        preprocess_params=config["preprocess"],
        cache_path=Path(data_cfg["cache_dir"]) / "test_pairs.npz",
        xy_std=xy_std,
        yaw_std_deg=noise_cfg["yaw_std_deg"],
        comm_range_m=data_cfg["comm_range_m"],
        train=False,
        seed=noise_cfg["eval_seed"],
        pcd_cache_root=data_cfg.get("test_pcd_cache"),
    )
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        shuffle=False,
        pin_memory=True,
        collate_fn=dataset.collate,
    )

    model.eval()
    predicted = MetricAccumulator(thresholds)
    # Control: predicting zero correction, i.e. trusting the noisy pose as-is.
    uncorrected = MetricAccumulator(thresholds)

    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            with torch.autocast("cuda", enabled=bool(train_cfg["amp"])):
                prediction = model(batch)
            target = batch["pose_error"]
            predicted.update(prediction.float(), target)
            uncorrected.update(torch.zeros_like(target), target)

    return {
        "coloca_qua": predicted.compute(),
        "no_correction": uncorrected.compute(),
    }


# Paper Table II(A), V2Xset, V2V block. Reported on V2XSet, not OPV2V, so these
# are a reference point rather than a like-for-like target -- see
# docs/coloca_qua_baseline.md for what differs.
PAPER_V2V_REFERENCE = {
    "2.0 m": {"mae_m": 0.23, "rmse_m": 0.29, "below_1.0m": 0.9943,
              "below_0.8m": 0.9815, "below_0.5m": 0.9361},
    "1.0 m": {"mae_m": 0.18, "rmse_m": 0.22, "below_1.0m": 0.9973,
              "below_0.8m": 0.9948, "below_0.5m": 0.9777},
}


def format_table(results: Mapping[str, Any], thresholds: list[float]) -> str:
    header = ["sigma", "method", "MAE (m)", "RMSE (m)"]
    header += [f"err < {t} m" for t in thresholds] + ["yaw MAE (deg)", "n"]
    lines = [" | ".join(f"{h:>13}" for h in header), "-" * (16 * len(header))]
    for sigma, methods in results.items():
        for method, metrics in methods.items():
            row = [f"{sigma}", method, f"{metrics['mae_m']:.3f}", f"{metrics['rmse_m']:.3f}"]
            row += [f"{metrics[f'below_{t}m'] * 100:.2f}%" for t in thresholds]
            row += [f"{metrics['yaw_mae_deg']:.3f}", f"{int(metrics['count'])}"]
            lines.append(" | ".join(f"{cell:>13}" for cell in row))
        reference = PAPER_V2V_REFERENCE.get(sigma)
        if reference:
            row = [f"{sigma}", "paper(V2XSet)", f"{reference['mae_m']:.3f}",
                   f"{reference['rmse_m']:.3f}"]
            row += [f"{reference[f'below_{t}m'] * 100:.2f}%" for t in thresholds]
            row += ["n/a", "n/a"]
            lines.append(" | ".join(f"{cell:>13}" for cell in row))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CoLoca-QuA on OPV2V test")
    parser.add_argument("--config", default="configs/coloca_qua_opv2v.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None, help="where to write the metrics json")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("a CUDA device is required to evaluate CoLoca-QuA")

    model = build_model(config, device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"loaded {args.checkpoint}")

    results = {
        f"{xy_std} m": evaluate_noise_level(model, config, xy_std, device)
        for xy_std in config["noise"]["eval_xy_std"]
    }

    print("\nOPV2V test split, V2V cooperative localization")
    print(format_table(results, config["eval"]["thresholds"]))

    output_path = Path(args.output or Path(args.checkpoint).parent / "test_metrics.json")
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {output_path}")


if __name__ == "__main__":
    main()
