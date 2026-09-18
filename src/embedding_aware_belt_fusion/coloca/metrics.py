"""Loss and evaluation metrics for CoLoca-QuA.

Loss follows Eq. (14): a per-component weighted MSE with the paper's chosen
weights ``(alpha_x, alpha_y, alpha_psi) = (2, 2, 1)`` (Section IV-F, setting S3).

Metrics follow Section IV-C.  MAE and RMSE are reported in metres over the
*planar translation* residual, and the threshold metrics report the fraction of
samples whose residual falls below 1.0 m, 0.8 m and 0.5 m.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

# Paper Section IV-F: setting S3 gave the lowest RMSE and is used for the
# final loss formulation.
DEFAULT_LOSS_WEIGHTS = (2.0, 2.0, 1.0)
# Paper Section IV-C.
DEFAULT_ERROR_THRESHOLDS = (1.0, 0.8, 0.5)


def pose_error_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: Sequence[float] = DEFAULT_LOSS_WEIGHTS,
) -> torch.Tensor:
    """Weighted MSE over ``[dx, dy, dpsi]`` (Eq. 14)."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction shape {tuple(prediction.shape)} != target shape {tuple(target.shape)}"
        )
    weight_tensor = torch.as_tensor(weights, dtype=prediction.dtype, device=prediction.device)
    squared_error = (prediction - target) ** 2
    return (squared_error * weight_tensor).sum(dim=-1).mean()


def translation_residual(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample Euclidean norm of the ``(dx, dy)`` residual, in metres."""
    return torch.linalg.norm(prediction[:, :2] - target[:, :2], dim=-1)


def heading_residual(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample absolute yaw residual, in degrees."""
    return (prediction[:, 2] - target[:, 2]).abs()


def localization_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    thresholds: Sequence[float] = DEFAULT_ERROR_THRESHOLDS,
) -> dict[str, float]:
    """Compute the paper's localization metrics for one batch of predictions.

    Returns MAE (m), RMSE (m), the mean absolute yaw error (deg), and the
    fraction of samples below each translation-error threshold.
    """
    residual = translation_residual(prediction, target)
    metrics = {
        "mae_m": residual.mean().item(),
        "rmse_m": residual.pow(2).mean().sqrt().item(),
        "yaw_mae_deg": heading_residual(prediction, target).mean().item(),
        "count": float(residual.numel()),
    }
    for threshold in thresholds:
        metrics[f"below_{threshold}m"] = (residual < threshold).float().mean().item()
    return metrics


class MetricAccumulator:
    """Streams per-batch residuals into dataset-level metrics.

    Averaging per-batch metrics would bias the result when batches differ in
    size, so the raw residuals are accumulated instead.
    """

    def __init__(self, thresholds: Sequence[float] = DEFAULT_ERROR_THRESHOLDS) -> None:
        self.thresholds = tuple(thresholds)
        self._translation: list[torch.Tensor] = []
        self._heading: list[torch.Tensor] = []

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        self._translation.append(translation_residual(prediction, target).detach().cpu())
        self._heading.append(heading_residual(prediction, target).detach().cpu())

    def compute(self) -> dict[str, float]:
        if not self._translation:
            raise RuntimeError("no predictions were accumulated")
        translation = torch.cat(self._translation)
        heading = torch.cat(self._heading)

        metrics = {
            "mae_m": translation.mean().item(),
            "rmse_m": translation.pow(2).mean().sqrt().item(),
            "yaw_mae_deg": heading.mean().item(),
            "yaw_rmse_deg": heading.pow(2).mean().sqrt().item(),
            "count": float(translation.numel()),
        }
        for threshold in self.thresholds:
            metrics[f"below_{threshold}m"] = (translation < threshold).float().mean().item()
        return metrics


def format_metrics(metrics: Mapping[str, float]) -> str:
    """Render a metrics dict as a compact single line for logs."""
    ordered = ["mae_m", "rmse_m", "yaw_mae_deg"]
    parts = [f"{key}={metrics[key]:.4f}" for key in ordered if key in metrics]
    parts += [
        f"{key}={metrics[key] * 100:.2f}%" for key in sorted(metrics) if key.startswith("below_")
    ]
    if "count" in metrics:
        parts.append(f"n={int(metrics['count'])}")
    return " ".join(parts)
