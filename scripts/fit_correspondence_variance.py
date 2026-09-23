"""Fit the per-correspondence disagreement variance model on validation.

Task 18 measured that cross-agent correspondence disagreement is
heteroscedastic in range: matched detections of the same physical object
disagree by 0.230 m / 6.65 deg beyond |x| = 70.4 m against 0.151 m / 3.69 deg
in the near field. Those correspondences enter the Procrustes fit unweighted.
Weighting them by their inverse variance is the statistically correct
estimator; this script is where the variance model is measured.

It is measured on the **validation** slice only, under the **true** pose
projection (sigma = 0), so what it sees is detector disagreement and never pose
error, and it is never allowed to look at the test split.

Five candidate forms are fitted for each of the two channels, by minimizing the
Gamma deviance of the squared residual against its modelled mean -- the MLE for
a 2-D isotropic Gaussian centre error, and a consistent quasi-likelihood for
the scalar heading error:

1. ``const``           -- homoscedastic, the estimator that was deployed
2. ``range``           -- ``sigma(r) = sigma_0 (1 + r / r_0)``
3. ``score_min``       -- ``sigma(s) = sigma_1 (min(s_ego, s_cav) / s_ref)^-b``
4. ``score_additive``  -- ``Var = v(s_ego) + v(s_cav)``, ``v(s) = sigma_1^2 (s/s_ref)^-2b``
5. ``score_additive_range`` -- 4, times ``(1 + r / r_0)^2``

**The selection rule is fixed in advance**: form 4, because it is the one
implied by the physics -- the disagreement is the difference of two independent
detections, so its variance is the sum of theirs -- unless a form fits
materially better. Forms 1, 2, 3 and 5 are fitted and reported so that
"materially better" is a measurement rather than an assertion, and so the
number of forms tried is on the record.

Usage::

    python scripts/fit_correspondence_variance.py \\
        --config configs/alignformer_r140.yaml \\
        --output outputs/alignformer/r140/correspondence_variance_result.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
import yaml
from scipy.optimize import minimize
from torch.utils.data import DataLoader

from embedding_aware_belt_fusion.alignformer.boxes import BOX_YAW
from embedding_aware_belt_fusion.alignformer.dataset import collate
from embedding_aware_belt_fusion.alignformer.train import (
    build_eval_dataset,
    build_pair_split,
)

# Bin edges in metres for the reported range profile. The 70.4 m edge is the
# superseded detector range, kept so the table is comparable with task 18's.
_RANGE_EDGES = (0.0, 20.0, 35.0, 50.0, 70.4, 90.0, 110.0, 145.0)
_BATCH_SIZE = 64
# The confidence the fitted sigmas are quoted at. Near the median of the
# observed score distribution, so the reported sigma is a typical
# correspondence's rather than an extrapolation.
SCORE_REFERENCE = 0.4
# The form this script commits to in advance; see the module docstring.
SELECTED_FORM = "score_additive"


def _fold_yaw(delta: torch.Tensor) -> torch.Tensor:
    """Wrap a yaw difference into [-pi/2, pi/2).

    The detector reports a box's *axis*, not its direction
    (``procrustes.heading_orientation``), so a 180 degree disagreement is the
    same heading. Folding to the half-circle is the same invariance the model
    applies, measured the same way.
    """
    return (delta + math.pi / 2.0) % math.pi - math.pi / 2.0


def collect_correspondences(config: Dict, num_workers: int) -> Dict[str, np.ndarray]:
    """Per-correspondence disagreement over the validation slice, at sigma = 0.

    A correspondence is a pair of detections the two agents both matched to the
    same OPV2V physical object id, which is what ``ego_match`` records.
    """
    _, val_pairs, _, val_scenarios = build_pair_split(config)
    dataset = build_eval_dataset(config, val_pairs, 0.0)
    loader = DataLoader(
        dataset,
        batch_size=_BATCH_SIZE,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=False,
    )

    columns: Dict[str, List[np.ndarray]] = {
        key: [] for key in ("range_m", "delta_m", "delta_yaw_rad", "ego_score", "cav_score")
    }
    for batch in loader:
        matched = (batch["ego_match"] >= 0) & batch["ego_mask"]
        if not bool(matched.any()):
            continue
        rows, cols = torch.nonzero(matched, as_tuple=True)
        partner = batch["ego_match"][rows, cols]
        ego = batch["ego_boxes"][rows, cols]
        cav = batch["cav_boxes"][rows, partner]

        columns["range_m"].append(ego[:, :2].norm(dim=-1).numpy())
        columns["delta_m"].append((ego[:, :2] - cav[:, :2]).norm(dim=-1).numpy())
        columns["delta_yaw_rad"].append(
            _fold_yaw(ego[:, BOX_YAW] - cav[:, BOX_YAW]).numpy()
        )
        columns["ego_score"].append(batch["ego_scores"][rows, cols].numpy())
        columns["cav_score"].append(batch["cav_scores"][rows, partner].numpy())

    collected = {key: np.concatenate(value) for key, value in columns.items()}
    collected["val_pairs"] = np.array([len(val_pairs)])
    collected["val_scenarios"] = np.array(val_scenarios)
    return collected


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else math.nan


def _profile(
    key: np.ndarray, edges: Sequence[float], delta: np.ndarray, yaw: np.ndarray
) -> List[Dict[str, float]]:
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        selection = (key >= low) & (key < high)
        rows.append(
            {
                "low": float(low),
                "high": float(high),
                "n": int(selection.sum()),
                "mean_key": float(key[selection].mean()) if selection.any() else math.nan,
                "rms_translation_m": _rms(delta[selection]),
                "rms_yaw_deg": math.degrees(_rms(yaw[selection])),
            }
        )
    return rows


def _decile_profile(
    key: np.ndarray, delta: np.ndarray, yaw: np.ndarray
) -> List[Dict[str, float]]:
    edges = np.percentile(key, np.arange(0, 101, 10))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    return _profile(key, edges.tolist(), delta, yaw)


def _gamma_deviance(
    predictor: Callable[[np.ndarray], np.ndarray], squared: np.ndarray
) -> Callable[[np.ndarray], float]:
    """``mean(log v + r^2 / v)``: the Gamma/exponential quasi-deviance, up to a constant.

    Minimized by ``v = E[r^2]`` for a constant model, which is what makes it
    the right criterion for fitting a *variance* rather than a mean.
    """

    def objective(parameters: np.ndarray) -> float:
        variance = predictor(parameters)
        if not np.all(np.isfinite(variance)) or np.any(variance <= 0.0):
            return 1e12
        return float(np.mean(np.log(variance) + squared / variance))

    return objective


def _candidates(
    ranges: np.ndarray, ego: np.ndarray, cav: np.ndarray, sigma_guess: float
) -> Dict[str, Tuple[Callable, List[float], List[str]]]:
    """The five pre-registered forms, as (variance function, start, parameter names)."""
    ego_ratio, cav_ratio = ego / SCORE_REFERENCE, cav / SCORE_REFERENCE
    minimum_ratio = np.minimum(ego_ratio, cav_ratio)
    return {
        "const": (
            lambda p: np.full_like(ranges, p[0] ** 2),
            [sigma_guess],
            ["sigma_m"],
        ),
        "range": (
            lambda p: (p[0] * (1.0 + ranges / p[1])) ** 2,
            [sigma_guess * 0.8, 150.0],
            ["sigma_0", "range_scale_m"],
        ),
        "score_min": (
            lambda p: (p[0] * minimum_ratio ** (-p[1])) ** 2,
            [sigma_guess, 1.0],
            ["sigma_1", "exponent"],
        ),
        SELECTED_FORM: (
            lambda p: (p[0] * ego_ratio ** (-p[1])) ** 2
            + (p[0] * cav_ratio ** (-p[1])) ** 2,
            [sigma_guess / math.sqrt(2.0), 1.0],
            ["sigma_1", "exponent"],
        ),
        "score_additive_range": (
            lambda p: (
                (p[0] * ego_ratio ** (-p[1])) ** 2 + (p[0] * cav_ratio ** (-p[1])) ** 2
            )
            * (1.0 + ranges / p[2]) ** 2,
            [sigma_guess / math.sqrt(2.0), 1.0, 300.0],
            ["sigma_1", "exponent", "range_scale_m"],
        ),
    }


def fit_channel(
    residual: np.ndarray, ranges: np.ndarray, ego: np.ndarray, cav: np.ndarray
) -> Dict[str, Dict]:
    """Fit every candidate form to one channel and return their deviances."""
    squared = np.square(residual)
    sigma_guess = _rms(residual)
    fits: Dict[str, Dict] = {}
    for name, (predictor, start, names) in _candidates(
        ranges, ego, cav, sigma_guess
    ).items():
        result = minimize(
            _gamma_deviance(predictor, squared),
            start,
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-7, "fatol": 1e-10},
        )
        fits[name] = {
            "deviance": float(result.fun),
            "parameters": dict(zip(names, [float(v) for v in result.x])),
            "converged": bool(result.success),
        }
    baseline = fits["const"]["deviance"]
    for entry in fits.values():
        entry["deviance_improvement_over_const"] = baseline - entry["deviance"]
    return fits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/alignformer_r140.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=10)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    data = collect_correspondences(config, args.num_workers)
    ranges = data["range_m"]
    delta = data["delta_m"]
    yaw = data["delta_yaw_rad"]
    ego, cav = data["ego_score"], data["cav_score"]
    minimum_score = np.minimum(ego, cav)

    translation = fit_channel(delta, ranges, ego, cav)
    heading = fit_channel(yaw, ranges, ego, cav)
    selected_t = translation[SELECTED_FORM]["parameters"]
    selected_y = heading[SELECTED_FORM]["parameters"]

    payload = {
        "method": "correspondence_variance_fit",
        "config": str(args.config),
        "cache_root": config["data"]["cache_root"],
        "split": (
            f"{config['data']['train_root']} :: validation scenarios "
            f"(scenario-disjoint, val_scenario_fraction="
            f"{config['data']['val_scenario_fraction']}, split_seed="
            f"{config['data']['split_seed']})"
        ),
        "sigma_m": 0.0,
        "correspondences": int(delta.size),
        "val_pairs": int(data["val_pairs"][0]),
        "val_scenarios": data["val_scenarios"].tolist(),
        "overall": {
            "rms_translation_m": _rms(delta),
            "rms_yaw_deg": math.degrees(_rms(yaw)),
        },
        "range_profile": _profile(ranges, _RANGE_EDGES, delta, yaw),
        "range_decile_profile": _decile_profile(ranges, delta, yaw),
        "score_decile_profile": _decile_profile(minimum_score, delta, yaw),
        "range_score_correlation": float(np.corrcoef(ranges, minimum_score)[0, 1]),
        "candidate_forms": {"translation": translation, "heading": heading},
        "selected_form": SELECTED_FORM,
        "selection_rule": (
            "fixed in advance: the additive form, because a correspondence's "
            "disagreement is the difference of two independent detections and its "
            "variance is therefore the sum of theirs. The other four forms are "
            "fitted and reported so 'nothing else fits materially better' is a "
            "measurement; five forms were tried per channel, and no further ones."
        ),
        "correspondence_variance": {
            "mode": "split",
            "sigma_translation_m": selected_t["sigma_1"],
            "translation_exponent": selected_t["exponent"],
            "sigma_yaw_rad": selected_y["sigma_1"],
            "sigma_yaw_deg": math.degrees(selected_y["sigma_1"]),
            "yaw_exponent": selected_y["exponent"],
            "score_reference": SCORE_REFERENCE,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"{payload['correspondences']} correspondences over {payload['val_pairs']} pairs")
    print("range profile (RMS):")
    for row in payload["range_profile"]:
        print(
            f"  {row['low']:6.1f}-{row['high']:6.1f} m  n={row['n']:7d}  "
            f"t={row['rms_translation_m']:.4f} m  yaw={row['rms_yaw_deg']:.3f} deg"
        )
    print("min(score) deciles (RMS):")
    for row in payload["score_decile_profile"]:
        print(
            f"  {row['low']:5.3f}-{row['high']:5.3f}  n={row['n']:7d}  "
            f"t={row['rms_translation_m']:.4f} m  yaw={row['rms_yaw_deg']:.3f} deg"
        )
    print(f"corr(range, min score) = {payload['range_score_correlation']:.3f}")
    for channel, fits in payload["candidate_forms"].items():
        print(f"{channel} candidate forms (Gamma deviance, lower is better):")
        for name, entry in fits.items():
            print(
                f"  {name:22s} dev={entry['deviance']:+.5f} "
                f"(improvement {entry['deviance_improvement_over_const']:+.5f}) "
                f"{ {k: round(v, 4) for k, v in entry['parameters'].items()} }"
            )
    block = payload["correspondence_variance"]
    print(
        "selected: v_t(s) = ({:.4f} m * (s/{:g})^-{:.4f})^2, "
        "v_psi(s) = ({:.4f} deg * (s/{:g})^-{:.4f})^2".format(
            block["sigma_translation_m"], block["score_reference"],
            block["translation_exponent"], block["sigma_yaw_deg"],
            block["score_reference"], block["yaw_exponent"],
        )
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
