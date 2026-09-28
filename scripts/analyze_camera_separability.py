"""Does camera colour carry cross-agent object identity on OPV2V?

The question, and why it is still open
--------------------------------------

AlignFormer's LiDAR appearance embedding contributes nothing to association:
``+0.0001``, 95% CI ``[-0.0013, +0.0014]`` (``scripts/analyze_association.py``,
commit ``a9e8449``). Three reasons were recorded. The first is about **shape**
-- AUC 0.560 separating a true partner from its nearest competitor, because
CARLA's asset library is small and all 65,774 CAV box widths are
``2.014 +/- 0.081`` m. The second is a resolution bound on the 0.8 m/cell BEV
map. Neither says anything about **colour**, which LiDAR cannot measure, and
nothing in this repository had read the cameras.

A first probe over 118 vehicle views put the mean-BGR spread *within* one
identity at 3.2 and *between* identities at 32.6. That is same-agent, which is
the easy case. This script asks the cross-agent version: ego's cameras against
the CAV's, different viewpoint, different range, different incident light.

The third recorded reason is untouched and still binds. **No ego object in the
validation split has a competitor within 2 m.** Where geometry is unambiguous
no feature can help, so the only cell where colour could pay is large
localization error, and this script reports the geometrically ambiguous subset
separately for that reason.

THE DECISION RULE, FIXED BEFORE THE FIRST NUMBER WAS READ
---------------------------------------------------------

Camera appearance is worth building into the matcher if and only if **all
three** hold on the scenario-disjoint validation slice:

1. **Signal.** Cross-agent AUC on the geometrically ambiguous subset (a
   competitor within ``PROBE_RADIUS`` = 8 m, the radius the LiDAR probe used)
   is at least **0.80**. The LiDAR embedding reached 0.627 and bought nothing,
   so a bar below that is a bar that has already been shown not to matter.
2. **Coverage.** At least **40%** of cross-agent object correspondences have a
   usable crop in *both* agents. A descriptor available on a fifth of objects
   cannot move an average, however separable it is.
3. **Reach.** Coverage in the **40-70 m** band -- where the pose fit degrades
   from 0.021 m to 0.112 m and where the clean-mAP loss lives -- is at least
   **25%**. A feature that exists only where geometry already works is a
   feature with nothing to add.

Failing any of the three is a null, and a null is reported as one. The shuffle
control must land within 0.02 of 0.5 or the whole run is void, whichever way
the headline points.

This reads the validation slice only -- a scenario-disjoint 15% of ``train/``
at ``split_seed 0`` -- through ``alignformer.splits.resolve_split`` with
``allow_test=False``, because it selects rather than reports.

Usage::

    python scripts/analyze_camera_separability.py \\
        --config configs/alignformer_r140.yaml \\
        --pairs 400 \\
        --output outputs/alignformer/r140/camera_separability_result.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from embedding_aware_belt_fusion.alignformer.camera import (
    MIN_CROP_PIXELS,
    appearance,
    cosine_similarity,
    occlusion_fraction,
    project_corners,
    silhouette_mask,
    vehicle_world_corners,
    world_from_pose,
)
from embedding_aware_belt_fusion.alignformer.splits import resolve_split
from embedding_aware_belt_fusion.alignformer.train import build_pair_split

# The radius scripts/analyze_association.py used for its distractor. Reused so
# the two AUCs answer the same question about the same population.
PROBE_RADIUS_M = 8.0

# A crop more than this fraction covered by nearer vehicles is the occluder.
MAX_OCCLUSION = 0.5

CAMERA_COUNT = 4
RANGE_EDGES_M = (0.0, 25.0, 40.0, 70.0, 1e9)

DESCRIPTORS = ("hue_saturation", "mean_colour")

# The pre-registered bars, in one place so the report can state whether each
# was met without a reader having to re-derive it from prose.
BAR_AUC = 0.80
BAR_COVERAGE = 0.40
BAR_FAR_COVERAGE = 0.25
SHUFFLE_TOLERANCE = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_annotation(path: Path) -> Dict:
    """OPV2V's own YAML, including the legacy numpy tags safe_load rejects."""
    with path.open("r", encoding="utf-8") as stream:
        return yaml.unsafe_load(stream)


def agent_directory(root: Path, scenario: str, agent: str) -> Path:
    return root / scenario / agent


def _views(
    annotation: Dict, directory: Path, timestamp: str, vehicles: Dict
) -> Dict[str, Dict[str, np.ndarray]]:
    """Best usable camera crop per vehicle id, for one agent at one frame.

    "Best" is the least occluded view with enough masked pixels; ties go to
    the nearer one. Reading four 800x600 PNGs per agent is the cost of this
    script, so the images are loaded once and shared across all vehicles.
    """
    images: List[Optional[np.ndarray]] = []
    metas: List[Optional[Dict]] = []
    for index in range(CAMERA_COUNT):
        key = f"camera{index}"
        path = directory / f"{timestamp}_{key}.png"
        meta = annotation.get(key)
        image = cv2.imread(str(path)) if meta is not None and path.exists() else None
        images.append(image)
        metas.append(meta)

    corners = {vid: vehicle_world_corners(veh) for vid, veh in vehicles.items()}
    found: Dict[str, Dict[str, np.ndarray]] = {}
    for index, (image, meta) in enumerate(zip(images, metas)):
        if image is None or meta is None:
            continue
        projected = {
            vid: project_corners(points, meta, image.shape)
            for vid, points in corners.items()
        }
        neighbours = [value for value in projected.values() if value is not None]
        for vid, view in projected.items():
            if view is None:
                continue
            mask = silhouette_mask(view.pixels, image.shape)
            pixel_count = int(mask.sum())
            if pixel_count < MIN_CROP_PIXELS:
                continue
            covered = occlusion_fraction(view, neighbours, image.shape)
            if covered > MAX_OCCLUSION:
                continue
            descriptors = appearance(image, mask)
            if descriptors is None:
                continue
            candidate = {
                **descriptors,
                "occlusion": covered,
                "pixels": pixel_count,
                "depth": view.depth,
                "camera": index,
            }
            previous = found.get(vid)
            if previous is None or covered < previous["occlusion"]:
                # The displaced best becomes the runner-up: a SECOND camera on
                # the same agent seeing the same vehicle is a viewpoint change
                # with no cross-agent gap, which is what separates "colour
                # carries no identity" from "colour does, and the cross-agent
                # viewpoint destroys it". Only the second answer leaves a
                # learned encoder anything to recover.
                if previous is not None:
                    candidate["runner_up"] = previous
                elif vid in found:
                    candidate["runner_up"] = found[vid].get("runner_up")
                found[vid] = candidate
            elif "runner_up" not in found[vid] or found[vid]["runner_up"] is None:
                found[vid]["runner_up"] = candidate
    return found


def _lidar_range(vehicle: Dict, lidar_pose: Sequence[float]) -> float:
    centre = vehicle_world_corners(vehicle).mean(axis=0)
    local = np.linalg.inv(world_from_pose(lidar_pose)) @ np.append(centre, 1.0)
    return float(np.linalg.norm(local[:2]))


def _world_centre(vehicle: Dict) -> np.ndarray:
    return vehicle_world_corners(vehicle).mean(axis=0)[:2]


def _range_bucket(distance: float) -> str:
    for low, high in zip(RANGE_EDGES_M, RANGE_EDGES_M[1:]):
        if low <= distance < high:
            return f"{low:g}-{high:g}m" if high < 1e8 else f"{low:g}m+"
    return "unbucketed"


def collect(root: Path, pairs, rng: random.Random) -> Dict:
    """Walk the sampled ego-CAV pairs and gather partner/distractor scores."""
    scores = {name: {"partner": [], "distractor": []} for name in DESCRIPTORS}
    shuffled = {name: {"partner": [], "distractor": []} for name in DESCRIPTORS}
    coverage = {"both": 0, "shared": 0}
    by_range = defaultdict(lambda: {"both": 0, "shared": 0})
    ambiguous = {name: {"partner": [], "distractor": []} for name in DESCRIPTORS}
    same_agent = {name: {"partner": [], "distractor": []} for name in DESCRIPTORS}
    pixel_rows: List[Tuple[float, int]] = []

    for done, pair in enumerate(pairs, start=1):
        ego_dir = agent_directory(root, pair.scenario, pair.ego_id)
        cav_dir = agent_directory(root, pair.scenario, pair.cav_id)
        ego_yaml = ego_dir / f"{pair.timestamp}.yaml"
        cav_yaml = cav_dir / f"{pair.timestamp}.yaml"
        if not ego_yaml.exists() or not cav_yaml.exists():
            continue
        ego_ann, cav_ann = load_annotation(ego_yaml), load_annotation(cav_yaml)
        ego_vehicles = ego_ann.get("vehicles", {}) or {}
        cav_vehicles = cav_ann.get("vehicles", {}) or {}
        common = sorted(set(ego_vehicles) & set(cav_vehicles), key=str)
        if len(common) < 2:
            continue

        ego_views = _views(ego_ann, ego_dir, pair.timestamp, ego_vehicles)
        cav_views = _views(cav_ann, cav_dir, pair.timestamp, cav_vehicles)
        centres = {vid: _world_centre(ego_vehicles[vid]) for vid in common}

        for vid in common:
            distance = _lidar_range(ego_vehicles[vid], ego_ann["lidar_pose"])
            bucket = _range_bucket(distance)
            coverage["shared"] += 1
            by_range[bucket]["shared"] += 1
            usable = vid in ego_views and vid in cav_views
            if not usable:
                continue
            coverage["both"] += 1
            by_range[bucket]["both"] += 1
            pixel_rows.append((distance, int(ego_views[vid]["pixels"])))

            others = [
                other
                for other in common
                if other != vid and other in cav_views
            ]
            if not others:
                continue
            nearest = min(
                others, key=lambda o: float(np.linalg.norm(centres[o] - centres[vid]))
            )
            gap = float(np.linalg.norm(centres[nearest] - centres[vid]))

            for name in DESCRIPTORS:
                partner = cosine_similarity(
                    ego_views[vid][name], cav_views[vid][name]
                )
                distractor = cosine_similarity(
                    ego_views[vid][name], cav_views[nearest][name]
                )
                scores[name]["partner"].append(partner)
                scores[name]["distractor"].append(distractor)
                if gap <= PROBE_RADIUS_M:
                    ambiguous[name]["partner"].append(partner)
                    ambiguous[name]["distractor"].append(distractor)
            # The null control: break the ego-to-CAV identity on BOTH sides,
            # so the "partner" is a random CAV object and so is the
            # "distractor". Anything that manufactures separation -- a crop
            # that is mostly road, a descriptor that tracks apparent size --
            # separates these two as well, and the AUC here has to come back
            # at 0.5 or the headline is not measuring identity. Drawn once per
            # object rather than per descriptor so both descriptors are
            # controlled against the same draw.
            # The same-agent upper bound: ego's SECOND camera on this vehicle
            # against ego's best camera on the nearest other vehicle. Same
            # agent, same instant, same illumination -- only the viewpoint
            # differs. Whatever this scores is what a perfect cross-agent
            # descriptor could hope for.
            second = ego_views[vid].get("runner_up")
            if second is not None and nearest in ego_views:
                for name in DESCRIPTORS:
                    same_agent[name]["partner"].append(
                        cosine_similarity(second[name], ego_views[vid][name])
                    )
                    same_agent[name]["distractor"].append(
                        cosine_similarity(second[name], ego_views[nearest][name])
                    )

            decoys = rng.sample(list(cav_views), 2) if len(cav_views) >= 2 else None
            if decoys is not None:
                for name in DESCRIPTORS:
                    shuffled[name]["partner"].append(
                        cosine_similarity(
                            ego_views[vid][name], cav_views[decoys[0]][name]
                        )
                    )
                    shuffled[name]["distractor"].append(
                        cosine_similarity(
                            ego_views[vid][name], cav_views[decoys[1]][name]
                        )
                    )

        if done % 25 == 0:
            print(f"  {done} pairs, {coverage['both']}/{coverage['shared']} usable")

    return {
        "scores": scores,
        "ambiguous": ambiguous,
        "same_agent": same_agent,
        "shuffled": shuffled,
        "coverage": coverage,
        "by_range": {key: dict(value) for key, value in by_range.items()},
        "pixels": pixel_rows,
    }


def summarize(partner: Sequence[float], distractor: Sequence[float]) -> Dict:
    """Mann-Whitney AUC and friends, the same statistic the LiDAR probe used."""
    partner = np.asarray(partner, dtype=np.float64)
    distractor = np.asarray(distractor, dtype=np.float64)
    if partner.size == 0 or distractor.size == 0:
        return {"n": 0, "auc": None}
    combined = np.concatenate([partner, distractor])
    ranks = combined.argsort().argsort().astype(np.float64) + 1.0
    rank_sum = ranks[: partner.size].sum()
    auc = (rank_sum - partner.size * (partner.size + 1) / 2.0) / (
        partner.size * distractor.size
    )
    return {
        "n": int(partner.size),
        "auc": float(auc),
        "pairwise_accuracy": float((partner > distractor).mean()),
        "partner_mean": float(partner.mean()),
        "distractor_mean": float(distractor.mean()),
        "mean_gap": float(partner.mean() - distractor.mean()),
        "cohens_d": float(
            (partner.mean() - distractor.mean())
            / np.sqrt(0.5 * (partner.var() + distractor.var()) + 1e-12)
        ),
    }


def verdict(report: Dict) -> Dict:
    """Apply the three pre-registered bars, and say which of them failed."""
    best = max(
        DESCRIPTORS,
        key=lambda name: report["ambiguous"][name]["auc"] or 0.0,
    )
    auc = report["ambiguous"][best]["auc"] or 0.0
    coverage = report["coverage"]["fraction"]
    far = report["coverage"]["by_range"].get("40-70m", {}).get("fraction", 0.0)
    shuffle_ok = all(
        abs((report["shuffled_control"][name]["auc"] or 0.5) - 0.5) <= SHUFFLE_TOLERANCE
        for name in DESCRIPTORS
    )
    clauses = {
        "signal_auc_at_least_0.80": auc >= BAR_AUC,
        "coverage_at_least_0.40": coverage >= BAR_COVERAGE,
        "far_field_coverage_at_least_0.25": far >= BAR_FAR_COVERAGE,
    }
    return {
        "best_descriptor": best,
        "ambiguous_auc": auc,
        "coverage": coverage,
        "far_field_coverage": far,
        "clauses": clauses,
        "shuffle_control_valid": shuffle_ok,
        "build_camera_branch": bool(all(clauses.values()) and shuffle_ok),
    }


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    root = resolve_split(config["data"]["train_root"], allow_test=False)

    _, val_pairs, _, val_scenarios = build_pair_split(config)
    rng = random.Random(args.seed)
    sampled = val_pairs if len(val_pairs) <= args.pairs else rng.sample(
        list(val_pairs), args.pairs
    )
    print(
        f"{len(val_pairs)} validation pairs over {len(val_scenarios)} scenarios; "
        f"reading {len(sampled)}"
    )

    collected = collect(root, sampled, rng)

    shared = max(collected["coverage"]["shared"], 1)
    report = {
        "method": "camera_separability",
        "config": str(args.config),
        "split": (
            f"{root} :: validation scenarios (scenario-disjoint, "
            f"val_scenario_fraction={config['data']['val_scenario_fraction']}, "
            f"split_seed={config['data']['split_seed']})"
        ),
        "val_scenarios": val_scenarios,
        "pairs_read": len(sampled),
        "probe_radius_m": PROBE_RADIUS_M,
        "min_crop_pixels": MIN_CROP_PIXELS,
        "max_occlusion": MAX_OCCLUSION,
        "pre_registered_bars": {
            "signal_auc": BAR_AUC,
            "coverage": BAR_COVERAGE,
            "far_field_coverage": BAR_FAR_COVERAGE,
        },
        "lidar_reference": {
            "roi_auc": 0.5595922765932373,
            "embedding_auc": 0.6272729723188022,
            "association_delta": 0.0001,
            "association_ci": [-0.0013, 0.0014],
        },
        "coverage": {
            "shared_objects": collected["coverage"]["shared"],
            "usable_in_both": collected["coverage"]["both"],
            "fraction": collected["coverage"]["both"] / shared,
            "by_range": {
                key: {
                    **value,
                    "fraction": value["both"] / max(value["shared"], 1),
                }
                for key, value in sorted(collected["by_range"].items())
            },
        },
        "all_objects": {
            name: summarize(
                collected["scores"][name]["partner"],
                collected["scores"][name]["distractor"],
            )
            for name in DESCRIPTORS
        },
        "ambiguous": {
            name: summarize(
                collected["ambiguous"][name]["partner"],
                collected["ambiguous"][name]["distractor"],
            )
            for name in DESCRIPTORS
        },
        "same_agent_bound": {
            name: summarize(
                collected["same_agent"][name]["partner"],
                collected["same_agent"][name]["distractor"],
            )
            for name in DESCRIPTORS
        },
        "shuffled_control": {
            name: summarize(
                collected["shuffled"][name]["partner"],
                collected["shuffled"][name]["distractor"],
            )
            for name in DESCRIPTORS
        },
    }
    report["verdict"] = verdict(report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["verdict"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
