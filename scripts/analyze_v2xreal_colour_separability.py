"""Does camera colour carry cross-agent object identity on V2X-Real?

Why ask again
-------------
On OPV2V camera colour failed all three pre-registered bars
(``scripts/analyze_camera_separability.py``): cross-agent AUC 0.60 on the
ambiguous subset against a same-agent bound of 0.71, coverage collapsing past
40 m. Two of the reasons were CARLA's: a small asset library, and rendered
paint. Neither holds on real traffic. The third, the cross-agent viewpoint
change, might.

What is different on V2X-Real, and why the probe is worth a day: the pairs
sharing one or two objects are 23 % of test frames (9 % on OPV2V). With one
or two shared objects there is no geometric context, and whether the single
match is right is the whole question; the deployed estimator loses 0.09
AP@0.7 there in the clean case by answering with wrong matches. An identity
cue has a measured home on this dataset. This script therefore reports the
separability by shared-object bucket as well as by range.

THE DECISION RULE, FIXED BEFORE THE FIRST NUMBER WAS READ
---------------------------------------------------------
Camera colour is worth building into the matcher if and only if, on the
official validation split:

1. **Signal.** Cross-agent AUC on the geometrically ambiguous subset (a
   competitor within ``PROBE_RADIUS_M`` = 8 m) is at least **0.80**.
2. **Coverage.** At least **40 %** of shared objects have a usable crop in
   both agents.
3. **Reach.** Coverage in the 40-70 m band is at least **25 %**.

Failing any of the three is a null, reported as one. The shuffle control must
land within 0.02 of 0.5 or the whole run is void. The ``shared_1_2`` rows are
reported, not gated: they say where a passing cue would pay.

Reads ``val/`` only; a path ending in ``test`` is refused.

Usage::

    python scripts/analyze_v2xreal_colour_separability.py \\
        --root /media/chenyi/basement2/dataset/v2x-real/val --pairs 400 \\
        --output outputs/v2xreal/colour_separability_val_result.json

The probe is cue-agnostic: ``--descriptors dinov2_small`` (or ``_base``,
or ``colour+dinov2_small`` for both on the same candidates) scores a frozen
DINOv2 crop embedding (``alignformer.foundation_features``) through the
same pairs, gates, controls and bars. Same bars, fixed before the first
number: a foundation model is held to the standard colour failed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embedding_aware_belt_fusion.alignformer.camera import (  # noqa: E402
    MIN_CROP_PIXELS,
    Projection,
    appearance,
    cosine_similarity,
    occlusion_fraction,
    silhouette_mask,
    vehicle_world_corners,
    world_from_pose,
)
from embedding_aware_belt_fusion.alignformer.camera_features import (  # noqa: E402
    MIN_DEPTH_M,
    calibration_from_yaml,
    load_image,
    project_points,
)
from embedding_aware_belt_fusion.alignformer.v2xreal import (  # noqa: E402
    VEHICLE_TYPES,
    fast_load_yaml,
)

PROBE_RADIUS_M = 8.0
MAX_OCCLUSION = 0.5
RANGE_EDGES_M = (0.0, 25.0, 40.0, 70.0, 1e9)
DESCRIPTORS = ("hue_saturation", "mean_colour")
SHARED_BUCKETS = ("shared_1_2", "shared_3plus")
BAR_AUC = 0.80
BAR_COVERAGE = 0.40
BAR_FAR_COVERAGE = 0.25
SHUFFLE_TOLERANCE = 0.02
PROGRESS_EVERY = 25


class Pair(NamedTuple):
    scenario: str
    timestamp: str
    ego: str
    cav: str


# ---------------------------------------------------------------------------
# Descriptors: the probe is the same whatever cue is scored
# ---------------------------------------------------------------------------


class Describer(NamedTuple):
    """Named descriptors of one projected vehicle: ``describe(image_bgr, mask, view)``
    returns ``{name: vector}`` for every name in ``names``, or ``None`` to skip."""

    names: Tuple[str, ...]
    describe: Callable[[np.ndarray, np.ndarray, Projection], Optional[Dict[str, np.ndarray]]]


def colour_describer() -> Describer:
    return Describer(DESCRIPTORS, lambda image_bgr, mask, view: appearance(image_bgr, mask))


def foundation_describer(size: str) -> Describer:
    """Frozen DINOv2 (``small`` or ``base``) on the letterboxed crop with context."""
    from embedding_aware_belt_fusion.alignformer.foundation_features import DESCRIPTOR_NAMES, FoundationBackbone

    backbone = FoundationBackbone(model=size)
    names = tuple(f"{n}_{size}" for n in DESCRIPTOR_NAMES)

    def describe(image_bgr, mask, view):
        raw = backbone.describe_projection(image_bgr[..., ::-1], mask, view.box)
        return {f"{n}_{size}": raw[n] for n in DESCRIPTOR_NAMES}

    return Describer(names, describe)


def combine(*parts: Describer) -> Describer:
    """One describer scoring every cue on the same candidates; any part declining declines all."""

    def describe(image_bgr, mask, view):
        merged: Dict[str, np.ndarray] = {}
        for part in parts:
            out = part.describe(image_bgr, mask, view)
            if out is None:
                return None
            merged.update(out)
        return merged

    return Describer(tuple(n for part in parts for n in part.names), describe)


DESCRIBER_CHOICES = ("colour", "dinov2_small", "dinov2_base", "colour+dinov2_small", "colour+dinov2_base")


def build_describer(choice: str) -> Describer:
    parts = []
    for name in choice.split("+"):
        if name == "colour":
            parts.append(colour_describer())
        elif name.startswith("dinov2_"):
            parts.append(foundation_describer(name.split("_", 1)[1]))
        else:
            raise ValueError(f"unknown descriptor set {name!r}; one of {DESCRIBER_CHOICES}")
    return parts[0] if len(parts) == 1 else combine(*parts)


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


def refuse_test_split(root: Path) -> None:
    if root.name == "test":
        raise ValueError(f"{root} is the test split; this probe selects, so it reads val only")


def _timestamps(agent_dir: Path) -> List[str]:
    return sorted(p.stem for p in agent_dir.glob("*.yaml") if not p.name.startswith("._"))


def enumerate_pairs(root: Path) -> List[Pair]:
    """Every ordered (ego, cav) agent pair at every timestamp both recorded."""
    pairs: List[Pair] = []
    for scenario in sorted(p for p in root.iterdir() if p.is_dir()):
        stamps_by_agent = {a.name: set(_timestamps(a)) for a in scenario.iterdir() if a.is_dir()}
        for stamp in sorted(set().union(*stamps_by_agent.values()) if stamps_by_agent else ()):
            present = sorted(a for a, stamps in stamps_by_agent.items() if stamp in stamps)
            pairs.extend(Pair(scenario.name, stamp, e, c) for e in present for c in present if e != c)
    return pairs


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def vehicle_lidar_corners(vehicle: Dict, lidar_pose: Sequence[float]) -> np.ndarray:
    """``(8, 3)`` corners of one annotation in the agent's LiDAR frame."""
    world = vehicle_world_corners(vehicle)
    lidar_from_world = np.linalg.inv(world_from_pose(lidar_pose))
    homogeneous = np.concatenate((world, np.ones((8, 1))), axis=1)
    return (lidar_from_world @ homogeneous.T).T[:, :3]


def _projection(corners_lidar: np.ndarray, block: Dict, image_shape) -> Optional[Projection]:
    pixels, depth = project_points(corners_lidar, calibration_from_yaml(block))
    if np.any(depth < MIN_DEPTH_M):
        return None
    height, width = image_shape[:2]
    x1, y1 = max(0, int(np.floor(pixels[:, 0].min()))), max(0, int(np.floor(pixels[:, 1].min())))
    x2, y2 = min(width - 1, int(np.ceil(pixels[:, 0].max()))), min(height - 1, int(np.ceil(pixels[:, 1].max())))
    if x1 >= x2 or y1 >= y2:
        return None
    return Projection(pixels=pixels, box=(x1, y1, x2, y2), depth=float(depth.mean()))


def shared_bucket(count: int) -> str:
    return SHARED_BUCKETS[0] if count <= 2 else SHARED_BUCKETS[1]


def _range_bucket(distance: float) -> str:
    for low, high in zip(RANGE_EDGES_M, RANGE_EDGES_M[1:]):
        if low <= distance < high:
            return f"{low:g}-{high:g}m" if high < 1e8 else f"{low:g}m+"
    return "unbucketed"


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


def _vehicles(params: Dict) -> Dict[str, Dict]:
    return {
        str(vid): entry
        for vid, entry in (params.get("vehicles") or {}).items()
        if entry.get("obj_type") in VEHICLE_TYPES
    }


def _candidate(image_bgr: np.ndarray, view: Projection, neighbours, camera: str, describe: Describer) -> Optional[Dict]:
    mask = silhouette_mask(view.pixels, image_bgr.shape)
    pixel_count = int(mask.sum())
    if pixel_count < MIN_CROP_PIXELS:
        return None
    covered = occlusion_fraction(view, neighbours, image_bgr.shape)
    if covered > MAX_OCCLUSION:
        return None
    descriptors = describe.describe(image_bgr, mask, view)
    if descriptors is None:
        return None
    return {**descriptors, "occlusion": covered, "pixels": pixel_count, "depth": view.depth, "camera": camera}


def _keep_best(found: Dict[str, Dict], vid: str, candidate: Dict) -> None:
    previous = found.get(vid)
    if previous is None or candidate["occlusion"] < previous["occlusion"]:
        candidate["runner_up"] = previous if previous is not None else None
        found[vid] = candidate
    elif previous.get("runner_up") is None:
        previous["runner_up"] = candidate


def views_for_agent_frame(
    params: Dict, images_bgr: Dict[str, np.ndarray], describe: Optional[Describer] = None
) -> Dict[str, Dict]:
    """Best usable camera crop per vehicle id for one agent-frame.

    ``images_bgr`` maps a camera block name (``cam1``) to its BGR image; a
    camera whose jpeg is missing is simply absent. "Best" is the least
    occluded view with enough silhouette pixels; the displaced view is kept as
    the same-agent runner-up (the viewpoint-only bound).
    """
    describe = describe if describe is not None else colour_describer()
    corners = {vid: vehicle_lidar_corners(v, params["lidar_pose"]) for vid, v in _vehicles(params).items()}
    found: Dict[str, Dict] = {}
    for camera, image in images_bgr.items():
        block = params.get(camera)
        if block is None:
            continue
        projected = {vid: _projection(c, block, image.shape) for vid, c in corners.items()}
        neighbours = [p for p in projected.values() if p is not None]
        for vid, view in projected.items():
            if view is None:
                continue
            candidate = _candidate(image, view, neighbours, camera, describe)
            if candidate is not None:
                _keep_best(found, vid, candidate)
    return found


def _load_views(root: Path, pair_dir: Path, timestamp: str, describe: Describer) -> Tuple[Dict, Dict[str, Dict]]:
    params = fast_load_yaml(str(pair_dir / f"{timestamp}.yaml"))
    images: Dict[str, np.ndarray] = {}
    for camera in sorted(k for k in params if k.startswith("cam")):
        path = pair_dir / f"{timestamp}_{camera}.jpeg"
        if path.exists() and path.stat().st_size > 0:
            images[camera] = load_image(path)[..., ::-1]  # appearance() wants BGR
    return params, views_for_agent_frame(params, images, describe)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _new_scores(names: Sequence[str]):
    return {name: {"partner": [], "distractor": []} for name in names}


def _lidar_range(vehicle: Dict, lidar_pose: Sequence[float]) -> float:
    return float(np.linalg.norm(vehicle_lidar_corners(vehicle, lidar_pose).mean(axis=0)[:2]))


def _append(scores, name: str, partner: float, distractor: float) -> None:
    scores[name]["partner"].append(partner)
    scores[name]["distractor"].append(distractor)


def _score_object(vid, common_bucket, range_bucket, ego_views, cav_views, gap_centres, acc, rng, names) -> None:
    others = [o for o in cav_views if o != vid]
    if not others:
        return
    nearest = min(others, key=lambda o: float(np.linalg.norm(gap_centres[o] - gap_centres[vid])))
    gap = float(np.linalg.norm(gap_centres[nearest] - gap_centres[vid]))
    for name in names:
        partner = cosine_similarity(ego_views[vid][name], cav_views[vid][name])
        distractor = cosine_similarity(ego_views[vid][name], cav_views[nearest][name])
        _append(acc["scores"], name, partner, distractor)
        _append(acc["by_shared"][common_bucket], name, partner, distractor)
        if gap <= PROBE_RADIUS_M:
            _append(acc["ambiguous"], name, partner, distractor)
            _append(acc["ambiguous_by_shared"][common_bucket], name, partner, distractor)
            _append(acc["ambiguous_by_range"].setdefault(range_bucket, _new_scores(names)), name, partner, distractor)
    second = ego_views[vid].get("runner_up")
    if second is not None and nearest in ego_views:
        for name in names:
            _append(acc["same_agent"], name, cosine_similarity(second[name], ego_views[vid][name]),
                    cosine_similarity(second[name], ego_views[nearest][name]))
    if len(cav_views) >= 2:
        decoys = rng.sample(list(cav_views), 2)
        for name in names:
            _append(acc["shuffled"], name, cosine_similarity(ego_views[vid][name], cav_views[decoys[0]][name]),
                    cosine_similarity(ego_views[vid][name], cav_views[decoys[1]][name]))


def collect(root: Path, pairs: Sequence[Pair], rng: random.Random, describe: Optional[Describer] = None) -> Dict:
    """Walk the sampled pairs and gather partner/distractor scores and coverage."""
    describe = describe if describe is not None else colour_describer()
    names = describe.names
    acc = {
        "scores": _new_scores(names), "ambiguous": _new_scores(names), "same_agent": _new_scores(names),
        "shuffled": _new_scores(names),
        "by_shared": {b: _new_scores(names) for b in SHARED_BUCKETS},
        "ambiguous_by_shared": {b: _new_scores(names) for b in SHARED_BUCKETS},
        "ambiguous_by_range": {},
    }
    coverage = {"both": 0, "shared": 0}
    by_range = defaultdict(lambda: {"both": 0, "shared": 0})
    by_shared = {b: {"both": 0, "shared": 0, "pairs": 0} for b in SHARED_BUCKETS}
    for done, pair in enumerate(pairs, start=1):
        ego_params, ego_views = _load_views(root, root / pair.scenario / pair.ego, pair.timestamp, describe)
        cav_params, cav_views = _load_views(root, root / pair.scenario / pair.cav, pair.timestamp, describe)
        ego_vehicles, cav_vehicles = _vehicles(ego_params), _vehicles(cav_params)
        common = sorted(set(ego_vehicles) & set(cav_vehicles))
        if not common:
            continue
        bucket = shared_bucket(len(common))
        by_shared[bucket]["pairs"] += 1
        centres = {vid: vehicle_world_corners(v).mean(axis=0)[:2] for vid, v in cav_vehicles.items()}
        for vid in common:
            range_bucket = _range_bucket(_lidar_range(ego_vehicles[vid], ego_params["lidar_pose"]))
            coverage["shared"] += 1
            by_range[range_bucket]["shared"] += 1
            by_shared[bucket]["shared"] += 1
            if vid not in ego_views or vid not in cav_views:
                continue
            coverage["both"] += 1
            by_range[range_bucket]["both"] += 1
            by_shared[bucket]["both"] += 1
            _score_object(vid, bucket, range_bucket, ego_views, cav_views, centres, acc, rng, names)
        if done % PROGRESS_EVERY == 0:
            print(f"  {done} pairs, {coverage['both']}/{coverage['shared']} usable")
    return {**acc, "names": names, "coverage": coverage, "by_range": {k: dict(v) for k, v in by_range.items()},
            "by_shared_coverage": by_shared}


def auc(partner: Sequence[float], distractor: Sequence[float]) -> Optional[float]:
    """P(partner score > distractor score), ties at one half; paired by object."""
    if not partner:
        return None
    p, d = np.asarray(partner), np.asarray(distractor)
    return float(np.mean((p > d) + 0.5 * (p == d)))


def summarize(partner: Sequence[float], distractor: Sequence[float]) -> Dict:
    return {
        "n": len(partner),
        "auc_paired": auc(partner, distractor),
        "partner_mean": float(np.mean(partner)) if partner else None,
        "distractor_mean": float(np.mean(distractor)) if distractor else None,
    }


def _summaries(scores, names: Sequence[str]) -> Dict:
    return {name: summarize(scores[name]["partner"], scores[name]["distractor"]) for name in names}


def verdict(report: Dict, names: Sequence[str] = DESCRIPTORS) -> Dict:
    best = max(names, key=lambda n: report["ambiguous"][n]["auc_paired"] or 0.0)
    signal = report["ambiguous"][best]["auc_paired"] or 0.0
    control = report["shuffled_control"][best]["auc_paired"]
    far = report["coverage"]["by_range"].get("40-70m", {}).get("fraction", 0.0)
    passed = {
        "signal": signal >= BAR_AUC,
        "coverage": report["coverage"]["fraction"] >= BAR_COVERAGE,
        "far_field_coverage": far >= BAR_FAR_COVERAGE,
    }
    valid = control is not None and abs(control - 0.5) <= SHUFFLE_TOLERANCE
    return {
        "best_descriptor": best, "ambiguous_auc": signal, "coverage": report["coverage"]["fraction"],
        "far_field_coverage": far, "shuffle_control_auc": control, "control_valid": valid,
        "bars": passed, "worth_building": valid and all(passed.values()),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="the V2X-Real val directory")
    parser.add_argument("--pairs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--descriptors", choices=DESCRIBER_CHOICES, default="colour",
                        help="which cue(s) to score on the same candidates; DINOv2 is frozen, zero-shot")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    refuse_test_split(args.root)
    pairs = enumerate_pairs(args.root)
    rng = random.Random(args.seed)
    sampled = pairs if len(pairs) <= args.pairs else rng.sample(pairs, args.pairs)
    print(f"{len(pairs)} agent pairs in {args.root}; reading {len(sampled)}")
    describe = build_describer(args.descriptors)
    collected = collect(args.root, sampled, rng, describe)
    names = collected["names"]
    shared = max(collected["coverage"]["shared"], 1)
    report = {
        "method": "v2xreal_camera_appearance_separability",
        "descriptors": args.descriptors, "descriptor_names": list(names),
        "split": str(args.root),
        "pairs_read": len(sampled),
        "probe_radius_m": PROBE_RADIUS_M, "min_crop_pixels": MIN_CROP_PIXELS, "max_occlusion": MAX_OCCLUSION,
        "pre_registered_bars": {"signal_auc": BAR_AUC, "coverage": BAR_COVERAGE, "far_field_coverage": BAR_FAR_COVERAGE},
        "opv2v_reference": {"ambiguous_auc_hue_sat": 0.598, "same_agent_bound": 0.707, "coverage": 0.392},
        "coverage": {
            "shared_objects": collected["coverage"]["shared"], "usable_in_both": collected["coverage"]["both"],
            "fraction": collected["coverage"]["both"] / shared,
            "by_range": {k: {**v, "fraction": v["both"] / max(v["shared"], 1)} for k, v in sorted(collected["by_range"].items())},
            "by_shared": {k: {**v, "fraction": v["both"] / max(v["shared"], 1)} for k, v in collected["by_shared_coverage"].items()},
        },
        "all_objects": _summaries(collected["scores"], names),
        "ambiguous": _summaries(collected["ambiguous"], names),
        "by_shared": {b: _summaries(collected["by_shared"][b], names) for b in SHARED_BUCKETS},
        "ambiguous_by_shared": {b: _summaries(collected["ambiguous_by_shared"][b], names) for b in SHARED_BUCKETS},
        "ambiguous_by_range": {k: _summaries(v, names) for k, v in sorted(collected["ambiguous_by_range"].items())},
        "same_agent_bound": _summaries(collected["same_agent"], names),
        "shuffled_control": _summaries(collected["shuffled"], names),
    }
    report["verdict"] = verdict(report, names)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["verdict"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
