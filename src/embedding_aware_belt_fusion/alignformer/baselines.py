"""Intermediate-fusion baselines under AlignFormer's own localization-noise sweep.

The project's central claim is *robustness per byte*: under one noise sweep,
one evaluator and one split, AlignFormer's late fusion should retain more
AP@0.7 than intermediate-fusion methods that transmit dense BEV feature maps.
Published numbers cannot be quoted beside ours -- different detectors, splits
and noise conventions make that meaningless -- so every baseline is run here,
locally, on the same 2170-frame OPV2V test split, with the same
``sigma_yaw(deg) = sigma_xy(m)`` perturbation applied to the CAV pose only, and
scored by the same ``fusion.average_precision``.

**Where the noise enters.** Every OpenCOOD intermediate-fusion model in this
repository runs with ``proj_first = True``: each CAV projects its own point
cloud into the ego frame with ``params['transformation_matrix']`` *before*
voxelization, so that matrix is the single place a localization error can act.
This module recomputes it from a perturbed CAV pose and leaves everything else
-- ``lidar_pose`` (so the ``COM_RANGE`` selection is unchanged), ``vehicles``
(so the ground truth is unchanged) -- exactly as OpenCOOD produced it.

**Why not OpenCOOD's own ``wild_setting``.** ``BaseDataset.add_loc_noise``
calls ``np.random.seed(self.seed)`` with a *constant* seed on every invocation,
so every CAV in every frame receives the identical displacement, and it also
perturbs ``z``. Neither matches this project's convention. Any ``wild_setting``
found in a baseline's config is therefore disabled here, and the fact is
recorded in the result JSON.

**Pairing with AlignFormer.** The perturbation is drawn from
``noisy_fusion._sweep_rng(seed, sigma, frame, agent)`` with the same seed and
the same agent ordering the AlignFormer sweep uses, so a baseline and
AlignFormer see byte-identical displacements on the same frame. The comparison
is paired, not merely matched in distribution.

Nothing under ``external/OpenCOOD`` is modified: the dataset's own
``get_item_single_car``, ``collate_batch_test``, ``post_process`` and
``generate_gt_bbx`` are called verbatim.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

MODELS_ROOT = Path("/media/chenyi/Elements1/models/opv2v")

# One entry per baseline: where its config and weights live, and what the
# weights actually are. ``trained_with_pose_noise`` is the honesty field --
# a method designed for pose error that was nevertheless trained clean is not
# being shown at its best, and the comparison must say so.
BASELINES: Dict[str, Dict[str, Any]] = {
    "v2xvit": {
        "label": "V2X-ViT",
        "config": MODELS_ROOT / "pointpillar_v2xvit" / "config.yaml",
        "checkpoint": MODELS_ROOT / "pointpillar_v2xvit" / "net_epoch60.pth",
        "message": "spatial_features_2d",
        "notes": "OpenCOOD point_pillar_transformer (V2X-ViT). Its own config "
                 "carries wild_setting async=True loc_err=True xyz_std=0.2 "
                 "ryp_std=0.2, i.e. it was trained/evaluated under OpenCOOD's "
                 "noisy setting; that setting is disabled for this sweep.",
    },
    "coalign": {
        "label": "CoAlign (fusion only)",
        "config": MODELS_ROOT / "pointpillar_coalign" / "point_pillar_coalign" / "config.yaml",
        "checkpoint": None,  # resolved to the newest net_epoch*.pth
        "message": "backbone_layer0",
        "notes": "OpenCOOD point_pillar_coalign. The model file states in its "
                 "header that it contains CoAlign's multiscale intermediate "
                 "feature fusion ONLY, not the agent-object pose graph that is "
                 "CoAlign's pose-error correction. This row is therefore "
                 "CoAlign's fusion, not CoAlign's robustness mechanism.",
    },
    "cobevt": {
        "label": "CoBEVT",
        "config": MODELS_ROOT / "pointpillar_cobevt" / "no_compression" / "config.yaml",
        "checkpoint": MODELS_ROOT / "pointpillar_cobevt" / "no_compression" / "net_epoch19.pth",
        "message": "spatial_features_2d",
        "notes": "OpenCOOD point_pillar_cobevt, no-compression variant.",
    },
    "attfuse": {
        "label": "AttFuse",
        "config": MODELS_ROOT / "pointpillar_attentive_fusion" / "config.yaml",
        "checkpoint": MODELS_ROOT / "pointpillar_attentive_fusion" / "latest.pth",
        "message": "spatial_features",
        "notes": "OpenCOOD point_pillar_intermediate (AttFuse / the OPV2V "
                 "paper's attentive fusion).",
    },
    "where2comm": {
        "label": "Where2comm",
        "config": MODELS_ROOT / "pointpillar_where2comm" / "config.yaml",
        "checkpoint": MODELS_ROOT / "pointpillar_where2comm" / "net_epoch50.pth",
        "message": "spatial_features",
        "notes": "OpenCOOD point_pillar_where2comm. Its config carries "
                 "wild_setting async=True loc_err=True; disabled for this sweep.",
    },
    "fcooper": {
        "label": "F-Cooper",
        "config": MODELS_ROOT / "f_cooper" / "config.yaml",
        "checkpoint": MODELS_ROOT / "f_cooper" / "latest.pth",
        "message": "spatial_features_2d",
        "notes": "OpenCOOD point_pillar_fcooper (maxout spatial fusion).",
    },
    "v2vam": {
        "label": "V2VAM",
        "config": MODELS_ROOT / "pointpillar_v2vam" / "no-compression" / "config.yaml",
        "checkpoint": None,
        "message": "spatial_features_2d",
        "notes": "OpenCOOD point_pillar_intermediate_V2VAM, no-compression.",
    },
    "ermvp": {
        "label": "ERMVP",
        "config": MODELS_ROOT / "ermvp" / "config.yaml",
        "checkpoint": MODELS_ROOT / "ermvp" / "net_epoch19.pth",
        "message": "spatial_features_2d",
        "notes": "point_pillar_ermvp; may not exist in this OpenCOOD fork.",
    },
}


def resolve_checkpoint(spec: Dict[str, Any]) -> Path:
    """Return the weights path, picking the highest ``net_epoch*.pth`` if unset."""
    if spec["checkpoint"] is not None:
        return Path(spec["checkpoint"])
    directory = Path(spec["config"]).parent
    candidates = sorted(
        directory.glob("net_epoch*.pth"),
        key=lambda path: int("".join(c for c in path.stem if c.isdigit()) or 0),
    )
    if candidates:
        return candidates[-1]
    latest = directory / "latest.pth"
    if latest.exists():
        return latest
    raise FileNotFoundError(f"no net_epoch*.pth or latest.pth beside {spec['config']}")


def install_pcd_cache(split_root: Path, cache_root: Path) -> int:
    """Read point clouds from the binary ``.npy`` mirror instead of the ASCII pcd.

    OPV2V ships ~3 MB ASCII ``.pcd`` files on a USB spinning disk here. The
    mirror written by ``coloca.pcd_cache`` is byte-for-byte what
    ``pcd_to_np`` returns, and this swaps it in *without touching OpenCOOD* by
    rebinding the attribute the dataset resolves at call time. Returns the
    number of cached files found, or 0 if the mirror is absent (in which case
    nothing is patched and the ASCII reader stands).
    """
    import opencood.utils.pcd_utils as pcd_utils
    from embedding_aware_belt_fusion.coloca.pcd_cache import cached_pcd_path

    if not cache_root.is_dir():
        return 0

    original = pcd_utils.pcd_to_np

    def cached_pcd_to_np(pcd_path):
        try:
            cached = cached_pcd_path(cache_root, Path(pcd_path), split_root)
        except ValueError:
            return original(pcd_path)
        if cached.exists():
            return np.load(cached)
        return original(pcd_path)

    pcd_utils.pcd_to_np = cached_pcd_to_np
    return sum(1 for _ in cache_root.rglob("*.npy"))


# ``point_pillar_intermediate_V2VAM`` opens with two imports this OpenCOOD fork
# cannot satisfy -- ``sub_modules.noise`` does not exist, and
# ``fuse_modules.self_attn`` has been trimmed of ``regroup`` -- and then uses
# neither. The imports alone make V2VAM unloadable.
_V2VAM_DEAD_IMPORTS = (
    ("opencood.models.sub_modules.noise",
     ("data_dropout", "data_dropout_uniform", "transmission_with_noise")),
    ("opencood.models.fuse_modules.self_attn", ("regroup",)),
)


def _shim_missing_v2vam_import() -> List[str]:
    """Supply the dead imports ``point_pillar_intermediate_V2VAM`` makes.

    Every name involved is verified dead: none appears anywhere in that model
    file below its import line. Binding stand-ins is therefore what lets the
    row be *measured* rather than silently dropped, and it is safe precisely
    because the stubs raise -- a V2VAM that ever called one would fail loudly
    instead of running a fabricated noise model. ``external/OpenCOOD`` is not
    modified; the names are bound in memory. Returns the names shimmed so the
    result JSON can declare them.
    """
    import importlib
    import sys
    import types

    def unavailable(*_args, **_kwargs):
        raise NotImplementedError(
            "this symbol is absent from this OpenCOOD fork and was stubbed "
            "only because point_pillar_intermediate_V2VAM's import of it is dead"
        )

    shimmed: List[str] = []
    for module_name, symbols in _V2VAM_DEAD_IMPORTS:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            module = types.ModuleType(module_name)
            sys.modules[module_name] = module
        for symbol in symbols:
            if not hasattr(module, symbol):
                setattr(module, symbol, unavailable)
                shimmed.append(f"{module_name}.{symbol}")
    return shimmed


def load_baseline(spec: Dict[str, Any], split: Path, device):
    """Build the baseline's dataset and load its trained weights.

    ``wild_setting`` is stripped before the dataset is constructed so that
    OpenCOOD's own (constant-seed, z-perturbing) localization noise and its
    asynchrony model cannot run alongside this sweep's perturbation.
    """
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.tools.train_utils import create_model

    hypes = load_yaml(str(spec["config"]), None)
    removed_wild_setting = hypes.pop("wild_setting", None)
    hypes["validate_dir"] = str(split.resolve())
    hypes["root_dir"] = str(split.resolve())

    dataset = build_dataset(hypes, visualize=False, train=False)

    shimmed = _shim_missing_v2vam_import()
    model = create_model(hypes)
    checkpoint_path = resolve_checkpoint(spec)
    state = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(state, dict):
        state = state.get("model_state_dict", state)
    model.load_state_dict(state)
    model = model.to(device).eval()

    if shimmed:
        removed_wild_setting = dict(removed_wild_setting or {})
        removed_wild_setting["_shimmed_dead_imports"] = shimmed
    return hypes, dataset, model, checkpoint_path, removed_wild_setting


def noisy_transforms(
    base_data_dict: "OrderedDict",
    ego_id,
    ego_pose: Sequence[float],
    *,
    sigma: float,
    seed: int,
    frame: int,
) -> Dict[Any, np.ndarray]:
    """CAV-to-ego matrices recomputed from perturbed CAV poses.

    The agent ordering and the RNG are ``noisy_fusion``'s, so the displacement
    a CAV receives here is the displacement AlignFormer's own sweep gave it on
    the same frame at the same sigma. That requires enumerating exactly the
    agents AlignFormer enumerates: ``_build_test_frame`` drops every CAV beyond
    ``COM_RANGE`` *before* sorting, so an out-of-range CAV left in the ordering
    here would shift every later agent's draw and silently unpair the two runs.
    """
    import opencood.data_utils.datasets as opencood_datasets
    from opencood.utils.transformation_utils import x1_to_x2

    from embedding_aware_belt_fusion.alignformer.dataset import YAW_STD_PER_XY_STD
    from embedding_aware_belt_fusion.alignformer.noisy_fusion import _sweep_rng
    from embedding_aware_belt_fusion.coloca.geometry import perturb_pose_2d

    def in_range(cav_id) -> bool:
        pose = base_data_dict[cav_id]["params"]["lidar_pose"]
        distance = math.hypot(pose[0] - ego_pose[0], pose[1] - ego_pose[1])
        return distance <= opencood_datasets.COM_RANGE

    others = sorted(
        (key for key in base_data_dict if key != ego_id and in_range(key)), key=str
    )
    transforms: Dict[Any, np.ndarray] = {}
    for agent, cav_id in enumerate(others):
        pose = base_data_dict[cav_id]["params"]["lidar_pose"]
        rng = _sweep_rng(seed, sigma, frame, agent)
        noisy = perturb_pose_2d(pose, sigma, sigma * YAW_STD_PER_XY_STD, rng)
        transforms[cav_id] = x1_to_x2(noisy, list(ego_pose))
    return transforms


def _apply_transforms(base_data_dict, transforms: Dict[Any, np.ndarray]):
    """A shallow copy of the frame with the named CAVs' transforms replaced.

    Copies rather than mutates: the same ``base_data_dict`` is reused for every
    sigma of the frame, and an in-place write would leak one level's noise into
    the next.
    """
    perturbed = OrderedDict()
    for cav_id, content in base_data_dict.items():
        entry = dict(content)
        if cav_id in transforms:
            params = dict(content["params"])
            params["transformation_matrix"] = transforms[cav_id]
            entry["params"] = params
        perturbed[cav_id] = entry
    return perturbed


def build_sample(dataset, base_data_dict, scenario: str, timestamp: str) -> Dict:
    """Mirror ``IntermediateFusionDataset.__getitem__`` on a supplied frame.

    Two deliberate differences, and no others: the frame is passed in rather
    than re-loaded (one point-cloud decode serves every sigma), and ``np.random``
    is reseeded per agent from ``(scenario, cav_id, timestamp)`` -- the seed
    formula ``alignformer.cache`` and ``evaluate._build_test_frame`` already use
    -- so ``shuffle_points`` inside ``get_item_single_car`` draws the same
    permutation at every sigma. Without that reseed the voxelization would
    differ between noise levels and the sweep would measure shuffling.
    """
    import opencood.data_utils.datasets as opencood_datasets

    from embedding_aware_belt_fusion.alignformer.cache import frame_seed

    ego_id, ego_lidar_pose = None, None
    for cav_id, cav_content in base_data_dict.items():
        if cav_content["ego"]:
            ego_id, ego_lidar_pose = cav_id, cav_content["params"]["lidar_pose"]
            break
    if ego_id is None:
        raise RuntimeError(f"{scenario}/{timestamp} has no ego vehicle")

    pairwise_t_matrix = dataset.get_pairwise_transformation(
        base_data_dict, dataset.max_cav
    )

    processed_features: List[Dict] = []
    object_stack: List[np.ndarray] = []
    object_id_stack: List[Any] = []
    velocity: List[float] = []
    time_delay: List[float] = []
    infra: List[float] = []
    spatial_correction_matrix: List[np.ndarray] = []

    for cav_id, selected_cav_base in base_data_dict.items():
        pose = selected_cav_base["params"]["lidar_pose"]
        distance = math.hypot(pose[0] - ego_lidar_pose[0], pose[1] - ego_lidar_pose[1])
        if distance > opencood_datasets.COM_RANGE:
            continue

        np.random.seed(frame_seed(scenario, cav_id, timestamp))
        processed = dataset.get_item_single_car(selected_cav_base, ego_lidar_pose)

        object_stack.append(processed["object_bbx_center"])
        object_id_stack += processed["object_ids"]
        processed_features.append(processed["processed_features"])
        velocity.append(processed["velocity"])
        time_delay.append(float(selected_cav_base["time_delay"]))
        spatial_correction_matrix.append(
            selected_cav_base["params"]["spatial_correction_matrix"]
        )
        infra.append(1 if int(cav_id) < 0 else 0)

    unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]
    object_stack = np.vstack(object_stack)[unique_indices]

    max_num = dataset.params["postprocess"]["max_num"]
    object_bbx_center = np.zeros((max_num, 7))
    mask = np.zeros(max_num)
    object_bbx_center[: object_stack.shape[0], :] = object_stack
    mask[: object_stack.shape[0]] = 1

    anchor_box = dataset.post_processor.generate_anchor_box()
    label_dict = dataset.post_processor.generate_label(
        gt_box_center=object_bbx_center, anchors=anchor_box, mask=mask
    )

    pad = dataset.max_cav - len(velocity)
    if pad < 0:
        raise RuntimeError(
            f"{scenario}/{timestamp}: {len(velocity)} in-range CAVs exceeds "
            f"max_cav={dataset.max_cav}"
        )
    spatial_correction_matrix = np.concatenate(
        [np.stack(spatial_correction_matrix), np.tile(np.eye(4)[None], (pad, 1, 1))],
        axis=0,
    )

    return {
        "ego": {
            "object_bbx_center": object_bbx_center,
            "object_bbx_mask": mask,
            "object_ids": [object_id_stack[i] for i in unique_indices],
            "anchor_box": anchor_box,
            "processed_lidar": dataset.merge_features_to_dict(processed_features),
            "label_dict": label_dict,
            "cav_num": len(processed_features),
            "velocity": velocity + pad * [0.0],
            "time_delay": time_delay + pad * [0.0],
            "infra": infra + pad * [0.0],
            "spatial_correction_matrix": spatial_correction_matrix,
            "pairwise_t_matrix": pairwise_t_matrix,
        }
    }


def late_fusion_convention_ground_truth(dataset, base_data_dict, ego_id, ego_pose):
    """The ground truth ``LateFusionDataset`` would report for this frame.

    OpenCOOD builds the two fusion families' ground truth differently, and the
    difference is not cosmetic. ``generate_object_center`` filters objects
    against ``GT_RANGE`` **in the reference agent's own frame, in 3D** -- so the
    box is rotated with that agent's heading and clipped in z -- while the final
    ``generate_gt_bbx`` mask checks x and y only. Late fusion references each
    CAV to *itself*; intermediate fusion references every CAV to the *ego*. On a
    sloped or sharply-turned frame the two admit different object sets.

    AlignFormer's published AP is on the late-fusion set. Reproducing that set
    here, from the same frame the baseline is scored on, is what makes one
    common ground truth possible; the native (ego-referenced) set is reported
    beside it so the choice is visible rather than assumed.
    """
    import opencood.data_utils.datasets as opencood_datasets
    from opencood.utils import box_utils
    from opencood.utils.transformation_utils import x1_to_x2

    order = dataset.post_processor.params["order"]
    corner_list, object_ids = [], []
    for cav_id, content in base_data_dict.items():
        pose = content["params"]["lidar_pose"]
        if math.hypot(pose[0] - ego_pose[0], pose[1] - ego_pose[1]) > \
                opencood_datasets.COM_RANGE:
            continue
        centers, mask, ids = dataset.post_processor.generate_object_center(
            [content], pose
        )
        centers = centers[mask == 1]
        if centers.shape[0] == 0:
            continue
        corners = box_utils.boxes_to_corners_3d(
            torch.from_numpy(centers).float(), order
        )
        transform = torch.from_numpy(np.asarray(x1_to_x2(pose, list(ego_pose)))).float()
        corner_list.append(box_utils.project_box3d(corners, transform))
        object_ids += ids

    if not corner_list:
        return torch.zeros((0, 7), dtype=torch.float32)

    stacked = torch.vstack(corner_list)
    unique = [object_ids.index(x) for x in set(object_ids)]
    stacked = stacked[unique]
    stacked = stacked[box_utils.get_mask_for_boxes_within_range_torch(stacked)]
    return _centers_from_corners(stacked, order)


def _to_device(batch: Dict, device) -> Dict:
    moved = {}
    for key, value in batch["ego"].items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        elif isinstance(value, dict):
            moved[key] = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in value.items()
            }
        else:
            moved[key] = value
    return {"ego": moved}


def _centers_from_corners(corners, order: str):
    """OpenCOOD corner output -> the ``(N, 7)`` centre form the AP code takes."""
    from opencood.utils import box_utils

    if corners is None or corners.shape[0] == 0:
        return torch.zeros((0, 7), dtype=torch.float32)
    centers = box_utils.corner_to_center(corners.detach().cpu().numpy(), order=order)
    return torch.from_numpy(centers).float()


@torch.no_grad()
def run_baseline_noise_sweep(
    dataset,
    model,
    device,
    *,
    sigmas: Sequence[float],
    seed: int,
    max_frames: Optional[int] = None,
    progress_interval: int = 100,
) -> Tuple[Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]], Dict[str, List[torch.Tensor]], Dict]:
    """Fuse and decode every frame at every sigma; return predictions and truth.

    Two ground truths come back, keyed ``native`` (OpenCOOD's intermediate
    convention, every CAV's objects referenced to the ego) and ``late_fusion``
    (each CAV's objects referenced to itself, which is what AlignFormer's
    numbers are on). Scoring against both is what lets one table hold both
    families without either being quietly re-based.

    Each is built once per frame and asserted invariant across sigmas: the
    perturbation touches only the CAV-to-ego transform, never the object list
    or the ego pose, so a ground truth that moved would mean the noise had
    leaked somewhere it must not.
    """
    from opencood.utils import box_utils

    from embedding_aware_belt_fusion.alignformer.evaluate import _frame_identity

    order = dataset.post_processor.params["order"]
    predictions: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {
        f"sigma_{sigma:g}m": [] for sigma in sigmas
    }
    ground_truth: Dict[str, List[torch.Tensor]] = {"native": [], "late_fusion": []}
    cur_ego_pose_flag = getattr(dataset, "cur_ego_pose_flag", True)

    frame_count = len(dataset) if max_frames is None else min(max_frames, len(dataset))
    started = time.time()
    empty_frames = 0
    gt_checks = 0

    for index in range(frame_count):
        scenario, timestamp = _frame_identity(dataset, index)
        base = dataset.retrieve_base_data(index, cur_ego_pose_flag=cur_ego_pose_flag)

        ego_id, ego_pose = None, None
        for cav_id, content in base.items():
            if content["ego"]:
                ego_id, ego_pose = cav_id, content["params"]["lidar_pose"]
                break

        frame_gt = None
        for sigma in sigmas:
            transforms = noisy_transforms(
                base, ego_id, ego_pose, sigma=sigma, seed=seed, frame=index
            )
            sample = build_sample(
                dataset, _apply_transforms(base, transforms), scenario, timestamp
            )
            batch = _to_device(dataset.collate_batch_test([sample]), device)

            output = model(batch["ego"])
            corners, scores = dataset.post_processor.post_process(
                batch, {"ego": output}
            )
            if corners is None:
                empty_frames += 1
                boxes = torch.zeros((0, 7), dtype=torch.float32)
                scores = torch.zeros((0,), dtype=torch.float32)
            else:
                boxes = _centers_from_corners(corners, order)
                scores = scores.detach().cpu().float()
            predictions[f"sigma_{sigma:g}m"].append((boxes, scores))

            gt_corners = dataset.post_processor.generate_gt_bbx(batch)
            gt_boxes = torch.from_numpy(
                box_utils.corner_to_center(gt_corners.detach().cpu().numpy(), order=order)
            ).float()
            if frame_gt is None:
                frame_gt = gt_boxes
            else:
                gt_checks += 1
                if not torch.allclose(frame_gt, gt_boxes, atol=1e-5):
                    raise RuntimeError(
                        f"frame {index} ground truth moved with sigma={sigma}"
                    )
        ground_truth["native"].append(frame_gt)
        ground_truth["late_fusion"].append(
            late_fusion_convention_ground_truth(dataset, base, ego_id, ego_pose)
        )

        if (index + 1) % progress_interval == 0:
            rate = (index + 1) / (time.time() - started)
            print(f"  {index + 1}/{frame_count} frames  {rate:.2f} fr/s", flush=True)

    return (
        predictions,
        ground_truth,
        {
            "frames": len(ground_truth["native"]),
            "native_gt_boxes": sum(int(g.shape[0]) for g in ground_truth["native"]),
            "late_fusion_gt_boxes": sum(
                int(g.shape[0]) for g in ground_truth["late_fusion"]
            ),
            "frames_where_gt_conventions_differ": sum(
                1
                for a, b in zip(ground_truth["native"], ground_truth["late_fusion"])
                if a.shape[0] != b.shape[0]
            ),
            "empty_prediction_cells": empty_frames,
            "ground_truth_invariance_checks": gt_checks,
            "seconds": time.time() - started,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, choices=sorted(BASELINES))
    parser.add_argument(
        "--split", type=Path, default=Path("/media/chenyi/Elements1/Dataset/OPV2V/test")
    )
    parser.add_argument(
        "--pcd-cache", type=Path, default=Path("/media/chenyi/basement2/cache/opv2v_coloca/test")
    )
    parser.add_argument(
        "--sweep", type=float, nargs="+", default=[0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    from embedding_aware_belt_fusion.alignformer.evaluate import _ap_report, _json_safe

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = BASELINES[args.baseline]

    cached = install_pcd_cache(args.split.resolve(), args.pcd_cache.resolve())
    print(f"pcd cache: {cached} npy files", flush=True)

    hypes, dataset, model, checkpoint_path, wild = load_baseline(spec, args.split, device)
    print(
        f"{spec['label']}: {len(dataset)} frames, model "
        f"{hypes['model']['core_method']}, weights {checkpoint_path.name}",
        flush=True,
    )

    predictions, ground_truth, stats = run_baseline_noise_sweep(
        dataset, model, device,
        sigmas=list(args.sweep), seed=args.seed, max_frames=args.max_frames,
    )

    ap: Dict[str, Dict[str, Any]] = {}
    for convention, truth in ground_truth.items():
        ap[convention] = {}
        for name, frames in predictions.items():
            ap[convention][name] = _ap_report(frames, truth)
            report = ap[convention][name]
            print(
                f"  [{convention:<11}] {name:<14} "
                f"AP@0.3={report['ap_30']['global_sorted']:.4f} "
                f"AP@0.5={report['ap_50']['global_sorted']:.4f} "
                f"AP@0.7={report['ap_70']['global_sorted']:.4f}",
                flush=True,
            )

    result = {
        "method": args.baseline,
        "label": spec["label"],
        "metric": "baseline_noisy_ap",
        "notes": spec["notes"],
        "config": str(spec["config"]),
        "checkpoint": str(checkpoint_path),
        "model_core_method": hypes["model"]["core_method"],
        "fusion_core_method": hypes["fusion"]["core_method"],
        "disabled_wild_setting": wild,
        "cav_lidar_range": hypes["preprocess"]["cav_lidar_range"],
        "score_threshold": hypes["postprocess"]["target_args"]["score_threshold"],
        "nms_thresh": hypes["postprocess"]["nms_thresh"],
        "max_cav": dataset.max_cav,
        "split": str(args.split),
        "sweep_sigmas_m": list(args.sweep),
        "seed": args.seed,
        "run": stats,
        "ground_truth_conventions": {
            "native": "OpenCOOD IntermediateFusionDataset: every CAV's objects "
                      "range-filtered in the EGO frame.",
            "late_fusion": "OpenCOOD LateFusionDataset: every CAV's objects "
                           "range-filtered in that CAV's OWN frame, then "
                           "projected to ego. This is the convention "
                           "AlignFormer's AP is measured under.",
        },
        "ap": ap,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_safe(result), indent=2))
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
