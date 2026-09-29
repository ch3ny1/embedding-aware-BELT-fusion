"""Bytes per frame per agent for AlignFormer and for each intermediate baseline.

The project's claim is *robustness per byte*, so the byte axis has to be
measured rather than asserted -- on both sides.

**What counts as the message.** For an intermediate-fusion model it is the
smallest per-agent tensor from which the ego can reconstruct everything that
agent contributes to fusion. Every model here shares one encoder, so a
receiver holding the first tensor the fusion stage consumes can run the rest of
the shared chain itself; counting the deeper scales as well would charge the
baseline for bytes it does not need to send. This is therefore the
interpretation most favourable to the baselines, and it is measured with a
forward hook on the model's own fusion site rather than derived from the config.

For AlignFormer the message is per **object**: the 7-float box plus the
``embed_dim``-float embedding. The object count is not assumed -- it is counted
over the cached detections the sweep actually transmits, after the same
``MAX_OBJECTS`` truncation.

Precision is float32 on both sides, which is what the tensors are at runtime.
Neither side is credited with a quantizer that is not in the measured pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

BYTES_PER_FLOAT32 = 4

# Where each model's per-agent message can be observed. ``site`` is an attribute
# path from the model; ``when`` is "output" for a module whose output *is* the
# message, or "input" for a fusion module whose first argument is.
MESSAGE_SITES: Dict[str, Dict[str, Any]] = {
    "point_pillar_transformer": {
        "site": "shrink_conv", "when": "output",
        "what": "spatial_features_2d after the shrink header -- exactly what "
                "regroup() packs for the V2X-ViT fusion transformer.",
    },
    "point_pillar_cobevt": {
        "site": "shrink_conv", "when": "output",
        "what": "spatial_features_2d after the shrink header, the input to "
                "SwapFusionEncoder.",
    },
    "point_pillar_fcooper": {
        "site": "shrink_conv", "when": "output",
        "what": "spatial_features_2d after the shrink header, the input to "
                "SpatialFusion (maxout).",
    },
    "point_pillar_intermediate_V2VAM": {
        "site": "shrink_conv", "when": "output",
        "what": "spatial_features_2d after the shrink header.",
    },
    "point_pillar_coalign": {
        "site": "fusion_net.0", "when": "input",
        "what": "the first multiscale feature. The model file's own comment "
                "names this as the transmitted tensor; the deeper scales are "
                "derived from it by the shared backbone.",
    },
    "point_pillar_intermediate": {
        "site": "backbone.fuse_modules.0", "when": "input",
        "what": "the first scale AttBEVBackbone fuses. The block chain that "
                "produces the deeper scales is unfused, so a receiver can run "
                "it on this tensor alone.",
    },
    "point_pillar_where2comm": {
        "site": "fusion_net", "when": "input",
        "what": "the high-resolution spatial_features Where2comm fuses over "
                "(multi_scale), before its communication mask sparsifies it.",
    },
}


def _resolve(model, path: str):
    """Follow a dotted attribute/index path from the model to a submodule."""
    current = model
    for part in path.split("."):
        current = current[int(part)] if part.isdigit() else getattr(current, part)
    return current


def _feature_tensor(value):
    """Pull the BEV feature out of whatever the hooked site handed us."""
    if torch.is_tensor(value):
        return value
    if isinstance(value, dict):
        for key in ("spatial_features_2d", "spatial_features"):
            if key in value:
                return value[key]
    if isinstance(value, (tuple, list)) and value:
        return _feature_tensor(value[0])
    raise TypeError(f"no feature tensor at hook site: {type(value)}")


class MessageProbe:
    """Records the per-agent message size at a model's own fusion site."""

    def __init__(self, model, core_method: str) -> None:
        if core_method not in MESSAGE_SITES:
            raise KeyError(f"no message site registered for {core_method}")
        self.spec = MESSAGE_SITES[core_method]
        self.shapes: List[Tuple[int, ...]] = []
        self.agent_elements: List[int] = []
        module = _resolve(model, self.spec["site"])
        if self.spec["when"] == "output":
            self._handle = module.register_forward_hook(self._on_output)
        else:
            self._handle = module.register_forward_pre_hook(self._on_input)

    def _record(self, tensor) -> None:
        # (N_agents, C, H, W): the leading dim is the agent, so per-agent size
        # is the rest of the tensor.
        self.shapes.append(tuple(tensor.shape))
        self.agent_elements.append(int(np.prod(tensor.shape[1:])))

    def _on_output(self, module, inputs, output) -> None:
        self._record(_feature_tensor(output))

    def _on_input(self, module, inputs) -> None:
        self._record(_feature_tensor(inputs))

    def close(self) -> None:
        self._handle.remove()

    def summary(self) -> Dict[str, Any]:
        if not self.agent_elements:
            return {"frames": 0}
        elements = np.array(self.agent_elements, dtype=np.float64)
        return {
            "frames": int(elements.size),
            "site": self.spec["site"],
            "what": self.spec["what"],
            "tensor_shape": list(self.shapes[0]),
            "elements_per_agent": float(elements.mean()),
            "bytes_per_frame_per_agent": float(elements.mean() * BYTES_PER_FLOAT32),
        }


def alignformer_message_bytes(
    cache_root: Path, embed_dim: int, max_objects: int, box_floats: int = 7
) -> Dict[str, Any]:
    """Count AlignFormer's own message over the detections it actually sends.

    Reads only the ``boxes`` array's shape from each cached agent-frame, so the
    object count is the detector's real per-agent output after the sweep's
    ``MAX_OBJECTS`` truncation -- not a nominal 64.
    """
    counts: List[int] = []
    for path in sorted(cache_root.rglob("*.npz")):
        with np.load(path) as handle:
            counts.append(min(int(handle["boxes"].shape[0]), max_objects))
    if not counts:
        raise FileNotFoundError(f"no cached agent-frames under {cache_root}")

    objects = np.array(counts, dtype=np.float64)
    per_object = (box_floats + embed_dim) * BYTES_PER_FLOAT32
    return {
        "agent_frames": int(objects.size),
        "objects_per_agent_mean": float(objects.mean()),
        "objects_per_agent_median": float(np.median(objects)),
        "objects_per_agent_max": int(objects.max()),
        "truncated_at_max_objects": int((objects >= max_objects).sum()),
        "box_floats": box_floats,
        "embed_dim": embed_dim,
        "bytes_per_object": per_object,
        "bytes_per_frame_per_agent": float(objects.mean() * per_object),
        "bytes_per_frame_per_agent_boxes_only": float(
            objects.mean() * box_floats * BYTES_PER_FLOAT32
        ),
    }


# The three floats freealign.edge_features reads: centre x, centre y, yaw.
# A message with fewer is not one either method could align from.
MIN_BOX_FLOATS = 3


def alignment_overhead_bytes(
    objects_per_agent: float, embed_dim: int, box_floats: int = 7
) -> Dict[str, Any]:
    """Split one agent's message into the shared payload and the aligner's cost.

    Both methods here are LATE fusion: the ego cannot fuse a CAV detection
    without its full box, so the boxes are on the wire whatever aligns them.
    Charging them to the aligner would make every ratio in the report wrong,
    so they are reported as ``shared_box_bytes_per_frame_per_agent`` and
    attributed to neither.

    What separates the two methods is the remainder. FreeAlign adds nothing --
    it builds its relative-distance graph from the centres and yaws already in
    the boxes (``freealign.edge_features``, and its own docstring: "a
    boxes-only method"). AlignFormer as configured adds ``embed_dim`` floats
    per object, which at 128 is 512 bytes against the box's 28.

    Pass ``embed_dim=0`` for FreeAlign or for the boxes-only trunk; the two are
    then the same number by construction, which is the point.
    """
    if embed_dim < 0:
        raise ValueError(f"embed_dim must be non-negative, got {embed_dim}")
    if box_floats < MIN_BOX_FLOATS:
        raise ValueError(
            f"box_floats must be at least {MIN_BOX_FLOATS} (centre x, centre y "
            f"and yaw are what alignment reads), got {box_floats}"
        )
    shared = float(objects_per_agent) * box_floats * BYTES_PER_FLOAT32
    overhead = float(objects_per_agent) * embed_dim * BYTES_PER_FLOAT32
    return {
        "objects_per_agent": float(objects_per_agent),
        "box_floats": box_floats,
        "embed_dim": embed_dim,
        "shared_box_bytes_per_frame_per_agent": shared,
        "alignment_overhead_bytes_per_frame_per_agent": overhead,
        "total_bytes_per_frame_per_agent": shared + overhead,
    }


@torch.no_grad()
def measure_baseline(baseline: str, split: Path, pcd_cache: Path, frames: int, device):
    """Run ``frames`` frames of one baseline at sigma 0 with the probe attached."""
    from embedding_aware_belt_fusion.alignformer.baselines import (
        BASELINES,
        build_sample,
        install_pcd_cache,
        load_baseline,
    )
    from embedding_aware_belt_fusion.alignformer.baselines import _to_device
    from embedding_aware_belt_fusion.alignformer.evaluate import _frame_identity

    install_pcd_cache(split.resolve(), pcd_cache.resolve())
    spec = BASELINES[baseline]
    hypes, dataset, model, checkpoint, _ = load_baseline(spec, split, device)
    core_method = hypes["model"]["core_method"]
    probe = MessageProbe(model, core_method)

    agents: List[int] = []
    communication_rates: List[float] = []
    cur_ego_pose_flag = getattr(dataset, "cur_ego_pose_flag", True)
    for index in range(min(frames, len(dataset))):
        scenario, timestamp = _frame_identity(dataset, index)
        base = dataset.retrieve_base_data(index, cur_ego_pose_flag=cur_ego_pose_flag)
        sample = build_sample(dataset, base, scenario, timestamp)
        agents.append(int(sample["ego"]["cav_num"]))
        batch = _to_device(dataset.collate_batch_test([sample]), device)
        output = model(batch["ego"])
        if "com" in output and output["com"] is not None:
            rate = output["com"]
            communication_rates.append(
                float(rate.mean()) if torch.is_tensor(rate) else float(rate)
            )

    summary = probe.summary()
    probe.close()
    summary.update(
        {
            "baseline": baseline,
            "label": spec["label"],
            "model_core_method": core_method,
            "checkpoint": str(checkpoint),
            "cav_lidar_range": hypes["preprocess"]["cav_lidar_range"],
            "voxel_size": hypes["preprocess"]["args"]["voxel_size"],
            "agents_per_frame_mean": float(np.mean(agents)) if agents else None,
        }
    )
    if communication_rates:
        rate = float(np.mean(communication_rates))
        summary["communication_rate"] = rate
        summary["bytes_per_frame_per_agent_after_selection"] = (
            summary["bytes_per_frame_per_agent"] * rate
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", nargs="*", default=None)
    parser.add_argument(
        "--split", type=Path, default=Path("/media/chenyi/Elements1/Dataset/OPV2V/test")
    )
    parser.add_argument(
        "--pcd-cache", type=Path,
        default=Path("/media/chenyi/basement2/cache/opv2v_coloca/test"),
    )
    parser.add_argument(
        "--alignformer-cache", type=Path,
        default=Path("/media/chenyi/basement2/cache/alignformer/test"),
    )
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--max-objects", type=int, default=64)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    from embedding_aware_belt_fusion.alignformer.baselines import BASELINES

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result: Dict[str, Any] = {
        "metric": "bytes_per_frame_per_agent",
        "frames_measured": args.frames,
        "precision": "float32",
        "alignformer": alignformer_message_bytes(
            args.alignformer_cache, args.embed_dim, args.max_objects
        ),
        "baselines": {},
        "failed": {},
    }
    print(
        "AlignFormer: {:.0f} B/frame/agent over {:.2f} objects".format(
            result["alignformer"]["bytes_per_frame_per_agent"],
            result["alignformer"]["objects_per_agent_mean"],
        ),
        flush=True,
    )

    names = args.baselines or [b for b in sorted(BASELINES) if b != "ermvp"]
    for name in names:
        try:
            summary = measure_baseline(
                name, args.split, args.pcd_cache, args.frames, device
            )
        except Exception as error:  # noqa: BLE001 - a failed row is reported, not dropped
            result["failed"][name] = f"{type(error).__name__}: {error}"
            print(f"{name}: FAILED {type(error).__name__}: {error}", flush=True)
            continue
        result["baselines"][name] = summary
        print(
            "{:<12} {:>12.0f} B/frame/agent  shape {}".format(
                name, summary["bytes_per_frame_per_agent"], summary["tensor_shape"]
            ),
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
