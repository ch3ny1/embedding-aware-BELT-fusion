"""Figures of the CVPR draft, drawn from the result files in this repository.

    python scripts/plot_paper_figures.py --out-dir ../AlignFormer-2027/figures \\
        --teaser-frame outputs/v2xreal/teaser_frame_468.json \\
        --v2xreal-test outputs/v2xreal/B_boxes_only_icprw_test_result.json \\
        --diagnosis outputs/v2xreal/dense_resolve_diagnosis_val_result.json

Teaser (Fig. 1): one dense val frame at sigma 2 m beside AP@0.7 against sigma on
V2X-Real test. Residuals (Fig. 3): the ECDF of the answered translation
residual on the dense val pairs at sigma 2 m, one line per fit. Colours are
four categorical slots (blue, orange, aqua, violet) validated for colour-vision deficiency across every pair; black is reserved for
the oracle and grey for the uncorrected baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
GREY, INK, POINTS = "#8a8a85", "#1a1a19", "#d9d9d4"
SELECTED_ARM = "alignformer_abstain_0.2_agree_1_c4"
COLUMN_IN = 3.25
RESIDUAL_LINES = (  # key in the diagnosis file, label, colour, dashes
    ("residual_irls", "soft proposal", VIOLET, ()),
    ("residual_icp", "least-squares over its pairs", AQUA, ()),
    ("residual_freealign", "FreeAlign", ORANGE, ()),
    ("residual_icp_pairs_ransac", "consensus over the same pairs (ours)", BLUE, ()),
    ("residual_oracle_pairs_exact", "oracle pairs", INK, (3, 2)),
)


def ap_curve(result: Dict, condition: str, ap_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(sigmas, mean, standard error)`` of ``ap_key`` over the noise seeds, per sweep sigma.

    A condition without a sigma suffix in the file (the oracle) is repeated across the sweep."""
    sigmas = np.asarray(result["sweep_sigmas_m"], dtype=float)
    seeds = [result["ap_by_seed"][str(s)] for s in result["noise_seeds"]]

    def value(seed: Dict, sigma: float) -> float:
        key = f"{condition}_sigma_{sigma:g}m"
        return float(seed[key if key in seed else condition][ap_key]["global_sorted"])

    table = np.array([[value(seed, s) for s in sigmas] for seed in seeds])
    spread = table.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros(len(sigmas))
    return sigmas, table.mean(axis=0), spread / np.sqrt(len(seeds))


def ecdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    return x, np.arange(1, x.size + 1) / x.size


def residual_series(diagnosis: Dict, sigma_key: str) -> Dict[str, np.ndarray]:
    """Every ``residual_*`` kind in the diagnosis, over the pairs that report it."""
    rows = diagnosis["pairs"][sigma_key]
    keys = sorted({k for r in rows for k in r if k.startswith("residual_")})
    return {k: np.array([r[k] for r in rows if k in r], dtype=float) for k in keys}


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot  # noqa: F401 - registers the pyplot attribute used by the drawing functions

    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"],
        "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5, "legend.fontsize": 6.5, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "stix", "axes.edgecolor": GREY,
        "xtick.color": GREY, "ytick.color": GREY, "axes.labelcolor": INK, "text.color": INK, "legend.frameon": False,
    })
    return matplotlib


def _polygons(ax, boxes, colour, dashes=(), width=0.8, z=3):
    from matplotlib.collections import PolyCollection

    from dump_v2xreal_teaser_frame import box_corners

    kw = {"linestyles": [(0, dashes)] if dashes else "solid"}
    ax.add_collection(PolyCollection(box_corners(np.asarray(boxes)), facecolors="none", edgecolors=colour, linewidths=width, zorder=z, **kw))


def _scene_window(frame: Dict, margin_m: float = 3.0):
    """A square BEV window around the ego's and the CAV's (true) boxes."""
    centres = np.concatenate([np.asarray(frame["ego"]["boxes"])[:, :2], np.asarray(frame["cav"]["boxes_true"])[:, :2]])
    lo, hi = centres.min(0) - margin_m, centres.max(0) + margin_m
    side = float(max(hi - lo))
    centre = (lo + hi) / 2
    return centre, side


def _teaser_frame_panel(ax, frame: Dict):
    """Returns the legend handles; the legend itself is drawn at figure level."""
    from matplotlib.lines import Line2D

    points = frame["ego"].get("lidar_xy")
    if points:
        xy = np.asarray(points)
        ax.scatter(xy[:, 0], xy[:, 1], s=0.08, c="#e6e6e2", linewidths=0, zorder=1, rasterized=True)
    _polygons(ax, frame["ego"]["boxes"], INK, width=0.8)
    _polygons(ax, frame["cav"]["boxes_received"], ORANGE, dashes=(2.5, 1.5), width=0.9)
    _polygons(ax, frame["cav"]["boxes_alignformer"], BLUE, width=0.9, z=4)
    ax.plot([0], [0], marker="^", ms=4, color=INK, mec="none", zorder=5)
    centre, side = _scene_window(frame)
    ax.set_xlim(centre[0] - side / 2, centre[0] + side / 2)
    ax.set_ylim(centre[1] - side / 2, centre[1] + side / 2)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for side_name in ("top", "right", "bottom", "left"):
        ax.spines[side_name].set_visible(False)
    x0, y0 = centre[0] - side / 2 + 0.05 * side, centre[1] - side / 2 + 0.05 * side
    ax.plot([x0, x0 + 10], [y0, y0], color=INK, lw=0.8)
    ax.text(x0 + 5, y0 + 0.015 * side, "10 m", ha="center", va="bottom", fontsize=6)
    return [Line2D([], [], color=INK, lw=0.8, label="ego detections"),
            Line2D([], [], color=ORANGE, lw=0.9, dashes=(2.5, 1.5), label="CAV boxes as received"),
            Line2D([], [], color=BLUE, lw=0.9, label="CAV boxes after AlignFormer")]


def _teaser_ap_panel(ax, result: Dict):
    """Returns the legend handles; the legend itself is drawn at figure level."""
    series = (("uncorrected", "late fusion, uncorrected", GREY, ()), ("freealign", "FreeAlign", ORANGE, ()),
              (SELECTED_ARM, "AlignFormer (ours)", BLUE, ()), ("oracle", "true-pose oracle", INK, (3, 2)))
    handles = []
    for condition, label, colour, dashes in series:
        sigmas, mean, _ = ap_curve(result, condition, "ap_70")
        handles += ax.plot(sigmas, mean, color=colour, lw=1.2, dashes=dashes or (None, None), marker="o" if not dashes else None,
                           ms=2.2, mec="white", mew=0.4, label=label, zorder=3 if condition == SELECTED_ARM else 2)
    ax.set_xlabel("localization noise $\\sigma$ (m)")
    ax.set_ylabel("AP@0.7")
    ax.text(0.97, 0.9, "V2X-Real test", transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=INK)
    ax.set_xlim(-0.05, 2.05)
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.grid(axis="y", color="#ededea", lw=0.5)
    ax.set_axisbelow(True)
    for side_name in ("top", "right"):
        ax.spines[side_name].set_visible(False)
    return handles


def draw_teaser(frame: Dict, result: Dict, out: Path) -> None:
    plt = _style().pyplot
    fig, (left, right) = plt.subplots(1, 2, figsize=(COLUMN_IN, 1.7), gridspec_kw={"width_ratios": [1.0, 1.1], "wspace": 0.28})
    handles = _teaser_frame_panel(left, frame) + _teaser_ap_panel(right, result)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, handlelength=1.8, columnspacing=1.0,
               labelspacing=0.25, borderaxespad=0)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def draw_residuals(diagnosis: Dict, sigma_key: str, out: Path) -> None:
    """Fraction of answered pairs whose residual exceeds x: the tail is what separates the fits."""
    plt = _style().pyplot
    series = residual_series(diagnosis, sigma_key)
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 1.7))
    for key, label, colour, dashes in RESIDUAL_LINES:
        x, y = ecdf(series[key])
        ax.step(x, 1.0 - y, where="post", color=colour, lw=1.1, dashes=dashes or (None, None),
                label=f"{label} ({series[key].mean():.2f} m)")
    for bar in (0.5, 1.0):
        ax.axvline(bar, color="#ededea", lw=0.6, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(0.04, 60)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("translation residual $r$ after correction (m)")
    ax.set_ylabel("pairs with residual $> r$")
    ax.legend(loc="upper right", handlelength=1.8, labelspacing=0.25, borderaxespad=0.2, title="fit (mean residual)", title_fontsize=6.5, alignment="left")
    for side_name in ("top", "right"):
        ax.spines[side_name].set_visible(False)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--teaser-frame", type=Path, help="dump_v2xreal_teaser_frame.py output")
    parser.add_argument("--v2xreal-test", type=Path, help="the selected arm's V2X-Real test sweep")
    parser.add_argument("--diagnosis", type=Path, help="dense_resolve_diagnosis_val_result.json")
    parser.add_argument("--diagnosis-sigma", default="2")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.teaser_frame and args.v2xreal_test:
        draw_teaser(json.loads(args.teaser_frame.read_text()), json.loads(args.v2xreal_test.read_text()), args.out_dir / "teaser.pdf")
        print(f"wrote {args.out_dir / 'teaser.pdf'}")
    if args.diagnosis:
        draw_residuals(json.loads(args.diagnosis.read_text()), args.diagnosis_sigma, args.out_dir / "residuals.pdf")
        print(f"wrote {args.out_dir / 'residuals.pdf'}")


if __name__ == "__main__":
    main()
