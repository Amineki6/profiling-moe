#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publication-grade figures for the 2-block Kolibri E2E benchmark (11_profile_2block_e2e.py).

Single-column layout (3.5 in wide), vector PDF with embedded TrueType fonts + high-dpi PNG.
Palette and font stack are imported from generate_figures.py so all repo figures match.

Statistics are computed from the per-iteration samples (<stem>_raw.csv), never from mean ± std:
  - latencies:  median with interquartile range
  - speedups:   median of *paired* per-iteration ratios (baseline_i / fused_i). Iterations are
                interleaved by the benchmark, so each pair ran back-to-back at the same clock and
                residual clock drift cancels. 95% CIs via percentile bootstrap.

Figures:
  fig1_region_latency   per-region latency, baseline vs fused (R1, R2, R3)
  fig2_region_speedup   per-region + total fusion speedup vs N
  fig3_e2e_breakdown    E2E time breakdown normalised to baseline (Amdahl view)
  fig4_amdahl           measured E2E speedup vs Amdahl prediction and bound

Usage:
  python benchmarks/generate_e2e_figures.py
  python benchmarks/generate_e2e_figures.py --primary free --formats pdf png svg
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, ticker
from matplotlib.patches import Patch

from generate_figures import (
    BG_COLOR,
    TEXT_COLOR,
    MUTED_TEXT,
    SPINE_COLOR,
    COLOR_TAUPE_EDGE,
    COLOR_TAUPE_FILL,
    COLOR_AMBER_EDGE,
    COLOR_AMBER_FILL,
    COLOR_TEAL_EDGE,
    COLOR_TEAL_FILL,
    COLOR_BERRY_EDGE,
    COLOR_BERRY_FILL,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
OUT_DIR = REPO_ROOT / "assets" / "e2e"
FONT_DIR = REPO_ROOT / "assets" / "fonts"  # optional: drop Inter-*.ttf here for a pinned font

COL_W = 3.5  # single-column width (in)

# Single-column sizing on top of the shared repo theme
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.labelweight": "semibold",
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.size": 0,
    "ytick.minor.size": 1.5,
    "ytick.minor.width": 0.5,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "pdf.fonttype": 42,  # embed TrueType (camera-ready requirement for most venues)
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "savefig.facecolor": BG_COLOR,
})

# (region key, legend label, edge colour, fill colour, marker)
REGIONS = [
    ("R1_sandwich_norm", "R1 Sandwich norm", COLOR_TEAL_EDGE, COLOR_TEAL_FILL, "o"),
    ("R2_moe_prologue", "R2 MoE prologue", COLOR_BERRY_EDGE, COLOR_BERRY_FILL, "s"),
    ("R3_post_moe_norm", "R3 Post-MoE norm", COLOR_AMBER_EDGE, COLOR_AMBER_FILL, "^"),
]
FUSION = [r[0] for r in REGIONS]
N_BOOT = 10_000
RNG = np.random.default_rng(0)


# ----------------------------------------------------------------------------- data

def load_raw(path):
    """{(n, region): {variant: samples ordered by iteration}} + synthetic 'fusion_total'."""
    acc = defaultdict(lambda: defaultdict(dict))
    with open(path) as f:
        for r in csv.DictReader(f):
            acc[(int(r["n"]), r["region"])][r["variant"]][int(r["iter"])] = float(r["us"])
    data = {
        key: {v: np.array([it[i] for i in sorted(it)]) for v, it in by_v.items()}
        for key, by_v in acc.items()
    }
    for n in {k[0] for k in data}:
        data[(n, "fusion_total")] = {
            v: sum(data[(n, r)][v] for r in FUSION) for v in ("baseline", "fused")
        }
    return data, sorted({k[0] for k in data})


def paired_speedup(data, n, region):
    """Median paired speedup and its 95% bootstrap CI."""
    ratio = data[(n, region)]["baseline"] / data[(n, region)]["fused"]
    idx = RNG.integers(0, ratio.size, (N_BOOT, ratio.size))
    lo, hi = np.percentile(np.median(ratio[idx], axis=1), [2.5, 97.5])
    return float(np.median(ratio)), float(lo), float(hi)


def quartiles(x):
    return np.percentile(x, [25, 50, 75])


# ----------------------------------------------------------------------------- styling

def resolve_font():
    if FONT_DIR.is_dir():
        for ttf in FONT_DIR.glob("*.[ot]tf"):
            font_manager.fontManager.addfont(str(ttf))
    wanted = plt.rcParams["font.sans-serif"][0]
    got = Path(font_manager.findfont(wanted, fallback_to_default=True)).stem
    if wanted.lower() not in got.lower():
        print(f"[!] '{wanted}' not installed; falling back to '{got}'. "
              f"Install it or drop the .ttf files into {FONT_DIR} for camera-ready output.")
    else:
        print(f"[*] Font: {got}")


def n_label(n):
    return f"{n // 1024}K" if n >= 1024 and n % 1024 == 0 else str(n)


def log2_axis(ax, ns):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([n_label(n) for n in ns])
    ax.set_xlim(ns[0] / 1.5, ns[-1] * 1.5)
    ax.grid(axis="y", which="major")


def save(fig, name, args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in args.formats:
        path = args.out_dir / f"{name}.{fmt}"
        fig.savefig(path, dpi=args.dpi if fmt == "png" else None)
        print(f"Generated: {path}")
    plt.close(fig)


# ----------------------------------------------------------------------------- figures

def fig1_region_latency(data, ns, args):
    """Figure 1: per-region latency (median, IQR band), baseline vs fused."""
    fig, axes = plt.subplots(3, 1, figsize=(COL_W, 3.9), sharex=True)
    for ax, (key, label, *_rest) in zip(axes, REGIONS):
        for variant, name, edge, fill, ls, mk in (
            ("baseline", "TRT-LLM baseline", COLOR_TAUPE_EDGE, COLOR_TAUPE_FILL, "-", "o"),
            ("fused", "Fused (ours)", COLOR_TEAL_EDGE, COLOR_TEAL_FILL, "--", "s"),
        ):
            q = np.array([quartiles(data[(n, key)][variant]) for n in ns])
            ax.fill_between(ns, q[:, 0], q[:, 2], color=fill, alpha=0.9, lw=0, zorder=2)
            ax.plot(ns, q[:, 1], ls=ls, marker=mk, color=edge, mfc="white", mew=0.9, label=name, zorder=3)
        ax.set_yscale("log")
        vals = [np.median(data[(n, key)][v]) for n in ns for v in ("baseline", "fused")]
        ax.set_ylim(min(vals) / 1.3, max(vals) * 1.3)
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        log2_axis(ax, ns)
        ax.text(0.02, 0.93, label, transform=ax.transAxes, ha="left", va="top",
                fontsize=7.5, fontweight="semibold", color=TEXT_COLOR)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.32), ncol=2)
    axes[-1].set_xlabel("Tokens per forward pass, N")
    fig.supylabel("Region latency, 2 blocks (µs) ↓", fontsize=8, fontweight="semibold", x=0.0)
    fig.align_ylabels(axes)
    save(fig, "fig1_region_latency", args)


def fig2_region_speedup(data, ns, args):
    """Figure 2: paired per-region speedup (median, 95% bootstrap CI)."""
    fig, ax = plt.subplots(figsize=(COL_W, 2.35))
    series = REGIONS + [("fusion_total", "R1+R2+R3 total", TEXT_COLOR, "#cbd5e1", "D")]
    for key, label, edge, fill, mk in series:
        s = np.array([paired_speedup(data, n, key) for n in ns])
        ax.fill_between(ns, s[:, 1], s[:, 2], color=edge, alpha=0.15, lw=0, zorder=2)
        ax.plot(ns, s[:, 0], marker=mk, color=edge, mfc="white", mew=0.9,
                ls="--" if key == "fusion_total" else "-", label=label, zorder=3)
    ax.axhline(1.0, color=SPINE_COLOR, lw=0.6, ls=":", zorder=1)
    # Below ~N=128 the kernels are launch/latency-bound (flat latency in Fig. 1)
    ax.axvspan(ns[0] / 1.5, 128, color="#f8fafc", zorder=0)
    ax.text(np.sqrt(ns[0] / 1.5 * 128), 1.03, "launch-bound", ha="center", va="bottom",
            fontsize=6.5, color=MUTED_TEXT, style="italic")
    log2_axis(ax, ns)
    ax.set_ylim(1.0, None)
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.1f}×")
    ax.set_xlabel("Tokens per forward pass, N")
    ax.set_ylabel("Speedup over baseline ↑")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=1.2)
    save(fig, "fig2_region_speedup", args)


def fig3_e2e_breakdown(data, ns, args):
    """Figure 3: E2E time breakdown, normalised to the baseline E2E median (broken y-axis)."""
    rows = []
    for n in ns:
        base_e2e = np.median(data[(n, "e2e")]["baseline"])
        for v in ("baseline", "fused"):
            regs = [np.median(data[(n, r)][v]) for r in FUSION]
            e2e = np.median(data[(n, "e2e")][v])
            rows.append([100 * (e2e - sum(regs)) / base_e2e] + [100 * r / base_e2e for r in regs])
    rows = np.array(rows).reshape(len(ns), 2, 1 + len(FUSION))

    lo_top = np.floor(rows[:, :, 0].min()) - 2
    fig, (ax, ax_lo) = plt.subplots(2, 1, figsize=(COL_W, 2.6), sharex=True,
                                    gridspec_kw=dict(height_ratios=[6, 1], hspace=0.08))
    x = np.arange(len(ns))
    w = 0.38
    colours = [(COLOR_TAUPE_EDGE, COLOR_TAUPE_FILL)] + [(r[2], r[3]) for r in REGIONS]
    for j, (v, off, hatch) in enumerate((("baseline", -w / 2, None), ("fused", w / 2, "////"))):
        for a in (ax, ax_lo):
            bottom = np.zeros(len(ns))
            for k, (edge, fill) in enumerate(colours):
                a.bar(x + off, rows[:, j, k], w, bottom=bottom, color=fill, edgecolor=edge,
                      hatch=hatch, linewidth=0.6, zorder=3)
                bottom += rows[:, j, k]
    for i, n in enumerate(ns):
        s, _, _ = paired_speedup(data, n, "e2e")
        ax.text(x[i], 100.3, f"{s:.3f}×", ha="center", va="bottom",
                fontsize=6, color=COLOR_TEAL_EDGE, fontweight="semibold")

    ax.set_ylim(lo_top, 102.2)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax_lo.set_ylim(0, 5)
    ax_lo.set_yticks([0])
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y")
    # Break marks
    d = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=5, linestyle="none", color=SPINE_COLOR,
             mec=SPINE_COLOR, mew=0.6, clip_on=False)
    ax.plot([0], [0], transform=ax.transAxes, **d)
    ax_lo.plot([0], [1], transform=ax_lo.transAxes, **d)

    ax_lo.set_xticks(x)
    ax_lo.set_xticklabels([n_label(n) for n in ns])
    ax_lo.set_xlabel("Tokens per forward pass, N")
    ax.set_ylabel("E2E time (% of baseline) ↓")
    handles = [Patch(facecolor=COLOR_TAUPE_FILL, edgecolor=COLOR_TAUPE_EDGE, lw=0.6, label="Unchanged ops")]
    handles += [Patch(facecolor=r[3], edgecolor=r[2], lw=0.6, label=r[1]) for r in REGIONS]
    handles += [Patch(facecolor="white", edgecolor=MUTED_TEXT, hatch="////", lw=0.6, label="Fused variant")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, columnspacing=1.0)
    save(fig, "fig3_e2e_breakdown", args)


def fig4_amdahl(data, ns, args, label, control=None, control_label=None):
    """Figure 4: measured paired E2E speedup vs Amdahl prediction from the region speedups."""
    pred, bound = [], []
    for n in ns:
        e2e_b = np.median(data[(n, "e2e")]["baseline"])
        fus_b = np.median(data[(n, "fusion_total")]["baseline"])
        fus_f = np.median(data[(n, "fusion_total")]["fused"])
        pred.append(e2e_b / (e2e_b - fus_b + fus_f))
        bound.append(e2e_b / (e2e_b - fus_b))

    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    ax.plot(ns, bound, ls=":", color=COLOR_AMBER_EDGE, lw=1.0, label="Bound (R1–R3 → 0)", zorder=2)
    ax.plot(ns, pred, ls="--", color=TEXT_COLOR, lw=1.0, label="Amdahl prediction", zorder=3)

    def measured(d, xs, label, mfc):
        s = np.array([paired_speedup(d, n, "e2e") for n in ns])
        ax.errorbar(xs, s[:, 0], yerr=[s[:, 0] - s[:, 1], s[:, 2] - s[:, 0]], fmt="o",
                    color=COLOR_TEAL_EDGE, mfc=mfc, mew=0.9, ms=4, elinewidth=0.8, capsize=1.8,
                    label=label, zorder=4)

    measured(data, ns, label, COLOR_TEAL_EDGE)
    if control is not None:
        measured(control, [n * 1.12 for n in ns], control_label, "white")

    log2_axis(ax, ns)
    ax.set_ylim(1.0, None)
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.2f}×")
    ax.set_xlabel("Tokens per forward pass, N")
    ax.set_ylabel("E2E speedup over baseline ↑")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=1.2)
    save(fig, "fig4_amdahl", args)


# ----------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description="Publication figures for the 2-block Kolibri E2E benchmark")
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--stem", default="e2e_benchmark_report", help="CSV stem written by --export-csv")
    p.add_argument("--primary", choices=["pinned", "free"], default="pinned",
                   help="Dataset for all figures; the other is overlaid in Fig. 4 as a control")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--formats", nargs="+", default=["pdf", "png"])
    p.add_argument("--dpi", type=int, default=600)
    args = p.parse_args()

    resolve_font()
    paths = {
        "free": args.results_dir / f"{args.stem}_raw.csv",
        "pinned": args.results_dir / f"{args.stem}_pinned_raw.csv",
    }
    data, ns = load_raw(paths[args.primary])
    other = "free" if args.primary == "pinned" else "pinned"
    control = load_raw(paths[other])[0] if paths[other].exists() else None
    labels = {"free": "Free routing", "pinned": "Pinned routing"}

    print("=" * 80)
    print(f"Generating E2E publication figures from {paths[args.primary].name}")
    print("=" * 80)
    fig1_region_latency(data, ns, args)
    fig2_region_speedup(data, ns, args)
    fig3_e2e_breakdown(data, ns, args)
    fig4_amdahl(data, ns, args, f"{labels[args.primary]} (95% CI)", control, labels[other])


if __name__ == "__main__":
    main()
