# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared figure style for generate_e2e_figures.py and generate_r2_figures.py.

Two presets:
  detail  every N, confidence bands, legends; 5 in wide (backup slides / appendix / paper)
  slide   one message per figure: three representative N (decode / mid / prefill), medians only,
          direct labels instead of legends, large type; sized for a 16:9 slide

Classic matplotlib look (boxed axes, DejaVu Sans, Computer Modern math) matching
docs/iterative_optimizations/figures/plot_dispatch_mapping.py, without grid lines.
"""

from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, ticker

REPO_ROOT = Path(__file__).resolve().parent.parent
FONT_DIR = REPO_ROOT / "assets" / "fonts"  # optional: extra .ttf/.otf files to register

MODES = ("detail", "slide")
DETAIL_W = 5.0             # detail figure width (in)
SLIDE_SIZE = (6.5, 3.6)    # slide figure size (in), ~16:9 content area
SLIDE_NS = (1, 512, 4096)  # decode / mid / prefill

# Palette (same hues as plot_dispatch_mapping.py)
C_BASE = "#D97706"     # amber: TRT-LLM baseline
C_FUSED = "#2B6CB0"    # sapphire blue: fused kernels (ours)
C_TOTAL = "#1A202C"    # near-black: totals / predictions
C_MUTED = "#4A5568"    # slate: secondary text, legend style keys
C_PARITY = "#E53E3E"   # red: 1.0x parity / bounds
C_BLUE, C_BLUE_FILL = "#2B6CB0", "#A3BFE3"
C_RED, C_RED_FILL = "#D9534F", "#F2B8B6"
C_GREEN, C_GREEN_FILL = "#2F855A", "#A7D8BA"
C_GRAY, C_GRAY_FILL = "#718096", "#E2E8F0"

_COMMON = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "mathtext.fontset": "cm",
    "axes.grid": False,
    "axes.edgecolor": "black",
    "axes.axisbelow": True,
    "xtick.minor.size": 0,
    "legend.frameon": False,
    "pdf.fonttype": 42,  # embed TrueType (camera-ready requirement for most venues)
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
}
_PRESETS = {
    "detail": {
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "lines.linewidth": 1.5,
        "lines.markersize": 4.5,
        "legend.handlelength": 2.0,
        "savefig.pad_inches": 0.03,
    },
    "slide": {
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 12.5,
        "legend.fontsize": 12.5,
        "axes.linewidth": 1.1,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "lines.linewidth": 2.6,
        "lines.markersize": 8,
        "legend.handlelength": 1.6,
        "savefig.pad_inches": 0.06,
    },
}

plt.rcParams.update(_COMMON)
plt.rcParams.update(_PRESETS["detail"])


@contextmanager
def style(mode):
    with plt.rc_context({**_COMMON, **_PRESETS[mode]}):
        yield


def resolve_font():
    if FONT_DIR.is_dir():
        for ttf in FONT_DIR.glob("*.[ot]tf"):
            font_manager.fontManager.addfont(str(ttf))
    wanted = plt.rcParams["font.sans-serif"][0]
    got = Path(font_manager.findfont(wanted, fallback_to_default=True)).stem
    if wanted.lower() not in got.lower():
        print(f"[!] '{wanted}' not installed; falling back to '{got}'. "
              f"Install it or drop the .ttf files into {FONT_DIR}.")
    else:
        print(f"[*] Font: {got}")


def save(fig, out_dir, mode, name, args):
    out = Path(out_dir) / mode
    out.mkdir(parents=True, exist_ok=True)
    for fmt in args.formats:
        path = out / f"{name}.{fmt}"
        fig.savefig(path, dpi=args.dpi if fmt == "png" else None)
        print(f"Generated: {path}")
    plt.close(fig)


# ----------------------------------------------------------------------------- axes helpers

def n_label(n):
    return f"{n // 1024}K" if n >= 1024 and n % 1024 == 0 else str(n)


def slide_n_label(n):
    """Two-line category label for the slide N values."""
    regime = {SLIDE_NS[0]: "Decode", SLIDE_NS[-1]: "Prefill"}.get(n, "")
    return f"{regime}\nN = {n_label(n)}".strip()


def log2_axis(ax, ns):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([n_label(n) for n in ns])
    ax.set_xlim(ns[0] / 1.5, ns[-1] * 1.5)


def log_yaxis(ax):
    """1-2-5 labelled ticks on a log y-axis."""
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())


# ----------------------------------------------------------------------------- slide patterns

def slide_latency_panels(data, items, speedup):
    """One panel per slide N; baseline vs fused bars per item, paired speedup printed above.

    items: [(key, short label)], speedup: f(data, n, key) -> (median, lo, hi).
    """
    fig, axes = plt.subplots(1, len(SLIDE_NS), figsize=SLIDE_SIZE, gridspec_kw=dict(wspace=0.32))
    x = np.arange(len(items))
    w = 0.38
    for ax, n in zip(axes, SLIDE_NS):
        base = [np.median(data[(n, k)]["baseline"]) for k, _ in items]
        fused = [np.median(data[(n, k)]["fused"]) for k, _ in items]
        ax.bar(x - w / 2, base, w, color=C_BASE, label="TRT-LLM baseline")
        ax.bar(x + w / 2, fused, w, color=C_FUSED, label="Fused (ours)")
        top = max(base)
        for i, (k, _) in enumerate(items):
            s, _, _ = speedup(data, n, k)
            ax.text(x[i], max(base[i], fused[i]) + 0.03 * top, f"{s:.1f}×", ha="center", va="bottom",
                    fontweight="bold", fontsize=12)
        ax.set_ylim(0, top * 1.22)
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in items], fontsize=10.5 if max(len(lab) for _, lab in items) > 3 else None)
        ax.set_title(slide_n_label(n).replace("\n", "  "), pad=8)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(3, integer=True))
    axes[0].set_ylabel("Latency ($\\mu$s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    return fig


def slide_speedup_lines(data, items, total, speedup):
    """Speedup at the slide N: faded component lines + emphasised total, labelled in place.

    items: [(key, short label, colour)], total: (key, label).
    """
    fig, ax = plt.subplots(figsize=SLIDE_SIZE)
    x = np.arange(len(SLIDE_NS))
    ends = []
    for key, label, colour in items:
        s = [speedup(data, n, key)[0] for n in SLIDE_NS]
        ax.plot(x, s, color=colour, marker="o", lw=1.8, ms=6, alpha=0.5)
        ends.append([s[-1], label, colour, 0.75])
    s = [speedup(data, n, total[0])[0] for n in SLIDE_NS]
    ax.plot(x, s, color=C_TOTAL, marker="D", lw=3.4, ms=10, zorder=5)
    for xi, v in zip(x, s):
        ax.annotate(f"{v:.2f}×", (xi, v), xytext=(0, 12), textcoords="offset points", ha="center",
                    fontweight="bold", fontsize=13, color=C_TOTAL)
    ends.append([s[-1], total[1], C_TOTAL, 1.0])
    _label_line_ends(ax, x[-1], ends)
    ax.axhline(1.0, color=C_PARITY, lw=1.2, ls=":", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([slide_n_label(n) for n in SLIDE_NS])
    ax.set_xlim(-0.3, len(x) - 1 + 0.75)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.1f}×")
    ax.set_ylim(0.9, None)
    ax.set_ylabel("Speedup over baseline")
    return fig


def _label_line_ends(ax, x_end, ends, min_gap_frac=0.07):
    """Label lines at their right end, nudging labels apart so they never overlap."""
    ends.sort(key=lambda e: e[0])
    lo, hi = ax.get_ylim()
    lo, hi = min(lo, min(e[0] for e in ends)), max(hi, max(e[0] for e in ends))
    gap = (hi - lo) * min_gap_frac
    ys = [e[0] for e in ends]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    for y, (_v, label, colour, alpha) in zip(ys, ends):
        ax.text(x_end + 0.12, y, label, color=colour, alpha=alpha, va="center", ha="left",
                fontweight="bold" if alpha == 1.0 else "normal")
