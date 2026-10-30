#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Figures for the 2-block Kolibri E2E benchmark (11_profile_2block_e2e.py).

Every figure is produced in two versions (see figstyle.py):
  assets/e2e/detail/  every N, confidence bands, legends
  assets/e2e/slide/   one message, N = 1 / 512 / 4K, direct labels, large type

Statistics are computed from the per-iteration samples (<stem>_raw.csv), never from mean ± std:
  - latencies:  median with interquartile range
  - speedups:   median of *paired* per-iteration ratios (baseline_i / fused_i). Iterations are
                interleaved by the benchmark, so each pair ran back-to-back at the same clock and
                residual clock drift cancels. 95% CIs via percentile bootstrap.

Figures:
  fig1_region_latency   per-region latency, baseline vs fused (R1, R2, R3)
  fig2_region_speedup   per-region + total fusion speedup
  fig3_e2e_breakdown    share of E2E time spent in the fusion regions (Amdahl view)
  fig4_amdahl           measured E2E speedup vs Amdahl prediction
  fig5_full_model       time saved per forward pass, projected to the full --model-layers stack
  fig6_cumulative       decode GPU time saved vs tokens generated (full stack)

Full-model projection (fig5/fig6): every decoder layer has the same shapes, so the paired per-pass
saving of the measured blocks scales linearly with depth (x model_layers / measured_layers). The
*speedup ratio* does not grow with depth -- it is the same 1.05x at 2 or 50 layers -- but the absolute
time saved does. The projection covers the decoder stack only (no embedding / LM head / sampling).
R1-R3 are per-token ops independent of context length, so the saved time per token also holds as the
KV cache grows during decode, while the % speedup shrinks as attention gets more expensive.

Usage:
  python benchmarks/generate_e2e_figures.py
  python benchmarks/generate_e2e_figures.py --mode slide --formats pdf png svg
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.patches import Patch

from figstyle import (
    C_BASE,
    C_BLUE,
    C_BLUE_FILL,
    C_FUSED,
    C_GRAY,
    C_GRAY_FILL,
    C_GREEN,
    C_GREEN_FILL,
    C_MUTED,
    C_PARITY,
    C_RED,
    C_RED_FILL,
    C_TOTAL,
    DETAIL_W,
    MODES,
    REPO_ROOT,
    SLIDE_NS,
    SLIDE_SIZE,
    log2_axis,
    log_yaxis,
    n_label,
    resolve_font,
    save,
    slide_latency_panels,
    slide_n_label,
    slide_speedup_lines,
    style,
)

RESULTS_DIR = REPO_ROOT / "results"
OUT_DIR = REPO_ROOT / "assets" / "e2e"

# (region key, legend label, edge colour, fill colour, marker)
REGIONS = [
    ("R1_sandwich_norm", "R1 Sandwich norm", C_BLUE, C_BLUE_FILL, "o"),
    ("R2_moe_prologue", "R2 MoE prologue", C_RED, C_RED_FILL, "s"),
    ("R3_post_moe_norm", "R3 Post-MoE norm", C_GREEN, C_GREEN_FILL, "^"),
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


def paired_saving(data, n, region="e2e"):
    """Median paired saving baseline_i - fused_i (us) and its 95% bootstrap CI."""
    diff = data[(n, region)]["baseline"] - data[(n, region)]["fused"]
    idx = RNG.integers(0, diff.size, (N_BOOT, diff.size))
    lo, hi = np.percentile(np.median(diff[idx], axis=1), [2.5, 97.5])
    return float(np.median(diff)), float(lo), float(hi)


def human_time(sec):
    for unit, size in (("days", 86400), ("h", 3600), ("min", 60), ("s", 1), ("ms", 1e-3)):
        if sec >= size:
            v = sec / size
            return f"{v:.1f} {unit}" if v < 10 else f"{v:.0f} {unit}"
    return f"{sec * 1e6:.0f} µs"


def quartiles(x):
    return np.percentile(x, [25, 50, 75])


def time_shares(data, ns):
    """[n][variant] -> [unchanged ops, R1, R2, R3] as % of the baseline E2E median."""
    rows = []
    for n in ns:
        base_e2e = np.median(data[(n, "e2e")]["baseline"])
        for v in ("baseline", "fused"):
            regs = [np.median(data[(n, r)][v]) for r in FUSION]
            e2e = np.median(data[(n, "e2e")][v])
            rows.append([100 * (e2e - sum(regs)) / base_e2e] + [100 * r / base_e2e for r in regs])
    return np.array(rows).reshape(len(ns), 2, 1 + len(FUSION))


def amdahl_prediction(data, n):
    e2e_b = np.median(data[(n, "e2e")]["baseline"])
    fus_b = np.median(data[(n, "fusion_total")]["baseline"])
    fus_f = np.median(data[(n, "fusion_total")]["fused"])
    return e2e_b / (e2e_b - fus_b + fus_f), e2e_b / (e2e_b - fus_b)


# ----------------------------------------------------------------------------- figure 1

def fig1_region_latency(data, ns, args, mode):
    """Figure 1: per-region latency, baseline vs fused."""
    if mode == "slide":
        fig = slide_latency_panels(data, [(k, k[:2]) for k in FUSION], paired_speedup)
        return save(fig, OUT_DIR, mode, "fig1_region_latency", args)

    fig, axes = plt.subplots(3, 1, figsize=(DETAIL_W, 5.0), sharex=True, gridspec_kw=dict(hspace=0.4))
    for ax, (key, label, *_rest) in zip(axes, REGIONS):
        for variant, name, colour, ls, mk in (
            ("baseline", "TRT-LLM baseline", C_BASE, "--", "^"),
            ("fused", "Fused (mine)", C_FUSED, "-", "s"),
        ):
            q = np.array([quartiles(data[(n, key)][variant]) for n in ns])
            ax.fill_between(ns, q[:, 0], q[:, 2], color=colour, alpha=0.2, lw=0, zorder=2)
            ax.plot(ns, q[:, 1], ls=ls, marker=mk, color=colour, label=name, zorder=3)
        ax.set_yscale("log")
        log_yaxis(ax)
        vals = [np.median(data[(n, key)][v]) for n in ns for v in ("baseline", "fused")]
        ax.set_ylim(min(vals) / 1.3, max(vals) * 1.3)
        log2_axis(ax, ns)
        ax.text(0.02, 0.93, label, transform=ax.transAxes, ha="left", va="top",
                fontweight="bold", color=C_TOTAL)
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=2)
    axes[-1].set_xlabel("Tokens per forward pass, N")
    fig.supylabel("Region latency, 2 blocks ($\\mu$s) ↓", fontsize=8.5, x=0.0)
    fig.align_ylabels(axes)
    save(fig, OUT_DIR, mode, "fig1_region_latency", args)


# ----------------------------------------------------------------------------- figure 2

def fig2_region_speedup(data, ns, args, mode):
    """Figure 2: paired per-region speedup; total emphasised."""
    if mode == "slide":
        fig = slide_speedup_lines(data, [(k, k[:2], c) for k, _l, c, _f, _m in REGIONS],
                                  ("fusion_total", "Total"), paired_speedup)
        return save(fig, OUT_DIR, mode, "fig2_region_speedup", args)

    fig, ax = plt.subplots(figsize=(DETAIL_W, 3.0))
    for key, label, edge, _fill, mk in REGIONS:
        s = np.array([paired_speedup(data, n, key) for n in ns])
        ax.fill_between(ns, s[:, 1], s[:, 2], color=edge, alpha=0.07, lw=0, zorder=2)
        ax.plot(ns, s[:, 0], marker=mk, color=edge, lw=1.0, ms=3.5, alpha=0.45, label=label, zorder=3)
    s = np.array([paired_speedup(data, n, "fusion_total") for n in ns])
    ax.fill_between(ns, s[:, 1], s[:, 2], color=C_TOTAL, alpha=0.15, lw=0, zorder=4)
    ax.plot(ns, s[:, 0], marker="D", color=C_TOTAL, lw=2.2, ms=5, label="R1+R2+R3 total", zorder=5)
    ax.axhline(1.0, color=C_PARITY, lw=1.0, ls=":", zorder=1)
    log2_axis(ax, ns)
    ax.set_ylim(0.9, None)
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.1f}×")
    ax.set_xlabel("Tokens per forward pass, N")
    ax.set_ylabel("Speedup over baseline ↑")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=1.2)
    save(fig, OUT_DIR, mode, "fig2_region_speedup", args)


# ----------------------------------------------------------------------------- figure 3

def fig3_e2e_breakdown(data, ns, args, mode):
    """Figure 3: E2E time breakdown, normalised to the baseline E2E median."""
    if mode == "slide":
        return _fig3_slide(data, args)

    rows = time_shares(data, ns)
    lo_top = np.floor(rows[:, :, 0].min()) - 2
    fig, (ax, ax_lo) = plt.subplots(2, 1, figsize=(DETAIL_W, 3.2), sharex=True,
                                    gridspec_kw=dict(height_ratios=[6, 1], hspace=0.08))
    x = np.arange(len(ns)) * 1.25  # extra gap between N groups
    w = 0.34
    gap = 0.05  # gap between the baseline and fused bar of a pair
    colours = [(C_GRAY, C_GRAY_FILL)] + [(r[2], r[3]) for r in REGIONS]
    for j, (_v, off, hatch) in enumerate((("baseline", -(w + gap) / 2, None), ("fused", (w + gap) / 2, "////"))):
        for a in (ax, ax_lo):
            bottom = np.zeros(len(ns))
            for k, (edge, fill) in enumerate(colours):
                a.bar(x + off, rows[:, j, k], w, bottom=bottom, color=fill, edgecolor=edge,
                      hatch=hatch, linewidth=0.6, zorder=3)
                bottom += rows[:, j, k]
    for i, n in enumerate(ns):
        s, _, _ = paired_speedup(data, n, "e2e")
        ax.text(x[i], 100.3, f"{s:.3f}×", ha="center", va="bottom",
                fontsize=6.5, color=C_FUSED, fontweight="bold")

    ax.set_ylim(lo_top, 102.2)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax_lo.set_ylim(0, 5)
    ax_lo.set_yticks([0])
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    # Break marks
    ax_lo.spines["top"].set_visible(False)
    d = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=5, linestyle="none", color="black",
             mec="black", mew=0.8, clip_on=False)
    ax.plot([0, 1], [0, 0], transform=ax.transAxes, **d)
    ax_lo.plot([0, 1], [1, 1], transform=ax_lo.transAxes, **d)

    ax_lo.set_xticks(x)
    ax_lo.set_xticklabels([n_label(n) for n in ns])
    ax_lo.set_xlabel("Tokens per forward pass, N")
    ax.set_ylabel("E2E time (% of baseline) ↓")
    handles = [Patch(facecolor=C_GRAY_FILL, edgecolor=C_GRAY, lw=0.6, label="Unchanged ops")]
    handles += [Patch(facecolor=r[3], edgecolor=r[2], lw=0.6, label=r[1]) for r in REGIONS]
    handles += [Patch(facecolor="white", edgecolor=C_MUTED, hatch="////", lw=0.6, label="Fused variant")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, columnspacing=1.0)
    save(fig, OUT_DIR, "detail", "fig3_e2e_breakdown", args)


def _fig3_slide(data, args):
    """Slide: per N, baseline vs fused E2E bar with R1–R3 merged into one segment."""
    rows = time_shares(data, SLIDE_NS)
    fig, axes = plt.subplots(len(SLIDE_NS), 1, figsize=SLIDE_SIZE, sharex=True,
                             gridspec_kw=dict(hspace=0.55))
    for ax, n, r in zip(axes, SLIDE_NS, rows):
        other, regions = r[:, 0], r[:, 1:].sum(axis=1)
        y = [1, 0]  # baseline on top
        ax.barh(y, other, 0.7, color=C_GRAY_FILL, edgecolor=C_GRAY, lw=0.8)
        ax.barh(y, regions, 0.7, left=other, color=C_RED, edgecolor=C_RED, lw=0.8)
        s, _, _ = paired_speedup(data, n, "e2e")
        ax.text(other[0] + regions[0] + 1.5, y[0], f"{regions[0]:.0f}% in R1–R3", va="center",
                color=C_RED, fontsize=12)
        ax.text(other[1] + regions[1] + 1.5, y[1], f"{s:.3f}× faster", va="center",
                color=C_FUSED, fontsize=12, fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(["Baseline", "Fused"])
        ax.set_xlim(0, 142)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(-0.6, 1.6)
        ax.set_title(slide_n_label(n).replace("\n", "  "), loc="left", fontsize=13, pad=4)
        ax.tick_params(axis="y", length=0)
    axes[-1].set_xticks([0, 25, 50, 75, 100])
    axes[-1].set_xlabel("E2E time (% of baseline)")
    handles = [Patch(facecolor=C_GRAY_FILL, edgecolor=C_GRAY, label="Unchanged ops"),
               Patch(facecolor=C_RED, edgecolor=C_RED, label="Fusion regions R1–R3")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, OUT_DIR, "slide", "fig3_e2e_breakdown", args)


# ----------------------------------------------------------------------------- figure 4

def fig4_amdahl(data, ns, args, mode, label, control=None, control_label=None):
    """Figure 4: measured paired E2E speedup vs Amdahl prediction from the region speedups."""
    if mode == "slide":
        return _fig4_slide(data, args)

    pred, bound = zip(*(amdahl_prediction(data, n) for n in ns))
    fig, ax = plt.subplots(figsize=(DETAIL_W, 3.0))
    ax.plot(ns, bound, ls=":", color=C_PARITY, lw=1.4, label="Bound (R1–R3 → 0)", zorder=2)
    ax.plot(ns, pred, ls="--", color=C_TOTAL, lw=1.3, label="Amdahl prediction", zorder=3)

    def measured(d, xs, label, mfc):
        s = np.array([paired_speedup(d, n, "e2e") for n in ns])
        ax.errorbar(xs, s[:, 0], yerr=[s[:, 0] - s[:, 1], s[:, 2] - s[:, 0]], fmt="o",
                    color=C_FUSED, mfc=mfc, mew=1.0, ms=4.5, elinewidth=0.9, capsize=2,
                    label=label, zorder=4)

    measured(data, ns, label, C_FUSED)
    if control is not None:
        measured(control, [n * 1.12 for n in ns], control_label, "white")

    log2_axis(ax, ns)
    ax.set_ylim(1.0, None)
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.2f}×")
    ax.set_xlabel("Tokens per forward pass, N")
    ax.set_ylabel("E2E speedup over baseline ↑")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=1.2)
    save(fig, OUT_DIR, "detail", "fig4_amdahl", args)


def _fig4_slide(data, args):
    """Slide: measured E2E speedup sits on the Amdahl prediction."""
    x = np.arange(len(SLIDE_NS))
    meas = [paired_speedup(data, n, "e2e")[0] for n in SLIDE_NS]
    pred = [amdahl_prediction(data, n)[0] for n in SLIDE_NS]
    fig, ax = plt.subplots(figsize=SLIDE_SIZE)
    ax.plot(x, pred, ls="--", color=C_TOTAL, lw=2.0, zorder=3, label="Predicted (Amdahl)")
    ax.plot(x, meas, color=C_FUSED, marker="o", ms=10, zorder=4, label="Measured")
    for xi, v in zip(x, meas):
        ax.annotate(f"{v:.3f}×", (xi, v), xytext=(0, 13), textcoords="offset points", ha="center",
                    fontweight="bold", color=C_FUSED)
    ax.legend(loc="upper right")
    ax.set_xticks(x)
    ax.set_xticklabels([slide_n_label(n) for n in SLIDE_NS])
    ax.set_xlim(-0.35, len(x) - 1 + 0.35)
    ax.set_ylim(1.0, max(meas + pred) * 1.015)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.2f}×")
    ax.set_ylabel("E2E speedup")
    save(fig, OUT_DIR, "slide", "fig4_amdahl", args)


# ----------------------------------------------------------------------------- figure 5

def fig5_full_model(data, ns, args, mode):
    """Figure 5: time saved per forward pass, projected to the full decoder stack."""
    scale = args.model_layers / args.measured_layers
    save_ms = np.array([paired_saving(data, n) for n in ns]) * scale / 1e3
    base_ms = {n: np.median(data[(n, "e2e")]["baseline"]) * scale / 1e3 for n in ns}
    fused_ms = {n: np.median(data[(n, "e2e")]["fused"]) * scale / 1e3 for n in ns}
    if mode == "slide":
        return _fig5_slide(save_ms[[ns.index(n) for n in SLIDE_NS]], base_ms, fused_ms, args)

    fig, ax = plt.subplots(figsize=(DETAIL_W, 3.0))
    ax.fill_between(ns, save_ms[:, 1], save_ms[:, 2], color=C_FUSED, alpha=0.18, lw=0)
    ax.plot(ns, save_ms[:, 0], color=C_FUSED, marker="o", label=f"Saved per pass, {args.model_layers} layers (95% CI)")
    for n, v in zip(ns, save_ms[:, 0]):
        if n in SLIDE_NS:
            off, ha, va = {ns[0]: ((3, 7), "left", "bottom"), ns[-1]: ((-7, 0), "right", "center")}.get(
                n, ((-5, 7), "right", "bottom"))
            ax.annotate(f"{v:.2f} ms", (n, v), xytext=off, textcoords="offset points", ha=ha, va=va,
                        fontsize=7.5, color=C_FUSED)
    log2_axis(ax, ns)
    ax.set_ylim(0, save_ms[:, 2].max() * 1.3)
    ax.set_xlabel("Tokens per forward pass, N")
    ax.set_ylabel("Time saved per pass (ms) ↑")
    ax.legend(loc="upper left")
    save(fig, OUT_DIR, "detail", "fig5_full_model", args)


def _fig5_slide(save_ms, base_ms, fused_ms, args):
    """Slide: milliseconds saved per forward pass of the full model."""
    x = np.arange(len(SLIDE_NS))
    v = save_ms[:, 0]
    fig, ax = plt.subplots(figsize=SLIDE_SIZE)
    ax.bar(x, v, 0.55, color=C_FUSED)
    for xi, n, vi in zip(x, SLIDE_NS, v):
        ax.text(xi, vi + 0.03 * v.max(), f"{base_ms[n]:.1f} → {fused_ms[n]:.1f} ms", ha="center",
                va="bottom", fontsize=10.5, color=C_MUTED)
        ax.text(xi, vi + 0.13 * v.max(), f"−{vi:.2f} ms", ha="center", va="bottom",
                fontweight="bold", fontsize=14, color=C_FUSED)
    ax.set_xticks(x)
    ax.set_xticklabels([slide_n_label(n) for n in SLIDE_NS])
    ax.set_ylim(0, v.max() * 1.32)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4, integer=True))
    ax.set_ylabel("Saved per pass (ms)")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, OUT_DIR, "slide", "fig5_full_model", args)


# ----------------------------------------------------------------------------- figure 6

TIME_TICKS = [(0.01, "10 ms"), (1, "1 s"), (60, "1 min"), (3600, "1 h"), (86400, "1 day"), (7 * 86400, "1 week")]
MILESTONES = [(1e3, "1K-token reply"), (1e6, "1M tokens"), (1e9, "1B tokens")]


def fig6_cumulative(data, ns, args, mode):
    """Figure 6: decode GPU time saved vs tokens generated (one token per forward pass)."""
    scale = args.model_layers / args.measured_layers
    per_tok = np.array(paired_saving(data, 1)) * scale / 1e6  # s saved per generated token
    tokens = np.logspace(2.5, 10.2, 200)
    fig, ax = plt.subplots(figsize=(DETAIL_W, 3.0) if mode == "detail" else SLIDE_SIZE)
    if mode == "detail":
        ax.fill_between(tokens, tokens * per_tok[1], tokens * per_tok[2], color=C_FUSED, alpha=0.18, lw=0)
        n_pf = SLIDE_NS[-1]
        pf_tok = paired_saving(data, n_pf)[0] * scale / 1e6 / n_pf
        ax.plot(tokens, tokens * pf_tok, color=C_MUTED, ls="--", lw=1.2,
                label=f"Prefill ({n_label(n_pf)}-token prompts), per prompt token")
    ax.plot(tokens, tokens * per_tok[0], color=C_FUSED,
            label=f"Decode, {per_tok[0] * 1e3:.2f} ms saved per token" if mode == "detail" else None)
    for t, name in MILESTONES:
        y = t * per_tok[0]
        last = t == MILESTONES[-1][0]
        ax.plot(t, y, "o", color=C_TOTAL, ms=5 if mode == "detail" else 9, zorder=5)
        ax.annotate(f"{name}\n{human_time(y)} saved", (t, y), xytext=(-10, 6) if last else (10, -6),
                    textcoords="offset points", ha="right" if last else "left", va="bottom" if last else "top",
                    fontweight="bold", color=C_TOTAL,
                    fontsize=7.5 if mode == "detail" else 12.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(tokens[0], tokens[-1])
    ax.set_ylim(0.005 if mode == "detail" else 0.03, tokens[-1] * per_tok[0] * 1.5)
    ax.xaxis.set_major_locator(ticker.FixedLocator([1e3, 1e6, 1e9]))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(["1K", "1M", "1B"]))
    ax.yaxis.set_major_locator(ticker.FixedLocator([t for t, _ in TIME_TICKS]))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter([lab for _, lab in TIME_TICKS]))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel("Tokens generated")
    ax.set_ylabel("GPU time saved")
    if mode == "detail":
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=1.2)
    else:
        ax.spines[["top", "right"]].set_visible(False)
    save(fig, OUT_DIR, mode, "fig6_cumulative", args)


# ----------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description="Figures for the 2-block Kolibri E2E benchmark")
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--stem", default="e2e_benchmark_report", help="CSV stem written by --export-csv")
    p.add_argument("--primary", choices=["pinned", "free"], default="pinned",
                   help="Dataset for all figures; the other is overlaid in the detailed Fig. 4 as a control")
    p.add_argument("--mode", choices=[*MODES, "both"], default="both")
    p.add_argument("--model-layers", type=int, default=50, help="Decoder layers of the full model (fig5/fig6)")
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
    meta = paths[args.primary].with_name(paths[args.primary].name.replace("_raw.csv", "_meta.json"))
    args.measured_layers = json.loads(meta.read_text())["args"]["layers"] if meta.exists() else 2

    print("=" * 80)
    print(f"Generating E2E figures from {paths[args.primary].name}")
    print("=" * 80)
    for mode in MODES if args.mode == "both" else [args.mode]:
        with style(mode):
            fig1_region_latency(data, ns, args, mode)
            fig2_region_speedup(data, ns, args, mode)
            fig3_e2e_breakdown(data, ns, args, mode)
            fig4_amdahl(data, ns, args, mode, f"{labels[args.primary]} (95% CI)", control, labels[other])
            fig5_full_model(data, ns, args, mode)
            fig6_cumulative(data, ns, args, mode)


if __name__ == "__main__":
    main()
