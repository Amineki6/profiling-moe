#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Publication-grade figures for the sub-stages within Region 2 (MoE Prologue).

Region 2 is the only multi-kernel fusion region in the pipeline, consisting of 3 stages:
  - Step 1: Top-6 Router Reduction (kolibri_router vs trtllm_router)
  - Step 2: Dynamic Hybrid Dispatch Mapping (kolibri_dispatch_map vs trtllm_build_expert_maps)
  - Step 3: 128-bit Vectorized Activation Row Expansion (kolibri_activation_expansion vs trtllm_activation_expansion)
  - Combined: Full MoE Prologue Pipeline (fused_moe_prologue vs trtllm_moe_prologue)

Figures generated, each in a detail and a slide version (see figstyle.py):
  1. fig1a_r2_stage_latency:
     Log-log execution latency curves for each sub-stage (Baseline vs Fused).
  2. fig1b_r2_stage_share:
     Latency share (%) stacked bars, showing how Step 3 grows to dominate R2 at large N.
  3. fig2_r2_stage_speedup:
     Speedup curves vs N for Step 1, Step 2, Step 3, and Total R2 with 95% bootstrap CIs.

Outputs go to assets/r2/detail/ and assets/r2/slide/.

Usage:
  python benchmarks/generate_r2_figures.py
  python benchmarks/generate_r2_figures.py --mode slide
  python benchmarks/generate_r2_figures.py --runs 50 --recompute
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Add repo roots
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "fusions" / "03_moe_prologue" / "fused"))
sys.path.insert(0, str(REPO_ROOT / "fusions" / "03_moe_prologue" / "baseline"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from figstyle import (
    C_BLUE,
    C_BLUE_FILL,
    C_GREEN,
    C_GREEN_FILL,
    C_MUTED,
    C_PARITY,
    C_RED,
    C_RED_FILL,
    C_TOTAL,
    DETAIL_W,
    MODES,
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
OUT_DIR = REPO_ROOT / "assets" / "r2"

# Color theme for R2 sub-stages (shared palette in figstyle.py)
C_ROUTER_EDGE, C_ROUTER_FILL = C_RED, C_RED_FILL      # Step 1 Router
C_MAP_EDGE, C_MAP_FILL = C_BLUE, C_BLUE_FILL          # Step 2 Dispatch Map
C_EXP_EDGE, C_EXP_FILL = C_GREEN, C_GREEN_FILL        # Step 3 Activation Expansion
C_TOTAL_EDGE, C_TOTAL_FILL = C_TOTAL, "#CBD5E0"       # Total R2 Pipeline

N_BOOT = 10_000
RNG = np.random.default_rng(42)


# =============================================================================
# Benchmarking Engine
# =============================================================================

def benchmark_r2_stages(seq_lens, runs=50, warmup=10, dev_str="cuda:0"):
    import torch
    import fused_moe_prologue as fmp
    import trtllm_baseline as trt_base

    dev = torch.device(dev_str)
    H, E, K = 2560, 384, 6

    raw_records = []
    summary_records = []

    print("=" * 80)
    print(f"  BENCHMARKING REGION 2 SUB-STAGES ({runs} interleaved runs per N on {dev})")
    print("=" * 80)

    for n in seq_lens:
        x = torch.randn(n, H, device=dev, dtype=torch.bfloat16)
        logits = torch.randn(n, E, device=dev, dtype=torch.float32)
        bias = torch.randn(E, device=dev, dtype=torch.float32)

        # Warmup
        for _ in range(warmup):
            trt_base.trtllm_moe_prologue(x, logits, bias)
            fmp.fused_moe_prologue(x, logits, bias)
        torch.cuda.synchronize(dev)

        stage_fns = {
            "step1_router": {
                "baseline": lambda: trt_base.trtllm_router(logits, bias),
                "fused": lambda: fmp.kolibri_router(logits, bias),
            },
            "step2_dispatch": {
                "baseline": None,  # dynamically bound
                "fused": None,
            },
            "step3_expansion": {
                "baseline": None,  # dynamically bound
                "fused": None,
            },
            "total_prologue": {
                "baseline": lambda: trt_base.trtllm_moe_prologue(x, logits, bias),
                "fused": lambda: fmp.fused_moe_prologue(x, logits, bias),
            },
        }

        # Cache inputs for step 2 and step 3
        wb, sb = trt_base.trtllm_router(logits, bias)
        wf, sf = fmp.kolibri_router(logits, bias)
        stage_fns["step2_dispatch"]["baseline"] = lambda: trt_base.trtllm_build_expert_maps(sb, E)
        stage_fns["step2_dispatch"]["fused"] = lambda: fmp.kolibri_dispatch_map(sf, E)

        maps_b = trt_base.trtllm_build_expert_maps(sb, E)
        maps_f = fmp.kolibri_dispatch_map(sf, E)
        stage_fns["step3_expansion"]["baseline"] = lambda: trt_base.trtllm_activation_expansion(x, maps_b[0])
        stage_fns["step3_expansion"]["fused"] = lambda: fmp.kolibri_activation_expansion(x, maps_f[0])

        stage_samples = defaultdict(lambda: defaultdict(list))

        for stage, fns in stage_fns.items():
            evs = {
                "baseline": [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(runs)],
                "fused": [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(runs)],
            }
            for i in range(runs):
                order = ["baseline", "fused"] if i % 2 == 0 else ["fused", "baseline"]
                for v in order:
                    evs[v][i][0].record()
                    fns[v]()
                    evs[v][i][1].record()
            torch.cuda.synchronize(dev)

            for v in ("baseline", "fused"):
                durations = [s.elapsed_time(e) * 1e3 for s, e in evs[v]]
                stage_samples[stage][v] = durations
                for i, d in enumerate(durations):
                    raw_records.append((n, stage, v, i, d))

        row = {"n": n}
        for stage in ("step1_router", "step2_dispatch", "step3_expansion", "total_prologue"):
            for v in ("baseline", "fused"):
                arr = np.array(stage_samples[stage][v])
                row[f"{stage}_{v}_med"] = float(np.median(arr))
                row[f"{stage}_{v}_mean"] = float(np.mean(arr))
                row[f"{stage}_{v}_std"] = float(np.std(arr, ddof=1))
            # Paired speedup
            ratios = np.array(stage_samples[stage]["baseline"]) / np.array(stage_samples[stage]["fused"])
            row[f"{stage}_speedup"] = float(np.median(ratios))

        summary_records.append(row)
        print(
            f"  N = {n:4d} | "
            f"Step 1 (Router): {row['step1_router_baseline_med']:5.1f} -> {row['step1_router_fused_med']:5.1f} us ({row['step1_router_speedup']:4.2f}x) | "
            f"Step 2 (Map): {row['step2_dispatch_baseline_med']:5.1f} -> {row['step2_dispatch_fused_med']:5.1f} us ({row['step2_dispatch_speedup']:4.2f}x) | "
            f"Step 3 (Exp): {row['step3_expansion_baseline_med']:5.1f} -> {row['step3_expansion_fused_med']:5.1f} us ({row['step3_expansion_speedup']:4.2f}x) | "
            f"Total: {row['total_prologue_baseline_med']:5.1f} -> {row['total_prologue_fused_med']:5.1f} us ({row['total_prologue_speedup']:4.2f}x)"
        )

    print("=" * 80 + "\n")
    return summary_records, raw_records


# =============================================================================
# Data Loading & Statistics
# =============================================================================

def load_or_compute_data(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RESULTS_DIR / "r2_stage_breakdown_raw.csv"
    sum_path = RESULTS_DIR / "r2_stage_breakdown.csv"

    if args.recompute or not raw_path.exists():
        seq_lens = [1, 16, 64, 128, 256, 512, 1024, 2048, 4096]
        summary_records, raw_records = benchmark_r2_stages(seq_lens, runs=args.runs, warmup=args.warmup, dev_str=args.device)

        # Save raw samples
        with open(raw_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["n", "stage", "variant", "iter", "us"])
            writer.writerows(raw_records)

        # Save summary CSV
        with open(sum_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_records[0].keys()))
            writer.writeheader()
            writer.writerows(summary_records)

    # Load from raw
    acc = defaultdict(lambda: defaultdict(dict))
    with open(raw_path) as f:
        for r in csv.DictReader(f):
            acc[(int(r["n"]), r["stage"])][r["variant"]][int(r["iter"])] = float(r["us"])

    data = {
        key: {v: np.array([it[i] for i in sorted(it)]) for v, it in by_v.items()}
        for key, by_v in acc.items()
    }
    ns = sorted({k[0] for k in data})
    return data, ns


def paired_speedup_ci(data, n, stage):
    ratios = data[(n, stage)]["baseline"] / data[(n, stage)]["fused"]
    idx = RNG.integers(0, ratios.size, (N_BOOT, ratios.size))
    lo, hi = np.percentile(np.median(ratios[idx], axis=1), [2.5, 97.5])
    return float(np.median(ratios)), float(lo), float(hi)


# =============================================================================
# Plotting Routines
# =============================================================================

STAGES = [
    ("step1_router", "Step 1: Router (top-6)", C_ROUTER_EDGE, C_ROUTER_FILL, "o"),
    ("step2_dispatch", "Step 2: Dispatch map", C_MAP_EDGE, C_MAP_FILL, "s"),
    ("step3_expansion", "Step 3: Expansion (128-bit)", C_EXP_EDGE, C_EXP_FILL, "^"),
]
TOTAL = ("total_prologue", "Total R2 prologue", C_TOTAL_EDGE, C_TOTAL_FILL, "D")
SHORT = {"step1_router": "Router", "step2_dispatch": "Map", "step3_expansion": "Expand",
         "total_prologue": "Total"}


def plot_fig1a_r2_stage_latency(data, ns, args, mode):
    """Figure 1a: latency of each R2 sub-stage and the total, baseline (dashed) vs fused (solid)."""
    if mode == "slide":
        fig = slide_latency_panels(data, [(k, SHORT[k]) for k, *_ in STAGES], paired_speedup_ci)
        return save(fig, OUT_DIR, mode, "fig1a_r2_stage_latency", args)

    fig, ax = plt.subplots(figsize=(DETAIL_W, 3.4))
    for key, _label, edge, _fill, mk in STAGES + [TOTAL]:
        total = key == TOTAL[0]
        lw, ms = (2.2, 5) if total else (1.3, 4)
        base = [np.median(data[(n, key)]["baseline"]) for n in ns]
        fused = [np.median(data[(n, key)]["fused"]) for n in ns]
        ax.plot(ns, base, color=edge, ls="--", lw=lw, marker=mk, ms=ms, mfc="white", mew=1.0,
                alpha=0.7, zorder=4 if total else 3)
        ax.plot(ns, fused, color=edge, ls="-", lw=lw, marker=mk, ms=ms, zorder=5 if total else 3)

    log2_axis(ax, ns)
    ax.set_yscale("log")
    log_yaxis(ax)
    ax.set_ylabel("Execution latency ($\\mu$s) ↓")
    ax.set_xlabel("Tokens per forward pass, N")

    handles = [Line2D([], [], color=e, marker=m, lw=2.2 if k == TOTAL[0] else 1.3, label=l)
               for k, l, e, _f, m in STAGES + [TOTAL]]
    handles += [Line2D([], [], color=C_MUTED, ls="--", label="TRT-LLM baseline"),
                Line2D([], [], color=C_MUTED, ls="-", label="Fused (ours)")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, columnspacing=1.2)
    save(fig, OUT_DIR, mode, "fig1a_r2_stage_latency", args)


def plot_fig1b_r2_stage_share(data, ns, args, mode):
    """Figure 1b: share of each sub-stage in R2 latency (baseline plain, fused hatched)."""
    if mode == "slide":
        return _fig1b_slide(data, args)

    fig, ax = plt.subplots(figsize=(DETAIL_W, 3.0))
    x = np.arange(len(ns)) * 1.25
    w, gap = 0.34, 0.05

    for v, off, hatch in (("baseline", -(w + gap) / 2, None), ("fused", (w + gap) / 2, "////")):
        parts = np.array([[np.median(data[(n, k)][v]) for k, *_ in STAGES] for n in ns])
        pct = 100 * parts / parts.sum(axis=1, keepdims=True)
        bottom = np.zeros(len(ns))
        for j, (_k, _l, edge, fill, _m) in enumerate(STAGES):
            ax.bar(x + off, pct[:, j], w, bottom=bottom, color=fill, edgecolor=edge,
                   hatch=hatch, linewidth=0.6, zorder=3)
            bottom += pct[:, j]

    ax.set_xticks(x)
    ax.set_xticklabels([n_label(n) for n in ns])
    ax.set_xlabel("Tokens per forward pass, N")
    ax.set_ylabel("Latency share within R2 (%)")
    ax.set_ylim(0, 100)

    handles = [Patch(facecolor=f, edgecolor=e, lw=0.6, label=l) for _k, l, e, f, _m in STAGES]
    handles += [Patch(facecolor="white", edgecolor=C_MUTED, hatch="////", lw=0.6, label="Fused variant")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, columnspacing=1.2)
    save(fig, OUT_DIR, mode, "fig1b_r2_stage_share", args)


def _fig1b_slide(data, args):
    """Slide: where baseline R2 time goes at decode / mid / prefill (shares printed in the bars)."""
    fig, ax = plt.subplots(figsize=SLIDE_SIZE)
    y = np.arange(len(SLIDE_NS))[::-1]
    parts = np.array([[np.median(data[(n, k)]["baseline"]) for k, *_ in STAGES] for n in SLIDE_NS])
    pct = 100 * parts / parts.sum(axis=1, keepdims=True)
    left = np.zeros(len(SLIDE_NS))
    for j, (key, _label, edge, fill, _m) in enumerate(STAGES):
        ax.barh(y, pct[:, j], 0.62, left=left, color=fill, edgecolor=edge, lw=1.0, label=SHORT[key])
        for yi, l, v in zip(y, left, pct[:, j]):
            if v >= 8:
                ax.text(l + v / 2, yi, f"{v:.0f}%", ha="center", va="center", fontweight="bold",
                        color=edge)
        left += pct[:, j]
    ax.set_yticks(y)
    ax.set_yticklabels([slide_n_label(n).replace("\n", "  ") for n in SLIDE_NS])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share of baseline R2 time (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    save(fig, OUT_DIR, "slide", "fig1b_r2_stage_share", args)


def plot_fig2_r2_stage_speedup(data, ns, args, mode):
    """Figure 2: paired sub-stage speedup (median, 95% bootstrap CI); total emphasised."""
    if mode == "slide":
        fig = slide_speedup_lines(data, [(k, SHORT[k], e) for k, _l, e, _f, _m in STAGES],
                                  (TOTAL[0], "Total R2"), paired_speedup_ci)
        return save(fig, OUT_DIR, mode, "fig2_r2_stage_speedup", args)

    fig, ax = plt.subplots(figsize=(DETAIL_W, 3.0))
    lows = []
    for key, label, edge, _fill, mk in STAGES + [TOTAL]:
        total = key == TOTAL[0]
        s = np.array([paired_speedup_ci(data, n, key) for n in ns])
        lows.append(s[:, 1].min())
        ax.fill_between(ns, s[:, 1], s[:, 2], color=edge, alpha=0.18 if total else 0.07, lw=0,
                        zorder=4 if total else 2)
        if total:
            ax.plot(ns, s[:, 0], color=edge, marker=mk, lw=2.2, ms=5, label=label, zorder=5)
        else:
            ax.plot(ns, s[:, 0], color=edge, marker=mk, lw=1.0, ms=3.5, alpha=0.45, label=label, zorder=3)

    log2_axis(ax, ns)
    ax.axhline(1.0, color=C_PARITY, lw=1.0, ls=":", zorder=1)
    ax.set_ylim(min(0.9, min(lows) - 0.05), None)
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.2f}×")
    ax.set_xlabel("Tokens per forward pass, N")
    ax.set_ylabel("Speedup over baseline ↑")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, columnspacing=1.2)
    save(fig, OUT_DIR, mode, "fig2_r2_stage_speedup", args)


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="Publication figures for Region 2 (MoE Prologue) sub-stages")
    p.add_argument("--runs", type=int, default=50, help="Benchmark runs per sequence length")
    p.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    p.add_argument("--device", type=str, default="cuda:0", help="CUDA device")
    p.add_argument("--recompute", action="store_true", help="Force re-running GPU benchmark instead of loading cached CSV")
    p.add_argument("--mode", choices=[*MODES, "both"], default="both")
    p.add_argument("--formats", nargs="+", default=["pdf", "png"])
    p.add_argument("--dpi", type=int, default=600)
    args = p.parse_args()

    resolve_font()
    data, ns = load_or_compute_data(args)

    print("=" * 80)
    print("Generating Region 2 sub-stage publication figures...")
    print("=" * 80)
    for mode in MODES if args.mode == "both" else [args.mode]:
        with style(mode):
            plot_fig1a_r2_stage_latency(data, ns, args, mode)
            plot_fig1b_r2_stage_share(data, ns, args, mode)
            plot_fig2_r2_stage_speedup(data, ns, args, mode)
    print("\n[+] All Region 2 figures successfully generated in: assets/r2/")


if __name__ == "__main__":
    main()

