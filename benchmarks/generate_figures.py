#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publication-grade visualization generator for Kolibri 1 MoE Profiling.

Matches the clean, academic aesthetic of the Kolibri / FineWeb publications:
- Pure white background (#ffffff)
- Subtle, crisp axis spines and faint gridlines
- Signature Kolibri palette: Deep Teal (#0d5c5a / #a7f3d0), Deep Berry/Magenta (#9b1158 / #fce7f3),
  Warm Amber (#d97706 / #fef3c7), and Warm Neutral Taupe (#78716c / #f5f5f4)
- Academic typography with diagonal hatches and clean value annotations
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set up paths
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
ASSETS_DIR = REPO_ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# Theme Palette (Academic White & Kolibri Signature Colors)
BG_COLOR = "#ffffff"
TEXT_COLOR = "#1f2937"        # Dark slate / charcoal
MUTED_TEXT = "#4b5563"        # Medium gray
SPINE_COLOR = "#4b5563"       # Axis spine gray
GRID_COLOR = "#f1f5f9"        # Very faint gray grid

# Series Palettes (Border + Fill + Hatch)
# 1. Baseline / Eager (Warm Taupe / Neutral Slate)
COLOR_TAUPE_EDGE = "#78716c"
COLOR_TAUPE_FILL = "#f5f5f4"

# 2. torch.compile (Warm Amber / Ochre)
COLOR_AMBER_EDGE = "#d97706"
COLOR_AMBER_FILL = "#fef3c7"

# 3. Fused Triton / Option 2 (Kolibri Signature Deep Teal & Mint)
COLOR_TEAL_EDGE = "#0d5c5a"
COLOR_TEAL_FILL = "#a7f3d0"

# 4. Fused CUDA SWA / MoE Router (Kolibri Signature Deep Berry / Magenta)
COLOR_BERRY_EDGE = "#9b1158"
COLOR_BERRY_FILL = "#fce7f3"

# Global plot styling
plt.rcParams.update({
    "font.sans-serif": ["Inter", "DejaVu Sans", "Helvetica Neue", "Arial", "sans-serif"],
    "font.family": "sans-serif",
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": BG_COLOR,
    "axes.edgecolor": SPINE_COLOR,
    "axes.linewidth": 0.85,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.major.width": 0.85,
    "ytick.major.width": 0.85,
    "text.color": TEXT_COLOR,
    "grid.color": GRID_COLOR,
    "grid.linestyle": "-",
    "grid.linewidth": 0.8,
    "grid.alpha": 1.0,
    "hatch.linewidth": 0.85,
})


def plot_figure_1_router_benchmark():
    """Figure 1: MoE Router Microbenchmark (Batch Size Scaling)."""
    router_json_path = RESULTS_DIR / "01_router_results.json"
    with open(router_json_path, "r") as f:
        data = json.load(f)["results"]

    batch_sizes = [d["batch_size"] for d in data]
    eager_us = [d["eager_us"] for d in data]
    compiled_us = [d["compiled_us"] for d in data]
    triton_us = [d["triton_us"] for d in data]
    triton_speedups = [d["triton_speedup"] for d in data]

    x = np.arange(len(batch_sizes))
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=300)

    # Bars with paper-matching borders, pastel fills, and diagonal hatches
    rects1 = ax.bar(
        x - bar_width, eager_us, bar_width,
        label="PyTorch Eager Baseline",
        color=COLOR_TAUPE_FILL, edgecolor=COLOR_TAUPE_EDGE, hatch="///", linewidth=0.9, zorder=3
    )
    rects2 = ax.bar(
        x, compiled_us, bar_width,
        label="torch.compile (Inductor)",
        color=COLOR_AMBER_FILL, edgecolor=COLOR_AMBER_EDGE, linewidth=0.9, zorder=3
    )
    rects3 = ax.bar(
        x + bar_width, triton_us, bar_width,
        label="Custom Fused Triton Kernel",
        color=COLOR_TEAL_FILL, edgecolor=COLOR_TEAL_EDGE, hatch="///", linewidth=0.9, zorder=3
    )

    # Speedup annotations above Triton bars
    for rect, spd in zip(rects3, triton_speedups):
        height = rect.get_height()
        ax.annotate(
            f"{spd}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=8.5, fontweight="bold",
            color=COLOR_TEAL_EDGE
        )

    # Spines & Grid
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels([f"b = {b}" for b in batch_sizes], fontsize=9.5)
    ax.set_xlabel("Batch size (tokens)", fontsize=10.5, fontweight="semibold", labelpad=8)
    ax.set_ylabel("Kernel latency (µs) ↓", fontsize=10.5, fontweight="semibold", labelpad=8)
    ax.set_ylim(0, 31)

    # Clean centered Title & Subtitle
    ax.set_title("Kolibri 1 MoE Router Microbenchmark (E = 384, k = 6)\n"
                 "NVIDIA RTX 4090 • Pre-activation logit bias + Top-6 + Unnormalized sigmoid",
                 fontsize=11.5, fontweight="bold", color=TEXT_COLOR, pad=14, loc="center")

    # Legend at the top
    legend = ax.legend(
        frameon=True, facecolor="#ffffff", edgecolor="#d1d5db",
        fontsize=9, loc="upper left", framealpha=0.95
    )
    legend.get_frame().set_linewidth(0.8)

    # Summary callout box
    ax.text(
        0.98, 0.94,
        "~3.0× speedup across all batch sizes\n"
        "Fuses 5 disjoint CUDA kernels into 1 SRAM pass",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8.5, color=MUTED_TEXT,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=0.8)
    )

    plt.tight_layout()
    output_path = ASSETS_DIR / "01_router_microbenchmark.png"
    plt.savefig(output_path, dpi=300, facecolor=BG_COLOR)
    plt.close()
    print(f"Generated: {output_path}")


def plot_figure_2a_swa_benchmark():
    """Figure 2A: Sliding Window Attention (SWA) 40-Layer Norm & RoPE Fusion."""
    attn_json_path = RESULTS_DIR / "02_attention_results.json"
    with open(attn_json_path, "r") as f:
        full_data = json.load(f)

    swa_data = full_data["swa_results"]
    batch_sizes = [d["batch_size"] for d in swa_data]
    x = np.arange(len(batch_sizes))
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(8.8, 4.6), dpi=300)

    swa_unfused = [d["unfused_us"] for d in swa_data]
    swa_comp = [d["compiled_us"] for d in swa_data]
    swa_fused = [d["fused_cuda_us"] for d in swa_data]
    swa_speedups = [d["speedup"] for d in swa_data]

    r1 = ax.bar(
        x - bar_width, swa_unfused, bar_width,
        label="Unfused Baseline (PyTorch Eager)",
        color=COLOR_TAUPE_FILL, edgecolor=COLOR_TAUPE_EDGE, hatch="///", linewidth=0.9, zorder=3
    )
    r2 = ax.bar(
        x, swa_comp, bar_width,
        label="torch.compile (Inductor)",
        color=COLOR_AMBER_FILL, edgecolor=COLOR_AMBER_EDGE, linewidth=0.9, zorder=3
    )
    r3 = ax.bar(
        x + bar_width, swa_fused, bar_width,
        label="Fused CUDA Kernel (TRT-LLM)",
        color=COLOR_BERRY_FILL, edgecolor=COLOR_BERRY_EDGE, hatch="///", linewidth=0.9, zorder=3
    )

    for rect, spd in zip(r3, swa_speedups):
        height = rect.get_height()
        ax.annotate(
            spd,
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=8.5, fontweight="bold",
            color=COLOR_BERRY_EDGE
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"b = {b}" for b in batch_sizes], fontsize=9.5)
    ax.set_xlabel("Batch size (tokens)", fontsize=10.5, fontweight="semibold", labelpad=8)
    ax.set_ylabel("Kernel latency (µs) ↓", fontsize=10.5, fontweight="semibold", labelpad=8)
    ax.set_ylim(0, 26)

    ax.set_title("Kolibri 1 Sliding Window Attention (SWA) Norm & RoPE Fusion\n"
                 "40 Layers (80% of model) • RMSNorm + Rotary Embeddings • NVIDIA RTX 4090",
                 fontsize=11, fontweight="bold", color=TEXT_COLOR, pad=12, loc="center")

    leg = ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#d1d5db", fontsize=9, loc="upper left")
    leg.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    output_path = ASSETS_DIR / "02_attention_norm_swa.png"
    plt.savefig(output_path, dpi=300, facecolor=BG_COLOR)
    plt.close()
    print(f"Generated: {output_path}")


def plot_figure_2b_full_attention_benchmark():
    """Figure 2B: Full Attention (RNoPE) 10-Layer Norm Fusion via Zero-Position Workaround."""
    attn_json_path = RESULTS_DIR / "02_attention_results.json"
    with open(attn_json_path, "r") as f:
        full_data = json.load(f)

    full_data_res = full_data["full_attention_results"]
    batch_sizes = [d["batch_size"] for d in full_data_res]
    x = np.arange(len(batch_sizes))
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(8.8, 4.6), dpi=300)

    full_unfused = [d["unfused_us"] for d in full_data_res]
    full_comp = [d["compiled_us"] for d in full_data_res]
    full_workaround = [d["workaround_fused_us"] for d in full_data_res]
    full_speedups = [d["workaround_speedup"] for d in full_data_res]

    r1 = ax.bar(
        x - bar_width, full_unfused, bar_width,
        label="Unfused Baseline (RNoPE RMSNorm)",
        color=COLOR_TAUPE_FILL, edgecolor=COLOR_TAUPE_EDGE, hatch="///", linewidth=0.9, zorder=3
    )
    r2 = ax.bar(
        x, full_comp, bar_width,
        label="torch.compile (Inductor)",
        color=COLOR_AMBER_FILL, edgecolor=COLOR_AMBER_EDGE, linewidth=0.9, zorder=3
    )
    r3 = ax.bar(
        x + bar_width, full_workaround, bar_width,
        label="Option 2: Zero-Pos Fused (TRT-LLM)",
        color=COLOR_TEAL_FILL, edgecolor=COLOR_TEAL_EDGE, hatch="///", linewidth=0.9, zorder=3
    )

    for rect, spd in zip(r3, full_speedups):
        height = rect.get_height()
        ax.annotate(
            spd,
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=8.5, fontweight="bold",
            color=COLOR_TEAL_EDGE
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"b = {b}" for b in batch_sizes], fontsize=9.5)
    ax.set_xlabel("Batch size (tokens)", fontsize=10.5, fontweight="semibold", labelpad=8)
    ax.set_ylabel("Kernel latency (µs) ↓", fontsize=10.5, fontweight="semibold", labelpad=8)
    ax.set_ylim(0, 31)

    ax.set_title("Kolibri 1 Full Attention (RNoPE) Norm Fusion\n"
                 "10 Layers (20% of model) • Pure RMSNorm via Zero-Position Identity Bypass • NVIDIA RTX 4090",
                 fontsize=11, fontweight="bold", color=TEXT_COLOR, pad=12, loc="center")

    leg = ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#d1d5db", fontsize=9, loc="upper left")
    leg.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    output_path = ASSETS_DIR / "02_attention_norm_full.png"
    plt.savefig(output_path, dpi=300, facecolor=BG_COLOR)
    plt.close()
    print(f"Generated: {output_path}")


def plot_figure_3_decode_impact():
    """Figure 3: Full 50-Layer Stack Decode Impact per Token (Hero Chart)."""
    categories = ["Baseline Architecture\n(All Unfused)", "Optimized Pipeline\n(Fused CUDA + Triton)"]

    attn_times = [1.678, 0.559]
    router_times = [0.984, 0.368]
    totals = [a + r for a, r in zip(attn_times, router_times)]

    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=300)

    y_pos = np.arange(len(categories))
    bar_height = 0.40

    # Horizontal stacked bars
    p1 = ax.barh(
        y_pos, attn_times, bar_height,
        label="50x Attention Norm (SWA + RNoPE)",
        color=COLOR_TEAL_FILL, edgecolor=COLOR_TEAL_EDGE, hatch="///", linewidth=0.9, zorder=3
    )
    p2 = ax.barh(
        y_pos, router_times, bar_height, left=attn_times,
        label="50x MoE Routing (384 Experts, Top-6)",
        color=COLOR_BERRY_FILL, edgecolor=COLOR_BERRY_EDGE, hatch="\\\\", linewidth=0.9, zorder=3
    )

    # High-contrast segment text inside the bars with clean white badges to avoid hatch interference
    badge_kw_teal = dict(boxstyle="round,pad=0.24", facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=0.8, alpha=0.96)
    badge_kw_berry = dict(boxstyle="round,pad=0.24", facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=0.8, alpha=0.96)

    ax.text(attn_times[0] / 2, 0, f"{attn_times[0]:.2f} ms", ha="center", va="center", color=COLOR_TEAL_EDGE, fontweight="bold", fontsize=9.5, bbox=badge_kw_teal, zorder=5)
    ax.text(attn_times[0] + router_times[0] / 2, 0, f"{router_times[0]:.2f} ms", ha="center", va="center", color=COLOR_BERRY_EDGE, fontweight="bold", fontsize=9.5, bbox=badge_kw_berry, zorder=5)

    ax.text(attn_times[1] / 2, 1, f"{attn_times[1]:.2f} ms", ha="center", va="center", color=COLOR_TEAL_EDGE, fontweight="bold", fontsize=9.5, bbox=badge_kw_teal, zorder=5)
    ax.text(attn_times[1] + router_times[1] / 2, 1, f"{router_times[1]:.2f} ms", ha="center", va="center", color=COLOR_BERRY_EDGE, fontweight="bold", fontsize=9.5, bbox=badge_kw_berry, zorder=5)

    # End labels
    ax.text(totals[0] + 0.06, 0, f"{totals[0]:.2f} ms total", ha="left", va="center", color=TEXT_COLOR, fontweight="bold", fontsize=10.5)
    ax.text(totals[1] + 0.06, 1, f"{totals[1]:.2f} ms total  (2.87× faster)", ha="left", va="center", color=COLOR_TEAL_EDGE, fontweight="bold", fontsize=10.5)

    # Latency saved bracket span
    delta_ms = totals[0] - totals[1]
    pct_reduction = (delta_ms / totals[0]) * 100

    ax.hlines(y=0.5, xmin=totals[1], xmax=totals[0], color=COLOR_TEAL_EDGE, linestyle="--", linewidth=1.2, zorder=2)
    ax.plot([totals[1], totals[1]], [0.44, 0.56], color=COLOR_TEAL_EDGE, linewidth=1.4, zorder=2)
    ax.plot([totals[0], totals[0]], [0.44, 0.56], color=COLOR_TEAL_EDGE, linewidth=1.4, zorder=2)
    ax.text(
        (totals[0] + totals[1]) / 2, 0.5,
        f"  −{delta_ms:.2f} ms per token saved ({pct_reduction:.1f}% reduction)  ",
        ha="center", va="center", color=COLOR_TEAL_EDGE, fontweight="bold", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f0fdf4", edgecolor=COLOR_TEAL_EDGE, linewidth=1.0),
        zorder=5
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", zorder=0)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=10.5, fontweight="semibold")
    ax.set_xlabel("Decode latency per token (ms) ↓", fontsize=10.5, fontweight="semibold", labelpad=8)
    ax.set_xlim(0, 3.2)
    ax.invert_yaxis()  # Baseline on top

    ax.set_title("Kolibri 1 (78B MoE) 50-Layer Non-GEMM Decode Overhead\n"
                 "Single-token autoregressive generation step (batch = 1) on NVIDIA RTX 4090",
                 fontsize=11.5, fontweight="bold", color=TEXT_COLOR, pad=12, loc="center")

    legend = ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#d1d5db", fontsize=9, loc="lower right")
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    output_path = ASSETS_DIR / "03_end_to_end_decode_impact.png"
    plt.savefig(output_path, dpi=300, facecolor=BG_COLOR)
    plt.close()
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    print("=" * 80)
    print("Generating Academic White Palette Visualizations for Kolibri 1 Profiling Repo")
    print("=" * 80)
    plot_figure_1_router_benchmark()
    plot_figure_2a_swa_benchmark()
    plot_figure_2b_full_attention_benchmark()
    plot_figure_3_decode_impact()
    print("\nAll 4 figures generated successfully in assets/ directory!")
