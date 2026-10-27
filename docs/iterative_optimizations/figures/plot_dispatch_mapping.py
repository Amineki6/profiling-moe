import os
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------------------------------------------
# Publication Styling Configuration (IEEE / ACM / NeurIPS Grade)
# -------------------------------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10.5,
    "figure.titlesize": 15,
    "lines.linewidth": 2.2,
    "lines.markersize": 7,
    "axes.linewidth": 1.2,
    "grid.linewidth": 0.8,
    "grid.alpha": 0.35,
    "mathtext.fontset": "cm",
})

# -------------------------------------------------------------------------------------------------
# Measured Empirical Hardware Data (RTX 4090, sm_89, E = 384, K = 6)
# -------------------------------------------------------------------------------------------------
N = np.array([16, 32, 64, 96, 128, 192, 256, 384, 512, 640, 768, 1024, 1152, 2048, 4096, 8192, 98304])

# Latencies in microseconds (µs)
trt_lat = np.array([3.64, 3.68, 3.75, 4.25, 4.88, 5.76, 6.71, 9.50, 12.50, 16.20, 20.80, 29.50, 34.80, 68.20, 136.50, 268.00, 2180.00])
sb_lat  = np.array([4.10, 3.87, 4.34, 4.51, 4.63, 5.46, 6.39, 7.15, 8.79, 10.16, 11.41, 19.44, 27.19, 50.16, 91.61, 176.82, 2061.58])
mb_lat  = np.array([14.94, 6.51, 7.67, 7.31, 7.45, 8.34, 9.46, 8.45, 9.25, 9.85, 9.11, 11.00, 9.54, 10.60, 10.54, 11.18, 31.54])

speedup_mb_sb  = sb_lat / mb_lat
speedup_mb_trt = trt_lat / mb_lat

# -------------------------------------------------------------------------------------------------
# Create 2-Panel Figure
# -------------------------------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

# Colors
c_trt = "#D97706"     # Amber / Orange (TRT-LLM Baseline)
c_sb  = "#D9534F"     # Crimson / Coral (Single-Block Custom)
c_mb  = "#2B6CB0"     # Sapphire Blue (Multi-Block 3-Stage)
c_zone_dec  = "#F0F4F8"
c_zone_pref = "#FDF8F0"

# =================================================================================================
# Panel (a): Latency vs Sequence Length
# =================================================================================================
# Background regime shading
ax1.axvspan(12, 512, color=c_zone_dec, alpha=0.55, zorder=0, label="Decode Regime (Single-Block Optimal)")
ax1.axvspan(512, 130000, color=c_zone_pref, alpha=0.55, zorder=0, label="Prefill Regime (Multi-Block Optimal)")

# Vertical tipping point divider
ax1.axvline(x=512, color="#4A5568", linestyle="--", linewidth=1.5, zorder=2)

# Plot curves
ax1.plot(N, trt_lat, marker="^", color=c_trt, linestyle="--", label="TRT-LLM Baseline (CUB / Fallback)", zorder=3)
ax1.plot(N, sb_lat,  marker="o", color=c_sb,  linestyle="-",  label="Single-Block Baseline (1 SM)", zorder=4)
ax1.plot(N, mb_lat,  marker="s", color=c_mb,  linestyle="-",  label="Multi-Block 3-Stage Pipeline (128 SMs)", zorder=5)

ax1.set_xscale("log", base=2)
ax1.set_yscale("log")
ax1.set_xlabel(r"Sequence Length ($N$ tokens, $\mathrm{Top}\text{-}K=6, E=384$)")
ax1.set_ylabel(r"Execution Latency ($\mu\mathrm{s}$)")
ax1.set_title("(a) Dispatch Mapping Latency vs. Sequence Length", loc="left", fontweight="bold")
ax1.grid(True, which="both", linestyle=":")

# Clean non-overlapping badge for Tipping Point (placed to the left of the divider in open space)
ax1.text(
    450, 60,
    "Tipping Point (N = 512)\nMulti-Block overtakes Single-Block",
    fontsize=9.5,
    fontweight="semibold",
    color="#2D3748",
    va="center",
    ha="right",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#CBD5E0", alpha=0.92),
    zorder=6
)

# Text annotation directly next to the N=98304 data points (no arrows)
ax1.text(
    75000, 31.54,
    "31.5 µs (65.4× – 69.1× faster)",
    fontsize=9.5,
    fontweight="bold",
    color=c_mb,
    ha="right",
    va="center",
    zorder=6
)
ax1.text(
    75000, 2100.0,
    ">2,000 µs (1-SM / Fallback Wall)",
    fontsize=9.5,
    fontweight="bold",
    color=c_sb,
    ha="right",
    va="center",
    zorder=6
)

ax1.set_xlim(12, 130000)
ax1.set_ylim(2.5, 4500)
ax1.legend(loc="upper left", framealpha=0.92, facecolor="white")

# =================================================================================================
# Panel (b): Speedup vs Sequence Length
# =================================================================================================
ax2.axvspan(12, 512, color=c_zone_dec, alpha=0.55, zorder=0, label="Decode Regime (N < 512)")
ax2.axvspan(512, 130000, color=c_zone_pref, alpha=0.55, zorder=0, label="Prefill Regime (N ≥ 512)")
ax2.axvline(x=512, color="#4A5568", linestyle="--", linewidth=1.5, zorder=2)
ax2.axhline(y=1.0, color="#E53E3E", linestyle=":", linewidth=1.6, label="1.0× Parity Baseline", zorder=3)

ax2.plot(N, speedup_mb_sb,  marker="s", color=c_mb,  linestyle="-",  label="Multi-Block vs. Single-Block", zorder=4)
ax2.plot(N, speedup_mb_trt, marker="^", color=c_trt, linestyle="--", label="Multi-Block vs. TRT-LLM Baseline", zorder=5)

ax2.set_xscale("log", base=2)
ax2.set_yscale("log")
ax2.set_xlabel(r"Sequence Length ($N$ tokens, $\mathrm{Top}\text{-}K=6, E=384$)")
ax2.set_ylabel(r"Speedup Factor ($\times$)")
ax2.set_title("(b) Multi-Block Speedup vs. Baselines", loc="left", fontweight="bold")
ax2.grid(True, which="both", linestyle=":")

# Tipping point text at bottom near x=512
ax2.text(
    560, 0.16,
    "Tipping Point (N = 512)\nSpeedup ≥ 1.0×",
    fontsize=9.5,
    fontweight="semibold",
    color="#2D3748",
    va="bottom",
    ha="left",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#CBD5E0", alpha=0.92),
    zorder=6
)

# Peak speedup text directly near end point (no arrows)
ax2.text(
    75000, 68.0,
    "65.4× vs SB (69.1× vs TRT)",
    fontsize=9.5,
    fontweight="bold",
    color=c_mb,
    ha="right",
    va="center",
    zorder=6
)

ax2.set_xlim(12, 130000)
ax2.set_ylim(0.12, 150)
ax2.legend(loc="upper left", framealpha=0.92, facecolor="white")

plt.tight_layout()

# Save publication formats
out_dir = os.path.dirname(os.path.abspath(__file__))
png_path = os.path.join(out_dir, "build_expert_maps_tipping_point.png")
pdf_path = os.path.join(out_dir, "build_expert_maps_tipping_point.pdf")

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
print(f"Saved high-res PNG: {png_path}")
print(f"Saved vector PDF:   {pdf_path}")
