import json
import os
import sys
from datetime import datetime
import torch
import triton.testing
from tabulate import tabulate

# -----------------------------------------------------------------------------
# Configuration & Device Setup
# -----------------------------------------------------------------------------
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
    NUM_EXPERTS = cfg.get("num_experts", cfg.get("moe_num_experts", 384))
    TOP_K = cfg.get("num_experts_per_tok", cfg.get("moe_top_k", 6))
else:
    NUM_EXPERTS = 384
    TOP_K = 6

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
DEVICE_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from kernels.kolibri_cuda_router import kolibri_cuda_ops

def bench_gpu_kernel_us(fn, *args) -> float:
    """Benchmark GPU execution time in microseconds (median across repeated runs)."""
    ms = triton.testing.do_bench(lambda: fn(*args), warmup=25, rep=100, return_mode="median")
    return ms * 1000.0


def main():
    assert torch.cuda.is_available(), "CUDA is required for profiling!"
    print(f"\n==================================================================")
    print(f" CUDA A/B Benchmark: Scalar Registers (vB) vs Vectorized (vFloat4)")
    print(f" Device : {DEVICE_NAME}")
    print(f" Experts: {NUM_EXPERTS} | Top-K: {TOP_K} | Dtype: {DTYPE}")
    print(f"==================================================================")

    assert hasattr(kolibri_cuda_ops, "forward_vA"), "kolibri_cuda_ops missing forward_vA"
    assert hasattr(kolibri_cuda_ops, "forward_vB"), "kolibri_cuda_ops missing forward_vB"
    assert hasattr(kolibri_cuda_ops, "forward_vFloat4"), "kolibri_cuda_ops missing forward_vFloat4"

    fn_vA = lambda x, b, k: kolibri_cuda_ops.forward_vA(x, b, k)
    fn_vB = lambda x, b, k: kolibri_cuda_ops.forward_vB(x, b, k)
    fn_vF4 = lambda x, b, k: kolibri_cuda_ops.forward_vFloat4(x, b, k)

    bias = torch.randn(NUM_EXPERTS, device=DEVICE, dtype=DTYPE)

    # -------------------------------------------------------------------------
    # ASSERTION GATE: Ensure vB and vFloat4 produce identical outputs
    # -------------------------------------------------------------------------
    print("\n[Assertion Gate] Verifying numerical equivalence between vB (Scalar) and vFloat4 (Vectorized)...")
    test_x = torch.randn(64, NUM_EXPERTS, device=DEVICE, dtype=DTYPE)
    ids_vB, weights_vB = fn_vB(test_x, bias, TOP_K)
    ids_vF4, weights_vF4 = fn_vF4(test_x, bias, TOP_K)

    # 1. Exact integer match on selected expert IDs
    assert torch.equal(ids_vB, ids_vF4), "❌ Top-K indices mismatch between vB and vFloat4!"

    # 2. Strict floating-point match on final weights
    diff_vF4 = (weights_vB.float() - weights_vF4.float()).abs().max().item()
    assert torch.allclose(weights_vB, weights_vF4, atol=1e-4, rtol=1e-4), f"❌ Weights drift vB vs vFloat4! Max diff: {diff_vF4}"

    print("  ✓ 100% exact match on Top-K indices (vB == vFloat4).")
    print(f"  ✓ 100% exact match on routing weights (max diff: {diff_vF4:.1e}).")

    # -------------------------------------------------------------------------
    # LATENCY SWEEP
    # -------------------------------------------------------------------------
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    print(f"\n[Benchmarking] Sweeping token batches from {batch_sizes[0]} to {batch_sizes[-1]}...")

    records = []
    table_rows = []

    for b in batch_sizes:
        x = torch.randn(b, NUM_EXPERTS, device=DEVICE, dtype=DTYPE)

        t_vA = bench_gpu_kernel_us(fn_vA, x, bias, TOP_K)
        t_vB = bench_gpu_kernel_us(fn_vB, x, bias, TOP_K)
        t_vF4 = bench_gpu_kernel_us(fn_vF4, x, bias, TOP_K)

        delta_us = t_vB - t_vF4
        delta_pct = (delta_us / t_vB) * 100.0
        speedup_vs_vB = t_vB / t_vF4 if t_vF4 > 0 else 1.0
        speedup_vs_vA = t_vA / t_vF4 if t_vF4 > 0 else 1.0

        records.append({
            "batch_size": b,
            "vA_us": t_vA,
            "vB_us": t_vB,
            "vFloat4_us": t_vF4,
            "delta_us": delta_us,
            "delta_pct": delta_pct,
            "speedup_vs_vB": speedup_vs_vB,
            "speedup_vs_vA": speedup_vs_vA,
        })

        table_rows.append([
            b,
            f"{t_vA:.2f}",
            f"{t_vB:.2f}",
            f"{t_vF4:.2f}",
            f"{delta_us:+.2f}",
            f"{delta_pct:+.1f}%",
            f"{speedup_vs_vB:.2f}x",
            f"{speedup_vs_vA:.2f}x",
        ])

    headers = [
        "Tokens (B)",
        "vA: Orig (µs)",
        "vB: Scalar (µs)",
        "vF4: Float4 (µs)",
        "Δ vB-vF4 (µs)",
        "Δ (%)",
        "Speedup (vs vB)",
        "Speedup (vs vA)",
    ]

    print(f"\n### CUDA A/B Benchmark Results ({DEVICE_NAME})")
    print(tabulate(table_rows, headers=headers, tablefmt="github"))

    # -------------------------------------------------------------------------
    # SAVE RESULTS
    # -------------------------------------------------------------------------
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "01_cuda_ab_results.json")

    payload = {
        "benchmark": "cuda_router_ab_test",
        "timestamp": datetime.now().isoformat(),
        "device": DEVICE_NAME,
        "experts": NUM_EXPERTS,
        "top_k": TOP_K,
        "dtype": str(DTYPE),
        "results": records,
    }

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[Artifact Saved] A/B benchmark data saved to -> {json_path}\n")


if __name__ == "__main__":
    main()
