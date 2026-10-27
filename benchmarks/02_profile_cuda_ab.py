import argparse
import os
import sys
import torch
import torch.cuda.nvtx as nvtx

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from kernels.cuda import kolibri_cuda_ops

def main():
    parser = argparse.ArgumentParser(description="Profile Kolibri Router CUDA Kernels")
    parser.add_argument("--tokens", type=int, default=1152, help="Token count / batch size (default: 1152)")
    parser.add_argument("--iterations", type=int, default=50, help="Profiling iterations (default: 50, use 1 for NCU)")
    parser.add_argument("--variant", type=str, choices=["all", "standard", "vB", "vFloat4"], default="standard", help="Variant to profile (default: standard)")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup iterations (recommended for NCU profiling)")
    args = parser.parse_args()

    # Map legacy vB to standard
    target_variant = "standard" if args.variant == "vB" else args.variant

    assert torch.cuda.is_available(), "CUDA is required for profiling!"
    device = "cuda"
    dtype = torch.float32
    n_experts = 384
    top_k = 6
    batch_size = args.tokens
    iters = args.iterations

    print(f"Profiling Kolibri Router: tokens={batch_size}, E={n_experts}, top_k={top_k}, iters={iters}, variant={target_variant}")

    x = torch.randn(batch_size, n_experts, device=device, dtype=dtype)
    bias = torch.randn(n_experts, device=device, dtype=dtype)

    # 1. Warm-up (skipped when iters == 1 or --no-warmup to keep NCU fast and lightweight)
    warmup_iters = 0 if (args.no_warmup or iters == 1) else 10
    if warmup_iters > 0:
        print(f"[1/3] Warming up kernels ({warmup_iters} iters)...")
        for _ in range(warmup_iters):
            if target_variant in ["all", "standard"]:
                kolibri_cuda_ops.forward(x, bias, top_k)
            if target_variant in ["all", "vFloat4"]:
                kolibri_cuda_ops.forward_vFloat4(x, bias, top_k)
        torch.cuda.synchronize()
    else:
        print("[1/3] Warm-up skipped for profiling.")

    # 2. Profile only the targeted executions
    print("[2/3] Profiling under NVTX ranges...")
    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()

    # --- Standard Production Kolibri Router (customKolibriRoutingKernels) ---
    if target_variant in ["all", "standard"]:
        nvtx.range_push("Kolibri Router: Standard (customKolibriRoutingKernels)")
        for _ in range(iters):
            kolibri_cuda_ops.forward(x, bias, top_k)
        torch.cuda.synchronize()
        nvtx.range_pop()

    # --- Variant Float4: Vectorized 128-bit Loads ---
    if target_variant in ["all", "vFloat4"]:
        nvtx.range_push("Variant Float4: Vectorized (customKolibriRoutingKernelsFloat4)")
        for _ in range(iters):
            kolibri_cuda_ops.forward_vFloat4(x, bias, top_k)
        torch.cuda.synchronize()
        nvtx.range_pop()

    cudart.cudaProfilerStop()
    print("[3/3] Profiling completed successfully.")

if __name__ == "__main__":
    main()
