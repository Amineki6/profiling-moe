#!/usr/bin/env python3
"""Nsight Systems Profiling Script for Fused Sandwich Norm vs Eager Unfused."""

import os
import sys
import torch
import torch.cuda.nvtx as nvtx

# Ensure environment paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from kernels.cuda.kolibri_sandwich_norm import fused_sandwich_norm, copy_baseline


def eager_sandwich_norm(attn_out, residual_in, gamma1, gamma2, eps=1e-5):
    d = attn_out.shape[-1]
    post_attn = torch.nn.functional.rms_norm(attn_out, (d,), gamma1, eps=eps)
    updated_residual = residual_in + post_attn
    moe_in = torch.nn.functional.rms_norm(updated_residual, (d,), gamma2, eps=eps)
    return updated_residual, moe_in


import argparse


def main():
    parser = argparse.ArgumentParser(description="Profile Fused Sandwich Norm vs Eager")
    parser.add_argument("--tokens", type=int, default=512, help="Token count for profiling (default: 512)")
    parser.add_argument("--iterations", type=int, default=50, help="Number of capture iterations (default: 50, use 1 for NCU)")
    parser.add_argument("--include-copy-baseline", action="store_true", help="Include 128-bit pure copy kernel baseline")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required for profiling!"
    device = "cuda"
    dtype = torch.bfloat16
    hidden_dim = 2560
    num_tokens = args.tokens
    eps = 1e-5

    print(f"Profiling configuration: {num_tokens} tokens, hidden_dim={hidden_dim}, dtype={dtype}, iterations={args.iterations}")

    attn_out = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
    residual_in = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
    gamma1 = torch.ones(hidden_dim, device=device, dtype=dtype)
    gamma2 = torch.ones(hidden_dim, device=device, dtype=dtype)

    # 1. Warm-up outside profiler range
    print("[1/3] Warming up kernels...")
    for _ in range(20):
        _ = eager_sandwich_norm(attn_out, residual_in, gamma1, gamma2, eps=eps)
        _ = fused_sandwich_norm(attn_out, residual_in, gamma1, gamma2, eps=eps)
        if args.include_copy_baseline:
            _ = copy_baseline(attn_out, residual_in)
    torch.cuda.synchronize()

    # 2. Targeted profile capture
    print(f"[2/3] Capturing profile ({args.iterations} iteration(s))...")
    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()

    # --- Copy Kernel Baseline (Speed of Light) ---
    if args.include_copy_baseline:
        nvtx.range_push("Pure_Copy_Baseline_128")
        for _ in range(args.iterations):
            _ = copy_baseline(attn_out, residual_in)
        torch.cuda.synchronize()
        nvtx.range_pop()

    # --- Custom Fused Kernel ---
    nvtx.range_push("Custom_Fused_SandwichNorm")
    for _ in range(args.iterations):
        _ = fused_sandwich_norm(attn_out, residual_in, gamma1, gamma2, eps=eps)
    torch.cuda.synchronize()
    nvtx.range_pop()

    cudart.cudaProfilerStop()
    print("[3/3] Profiling completed.")


if __name__ == "__main__":
    main()
