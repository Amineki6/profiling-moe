#!/usr/bin/env python3
"""Nsight Systems Profiling Script for Candidate 2: Fused Add + Post-MoE Norm."""

import argparse
import os
import sys
import torch
import torch.cuda.nvtx as nvtx

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from kernels.cuda.kolibri_post_moe_norm import fused_add_post_moe_norm, copy_baseline


def eager_add_post_moe_norm(routed_out, shared_out, gamma, eps=1e-5):
    d = routed_out.shape[-1]
    combined = routed_out + shared_out
    moe_out = torch.nn.functional.rms_norm(combined, (d,), gamma, eps=eps)
    return moe_out


def main():
    parser = argparse.ArgumentParser(description="Profile Fused Add + Post-MoE Norm vs Eager")
    parser.add_argument("--tokens", type=int, default=512, help="Token count for profiling (default: 512)")
    parser.add_argument("--iterations", type=int, default=50, help="Number of capture iterations (default: 50, use 1 for NCU)")
    parser.add_argument("--include-copy-baseline", action="store_true", help="Include 128-bit pure copy baseline")
    parser.add_argument("--skip-eager", action="store_true", help="Skip eager baseline (useful for NCU)")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required for profiling!"
    device = "cuda"
    dtype = torch.bfloat16
    hidden_dim = 2560
    num_tokens = args.tokens
    eps = 1e-5

    print(f"Profiling configuration: {num_tokens} tokens, hidden_dim={hidden_dim}, dtype={dtype}, iterations={args.iterations}")

    routed_out = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
    shared_out = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
    gamma = torch.ones(hidden_dim, device=device, dtype=dtype)

    # 1. Warm-up
    print("[1/3] Warming up kernels...")
    for _ in range(20):
        if not args.skip_eager:
            _ = eager_add_post_moe_norm(routed_out, shared_out, gamma, eps=eps)
        if args.include_copy_baseline:
            _ = copy_baseline(routed_out, shared_out)
        _ = fused_add_post_moe_norm(routed_out, shared_out, gamma, eps=eps)
    torch.cuda.synchronize()

    # 2. Capture
    print(f"[2/3] Capturing profile ({args.iterations} iteration(s))...")
    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()

    if args.include_copy_baseline:
        nvtx.range_push("Pure_Copy_Baseline_128")
        for _ in range(args.iterations):
            _ = copy_baseline(routed_out, shared_out)
        torch.cuda.synchronize()
        nvtx.range_pop()

    if not args.skip_eager:
        nvtx.range_push("Eager_Unfused_AddPostMoeNorm")
        for _ in range(args.iterations):
            _ = eager_add_post_moe_norm(routed_out, shared_out, gamma, eps=eps)
        torch.cuda.synchronize()
        nvtx.range_pop()

    nvtx.range_push("Custom_Fused_AddPostMoeNorm")
    for _ in range(args.iterations):
        _ = fused_add_post_moe_norm(routed_out, shared_out, gamma, eps=eps)
    torch.cuda.synchronize()
    nvtx.range_pop()

    cudart.cudaProfilerStop()
    print("[3/3] Profiling completed.")


if __name__ == "__main__":
    main()
