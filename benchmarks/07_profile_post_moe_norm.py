#!/usr/bin/env python3
"""Nsight Systems Profiling Script for Candidate 2: Fused Add + Post-MoE Norm."""

import argparse
import os
import sys
import torch
import torch.cuda.nvtx as nvtx

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from kernels.cuda.kolibri_post_moe_norm import fused_add_post_moe_norm


def eager_add_post_moe_norm(routed_out, shared_out, gamma, eps=1e-5):
    d = routed_out.shape[-1]
    combined = routed_out + shared_out
    moe_out = torch.nn.functional.rms_norm(combined, (d,), gamma, eps=eps)
    return moe_out


def main():
    parser = argparse.ArgumentParser(description="Profile Fused Add + Post-MoE Norm vs Eager")
    parser.add_argument("--tokens", type=int, default=512, help="Token count for profiling (default: 512)")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required for profiling!"
    device = "cuda"
    dtype = torch.bfloat16
    hidden_dim = 2560
    num_tokens = args.tokens
    eps = 1e-5

    print(f"Profiling configuration: {num_tokens} tokens, hidden_dim={hidden_dim}, dtype={dtype}")

    routed_out = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
    shared_out = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
    gamma = torch.ones(hidden_dim, device=device, dtype=dtype)

    # 1. Warm-up
    print("[1/3] Warming up kernels...")
    for _ in range(50):
        _ = eager_add_post_moe_norm(routed_out, shared_out, gamma, eps=eps)
        _ = fused_add_post_moe_norm(routed_out, shared_out, gamma, eps=eps)
    torch.cuda.synchronize()

    # 2. Capture
    print("[2/3] Capturing profile with Nsight Systems...")
    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()

    nvtx.range_push("Eager_Unfused_AddPostMoeNorm")
    for _ in range(50):
        _ = eager_add_post_moe_norm(routed_out, shared_out, gamma, eps=eps)
    torch.cuda.synchronize()
    nvtx.range_pop()

    nvtx.range_push("Custom_Fused_AddPostMoeNorm")
    for _ in range(50):
        _ = fused_add_post_moe_norm(routed_out, shared_out, gamma, eps=eps)
    torch.cuda.synchronize()
    nvtx.range_pop()

    cudart.cudaProfilerStop()
    print("[3/3] Profiling completed.")


if __name__ == "__main__":
    main()
