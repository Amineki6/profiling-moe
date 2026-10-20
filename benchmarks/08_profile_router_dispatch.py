#!/usr/bin/env python3
"""Nsight Systems Profiling Script for Candidate 3: Fused Kolibri Router + Dispatch + Activation Expansion."""

import argparse
import os
import sys
import torch
import torch.cuda.nvtx as nvtx

# Ensure environment paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from kernels.cuda.kolibri_router_dispatch import fused_kolibri_router_dispatch


def eager_kolibri_router(scores, bias, topk=6):
    biased_scores = scores + bias
    topk_vals, topk_indices = torch.topk(biased_scores, topk, dim=-1)
    raw_selected = torch.gather(scores, 1, topk_indices)
    weights = torch.sigmoid(raw_selected)
    return weights, topk_indices


def eager_router_dispatch(x, scores, bias, topk=6, num_experts=384):
    num_tokens, hidden_size = x.shape
    weights, indices = eager_kolibri_router(scores, bias, topk=topk)

    flat_experts = indices.t().contiguous().view(-1)
    flat_tokens = torch.arange(num_tokens, device=x.device).repeat(topk)
    flat_ranks = torch.arange(topk, device=x.device).unsqueeze(1).repeat(1, num_tokens).view(-1)
    flat_weights = weights.t().contiguous().view(-1)

    sorted_permuted_indices = torch.argsort(flat_experts, stable=True)

    permuted_tokens = flat_tokens[sorted_permuted_indices]
    permuted_ranks = flat_ranks[sorted_permuted_indices]
    permuted_weights = flat_weights[sorted_permuted_indices]
    permuted_row_to_unpermuted_row = (permuted_ranks * num_tokens + permuted_tokens).to(torch.int32)

    counts = torch.bincount(flat_experts, minlength=num_experts)
    offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=x.device)
    offsets[1:] = torch.cumsum(counts, dim=0)

    permuted_output = x[permuted_tokens]

    return permuted_output, permuted_weights, permuted_row_to_unpermuted_row, offsets


def main():
    parser = argparse.ArgumentParser(description="Profile Fused Router Dispatch vs Eager")
    parser.add_argument("--tokens", type=int, default=64, help="Token count for profiling (default: 64)")
    parser.add_argument("--iters", type=int, default=50, help="Number of profiling iterations (default: 50)")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required for profiling!"
    device = "cuda"
    dtype = torch.bfloat16
    hidden_dim = 2560
    num_experts = 384
    num_tokens = args.tokens
    iters = args.iters

    print(f"Profiling configuration: {num_tokens} tokens, hidden_dim={hidden_dim}, E={num_experts}, dtype={dtype}, iters={iters}")

    x = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
    scores = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    bias = torch.randn(num_experts, device=device, dtype=torch.float32)

    # 1. Warm-up outside profiler range
    print("[1/3] Warming up kernels...")
    for _ in range(25):
        _ = eager_router_dispatch(x, scores, bias, topk=6, num_experts=num_experts)
        _ = fused_kolibri_router_dispatch(x, scores, bias)
    torch.cuda.synchronize()

    # 2. Targeted profile capture
    print("[2/3] Capturing profile with Nsight Systems...")
    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()

    # --- Eager Unfused Baseline ---
    nvtx.range_push("Eager_Unfused_RouterDispatch")
    for _ in range(iters):
        _ = eager_router_dispatch(x, scores, bias, topk=6, num_experts=num_experts)
    torch.cuda.synchronize()
    nvtx.range_pop()

    # --- Custom Fused Kernel Pipeline ---
    nvtx.range_push("Custom_Fused_RouterDispatch")
    for _ in range(iters):
        _ = fused_kolibri_router_dispatch(x, scores, bias)
    torch.cuda.synchronize()
    nvtx.range_pop()

    cudart.cudaProfilerStop()
    print("[3/3] Profiling completed.")


if __name__ == "__main__":
    main()
