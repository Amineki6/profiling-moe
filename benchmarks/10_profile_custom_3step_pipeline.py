#!/usr/bin/env python3
"""Nsight Systems Profiling Script for Full 3-Step Custom Kolibri MoE Pipeline.

Pipeline:
  Step 1: Native sm_89 Multi-Block CUDA Router (kolibri_gate_forward_kernel)
  Step 2: Custom 256-Thread 3 KB SRAM Dispatch Mapping (kolibri_build_expert_maps_kernel)
  Step 3: Multi-Block 128-bit Vectorized Activation Expansion (kolibri_activation_expansion_from_map_kernel)
"""

import argparse
import os
import sys
import torch
import torch.cuda.nvtx as nvtx

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from kernels.kolibri_cuda_router import kolibri_cuda_ops
from kernels.cuda.kolibri_build_expert_maps import (
    kolibri_build_expert_maps,
    kolibri_activation_expansion,
)


def run_pipeline(x, scores, bias, num_experts=384, topk=6):
    # Step 1: Router
    out_indices, out_weights = kolibri_cuda_ops.forward_vB(scores, bias, topk)

    # Step 2: Dispatch Mapping
    permuted_row_to_unpermuted, unpermuted_to_permuted, offsets = kolibri_build_expert_maps(out_indices, num_experts)

    # Step 3: Activation Row Expansion
    permuted_output = kolibri_activation_expansion(x, permuted_row_to_unpermuted)

    return permuted_output, out_weights, permuted_row_to_unpermuted, offsets


def main():
    parser = argparse.ArgumentParser(description="Profile Full 3-Step Custom Kolibri Pipeline")
    parser.add_argument("--tokens", type=int, default=16, help="Token count (default: 16)")
    parser.add_argument("--iters", type=int, default=100, help="Profiling iterations (default: 100)")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required for profiling!"
    device = "cuda"
    dtype = torch.bfloat16
    hidden_dim = 2560
    num_experts = 384
    topk = 6
    num_tokens = args.tokens
    iters = args.iters

    print(f"Profiling 3-Step Pipeline: tokens={num_tokens}, hidden_dim={hidden_dim}, E={num_experts}, iters={iters}")

    torch.manual_seed(42)
    x = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
    scores = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    bias = torch.randn(num_experts, device=device, dtype=torch.float32)

    # 1. Warm-up
    print("[1/3] Warming up pipeline...")
    for _ in range(50):
        _ = run_pipeline(x, scores, bias, num_experts, topk)
    torch.cuda.synchronize()

    # 2. Targeted profile capture
    print("[2/3] Capturing profile with Nsight Systems...")
    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()

    nvtx.range_push(f"Full_3Step_Pipeline_N{num_tokens}")
    for _ in range(iters):
        _ = run_pipeline(x, scores, bias, num_experts, topk)
    torch.cuda.synchronize()
    nvtx.range_pop()

    cudart.cudaProfilerStop()
    print("[3/3] Profiling completed successfully.")


if __name__ == "__main__":
    main()
