#!/usr/bin/env python3
"""Nsight Systems Profiling Script: Custom 256-Thread Build Expert Maps vs TRT-LLM CUB B32 (N = 16)."""

import argparse
import os
import sys
import torch
import torch.cuda.nvtx as nvtx

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from kernels.cuda.kolibri_build_expert_maps import (
    kolibri_build_expert_maps,
    trtllm_build_expert_maps_b32,
)


def main():
    parser = argparse.ArgumentParser(description="Profile Custom Step 2 vs TRT-LLM CUB B32")
    parser.add_argument("--tokens", type=int, default=16, help="Token count for profiling (default: 16)")
    parser.add_argument("--iters", type=int, default=100, help="Number of profiling iterations (default: 100)")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required for profiling!"
    device = "cuda"
    num_experts = 384
    top_k = 6
    num_tokens = args.tokens
    iters = args.iters

    print(f"Profiling Step 2: tokens={num_tokens}, E={num_experts}, top_k={top_k}, iters={iters}")

    torch.manual_seed(42)
    token_selected_experts = torch.randint(0, num_experts, (num_tokens, top_k), device=device, dtype=torch.int32)

    # 1. Warm-up
    print("[1/3] Warming up kernels...")
    for _ in range(50):
        _ = trtllm_build_expert_maps_b32(token_selected_experts, num_experts)
        _ = kolibri_build_expert_maps(token_selected_experts, num_experts)
    torch.cuda.synchronize()

    # 2. Targeted profile capture
    print("[2/3] Capturing profile with Nsight Systems...")
    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()

    # --- Baseline: TRT-LLM CUB B32 ---
    nvtx.range_push("TRTLLM_CUB_B32_Step2")
    for _ in range(iters):
        _ = trtllm_build_expert_maps_b32(token_selected_experts, num_experts)
    torch.cuda.synchronize()
    nvtx.range_pop()

    # --- Custom: 256-Thread 3 KB SRAM Kernel ---
    nvtx.range_push("Custom_256Th_3KBSmem_Step2")
    for _ in range(iters):
        _ = kolibri_build_expert_maps(token_selected_experts, num_experts)
    torch.cuda.synchronize()
    nvtx.range_pop()

    cudart.cudaProfilerStop()
    print("[3/3] Profiling completed successfully.")


if __name__ == "__main__":
    main()
