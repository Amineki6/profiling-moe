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
from kernels.cuda.multi_block_expert_maps.kolibri_multi_block_build_expert_maps import (
    kolibri_multi_block_build_expert_maps,
)


def main():
    parser = argparse.ArgumentParser(description="Profile Kolibri Build Expert Maps CUDA Kernel")
    parser.add_argument("--tokens", type=int, default=98304, help="Token count (default: 98304, 16 waves router equivalent)")
    parser.add_argument("--iterations", type=int, default=1, help="Profiling iterations (default: 1 for NCU)")
    parser.add_argument("--variant", type=str, choices=["custom", "trtllm", "multiblock", "all"], default="multiblock", help="Variant to profile (default: multiblock)")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warm-up iterations (recommended for NCU)")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required for profiling!"
    device = "cuda"
    num_experts = 384
    top_k = 6
    num_tokens = args.tokens
    iters = args.iterations
    variant = args.variant

    print(f"Profiling Build Expert Maps: tokens={num_tokens}, E={num_experts}, top_k={top_k}, iters={iters}, variant={variant}")

    torch.manual_seed(42)
    token_selected_experts = torch.randint(0, num_experts, (num_tokens, top_k), device=device, dtype=torch.int32)

    # 1. Warm-up (skipped when iters == 1 or --no-warmup to keep NCU fast and lightweight)
    warmup_iters = 0 if (args.no_warmup or iters == 1) else 10
    if warmup_iters > 0:
        print(f"[1/3] Warming up kernels ({warmup_iters} iters)...")
        for _ in range(warmup_iters):
            if variant in ["all", "trtllm"]:
                _ = trtllm_build_expert_maps_b32(token_selected_experts, num_experts)
            if variant in ["all", "custom"]:
                _ = kolibri_build_expert_maps(token_selected_experts, num_experts)
            if variant in ["all", "multiblock"]:
                _ = kolibri_multi_block_build_expert_maps(token_selected_experts, num_experts)
        torch.cuda.synchronize()
    else:
        print("[1/3] Warm-up skipped for profiling.")

    # 2. Targeted profile capture
    print("[2/3] Profiling under NVTX ranges...")
    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()

    # --- Baseline: TRT-LLM CUB B32 ---
    if variant in ["all", "trtllm"]:
        nvtx.range_push("TRTLLM_CUB_B32_Step2")
        for _ in range(iters):
            _ = trtllm_build_expert_maps_b32(token_selected_experts, num_experts)
        torch.cuda.synchronize()
        nvtx.range_pop()

    # --- Custom: 256-Thread 3 KB SRAM Kernel (Single Block) ---
    if variant in ["all", "custom"]:
        nvtx.range_push("Kolibri Build Expert Maps: Custom (256-thread 3KB Smem)")
        for _ in range(iters):
            _ = kolibri_build_expert_maps(token_selected_experts, num_experts)
        torch.cuda.synchronize()
        nvtx.range_pop()

    # --- Multi-Block: 3-Stage Pipeline (128 SMs) ---
    if variant in ["all", "multiblock"]:
        nvtx.range_push("Kolibri Build Expert Maps: Multi-Block 3-Stage")
        for _ in range(iters):
            _ = kolibri_multi_block_build_expert_maps(token_selected_experts, num_experts)
        torch.cuda.synchronize()
        nvtx.range_pop()

    cudart.cudaProfilerStop()
    print("[3/3] Profiling completed successfully.")


if __name__ == "__main__":
    main()
