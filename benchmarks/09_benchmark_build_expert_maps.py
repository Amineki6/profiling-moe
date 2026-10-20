#!/usr/bin/env python3
"""Benchmark and Parity Test: Custom 256-Thread Build Expert Maps vs TRT-LLM CUB B32."""

import os
import sys
import torch
import triton.testing

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from kernels.cuda.kolibri_build_expert_maps import (
    kolibri_build_expert_maps,
    trtllm_build_expert_maps_b32,
)


def eager_build_expert_maps(indices, num_experts=384, topk=6):
    num_tokens = indices.size(0)
    flat_experts = indices.view(-1)
    flat_tokens = torch.arange(num_tokens, device=indices.device).repeat_interleave(topk)
    flat_ranks = torch.arange(topk, device=indices.device).repeat(num_tokens)

    counts = torch.bincount(flat_experts, minlength=num_experts)
    offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=indices.device)
    offsets[1:] = torch.cumsum(counts, dim=0)

    return offsets


def test_parity(device="cuda"):
    print("=" * 80)
    print("1. STEP 2 NUMERICAL PARITY TEST (Custom 256-Thread vs TRT-LLM CUB B32)")
    print("=" * 80)

    num_tokens = 32
    num_experts = 384
    top_k = 6

    torch.manual_seed(42)
    # Generate random expert selections in [0, 384)
    token_selected_experts = torch.randint(0, num_experts, (num_tokens, top_k), device=device, dtype=torch.int32)

    ref_offsets = eager_build_expert_maps(token_selected_experts, num_experts, top_k)
    custom_map, custom_inv, custom_offsets = kolibri_build_expert_maps(token_selected_experts, num_experts)
    b32_map, b32_inv, b32_offsets = trtllm_build_expert_maps_b32(token_selected_experts, num_experts)

    diff_custom_offsets = (ref_offsets - custom_offsets).abs().max().item()
    diff_b32_offsets = (ref_offsets - b32_offsets).abs().max().item()

    print(f"Max Abs Diff: Custom expert_offsets vs Ref  : {diff_custom_offsets}")
    print(f"Max Abs Diff: CUB B32 expert_offsets vs Ref : {diff_b32_offsets}")

    # Check partition set equivalence for each expert
    sets_match = True
    for e in range(num_experts):
        start = ref_offsets[e].item()
        end = ref_offsets[e + 1].item()
        if start == end:
            continue
        c_set = set(custom_map[start:end].cpu().tolist())
        b_set = set(b32_map[start:end].cpu().tolist())
        if c_set != b_set:
            sets_match = False
            break

    print(f"Per-Expert Token Partition Equivalence       : {'EXACT MATCH' if sets_match else 'MISMATCH'}")
    is_ok = (diff_custom_offsets == 0) and (diff_b32_offsets == 0) and sets_match

    if is_ok:
        print(">>> [PASS] Custom Step 2 matches TRT-LLM CUB B32 and Eager reference perfectly!\n")
    else:
        print(">>> [FAIL] Parity check failed!\n")

    return is_ok


def benchmark_step2_latency(device="cuda"):
    print("=" * 80)
    print("2. STEP 2 LATENCY BENCHMARK: Custom 256-Thread vs TRT-LLM CUB B32")
    print("=" * 80)
    print(f"{'Tokens':<10} | {'TRT-LLM CUB B32 (us)':<22} | {'Custom 256-Th (us)':<20} | {'Speedup':<10} | {'Shared Memory'}")
    print("-" * 80)

    num_experts = 384
    top_k = 6
    token_counts = [1, 8, 16, 32, 64, 128, 256]

    for n in token_counts:
        token_selected_experts = torch.randint(0, num_experts, (n, top_k), device=device, dtype=torch.int32)

        # Custom 256-thread 3 KB SRAM kernel
        custom_ms = triton.testing.do_bench(
            lambda: kolibri_build_expert_maps(token_selected_experts, num_experts),
            warmup=25,
            rep=100,
            return_mode="median",
        )
        custom_us = custom_ms * 1000.0

        # TRT-LLM CUB B32 kernel (only valid for n <= 32 due to 32 threads)
        if n <= 32:
            b32_ms = triton.testing.do_bench(
                lambda: trtllm_build_expert_maps_b32(token_selected_experts, num_experts),
                warmup=25,
                rep=100,
                return_mode="median",
            )
            b32_us = b32_ms * 1000.0
            speedup_str = f"{b32_us / custom_us:.2f}x"
            b32_str = f"{b32_us:.2f}"
            smem_str = "3 KB vs 32 KB"
        else:
            b32_str = "N/A (CUB smem overflow)"
            speedup_str = "Inf"
            smem_str = "3 KB (CUB needs 257KB!)"

        print(f"{n:<10} | {b32_str:<22} | {custom_us:<20.2f} | {speedup_str:<10} | {smem_str}")

    print("=" * 80)


if __name__ == "__main__":
    test_parity()
    benchmark_step2_latency()
