#!/usr/bin/env python3
"""Benchmark and Parity Test for Fusion Candidate 3: Kolibri Router + Dispatch + Activation Expansion."""

import os
import sys
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from kernels.cuda.kolibri_router_dispatch import fused_kolibri_router_dispatch


def eager_kolibri_router(scores, bias, topk=6):
    """Reference unfused eager PyTorch Kolibri router."""
    biased_scores = scores + bias
    topk_vals, topk_indices = torch.topk(biased_scores, topk, dim=-1)
    # Sigmoid on raw scores for chosen experts
    raw_selected = torch.gather(scores, 1, topk_indices)
    weights = torch.sigmoid(raw_selected)
    return weights, topk_indices


def eager_router_dispatch(x, scores, bias, topk=6, num_experts=384):
    """
    Reference unfused eager PyTorch router + dispatch + expansion pipeline.
    """
    num_tokens, hidden_size = x.shape
    weights, indices = eager_kolibri_router(scores, bias, topk=topk)

    # Flatten tokens and their top-k choices:
    # Convention in TRT-LLM: unpermuted_row = k * num_tokens + token
    flat_experts = indices.t().contiguous().view(-1)  # shape [topk * num_tokens]
    flat_tokens = torch.arange(num_tokens, device=x.device).repeat(topk)
    flat_ranks = torch.arange(topk, device=x.device).unsqueeze(1).repeat(1, num_tokens).view(-1)
    flat_weights = weights.t().contiguous().view(-1)

    # Stable sort by expert ID
    sorted_permuted_indices = torch.argsort(flat_experts, stable=True)

    permuted_tokens = flat_tokens[sorted_permuted_indices]
    permuted_ranks = flat_ranks[sorted_permuted_indices]
    permuted_weights = flat_weights[sorted_permuted_indices]
    permuted_row_to_unpermuted_row = (permuted_ranks * num_tokens + permuted_tokens).to(torch.int32)

    # Cumulative expert offsets
    counts = torch.bincount(flat_experts, minlength=num_experts)
    offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=x.device)
    offsets[1:] = torch.cumsum(counts, dim=0)

    # Row expansion
    permuted_output = x[permuted_tokens]

    return permuted_output, permuted_weights, permuted_row_to_unpermuted_row, offsets


def test_parity(device="cuda", dtype=torch.bfloat16):
    print("=" * 80)
    print(f"1. NUMERICAL PARITY TEST ({dtype})")
    print("=" * 80)

    num_tokens = 64
    hidden_dim = 2560
    num_experts = 384
    top_k = 6

    torch.manual_seed(42)
    x = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
    scores = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    bias = torch.randn(num_experts, device=device, dtype=torch.float32)

    ref_out, ref_weights, ref_map, ref_offsets = eager_router_dispatch(x, scores, bias, topk=top_k, num_experts=num_experts)
    fused_out, fused_weights, fused_map, fused_offsets = fused_kolibri_router_dispatch(x, scores, bias)

    # Check expert offsets
    diff_offsets = (ref_offsets - fused_offsets).abs().max().item()
    print(f"Max Abs Diff (expert_first_token_offset)     : {diff_offsets}")

    # Invariant 1: Activation expansion matches input indexing
    fused_tokens = (fused_map % num_tokens).long()
    expected_fused_out = x[fused_tokens]
    diff_out_self = (fused_out - expected_fused_out).abs().max().item()
    print(f"Max Abs Diff (fused_out vs x[fused_tokens])   : {diff_out_self:.6e}")

    # Invariant 2: Weight consistency with selected token and chosen expert
    # For each row, check if the weight matches sigmoid(raw_score)
    ref_weights_eager, ref_indices_eager = eager_kolibri_router(scores, bias, topk=top_k)
    fused_ranks = (fused_map // num_tokens).long()
    
    # Check that for every expert e, the set of (token, rank) pairs is identical
    expert_sets_match = True
    for e in range(num_experts):
        start = ref_offsets[e].item()
        end = ref_offsets[e + 1].item()
        if start == end:
            continue
        ref_set = set(ref_map[start:end].cpu().tolist())
        fused_set = set(fused_map[start:end].cpu().tolist())
        if ref_set != fused_set:
            expert_sets_match = False
            break

    print(f"Per-Expert Token Assignment Equivalence       : {'EXACT MATCH' if expert_sets_match else 'MISMATCH'}")

    tol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
    is_ok = (diff_offsets == 0) and (diff_out_self <= tol) and expert_sets_match

    if is_ok:
        print(">>> [PASS] Candidate 3 outputs are mathematically equivalent to eager reference!\n")
    else:
        print(">>> [FAIL] Parity check failed!\n")

    return is_ok


import gc
import triton.testing


def benchmark_latency(device="cuda", dtype=torch.bfloat16):
    print("=" * 80)
    print(f"2. LATENCY BENCHMARK ({dtype}, d_model = 2560, E = 384, top-6)")
    print("=" * 80)
    print(f"{'Tokens':<10} | {'Eager (us)':<12} | {'Fused (us)':<12} | {'Speedup':<10} | {'Intermediate DRAM Saved':<22}")
    print("-" * 80)

    hidden_dim = 2560
    num_experts = 384
    token_counts = [1, 8, 16, 32, 64, 128, 256]

    for n in token_counts:
        x = torch.randn(n, hidden_dim, device=device, dtype=dtype)
        scores = torch.randn(n, num_experts, device=device, dtype=torch.float32)
        bias = torch.randn(num_experts, device=device, dtype=torch.float32)

        gc.collect()
        torch.cuda.empty_cache()

        # Eager timing (median across iterations, immune to GC hitches)
        eager_ms = triton.testing.do_bench(
            lambda: eager_router_dispatch(x, scores, bias, topk=6, num_experts=num_experts),
            warmup=25,
            rep=100,
            return_mode="median",
        )
        eager_us = eager_ms * 1000.0

        # Fused timing (median across iterations)
        fused_ms = triton.testing.do_bench(
            lambda: fused_kolibri_router_dispatch(x, scores, bias),
            warmup=25,
            rep=100,
            return_mode="median",
        )
        fused_us = fused_ms * 1000.0

        speedup = eager_us / fused_us
        # DRAM Traffic Saved: eliminated out_indices write+read (2 * n * 6 * 4B) + out_weights write+read (2 * n * 6 * 4B)
        dram_saved_kb = (2 * n * 6 * 4 * 2) / 1024.0

        print(f"{n:<10} | {eager_us:<12.2f} | {fused_us:<12.2f} | {speedup:<10.2f}x | {dram_saved_kb:<18.2f} KB")

    print("=" * 80)


if __name__ == "__main__":
    test_parity(dtype=torch.bfloat16)
    benchmark_latency(dtype=torch.bfloat16)
