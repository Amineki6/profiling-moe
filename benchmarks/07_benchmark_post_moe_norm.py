#!/usr/bin/env python3
"""Benchmark and Parity Test for Fusion Candidate 2: Routed + Shared Add + Post-MoE Norm."""

import os
import sys
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from kernels.cuda.kolibri_post_moe_norm import fused_add_post_moe_norm


def eager_add_post_moe_norm(routed_out, shared_out, gamma, eps=1e-5):
    """Reference unfused eager PyTorch implementation."""
    d = routed_out.shape[-1]
    combined = routed_out + shared_out
    moe_out = torch.nn.functional.rms_norm(combined, (d,), gamma, eps=eps)
    return moe_out


def test_parity(device="cuda", dtype=torch.bfloat16):
    print("=" * 70)
    print(f"1. NUMERICAL PARITY TEST ({dtype})")
    print("=" * 70)

    num_tokens = 64
    hidden_dim = 2560
    eps = 1e-5

    torch.manual_seed(42)
    routed_out = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
    shared_out = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
    gamma = torch.ones(hidden_dim, device=device, dtype=dtype) + 0.1 * torch.randn(hidden_dim, device=device, dtype=dtype)

    ref_out = eager_add_post_moe_norm(routed_out, shared_out, gamma, eps=eps)
    fused_out = fused_add_post_moe_norm(routed_out, shared_out, gamma, eps=eps)

    diff = (ref_out - fused_out).abs().max().item()
    tol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4

    print(f"Max Abs Diff (moe_out) : {diff:.6e} (tolerance: {tol})")
    is_ok = torch.allclose(ref_out, fused_out, atol=tol, rtol=tol)

    if is_ok:
        print(">>> [PASS] Candidate 2 outputs match PyTorch eager reference perfectly!\n")
    else:
        print(">>> [FAIL] Parity check failed!\n")
        assert False, "Parity check failed"


def benchmark_latency(device="cuda", dtype=torch.bfloat16):
    print("=" * 70)
    print(f"2. LATENCY BENCHMARK ({dtype}, d_model = 2560)")
    print("=" * 70)
    print(f"{'Tokens':<10} | {'Eager (us)':<12} | {'Fused (us)':<12} | {'Speedup':<10} | {'DRAM Traffic Saved':<18}")
    print("-" * 70)

    hidden_dim = 2560
    eps = 1e-5
    token_counts = [1, 16, 64, 128, 512, 1024]

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    warmup = 50
    iters = 300
    elem_bytes = 2 if dtype in (torch.float16, torch.bfloat16) else 4

    for n in token_counts:
        routed_out = torch.randn(n, hidden_dim, device=device, dtype=dtype)
        shared_out = torch.randn(n, hidden_dim, device=device, dtype=dtype)
        gamma = torch.ones(hidden_dim, device=device, dtype=dtype)

        # Eager timing
        for _ in range(warmup):
            _ = eager_add_post_moe_norm(routed_out, shared_out, gamma, eps=eps)
        torch.cuda.synchronize()

        start_event.record()
        for _ in range(iters):
            _ = eager_add_post_moe_norm(routed_out, shared_out, gamma, eps=eps)
        end_event.record()
        torch.cuda.synchronize()
        eager_us = (start_event.elapsed_time(end_event) / iters) * 1000.0

        # Fused timing
        for _ in range(warmup):
            _ = fused_add_post_moe_norm(routed_out, shared_out, gamma, eps=eps)
        torch.cuda.synchronize()

        start_event.record()
        for _ in range(iters):
            _ = fused_add_post_moe_norm(routed_out, shared_out, gamma, eps=eps)
        end_event.record()
        torch.cuda.synchronize()
        fused_us = (start_event.elapsed_time(end_event) / iters) * 1000.0

        speedup = eager_us / fused_us
        # DRAM Traffic Saved: eliminated moe_combined write + read = 2 * n * hidden_dim * elem_bytes
        bytes_saved = 2 * n * hidden_dim * elem_bytes
        saved_str = f"{bytes_saved / 1024:.1f} KB / step"

        print(f"{n:<10} | {eager_us:<12.2f} | {fused_us:<12.2f} | {speedup:<10.2f}x | {saved_str:<18}")

    print("=" * 70)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available!")
        sys.exit(1)

    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    test_parity(dtype=torch.float16)
    benchmark_latency(dtype=torch.float16)

    test_parity(dtype=torch.bfloat16)
    benchmark_latency(dtype=torch.bfloat16)
