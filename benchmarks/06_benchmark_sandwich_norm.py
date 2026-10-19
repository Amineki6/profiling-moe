#!/usr/bin/env python3
"""Benchmark and Parity Test for Custom Kolibri Fused Sandwich Norm Kernel.

Compares:
1. Eager Unfused PyTorch:
     post_attn = RMSNorm(attn_out, gamma1)
     updated_residual = residual_in + post_attn
     moe_in = RMSNorm(updated_residual, gamma2)
2. Custom Fused CUDA Kernel (customKolibriSandwichNormKernels):
     Single pass with on-chip shared memory caching & in-register residual accumulation.
"""

import os
import sys
import torch

# Add repository root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from kernels.cuda.kolibri_sandwich_norm import fused_sandwich_norm


def eager_sandwich_norm(attn_out, residual_in, gamma1, gamma2, eps=1e-5):
    """Reference unfused eager PyTorch implementation."""
    d = attn_out.shape[-1]
    post_attn = torch.nn.functional.rms_norm(attn_out, (d,), gamma1, eps=eps)
    updated_residual = residual_in + post_attn
    moe_in = torch.nn.functional.rms_norm(updated_residual, (d,), gamma2, eps=eps)
    return updated_residual, moe_in


def test_parity(device="cuda", dtype=torch.float16):
    print("=" * 70)
    print(f"1. NUMERICAL PARITY TEST ({dtype})")
    print("=" * 70)

    num_tokens = 64
    hidden_dim = 2560
    eps = 1e-5

    torch.manual_seed(42)
    attn_out = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
    residual_in = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
    gamma1 = torch.ones(hidden_dim, device=device, dtype=dtype) + 0.1 * torch.randn(hidden_dim, device=device, dtype=dtype)
    gamma2 = torch.ones(hidden_dim, device=device, dtype=dtype) + 0.1 * torch.randn(hidden_dim, device=device, dtype=dtype)

    # Reference Eager output
    ref_res, ref_moe = eager_sandwich_norm(attn_out, residual_in, gamma1, gamma2, eps=eps)

    # Fused CUDA output
    fused_res, fused_moe = fused_sandwich_norm(attn_out, residual_in, gamma1, gamma2, eps=eps)

    # Compute differences
    res_diff = (ref_res - fused_res).abs().max().item()
    moe_diff = (ref_moe - fused_moe).abs().max().item()

    tol = 1e-2 if dtype == torch.float16 or dtype == torch.bfloat16 else 1e-4

    print(f"Max Abs Diff (updated_residual) : {res_diff:.6e} (tolerance: {tol})")
    print(f"Max Abs Diff (moe_in)           : {moe_diff:.6e} (tolerance: {tol})")

    res_ok = torch.allclose(ref_res, fused_res, atol=tol, rtol=tol)
    moe_ok = torch.allclose(ref_moe, fused_moe, atol=tol, rtol=tol)

    if res_ok and moe_ok:
        print(">>> [PASS] Custom Kernel outputs match PyTorch eager reference perfectly!\n")
    else:
        print(">>> [FAIL] Parity check failed!\n")
        assert False, "Parity check failed"


def benchmark_latency(device="cuda", dtype=torch.float16):
    print("=" * 70)
    print(f"2. LATENCY BENCHMARK ({dtype}, d_model = 2560)")
    print("=" * 70)
    print(f"{'Tokens':<10} | {'Eager (us)':<12} | {'Fused (us)':<12} | {'Speedup':<10} | {'DRAM Traffic Saved':<18}")
    print("-" * 70)

    hidden_dim = 2560
    eps = 1e-5
    token_counts = [1, 16, 64, 128, 512, 1024, 2048]

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    warmup = 50
    iters = 300

    elem_bytes = 2 if dtype in (torch.float16, torch.bfloat16) else 4

    for n in token_counts:
        attn_out = torch.randn(n, hidden_dim, device=device, dtype=dtype)
        residual_in = torch.randn(n, hidden_dim, device=device, dtype=dtype)
        gamma1 = torch.ones(hidden_dim, device=device, dtype=dtype)
        gamma2 = torch.ones(hidden_dim, device=device, dtype=dtype)

        # ---------------- Eager Timing ----------------
        for _ in range(warmup):
            _ = eager_sandwich_norm(attn_out, residual_in, gamma1, gamma2, eps=eps)
        torch.cuda.synchronize()

        start_event.record()
        for _ in range(iters):
            _ = eager_sandwich_norm(attn_out, residual_in, gamma1, gamma2, eps=eps)
        end_event.record()
        torch.cuda.synchronize()
        eager_us = (start_event.elapsed_time(end_event) / iters) * 1000.0

        # ---------------- Fused Timing ----------------
        for _ in range(warmup):
            _ = fused_sandwich_norm(attn_out, residual_in, gamma1, gamma2, eps=eps)
        torch.cuda.synchronize()

        start_event.record()
        for _ in range(iters):
            _ = fused_sandwich_norm(attn_out, residual_in, gamma1, gamma2, eps=eps)
        end_event.record()
        torch.cuda.synchronize()
        fused_us = (start_event.elapsed_time(end_event) / iters) * 1000.0

        speedup = eager_us / fused_us
        # DRAM Traffic Saved: eliminated post_attn write + read = 2 * n * hidden_dim * elem_bytes
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

    # Also test BF16
    test_parity(dtype=torch.bfloat16)
    benchmark_latency(dtype=torch.bfloat16)
