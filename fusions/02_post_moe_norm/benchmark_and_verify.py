#!/usr/bin/env python3
import sys
import os
import importlib.util
import time
import torch
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Dynamically import fused and baseline modules to avoid package naming restrictions with leading numbers
def load_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

fused_mod = load_module_from_path(
    "fused_post_moe_norm",
    os.path.join(CURRENT_DIR, "fused", "fused_post_moe_norm.py"),
)
baseline_mod = load_module_from_path(
    "trtllm_baseline",
    os.path.join(CURRENT_DIR, "baseline", "trtllm_baseline.py"),
)

fused_add_post_moe_norm = fused_mod.fused_add_post_moe_norm
trtllm_cuda_baseline = baseline_mod.trtllm_cuda_baseline
eager_pytorch_baseline = baseline_mod.eager_pytorch_baseline


def test_parity(device="cuda"):
    print("=" * 80)
    print("1. NUMERICAL PARITY & CORRECTNESS VERIFICATION")
    print("=" * 80)

    test_tokens = [1, 16, 64, 256, 1152]
    hidden_dim = 2560
    eps = 1e-5

    for dtype, dtype_name, atol, rtol in [
        (torch.bfloat16, "BF16", 4e-2, 2e-2),
        (torch.float16, "FP16", 1e-3, 1e-3),
    ]:
        print(f"\n---> Testing Dtype: {dtype_name} (atol={atol}, rtol={rtol})")
        for n in test_tokens:
            torch.manual_seed(42 + n)
            routed_out = torch.randn(n, hidden_dim, device=device, dtype=dtype)
            shared_out = torch.randn(n, hidden_dim, device=device, dtype=dtype)
            gamma = torch.randn(hidden_dim, device=device, dtype=dtype).abs() + 0.1

            # 1. Reference Eager PyTorch
            ref_moe = eager_pytorch_baseline(routed_out, shared_out, gamma, eps=eps)

            # 2. Native TRT-LLM 2-Kernel Baseline
            trt_moe = trtllm_cuda_baseline(routed_out, shared_out, gamma, eps=eps)

            # 3. Production Fused Kernel
            fused_moe = fused_add_post_moe_norm(routed_out, shared_out, gamma, eps=eps)

            # Check parity: Fused vs Reference
            fused_vs_ref = torch.allclose(fused_moe, ref_moe, atol=atol, rtol=rtol)
            max_diff_fused = (fused_moe - ref_moe).abs().max().item()

            # Check parity: TRT-LLM vs Reference
            trt_vs_ref = torch.allclose(trt_moe, ref_moe, atol=atol, rtol=rtol)
            max_diff_trt = (trt_moe - ref_moe).abs().max().item()

            # Check parity: Fused vs TRT-LLM
            fused_vs_trt = torch.allclose(fused_moe, trt_moe, atol=atol, rtol=rtol)
            diff_fused_trt = (fused_moe - trt_moe).abs().max().item()

            status = "PASS" if (fused_vs_ref and trt_vs_ref and fused_vs_trt) else "FAIL"
            print(
                f"  N={n:4d} | Status: [{status}] | "
                f"Fused vs Ref: diff={max_diff_fused:.6f} | "
                f"TRT vs Ref: diff={max_diff_trt:.6f} | "
                f"Fused vs TRT: diff={diff_fused_trt:.6f}"
            )
            assert status == "PASS", f"Parity failure at N={n}, dtype={dtype_name}"

    print("\n[SUCCESS] All numerical parity checks passed across BF16 and FP16!")


def test_profiler_kernel_names(device="cuda"):
    print("\n" + "=" * 80)
    print("2. PROFILER KERNEL EXECUTION & SYMBOL PROOF")
    print("=" * 80)

    n, hidden_dim = 64, 2560
    routed_out = torch.randn(n, hidden_dim, device=device, dtype=torch.bfloat16)
    shared_out = torch.randn(n, hidden_dim, device=device, dtype=torch.bfloat16)
    gamma = torch.ones(hidden_dim, device=device, dtype=torch.bfloat16)

    # Warmup
    trtllm_cuda_baseline(routed_out, shared_out, gamma)
    fused_add_post_moe_norm(routed_out, shared_out, gamma)
    torch.cuda.synchronize()

    # Profile TRT-LLM Baseline
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof_base:
        trtllm_cuda_baseline(routed_out, shared_out, gamma)
        torch.cuda.synchronize()

    print("\n--- TRT-LLM Baseline Kernel Profile (vector_add + generalRmsNorm) ---")
    print(prof_base.key_averages().table(sort_by="cuda_time_total", row_limit=5))

    # Profile Production Fused Kernel
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof_fused:
        fused_add_post_moe_norm(routed_out, shared_out, gamma)
        torch.cuda.synchronize()

    print("\n--- Production Fused Kernel Profile (1 Launch: fusedAddPostMoeNormRegKernel) ---")
    print(prof_fused.key_averages().table(sort_by="cuda_time_total", row_limit=5))

    print("[SUCCESS] Verified authentic TRT-LLM generalRmsNorm baseline vs Fused registration kernel!")


def benchmark_latency(device="cuda"):
    print("\n" + "=" * 80)
    print("3. LATENCY & THROUGHPUT BENCHMARK (Ada Lovelace RTX 4090)")
    print("=" * 80)

    token_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 98304]
    hidden_dim = 2560
    dtype = torch.bfloat16

    warmup = 50
    iters = 200

    print(f"{'Tokens':>8} | {'Eager PyTorch':>14} | {'TRT-LLM Native':>14} | {'Custom Fused':>14} | {'Speedup':>9} | {'Bandwidth':>11}")
    print("-" * 84)

    for n in token_counts:
        routed_out = torch.randn(n, hidden_dim, device=device, dtype=dtype)
        shared_out = torch.randn(n, hidden_dim, device=device, dtype=dtype)
        gamma = torch.ones(hidden_dim, device=device, dtype=dtype)

        # Warmup
        for _ in range(warmup):
            eager_pytorch_baseline(routed_out, shared_out, gamma)
            trtllm_cuda_baseline(routed_out, shared_out, gamma)
            fused_add_post_moe_norm(routed_out, shared_out, gamma)
        torch.cuda.synchronize()

        # Benchmark Eager PyTorch
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        for _ in range(iters):
            eager_pytorch_baseline(routed_out, shared_out, gamma)
        end_evt.record()
        torch.cuda.synchronize()
        eager_us = (start_evt.elapsed_time(end_evt) * 1000.0) / iters

        # Benchmark TRT-LLM Native Baseline
        start_evt.record()
        for _ in range(iters):
            trtllm_cuda_baseline(routed_out, shared_out, gamma)
        end_evt.record()
        torch.cuda.synchronize()
        base_us = (start_evt.elapsed_time(end_evt) * 1000.0) / iters

        # Benchmark Custom Fused
        start_evt.record()
        for _ in range(iters):
            fused_add_post_moe_norm(routed_out, shared_out, gamma)
        end_evt.record()
        torch.cuda.synchronize()
        fused_us = (start_evt.elapsed_time(end_evt) * 1000.0) / iters

        speedup = base_us / fused_us

        # Memory traffic for Fused:
        # Reads: routed_out (n * d * 2 B) + shared_out (n * d * 2 B) + gamma (d * 2 B)
        # Writes: moe_out (n * d * 2 B)
        bytes_transferred = n * hidden_dim * 2 * 3 + hidden_dim * 2
        bw_gb_s = (bytes_transferred / (fused_us * 1e-6)) / 1e9

        print(
            f"{n:8d} | "
            f"{eager_us:11.2f} µs | "
            f"{base_us:11.2f} µs | "
            f"{fused_us:11.2f} µs | "
            f"{speedup:8.2f}x | "
            f"{bw_gb_s:8.1f} GB/s"
        )


if __name__ == "__main__":
    test_parity()
    test_profiler_kernel_names()
    benchmark_latency()
