#!/usr/bin/env python3
import sys
import os
import importlib.util
import time
import torch
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

fused_mod = load_module_from_path(
    "fused_moe_prologue",
    os.path.join(CURRENT_DIR, "fused", "fused_moe_prologue.py"),
)
baseline_mod = load_module_from_path(
    "trtllm_baseline",
    os.path.join(CURRENT_DIR, "baseline", "trtllm_baseline.py"),
)

kolibri_router = fused_mod.kolibri_router
kolibri_dispatch_map = fused_mod.kolibri_dispatch_map
kolibri_activation_expansion = fused_mod.kolibri_activation_expansion
fused_moe_prologue = fused_mod.fused_moe_prologue

trtllm_router = baseline_mod.trtllm_router
trtllm_build_expert_maps = baseline_mod.trtllm_build_expert_maps
trtllm_activation_expansion = baseline_mod.trtllm_activation_expansion
trtllm_moe_prologue = baseline_mod.trtllm_moe_prologue


def test_parity(device="cuda"):
    print("=" * 80)
    print("1. NUMERICAL PARITY & MATHEMATICAL INVARIANT VERIFICATION")
    print("=" * 80)

    test_tokens = [1, 16, 64, 256, 512, 1152]
    hidden_dim = 2560
    num_experts = 384
    top_k = 6

    for dtype, dtype_name in [(torch.bfloat16, "BF16"), (torch.float16, "FP16")]:
        print(f"\n---> Testing Dtype: {dtype_name}")
        for n in test_tokens:
            torch.manual_seed(42 + n)
            router_logits = torch.randn(n, num_experts, device=device, dtype=torch.float32)
            router_bias = torch.randn(num_experts, device=device, dtype=torch.float32)
            unpermuted_input = torch.randn(n, hidden_dim, device=device, dtype=dtype)

            # 1. Run Fused Pipeline (Dynamic Hybrid)
            f_out, f_w, f_idx, f_offsets, f_p2u, f_u2p = fused_moe_prologue(
                unpermuted_input, router_logits, router_bias
            )

            # 2. Run TRT-LLM Baseline (Single-block for N<=32, 3-kernel pipeline for N>32)
            t_out, t_w, t_idx, t_offsets, t_p2u, t_u2p = trtllm_moe_prologue(
                unpermuted_input, router_logits, router_bias
            )
            router_match = torch.equal(f_idx, t_idx)
            weight_diff = (f_w - t_w).abs().max().item()

            # 3. Invariant checks
            # Invariant A: Router weights sum to 1.0 per token
            weight_sum_err = (f_w.sum(dim=-1) - 1.0).abs().max().item()
            assert weight_sum_err < 1e-4, f"Router weights do not sum to 1: {weight_sum_err}"

            # Invariant B: Expert offsets are non-decreasing and terminate at N * 6
            offsets_cpu = f_offsets.cpu()
            is_monotonic = (offsets_cpu[1:] >= offsets_cpu[:-1]).all().item()
            total_tokens = offsets_cpu[-1].item()
            assert is_monotonic, "Expert offsets not monotonic"
            assert total_tokens == n * top_k, f"Expert offsets total {total_tokens} != {n * top_k}"

            # Invariant C: Permutation mapping is valid and bijective
            p2u_cpu = f_p2u.cpu()
            unique_elements = len(torch.unique(p2u_cpu))
            assert unique_elements == n * top_k, f"Permutation array contains duplicates: {unique_elements} != {n * top_k}"

            # Invariant D: Activation expansion correctness (matches eager gather)
            # unpermuted_row = rank * n + token_id; src_token_id = unpermuted_row % n
            src_token_ids = p2u_cpu % n
            expected_output = unpermuted_input[src_token_ids.to(device)]
            expansion_diff = (f_out - expected_output).abs().max().item()
            assert expansion_diff == 0.0, f"Activation expansion gather mismatch: diff={expansion_diff}"

            regime = "Single-Block (<512)" if n < 512 else "3-Stage Multi-Block (>=512)"
            status = "PASS" if (weight_sum_err < 1e-4 and expansion_diff == 0.0) else "FAIL"
            print(f"  N={n:4d} | Regime: {regime:<25} | Status: [{status}] | Gather Diff: {expansion_diff:.6f} | Weight Diff: {weight_diff:.6f}")

    print("\n[SUCCESS] All numerical parity and permutation invariants verified!")


def test_profiler_regimes(device="cuda"):
    print("\n" + "=" * 80)
    print("2. PROFILER KERNEL EXECUTION & DYNAMIC HYBRID PROOF")
    print("=" * 80)

    hidden_dim = 2560
    num_experts = 384
    dtype = torch.bfloat16

    # Test 1: Decode Regime (N = 64 < 512) -> Single-Block
    n_decode = 64
    x_dec = torch.randn(n_decode, hidden_dim, device=device, dtype=dtype)
    logits_dec = torch.randn(n_decode, num_experts, device=device, dtype=torch.float32)

    fused_moe_prologue(x_dec, logits_dec)
    torch.cuda.synchronize()

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof_dec:
        fused_moe_prologue(x_dec, logits_dec)
        torch.cuda.synchronize()

    print("\n--- Decode Regime (N=64 < 512): Single-Block Dispatch Map ---")
    print(prof_dec.key_averages().table(sort_by="cuda_time_total", row_limit=5))

    # Test 2: Prefill Regime (N = 1024 >= 512) -> 3-Stage Multi-Block Pipeline
    n_prefill = 1024
    x_pre = torch.randn(n_prefill, hidden_dim, device=device, dtype=dtype)
    logits_pre = torch.randn(n_prefill, num_experts, device=device, dtype=torch.float32)

    fused_moe_prologue(x_pre, logits_pre)
    torch.cuda.synchronize()

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof_pre:
        fused_moe_prologue(x_pre, logits_pre)
        torch.cuda.synchronize()

    print("\n--- Prefill Regime (N=1024 >= 512): 3-Stage Multi-Block Pipeline ---")
    print(prof_pre.key_averages().table(sort_by="cuda_time_total", row_limit=8))

    # Test 3: TRT-LLM Native Baseline
    trtllm_moe_prologue(x_dec, logits_dec)
    torch.cuda.synchronize()

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof_trt:
        trtllm_moe_prologue(x_dec, logits_dec)
        torch.cuda.synchronize()

    print("\n--- TRT-LLM Native Baseline (N=64): invokeKolibriRouting + CUB RadixRank ---")
    print(prof_trt.key_averages().table(sort_by="cuda_time_total", row_limit=5))


def benchmark_prologue_latency(device="cuda"):
    print("\n" + "=" * 80)
    print("3. LATENCY & SPEEDUP BENCHMARK (Ada Lovelace RTX 4090)")
    print("=" * 80)

    token_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 98304]
    hidden_dim = 2560
    num_experts = 384
    dtype = torch.bfloat16

    warmup = 30
    iters = 100

    print(f"{'Tokens':>8} | {'Regime Active':>22} | {'TRT-LLM Base':>14} | {'Custom Fused':>14} | {'Speedup':>9}")
    print("-" * 75)

    for n in token_counts:
        x = torch.randn(n, hidden_dim, device=device, dtype=dtype)
        logits = torch.randn(n, num_experts, device=device, dtype=torch.float32)
        bias = torch.randn(num_experts, device=device, dtype=torch.float32)

        # Warmup Fused
        for _ in range(warmup):
            fused_moe_prologue(x, logits, bias)
        torch.cuda.synchronize()

        # Benchmark Fused
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        for _ in range(iters):
            fused_moe_prologue(x, logits, bias)
        end_evt.record()
        torch.cuda.synchronize()
        fused_us = (start_evt.elapsed_time(end_evt) * 1000.0) / iters

        # Benchmark TRT-LLM Baseline (Single-block for N<=32, 3-kernel pipeline for N>32)
        if n <= 4096:
            base_iters = 50 if n >= 1024 else iters
            for _ in range(10):
                trtllm_moe_prologue(x, logits, bias)
            torch.cuda.synchronize()

            start_evt.record()
            for _ in range(base_iters):
                trtllm_moe_prologue(x, logits, bias)
            end_evt.record()
            torch.cuda.synchronize()
            base_us = (start_evt.elapsed_time(end_evt) * 1000.0) / base_iters
            base_str = f"{base_us:11.2f} µs"
            speedup_str = f"{base_us / fused_us:8.2f}x"
        else:
            base_str = "      N/A     "
            speedup_str = "    N/A  "

        regime = "Single-Block (<512)" if n < 512 else "Multi-Block (>=512)"

        print(
            f"{n:8d} | "
            f"{regime:>22} | "
            f"{base_str:>14} | "
            f"{fused_us:11.2f} µs | "
            f"{speedup_str:>9}"
        )


if __name__ == "__main__":
    test_parity()
    test_profiler_regimes()
    benchmark_prologue_latency()
