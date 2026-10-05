import json
import os
from datetime import datetime
import torch
import triton.testing
from tabulate import tabulate

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
    NUM_EXPERTS = cfg.get("num_experts", cfg.get("moe_num_experts", 384))
    TOP_K = cfg.get("num_experts_per_tok", cfg.get("moe_top_k", 6))
else:
    NUM_EXPERTS = 384
    TOP_K = 6

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
DEVICE_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

print(f"Device: {DEVICE_NAME} | Experts: {NUM_EXPERTS} | Top-K: {TOP_K}")

# 1. Kolibri 1 Routing Logic
def eager_kolibri_router(x: torch.Tensor, bias: torch.Tensor, k: int):
    scores = x + bias
    topk_ids = torch.topk(scores, k=k, dim=-1).indices
    topk_weights = torch.sigmoid(torch.gather(x, -1, topk_ids))
    return topk_ids, topk_weights

@torch.compile(mode="reduce-overhead")
def compiled_kolibri_router(x: torch.Tensor, bias: torch.Tensor, k: int):
    scores = x + bias
    topk_ids = torch.topk(scores, k=k, dim=-1).indices
    topk_weights = torch.sigmoid(torch.gather(x, -1, topk_ids))
    return topk_ids, topk_weights

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from kernels import fused_kolibri_router

def bench_gpu_kernel_us(fn, *args):
    # warmup=25 ms, rep=100 ms (~6,000 iterations per kernel)
    ms = triton.testing.do_bench(lambda: fn(*args), warmup=25, rep=100, return_mode="median")
    return ms * 1000.0

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required for profiling!"

    batch_sizes = [1, 4, 8, 16, 32, 64, 128]
    bias = torch.randn(NUM_EXPERTS, device=DEVICE, dtype=DTYPE)

    table_data = []

    print("\nValidating numerical correctness & profiling sweeps...")
    for b in batch_sizes:
        x = torch.randn(b, NUM_EXPERTS, device=DEVICE, dtype=DTYPE)

        # 1. Warm up shape & verify mathematical equivalence
        eager_ids, eager_weights = eager_kolibri_router(x, bias, TOP_K)
        comp_ids, comp_weights = compiled_kolibri_router(x, bias, TOP_K)
        tri_ids, tri_weights = fused_kolibri_router(x, bias, TOP_K)

        assert torch.equal(eager_ids, comp_ids), f"Compile TopK mismatch at batch size {b}!"
        assert torch.allclose(eager_weights, comp_weights, atol=1e-3, rtol=1e-3), f"Compile Weight mismatch at batch size {b}!"

        # Verify Triton output correctness
        expected_tri_weights = torch.sigmoid(torch.gather(x, -1, tri_ids.long()))
        assert torch.allclose(tri_weights, expected_tri_weights, atol=1e-3, rtol=1e-3), f"Triton weight mismatch at batch size {b}!"

        # 2. Benchmark GPU execution time
        t_eager = bench_gpu_kernel_us(eager_kolibri_router, x, bias, TOP_K)
        t_comp = bench_gpu_kernel_us(compiled_kolibri_router, x, bias, TOP_K)
        t_tri = bench_gpu_kernel_us(fused_kolibri_router, x, bias, TOP_K)

        sp_comp = t_eager / t_comp if t_comp > 0 else 1.0
        sp_tri = t_eager / t_tri if t_tri > 0 else 1.0

        table_data.append([
            b,
            f"{t_eager:.2f}",
            f"{t_comp:.2f}",
            f"{t_tri:.2f}",
            f"{sp_comp:.2f}x",
            f"{sp_tri:.2f}x",
        ])

    print(f"\n### Kolibri 1 MoE Router Microbenchmark ({DEVICE_NAME})")
    print(tabulate(
        table_data,
        headers=[
            "Tokens (Batch)",
            "PyTorch Eager (µs)",
            "torch.compile (µs)",
            "Fused Triton (µs)",
            "Compile Speedup",
            "Triton Speedup",
        ],
        tablefmt="github"
    ))

    # Auto-save results to benchmarks/results/
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "01_router_results.json")

    result_payload = {
        "benchmark": "01_router_microbenchmark",
        "timestamp": datetime.now().isoformat(),
        "device": DEVICE_NAME,
        "experts": NUM_EXPERTS,
        "top_k": TOP_K,
        "dtype": str(DTYPE),
        "results": [
            {
                "batch_size": row[0],
                "eager_us": float(row[1]),
                "compiled_us": float(row[2]),
                "triton_us": float(row[3]),
                "compile_speedup": row[4],
                "triton_speedup": row[5],
            }
            for row in table_data
        ],
    }

    with open(json_path, "w") as f:
        json.dump(result_payload, f, indent=2)

    print(f"\n[Artifact Saved] Benchmark data saved to -> {json_path}")