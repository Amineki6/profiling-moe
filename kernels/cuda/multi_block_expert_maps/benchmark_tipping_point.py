import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from kolibri_multi_block_build_expert_maps import kolibri_multi_block_build_expert_maps
from kolibri_build_expert_maps import kolibri_build_expert_maps

def benchmark_n(n: int, num_experts: int = 384, top_k: int = 6, iters: int = 100):
    device = torch.device("cuda:0")
    x = torch.randint(0, num_experts, (n, top_k), dtype=torch.int32, device=device)

    # --- 1. Eager Benchmark ---
    # Warmup
    for _ in range(10):
        kolibri_build_expert_maps(x, num_experts)
        kolibri_multi_block_build_expert_maps(x, num_experts)
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    # Single-Block Eager
    start_ev.record()
    for _ in range(iters):
        kolibri_build_expert_maps(x, num_experts)
    end_ev.record()
    torch.cuda.synchronize()
    sb_eager_us = (start_ev.elapsed_time(end_ev) / iters) * 1000

    # Multi-Block Eager
    start_ev.record()
    for _ in range(iters):
        kolibri_multi_block_build_expert_maps(x, num_experts)
    end_ev.record()
    torch.cuda.synchronize()
    mb_eager_us = (start_ev.elapsed_time(end_ev) / iters) * 1000

    # --- 2. CUDA Graph Benchmark ---
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())

    # Single-Block Graph Capture
    with torch.cuda.stream(s):
        for _ in range(3):
            kolibri_build_expert_maps(x, num_experts)
    s.synchronize()
    g_sb = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_sb, stream=s):
        kolibri_build_expert_maps(x, num_experts)

    # Multi-Block Graph Capture
    with torch.cuda.stream(s):
        for _ in range(3):
            kolibri_multi_block_build_expert_maps(x, num_experts)
    s.synchronize()
    g_mb = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_mb, stream=s):
        kolibri_multi_block_build_expert_maps(x, num_experts)

    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # Warmup graphs
    for _ in range(10):
        g_sb.replay()
        g_mb.replay()
    torch.cuda.synchronize()

    # Benchmark Single-Block Graph
    start_ev.record()
    for _ in range(iters):
        g_sb.replay()
    end_ev.record()
    torch.cuda.synchronize()
    sb_graph_us = (start_ev.elapsed_time(end_ev) / iters) * 1000

    # Benchmark Multi-Block Graph
    start_ev.record()
    for _ in range(iters):
        g_mb.replay()
    end_ev.record()
    torch.cuda.synchronize()
    mb_graph_us = (start_ev.elapsed_time(end_ev) / iters) * 1000

    return {
        "N": n,
        "selections": n * top_k,
        "sb_eager": sb_eager_us,
        "mb_eager": mb_eager_us,
        "sb_graph": sb_graph_us,
        "mb_graph": mb_graph_us,
    }

def main():
    print("=" * 95)
    print("Fine-Grained Tipping Point Benchmark: Single-Block vs Multi-Block 3-Stage Pipeline")
    print("Hardware: NVIDIA GeForce RTX 4090 (128 SMs) | E = 384 Experts, Top-K = 6")
    print("=" * 95)

    test_tokens = [
        16, 32, 64, 96, 128, 192, 256, 384, 512, 640, 768, 1024, 1152, 2048, 4096, 8192, 98304
    ]

    results = []
    print(f"\n{'N':>6} | {'Tokens*K':>9} | {'SB Eager':>10} | {'MB Eager':>10} | {'Eager Winner':>13} | {'SB Graph':>10} | {'MB Graph':>10} | {'Graph Winner':>13}")
    print("-" * 95)

    for n in test_tokens:
        iters = 50 if n >= 8192 else 100
        res = benchmark_n(n, iters=iters)
        results.append(res)

        eager_winner = "Multi-Block" if res['mb_eager'] < res['sb_eager'] else "Single-Block"
        eager_diff = f"{res['sb_eager'] / res['mb_eager']:.2f}x"
        graph_winner = "Multi-Block" if res['mb_graph'] < res['sb_graph'] else "Single-Block"
        graph_diff = f"{res['sb_graph'] / res['mb_graph']:.2f}x"

        print(f"{res['N']:6d} | {res['selections']:9d} | {res['sb_eager']:8.2f} us | {res['mb_eager']:8.2f} us | {eager_winner:>11} ({eager_diff}) | {res['sb_graph']:8.2f} us | {res['mb_graph']:8.2f} us | {graph_winner:>11} ({graph_diff})")

    # Analyze tipping points
    eager_tipping = None
    for r in results:
        if r['mb_eager'] < r['sb_eager']:
            eager_tipping = r['N']
            break

    graph_tipping = None
    for r in results:
        if r['mb_graph'] < r['sb_graph']:
            graph_tipping = r['N']
            break

    print("\n" + "=" * 95)
    print("SUMMARY OF TIPPING POINTS:")
    print(f"  • Standard Eager Tipping Point:     N = {eager_tipping} tokens ({eager_tipping * 6} selections)")
    print(f"  • CUDA Graph Zero-Overhead Tipping: N = {graph_tipping} tokens ({graph_tipping * 6} selections)")
    print("=" * 95)

if __name__ == "__main__":
    main()
