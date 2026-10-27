import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kolibri_multi_block_build_expert_maps import kolibri_multi_block_build_expert_maps

def profile_breakdown(num_tokens=98304, num_experts=384, top_k=6, iters=100):
    device = torch.device("cuda:0")
    token_selected_experts = torch.randint(
        0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device
    )

    # Warmup
    for _ in range(20):
        kolibri_multi_block_build_expert_maps(token_selected_experts, num_experts=num_experts, max_blocks=128)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        kolibri_multi_block_build_expert_maps(token_selected_experts, num_experts=num_experts, max_blocks=128)
    end.record()
    torch.cuda.synchronize()

    total_us = (start.elapsed_time(end) / iters) * 1000
    print(f"Total Multi-Block Pipeline Latency (N={num_tokens}): {total_us:.2f} us")

    # Single profiled iteration for NCU
    torch.cuda.cudart().cudaProfilerStart()
    kolibri_multi_block_build_expert_maps(token_selected_experts, num_experts=num_experts, max_blocks=128)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

if __name__ == "__main__":
    profile_breakdown(98304)
