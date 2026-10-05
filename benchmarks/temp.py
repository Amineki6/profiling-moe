import os
import sys
import torch
from torch.profiler import profile, ProfilerActivity

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from kernels.fused_kolibri_router import fused_kolibri_router

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

x = torch.randn(1, 384, device="cuda", dtype=torch.bfloat16)
bias = torch.randn(384, device="cuda", dtype=torch.bfloat16)
k = 6

# Warmup compile and cuda graphs
for _ in range(50):
    eager_kolibri_router(x, bias, k)
    compiled_kolibri_router(x, bias, k)
    fused_kolibri_router(x, bias, k)
torch.cuda.synchronize()

print("\n" + "="*70)
print("=== 1. TORCH.COMPILE HARDWARE KERNEL BREAKDOWN (100 runs) ===")
print("="*70)
with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof_comp:
    for _ in range(100):
        compiled_kolibri_router(x, bias, k)
print(prof_comp.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("\n" + "="*70)
print("=== 2. FUSED TRITON HARDWARE KERNEL BREAKDOWN (100 runs) ===")
print("="*70)
with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof_tri:
    for _ in range(100):
        fused_kolibri_router(x, bias, k)
print(prof_tri.key_averages().table(sort_by="cuda_time_total", row_limit=10))

