import os
import sys
import torch
import torch.cuda.nvtx as nvtx

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from kernels import fused_kolibri_router
from kernels.cuda import kolibri_cuda_ops

def main():
    assert torch.cuda.is_available(), "CUDA is required for profiling!"
    device = "cuda"
    dtype = torch.float32
    n_experts = 384
    top_k = 6
    batch_size = 512  # Standard batch size for comparison

    x = torch.randn(batch_size, n_experts, device=device, dtype=dtype)
    bias = torch.randn(n_experts, device=device, dtype=dtype)

    # 1. Warm-up to trigger compilation & CUDA context initialization
    print("[1/3] Warming up Triton and Custom CUDA kernels...")
    for _ in range(25):
        fused_kolibri_router(x, bias, top_k)
        kolibri_cuda_ops.forward_vB(x, bias, top_k)
    torch.cuda.synchronize()

    # 2. Profile targeted executions under NVTX
    print("[2/3] Profiling Fused Triton vs Custom Kolibri CUDA...")
    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()

    # --- Fused Triton Kernel ---
    nvtx.range_push("Fused Triton Kernel")
    for _ in range(10):
        fused_kolibri_router(x, bias, top_k)
    torch.cuda.synchronize()
    nvtx.range_pop()

    # --- Custom Kolibri CUDA Kernel ---
    nvtx.range_push("Custom Kolibri CUDA Kernel")
    for _ in range(10):
        kolibri_cuda_ops.forward_vB(x, bias, top_k)
    torch.cuda.synchronize()
    nvtx.range_pop()

    cudart.cudaProfilerStop()
    print("[3/3] Profiling completed successfully.")

if __name__ == "__main__":
    main()
