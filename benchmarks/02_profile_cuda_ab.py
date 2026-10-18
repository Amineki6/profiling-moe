import os
import sys
import torch
import torch.cuda.nvtx as nvtx

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from kernels.cuda import kolibri_cuda_ops

def main():
    assert torch.cuda.is_available(), "CUDA is required for profiling!"
    device = "cuda"
    dtype = torch.float32
    n_experts = 384
    top_k = 6
    batch_size = 4096  # High token count to saturate GPU memory and compute

    x = torch.randn(batch_size, n_experts, device=device, dtype=dtype)
    bias = torch.randn(n_experts, device=device, dtype=dtype)

    # 1. Warm-up to trigger JIT compilation and CUDA context init
    print("[1/3] Warming up kernels...")
    for _ in range(20):
        kolibri_cuda_ops.forward_vB(x, bias, top_k)
        kolibri_cuda_ops.forward_vFloat4(x, bias, top_k)
    torch.cuda.synchronize()

    # 2. Profile only the targeted executions
    print("[2/3] Profiling Variant B (Scalar) and Variant Float4 (Vectorized) under NVTX ranges...")
    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()

    # --- Variant B: Scalar Loads ---
    nvtx.range_push("Variant B: Scalar (customKolibriRoutingKernels)")
    for _ in range(50):
        kolibri_cuda_ops.forward_vB(x, bias, top_k)
    torch.cuda.synchronize()
    nvtx.range_pop()

    # --- Variant Float4: Vectorized 128-bit Loads ---
    nvtx.range_push("Variant Float4: Vectorized (customKolibriRoutingKernelsFloat4)")
    for _ in range(50):
        kolibri_cuda_ops.forward_vFloat4(x, bias, top_k)
    torch.cuda.synchronize()
    nvtx.range_pop()

    cudart.cudaProfilerStop()
    print("[3/3] Profiling completed successfully.")

if __name__ == "__main__":
    main()
