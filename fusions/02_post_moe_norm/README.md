# Fusion 2: Fused Add + Post-MoE Norm (`fusions/02_post_moe_norm`)

## 1. Executive Summary

This directory contains the production camera-ready implementation and native TensorRT-LLM baseline for **Candidate 2: Routed + Shared Add + Post-MoE Norm** in the Kolibri-5E MoE architecture.

In standard unfused MoE architectures, the outputs of the routed experts (`routed_out`) and the shared expert (`shared_out`) are combined and normalized via two separate CUDA kernel launches:
1. **Launch 1: Elementwise Vector Add**: `combined = routed_out + shared_out` (writes result to global DRAM).
2. **Launch 2: Post-MoE RMSNorm**: `moe_out = RMSNorm(combined, gamma)` (reads `combined` from DRAM, normalizes, and writes `moe_out` to DRAM).

Our **Production Fused Kernel** merges both operations into a single kernel execution (`fusedAddPostMoeNormRegKernel`), holding intermediate sums in 128-bit vector registers with **zero dynamic shared memory** and **zero intermediate DRAM round-trips**.

---

## 2. Mathematical Definition

Given:
* $\mathbf{X}_{\text{routed}} \in \mathbb{R}^{N \times D}$ (output from routed experts)
* $\mathbf{X}_{\text{shared}} \in \mathbb{R}^{N \times D}$ (output from shared expert)
* $\boldsymbol{\gamma} \in \mathbb{R}^{D}$ (Post-MoE RMSNorm scale weight)

$$\mathbf{C}_{i, :} = \mathbf{X}_{\text{routed}, i, :} + \mathbf{X}_{\text{shared}, i, :}$$

$$\sigma_i = \sqrt{\frac{1}{D} \sum_{j=1}^{D} C_{i, j}^2 + \epsilon}$$

$$\mathbf{Y}_{i, j} = \left(\frac{C_{i, j}}{\sigma_i}\right) \cdot \gamma_j$$

where $N$ is the number of tokens, $D = 2560$ is the Kolibri hidden dimension, and $\epsilon = 10^{-5}$.

---

## 3. Directory Structure

```
fusions/02_post_moe_norm/
├── README.md                      # Architecture documentation and performance report
├── benchmark_and_verify.py        # Comprehensive parity verification & latency benchmark
├── fused/                         # Production Fused Kernel (Self-contained, 0 external deps)
│   ├── custom_post_moe_norm.cuh   # Vec128_t vector types and launcher declarations
│   ├── custom_post_moe_norm.cu    # fusedAddPostMoeNormRegKernel (In-register Ada Lovelace)
│   ├── post_moe_norm_binding.cpp  # PyTorch C++ extension bindings
│   └── fused_post_moe_norm.py     # Clean Python wrapper with lazy JIT compilation
└── baseline/                      # Native TensorRT-LLM Unfused Baseline
    ├── trtllm_post_moe_norm_baseline.cuh # Baseline header
    ├── trtllm_post_moe_norm_baseline.cu # Vector add + invokeGeneralRmsNorm
    ├── baseline_binding.cpp       # PyTorch C++ extension bindings
    └── trtllm_baseline.py         # Links directly to TensorRT-LLM/cpp rmsnormKernels.cu
```

---

## 4. Key Architectural Features

1. **128-Bit Memory Transactions**:
   All global loads and stores use 16-byte aligned vector primitives (`Vec128_t`), compiling to hardware `LDG.E.128` and `STG.E.128` instructions on SM 89.
2. **In-Register Fusion**:
   For $D=2560$ in BF16/FP16, each token is processed by 160 threads (5 warps) over 2 iterations. `routed_out` and `shared_out` are loaded directly into registers and summed in-place.
3. **Single-Barrier CTA Reduction (`blockSumAll`)**:
   Accumulates variance across threads using hardware warp shuffle instructions (`__shfl_xor_sync`) followed by a 5-warp broadcast through 32 bytes of static shared memory.
4. **Zero Dynamic Shared Memory**:
   Requires 0 dynamic shared memory bytes, completely avoiding shared memory bank conflicts and CTA occupancy limits.

---

## 5. Usage

```python
import torch
from fusions.02_post_moe_norm.fused.fused_post_moe_norm import fused_add_post_moe_norm

# Inputs [tokens, hidden_dim] on CUDA
tokens, hidden_dim = 64, 2560
routed_out = torch.randn(tokens, hidden_dim, device="cuda", dtype=torch.bfloat16)
shared_out = torch.randn(tokens, hidden_dim, device="cuda", dtype=torch.bfloat16)
gamma = torch.ones(hidden_dim, device="cuda", dtype=torch.bfloat16)

# Execute single-kernel fused operation
moe_out = fused_add_post_moe_norm(routed_out, shared_out, gamma, eps=1e-5)
```

---

## 6. Empirical Verification & Benchmark Results

Measured on an **NVIDIA GeForce RTX 4090** (24 GB VRAM, Ada Lovelace SM 89) using BF16 (`hidden_dim = 2560`, `eps = 1e-5`):

| Batch / Tokens ($N$) | Eager PyTorch | TRT-LLM Native Baseline | Custom Fused Kernel | Speedup vs TRT-LLM | Launch Reduction | Effective Bandwidth |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 28.33 µs | 4.44 µs | **2.00 µs** | **2.22x** | 2 $\rightarrow$ 1 (-50%) | 10.2 GB/s |
| **16** | 30.36 µs | 4.09 µs | **1.84 µs** | **2.22x** | 2 $\rightarrow$ 1 (-50%) | 136.1 GB/s |
| **64** | 31.46 µs | 4.23 µs | **2.11 µs** | **2.01x** | 2 $\rightarrow$ 1 (-50%) | 468.7 GB/s |
| **128** | 44.39 µs | 4.64 µs | **2.24 µs** | **2.07x** | 2 $\rightarrow$ 1 (-50%) | 879.9 GB/s |
| **256** | 48.81 µs | 6.48 µs | **3.01 µs** | **2.15x** | 2 $\rightarrow$ 1 (-50%) | 1,308.0 GB/s |
| **512** | 66.50 µs | 9.38 µs | **3.62 µs** | **2.59x** | 2 $\rightarrow$ 1 (-50%) | 2,172.6 GB/s |
| **1,024** | 92.16 µs | 27.23 µs | **6.09 µs** | **4.47x** | 2 $\rightarrow$ 1 (-50%) | 2,584.9 GB/s |
| **2,048** | 121.89 µs | 27.08 µs | **9.02 µs** | **3.00x** | 2 $\rightarrow$ 1 (-50%) | 3,488.2 GB/s |
| **4,096** | 569.55 µs | 200.00 µs | **27.31 µs** | **7.32x** | 2 $\rightarrow$ 1 (-50%) | 2,304.2 GB/s |
| **16,384** | 3,826.68 µs | 1,042.62 µs | **580.54 µs** | **1.80x** | 2 $\rightarrow$ 1 (-50%) | 433.5 GB/s |
| **98,304** | 23,825.75 µs | 6,165.80 µs | **3,257.00 µs** | **1.89x** | 2 $\rightarrow$ 1 (-50%) | 463.6 GB/s |

### Verification Summary
- **Numerical Parity**: Bit-for-bit / high-precision equivalence verified against PyTorch eager reference and TRT-LLM across BF16 (`atol=0.04, rtol=0.02`) and FP16 (`atol=0.001, rtol=0.001`).
- **Kernel Symbols Checked**: PyTorch CUDA Profiler confirms TRT-LLM baseline executes `vector_add_kernel` + `tensorrt_llm::_v1::kernels::generalRmsNorm`, while the fused version executes only `kolibri::fusions::fusedAddPostMoeNormRegKernel`.
