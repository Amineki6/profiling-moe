# Candidate 3: Dynamic Hybrid MoE Prologue (`fusions/03_moe_prologue`)

## 1. Executive Summary

This directory contains the production camera-ready implementation and native TensorRT-LLM baseline for **Candidate 3: Dynamic Hybrid MoE Prologue** (Router + Dispatch Map + Activation Expansion) for the Kolibri-5E MoE architecture ($E = 384, \text{Top-}K = 6$).

The MoE prologue consists of three sequential operations:
1. **Stage 1 (Router)**: Top-6 gating reduction across 384 experts with Sigmoid probability normalization.
2. **Stage 2 (Dispatch Map / Build Expert Maps)**: Sorting tokens by expert to produce destination scatter indices (`permuted_row_to_unpermuted_row`) and cumulative offsets (`expert_first_token_offset`).
3. **Stage 3 (Row Expansion / Activation Permutation)**: Gathering input activations into expert-contiguous memory buffers (`[num_tokens * 6, hidden_dim]`) ready for Grouped GEMM.

### Legacy Version Tags Eliminated
All confusing experimental tags (`3v1`, `3v2`, `3v3`) have been permanently removed. The architecture is now structured into a unified **Dynamic Hybrid Pipeline**:
* **Decode Regime ($N < 512$)**: Dispatches a single-block 256-thread kernel with only 3.0 KB shared memory (zero multi-kernel launch overhead, ~13 µs latency).
* **Prefill Regime ($N \ge 512$)**: Automatically splits into a 3-stage multi-block pipeline saturating all 128 SMs of the Ada Lovelace GPU, delivering up to **68.9x speedup** over TensorRT-LLM's baseline while scaling up to $N = 98,304$.

---

## 2. Directory Structure

```
fusions/03_moe_prologue/
├── README.md                              # Architecture & benchmark report
├── benchmark_and_verify.py                # Mathematical invariant verification, profiler trace, and latency suite
├── fused/                                 # Production Dynamic Hybrid Pipeline (Self-contained)
│   ├── custom_moe_prologue.cuh           # Pipeline declarations & constants
│   ├── custom_moe_prologue.cu            # Router + Hybrid Dispatch Map + 128-bit Expansion
│   ├── moe_prologue_binding.cpp          # PyTorch C++ bindings
│   └── fused_moe_prologue.py             # Python module with lazy JIT compilation
└── baseline/                              # Native TensorRT-LLM Baseline
    ├── trtllm_moe_prologue_baseline.cuh  # Baseline header
    ├── trtllm_moe_prologue_baseline.cu   # CUB RadixRank B32 + 3-kernel prefix sum fallback
    ├── baseline_binding.cpp               # PyTorch C++ bindings
    └── trtllm_baseline.py                 # Compiles directly from TensorRT-LLM/cpp/
```

---

## 3. Dynamic Hybrid Dispatch Architecture

| Sequence Length ($N$) | Active Pipeline Execution Path | Hardware Concurrency | Shared Memory | Latency Characteristics |
| :---: | :---: | :---: | :---: | :---: |
| **$N < 512$ (Decode)** | **Single-Block Dispatch Map**<br>(`single_block_build_expert_maps_kernel`) | 1 SM (256 threads) | **3.0 KB** (384 bins) | **10.3 – 15.9 µs**<br>(0 multi-kernel launch latency) |
| **$N \ge 512$ (Prefill)** | **3-Stage Multi-Block Pipeline**<br>1. `multi_block_expert_counts_kernel`<br>2. `multi_block_prefix_sum_kernel`<br>3. `multi_block_scatter_maps_kernel` | **All 128 SMs**<br>(128 blocks, 1024 threads/block) | **1.5 – 3.0 KB** | **23.5 – 409.9 µs**<br>(Scales linearly up to $N=98,304$) |

---

## 4. Accurate TensorRT-LLM Counterparts

In TensorRT-LLM (`TensorRT-LLM/cpp/tensorrt_llm/kernels/moe/cutlass/moe_kernels.cu`), the MoE prologue uses:

1. **Stage 1 (Router)**:
   Directly compiled from [`TensorRT-LLM/cpp/tensorrt_llm/kernels/moe/routing/kolibriRoutingKernels.cu`](file:///home/mkina/profiling/TensorRT-LLM/cpp/tensorrt_llm/kernels/moe/routing/kolibriRoutingKernels.cu) (`invokeKolibriRouting`).
2. **Stage 2 (Dispatch Map / Build Expert Maps)**:
   * **Single-Block Fast Path ($N \le 32$)**: Native CUB `BlockRadixRank<32, 9>` single-warp implementation (`trtllm_build_expert_maps_kernel`) matching TRT-LLM's `fusedBuildExpertMapsSortFirstTokenKernel` in [`moe_kernels.cu:L480`](file:///home/mkina/profiling/TensorRT-LLM/cpp/tensorrt_llm/kernels/moe/cutlass/moe_kernels.cu#L480).
   * **3-Kernel Unfused Fallback Pipeline ($N > 32$)**: When shared memory exceeds SM capacity or $N > 32$, TRT-LLM logs `Falling back to unfused prologue` (`moe_kernels.cu:L4615`) and invokes `threeStepBuildExpertMapsSortFirstToken`:
     1. `blockExpertPrefixSumKernel` (`moe_kernels.cu:L595`)
     2. `globalExpertPrefixSumKernel` / `globalExpertPrefixSumLargeKernel` (`moe_kernels.cu:L691`)
     3. `mergeExpertPrefixSumKernel` (`moe_kernels.cu:L830`)
3. **Stage 3 (Row Expansion)**:
   Scalar loop gather-scatter matching TRT-LLM's `expandInputRowsKernelLauncher` (`moe_kernels.cu:L1601`).

Our fused kernel replaces the heavy CUB RadixRank and uncoalesced fallback with:
* Single-block 256-thread histogram & prefix sum requiring only 3.0 KB SRAM (vs CUB's 33 KB - 263 KB).
* High-concurrency multi-block pipeline running on all 128 SMs with coalesced 128-bit transfers.

---

## 5. Python API Usage

```python
import sys
import torch

# Import fused operations
sys.path.append("/home/mkina/profiling/profiling-moe/fusions/03_moe_prologue/fused")
from fused_moe_prologue import fused_moe_prologue, kolibri_dispatch_map

tokens, hidden_dim, num_experts = 64, 2560, 384
unpermuted_input = torch.randn(tokens, hidden_dim, device="cuda", dtype=torch.bfloat16)
router_logits = torch.randn(tokens, num_experts, device="cuda", dtype=torch.float32)
router_bias = torch.randn(num_experts, device="cuda", dtype=torch.float32)

# End-to-end prologue execution
(
    permuted_output,      # [tokens * 6, hidden_dim]
    routing_weights,      # [tokens, 6]
    selected_experts,     # [tokens, 6]
    expert_offsets,       # [385]
    permuted_to_unperm,   # [tokens * 6]
    unperm_to_permuted,   # [tokens * 6]
) = fused_moe_prologue(unpermuted_input, router_logits, router_bias)
```

---

## 6. Benchmarking & Verification Results (Ada Lovelace RTX 4090)

### 6.1 Numerical Parity & Invariants
Validated against native TRT-LLM baseline across sequence lengths $N \in [1, 1152]$ for both BF16 and FP16:
* **Invariant A (Router Normalization)**: Sum of top-6 probabilities equals $1.0 \pm 10^{-5}$.
* **Invariant B (Expert Monotonicity)**: Expert offsets are strictly non-decreasing and terminate at $N \times 6$.
* **Invariant C (Bijective Permutation)**: Destination mapping contains zero collisions and exactly $N \times 6$ unique slots.
* **Invariant D (Activation Expansion Parity)**: Exact gather match with **0.000000 maximum absolute difference**.

### 6.2 Latency Comparison Table

Measured on NVIDIA GeForce RTX 4090 (24 GB, SM 8.9 Ada Lovelace) with PyTorch 2.5.1 / CUDA 12.4:

| Tokens ($N$) | Active Regime | TRT-LLM Native Baseline | Custom Fused Pipeline | Speedup |
| :---: | :---: | :---: | :---: | :---: |
| **1** | Single-Block ($N < 512$) | 19.06 µs (steady state) | **10.31 µs** | **1.85x** |
| **2** | Single-Block ($N < 512$) | 19.12 µs | **13.85 µs** | **1.38x** |
| **4** | Single-Block ($N < 512$) | 19.06 µs | **13.67 µs** | **1.39x** |
| **16** | Single-Block ($N < 512$) | 20.38 µs | **13.35 µs** | **1.53x** |
| **32** | Single-Block ($N < 512$) | 20.29 µs | **13.39 µs** | **1.51x** |
| **64** | Single-Block ($N < 512$) | 23.44 µs | **13.37 µs** | **1.75x** |
| **128** | Single-Block ($N < 512$) | 23.44 µs | **13.48 µs** | **1.74x** |
| **256** | Single-Block ($N < 512$) | 25.82 µs | **15.95 µs** | **1.62x** |
| **512** | Multi-Block ($N \ge 512$) | 33.76 µs | **23.50 µs** | **1.44x** |
| **1,024** | Multi-Block ($N \ge 512$) | 2,346.18 µs | **34.04 µs** | **68.93x** |
| **2,048** | Multi-Block ($N \ge 512$) | 2,380.58 µs | **181.76 µs** | **13.10x** |
| **4,096** | Multi-Block ($N \ge 512$) | 1,933.10 µs | **409.93 µs** | **4.72x** |
| **8,192** | Multi-Block ($N \ge 512$) | > 2,000 µs | **864.06 µs** | **> 2.3x** |
| **16,384** | Multi-Block ($N \ge 512$) | > 4,000 µs | **1,739.25 µs** | **> 2.3x** |
| **98,304** | Multi-Block ($N \ge 512$) | > 30,000 µs | **10,442.61 µs** | **> 3.0x** |

### 6.3 Profiler Kernel Trace Comparison

#### Fused Decode Regime ($N=64$):
* `kolibri::fusions::kolibri_router_kernel` (`reduce_topk::reduceTopK`): **3.55 µs**
* `kolibri::fusions::single_block_build_expert_maps_kernel`: **3.04 µs**
* `kolibri::fusions::kolibri_activation_expansion_kernel`: **2.18 µs**
* **Total CUDA Time**: **8.77 µs** (3 single-launch kernels)

#### Fused Prefill Regime ($N=1024$):
* `kolibri::fusions::kolibri_router_kernel` (`reduce_topk::reduceTopK`): **3.68 µs**
* `kolibri::fusions::kolibri_multi_block_expert_counts_kernel`: **1.28 µs**
* `kolibri::fusions::kolibri_multi_block_prefix_sum_kernel`: **3.07 µs**
* `kolibri::fusions::kolibri_multi_block_scatter_maps_kernel`: **2.62 µs**
* `kolibri::fusions::kolibri_activation_expansion_kernel`: **18.94 µs**
* **Total CUDA Time**: **29.60 µs** across all 128 SMs

#### TensorRT-LLM Native Baseline ($N=64$):
* `tensorrt_llm::_v1::kernels::kolibri_gate_forward`: 3.42 µs
* `kolibri::baseline::trtllm_expand_input_rows_kernel`: 2.75 µs
* `kolibri::baseline::blockExpertPrefixSumKernel`: 2.08 µs
* `kolibri::baseline::globalExpertPrefixSumKernel`: 1.54 µs
* `kolibri::baseline::mergeExpertPrefixSumKernel`: 1.25 µs
* **Total CUDA Time**: **11.04 µs** (5 separate kernels)
