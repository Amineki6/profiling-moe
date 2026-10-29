# CUDA Execution Flow: Single Baseline Kolibri Decoder Block

This document provides a comprehensive, low-level breakdown of every CUDA kernel executed during a single forward pass of one **Kolibri-1 decoder layer** in the **TensorRT-LLM baseline** configuration (unfused reference implementation).

---

## 1. Architectural Configuration

| Parameter | Symbol | Value | Notes |
| :--- | :--- | :--- | :--- |
| **Hidden Dimension** | $D$ | `2560` | Model representation width |
| **Attention Heads (Q)** | $N_Q$ | `48` | Query heads |
| **Attention Heads (KV)**| $N_{KV}$ | `4` | Grouped Query Attention (GQA 12:1 ratio) |
| **Head Dimension** | $d_k$ | `128` | Per-head vector length ($48 \times 128 = 6144$) |
| **QK Normalization** | — | `RMSNorm` | Applied per-head before attention dot product |
| **Routed Experts** | $E$ | `384` | Total sparse experts |
| **Active Experts (Top-K)** | $k$ | `6` | Selected experts per token |
| **Expert Intermediate** | $I_{\text{moe}}$ | `512` | Per-expert hidden projection dimension |
| **Shared Expert Intermediate** | $I_{\text{sh}}$ | `512` | Always active on all tokens |
| **FFN Activation** | — | `SwiGLU` | $\text{SiLU}(W_{\text{gate}} x) \odot W_{\text{up}} x$ |
| **Precision** | — | `BFloat16` | Weights and activations in BF16, Norm math in FP32 |

---

## 2. High-Level Kernel Sequence Overview

In a single baseline block forward pass, between **18 and 30+ distinct CUDA kernels** execute sequentially across 10 functional stages:

```mermaid
flowchart TD
    subgraph Pre-Attention
        K1["1. Input RMSNorm<br/><code>vectorized_layer_norm_kernel</code>"]
    end

    subgraph Attention Sublayer
        K2["2. QKV Projection GEMM<br/><code>ampere_bf16_s16816gemm</code>"]
        K3["3. Q-Head & K-Head RMSNorms<br/><code>vectorized_layer_norm_kernel (x2)</code>"]
        K4["4. Causal FlashAttention-2 (GQA)<br/><code>pytorch_flash::flash_fwd_kernel</code>"]
        K5["5. Out Projection GEMM<br/><code>ampere_s16816gemm + splitKreduce</code>"]
    end

    subgraph Region 1: Sandwich Norm (Unfused Baseline)
        K6["6. Post-Attention RMSNorm<br/><code>generalRmsNorm</code>"]
        K7["7. Residual Add<br/><code>vectorized_elementwise_kernel (add)</code>"]
        K8["8. Pre-MoE RMSNorm<br/><code>generalRmsNorm</code>"]
    end

    subgraph MoE Gating
        K9["9. Router Gate GEMM<br/><code>cutlass_80_tensorop_s16816gemm</code>"]
    end

    subgraph Region 2: MoE Prologue (Unfused Baseline)
        K10["10. Stage 1: Softmax & Top-6 Router<br/><code>kolibri_gate_forward_kernel</code>"]
        K11["11. Stage 2: 3-Step Dispatch Mapping<br/><code>blockExpertPrefixSumKernel</code><br/><code>globalExpertPrefixSumKernel</code><br/><code>mergeExpertPrefixSumKernel</code>"]
        K12["12. Stage 3: Activation Row Expansion<br/><code>trtllm_expand_input_rows_kernel</code>"]
    end

    subgraph MoE Computation
        K13["13. Routed Expert GEMMs (384 Experts)<br/><code>gemvx / cutlass GEMM + SwiGLU + Down GEMM</code>"]
        K14["14. MoE Finalize & Reduction<br/><code>vectorized_gather + elementwise_mul + reduce_sum</code>"]
        K15["15. Shared Expert FFN<br/><code>Gate-Up GEMM + SwiGLU + Down GEMM</code>"]
    end

    subgraph Region 3: Post-MoE Norm (Unfused Baseline)
        K16["16. Routed + Shared Add<br/><code>vectorized_elementwise_kernel (add)</code>"]
        K17["17. Post-FFN RMSNorm<br/><code>generalRmsNorm</code>"]
    end

    K1 --> K2 --> K3 --> K4 --> K5 --> K6 --> K7 --> K8 --> K9 --> K10 --> K11 --> K12 --> K13 --> K14 --> K16
    K8 -.-> K15 --> K16 --> K17
```

---

## 3. Detailed Stage-by-Stage Kernel Reference

### Stage 1: Input LayerNorm (`input_layernorm`)
Applies pre-normalization to the input token activations before the attention projection.
* **Kernel Symbol:** `void at::native::vectorized_layer_norm_kernel<c10::BFloat16, float, true>(...)`
* **Library/Source:** PyTorch CUDA / Native TRT-LLM `invokeGeneralRmsNorm`
* **Input:** $x \in \mathbb{R}^{N \times 2560}$ (or $x + \text{residual}$)
* **Output:** $h \in \mathbb{R}^{N \times 2560}$
* **Characteristics:** Pure memory-bandwidth bound. Loads activation vector, computes sum of squares across 2560 elements in registers/shared memory, multiplies by inverse root and `input_ln` gamma, and writes to global DRAM.

---

### Stage 2: Attention Sublayer (`attention`)

#### 2.1 QKV Projection GEMM
Projects normalized activations to fused Query, Key, and Value vectors.
* **Kernel Symbol:** `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_nn`
* **Library:** NVIDIA cuBLAS / CUTLASS
* **Math:** $[N, 2560] \times [2560, 7168] \to [N, 7168]$ (where $7168 = (48 + 2 \times 4) \times 128$)
* **Characteristics:** Compute/Tensor-Core bound on Ada Lovelace SMs.

#### 2.2 Query & Key RMSNorms (QK-Norm)
Applies independent RMSNorm per attention head for numerical stability.
* **Kernel Symbol:** `vectorized_layer_norm_kernel<c10::BFloat16, float, true>` (launched twice: once for Q heads, once for KV heads)
* **Math:** Normalizes 48 Q-heads ($128$ dims each) with `q_norm` and 4 KV-heads with `k_norm`.
* **Characteristics:** Small vector length reductions ($D=128$), memory-bound.

#### 2.3 Causal FlashAttention-2 with Grouped Query Attention (GQA)
Fuses causal masking, scaled dot-product attention softmax, and value projection.
* **Kernel Symbol:** `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<128, 64, 64, 4, ...>>`
* **Library:** FlashAttention-2 (Tri Dao / CUTLASS)
* **Math:** $O = \text{Softmax}\left(\frac{Q K^T}{\sqrt{128}} + M_{\text{causal}}\right) V$
* **Ratio:** 48 Query heads attend to 4 Key/Value heads ($12:1$ head sharing).
* **Characteristics:** High SRAM reuse via online softmax; avoids materializing the $[N, N]$ attention matrix in DRAM.

#### 2.4 Attention Out Projection GEMM
Projects multi-head concatenated output back to model dimension $D=2560$.
* **Kernel Symbol:** `ampere_s16816gemm_bf16_64x64_ldg8_stages_64x5_nn` followed by `cublasLt::splitKreduce_kernel`
* **Math:** $[N, 6144] \times [6144, 2560] \to [N, 2560]$

---

### Stage 3: Region 1 — Sandwich Norm (`R1_sandwich_norm`)
In the baseline, this region executes as **three independent kernel launches** requiring intermediate global memory roundtrips:

| Step | Kernel Symbol | Inputs $\to$ Outputs | Traffic (bytes/tok) | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Launch 1** | `tensorrt_llm::_v1::kernels::generalRmsNorm` | `attn_out` $\to$ `temp_post_attn` | $2560 \times 2 \times 2 = 10.24\text{ KB}$ | Post-attention RMSNorm with `gamma1` |
| **Launch 2** | `at::native::vectorized_elementwise_kernel<4, add>` | `residual` $+$ `temp_post_attn` $\to$ `updated_residual` | $2560 \times 2 \times 3 = 15.36\text{ KB}$ | Residual stream update |
| **Launch 3** | `tensorrt_llm::_v1::kernels::generalRmsNorm` | `updated_residual` $\to$ `moe_in` | $2560 \times 2 \times 2 = 10.24\text{ KB}$ | Pre-MoE RMSNorm with `gamma2` |

* **Total Baseline DRAM Traffic:** **$28\text{ KB}$ per token**.

---

### Stage 4: Router Gate GEMM (`gate_gemm`)
Computes unnormalized routing affinity scores across all 384 experts.
* **Kernel Symbol:** `cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_32x10_nn_align8>` followed by `splitKreduce_kernel`
* **Math:** $[N, 2560] \times [2560, 384] \to [N, 384]$
* **Type Conversion:** Followed by `direct_copy_kernel_cuda` converting logits to FP32 for routing precision.

---

### Stage 5: Region 2 — MoE Prologue (`R2_moe_prologue`)
The baseline TRT-LLM MoE prologue consists of **5 sequential kernel launches**:

#### 5.1 Stage 1: Router Softmax & Top-6 Selection
* **Kernel Symbol:** `tensorrt_llm::_v1::kernels::kolibri_gate_forward_kernel<float, float, 384, 6, true>`
* **Source:** `tensorrt_llm/kernels/moe/routing/kolibriRoutingKernels.cu`
* **Operation:** Adds `gate_bias` to FP32 logits, performs softmax across all 384 experts, selects the top-6 experts per token, and renormalizes the top-6 probabilities to sum to 1.0.
* **Outputs:** `routing_weights` ($[N, 6]$ float) and `selected_experts` ($[N, 6]$ int32).

#### 5.2 Stage 2: 3-Step Expert Dispatch Mapping
Calculates the scatter offsets and permutation tables mapping tokens to contiguous expert segments:
1. **Block Prefix Sum Kernel:**
   * `kolibri::baseline::blockExpertPrefixSumKernel<512>`
   * Computes local expert histograms and block-level exclusive prefix sums.
2. **Global Prefix Sum Kernel:**
   * `kolibri::baseline::globalExpertPrefixSumKernel<1024>`
   * Scans block-level totals across the entire sequence to compute global expert offsets.
3. **Merge Prefix Sum Kernel:**
   * `kolibri::baseline::mergeExpertPrefixSumKernel`
   * Scatters unpermuted row indices into `permuted_row_to_unpermuted_row` ($[N \times 6]$) and generates `expert_first_token_offset` ($[385]$).

*(Note: For $N \le 32$, TRT-LLM substitutes a single CUB block radix rank kernel `trtllm_build_expert_maps_kernel` with $32.9\text{ KB}$ shared memory footprint).*

#### 5.3 Stage 3: Activation Row Expansion
Gathers activations according to the permutation map to build grouped expert input matrices.
* **Kernel Symbol:** `kolibri::baseline::trtllm_expand_input_rows_kernel<__nv_bfloat16>`
* **Operation:** Reads source row $i = \text{permuted\_map}[j] \% N$ from `moe_in` ($[N, 2560]$) and writes to contiguous buffer `permuted_output` ($[N \times 6, 2560]$).
* **Characteristics:** Scalar memory access without 128-bit vectorization; bandwidth-bound ($917\text{ GB/s}$ at $N=4096$).

---

### Stage 6: Routed Expert GEMMs (`expert_gemms`)
Executes the grouped MLP for each expert $e \in [0, 383]$ that contains assigned tokens ($N_e > 0$):

For each active expert:
1. **Gate-Up GEMM:**
   * `internal::gemvx::kernel` or `cutlass_80_wmma_tensorop_...gemm`
   * Math: $[N_e, 2560] \times [2560, 1024] \to [N_e, 1024]$ (fused gate and up projections).
2. **SwiGLU Elementwise Kernel:**
   * `vectorized_elementwise_kernel<4, silu_kernel>` + `BinaryFunctor<MulFunctor>`
   * Math: $\text{SwiGLU}(G, U) = \text{SiLU}(G) \odot U \in \mathbb{R}^{N_e \times 512}$.
3. **Down Projection GEMM:**
   * `gemvNSP_kernel` or `cutlass_80_wmma_tensorop_...gemm`
   * Math: $[N_e, 512] \times [512, 2560] \to [N_e, 2560]$.

---

### Stage 7: MoE Finalize & Reduction (`finalize`)
Recombines the routed expert outputs back into the token dimension:
* **Step 1:** `at::native::vectorized_gather_kernel<16, long>` gathers expert output rows back to token index order using the inverse permutation map `u2p`.
* **Step 2:** `at::native::elementwise_kernel<128, 2, BinaryFunctor<MulFunctor>>` scales each token-expert output by its normalized routing weight.
* **Step 3:** `at::native::reduce_kernel<128, 4, ReduceOp<sum_functor>>` sums across the 6 active experts.
* **Step 4:** `vectorized_elementwise_kernel<4, bfloat16_copy_kernel_cuda>` casts the accumulated float results back to BF16 tensor `routed_out` ($[N, 2560]$).

---

### Stage 8: Shared Expert FFN (`shared_expert`)
Runs concurrently or sequentially over **all tokens** $N$:
1. **Shared Gate-Up GEMM:**
   * `cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_32x6_nn_align8>`
   * Math: $[N, 2560] \times [2560, 1024] \to [N, 1024]$.
2. **Shared SwiGLU:**
   * `elementwise_kernel<silu_kernel>` and `BinaryFunctor<MulFunctor>`.
3. **Shared Down GEMM:**
   * `cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_32x6_nn_align8>`
   * Math: $[N, 512] \times [512, 2560] \to [N, 2560]$ (`shared_out`).

---

### Stage 9: Region 3 — Post-MoE Norm (`R3_post_moe_norm`)
In the baseline, this region executes as **two independent kernel launches**:

| Step | Kernel Symbol | Inputs $\to$ Outputs | Traffic (bytes/tok) | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Launch 1** | `at::native::vectorized_elementwise_kernel<4, add>` | `routed_out` $+$ `shared_out` $\to$ `combined` | $2560 \times 2 \times 3 = 15.36\text{ KB}$ | Add routed and shared streams |
| **Launch 2** | `tensorrt_llm::_v1::kernels::generalRmsNorm` | `combined` $\to$ `moe_out` | $2560 \times 2 \times 2 = 10.24\text{ KB}$ | Post-FFN RMSNorm with `post_ffn_ln` |

* **Total Baseline DRAM Traffic:** **$20\text{ KB}$ per token**.

---

### Stage 10: Block Residual Accumulation
Adds the completed sublayer output back into the persistent residual stream:
* **Kernel Symbol:** `at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<c10::BFloat16>>`
* **Math:** `residual_out = residual_in + moe_out`.

---

## 4. Kernel Launch Summary for 1 Baseline Block

| Stage | Operation Description | Base Kernel Count | Dominant Limiter |
| :--- | :--- | :---: | :--- |
| **Pre-Attn** | Input RMSNorm | 1 | Memory Bandwidth |
| **Attention** | QKV GEMM, Q/K Norms, FlashAttn-2, Out GEMM | 5 | Compute / Tensor Core |
| **R1 (Sandwich)** | Post-Attn Norm $\to$ Add $\to$ Pre-MoE Norm | **3** | Launch Latency / DRAM BW |
| **Gating** | Router Gate GEMM & Typecast | 2 | Tensor Core |
| **R2 (Prologue)**| Top-6 Router $\to$ 3-Step Dispatch $\to$ Expand Rows | **5** | Launch Latency / DRAM Copy |
| **MoE Experts** | Grouped Gate-Up, SwiGLU, Down GEMMs | $3 \times E_{\text{active}}$ | Compute / Tensor Core |
| **Finalize** | Permutation Gather, Weight Mult, Sum Reduction | 4 | Memory Bandwidth / Gather |
| **Shared Expert**| Shared Gate-Up, SwiGLU, Down GEMMs | 3 | Tensor Core |
| **R3 (Post-MoE)**| Routed+Shared Add $\to$ Post-FFN Norm | **2** | Launch Latency / DRAM BW |
| **Block Output**| Final residual stream update | 1 | Memory Bandwidth |
| **Total** | Single Block Forward Pass | **$\mathbf{26 + 3 \times E_{\text{active}}}$** | Mixed |

