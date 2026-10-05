# 01: Kolibri MoE Router Fusion & Inductor Lowering Analysis

- **Investigation Date**: 2026-10-05
- **Hardware Target**: NVIDIA GeForce RTX 4090 (24 GB VRAM, 128 SMs, 72 MB L2 cache)
- **Software Stack**: PyTorch 2.14.0+cu130, Triton, Python 3.10
- **Model Target**: Aleph Alpha's Kolibri 1 (78B MoE, $E = 384$, $k = 6$)

---

## 1. Research Question

> **Can PyTorch Inductor (`torch.compile`) fuse Kolibri 1's non-standard routing operations into a single kernel, or is a custom fused Triton/CUDA kernel strictly necessary?**

### Mathematical Context & Architectural Divergence
Kolibri 1 diverges from standard MoE architectures (and existing TensorRT-LLM fused kernels) in three fundamental ways:

1. **Logit-Space Bias**: Kolibri adds bias directly to raw logits prior to Top-K selection:
   ```text
   scores = x + bias
   ```
   Existing fused kernels (e.g., TRT-LLM `MiniMax2`) apply bias in post-sigmoid space (`sigmoid(x) + bias`), which yields divergent expert rankings.
2. **Unnormalized Sigmoid Weights**: Kolibri assigns expert weights as:
   ```text
   topk_weights = sigmoid(x[topk_ids])
   ```
   where weights do not sum to 1 ($\sum w_i \neq 1$). Existing kernels (e.g., TRT-LLM `SigmoidRenorm`) strictly enforce sum normalization ($\sum w_i = 1$).
3. **High Expert Count**: Kolibri uses $E = 384$ experts per layer. TRT-LLM built-in fused routing traits (`RoutingCustomPolicy.cuh`) specify a hard compile-time limit of `Tier<128, 8>` ($E \le 128$).

---

## 2. Methodology & Profiling Harness

- **Timing Harness**: `triton.testing.do_bench(warmup=25, rep=100, return_mode="median")`.
  - Isolates pure GPU execution time from Python CPU dispatch latency.
  - Flushes L2 cache between runs to reflect DRAM memory bandwidth realistically.
- **Precision**: `bfloat16` inputs and weights ($E=384, k=6$).
- **Equivalence Check**: `assert torch.equal` (indices) and `assert torch.allclose` (weights) across all batch sizes.

---

## 3. Empirical Results

Microbenchmark conducted on **NVIDIA GeForce RTX 4090**:

| Tokens (Batch) | PyTorch Eager (µs) | `torch.compile` (µs) | Speedup | Phase |
| :--- | :--- | :--- | :--- | :--- |
| **1** | 17.41 | 16.38 | **1.06x** | Decode |
| **4** | 18.26 | 16.38 | **1.11x** | Decode |
| **8** | 17.54 | 16.38 | **1.07x** | Decode |
| **16** | 17.41 | 16.38 | **1.06x** | Decode |
| **32** | 17.70 | 17.41 | **1.02x** | Transition |
| **64** | 18.34 | 19.46 | **0.94x** | Prefill |
| **128** | 19.46 | 21.50 | **0.90x** | Prefill |

---

## 4. Under-the-Hood Proof (Compiler IR Inspection)

Inspecting the generated Inductor code (`TORCH_LOGS="output_code"`) in `/tmp/torchinductor_mkina/...` explains the exact reason why `torch.compile` provides virtually no speedup:

```python
def partition_0(args):
    arg1_1, arg2_1, s77 = args
    ...
    # STAGE 1: Logit bias addition
    buf0 = empty_strided_cuda((s77, 384), (384, 1), torch.bfloat16)
    triton_poi_fused_add_0.run(arg1_1, arg2_1, buf0, triton_poi_fused_add_0_xnumel, stream=raw_stream0)
    del arg2_1

    # STAGE 2: Top-K reduction fallback
    buf1 = torch.ops.aten.topk.default(buf0, 6, -1, True, True)
    del buf0
    buf3 = buf1[1]

    # STAGE 3: Unnormalized sigmoid on original logits
    buf4 = empty_strided_cuda((s77, 6), (6, 1), torch.bfloat16)
    triton_poi_fused_gather_sigmoid_1.run(buf3, arg1_1, buf4, triton_poi_fused_gather_sigmoid_1_xnumel, stream=raw_stream0)
    del arg1_1

    return (buf3, buf4, )
```

### Key Observations:
1. **Broken Fusion Boundary**: Inductor lacks a fused code generator for Top-K sorting/reductions. It lowers the graph into **3 separate stages**:
   - `triton_poi_fused_add_0` (pointwise addition)
   - `torch.ops.aten.topk.default` (external ATen C++ bitonic sort)
   - `triton_poi_fused_gather_sigmoid_1` (pointwise gather + sigmoid)
2. **Intermediate Global Memory Traffic**:
   - `buf0` allocates an intermediate tensor `(batch, 384)` in GPU memory.
   - Stage 1 writes all 384 scores out to memory/L2.
   - Stage 2 reads `buf0` back from memory, performs Top-K, and writes indices to `buf3`.
   - Stage 3 reads `buf3` and the original tensor `arg1_1` back to apply `sigmoid`.
3. **The Hardware Latency Floor**:
   - For a single token ($b=1$), the data payload is only ~768 bytes ($384 \times 2$ bytes). Raw compute is < 1 µs.
   - Chaining 3 sequential kernel launches and barriers on the GPU creates an irreducible latency floor of ~15–16 µs.
4. **Prefill Degradation (0.90x)**:
   - At larger batch sizes ($b=64, 128$), the dynamic batch symbol `s77` forces Inductor to calculate shape strides and allocations dynamically per invocation, creating higher overhead than PyTorch Eager's pre-allocated memory pool.

---

## 5. Architectural Conclusion & Next Steps

> **Conclusion**: `torch.compile` is **insufficient** for Kolibri 1 MoE routing. It fails to fuse across Top-K and incurs a 3-kernel launch floor of ~16 µs.
