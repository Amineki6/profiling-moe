# Kolibri 1 MoE Inference Profiling & Kernel Evaluation

An empirical performance analysis and kernel design study for deploying **Aleph Alpha's Kolibri 1 (78B MoE)** architecture in **NVIDIA TensorRT-LLM**.

This repository evaluates whether monolithic C++/CUDA kernels are strictly required for Kolibri's non-standard routing and attention schedules, demonstrating that lightweight fused Triton kernels and zero-position identity bypasses deliver **2.87× faster non-GEMM decode throughput** without modifying C++ source code.

---

## End-to-End Decode Impact (50 Layers)

![Kolibri 1 Decode Latency Impact](assets/03_end_to_end_decode_impact.png)

* **−1.73 ms per token saved (65.2% latency reduction)**: Non-GEMM decode overhead across all 50 layers drops from **2.66 ms** to **0.93 ms** (**2.87× speedup**) during single-token generation on an NVIDIA RTX 4090.
* **Key Optimizations**:
  1. **MoE Router Fusion ($E=384, k=6$)**: Fused 5 disjoint PyTorch kernels into 1 on-chip Triton kernel, saving **~0.61 ms/token** across 50 layers (**3.4× faster**).
  2. **Hybrid Attention Norm & RoPE Fusion (50 Layers)**: Bypassed TRT-LLM's conservative `skip_rope` assertion using layer-selective and zero-position identity fusion, saving **~1.12 ms/token** across 50 layers (**3.0× faster**).

---

## Architectural Challenges

When compiling Kolibri 1 with standard TensorRT-LLM fused kernels, two architectural divergences cause assertions or fallback to slow unfused kernels:

1. **MoE Routing ($E = 384$, $k = 6$)**:
   * **Pre-Activation Logit Bias**: Kolibri adds bias directly to raw logits (`x + bias`) before Top-K selection. TRT-LLM's `MiniMax2` kernel assumes post-sigmoid bias (`sigmoid(x) + bias`), altering expert selection order.
   * **Unnormalized Sigmoids**: Weights are raw sigmoids without sum-to-1 normalization ($\sum w_i \neq 1$), conflicting with TRT-LLM's `SigmoidRenorm` kernel.
   * **Expert Limit**: TRT-LLM's `RoutingCustomPolicy.cuh` caps compilation at $E \le 128$, whereas Kolibri routes across 384 experts.

2. **Hybrid Attention Schedule**:
   * **40 SWA Layers (80%)**: Sliding Window Attention with rotary embeddings (RoPE) and per-head RMSNorm.
   * **10 Full-Attention Layers (20%)**: Full-context attention with **No Rotary Embeddings (RNoPE)**, requiring pure RMSNorm.
   * **Framework Assertion**: TRT-LLM's `qk_norm_attention.py` hardcodes `assert not (fuse_qk_norm_rope and skip_rope)`, globally disabling fusion across the entire model.

---

## Investigation 01: MoE Router Fusion

* **Bottleneck**: PyTorch Eager and `torch.compile` split Kolibri's routing into 5 separate kernels (`add`, `gatherTopK`, `bitonicSort`, `gather`, `sigmoid`), generating DRAM roundtrips and an irreducible ~17 µs launch latency floor.
* **Solution**: Developed a fused Triton kernel ([kernels/fused_kolibri_router.py](file:///home/mkina/profiling/profiling-moe/kernels/fused_kolibri_router.py)) performing bias addition, iterative Top-6 extraction, and unnormalized sigmoid entirely in SRAM and registers in a single pass.
* **Results**: Per-layer router latency reduced from **17.41 µs** to **5.12 µs** (**3.40× macro speedup**; raw SM runtime dropped from 19.51 µs to 1.87 µs on Nsight Systems).

![Kolibri 1 MoE Router Microbenchmark](assets/01_router_microbenchmark.png)

> Full architectural proof, Triton implementation, and Nsight traces: [docs/01_router_fusion.md](file:///home/mkina/profiling/profiling-moe/docs/01_router_fusion.md).

---

## Investigation 02: Hybrid Attention Norm & RoPE Fusion

* **Bottleneck**: Because TRT-LLM disables fusion when `skip_rope=True`, all 50 layers default to an unfused fallback dispatching 3 separate CUDA kernels (`q_norm` + `k_norm` + `apply_rope`) per layer, adding 1.68 ms per token during decode.
* **Root Cause & Mathematical Workaround**: The assertion is unnecessary. When `skip_rope=True`, setting `position_ids = 0` causes the RoPE rotation matrix to collapse into the exact identity matrix ($R(\theta=0) = I$ since $\cos(0) = 1, \sin(0) = 0$). The existing C++ kernel executes pure per-head RMSNorm at full fused speed with **0.000000 bit-exact equivalence**.

### A. Sliding Window Attention (SWA) — 40 Layers (80%)
SWA layers apply both RMSNorm and RoPE. Enabling the fused CUDA kernel yields **2.0×–3.1× speedup** over unfused PyTorch.

![Kolibri 1 SWA Attention Norm Benchmark](assets/02_attention_norm_swa.png)

### B. Full Attention (RNoPE) — 10 Layers (20%)
Full-attention layers skip RoPE. Using the zero-position identity bypass unlocks the fused kernel for pure RMSNorm, delivering **1.6×–2.2× speedup** over the unfused baseline.

![Kolibri 1 Full Attention Norm Benchmark](assets/02_attention_norm_full.png)

> Bit-exact mathematical proof, verification script, and upstream integration paths: [docs/02_hybrid_attention.md](file:///home/mkina/profiling/profiling-moe/docs/02_hybrid_attention.md).

---

## Project Structure & Reproduction

```text
profiling-moe/
├── assets/                          # Publication-grade benchmark figures
│   ├── 01_router_microbenchmark.png
│   ├── 02_attention_norm_swa.png
│   ├── 02_attention_norm_full.png
│   └── 03_end_to_end_decode_impact.png
├── benchmarks/
│   ├── 01_benchmark_router.py       # MoE router benchmark (E=384, k=6)
│   ├── 02_benchmark_attention_norm.py # SWA & Full-Attention benchmark
│   ├── 02_verify_zero_position_rnope.py # Bit-exact numerical equivalence test
│   └── generate_figures.py          # Reproducible figure rendering pipeline
├── kernels/
│   └── fused_kolibri_router.py      # Custom single-pass Triton router
├── docs/
│   ├── 01_router_fusion.md          # Router deep-dive & Triton analysis
│   └── 02_hybrid_attention.md       # Attention norm analysis & RNoPE proof
```

### Reproduce Benchmarks & Figures

```bash
# 1. Benchmark MoE Router (Eager vs Inductor vs Fused Triton)
python benchmarks/01_benchmark_router.py

# 2. Benchmark Attention Norm & verify numerical equivalence
python benchmarks/02_benchmark_attention_norm.py
python benchmarks/02_verify_zero_position_rnope.py

# 3. Generate all publication figures
python benchmarks/generate_figures.py
```