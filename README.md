# Kolibri 1 MoE Inference Profiling & Kernel Evaluation

An empirical performance analysis and kernel design study for deploying **Aleph Alpha's Kolibri 1 (78B MoE)** architecture in **NVIDIA TensorRT-LLM**.

This repository investigates whether writing custom monolithic C++/CUDA kernels is strictly required for Kolibri's non-standard routing and attention layers, or if lighter execution paths (PyTorch compiled, Triton kernels, separated routing) offer comparable throughput with significantly reduced engineering overhead.

---

## Background & Architecture Divergence

When integrating Kolibri 1 into TensorRT-LLM, standard fused C++/CUDA kernels fail due to mathematical and architectural differences:

### 1. MoE Routing ($E = 384$, $k = 6$)
* **Logit-Space Bias:** Kolibri adds bias directly to raw logits (`scores = x + bias`) prior to Top-K selection. Existing TRT-LLM fused kernels (e.g., `MiniMax2`) apply bias in post-sigmoid space (`sigmoid(x) + bias`), which yields mathematically divergent expert rankings.
* **Unnormalized Sigmoids:** Kolibri assigns weights as `topk_weights = sigmoid(x[topk_ids])` where weights do not sum to 1 ($\sum w_i \neq 1$). TRT-LLM's `SigmoidRenorm` kernel strictly enforces sum normalization ($\sum w_i = 1$).
* **Expert Count Cap:** Built-in CUDA policy traits (`RoutingCustomPolicy.cuh`) specify a hard compilation limit of `Tier<128, 8>` ($E \le 128$). Kolibri uses 384 experts.

### 2. Attention Layer Structure
* **Interleaved SWA / RNoPE:** Kolibri alternates between Sliding Window Attention (with RoPE + RMSNorm) and Full Attention without rotary embeddings (RMSNorm only).
* **Fused Kernel Assertion:** TRT-LLM's `fusedQKNormRopeKernel.cu` does not support skipping RoPE while retaining per-head RMSNorm (`assert not (fuse_qk_norm_rope and skip_rope)`).

---

## Research Question

> **Can we fuse Kolibri's MoE routing operations (logit bias + top-k + unnormalized sigmoid) into a single CUDA/Triton kernel instead of relying on PyTorch's requires_separated_routing = True?**

- **Context**: Kolibri 1 routes tokens across 384 experts by adding an `e_score_correction_bias` directly to raw logits, selecting top-k (k=6), and applying an unnormalized sigmoid activation ($\sum w_i \neq 1$).
- **Challenge**: Existing trtllmGen routing kernels (`SigmoidRenorm` and `MiniMax2`) either force sum-to-1 normalization, do not support pre-activation logit biases, or are constrained to $\le 128$ experts.
- **Investigation**: Does profiling show that the 4 separate PyTorch operations create kernel launch overhead during decoding, and would extending trtllmGen with a (LogitBias -> TopK -> SigmoidNoNorm) policy (or a custom Triton kernel) yield measurable end-to-end latency gains?

> **Can the attention layer leverage fuse_qk_norm_rope = True despite Kolibri's alternating hybrid schedule (Sliding-Window with RoPE vs. Full-Attention with RNoPE)?**

- **Context**: Kolibri alternates every 5 layers: 4 Sliding-Window Attention (SWA) layers that apply rotary embeddings (RoPE), and 1 Full-Attention layer that skips RoPE completely (RNoPE).
- **Challenge**: The current CUDA kernel (`fusedQKNormRopeKernel.cu`) enforces `assert not (fuse_qk_norm_rope and skip_rope)`, disallowing normalization without rotation.
- **Investigation**: Can we either safely enable `fuse_qk_norm_rope = not self.is_full_attention` selectively on the 4 SWA layers, or add a pass-through bypass flag to `fusedQKNormRopeKernel.cu` so that all layers can benefit from single-pass SRAM fusion without crashing on RNoPE layers?

---

## Project Structure

```text
profiling-moe/
├── benchmarks/
│   ├── 01_benchmark_router.py
│   └── results/                     # Auto-generated JSON benchmark results
├── kernels/                         # Custom Triton/CUDA kernel implementations
├── traces/                          # NVIDIA Nsight Systems (.nsys-rep) captures
├── docs/                            # Deep-dive architectural analyses & IR proofs
│   └── 01_router_fusion.md
├── config.json                      # Kolibri 1 model architectural configuration
├── requirements.txt
└── README.md
```

---

## Investigation Progress & Results

| # | Investigation | Target Layer | Eager Baseline | `torch.compile` | Custom Kernel | Status | Deep-Dive Doc |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|
| **01** | **MoE Router Fusion** | Logit Bias + Top-6 ($E=384$) | 17.41 µs | 16.38 µs (1.06x) | *In Progress* | 🔬 Investigating | [01_router_fusion.md](docs/01_router_fusion.md) |
| **02** | **Hybrid Attention Norm** | SWA RoPE vs RNoPE | TBD | TBD | TBD | 📋 Planned | `docs/02_hybrid_attention.md` |