# Iterative Performance Optimization Log: Kolibri 1 TRT-LLM Integration

## 1. Document Purpose
The purpose of this document is to track performance metrics and architectural modifications iteratively as optimizations are applied to the Kolibri 1 Mixture-of-Experts (MoE) pipeline in TensorRT-LLM. 

Each section records:
- The specific optimization applied.
- Systematic benchmark numbers across execution modes.
- Kernel-level profiling breakdown from NVIDIA Nsight Systems (`nsys`).
- Identified bottlenecks and areas targeted for subsequent iterations.

---

## 2. Iteration 0: Baseline (Native CUDA Router Integration)

### 2.1 Configuration & Setup
- **Target Architecture**: Kolibri 1 MoE
  - Hidden Dimension ($d_{\text{model}}$): `2560`
  - Attention Heads: `48` Query Heads, `4` Key/Value Heads, `head_dim = 128`
  - Sliding Window: `513` tokens
  - MoE Routed Experts ($E$): `384` fine-grained experts (`intermediate_size = 512`)
  - Top-$K$ Selection: `6` experts per token
  - Shared Expert: `1` dense expert (`intermediate_size = 512`)
  - Data Type: `bfloat16`
- **Hardware Platform**: NVIDIA GeForce RTX 4090 (24 GB VRAM, Compute Capability 8.9)
- **Software Stack**:
  - PyTorch 2.14.0+cu130
  - TensorRT-LLM v1.3.0rc29
  - CUDA Compilation: `nvcc` sm_89 with `-O3`
- **Routing Implementation**: Custom sm_89 CUDA kernel (`kolibri_gate_forward_kernel`) replacing the previous Triton router implementation.

---

### 2.2 Baseline Metrics

#### Full Model Forward Pass (`Kolibri1ForCausalLM` - 2 Full Blocks)
*Benchmark command: `python benchmarks/05_test_trtllm_full_blocks.py --mode model --num-layers 2 --runs 10`*
- **Pipeline**: Token Embedding $\rightarrow$ SWA Attention $\rightarrow$ Sandwich Norm $\rightarrow$ MoE Block $\rightarrow$ LM Head
- **Parameters**: 3.26 Billion (`3,261,751,040`)
- **Weights VRAM**: 6.08 GB
- **Peak Allocated VRAM**: 6.37 GB
- **Input Sequence**: 16 tokens
- **Average Forward Latency**: **1.44 ms**

---

### 2.3 Kernel-Level Execution Profile (Nsight Systems)

From Nsight Systems trace analysis (`traces/trtllm_kolibri_full_blocks.nsys-rep`):

| Kernel Name | Role | Instances | Avg Duration | % GPU Time |
| :--- | :--- | :---: | :---: | :---: |
| `cutlass::Kernel2<cutlass_80_wmma_...gemm...>` | Gate / Up / Down Projections | 27 | 43.80 µs | 38.6% |
| `fused_moe_oss::run_global<...sm80...>` | 384 Routed Experts Execution | 6 | 168.69 µs | 33.0% |
| `cutlass::Kernel<...MoeFCGemm...>` | MoE Fully Connected GEMM | 6 | 83.61 µs | 16.4% |
| `kolibri_gate_forward_kernel` | Native CUDA Router | 6 | 3.40 µs | 0.7% |
| `fusedBuildExpertMapsSortFirstTokenKernel` | Token-to-expert mapping | 6 | 3.56 µs | 0.7% |
| `finalizeMoeRoutingKernel` | MoE output reduction | 6 | 5.57 µs | 1.1% |
| `fmha_v2_flash_attention_...sm89` | SWA FlashAttention | 6 | 4.53 µs | 0.9% |
| `fusedQKNormRopeKernel` | QK RMSNorm + RoPE | 6 | 2.28 µs | 0.4% |
| `kernel_cutlass_...RMSNormKernel` | Pre/Post Sandwich Norms | 27 | ~2.6 µs | 2.3% |

*Total pure GPU execution time per forward pass*: **~1.03 ms**

---

## 3. Notes & Active Pain Points

### Pain Point 1: CPU-Bound Execution & Launch Gaps
While the native CUDA router kernel finishes execution in **$3.40\ \mu\text{s}$**, Nsight Systems timeline analysis reveals significant GPU idle gaps between operations:
1. **Post-Router Dispatch Gap (~$111.87\ \mu\text{s}$)**:
   - A distinct dead-time occurs immediately after `kolibri_gate_forward_kernel` completes before `fused_moe_oss::run_global` begins execution.
   - Cause: Control returns across language boundaries from C++ to the Python runtime (`Kolibri1RoutingMethod.apply`), unpacks tensors, passes arguments to `create_moe`, and enters CUTLASS host setup routines before launching the GEMM kernel.
2. **Attention Setup Stalls**:
   - Small CPU stalls are observed prior to `tensorrt_llm::_v1::kernels::computeSeqAndPaddingOffsets`, driven by host-side sequence length and KV-cache metadata marshaling.
3. **Execution Bound**:
   - At small batch and sequence sizes ($seq\_len \le 128$), the GPU frequently drains the hardware queue faster than Python can issue kernel launch calls from the host.

---

---

## 4. Kernel Fusion Metrics Tracking (To Be Populated)

This section tracks the micro and macro metrics for each candidate fusion to quantify memory-traffic elimination and register health.

### 4.1 Fusion Candidate 1: Post-Attention Sandwich Norm (#8) + Pre-MoE FusedAddRMSNorm (#9)
- **Goal**: Eliminate intermediate global memory DRAM write and read of `post_attn`.
- **Baseline Kernels**:
  - #8: `RMSNormKernel` (~2.4 µs)
  - #9: `FusedAddRMSNormKernel` (~2.9 µs)

| Metric | Tool | Baseline (Separate) | Post-Fusion | Delta / Impact | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Combined Kernel Duration** | `nsys` | 5.30 µs (2.4 + 2.9) | `[ TBD µs ]` | `[ TBD % ]` | Pending Implementation |
| **Kernel Launch Count** | `nsys` | 2 launches | 1 launch | -1 launch | Pending Implementation |
| **Global DRAM Read/Write Volume** | `ncu` (`dram__bytes.sum`) | `[ TBD KB ]` | `[ TBD KB ]` | Target ~ -50% | Awaiting NCU Access |
| **DRAM Throughput % of Peak** | `ncu` | `[ TBD % ]` | `[ TBD % ]` | `[ TBD ]` | Awaiting NCU Access |
| **Registers per Thread** | `ncu` (`launch__registers...`) | `[ TBD ]` | `[ TBD ]` | Target $\le$ 32 | Awaiting NCU Access |
| **Local Memory Spills** | `ncu` (`wavefronts_shared...`) | 0 bytes | `[ TBD bytes ]` | Must be 0 | Awaiting NCU Access |
| **Achieved Warp Occupancy** | `ncu` (`sm__warps_active...`) | `[ TBD % ]` | `[ TBD % ]` | Target $\ge$ 60% | Awaiting NCU Access |
| **Full Block 2-Layer Latency** | `nsys` / Benchmark | 1.44 ms | `[ TBD ms ]` | `[ TBD % ]` | Pending Implementation |

---

### 4.2 Fusion Candidate 2: Routed + Shared Expert Add (#21) + Post-MoE Sandwich Norm (#22)
- **Goal**: Combine routed & shared outputs and apply sandwich norm in registers before writing `moe_out`.
- **Baseline Kernels**:
  - #21: `vectorized_elementwise_kernel` (~1.1 µs)
  - #22: `RMSNormKernel` (~2.4 µs)

| Metric | Tool | Baseline (Separate) | Post-Fusion | Delta / Impact | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Combined Kernel Duration** | `nsys` | 3.50 µs (1.1 + 2.4) | `[ TBD µs ]` | `[ TBD % ]` | Pending Implementation |
| **Kernel Launch Count** | `nsys` | 2 launches | 1 launch | -1 launch | Pending Implementation |
| **Global DRAM Read/Write Volume** | `ncu` (`dram__bytes.sum`) | `[ TBD KB ]` | `[ TBD KB ]` | Target ~ -50% | Awaiting NCU Access |
| **DRAM Throughput % of Peak** | `ncu` | `[ TBD % ]` | `[ TBD % ]` | `[ TBD ]` | Awaiting NCU Access |
| **Registers per Thread** | `ncu` (`launch__registers...`) | `[ TBD ]` | `[ TBD ]` | Target $\le$ 32 | Awaiting NCU Access |
| **Local Memory Spills** | `ncu` (`wavefronts_shared...`) | 0 bytes | `[ TBD bytes ]` | Must be 0 | Awaiting NCU Access |
| **Achieved Warp Occupancy** | `ncu` (`sm__warps_active...`) | `[ TBD % ]` | `[ TBD % ]` | Target $\ge$ 60% | Awaiting NCU Access |
| **Full Block 2-Layer Latency** | `nsys` / Benchmark | 1.44 ms | `[ TBD ms ]` | `[ TBD % ]` | Pending Implementation |

---

### 4.3 Fusion Candidate 3: Kolibri Router (#12) + Expert Dispatch Map (#13 + #14)
- **Goal**: Output CUTLASS-ready sorted token-to-expert mapping tables directly from top-K sorting.
- **Baseline Kernels**:
  - #12: `kolibri_gate_forward_kernel` (3.40 µs)
  - #13: `fusedBuildExpertMapsSortFirstTokenKernel` (3.56 µs)
  - #14: `expandInputRowsKernel` (2.22 µs)

| Metric | Tool | Baseline (Separate) | Post-Fusion | Delta / Impact | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Combined Kernel Duration** | `nsys` | 9.18 µs (3.40 + 3.56 + 2.22) | `[ TBD µs ]` | `[ TBD % ]` | Pending Implementation |
| **Kernel Launch Count** | `nsys` | 3 launches | 1 launch | -2 launches | Pending Implementation |
| **Global DRAM Read/Write Volume** | `ncu` (`dram__bytes.sum`) | `[ TBD KB ]` | `[ TBD KB ]` | Target ~ -60% | Awaiting NCU Access |
| **DRAM Throughput % of Peak** | `ncu` | `[ TBD % ]` | `[ TBD % ]` | `[ TBD ]` | Awaiting NCU Access |
| **Registers per Thread** | `ncu` (`launch__registers...`) | `[ TBD ]` | `[ TBD ]` | Target $\le$ 48 | Awaiting NCU Access |
| **Local Memory Spills** | `ncu` (`wavefronts_shared...`) | 0 bytes | `[ TBD bytes ]` | Must be 0 | Awaiting NCU Access |
| **Achieved Warp Occupancy** | `ncu` (`sm__warps_active...`) | `[ TBD % ]` | `[ TBD % ]` | Target $\ge$ 50% | Awaiting NCU Access |
| **Full Block 2-Layer Latency** | `nsys` / Benchmark | 1.44 ms | `[ TBD ms ]` | `[ TBD % ]` | Pending Implementation |

---

## 5. Planned Iterations Log

| Iteration | Focus Area | Status | Target Metric |
| :---: | :--- | :---: | :--- |
| **0** | Baseline: Native CUDA Router sm_89 Integration | **Completed** | 1.44 ms / 2-block pass, 3.4 µs router |
| **1** | Kernel Fusion 1: Post-Attention Sandwich Norm + Pre-MoE Norm (#8 + #9) | *In Progress* | 5.3 µs -> ~3.0 µs, 1 launch |
| **2** | Kernel Fusion 2: MoE Shared+Routed Add + Sandwich Norm (#21 + #22) | *Planned* | 3.5 µs -> ~2.0 µs, 1 launch |
| **3** | Kernel Fusion 3: Router + Expert Dispatch Map (#12 + #13 + #14) | *Planned* | 9.2 µs -> ~4.5 µs, 1 launch |
| **4** | Stream Concurrency: Overlap Shared & Routed Expert Compute | *Planned* | Concurrency via CUDA streams |
| **5** | CUDA Graphs: Zero-CPU Dispatch End-to-End | *Planned* | 1.44 ms -> < 1.0 ms full model |
| **6** | NCU Deep Dive: Populate DRAM & Micro-Kernel Metrics | *Planned* | Validate 0 spills, peak bandwidth |
