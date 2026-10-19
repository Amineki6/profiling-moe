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

From Nsight Systems trace analysis ([`traces/00_baseline/trtllm_kolibri_full_blocks.nsys-rep`](file:///home/mkina/profiling/profiling-moe/traces/00_baseline/trtllm_kolibri_full_blocks.nsys-rep)):

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
- **Implementation**: Custom sm_89 CUDA kernel [`kernels/cuda/customKolibriSandwichNormKernels.cu`](file:///home/mkina/profiling/profiling-moe/kernels/cuda/customKolibriSandwichNormKernels.cu) with dynamic shared memory caching and in-register residual addition.
- **Trace References**:
  - Decode Regime ($N = 64$): [`traces/02_sandwich_norm/decode_n64.nsys-rep`](file:///home/mkina/profiling/profiling-moe/traces/02_sandwich_norm/decode_n64.nsys-rep) & [`.sqlite`](file:///home/mkina/profiling/profiling-moe/traces/02_sandwich_norm/decode_n64.sqlite)
  - Saturated Prefill ($N = 512$): [`traces/02_sandwich_norm/prefill_n512.nsys-rep`](file:///home/mkina/profiling/profiling-moe/traces/02_sandwich_norm/prefill_n512.nsys-rep) & [`.sqlite`](file:///home/mkina/profiling/profiling-moe/traces/02_sandwich_norm/prefill_n512.sqlite)

#### Hardware Performance Across Operating Regimes ($d_{\text{model}} = 2560$, RTX 4090)

| Operating Regime | Metric | Tool | Baseline (Separate) | Post-Fusion (`customKolibriSandwichNorm`) | Delta / Impact |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Decode ($N = 64$ tokens)**<br>*Memory-Latency Bound* | **Pure GPU Duration** | `nsys` | 6.69 µs (2.72 + 1.25 + 2.72) | **2.91 µs** | **-56.5% (2.30x speedup)** |
| | **Kernel Launches** | `nsys` | 3 launches | **1 launch** | **-2 launches (-66%)** |
| | **Intermediate DRAM Traffic** | Calc | 640 KB | **0 KB** | **-100% `post_attn` DRAM** |
| **Saturated Prefill ($N = 512$ tokens)**<br>*4 Full Waves (512 CTAs on 128 SMs, blockDim=256)* | **Pure GPU Duration** | `nsys` | 10.20 µs (3.74 + 2.72 + 3.74) | **5.47 µs** | **-46.4% (1.86x speedup)** |
| | **Kernel Launches** | `nsys` | 3 launches | **1 launch** | **-2 launches (-66%)** |
| | **Intermediate DRAM Traffic** | Calc | 5.12 MB | **0 MB** | **-5.12 MB DRAM / step** |
| **All Regimes** | **Numerical Parity** | PyTorch | Reference | Matches eager (`diff < 0.001`) | `torch.allclose` PASS |
| | **Full Block 2-Layer Latency** | Benchmark | 1.44 ms | `[ Pending Integration ]` | Next Step |

#### 4.1.1 Microarchitectural Metrics Table (Pending NCU Access)
*To be populated once NVIDIA Nsight Compute (`ncu`) is enabled on the target system.*

| Category | Microarchitectural Metric | NCU Metric Identifier | Baseline (Unfused) | Post-Fusion (`customKolibriSandwichNorm`) | Target / Expectation | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **DRAM Subsystem** | **Total DRAM Read Traffic** | `dram__bytes_read.sum` | `[ TBD MB ]` | `[ TBD MB ]` | Reduced by $\sim 50\%$ | Pending NCU |
| | **Total DRAM Write Traffic** | `dram__bytes_write.sum` | `[ TBD MB ]` | `[ TBD MB ]` | Reduced by $N \cdot D \cdot 2$ B | Pending NCU |
| | **DRAM Throughput % of Peak** | `dram__throughput.avg.pct_of_peak_sustained_elapsed` | `[ TBD % ]` | `[ TBD % ]` | High utilization in prefill | Pending NCU |
| **Cache Hierarchy** | **L2 Cache Sector Hit Rate** | `lts__t_sector_hit_rate.pct` | `[ TBD % ]` | `[ TBD % ]` | Higher hit rate | Pending NCU |
| | **L1/Tex Cache Hit Rate** | `l1tex__t_sector_hit_rate.pct` | `[ TBD % ]` | `[ TBD % ]` | Improved locality | Pending NCU |
| **On-Chip SRAM** | **Dynamic Shared Memory / CTA** | `launch__shared_mem_per_block_allocated` | 0 B | **5,120 B** | Exact $D \times 2$ bytes (5 KB) | Expected 5.12 KB |
| | **Shared Memory Bank Conflicts** | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` | 0 | `[ TBD ]` | **0 conflicts** | Pending NCU |
| **Register & SM** | **Registers Allocated per Thread** | `launch__registers_per_thread` | `[ TBD ]` | `[ TBD ]` | Target $\le$ 32 registers | Pending NCU |
| | **Local Memory Spills (L1 / DRAM)**| `smsp__sass_inst_executed_op_local_ld.sum` | 0 B | **0 B** | **Zero spills** | Pending NCU |
| | **Theoretical Warp Occupancy** | `sm__warps_active.avg.pct_of_peak_sustained_active` | `[ TBD % ]` | `[ TBD % ]` | Target $\ge$ 50% | Pending NCU |
| | **Eligible Warps per Cycle** | `smsp__issue_active.avg.per_cycle_active` | `[ TBD ]` | `[ TBD ]` | Latency hiding indicator | Pending NCU |

> [!TIP]
> **Command to capture NCU metrics once access is granted:**
> ```bash
> ncu --set full -f \
>     --metrics dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum \
>     -o traces/02_sandwich_norm/ncu_sandwich_norm \
>     /home/mkina/miniconda3/envs/thesis/bin/python benchmarks/06_profile_sandwich_norm.py --tokens 512
> ```

---

### 4.2 Fusion Candidate 2: Routed + Shared Expert Add (#21) + Post-MoE Sandwich Norm (#22)
- **Goal**: Combine routed & shared outputs and apply sandwich norm in registers before writing `moe_out`.
- **Implementation**: Custom sm_89 CUDA kernel [`kernels/cuda/customKolibriPostMoeNormKernels.cu`](file:///home/mkina/profiling/profiling-moe/kernels/cuda/customKolibriPostMoeNormKernels.cu) with dynamic shared memory caching and in-register residual addition.
- **Trace References**:
  - Decode Regime ($N = 64$): [`traces/03_post_moe_norm/decode_n64.nsys-rep`](file:///home/mkina/profiling/profiling-moe/traces/03_post_moe_norm/decode_n64.nsys-rep) & [`.sqlite`](file:///home/mkina/profiling/profiling-moe/traces/03_post_moe_norm/decode_n64.sqlite)
  - Saturated Prefill ($N = 512$): [`traces/03_post_moe_norm/prefill_n512.nsys-rep`](file:///home/mkina/profiling/profiling-moe/traces/03_post_moe_norm/prefill_n512.nsys-rep) & [`.sqlite`](file:///home/mkina/profiling/profiling-moe/traces/03_post_moe_norm/prefill_n512.sqlite)

#### Hardware Performance Across Operating Regimes ($d_{\text{model}} = 2560$, RTX 4090)

| Operating Regime | Metric | Tool | Baseline (Separate) | Post-Fusion (`customKolibriPostMoeNorm`) | Delta / Impact |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Decode ($N = 64$ tokens)**<br>*Memory-Latency Bound* | **Pure GPU Duration** | `nsys` | 3.94 µs (1.22 + 2.72) | **2.14 µs** | **-45.7% (1.84x speedup)** |
| | **Kernel Launches** | `nsys` | 2 launches | **1 launch** | **-1 launch (-50%)** |
| | **Intermediate DRAM Traffic** | Calc | 640 KB | **0 KB** | **-100% `moe_combined` DRAM** |
| **Saturated Prefill ($N = 512$ tokens)**<br>*4 Full Waves (512 CTAs on 128 SMs, blockDim=256)* | **Pure GPU Duration** | `nsys` | 6.47 µs (2.69 + 3.78) | **4.06 µs** | **-37.2% (1.59x speedup)** |
| | **Kernel Launches** | `nsys` | 2 launches | **1 launch** | **-1 launch (-50%)** |
| | **Intermediate DRAM Traffic** | Calc | 5.12 MB | **0 MB** | **-5.12 MB DRAM / step** |
| **All Regimes** | **Numerical Parity** | PyTorch | Reference | Matches eager (`diff < 0.001`) | `torch.allclose` PASS |
| | **Full Block 2-Layer Latency** | Benchmark | 1.44 ms | `[ Pending Integration ]` | Next Step |

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
| **1** | Kernel Fusion 1: Post-Attention Sandwich Norm + Pre-MoE Norm (#8 + #9) | **Completed (Micro)** | 6.69 µs -> 2.91 µs (2.30x speedup), 1 launch |
| **2** | Kernel Fusion 2: MoE Shared+Routed Add + Sandwich Norm (#21 + #22) | **Completed (Micro)** | 3.94 µs -> 2.14 µs (1.84x speedup), 1 launch |
| **3** | Kernel Fusion 3: Router + Expert Dispatch Map (#12 + #13 + #14) | *Planned* | 9.2 µs -> ~4.5 µs, 1 launch |
| **4** | Stream Concurrency: Overlap Shared & Routed Expert Compute | *Planned* | Concurrency via CUDA streams |
| **5** | CUDA Graphs: Zero-CPU Dispatch End-to-End | *Planned* | 1.44 ms -> < 1.0 ms full model |
| **6** | NCU Deep Dive: Populate DRAM & Micro-Kernel Metrics | *Planned* | Validate 0 spills, peak bandwidth |
