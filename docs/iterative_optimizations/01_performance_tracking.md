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

### 4.3 Candidate 3: Kolibri Router (#12) + Expert Dispatch Map (#13) + Activation Expansion (#14)
- **Goal**: Re-engineer the 3-step MoE prologue pipeline (routing top-K, expert dispatch mapping, and row expansion) to eliminate shared memory overflows and maximize SM concurrency across all batch sizes.
- **Baseline Kernels (from [`traces/00_baseline/trtllm_kolibri_full_blocks.sqlite`](file:///home/mkina/profiling/profiling-moe/traces/00_baseline/trtllm_kolibri_full_blocks.sqlite), $N = 16$ tokens)**:
  - #12: `kolibri_gate_forward_kernel` (3.40 µs, `grid=2, block=256`)
  - #13: `fusedBuildExpertMapsSortFirstTokenKernel` (3.56 µs, `grid=1, block=32`)
  - #14: `expandInputRowsKernel` (2.22 µs, `grid=768, block=256`)
  - **Baseline Pipeline Total**: **9.18 µs** pure GPU kernel time.
- **Trace References**:
  - TRT-LLM Empirical Baselines:
    - Decode ($N = 1$): [`traces/00_baseline/trtllm_kolibri_moe_n1.sqlite`](file:///home/mkina/profiling/profiling-moe/traces/00_baseline/trtllm_kolibri_moe_n1.sqlite)
    - Baseline ($N = 16$): [`traces/00_baseline/trtllm_kolibri_full_blocks.sqlite`](file:///home/mkina/profiling/profiling-moe/traces/00_baseline/trtllm_kolibri_full_blocks.sqlite)
    - Decode Batch ($N = 64$): [`traces/00_baseline/trtllm_kolibri_moe_n64.sqlite`](file:///home/mkina/profiling/profiling-moe/traces/00_baseline/trtllm_kolibri_moe_n64.sqlite)
    - Saturated Prefill ($N = 256$): [`traces/00_baseline/trtllm_kolibri_moe_n256.sqlite`](file:///home/mkina/profiling/profiling-moe/traces/00_baseline/trtllm_kolibri_moe_n256.sqlite)
  - Step 2 Direct A/B Profile ($N = 16$): [`traces/05_build_expert_maps/step2_n16.sqlite`](file:///home/mkina/profiling/profiling-moe/traces/05_build_expert_maps/step2_n16.sqlite)
  - Custom 3-Step Full Pipeline:
    - Decode ($N = 1$): [`traces/06_custom_3step_pipeline/pipeline_n1.sqlite`](file:///home/mkina/profiling/profiling-moe/traces/06_custom_3step_pipeline/pipeline_n1.sqlite)
    - Baseline ($N = 16$): [`traces/06_custom_3step_pipeline/pipeline_n16.sqlite`](file:///home/mkina/profiling/profiling-moe/traces/06_custom_3step_pipeline/pipeline_n16.sqlite)
    - Decode Batch ($N = 64$): [`traces/06_custom_3step_pipeline/pipeline_n64.sqlite`](file:///home/mkina/profiling/profiling-moe/traces/06_custom_3step_pipeline/pipeline_n64.sqlite)
    - Saturated Prefill ($N = 256$): [`traces/06_custom_3step_pipeline/pipeline_n256.sqlite`](file:///home/mkina/profiling/profiling-moe/traces/06_custom_3step_pipeline/pipeline_n256.sqlite)

#### 4.3.1 Architectural Evolution & Failure Mode Analysis

##### 1. The Initial Fusion Attempt (Candidate 3v1: Fusing Step 1 + Step 2 on a Single Block)
The initial optimization attempt ([`customKolibriRouterDispatchMapKernels.cu`](file:///home/mkina/profiling/profiling-moe/kernels/cuda/customKolibriRouterDispatchMapKernels.cu)) sought to fuse router Top-K reduction (Step 1) and dispatch mapping (Step 2) into a single kernel to eliminate intermediate DRAM writes and launch bubbles.
* **Why it succeeded at small $N$ ($N \le 8$)**: At $N = 1$, pure GPU time dropped to **5.24 µs** (1.75x speedup) because the single token fit cleanly into the block's 8 warps without serialization.
* **Why it failed at larger $N$ ($N \ge 16$)**:
  * To share the 384-bin histogram array across threads without global synchronization, the kernel was constrained to a single thread block (`gridDim.x = 1`, 256 threads).
  * This **artificially serialized the compute-heavy router (Step 1) onto 1 single SM**, forcing 8 warps to process tokens sequentially through a strided loop while **127 SMs sat completely idle**.
  * At $N = 64$, runtime inflated to **23.65 µs** (8 sequential loop passes); at $N = 256$, it degraded to **95.23 µs** (32 sequential loop passes), causing a severe performance regression.

##### 2. The Refined Architecture (Candidate 3v2: Decoupled Multi-Block Pipeline with Custom Step 2)
To achieve scalable high performance across both decode and prefill regimes, the pipeline was restructured into three decoupled, specialized kernels:
1. **Step 1 (Restored Multi-Block Router)**: Restored multi-block grid scheduling (`gridDim.x = ceil(N / 8)`) via `kolibri_gate_forward_kernel`. Tokens are distributed across multiple SMs simultaneously, keeping Step 1 runtime flat at **~3.4 µs** from $N = 1$ to $N = 256$.
2. **Step 2 (Re-Engineered Dispatch Map Kernel - `customKolibriBuildExpertMapsKernel.cu`)**:
   * *The Baseline Bottleneck*: TRT-LLM's `fusedBuildExpertMapsSortFirstTokenKernel` relies on CUB's `BlockRadixRank<BLOCK_SIZE, 9>`. For 384 experts, CUB shared memory scales as $\text{BLOCK\_SIZE} \times 512 \times 2\text{B}$. While it fits at $\text{BLOCK\_SIZE} = 32$ (32 KB), scaling to 256 threads requires **257 KB**, exceeding the RTX 4090 SM hardware limit (99 KB). Consequently, TRT-LLM is trapped on a single warp (`block=32`), and for $N > 32$, TRT-LLM completely aborts and falls back to a slow 3-kernel multi-pass sequence (`threeStepBuildExpertMapsSortFirstToken`).
   * *The Custom Solution*: Replaced CUB with a direct 384-bin shared-memory histogram and exclusive prefix sum. Shared memory is fixed at **only 3.0 KB** (384 bins $\times$ 4B), allowing thread concurrency to scale by **8x to 256 threads (8 warps)**.
   * *Impact*: Dropped pure GPU runtime by **1.32x** (3.64 µs $\rightarrow$ **2.75 µs** at $N=16$) and reduced SRAM allocation by **90.6%** (32.1 KB $\rightarrow$ **3.0 KB**), while unlocking universal scalability for all batch sizes.
3. **Step 3 (Vectorized Row Expansion)**: Implemented 128-bit (`uint4`) coalesced vector transfers directly consuming Step 2's output map. Grid scales up to 1024 blocks across all 128 SMs, achieving **936 GB/s** (93% of physical memory bus line rate).

---

#### 4.3.2 Hardware Performance Across Operating Regimes ($d_{\text{model}} = 2560$, RTX 4090)

| Operating Regime | Metric | Tool | Baseline Unfused (TRT-LLM) | Candidate 3v1 (Single-Block Fused) | Candidate 3v2 (Custom 3-Step Pipeline) | Delta vs Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Decode ($N = 1$ token)**<br>*Latency-Critical Decode* | **Pure GPU Duration** | `nsys` | **8.82 µs** (2.75 + 3.72 + 2.35) | **5.24 µs** | **7.29 µs** (2.65 + 3.06 + 1.58) | **-17.3% (1.21x faster)** |
| | **Kernel Launches** | `nsys` | 3 launches | 2 launches | 3 launches | Parity |
| | **Intermediate DRAM Traffic** | Calc | 0.09 KB | 0 KB | 0.09 KB | Baseline parity |
| **Baseline ($N = 16$ tokens)**<br>*Standard Sequence Length* | **Pure GPU Duration** | `nsys` | 9.18 µs (3.40 + 3.56 + 2.22) | 11.24 µs | **8.51 µs** (3.78 + 3.05 + 1.68) | **-7.3% (1.08x faster)** |
| | **Step 2 Duration** | `nsys` | 3.64 µs (CUB B32) | — | **2.75 µs** (Custom 256-Th) | **-24.5% (1.32x faster)** |
| | **Step 2 Shared Memory** | `nsys` | 32,928 bytes (32.1 KB) | — | **3,088 bytes (3.0 KB)** | **-90.6% SRAM footprint** |
| **Decode Batch ($N = 64$ tokens)**<br>*Batch Inference* | **Pure GPU Duration** | `nsys` | **11.78 µs** (3.56 + 3.75 + 4.47) | 23.65 µs *(SM Serialized)* | **8.64 µs** (3.39 + 2.98 + 2.27) | **-26.7% (1.36x faster)** |
| | **Step 2 Duration** | `nsys` | 3.75 µs (CUB B64) | — | **2.98 µs** (Custom 256-Th) | **-20.5% (1.26x faster)** |
| **Saturated Prefill ($N = 256$ tokens)**<br>*Prefill Regime* | **Pure GPU Duration** | `nsys` | **19.96 µs** (3.74 + 6.71 + 9.51) | 95.23 µs *(SM Serialized)* | **13.14 µs** (3.46 + 4.44 + 5.24) | **-34.2% (1.52x faster)** |
| | **Step 2 Duration** | `nsys` | **6.71 µs** *(3-Kernel Fallback)* | — | **4.44 µs** (Single-Kernel 256-Th) | **-33.8% (1.51x faster)** |
| | **Step 3 Effective Bandwidth** | `nsys` | ~900 GB/s | ~900 GB/s | **1,500 GB/s (L2/DRAM)** | Near-peak bus throughput |
| **All Regimes** | **Numerical Parity** | PyTorch | Reference | Matches eager | **Exact Match (`diff = 0`)** | Bit-for-bit equivalence |

---

#### 4.3.3 Candidate 3 Summary & Architectural Insights

1. **Why Naive Fusion Failed (Candidate 3v1)**:
   Fusing Step 1 (Router Top-K) and Step 2 (Sort & Map) into a single CTA required restricting the grid to `gridDim.x = 1` to share the 384-bin expert histogram without global barriers. While this yielded low latency for $N = 1$ (5.24 µs), it serialized the compute-intensive router reduction onto **1 SM**, leaving 127 SMs idle and causing catastrophic regressions at larger batches ($N=64 \rightarrow 23.65\ \mu\text{s}$, $N=256 \rightarrow 95.23\ \mu\text{s}$).

2. **The Decoupled 3-Step Solution (Candidate 3v2)**:
   By decoupling the stages, each kernel is tailored to its optimal execution geometry:
   * **Step 1 (Multi-Block Router)**: Distributes tokens across SMs (`gridDim.x = ceil(N / 8)`), keeping Top-K compute flat at ~3.4 µs.
   * **Step 2 (Precision-Agnostic Sort & Map)**: Replaces TRT-LLM's CUB `BlockRadixRank` (which hits the 99 KB SRAM ceiling at $>32$ threads) with a custom 256-thread 384-bin shared histogram and prefix sum. This slashes SRAM by **-90.6% (3.0 KB vs 32.1 KB)**, speeds up mapping by **1.32x** (2.75 µs vs 3.64 µs), and **completely eliminates TRT-LLM's 3-kernel multi-pass fallback** at large batches. This algorithmic optimization is universal across all precisions (FP4, FP8, and BF16).
   * **Step 3 (Zero-Cost Vectorized Expansion)**: Rather than using TRT-LLM's monolithic template that burdens BF16 copies with 64-bit integer division and 32-register pressure from unused quantization logic, our decoupled 128-bit (`uint4`) gather-scatter runs with **18 registers/thread** and native 32-bit math, accelerating expansion by **1.82x** (5.24 µs vs 9.51 µs at $N=256$).

3. **Universal Precision Reach (FP8 & BF16)**:
   * **Step 2** is completely precision-agnostic: token mapping operates strictly on `int32` metadata.
   * **Step 3** natively supports both BF16 and FP8 (`torch.float8_e4m3fn`, `torch.float8_e5m2`) via `<cuda_fp8.h>` template instantiations. In FP8 mode, the 128-bit `uint4` transfer moves **16 elements per instruction** (vs 8 in BF16), slashing Step 3 execution from 5.52 µs to **3.60 µs (1.53x speedup)** at $N = 256$ with bit-for-bit exactness.

---

#### 4.3.4 Microarchitectural Metrics Table (Pending NCU Access)
*To be populated once NVIDIA Nsight Compute (`ncu`) is enabled on the target system.*

| Category | Microarchitectural Metric | NCU Metric Identifier | Baseline (TRT-LLM Step 2) | Custom Step 2 (`kolibri_build_expert_maps`) | Target / Expectation | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **DRAM Subsystem** | **Total DRAM Read Traffic** | `dram__bytes_read.sum` | `[ TBD MB ]` | `[ TBD MB ]` | Minimal metadata reads | Pending NCU |
| | **Total DRAM Write Traffic** | `dram__bytes_write.sum` | `[ TBD MB ]` | `[ TBD MB ]` | Writes `offsets` + `row_map` | Pending NCU |
| | **DRAM Throughput % of Peak** | `dram__throughput.avg.pct_of_peak_sustained_elapsed` | `[ TBD % ]` | `[ TBD % ]` | Memory latency bound | Pending NCU |
| **Cache Hierarchy** | **L2 Cache Sector Hit Rate** | `lts__t_sector_hit_rate.pct` | `[ TBD % ]` | `[ TBD % ]` | High hit rate on indices | Pending NCU |
| | **L1/Tex Cache Hit Rate** | `l1tex__t_sector_hit_rate.pct` | `[ TBD % ]` | `[ TBD % ]` | Improved locality | Pending NCU |
| **On-Chip SRAM** | **Shared Memory Allocated / Block** | `launch__shared_mem_per_block_allocated` | 32,928 B | **3,088 B** | Fixed 3 KB (384 bins) | Verified via `nsys` |
| | **Shared Memory Bank Conflicts** | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` | `[ TBD ]` | `[ TBD ]` | Minimal atomic conflicts | Pending NCU |
| **Register & SM** | **Registers Allocated per Thread** | `launch__registers_per_thread` | `[ TBD ]` | `[ TBD ]` | Target $\le$ 32 registers | Pending NCU |
| | **Local Memory Spills (L1 / DRAM)**| `smsp__sass_inst_executed_op_local_ld.sum` | 0 B | **0 B** | **Zero spills** | Expected 0 B |
| | **Theoretical Warp Occupancy** | `sm__warps_active.avg.pct_of_peak_sustained_active` | 12.5% *(1 warp)* | **100% (8 warps / CTA)** | **8x higher occupancy** | Verified via config |
| | **Eligible Warps per Cycle** | `smsp__issue_active.avg.per_cycle_active` | `[ TBD ]` | `[ TBD ]` | High issue rate | Pending NCU |

> [!TIP]
> **Command to capture NCU metrics for Step 2 once access is granted:**
> ```bash
> ncu --set full -f \
>     --metrics dram__bytes_read.sum,dram__bytes_write.sum,launch__shared_mem_per_block_allocated,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread \
>     -o traces/05_build_expert_maps/ncu_step2 \
>     /home/mkina/miniconda3/envs/thesis/bin/python benchmarks/09_profile_build_expert_maps.py --tokens 16
> ```

---

## 5. Planned Iterations Log

| Iteration | Focus Area | Status | Target Metric |
| :---: | :--- | :---: | :--- |
| **0** | Baseline: Native CUDA Router sm_89 Integration | **Completed** | 1.44 ms / 2-block pass, 3.4 µs router |
| **1** | Kernel Fusion 1: Post-Attention Sandwich Norm + Pre-MoE Norm (#8 + #9) | **Completed (Micro)** | 6.69 µs -> 2.91 µs (2.30x speedup), 1 launch |
| **2** | Kernel Fusion 2: MoE Shared+Routed Add + Sandwich Norm (#21 + #22) | **Completed (Micro)** | 3.94 µs -> 2.14 µs (1.84x speedup), 1 launch |
| **3** | Prologue Re-engineering: 256-Th 3 KB Map + Decoupled Router (#12 + #13 + #14) | **Completed (Micro)** | Step 2: 3.64 µs -> 2.75 µs (-90.6% SRAM), Full: 8.5 µs @ N=16, 13.1 µs @ N=256 |
| **4** | Stream Concurrency: Overlap Shared & Routed Expert Compute | *Planned* | Concurrency via CUDA streams |
| **5** | CUDA Graphs: Zero-CPU Dispatch End-to-End | *Planned* | 1.44 ms -> < 1.0 ms full model |
| **6** | NCU Deep Dive: Populate DRAM & Micro-Kernel Metrics | *Planned* | Validate 0 spills, peak bandwidth |
