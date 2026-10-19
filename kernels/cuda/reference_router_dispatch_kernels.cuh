#pragma once

/**
 * =================================================================================================
 * REFERENCE KERNELS FOR CANDIDATE 3: KOLIBRI ROUTER + CUTLASS EXPERT DISPATCH FUSION
 * =================================================================================================
 * 
 * Pipeline in TensorRT-LLM MoE execution:
 * 
 *   [Hidden Activations x: (N, 2560)]
 *                 |
 *                 +-----------------------------------------------------+
 *                 |                                                     |
 *                 v                                                     |
 *       (GEMM: Router FC)                                               |
 *                 |                                                     |
 *                 v                                                     |
 *         [scores: (N, 384)]                                            |
 *                 |                                                     |
 *                 v                                                     |
 *   +-----------------------------+                                     |
 *   | KERNEL 1: Router Top-K      | <--- gate_forward_kernel_vB          |
 *   +-----------------------------+                                     |
 *      |                     |                                          |
 *      v                     v                                          |
 *   [out_indices: (N, 6)]   [out_weights: (N, 6)]                       |
 *      |                                                                |
 *      v                                                                |
 *   +---------------------------------------+                           |
 *   | KERNEL 2: Expert Map Sorting / Offsets|                           |
 *   |           (cub::BlockRadixRank)       | <--- fusedBuildExpertMapsSortFirstTokenKernel
 *   +---------------------------------------+                           |
 *      |                        |                                       |
 *      v                        v                                       |
 *   [permuted_row_to_        [expert_first_token_offset: (385,)]        |
 *    unpermuted_row: (6*N,)]    |                                       |
 *      |                        |                                       |
 *      +------------------------+---------------------------------------+
 *                               |
 *                               v
 *                 +-----------------------------+
 *                 | KERNEL 3: Input Expansion   | <--- expandInputRowsKernel
 *                 +-----------------------------+
 *                               |
 *              +----------------+----------------+
 *              |                                 |
 *              v                                 v
 *   [permuted_output: (6*N, 2560)]    [permuted_scales: (6*N,)]
 *              |                                 |
 *              +----------------+----------------+
 *                               |
 *                               v
 *                 [CUTLASS Grouped GEMM 1 (FC1)]
 * =================================================================================================
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cstdint>
#include <cassert>

namespace cg = cooperative_groups;

namespace kolibri_moe_reference
{

// =================================================================================================
// 1. KERNEL #12: KOLIBRI ROUTER TOP-K (Variant B)
// Source: kernels/cuda/customKolibriRoutingKernels.cu
//
// Description:
//   - Each warp processes 1 token.
//   - Each thread in the warp handles 384 / 32 = 12 experts.
//   - Adds bias to raw routing logits, tracks top-6 using warp-level bitonic/radix reduction.
//   - Computes sigmoid on the raw (unbiased) score for the selected top-6 experts.
//   - Writes:
//       out_indices: [N, 6]  (expert IDs for each token)
//       out_weights: [N, 6]  (gating scale factors)
// =================================================================================================

static constexpr int WARP_SIZE = 32;

template <int nExperts = 384, int topK = 6>
__global__ void gate_forward_kernel_vB_reference(
    float const* __restrict__ scores_in,  // [N, nExperts] raw router logits
    float const* __restrict__ bias,       // [nExperts] bias terms
    float* __restrict__ out_weights,      // [N, topK] normalized routing weights
    int* __restrict__ out_indices,        // [N, topK] chosen expert IDs
    int batch_size)
{
    constexpr int kExpertsPerThread = nExperts / WARP_SIZE; // 384 / 32 = 12
    constexpr int kWarpsPerBlock = 8;

    __shared__ float smem_scores[kWarpsPerBlock][nExperts];

    int const global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int const local_warp_id  = (threadIdx.x / WARP_SIZE) % kWarpsPerBlock;
    int const lane_id        = threadIdx.x % WARP_SIZE;

    if (global_warp_id >= batch_size)
        return;

    auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
    float* my_smem = smem_scores[local_warp_id];
    float const* scores_row = scores_in + global_warp_id * nExperts;

    float biased_scores[kExpertsPerThread];
    int indices[kExpertsPerThread];

    // Load logits, add bias, and store un-biased score to shared memory
    #pragma unroll
    for (int e = 0; e < kExpertsPerThread; ++e)
    {
        int expert_id = lane_id + e * WARP_SIZE;
        float raw_score = scores_row[expert_id];
        biased_scores[e] = raw_score + bias[expert_id];
        indices[e] = expert_id;
        my_smem[expert_id] = raw_score;
    }
    __syncwarp();

    // Top-K reduction (using warp shuffles)
    float topk_values[topK];
    int32_t topk_indices[topK];
    // Note: in actual implementation, calls reduce_topk::reduceTopK(...)

    // Lane 0..topK-1 write results to global memory
    if (lane_id < topK)
    {
        int expert_id = topk_indices[lane_id];
        float raw_val = my_smem[expert_id];
        float s = 1.0f / (1.0f + __expf(-raw_val)); // Sigmoid on raw score

        out_weights[global_warp_id * topK + lane_id] = s;
        out_indices[global_warp_id * topK + lane_id] = expert_id;
    }
}


// =================================================================================================
// 2. KERNEL #13: CUTLASS EXPERT MAP BUILD & SORT
// Source: TensorRT-LLM/cpp/tensorrt_llm/kernels/moe/cutlass/moe_kernels.cu (lines 339-430)
//
// Description:
//   - Single block kernel (gridDim.x == 1) for decode token counts (N <= 256).
//   - Uses cub::BlockRadixRank to sort token expert assignments into bins (experts 0..383).
//   - Writes:
//       permuted_row_to_unpermuted_row: [N * 6] -> maps sorted slot -> original (token, topk_rank)
//       unpermuted_row_to_permuted_row: [N * 6] -> reverse map
//       expert_first_token_offset:      [385]   -> offset where each expert's tokens start
// =================================================================================================

template <int BLOCK_SIZE, int EXPERTS_PER_TOKEN = 6, int LOG2_NUM_EXPERTS = 9>
__global__ void fusedBuildExpertMapsSortFirstTokenKernel_reference(
    int const* const token_selected_experts,      // [N, EXPERTS_PER_TOKEN] (out_indices from router)
    int* const permuted_row_to_unpermuted_row,    // [N * EXPERTS_PER_TOKEN]
    int* const unpermuted_row_to_permuted_row,    // [N * EXPERTS_PER_TOKEN]
    int64_t* const expert_first_token_offset,     // [num_experts_per_node + 1]
    int64_t const num_tokens,
    int const experts_per_token,
    int const start_expert,
    int const end_expert,
    int const num_experts_per_node)               // 384 for Kolibri
{
    // Single block constraint for CUB collective
    assert(gridDim.x == 1);

    int const token = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    bool is_valid_token = token < num_tokens;

    int local_token_selected_experts[EXPERTS_PER_TOKEN];
    int local_token_permuted_indices[EXPERTS_PER_TOKEN];

    // 1. Gather expert IDs per token
    #pragma unroll
    for (int i = 0; i < EXPERTS_PER_TOKEN; i++)
    {
        int const expert = is_valid_token
            ? token_selected_experts[token * EXPERTS_PER_TOKEN + i]
            : num_experts_per_node;

        bool is_valid_expert = (expert >= start_expert && expert < end_expert);
        local_token_selected_experts[i] = !is_valid_token ? num_experts_per_node + 1
            : is_valid_expert                             ? (expert - start_expert)
                                                          : num_experts_per_node;
    }

    // 2. Sort expert map with CUB BlockRadixRank
    using BlockRadixRank = cub::BlockRadixRank<BLOCK_SIZE, LOG2_NUM_EXPERTS, false>;
    extern __shared__ unsigned char temp_storage[];
    auto& sort_temp = *reinterpret_cast<typename BlockRadixRank::TempStorage*>(temp_storage);

    int local_expert_first_token_offset[BlockRadixRank::BINS_TRACKED_PER_THREAD];
    cub::BFEDigitExtractor<int> extractor(0, LOG2_NUM_EXPERTS);

    BlockRadixRank(sort_temp).RankKeys(
        local_token_selected_experts,
        local_token_permuted_indices,
        extractor,
        local_expert_first_token_offset);

    // 3. Write permutation tables
    if (is_valid_token)
    {
        #pragma unroll
        for (int i = 0; i < EXPERTS_PER_TOKEN; i++)
        {
            int const unpermuted_row = i * num_tokens + token;
            int const permuted_row = local_token_permuted_indices[i];
            permuted_row_to_unpermuted_row[permuted_row] = unpermuted_row;
            unpermuted_row_to_permuted_row[unpermuted_row] = permuted_row;
        }
    }

    // 4. Write cumulative token offsets per expert
    #pragma unroll
    for (int expert_id = 0; expert_id < BlockRadixRank::BINS_TRACKED_PER_THREAD; expert_id++)
    {
        int out_expert_id = expert_id + token * BlockRadixRank::BINS_TRACKED_PER_THREAD;
        if (out_expert_id < num_experts_per_node + 1)
        {
            expert_first_token_offset[out_expert_id] = local_expert_first_token_offset[expert_id];
        }
    }
}


// =================================================================================================
// 3. KERNEL #14: ACTIVATION EXPANSION & PERMUTATION (Clean FP16/BF16 Path)
// Source: TensorRT-LLM/cpp/tensorrt_llm/kernels/moe/cutlass/moe_kernels.cu (lines 1393-1555)
//
// Description:
//   - Permutes and duplicates rows of input activations `x` (shape [N, 2560]) according to
//     the expert assignment order in `permuted_row_to_unpermuted_row`.
//   - Also permutes the routing scales `unpermuted_scales` into `permuted_scales`.
//   - Output tensor `permuted_output` (shape [6 * N, 2560]) is the contiguous input to FC1 GEMM.
// =================================================================================================

constexpr static int EXPAND_THREADS_PER_BLOCK = 256;

template <typename T> // half or __nv_bfloat16
__global__ void expandInputRowsKernel_reference(
    T const* __restrict__ unpermuted_input,             // [N, hidden_size] (hidden activations)
    T* __restrict__ permuted_output,                     // [6 * N, hidden_size]
    float const* __restrict__ unpermuted_scales,         // [N * 6] (out_weights from router)
    float* __restrict__ permuted_scales,                 // [6 * N]
    int const* __restrict__ permuted_row_to_unpermuted_row, // [6 * N]
    int64_t const num_tokens,
    int64_t const hidden_size,                           // 2560
    int64_t const k,                                     // 6
    int64_t const* __restrict__ expert_first_token_offset,
    int64_t const num_experts_per_node)                  // 384
{
    // Total number of valid expanded rows across all experts
    int64_t const num_valid_tokens = expert_first_token_offset[num_experts_per_node]; // 6 * N

    // Each block processes one permuted row (or strides over permuted rows)
    for (int64_t permuted_row = blockIdx.x; permuted_row < num_valid_tokens; permuted_row += gridDim.x)
    {
        // 1. Identify which original token and rank this permuted row corresponds to
        int64_t const unpermuted_row = permuted_row_to_unpermuted_row[permuted_row];
        int64_t const source_k_rank = unpermuted_row / num_tokens;
        int64_t const source_row    = unpermuted_row % num_tokens;

        // 2. 128-bit vectorized row copy: 8 elements of FP16/BF16 per 16-byte transfer
        using VecType = uint4; // 16 bytes = 8 x 2-byte elements
        constexpr int64_t ELEMS_PER_VEC = sizeof(VecType) / sizeof(T); // 8

        VecType const* src_vec = reinterpret_cast<VecType const*>(unpermuted_input + source_row * hidden_size);
        VecType* dst_vec       = reinterpret_cast<VecType*>(permuted_output + permuted_row * hidden_size);

        int64_t const num_vecs = hidden_size / ELEMS_PER_VEC; // 2560 / 8 = 320 vectors

        for (int64_t col = threadIdx.x; col < num_vecs; col += blockDim.x)
        {
            dst_vec[col] = src_vec[col];
        }

        // 3. Thread 0 copies the routing scale for this expanded token
        if (permuted_scales && threadIdx.x == 0)
        {
            int64_t const source_k_idx = source_row * k + source_k_rank;
            permuted_scales[permuted_row] = unpermuted_scales ? unpermuted_scales[source_k_idx] : 1.0f;
        }
    }
}

} // namespace kolibri_moe_reference
