/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "customKolibriRouterDispatchKernels.h"
#include "tensorrt_llm/kernels/moeTopKFuncs.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace cg = cooperative_groups;

TRTLLM_NAMESPACE_BEGIN
namespace kernels
{

#undef TLLM_CHECK_WITH_INFO
#define TLLM_CHECK_WITH_INFO(cond, msg) \
    do { if (!(cond)) throw std::runtime_error(std::string("Assertion failed: ") + (msg)); } while(0)

static constexpr int WARP_SIZE = 32;
static constexpr int KOLIBRI_NUM_EXPERTS = 384;
static constexpr int KOLIBRI_TOP_K = 6;
static constexpr int LOG2_NUM_EXPERTS = 9; // 2^9 = 512 >= 384

// =================================================================================================
// Fused Kolibri Router + Expert Dispatch Mapping Kernel Skeleton
// =================================================================================================
template <typename T, int NUM_EXPERTS = 384, int TOP_K = 6, int BLOCK_SIZE = 256>
__global__ void fused_kolibri_router_dispatch_kernel(
    T const* __restrict__ unpermuted_input,             // [num_tokens, hidden_size]
    float const* __restrict__ scores_in,                // [num_tokens, NUM_EXPERTS]
    float const* __restrict__ bias,                     // [NUM_EXPERTS]
    T* __restrict__ permuted_output,                    // [num_tokens * TOP_K, hidden_size]
    float* __restrict__ permuted_scales,                // [num_tokens * TOP_K]
    int* __restrict__ permuted_row_to_unpermuted_row,   // [num_tokens * TOP_K]
    int64_t* __restrict__ expert_first_token_offset,    // [NUM_EXPERTS + 1]
    int const num_tokens,
    int const hidden_size)
{
    // Single block constraint for CUB collective
    assert(gridDim.x == 1);

    constexpr int kExpertsPerThread = nExperts / WARP_SIZE; // 384 / 32 = 12
    constexpr int kWarpsPerBlock = 8;

    __shared__ float smem_scores[kWarpsPerBlock][nExperts];
    __shared__ float smem_weights[TOP_K * BLOCK_SIZE];
    __shared__ float smem_indicies[TOP_K * BLOCK_SIZE];


    int const global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int const local_warp_id  = (threadIdx.x / WARP_SIZE) % kWarpsPerBlock;
    int const lane_id        = threadIdx.x % WARP_SIZE;

    if (global_warp_id >= batch_size)
        return;

    auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
    float* my_smem = smem_scores[local_warp_id];
    

    float biased_scores[kExpertsPerThread];
    int indices[kExpertsPerThread];

#pragma unroll
    for (int token_id = global_warp_id; token_id < num_tokens; token_id += kWarpsPerBlock)
    {
        // Load logits, add bias, and store un-biased score to shared memory

        float const* scores_row = scores_in + token_id * nExperts;

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
            
            smem_weights[token_id * topk + lane_id] = s;
            smem_indices[token_id * topk + lane_id] = expert_id;
        }
    
    }

    __syncthreads();

    int my_experts[TOP_K];
    int local_token_permuted_indices[TOP_K];

    // If valid experts are in [0, NUM_EXPERTS - 1]:
    constexpr int DUMMY_EXPERT = NUM_EXPERTS;

#pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
        my_experts[k] = (threadIdx.x < num_tokens)
            ? smem_indices[threadIdx.x * TOP_K + k]
            : DUMMY_EXPERT; // Sentinel bucket beyond all valid experts
    }

    // 2. Sort expert map with CUB BlockRadixRank
    using BlockRadixRank = cub::BlockRadixRank<BLOCK_SIZE, LOG2_NUM_EXPERTS, false>;
    extern __shared__ unsigned char temp_storage[];
    auto& sort_temp = *reinterpret_cast<typename BlockRadixRank::TempStorage*>(temp_storage);

    int local_expert_first_token_offset[BlockRadixRank::BINS_TRACKED_PER_THREAD];
    cub::BFEDigitExtractor<int> extractor(0, LOG2_NUM_EXPERTS);

    BlockRadixRank(sort_temp).RankKeys(
        my_experts,
        local_token_permuted_indices,
        extractor,
        local_expert_first_token_offset);


    // =============================================================================================
    // [STEP 1]: ROUTER STAGE (Top-K selection per token warp)
    // 
    // Reference: kernels/cuda/customKolibriRoutingKernels.cu (gate_forward_kernel_vB)
    // Goal:
    //   - Each warp handles 1 token (or token stride).
    //   - Read logits, add bias, find top-6 experts using reduceTopK.
    //   - Apply sigmoid: s = 1.0f / (1.0f + __expf(-raw_score)).
    //   - Keep (topk_indices, topk_weights) in shared memory or registers instead of writing to DRAM!
    // =============================================================================================

    // TODO: Write Step 1 here
    // Hint: define shared memory for scores if needed, or coordinate across warps.


    // =============================================================================================
    // [STEP 2]: DISPATCH / SORTING STAGE (CUB Radix Rank across tokens)
    //
    // Reference: moe_kernels.cu (fusedBuildExpertMapsSortFirstTokenKernel)
    // Goal:
    //   - Rank the selected expert IDs using cub::BlockRadixRank<BLOCK_SIZE, LOG2_NUM_EXPERTS>.
    //   - Output the histogram offsets: expert_first_token_offset[0 .. NUM_EXPERTS].
    //   - Output the row permutation table: permuted_row_to_unpermuted_row[permuted_row].
    // =============================================================================================

    // TODO: Write Step 2 here
    // Hint: BlockRadixRank requires TempStorage in shared memory:
    //   using BlockRadixRank = cub::BlockRadixRank<BLOCK_SIZE, LOG2_NUM_EXPERTS, false>;
    //   __shared__ typename BlockRadixRank::TempStorage sort_temp;


    // =============================================================================================
    // [STEP 3]: ACTIVATION EXPANSION STAGE (Row permutation + 128-bit vector copy)
    //
    // Reference: moe_kernels.cu (expandInputRowsKernel)
    // Goal:
    //   - Copy rows from unpermuted_input to permuted_output based on permuted_row_to_unpermuted_row.
    //   - Also assign the routing scale for each permuted row: permuted_scales[permuted_row].
    //   - Use 128-bit vectorized transfers (uint4 = 8 x FP16/BF16) for maximum memory bandwidth.
    // =============================================================================================

    // TODO: Write Step 3 here
}


// =================================================================================================
// Host Launcher Template
// =================================================================================================
template <typename T>
void invokeFusedKolibriRouterDispatch(
    T const* unpermuted_input,
    float const* scores_in,
    float const* bias,
    T* permuted_output,
    float* permuted_scales,
    int* permuted_row_to_unpermuted_row,
    int64_t* expert_first_token_offset,
    int const num_tokens,
    int const hidden_size,
    int const num_experts,
    int const top_k,
    cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(num_experts == KOLIBRI_NUM_EXPERTS, "num_experts must be 384 for Kolibri");
    TLLM_CHECK_WITH_INFO(top_k == KOLIBRI_TOP_K, "top_k must be 6 for Kolibri");

    // Launch configuration:
    // For decode tokens (N <= 256), a single cooperative block or small grid can be used.
    constexpr int BLOCK_SIZE = 256;
    dim3 const grid(1);
    dim3 const block(BLOCK_SIZE);

    // TODO: Add dynamic shared memory calculation if needed, then launch:
    fused_kolibri_router_dispatch_kernel<T, KOLIBRI_NUM_EXPERTS, KOLIBRI_TOP_K, BLOCK_SIZE>
        <<<grid, block, 0, stream>>>(
            unpermuted_input,
            scores_in,
            bias,
            permuted_output,
            permuted_scales,
            permuted_row_to_unpermuted_row,
            expert_first_token_offset,
            num_tokens,
            hidden_size);
}

// Template explicit instantiations for half and nv_bfloat16
template void invokeFusedKolibriRouterDispatch<half>(
    half const* unpermuted_input,
    float const* scores_in,
    float const* bias,
    half* permuted_output,
    float* permuted_scales,
    int* permuted_row_to_unpermuted_row,
    int64_t* expert_first_token_offset,
    int const num_tokens,
    int const hidden_size,
    int const num_experts,
    int const top_k,
    cudaStream_t stream);

#ifdef ENABLE_BF16
template void invokeFusedKolibriRouterDispatch<__nv_bfloat16>(
    __nv_bfloat16 const* unpermuted_input,
    float const* scores_in,
    float const* bias,
    __nv_bfloat16* permuted_output,
    float* permuted_scales,
    int* permuted_row_to_unpermuted_row,
    int64_t* expert_first_token_offset,
    int const num_tokens,
    int const hidden_size,
    int const num_experts,
    int const top_k,
    cudaStream_t stream);
#endif

} // namespace kernels
TRTLLM_NAMESPACE_END
