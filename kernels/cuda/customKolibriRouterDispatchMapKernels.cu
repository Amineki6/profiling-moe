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
#include <cstdint>
#include <cassert>
#include <stdexcept>
#include <string>

namespace cg = cooperative_groups;

TRTLLM_NAMESPACE_BEGIN
namespace kernels
{

static constexpr int WARP_SIZE = 32;
static constexpr int KOLIBRI_NUM_EXPERTS = 384;
static constexpr int KOLIBRI_TOP_K = 6;

// =================================================================================================
// KERNEL 1: Fused Kolibri Router + Dispatch Map (Single Block)
//
// Computes top-6 routing and builds the dispatch permutation mapping in shared memory.
// Uses a 384-bin shared memory histogram and prefix sum (~3 KB), completely avoiding CUB overhead.
// =================================================================================================
template <int NUM_EXPERTS = 384, int TOP_K = 6, int BLOCK_SIZE = 256>
__global__ void kolibri_router_dispatch_map_kernel(
    float const* __restrict__ scores_in,                // [num_tokens, NUM_EXPERTS]
    float const* __restrict__ bias,                     // [NUM_EXPERTS]
    float* __restrict__ permuted_weights,                // [num_tokens * TOP_K]
    int* __restrict__ permuted_row_to_unpermuted_row,   // [num_tokens * TOP_K]
    int* __restrict__ permuted_token_ids,               // [num_tokens * TOP_K]
    int64_t* __restrict__ expert_first_token_offset,    // [NUM_EXPERTS + 1]
    int const num_tokens)
{
    // Single block constraint for block-wide shared histogram & prefix sum
    assert(gridDim.x == 1);

    constexpr int kExpertsPerThread = NUM_EXPERTS / WARP_SIZE;
    constexpr int kWarpsPerBlock = BLOCK_SIZE / WARP_SIZE;

    // Shared memory staging buffers (~27 KB total, comfortably within 48 KB limit)
    __shared__ float smem_scores[kWarpsPerBlock][NUM_EXPERTS];
    __shared__ float smem_weights[TOP_K * BLOCK_SIZE];
    __shared__ int smem_indices[TOP_K * BLOCK_SIZE];

    __shared__ int expert_counts[NUM_EXPERTS];
    __shared__ int expert_offsets[NUM_EXPERTS + 1];

    // Clear histogram bins across the block
    for (int e = threadIdx.x; e < NUM_EXPERTS; e += blockDim.x)
    {
        expert_counts[e] = 0;
    }
    __syncthreads();

    int const global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int const local_warp_id  = (threadIdx.x / WARP_SIZE) % kWarpsPerBlock;
    int const lane_id        = threadIdx.x % WARP_SIZE;

    auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
    float* my_smem = smem_scores[local_warp_id];

    float biased_scores[kExpertsPerThread];
    int indices[kExpertsPerThread];

    // Router stage: each warp processes tokens with a warp stride across the block
    #pragma unroll
    for (int token_id = global_warp_id; token_id < num_tokens; token_id += kWarpsPerBlock)
    {
        float const* scores_row = scores_in + token_id * NUM_EXPERTS;

        // Load raw routing logits, apply expert bias, and stage un-biased score in shared memory
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

        // Warp-level parallel top-k reduction across all expert candidates
        float topk_values[TOP_K];
        int32_t topk_indices[TOP_K];

        constexpr float minValue = -1e30f;
        reduce_topk::reduceTopK<TOP_K, float, kExpertsPerThread>(
            warp, topk_values, topk_indices, biased_scores, indices, minValue, TOP_K);

        // Compute sigmoid activation on raw un-biased score, store in shared memory, and tally histogram
        if (lane_id < TOP_K)
        {
            int expert_id = topk_indices[lane_id];
            float raw_val = my_smem[expert_id];
            float s = 1.0f / (1.0f + __expf(-raw_val));

            smem_weights[token_id * TOP_K + lane_id] = s;
            smem_indices[token_id * TOP_K + lane_id] = expert_id;
            atomicAdd(&expert_counts[expert_id], 1);
        }
    }

    // Synchronize so all warps complete their token routing and histogram counts
    __syncthreads();

    // Exclusive prefix sum over expert counts to determine starting offset for each expert
    if (threadIdx.x == 0)
    {
        int sum = 0;
        for (int e = 0; e < NUM_EXPERTS; ++e)
        {
            expert_offsets[e] = sum;
            sum += expert_counts[e];
        }
        expert_offsets[NUM_EXPERTS] = sum;
    }
    __syncthreads();

    // Export initial expert offsets to global memory for downstream CUTLASS Grouped GEMM
    for (int e = threadIdx.x; e < NUM_EXPERTS + 1; e += blockDim.x)
    {
        expert_first_token_offset[e] = expert_offsets[e];
    }

    // Scatter tokens into contiguous expert slots using atomic increment on the offset
    for (int idx = threadIdx.x; idx < num_tokens * TOP_K; idx += blockDim.x)
    {
        int const token_id = idx / TOP_K;
        int const rank = idx % TOP_K;
        int const expert_id = smem_indices[idx];
        float const weight = smem_weights[idx];

        int const dst_slot = atomicAdd(&expert_offsets[expert_id], 1);

        permuted_token_ids[dst_slot] = token_id;
        permuted_weights[dst_slot] = weight;
        permuted_row_to_unpermuted_row[dst_slot] = rank * num_tokens + token_id;
    }
}

void invokeKolibriRouterDispatchMap(
    float const* scores_in,
    float const* bias,
    float* permuted_weights,
    int* permuted_row_to_unpermuted_row,
    int* permuted_token_ids,
    int64_t* expert_first_token_offset,
    int const num_tokens,
    int const num_experts,
    int const top_k,
    cudaStream_t stream)
{
    if (num_experts != KOLIBRI_NUM_EXPERTS || top_k != KOLIBRI_TOP_K)
    {
        throw std::runtime_error("Kolibri router dispatch map kernel requires num_experts=384 and top_k=6");
    }

    constexpr int ROUTER_BLOCK_SIZE = 256;
    dim3 const router_grid(1);
    dim3 const router_block(ROUTER_BLOCK_SIZE);

    kolibri_router_dispatch_map_kernel<KOLIBRI_NUM_EXPERTS, KOLIBRI_TOP_K, ROUTER_BLOCK_SIZE>
        <<<router_grid, router_block, 0, stream>>>(
            scores_in,
            bias,
            permuted_weights,
            permuted_row_to_unpermuted_row,
            permuted_token_ids,
            expert_first_token_offset,
            num_tokens);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("Router dispatch map kernel launch failed: ") + cudaGetErrorString(err));
    }
}

} // namespace kernels
TRTLLM_NAMESPACE_END
