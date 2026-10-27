#pragma once

#include "tensorrt_llm/common/config.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cassert>

TRTLLM_NAMESPACE_BEGIN
namespace kernels
{

// =================================================================================================
// Custom Kolibri Dispatch Mapping Kernel (256 Threads, 3 KB Shared Memory)
// =================================================================================================
template <int NUM_EXPERTS = 384, int TOP_K = 6, int BLOCK_SIZE = 256>
__global__ void kolibri_build_expert_maps_kernel(
    int const* __restrict__ token_selected_experts,       // [num_tokens * TOP_K]
    int* __restrict__ permuted_row_to_unpermuted_row,     // [num_tokens * TOP_K]
    int* __restrict__ unpermuted_row_to_permuted_row,     // [num_tokens * TOP_K] (optional)
    int64_t* __restrict__ expert_first_token_offset,      // [NUM_EXPERTS + 1]
    int const num_tokens,
    int2* __restrict__ selection_scratch)                 // [num_tokens * TOP_K]
{
    assert(gridDim.x == 1);

    // Lean 3 KB shared memory footprint
    __shared__ int expert_counts[NUM_EXPERTS];
    __shared__ int expert_offsets[NUM_EXPERTS + 1];

    // Step 1: Zero out histogram bins across all 256 threads
    for (int e = threadIdx.x; e < NUM_EXPERTS; e += blockDim.x)
    {
        expert_counts[e] = 0;
    }
    __syncthreads();

    // Step 2: Cache each expert and its unique intra-expert offset while counting
    int const total_selections = num_tokens * TOP_K;
    for (int idx = threadIdx.x; idx < total_selections; idx += blockDim.x)
    {
        int const expert_id = token_selected_experts[idx];
        int local_offset = -1;
        if (expert_id >= 0 && expert_id < NUM_EXPERTS)
        {
            local_offset = atomicAdd(&expert_counts[expert_id], 1);
        }
        selection_scratch[idx] = make_int2(expert_id, local_offset);
    }
    __syncthreads();

    // Step 3: Fast exclusive prefix sum to compute starting slot for each expert
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

    // Step 4: Export starting offsets to global memory for CUTLASS Grouped GEMM
    for (int e = threadIdx.x; e < NUM_EXPERTS + 1; e += blockDim.x)
    {
        expert_first_token_offset[e] = expert_offsets[e];
    }

    // Step 5: Parallel scatter into contiguous expert partitions
    for (int idx = threadIdx.x; idx < total_selections; idx += blockDim.x)
    {
        int const token_id = idx / TOP_K;
        int const rank = idx % TOP_K;
        int2 const selection = selection_scratch[idx];
        int const expert_id = selection.x;
        int const local_offset = selection.y;

        if (local_offset >= 0)
        {
            int const dst_slot = expert_offsets[expert_id] + local_offset;
            int const unpermuted_row = rank * num_tokens + token_id;

            permuted_row_to_unpermuted_row[dst_slot] = unpermuted_row;
            if (unpermuted_row_to_permuted_row != nullptr)
            {
                unpermuted_row_to_permuted_row[unpermuted_row] = dst_slot;
            }
        }
    }
}

} // namespace kernels
TRTLLM_NAMESPACE_END
