#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cassert>
#include <algorithm>

#if __has_include("tensorrt_llm/common/config.h")
#include "tensorrt_llm/common/config.h"
#endif

#ifndef TRTLLM_NAMESPACE_BEGIN
#define TRTLLM_NAMESPACE_BEGIN namespace tensorrt_llm {
#define TRTLLM_NAMESPACE_END }
#endif

TRTLLM_NAMESPACE_BEGIN
namespace kernels
{

// =================================================================================================
// Kernel 1: Block-Independent Local Expert Counts
//
// Each block processes a contiguous chunk of token selections and tallies expert frequencies
// in fast 1.5 KB shared memory without any global atomics.
// =================================================================================================
template <int NUM_EXPERTS = 384, int BLOCK_SIZE = 1024>
__global__ void kolibri_multi_block_expert_counts_kernel(
    int const* __restrict__ token_selected_experts,       // [total_selections]
    int* __restrict__ block_expert_counts,                // [num_blocks * NUM_EXPERTS]
    int const total_selections,
    int const num_blocks)
{
    __shared__ int local_expert_counts[NUM_EXPERTS];

    // Initialize shared histogram bins
    for (int e = threadIdx.x; e < NUM_EXPERTS; e += blockDim.x)
    {
        local_expert_counts[e] = 0;
    }
    __syncthreads();

    // Contiguous chunk per block
    int const chunk_size = (total_selections + num_blocks - 1) / num_blocks;
    int const start_idx = blockIdx.x * chunk_size;
    int const end_idx = min(start_idx + chunk_size, total_selections);

    // Tally expert counts in shared memory (zero global atomics)
    for (int idx = start_idx + threadIdx.x; idx < end_idx; idx += blockDim.x)
    {
        int const expert_id = token_selected_experts[idx];
        if (expert_id >= 0 && expert_id < NUM_EXPERTS)
        {
            atomicAdd(&local_expert_counts[expert_id], 1);
        }
    }
    __syncthreads();

    // Coalesced write of block local counts to global memory
    int const block_offset = blockIdx.x * NUM_EXPERTS;
    for (int e = threadIdx.x; e < NUM_EXPERTS; e += blockDim.x)
    {
        block_expert_counts[block_offset + e] = local_expert_counts[e];
    }
}

// =================================================================================================
// Kernel 2: Prefix-Sum Across Blocks & Experts (Register-Cached)
//
// 512 threads (1 thread per expert, 1 block) compute:
// 1. Total tokens per expert across all blocks (read ONCE from global memory into registers)
// 2. Global starting offset for each expert (expert_first_token_offset)
// 3. Block-specific starting offset for each expert (read from registers, zero redundant global reads)
// All global reads and writes are 100% coalesced.
// =================================================================================================
template <int NUM_EXPERTS = 384, int MAX_BLOCKS = 128, int BLOCK_SIZE = 512>
__global__ void kolibri_multi_block_prefix_sum_kernel(
    int const* __restrict__ block_expert_counts,          // [num_blocks * NUM_EXPERTS]
    int* __restrict__ block_expert_offsets,               // [num_blocks * NUM_EXPERTS]
    int64_t* __restrict__ expert_first_token_offset,      // [NUM_EXPERTS + 1]
    int const num_blocks)
{
    static_assert(BLOCK_SIZE >= NUM_EXPERTS, "Kernel 2 block size must be >= NUM_EXPERTS");

    __shared__ int expert_totals[NUM_EXPERTS];
    __shared__ int expert_offsets[NUM_EXPERTS + 1];

    int const e = threadIdx.x;
    int local_counts[MAX_BLOCKS];

    // Step 2a: Read ONCE from global memory into private thread registers, sum totals
    if (e < NUM_EXPERTS)
    {
        int total = 0;
        for (int g = 0; g < num_blocks; ++g)
        {
            int const c = block_expert_counts[g * NUM_EXPERTS + e];
            local_counts[g] = c;
            total += c;
        }
        expert_totals[e] = total;
    }
    __syncthreads();

    // Step 2b: Thread 0 computes exclusive prefix sum across all expert totals
    if (threadIdx.x == 0)
    {
        int sum = 0;
        for (int i = 0; i < NUM_EXPERTS; ++i)
        {
            expert_offsets[i] = sum;
            sum += expert_totals[i];
        }
        expert_offsets[NUM_EXPERTS] = sum;
    }
    __syncthreads();

    // Step 2c: Export global starting offsets for CUTLASS Grouped GEMM
    if (threadIdx.x < NUM_EXPERTS + 1)
    {
        expert_first_token_offset[threadIdx.x] = static_cast<int64_t>(expert_offsets[threadIdx.x]);
    }

    // Step 2d: Compute per-block base offset from registers (ZERO redundant global reads)
    if (e < NUM_EXPERTS)
    {
        int running_offset = expert_offsets[e];
        for (int g = 0; g < num_blocks; ++g)
        {
            block_expert_offsets[g * NUM_EXPERTS + e] = running_offset;
            running_offset += local_counts[g]; // Fast register read
        }
    }
}

// =================================================================================================
// Kernel 3: Multi-Block Scatter Directly into Final Global Mapping Arrays
//
// Every block independently loads its base offsets, calculates local intra-block offsets in
// shared memory, and writes directly into permuted_row_to_unpermuted_row and
// unpermuted_row_to_permuted_row without global atomics or second passes.
// =================================================================================================
template <int NUM_EXPERTS = 384, int TOP_K = 6, int BLOCK_SIZE = 256>
__global__ void kolibri_multi_block_scatter_maps_kernel(
    int const* __restrict__ token_selected_experts,       // [total_selections]
    int const* __restrict__ block_expert_offsets,         // [num_blocks * NUM_EXPERTS]
    int* __restrict__ permuted_row_to_unpermuted_row,     // [total_selections]
    int* __restrict__ unpermuted_row_to_permuted_row,     // [total_selections] (optional)
    int const total_selections,
    int const num_tokens,
    int const num_blocks)
{
    __shared__ int local_expert_base_offsets[NUM_EXPERTS];
    __shared__ int local_expert_counters[NUM_EXPERTS];

    // Load this block's expert base offsets and reset local counters
    int const block_offset = blockIdx.x * NUM_EXPERTS;
    for (int e = threadIdx.x; e < NUM_EXPERTS; e += blockDim.x)
    {
        local_expert_base_offsets[e] = block_expert_offsets[block_offset + e];
        local_expert_counters[e] = 0;
    }
    __syncthreads();

    // Contiguous chunk matching Kernel 1
    int const chunk_size = (total_selections + num_blocks - 1) / num_blocks;
    int const start_idx = blockIdx.x * chunk_size;
    int const end_idx = min(start_idx + chunk_size, total_selections);

    for (int idx = start_idx + threadIdx.x; idx < end_idx; idx += blockDim.x)
    {
        int const expert_id = token_selected_experts[idx];
        if (expert_id >= 0 && expert_id < NUM_EXPERTS)
        {
            // Intra-block rank via shared memory atomic
            int const local_offset = atomicAdd(&local_expert_counters[expert_id], 1);
            int const dst_slot = local_expert_base_offsets[expert_id] + local_offset;

            int const token_id = idx / TOP_K;
            int const rank = idx % TOP_K;
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
