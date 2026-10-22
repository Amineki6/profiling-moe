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

#include "customKolibriBuildExpertMapsKernel.h"
#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <string>

TRTLLM_NAMESPACE_BEGIN
namespace kernels
{

static constexpr int KOLIBRI_NUM_EXPERTS = 384;
static constexpr int KOLIBRI_TOP_K = 6;

// =================================================================================================
// 1. Custom Kolibri Dispatch Mapping Kernel (256 Threads, 3 KB Shared Memory)
// =================================================================================================
template <int NUM_EXPERTS = 384, int TOP_K = 6, int BLOCK_SIZE = 256>
__global__ void kolibri_build_expert_maps_kernel(
    int const* __restrict__ token_selected_experts,       // [num_tokens * TOP_K]
    int* __restrict__ permuted_row_to_unpermuted_row,     // [num_tokens * TOP_K]
    int* __restrict__ unpermuted_row_to_permuted_row,     // [num_tokens * TOP_K] (optional)
    int64_t* __restrict__ expert_first_token_offset,      // [NUM_EXPERTS + 1]
    int const num_tokens)
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

    // Step 2: 256 threads cooperatively read token expert selections and update histogram
    int const total_selections = num_tokens * TOP_K;
    for (int idx = threadIdx.x; idx < total_selections; idx += blockDim.x)
    {
        int const expert_id = token_selected_experts[idx];
        if (expert_id >= 0 && expert_id < NUM_EXPERTS)
        {
            atomicAdd(&expert_counts[expert_id], 1);
        }
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
        int const expert_id = token_selected_experts[idx];

        if (expert_id >= 0 && expert_id < NUM_EXPERTS)
        {
            int const dst_slot = atomicAdd(&expert_offsets[expert_id], 1);
            int const unpermuted_row = rank * num_tokens + token_id;

            permuted_row_to_unpermuted_row[dst_slot] = unpermuted_row;
            if (unpermuted_row_to_permuted_row != nullptr)
            {
                unpermuted_row_to_permuted_row[unpermuted_row] = dst_slot;
            }
        }
    }
}

void invokeKolibriBuildExpertMaps(
    int const* token_selected_experts,
    int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row,
    int64_t* expert_first_token_offset,
    int const num_tokens,
    int const num_experts,
    int const top_k,
    cudaStream_t stream)
{
    if (num_experts != KOLIBRI_NUM_EXPERTS || top_k != KOLIBRI_TOP_K)
    {
        throw std::runtime_error("Kolibri build expert maps requires num_experts=384 and top_k=6");
    }

    constexpr int BLOCK_SIZE = 256;
    dim3 const grid(1);
    dim3 const block(BLOCK_SIZE);

    kolibri_build_expert_maps_kernel<KOLIBRI_NUM_EXPERTS, KOLIBRI_TOP_K, BLOCK_SIZE>
        <<<grid, block, 0, stream>>>(
            token_selected_experts,
            permuted_row_to_unpermuted_row,
            unpermuted_row_to_permuted_row,
            expert_first_token_offset,
            num_tokens);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("invokeKolibriBuildExpertMaps failed: ") + cudaGetErrorString(err));
    }
}


// =================================================================================================
// 2. TRT-LLM Baseline: CUB BlockRadixRank<32, 9> (Single Warp, 32 Threads)
// =================================================================================================
template <int BLOCK_SIZE = 32, int EXPERTS_PER_TOKEN = 6, int LOG2_NUM_EXPERTS = 9>
__global__ void trtllm_build_expert_maps_b32_kernel(
    int const* const token_selected_experts,
    int* const permuted_row_to_unpermuted_row,
    int* const unpermuted_row_to_permuted_row,
    int64_t* const expert_first_token_offset,
    int64_t const num_tokens,
    int const num_experts)
{
    assert(gridDim.x == 1);

    int const token = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    bool const is_valid_token = token < num_tokens;

    int local_token_selected_experts[EXPERTS_PER_TOKEN];
    int local_token_permuted_indices[EXPERTS_PER_TOKEN];

    #pragma unroll
    for (int i = 0; i < EXPERTS_PER_TOKEN; i++)
    {
        int const expert = is_valid_token ? token_selected_experts[token * EXPERTS_PER_TOKEN + i] : num_experts;
        bool const is_valid_expert = (expert >= 0 && expert < num_experts);
        local_token_selected_experts[i] = !is_valid_token ? (num_experts + 1)
            : is_valid_expert                             ? expert
                                                          : num_experts;
    }

    using BlockRadixRank = cub::BlockRadixRank<BLOCK_SIZE, LOG2_NUM_EXPERTS, false>;
    extern __shared__ unsigned char temp_storage[];
    auto& sort_temp = *reinterpret_cast<typename BlockRadixRank::TempStorage*>(temp_storage);

    int local_expert_first_token_offset[BlockRadixRank::BINS_TRACKED_PER_THREAD];
    cub::BFEDigitExtractor<int> extractor(0, LOG2_NUM_EXPERTS);

    BlockRadixRank(sort_temp).RankKeys(
        local_token_selected_experts, local_token_permuted_indices, extractor, local_expert_first_token_offset);

    if (is_valid_token)
    {
        #pragma unroll
        for (int i = 0; i < EXPERTS_PER_TOKEN; i++)
        {
            int const unpermuted_row = i * num_tokens + token;
            int const permuted_row = local_token_permuted_indices[i];
            permuted_row_to_unpermuted_row[permuted_row] = unpermuted_row;
            if (unpermuted_row_to_permuted_row != nullptr)
            {
                unpermuted_row_to_permuted_row[unpermuted_row] = permuted_row;
            }
        }
    }

    #pragma unroll
    for (int expert_id = 0; expert_id < BlockRadixRank::BINS_TRACKED_PER_THREAD; expert_id++)
    {
        int out_expert_id = expert_id + token * BlockRadixRank::BINS_TRACKED_PER_THREAD;
        if (out_expert_id < num_experts + 1)
        {
            expert_first_token_offset[out_expert_id] = local_expert_first_token_offset[expert_id];
        }
    }
}

void invokeTrtllmBuildExpertMapsB32(
    int const* token_selected_experts,
    int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row,
    int64_t* expert_first_token_offset,
    int const num_tokens,
    int const num_experts,
    int const top_k,
    cudaStream_t stream)
{
    constexpr int BLOCK_SIZE = 32;
    constexpr int LOG2_NUM_EXPERTS = 9; // 2^9 = 512 >= 384
    using BlockRadixRank = cub::BlockRadixRank<BLOCK_SIZE, LOG2_NUM_EXPERTS, false>;
    size_t const shared_size = sizeof(typename BlockRadixRank::TempStorage);

    dim3 const grid(1);
    dim3 const block(BLOCK_SIZE);

    trtllm_build_expert_maps_b32_kernel<BLOCK_SIZE, 6, LOG2_NUM_EXPERTS>
        <<<grid, block, shared_size, stream>>>(
            token_selected_experts,
            permuted_row_to_unpermuted_row,
            unpermuted_row_to_permuted_row,
            expert_first_token_offset,
            num_tokens,
            num_experts);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("invokeTrtllmBuildExpertMapsB32 failed: ") + cudaGetErrorString(err));
    }
}


// =================================================================================================
// 3. Step 3: Multi-Block Vectorized Activation Row Expansion From Map
// =================================================================================================
template <typename T>
__global__ void kolibri_activation_expansion_from_map_kernel(
    T const* __restrict__ unpermuted_input,             // [num_tokens, hidden_size]
    T* __restrict__ permuted_output,                    // [num_expanded_rows, hidden_size]
    int const* __restrict__ permuted_row_to_unpermuted, // [num_expanded_rows]
    int const num_tokens,
    int const num_expanded_rows,
    int const hidden_size)
{
    using VecType = uint4;
    constexpr int ELEMS_PER_VEC = sizeof(VecType) / sizeof(T);
    int const num_vecs = hidden_size / ELEMS_PER_VEC;

    for (int dst_row = blockIdx.x; dst_row < num_expanded_rows; dst_row += gridDim.x)
    {
        int const unpermuted_row = permuted_row_to_unpermuted[dst_row];
        int const src_token_id   = unpermuted_row % num_tokens;

        VecType const* src_vec = reinterpret_cast<VecType const*>(unpermuted_input + src_token_id * hidden_size);
        VecType* dst_vec       = reinterpret_cast<VecType*>(permuted_output + dst_row * hidden_size);

        for (int i = threadIdx.x; i < num_vecs; i += blockDim.x)
        {
            dst_vec[i] = src_vec[i];
        }
    }
}

template <typename T>
void invokeKolibriActivationExpansionFromMap(
    T const* unpermuted_input,
    T* permuted_output,
    int const* permuted_row_to_unpermuted_row,
    int const num_tokens,
    int const num_expanded_rows,
    int const hidden_size,
    cudaStream_t stream)
{
    constexpr int BLOCK_SIZE = 256;
    int const grid_size = std::min(num_expanded_rows, 1024);

    dim3 const grid(grid_size);
    dim3 const block(BLOCK_SIZE);

    kolibri_activation_expansion_from_map_kernel<T>
        <<<grid, block, 0, stream>>>(
            unpermuted_input,
            permuted_output,
            permuted_row_to_unpermuted_row,
            num_tokens,
            num_expanded_rows,
            hidden_size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("invokeKolibriActivationExpansionFromMap failed: ") + cudaGetErrorString(err));
    }
}

template void invokeKolibriActivationExpansionFromMap<half>(
    half const* unpermuted_input,
    half* permuted_output,
    int const* permuted_row_to_unpermuted_row,
    int const num_tokens,
    int const num_expanded_rows,
    int const hidden_size,
    cudaStream_t stream);

#ifdef ENABLE_BF16
template void invokeKolibriActivationExpansionFromMap<__nv_bfloat16>(
    __nv_bfloat16 const* unpermuted_input,
    __nv_bfloat16* permuted_output,
    int const* permuted_row_to_unpermuted_row,
    int const num_tokens,
    int const num_expanded_rows,
    int const hidden_size,
    cudaStream_t stream);
#endif

#ifdef ENABLE_FP8
template void invokeKolibriActivationExpansionFromMap<__nv_fp8_e4m3>(
    __nv_fp8_e4m3 const* unpermuted_input,
    __nv_fp8_e4m3* permuted_output,
    int const* permuted_row_to_unpermuted_row,
    int const num_tokens,
    int const num_expanded_rows,
    int const hidden_size,
    cudaStream_t stream);

template void invokeKolibriActivationExpansionFromMap<__nv_fp8_e5m2>(
    __nv_fp8_e5m2 const* unpermuted_input,
    __nv_fp8_e5m2* permuted_output,
    int const* permuted_row_to_unpermuted_row,
    int const num_tokens,
    int const num_expanded_rows,
    int const hidden_size,
    cudaStream_t stream);
#endif

} // namespace kernels
TRTLLM_NAMESPACE_END
