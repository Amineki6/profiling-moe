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

#include "customKolibriMultiBlockBuildExpertMapsKernels.h"
#include "customKolibriMultiBlockBuildExpertMapsKernels.cuh"
#include "customKolibriBuildExpertMapsKernels.cuh"
#include <stdexcept>
#include <string>
#include <algorithm>

TRTLLM_NAMESPACE_BEGIN
namespace kernels
{

static constexpr int KOLIBRI_NUM_EXPERTS = 384;
static constexpr int KOLIBRI_TOP_K = 6;
static constexpr int BLOCK_SIZE_K1 = 1024;
static constexpr int BLOCK_SIZE_K3 = 256;
static constexpr int SCAN_BLOCK_SIZE = 512;
static constexpr int ELEMENTS_PER_BLOCK = 2048;
static constexpr int TIPPING_POINT_TOKENS = 512;

void invokeKolibriMultiBlockBuildExpertMaps(
    int const* token_selected_experts,
    int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row,
    int64_t* expert_first_token_offset,
    int* scratch_buffer,
    int const num_tokens,
    int const num_experts,
    int const top_k,
    int const max_blocks,
    cudaStream_t stream)
{
    if (num_experts != KOLIBRI_NUM_EXPERTS || top_k != KOLIBRI_TOP_K)
    {
        throw std::runtime_error("Kolibri multi-block build expert maps requires num_experts=384 and top_k=6");
    }

    if (num_tokens <= 0)
    {
        return;
    }

    int const total_selections = num_tokens * top_k;

    // ---------------------------------------------------------------------------------------------
    // Dynamic Regime Dispatch Strategy:
    // Decode regime (num_tokens < 512): Single-Block kernel (2-3 µs faster, zero multi-kernel launch overhead)
    // Prefill regime (num_tokens >= 512): 3-Stage Multi-Block pipeline (scales across all 128 SMs, up to 65x faster)
    // If max_blocks < 0: Bypasses the threshold and forces multi-block with abs(max_blocks).
    // ---------------------------------------------------------------------------------------------
    if (max_blocks >= 0 && num_tokens < TIPPING_POINT_TOKENS)
    {
        bool allocated_scratch = false;
        int2* selection_scratch = reinterpret_cast<int2*>(scratch_buffer);
        size_t const required_scratch_bytes = static_cast<size_t>(total_selections) * sizeof(int2);

        if (selection_scratch == nullptr)
        {
            cudaError_t const alloc_err = cudaMallocAsync(
                reinterpret_cast<void**>(&selection_scratch),
                required_scratch_bytes,
                stream);
            if (alloc_err != cudaSuccess)
            {
                throw std::runtime_error(std::string("Selection scratch allocation failed: ")
                    + cudaGetErrorString(alloc_err));
            }
            allocated_scratch = true;
        }

        dim3 const grid_single(1);
        dim3 const block_single(BLOCK_SIZE_K3); // 256 threads

        kolibri_build_expert_maps_kernel<KOLIBRI_NUM_EXPERTS, KOLIBRI_TOP_K, BLOCK_SIZE_K3>
            <<<grid_single, block_single, 0, stream>>>(
                token_selected_experts,
                permuted_row_to_unpermuted_row,
                unpermuted_row_to_permuted_row,
                expert_first_token_offset,
                num_tokens,
                selection_scratch);

        cudaError_t err = cudaGetLastError();
        if (allocated_scratch && selection_scratch != nullptr)
        {
            cudaError_t const free_err = cudaFreeAsync(selection_scratch, stream);
            if (err == cudaSuccess)
            {
                err = free_err;
            }
        }
        if (err != cudaSuccess)
        {
            throw std::runtime_error(std::string("Single-block dispatch mapping failed: ")
                + cudaGetErrorString(err));
        }
        return;
    }

    // Determine grid size:
    // If max_blocks < 0: use fixed abs(max_blocks)
    // If max_blocks >= 0: use dynamic heuristic clamped to max_blocks (default 128)
    int num_blocks = 128;
    if (max_blocks < 0)
    {
        num_blocks = -max_blocks;
    }
    else
    {
        int const max_b = max_blocks > 0 ? max_blocks : 128;
        num_blocks = std::max(1, std::min(max_b, (total_selections + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK));
    }

    // Allocate scratch buffer if not provided
    // Buffer layout: [block_expert_counts: num_blocks * 384] [block_expert_offsets: num_blocks * 384]
    bool allocated_scratch = false;
    int* d_scratch = scratch_buffer;
    size_t const scratch_elements = static_cast<size_t>(num_blocks) * KOLIBRI_NUM_EXPERTS * 2;

    if (d_scratch == nullptr)
    {
        cudaError_t const alloc_err = cudaMallocAsync(
            reinterpret_cast<void**>(&d_scratch),
            scratch_elements * sizeof(int),
            stream);
        if (alloc_err != cudaSuccess)
        {
            throw std::runtime_error(std::string("Multi-block scratch allocation failed: ")
                + cudaGetErrorString(alloc_err));
        }
        allocated_scratch = true;
    }

    int* block_expert_counts = d_scratch;
    int* block_expert_offsets = d_scratch + static_cast<size_t>(num_blocks) * KOLIBRI_NUM_EXPERTS;

    dim3 const grid_g(num_blocks);
    dim3 const block_k1(BLOCK_SIZE_K1);
    dim3 const block_k3(BLOCK_SIZE_K3);
    dim3 const grid_1(1);
    dim3 const block_512(SCAN_BLOCK_SIZE);

    // Stage 1: Every block independently computes local expert counts in shared memory (1024 threads, 66.7% occupancy)
    kolibri_multi_block_expert_counts_kernel<KOLIBRI_NUM_EXPERTS, BLOCK_SIZE_K1>
        <<<grid_g, block_k1, 0, stream>>>(
            token_selected_experts,
            block_expert_counts,
            total_selections,
            num_blocks);

    // Stage 2: Prefix-sum counts across blocks & experts (register-cached, zero redundant global reads)
    if (num_blocks <= 64)
    {
        kolibri_multi_block_prefix_sum_kernel<KOLIBRI_NUM_EXPERTS, 64, SCAN_BLOCK_SIZE>
            <<<grid_1, block_512, 0, stream>>>(
                block_expert_counts,
                block_expert_offsets,
                expert_first_token_offset,
                num_blocks);
    }
    else
    {
        kolibri_multi_block_prefix_sum_kernel<KOLIBRI_NUM_EXPERTS, 128, SCAN_BLOCK_SIZE>
            <<<grid_1, block_512, 0, stream>>>(
                block_expert_counts,
                block_expert_offsets,
                expert_first_token_offset,
                num_blocks);
    }

    // Stage 3: Every block computes local positions and writes directly into global mapping arrays (256 threads)
    kolibri_multi_block_scatter_maps_kernel<KOLIBRI_NUM_EXPERTS, KOLIBRI_TOP_K, BLOCK_SIZE_K3>
        <<<grid_g, block_k3, 0, stream>>>(
            token_selected_experts,
            block_expert_offsets,
            permuted_row_to_unpermuted_row,
            unpermuted_row_to_permuted_row,
            total_selections,
            num_tokens,
            num_blocks);

    cudaError_t err = cudaGetLastError();

    if (allocated_scratch && d_scratch != nullptr)
    {
        cudaError_t const free_err = cudaFreeAsync(d_scratch, stream);
        if (err == cudaSuccess)
        {
            err = free_err;
        }
    }

    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("invokeKolibriMultiBlockBuildExpertMaps failed: ")
            + cudaGetErrorString(err));
    }
}

} // namespace kernels
TRTLLM_NAMESPACE_END
