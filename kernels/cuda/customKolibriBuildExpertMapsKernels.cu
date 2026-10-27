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

#include "customKolibriBuildExpertMapsKernels.cuh"
#include <stdexcept>
#include <string>

TRTLLM_NAMESPACE_BEGIN
namespace kernels
{

static constexpr int KOLIBRI_NUM_EXPERTS = 384;
static constexpr int KOLIBRI_TOP_K = 6;

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

    int2* selection_scratch = nullptr;
    if (num_tokens > 0)
    {
        cudaError_t const allocation_err = cudaMallocAsync(
            reinterpret_cast<void**>(&selection_scratch),
            static_cast<size_t>(num_tokens) * KOLIBRI_TOP_K * sizeof(int2), stream);
        if (allocation_err != cudaSuccess)
        {
            throw std::runtime_error(std::string("Kolibri selection scratch allocation failed: ")
                + cudaGetErrorString(allocation_err));
        }
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
            num_tokens,
            selection_scratch);

    cudaError_t err = cudaGetLastError();
    cudaError_t const free_err = selection_scratch != nullptr
        ? cudaFreeAsync(selection_scratch, stream) : cudaSuccess;
    if (err == cudaSuccess)
    {
        err = free_err;
    }
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("invokeKolibriBuildExpertMaps failed: ") + cudaGetErrorString(err));
    }
}

} // namespace kernels
TRTLLM_NAMESPACE_END
