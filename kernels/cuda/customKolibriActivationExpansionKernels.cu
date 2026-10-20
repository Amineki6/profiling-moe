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

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <string>

TRTLLM_NAMESPACE_BEGIN
namespace kernels
{

// =================================================================================================
// KERNEL 2: Multi-Block Row Expansion & Permutation
//
// Distributed across all SMs of the GPU using 128-bit vectorized (uint4) memory copies.
// Each block processes one expanded row (or strides across rows) to saturate memory bandwidth.
// =================================================================================================
template <typename T>
__global__ void kolibri_activation_expansion_kernel(
    T const* __restrict__ unpermuted_input,             // [num_tokens, hidden_size]
    T* __restrict__ permuted_output,                    // [num_expanded_rows, hidden_size]
    int const* __restrict__ permuted_token_ids,         // [num_expanded_rows]
    int const num_expanded_rows,
    int const hidden_size)
{
    // 128-bit vector copy: uint4 = 16 bytes = 8 elements of FP16/BF16
    using VecType = uint4;
    constexpr int ELEMS_PER_VEC = sizeof(VecType) / sizeof(T);
    int const num_vecs = hidden_size / ELEMS_PER_VEC;

    // Grid-stride loop over expanded rows across all thread blocks
    for (int dst_row = blockIdx.x; dst_row < num_expanded_rows; dst_row += gridDim.x)
    {
        int const src_token_id = permuted_token_ids[dst_row];

        VecType const* src_vec = reinterpret_cast<VecType const*>(unpermuted_input + src_token_id * hidden_size);
        VecType* dst_vec       = reinterpret_cast<VecType*>(permuted_output + dst_row * hidden_size);

        // Coalesced 128-bit transfers across threads in the block
        for (int i = threadIdx.x; i < num_vecs; i += blockDim.x)
        {
            dst_vec[i] = src_vec[i];
        }
    }
}

template <typename T>
void invokeKolibriActivationExpansion(
    T const* unpermuted_input,
    T* permuted_output,
    int const* permuted_token_ids,
    int const num_expanded_rows,
    int const hidden_size,
    cudaStream_t stream)
{
    constexpr int EXPAND_BLOCK_SIZE = 256;
    int const expand_grid_size = std::min(num_expanded_rows, 1024);

    dim3 const expand_grid(expand_grid_size);
    dim3 const expand_block(EXPAND_BLOCK_SIZE);

    kolibri_activation_expansion_kernel<T>
        <<<expand_grid, expand_block, 0, stream>>>(
            unpermuted_input,
            permuted_output,
            permuted_token_ids,
            num_expanded_rows,
            hidden_size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("Activation expansion kernel launch failed: ") + cudaGetErrorString(err));
    }
}

// Template explicit instantiations
template void invokeKolibriActivationExpansion<half>(
    half const* unpermuted_input,
    half* permuted_output,
    int const* permuted_token_ids,
    int const num_expanded_rows,
    int const hidden_size,
    cudaStream_t stream);

#ifdef ENABLE_BF16
template void invokeKolibriActivationExpansion<__nv_bfloat16>(
    __nv_bfloat16 const* unpermuted_input,
    __nv_bfloat16* permuted_output,
    int const* permuted_token_ids,
    int const num_expanded_rows,
    int const hidden_size,
    cudaStream_t stream);
#endif

} // namespace kernels
TRTLLM_NAMESPACE_END
