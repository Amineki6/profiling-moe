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
#include <stdexcept>
#include <string>

TRTLLM_NAMESPACE_BEGIN
namespace kernels
{

// =================================================================================================
// Host Launcher: Pipelined Two-Kernel Execution
//
// Launches:
//   Stage 1: invokeKolibriRouterDispatchMap (Single block, router top-6 + 384-bin histogram/scan)
//   Stage 2: invokeKolibriActivationExpansion (Multi-block grid across all SMs, 128-bit vector copy)
//
// Both kernels execute sequentially on the same stream without CPU synchronization.
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
    int* permuted_token_ids,
    int const num_tokens,
    int const hidden_size,
    int const num_experts,
    int const top_k,
    cudaStream_t stream)
{
    // Stage 1: Router Top-K + Dispatch Mapping (Single block, shared memory histogram)
    invokeKolibriRouterDispatchMap(
        scores_in,
        bias,
        permuted_scales,
        permuted_row_to_unpermuted_row,
        permuted_token_ids,
        expert_first_token_offset,
        num_tokens,
        num_experts,
        top_k,
        stream);

    // Stage 2: Multi-Block Vectorized Activation Row Expansion (Parallelized across all 128 SMs)
    int const num_expanded_rows = num_tokens * top_k;
    invokeKolibriActivationExpansion<T>(
        unpermuted_input,
        permuted_output,
        permuted_token_ids,
        num_expanded_rows,
        hidden_size,
        stream);
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
    int* permuted_token_ids,
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
    int* permuted_token_ids,
    int const num_tokens,
    int const hidden_size,
    int const num_experts,
    int const top_k,
    cudaStream_t stream);
#endif

} // namespace kernels
TRTLLM_NAMESPACE_END
