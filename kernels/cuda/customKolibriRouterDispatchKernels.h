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

#pragma once

#include "tensorrt_llm/common/config.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

/**
 * Kernel 1: Kolibri Router + CUTLASS Dispatch Mapping (Single Block)
 *
 * Computes top-6 routing and builds the dispatch permutation mapping in shared memory.
 * Uses a 384-bin shared memory histogram and prefix sum (~3 KB), completely avoiding CUB overhead.
 *
 * Inputs:
 *   - scores_in:                  [N, num_experts] (raw routing logits from router FC, FP32)
 *   - bias:                       [num_experts]    (routing bias terms, FP32)
 *
 * Outputs:
 *   - permuted_weights:           [N * top_k]              (routing sigmoid scale for each expanded row)
 *   - permuted_row_to_unpermuted: [N * top_k]              (mapping needed for FC2 unpermutation)
 *   - permuted_token_ids:         [N * top_k]              (source token index for each expanded row)
 *   - expert_first_token_offset:  [num_experts + 1]        (histogram prefix sums for CUTLASS Grouped GEMM)
 */
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
    cudaStream_t stream);

/**
 * Kernel 2: Activation Row Expansion & Permutation (Multi-Block Grid across all SMs)
 *
 * Distributes expanded rows across all available SMs on the GPU.
 * Uses 128-bit vectorized (uint4 = 16 bytes = 8 FP16/BF16 elements) coalesced memory transfers.
 *
 * Inputs:
 *   - unpermuted_input:           [N, hidden_size]         (hidden activations, FP16 or BF16)
 *   - permuted_token_ids:         [N * top_k]              (source token index for each expanded row)
 *
 * Outputs:
 *   - permuted_output:            [N * top_k, hidden_size] (CUTLASS FC1 GEMM input)
 */
template <typename T>
void invokeKolibriActivationExpansion(
    T const* unpermuted_input,
    T* permuted_output,
    int const* permuted_token_ids,
    int const num_expanded_rows,
    int const hidden_size,
    cudaStream_t stream);

/**
 * Pipelined Execution: Kernel 1 followed by Kernel 2 on the same CUDA stream.
 *
 * Eliminates kernel launch gaps and intermediate DRAM roundtrips on router outputs.
 */
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
    cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
