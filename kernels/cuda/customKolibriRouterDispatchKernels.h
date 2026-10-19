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
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

/**
 * Fused Kolibri Router + CUTLASS Expert Dispatch Mapping + Activation Expansion
 *
 * Fuses:
 *   1. Custom Kolibri Top-K Router (scores + bias -> top-6 + sigmoid on raw scores)
 *   2. CUB-based Expert Map Sort & Prefix Sum (out_indices -> expert_first_token_offset, permutation tables)
 *   3. Activation Row Expansion (unpermuted_input [N, d] -> permuted_output [6*N, d] + permuted_scales [6*N])
 *
 * Inputs:
 *   - unpermuted_input:           [N, hidden_size] (hidden activations, FP16 or BF16)
 *   - scores_in:                  [N, num_experts] (raw routing logits from router FC, FP32)
 *   - bias:                       [num_experts]    (routing bias terms, FP32)
 *
 * Outputs:
 *   - permuted_output:            [N * top_k, hidden_size] (CUTLASS FC1 GEMM input)
 *   - permuted_scales:            [N * top_k]              (routing scale for each expanded row)
 *   - permuted_row_to_unpermuted: [N * top_k]              (mapping needed for FC2 unpermutation)
 *   - expert_first_token_offset:  [num_experts + 1]        (histogram prefix sums for CUTLASS Grouped GEMM)
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
    int const num_tokens,
    int const hidden_size,
    int const num_experts,
    int const top_k,
    cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
