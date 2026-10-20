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
 * Step 2: Custom Kolibri Expert Dispatch Mapping (256 Threads, 3 KB Shared Memory)
 *
 * Replaces TRT-LLM's single-warp CUB BlockRadixRank kernel with a direct 384-bin
 * shared memory histogram and exclusive prefix sum.
 */
void invokeKolibriBuildExpertMaps(
    int const* token_selected_experts,
    int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row,
    int64_t* expert_first_token_offset,
    int const num_tokens,
    int const num_experts,
    int const top_k,
    cudaStream_t stream);

/**
 * TRT-LLM Baseline: CUB BlockRadixRank<32, 9> (Single warp, 32 threads)
 */
void invokeTrtllmBuildExpertMapsB32(
    int const* token_selected_experts,
    int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row,
    int64_t* expert_first_token_offset,
    int const num_tokens,
    int const num_experts,
    int const top_k,
    cudaStream_t stream);

/**
 * Step 3: Multi-Block Vectorized Activation Row Expansion
 *
 * Consumes permuted_row_to_unpermuted_row directly from Step 2.
 * Copies rows using 128-bit coalesced transfers across all 128 SMs.
 */
template <typename T>
void invokeKolibriActivationExpansionFromMap(
    T const* unpermuted_input,
    T* permuted_output,
    int const* permuted_row_to_unpermuted_row,
    int const num_tokens,
    int const num_expanded_rows,
    int const hidden_size,
    cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
