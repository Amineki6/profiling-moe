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

#include <cuda_runtime.h>
#include <cstdint>

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

/**
 * 3-Stage Multi-Block Kolibri Dispatch Mapping Pipeline:
 *
 * Kernel 1: Grid G blocks independently count local expert frequencies in 1.5 KB shared memory.
 * Kernel 2: 1 block with 512 threads prefix-sums counts to determine global base offsets.
 * Kernel 3: Grid G blocks compute intra-block ranks in shared memory and scatter directly to final maps.
 *
 * @param token_selected_experts: [num_tokens * top_k] (int32)
 * @param permuted_row_to_unpermuted_row: [num_tokens * top_k] (int32)
 * @param unpermuted_row_to_permuted_row: [num_tokens * top_k] (int32, optional)
 * @param expert_first_token_offset: [num_experts + 1] (int64)
 * @param scratch_buffer: [2 * max_blocks * num_experts] (int32, optional; if nullptr, allocated via cudaMallocAsync)
 * @param num_tokens: Number of input tokens
 * @param num_experts: Must be 384 for Kolibri
 * @param top_k: Must be 6 for Kolibri
 * @param max_blocks: Max thread blocks (default 128 for RTX 4090 SM saturation)
 * @param stream: CUDA stream
 */
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
    cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
