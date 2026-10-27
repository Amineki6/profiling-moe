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

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

/*
 * Fused Routed Expert Output + Shared Expert Output Addition + Post-MoE Sandwich Norm.
 *
 * Computes:
 *   moe_out = RMSNorm(routed_out + shared_out, gamma)
 */
template <typename T>
void invokeFusedAddPostMoeNorm(
    T const* routed_out,
    T const* shared_out,
    T const* gamma,
    T* moe_out,
    float const eps,
    int const tokens,
    int const hidden_dim,
    cudaStream_t stream);

template <typename T>
void invokeCopyBaselinePostMoe(
    T const* in1,
    T const* in2,
    T* out,
    int const tokens,
    int const hidden_dim,
    cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
