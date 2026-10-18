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

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "fused_sandwich_norm.h"

using namespace tensorrt_llm::common;

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

/*
 * Fused Sandwich Norm + Residual Add + Pre-MoE Norm Kernel
 * 
 * Target Mathematical Flow:
 *   1. post_attn = RMSNorm(attn_out, gamma1)
 *   2. updated_residual = residual + post_attn
 *   3. moe_in = RMSNorm(updated_residual, gamma2)
 *
 * Grid Configuration:
 *   - Grid Dim: (num_tokens) -> One thread block (CTA) per token row
 *   - Block Dim: min(hidden_dim, 1024) (must be multiple of 32 for warp shuffles)
 */
template <typename T>
__global__ void fusedSandwichResidualNormKernel(
    T const* attn_out,         // [tokens, hidden_dim]
    T const* residual_in,      // [tokens, hidden_dim]
    T const* gamma1,           // [hidden_dim] (Sandwich Norm weight)
    T const* gamma2,           // [hidden_dim] (Pre-MoE Norm weight)
    T* updated_residual_out,   // [tokens, hidden_dim]
    T* moe_in_out,             // [tokens, hidden_dim]
    float const eps,
    int const hidden_dim)
{
    constexpr auto num_elems_T = num_elems<T>::value;
    using float_packed_t = typename packed_as<float, num_elems_T>::type;

    int const tidx = threadIdx.x;
    int const bidx = blockIdx.x;
    int const n_elems = hidden_dim / num_elems_T;

    __shared__ float s_variance1;
    __shared__ float s_variance2;

    // TODO: Step 1 - Compute sum of squares for attn_out (variance 1)

    // TODO: Step 2 - Block reduction for variance 1 and compute 1/sqrt(var1 + eps)

    // TODO: Step 3 - Apply RMSNorm 1 (gamma1) in registers, add to residual_in, store to updated_residual_out

    // TODO: Step 4 - Compute sum of squares for updated_residual (variance 2)

    // TODO: Step 5 - Block reduction for variance 2 and compute 1/sqrt(var2 + eps)

    // TODO: Step 6 - Apply RMSNorm 2 (gamma2) and write to moe_in_out
}

template <typename T>
void invokeFusedSandwichResidualNorm(
    T const* attn_out,
    T const* residual_in,
    T const* gamma1,
    T const* gamma2,
    T* updated_residual_out,
    T* moe_in_out,
    float const eps,
    int const tokens,
    int const hidden_dim,
    cudaStream_t stream)
{
    dim3 grid(tokens);
    dim3 block(min(hidden_dim, 1024));
    block.x = 32 * ((block.x + 31) / 32);

    constexpr size_t vec_size = 2;
    bool const use_vec_type = (hidden_dim % vec_size == 0)
        && (std::is_same<T, half>::value
#ifdef ENABLE_BF16
            || std::is_same<T, __nv_bfloat16>::value
#endif
        );

    if (use_vec_type)
    {
        using Tp = typename packed_as<T, vec_size>::type;
        fusedSandwichResidualNormKernel<Tp><<<grid, block, 0, stream>>>(
            reinterpret_cast<Tp const*>(attn_out),
            reinterpret_cast<Tp const*>(residual_in),
            reinterpret_cast<Tp const*>(gamma1),
            reinterpret_cast<Tp const*>(gamma2),
            reinterpret_cast<Tp*>(updated_residual_out),
            reinterpret_cast<Tp*>(moe_in_out),
            eps,
            hidden_dim);
    }
    else
    {
        fusedSandwichResidualNormKernel<T><<<grid, block, 0, stream>>>(
            attn_out,
            residual_in,
            gamma1,
            gamma2,
            updated_residual_out,
            moe_in_out,
            eps,
            hidden_dim);
    }
}

#ifdef ENABLE_BF16
template void invokeFusedSandwichResidualNorm<__nv_bfloat16>(
    __nv_bfloat16 const* attn_out,
    __nv_bfloat16 const* residual_in,
    __nv_bfloat16 const* gamma1,
    __nv_bfloat16 const* gamma2,
    __nv_bfloat16* updated_residual_out,
    __nv_bfloat16* moe_in_out,
    float const eps,
    int const tokens,
    int const hidden_dim,
    cudaStream_t stream);
#endif

template void invokeFusedSandwichResidualNorm<half>(
    half const* attn_out,
    half const* residual_in,
    half const* gamma1,
    half const* gamma2,
    half* updated_residual_out,
    half* moe_in_out,
    float const eps,
    int const tokens,
    int const hidden_dim,
    cudaStream_t stream);

template void invokeFusedSandwichResidualNorm<float>(
    float const* attn_out,
    float const* residual_in,
    float const* gamma1,
    float const* gamma2,
    float* updated_residual_out,
    float* moe_in_out,
    float const eps,
    int const tokens,
    int const hidden_dim,
    cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
