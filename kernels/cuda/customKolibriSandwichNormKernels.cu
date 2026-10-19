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
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaFp8Utils.h"
#include "tensorrt_llm/common/quantTypeUtils.cuh"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "customKolibriSandwichNormKernels.h"

using namespace tensorrt_llm::common;

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

template <typename Tf, typename T>
__inline__ __device__ Tf compute_rmsnorm(Tf val, float s_variance, T const* gamma, int i)
{
    Tf ret = val * s_variance * cuda_cast<Tf>(gamma[i]);
    return ret;
}

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
template <typename T, bool USE_SHMEM>
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

    extern __shared__ __align__(sizeof(float)) char _shmem[];
    T* shmem = reinterpret_cast<T*>(_shmem);

    int const tidx = threadIdx.x;
    int const bidx = blockIdx.x;
    int const n_elems = hidden_dim / num_elems_T;

    __shared__ float s_variance1;
    __shared__ float s_variance2;

    // Compute variance for sandwich norm (attn_out)
    float local_var_sum = 0.0f;
    float sum = 0.0f;
    float variance = 0.0f;

    for (int i = tidx; i < n_elems; i += blockDim.x)
    {
        T val = attn_out[bidx * n_elems + i];
        
        // Cache loaded activation in shared memory to avoid redundant DRAM reads
        if (USE_SHMEM)
        {
            shmem[i] = val;
        }

        float_packed_t const val_f = cuda_cast<float_packed_t>(val);
        local_var_sum += cuda_sum<float>(val_f * val_f);
    }

    // Inter-warp block reduction across CTA
    float packed[1] = {local_var_sum};
    blockReduceSumV2<float, 1>(packed);
    sum = packed[0];

    // Compute scale factor 1/sqrt(var + eps) and broadcast to block
    if (threadIdx.x == 0)
    {
        variance = (sum / hidden_dim);
        s_variance1 = rsqrtf(variance + eps);
    }
    __syncthreads();

    // Post-attention norm, in-register residual addition, and accumulate pre-MoE variance
    local_var_sum = 0.0f;
    for (int i = tidx; i < n_elems; i += blockDim.x)
    {
        int const index = bidx * n_elems + i;

        // Read activation from shared memory cache
        float_packed_t const val_f = cuda_cast<float_packed_t>(USE_SHMEM ? shmem[i] : attn_out[index]);

        // Apply sandwich norm weight
        T val = cuda_cast<T>(compute_rmsnorm(val_f, s_variance1, gamma1, i));

        // Add residual stream directly in registers
        T residual = residual_in[index];
        T res = val + residual;

        // Cache updated residual in shared memory for final norm
        if (USE_SHMEM)
        {
            shmem[i] = res;
        }

        // Save updated residual to global memory for later skip connection
        updated_residual_out[index] = res;

        // Accumulate variance for pre-MoE norm
        float_packed_t const res_f = cuda_cast<float_packed_t>(res);
        local_var_sum += cuda_sum<float>(res_f * res_f);
    }

    // Inter-warp block reduction for updated residual variance
    packed[0] = local_var_sum;
    blockReduceSumV2<float, 1>(packed);
    sum = packed[0];

    // Compute pre-MoE scale factor and broadcast to block
    if (threadIdx.x == 0)
    {
        variance = (sum / hidden_dim);
        s_variance2 = rsqrtf(variance + eps);
    }
    __syncthreads();

    // Pre-MoE norm and write to output buffer
    for (int i = tidx; i < n_elems; i += blockDim.x)
    {
        int const index = bidx * n_elems + i;

        // Read updated residual from shared memory cache
        float_packed_t const val_f = cuda_cast<float_packed_t>(USE_SHMEM ? shmem[i] : updated_residual_out[index]);

        // Apply pre-MoE norm weight and write out
        T val = cuda_cast<T>(compute_rmsnorm(val_f, s_variance2, gamma2, i));
        moe_in_out[index] = val;
    }
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
    dim3 block(min(hidden_dim, 256));
    block.x = 32 * ((block.x + 31) / 32);

    constexpr int vec_size = 2;
    size_t const shmem_size = hidden_dim * sizeof(T);
    bool const use_vec_type = (hidden_dim % vec_size == 0)
        && (std::is_same<T, half>::value
#ifdef ENABLE_BF16
            || std::is_same<T, __nv_bfloat16>::value
#endif
        );

    if (use_vec_type)
    {
        using Tp = typename packed_as<T, vec_size>::type;
        fusedSandwichResidualNormKernel<Tp, true><<<grid, block, shmem_size, stream>>>(
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
        fusedSandwichResidualNormKernel<T, true><<<grid, block, shmem_size, stream>>>(
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
