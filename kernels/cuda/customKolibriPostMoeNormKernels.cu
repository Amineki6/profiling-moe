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
#include "customKolibriPostMoeNormKernels.h"

using namespace tensorrt_llm::common;

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

template <typename Tf, typename T>
__inline__ __device__ Tf compute_rmsnorm(Tf val, float s_variance, T const* gamma, int i)
{
    return val * s_variance * cuda_cast<Tf>(gamma[i]);
}

/*
 * Fused Routed + Shared Expert Output Addition + Post-MoE Sandwich Norm Kernel
 *
 * Mathematical Flow:
 *   1. combined = routed_out + shared_out  (computed directly in registers)
 *   2. moe_out = RMSNorm(combined, gamma)
 *
 * Grid Configuration:
 *   - Grid Dim: (num_tokens) -> One thread block (CTA) per token row
 *   - Block Dim: min(hidden_dim, 1024) (must be multiple of 32 for warp shuffles)
 */
template <typename T, bool USE_SHMEM>
__global__ void fusedAddPostMoeNormKernel(
    T const* routed_out,       // [tokens, hidden_dim]
    T const* shared_out,       // [tokens, hidden_dim]
    T const* gamma,            // [hidden_dim] (Post-MoE Norm weight)
    T* moe_out,                // [tokens, hidden_dim]
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

    __shared__ float s_variance;

    // In-register addition of routed and shared outputs + variance accumulation
    float local_var_sum = 0.0f;
    float sum = 0.0f;
    float variance = 0.0f;

    for (int i = tidx; i < n_elems; i += blockDim.x)
    {
        int const index = bidx * n_elems + i;
        T routed = routed_out[index];
        T shared = shared_out[index];
        T res = routed + shared;

        // Cache combined activation in shared memory for Phase 2
        if (USE_SHMEM)
        {
            shmem[i] = res;
        }

        float_packed_t const res_f = cuda_cast<float_packed_t>(res);
        local_var_sum += cuda_sum<float>(res_f * res_f);
    }

    // Inter-warp block reduction across CTA
    float packed[1] = {local_var_sum};
    blockReduceSumV2<float, 1>(packed);
    sum = packed[0];

    // Compute scale factor 1/sqrt(var + eps) and broadcast to block
    if (threadIdx.x == 0)
    {
        variance = (sum / hidden_dim);
        s_variance = rsqrtf(variance + eps);
    }
    __syncthreads();

    // Apply post-MoE norm weight and write out
    for (int i = tidx; i < n_elems; i += blockDim.x)
    {
        int const index = bidx * n_elems + i;

        // Read combined activation from shared memory cache
        T res = USE_SHMEM ? shmem[i] : (routed_out[index] + shared_out[index]);
        float_packed_t const val_f = cuda_cast<float_packed_t>(res);

        T val = cuda_cast<T>(compute_rmsnorm(val_f, s_variance, gamma, i));
        moe_out[index] = val;
    }
}

template <typename T>
void invokeFusedAddPostMoeNorm(
    T const* routed_out,
    T const* shared_out,
    T const* gamma,
    T* moe_out,
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
        fusedAddPostMoeNormKernel<Tp, true><<<grid, block, shmem_size, stream>>>(
            reinterpret_cast<Tp const*>(routed_out),
            reinterpret_cast<Tp const*>(shared_out),
            reinterpret_cast<Tp const*>(gamma),
            reinterpret_cast<Tp*>(moe_out),
            eps,
            hidden_dim);
    }
    else
    {
        fusedAddPostMoeNormKernel<T, true><<<grid, block, shmem_size, stream>>>(
            routed_out,
            shared_out,
            gamma,
            moe_out,
            eps,
            hidden_dim);
    }
}

#ifdef ENABLE_BF16
template void invokeFusedAddPostMoeNorm<__nv_bfloat16>(
    __nv_bfloat16 const* routed_out,
    __nv_bfloat16 const* shared_out,
    __nv_bfloat16 const* gamma,
    __nv_bfloat16* moe_out,
    float const eps,
    int const tokens,
    int const hidden_dim,
    cudaStream_t stream);
#endif

template void invokeFusedAddPostMoeNorm<half>(
    half const* routed_out,
    half const* shared_out,
    half const* gamma,
    half* moe_out,
    float const eps,
    int const tokens,
    int const hidden_dim,
    cudaStream_t stream);

template void invokeFusedAddPostMoeNorm<float>(
    float const* routed_out,
    float const* shared_out,
    float const* gamma,
    float* moe_out,
    float const eps,
    int const tokens,
    int const hidden_dim,
    cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
