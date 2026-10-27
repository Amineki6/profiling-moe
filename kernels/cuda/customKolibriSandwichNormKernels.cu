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

// 128-bit (16-byte) vectorized container matching NVIDIA TensorRT-LLM standard
template <typename T>
struct alignas(16) Vec128_t;

#ifdef ENABLE_BF16
template <>
struct alignas(16) Vec128_t<__nv_bfloat16>
{
    static constexpr int size = 8;
    __nv_bfloat16 data[8];

    __device__ inline float sum_of_squares() const
    {
        float s = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; ++k)
        {
            float const f = cuda_cast<float>(data[k]);
            s += f * f;
        }
        return s;
    }

    __device__ inline Vec128_t<__nv_bfloat16> rmsnorm(float s_var, Vec128_t<__nv_bfloat16> const& gamma) const
    {
        Vec128_t<__nv_bfloat16> out;
        #pragma unroll
        for (int k = 0; k < 8; ++k)
        {
            float const val_f = cuda_cast<float>(data[k]);
            float const g_f = cuda_cast<float>(gamma.data[k]);
            out.data[k] = cuda_cast<__nv_bfloat16>(val_f * s_var * g_f);
        }
        return out;
    }

    __device__ inline Vec128_t<__nv_bfloat16> add(Vec128_t<__nv_bfloat16> const& b) const
    {
        Vec128_t<__nv_bfloat16> out;
        #pragma unroll
        for (int k = 0; k < 8; ++k)
        {
            out.data[k] = data[k] + b.data[k];
        }
        return out;
    }
};
#endif

template <>
struct alignas(16) Vec128_t<half>
{
    static constexpr int size = 8;
    half data[8];

    __device__ inline float sum_of_squares() const
    {
        float s = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; ++k)
        {
            float const f = cuda_cast<float>(data[k]);
            s += f * f;
        }
        return s;
    }

    __device__ inline Vec128_t<half> rmsnorm(float s_var, Vec128_t<half> const& gamma) const
    {
        Vec128_t<half> out;
        #pragma unroll
        for (int k = 0; k < 8; ++k)
        {
            float const val_f = cuda_cast<float>(data[k]);
            float const g_f = cuda_cast<float>(gamma.data[k]);
            out.data[k] = cuda_cast<half>(val_f * s_var * g_f);
        }
        return out;
    }

    __device__ inline Vec128_t<half> add(Vec128_t<half> const& b) const
    {
        Vec128_t<half> out;
        #pragma unroll
        for (int k = 0; k < 8; ++k)
        {
            out.data[k] = data[k] + b.data[k];
        }
        return out;
    }
};

template <>
struct alignas(16) Vec128_t<float>
{
    static constexpr int size = 4;
    float data[4];

    __device__ inline float sum_of_squares() const
    {
        float s = 0.0f;
        #pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            s += data[k] * data[k];
        }
        return s;
    }

    __device__ inline Vec128_t<float> rmsnorm(float s_var, Vec128_t<float> const& gamma) const
    {
        Vec128_t<float> out;
        #pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            out.data[k] = data[k] * s_var * gamma.data[k];
        }
        return out;
    }

    __device__ inline Vec128_t<float> add(Vec128_t<float> const& b) const
    {
        Vec128_t<float> out;
        #pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            out.data[k] = data[k] + b.data[k];
        }
        return out;
    }
};

template <typename T>
__inline__ __device__ float compute_sum_of_squares(T const& val)
{
    constexpr auto num_elems_T = num_elems<T>::value;
    using float_packed_t = typename packed_as<float, num_elems_T>::type;
    float_packed_t const val_f = cuda_cast<float_packed_t>(val);
    return cuda_sum<float>(val_f * val_f);
}

template <typename T>
__inline__ __device__ T compute_vec_rmsnorm(T const& val, float s_variance, T const& gamma)
{
    constexpr auto num_elems_T = num_elems<T>::value;
    using float_packed_t = typename packed_as<float, num_elems_T>::type;
    float_packed_t const val_f = cuda_cast<float_packed_t>(val);
    float_packed_t const gamma_f = cuda_cast<float_packed_t>(gamma);
    return cuda_cast<T>(val_f * s_variance * gamma_f);
}

template <int NUM_WARPS>
__device__ __forceinline__ float blockSumAll(float v, float* red)
{
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
    {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    if ((threadIdx.x & 31) == 0)
    {
        red[threadIdx.x >> 5] = v;
    }
    __syncthreads();
    float t = 0.f;
    #pragma unroll
    for (int w = 0; w < NUM_WARPS; ++w)
    {
        t += red[w]; // broadcast read
    }
    return t; // valid across every thread in the CTA
}

// Ultra-fast in-register kernel:
// 1. Hoists all global memory loads up front (attn_out, residual_in, gamma1, gamma2).
// 2. Eliminates all intermediate shared memory round-trips (everything stays in ~32 registers).
// 3. Lightweight single-barrier reduction with warp-shuffle + broadcast (red1, red2).
template <typename T, int NUM_WARPS, int ITERS, int HIDDEN_DIM>
__global__ void __launch_bounds__(NUM_WARPS * 32)
fusedSandwichNormRegKernel(
    Vec128_t<T> const* __restrict__ attn_out,
    Vec128_t<T> const* __restrict__ residual_in,
    Vec128_t<T> const* __restrict__ gamma1,
    Vec128_t<T> const* __restrict__ gamma2,
    Vec128_t<T>* __restrict__ updated_residual_out,
    Vec128_t<T>* __restrict__ moe_in_out,
    float const eps)
{
    using V = Vec128_t<T>;
    constexpr int THREADS = NUM_WARPS * 32;
    constexpr int N_VEC   = THREADS * ITERS;
    constexpr float INV_HIDDEN_DIM = 1.0f / float(HIDDEN_DIM);

    // Two separate shared buffers eliminate write-after-read hazards
    __shared__ float red1[NUM_WARPS];
    __shared__ float red2[NUM_WARPS];

    size_t const row = size_t(blockIdx.x) * N_VEC;
    int const tid = threadIdx.x;

    V a[ITERS];
    V r[ITERS];
    V g1[ITERS];
    V g2[ITERS];

    // Issue all global loads up front into registers to maximize memory-level parallelism
    #pragma unroll
    for (int k = 0; k < ITERS; ++k)
    {
        int const i = tid + k * THREADS;
        a[k] = attn_out[row + i];
        r[k] = residual_in[row + i];
    }

    #pragma unroll
    for (int k = 0; k < ITERS; ++k)
    {
        int const i = tid + k * THREADS;
        g1[k] = gamma1[i];
        g2[k] = gamma2[i];
    }

    // 1. Compute first RMSNorm variance
    float ss = 0.f;
    #pragma unroll
    for (int k = 0; k < ITERS; ++k)
    {
        ss += a[k].sum_of_squares();
    }
    float const inv1 = rsqrtf(blockSumAll<NUM_WARPS>(ss, red1) * INV_HIDDEN_DIM + eps);

    // 2. Post-attention RMSNorm + Residual addition in-registers, store updated residual, accumulate pre-MoE variance
    ss = 0.f;
    #pragma unroll
    for (int k = 0; k < ITERS; ++k)
    {
        a[k] = a[k].rmsnorm(inv1, g1[k]).add(r[k]);
        updated_residual_out[row + tid + k * THREADS] = a[k];
        ss += a[k].sum_of_squares();
    }
    float const inv2 = rsqrtf(blockSumAll<NUM_WARPS>(ss, red2) * INV_HIDDEN_DIM + eps);

    // 3. Pre-MoE RMSNorm in-registers, store moe_in
    #pragma unroll
    for (int k = 0; k < ITERS; ++k)
    {
        moe_in_out[row + tid + k * THREADS] = a[k].rmsnorm(inv2, g2[k]);
    }
}

// 128-bit Vectorized Kernel (issues LDG.E.128 and STG.E.128)
template <typename T>
__global__ void fusedSandwichResidualNormKernel128(
    Vec128_t<T> const* attn_out,
    Vec128_t<T> const* residual_in,
    Vec128_t<T> const* gamma1,
    Vec128_t<T> const* gamma2,
    Vec128_t<T>* updated_residual_out,
    Vec128_t<T>* moe_in_out,
    float const eps,
    int const hidden_dim)
{
    constexpr int V_SIZE = Vec128_t<T>::size;
    int const n_vec = hidden_dim / V_SIZE;
    int const tidx = threadIdx.x;
    int const bidx = blockIdx.x;

    extern __shared__ __align__(16) char _shmem[];
    Vec128_t<T>* shmem = reinterpret_cast<Vec128_t<T>*>(_shmem);

    __shared__ float s_variance1;
    __shared__ float s_variance2;

    float local_var_sum = 0.0f;
    #pragma unroll 2
    for (int i = tidx; i < n_vec; i += blockDim.x)
    {
        // 128-bit Load (LDG.E.128)
        Vec128_t<T> val = attn_out[bidx * n_vec + i];
        shmem[i] = val;
        local_var_sum += val.sum_of_squares();
    }

    float packed[1] = {local_var_sum};
    blockReduceSumV2<float, 1>(packed);

    if (threadIdx.x == 0)
    {
        float const variance = (packed[0] / hidden_dim);
        s_variance1 = rsqrtf(variance + eps);
    }
    __syncthreads();

    local_var_sum = 0.0f;
    #pragma unroll 2
    for (int i = tidx; i < n_vec; i += blockDim.x)
    {
        int const index = bidx * n_vec + i;
        Vec128_t<T> val = shmem[i];
        Vec128_t<T> g1 = gamma1[i];
        Vec128_t<T> norm_val = val.rmsnorm(s_variance1, g1);

        // 128-bit Load residual (LDG.E.128)
        Vec128_t<T> residual = residual_in[index];
        Vec128_t<T> res = norm_val.add(residual);

        shmem[i] = res;
        // 128-bit Store updated residual (STG.E.128)
        updated_residual_out[index] = res;

        local_var_sum += res.sum_of_squares();
    }

    packed[0] = local_var_sum;
    blockReduceSumV2<float, 1>(packed);

    if (threadIdx.x == 0)
    {
        float const variance = (packed[0] / hidden_dim);
        s_variance2 = rsqrtf(variance + eps);
    }
    __syncthreads();

    #pragma unroll 2
    for (int i = tidx; i < n_vec; i += blockDim.x)
    {
        int const index = bidx * n_vec + i;
        Vec128_t<T> res = shmem[i];
        Vec128_t<T> g2 = gamma2[i];
        Vec128_t<T> moe_val = res.rmsnorm(s_variance2, g2);

        // 128-bit Store moe_in (STG.E.128)
        moe_in_out[index] = moe_val;
    }
}

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

        local_var_sum += compute_sum_of_squares(val);
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
        T const val = USE_SHMEM ? shmem[i] : attn_out[index];

        // Apply sandwich norm weight
        T norm_val = compute_vec_rmsnorm(val, s_variance1, gamma1[i]);

        // Add residual stream directly in registers
        T residual = residual_in[index];
        T res = norm_val + residual;

        // Cache updated residual in shared memory for final norm
        if (USE_SHMEM)
        {
            shmem[i] = res;
        }

        // Save updated residual to global memory for later skip connection
        updated_residual_out[index] = res;

        // Accumulate variance for pre-MoE norm
        local_var_sum += compute_sum_of_squares(res);
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
        T const val = USE_SHMEM ? shmem[i] : updated_residual_out[index];

        // Apply pre-MoE norm weight and write out
        T norm_val = compute_vec_rmsnorm(val, s_variance2, gamma2[i]);
        moe_in_out[index] = norm_val;
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

    constexpr int vec128_elements = 16 / sizeof(T);
    size_t const shmem_size = hidden_dim * sizeof(T);

    if (hidden_dim % vec128_elements == 0)
    {
        using V128 = Vec128_t<T>;
        int const n_vec = hidden_dim / vec128_elements;

        // Specialized zero-shmem in-register path for standard 2560 hidden dimension:
        // NUM_WARPS = 5 (160 threads), ITERS = 2, HIDDEN_DIM = 2560 (N_VEC = 320)
        if (hidden_dim == 2560 && (std::is_same<T, half>::value
#ifdef ENABLE_BF16
            || std::is_same<T, __nv_bfloat16>::value
#endif
        ))
        {
            fusedSandwichNormRegKernel<T, 5, 2, 2560><<<grid, 160, 0, stream>>>(
                reinterpret_cast<V128 const*>(attn_out),
                reinterpret_cast<V128 const*>(residual_in),
                reinterpret_cast<V128 const*>(gamma1),
                reinterpret_cast<V128 const*>(gamma2),
                reinterpret_cast<V128*>(updated_residual_out),
                reinterpret_cast<V128*>(moe_in_out),
                eps);
        }
        else
        {
            // Generic 128-bit vectorized fallback
            dim3 block128(min(n_vec, 160));
            block128.x = 32 * ((block128.x + 31) / 32);

            fusedSandwichResidualNormKernel128<T><<<grid, block128, shmem_size, stream>>>(
                reinterpret_cast<V128 const*>(attn_out),
                reinterpret_cast<V128 const*>(residual_in),
                reinterpret_cast<V128 const*>(gamma1),
                reinterpret_cast<V128 const*>(gamma2),
                reinterpret_cast<V128*>(updated_residual_out),
                reinterpret_cast<V128*>(moe_in_out),
                eps,
                hidden_dim);
        }
    }
    else
    {
        constexpr int vec_size = 2;
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

// Pure Copy-Kernel Baseline:
// Reads 2 streams (attn_out, residual_in) and writes 2 streams (out1, out2) via 128-bit loads/stores.
// With 1152 tokens and hidden_dim=2560 (bf16), exactly 11.8 MB read and 11.8 MB written.
// Matches identical grid (tokens) and block (160 threads, 2 iterations per thread).
template <typename T, int NUM_WARPS, int ITERS>
__global__ void __launch_bounds__(NUM_WARPS * 32)
copyBaselineKernel(
    Vec128_t<T> const* __restrict__ in1,
    Vec128_t<T> const* __restrict__ in2,
    Vec128_t<T>* __restrict__ out1,
    Vec128_t<T>* __restrict__ out2)
{
    constexpr int THREADS = NUM_WARPS * 32;
    constexpr int N_VEC   = THREADS * ITERS;

    size_t const row = size_t(blockIdx.x) * N_VEC;
    int const tid = threadIdx.x;

    #pragma unroll
    for (int k = 0; k < ITERS; ++k)
    {
        size_t const idx = row + tid + k * THREADS;
        out1[idx] = in1[idx];
        out2[idx] = in2[idx];
    }
}

template <typename T>
void invokeCopyBaseline(
    T const* in1,
    T const* in2,
    T* out1,
    T* out2,
    int const tokens,
    int const hidden_dim,
    cudaStream_t stream)
{
    dim3 grid(tokens);
    constexpr int vec128_elements = 16 / sizeof(T);
    using V128 = Vec128_t<T>;

    if (hidden_dim == 2560 && vec128_elements == 8)
    {
        copyBaselineKernel<T, 5, 2><<<grid, 160, 0, stream>>>(
            reinterpret_cast<V128 const*>(in1),
            reinterpret_cast<V128 const*>(in2),
            reinterpret_cast<V128*>(out1),
            reinterpret_cast<V128*>(out2));
    }
}

#ifdef ENABLE_BF16
template void invokeCopyBaseline<__nv_bfloat16>(
    __nv_bfloat16 const* in1,
    __nv_bfloat16 const* in2,
    __nv_bfloat16* out1,
    __nv_bfloat16* out2,
    int const tokens,
    int const hidden_dim,
    cudaStream_t stream);
#endif

template void invokeCopyBaseline<half>(
    half const* in1,
    half const* in2,
    half* out1,
    half* out2,
    int const tokens,
    int const hidden_dim,
    cudaStream_t stream);

template void invokeCopyBaseline<float>(
    float const* in1,
    float const* in2,
    float* out1,
    float* out2,
    int const tokens,
    int const hidden_dim,
    cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
