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
__device__ __forceinline__ Vec128_t<T> load_128_cg(Vec128_t<T> const* ptr)
{
    Vec128_t<T> ret;
    uint4* out = reinterpret_cast<uint4*>(&ret);
    asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(out->x), "=r"(out->y), "=r"(out->z), "=r"(out->w)
                 : "l"(ptr));
    return ret;
}

template <typename T>
__device__ __forceinline__ Vec128_t<T> load_128_ca(Vec128_t<T> const* ptr)
{
    Vec128_t<T> ret;
    uint4* out = reinterpret_cast<uint4*>(&ret);
    asm volatile("ld.global.ca.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(out->x), "=r"(out->y), "=r"(out->z), "=r"(out->w)
                 : "l"(ptr));
    return ret;
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
        t += red[w]; // broadcast read across CTA
    }
    return t;
}

// Variant A: Fast in-register reduction via blockSumAll (32 B static shmem, single __syncthreads)
template <typename T, int NUM_WARPS, int ITERS, int HIDDEN_DIM>
__global__ void __launch_bounds__(NUM_WARPS * 32)
fusedAddPostMoeNormRegKernel(
    Vec128_t<T> const* __restrict__ routed_out,
    Vec128_t<T> const* __restrict__ shared_out,
    Vec128_t<T> const* __restrict__ gamma,
    Vec128_t<T>* __restrict__ moe_out,
    float const eps)
{
    using V = Vec128_t<T>;
    constexpr int THREADS = NUM_WARPS * 32;
    constexpr int N_VEC   = THREADS * ITERS;
    constexpr float INV_HIDDEN_DIM = 1.0f / float(HIDDEN_DIM);

    __shared__ float red[NUM_WARPS];

    size_t const row = size_t(blockIdx.x) * N_VEC;
    int const tid = threadIdx.x;

    V c[ITERS];
    V g[ITERS];

    #pragma unroll
    for (int k = 0; k < ITERS; ++k)
    {
        int const i = tid + k * THREADS;
        V routed = load_128_cg(&routed_out[row + i]);
        V shared = load_128_cg(&shared_out[row + i]);
        c[k] = routed.add(shared);
        g[k] = load_128_ca(&gamma[i]);
    }

    float ss = 0.f;
    #pragma unroll
    for (int k = 0; k < ITERS; ++k)
    {
        ss += c[k].sum_of_squares();
    }

    float const sum = blockSumAll<NUM_WARPS>(ss, red);
    float const inv = rsqrtf(sum * INV_HIDDEN_DIM + eps);

    #pragma unroll
    for (int k = 0; k < ITERS; ++k)
    {
        moe_out[row + tid + k * THREADS] = c[k].rmsnorm(inv, g[k]);
    }
}

// Variant B: Using TensorRT-LLM library blockReduceSumV2<float, 1>
template <typename T, int NUM_WARPS, int ITERS, int HIDDEN_DIM>
__global__ void __launch_bounds__(NUM_WARPS * 32)
fusedAddPostMoeNormLibKernel(
    Vec128_t<T> const* __restrict__ routed_out,
    Vec128_t<T> const* __restrict__ shared_out,
    Vec128_t<T> const* __restrict__ gamma,
    Vec128_t<T>* __restrict__ moe_out,
    float const eps)
{
    using V = Vec128_t<T>;
    constexpr int THREADS = NUM_WARPS * 32;
    constexpr int N_VEC   = THREADS * ITERS;
    constexpr float INV_HIDDEN_DIM = 1.0f / float(HIDDEN_DIM);

    __shared__ float s_inv;

    size_t const row = size_t(blockIdx.x) * N_VEC;
    int const tid = threadIdx.x;

    V c[ITERS];
    V g[ITERS];

    #pragma unroll
    for (int k = 0; k < ITERS; ++k)
    {
        int const i = tid + k * THREADS;
        V routed = routed_out[row + i];
        V shared = shared_out[row + i];
        c[k] = routed.add(shared);
        g[k] = gamma[i];
    }

    float ss = 0.f;
    #pragma unroll
    for (int k = 0; k < ITERS; ++k)
    {
        ss += c[k].sum_of_squares();
    }

    // Using TRT-LLM library blockReduceSumV2
    float packed[1] = {ss};
    blockReduceSumV2<float, 1>(packed);

    if (threadIdx.x == 0)
    {
        s_inv = rsqrtf(packed[0] * INV_HIDDEN_DIM + eps);
    }
    __syncthreads();

    float const inv = s_inv;

    #pragma unroll
    for (int k = 0; k < ITERS; ++k)
    {
        moe_out[row + tid + k * THREADS] = c[k].rmsnorm(inv, g[k]);
    }
}

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
    constexpr int vec128_elements = 16 / sizeof(T);

    // Fast-path for standard hidden_dim=2560 (bf16/fp16):
    // NUM_WARPS = 5 (160 threads), ITERS = 2, HIDDEN_DIM = 2560 (N_VEC = 320)
    if (hidden_dim == 2560 && vec128_elements == 8 && (std::is_same<T, half>::value
#ifdef ENABLE_BF16
        || std::is_same<T, __nv_bfloat16>::value
#endif
    ))
    {
        using V128 = Vec128_t<T>;
        fusedAddPostMoeNormRegKernel<T, 5, 2, 2560><<<grid, 160, 0, stream>>>(
            reinterpret_cast<V128 const*>(routed_out),
            reinterpret_cast<V128 const*>(shared_out),
            reinterpret_cast<V128 const*>(gamma),
            reinterpret_cast<V128*>(moe_out),
            eps);
    }
    else
    {
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

// Pure Copy-Kernel Baseline for Post-MoE Norm:
// Reads 2 streams (in1, in2) and writes 1 stream (out) via 128-bit loads/stores.
// With 1152 tokens and hidden_dim=2560 (bf16), exactly 11.8 MB read and 5.9 MB written.
// Matches identical grid (tokens) and block (160 threads, 2 iterations per thread).
template <typename T, int NUM_WARPS, int ITERS>
__global__ void __launch_bounds__(NUM_WARPS * 32)
copyBaselinePostMoeKernel(
    Vec128_t<T> const* __restrict__ in1,
    Vec128_t<T> const* __restrict__ in2,
    Vec128_t<T>* __restrict__ out)
{
    constexpr int THREADS = NUM_WARPS * 32;
    constexpr int N_VEC   = THREADS * ITERS;

    size_t const row = size_t(blockIdx.x) * N_VEC;
    int const tid = threadIdx.x;

    #pragma unroll
    for (int k = 0; k < ITERS; ++k)
    {
        size_t const idx = row + tid + k * THREADS;
        out[idx] = in1[idx].add(in2[idx]);
    }
}

template <typename T>
void invokeCopyBaselinePostMoe(
    T const* in1,
    T const* in2,
    T* out,
    int const tokens,
    int const hidden_dim,
    cudaStream_t stream)
{
    dim3 grid(tokens);
    constexpr int vec128_elements = 16 / sizeof(T);
    using V128 = Vec128_t<T>;

    if (hidden_dim == 2560 && vec128_elements == 8)
    {
        copyBaselinePostMoeKernel<T, 5, 2><<<grid, 160, 0, stream>>>(
            reinterpret_cast<V128 const*>(in1),
            reinterpret_cast<V128 const*>(in2),
            reinterpret_cast<V128*>(out));
    }
}

#ifdef ENABLE_BF16
template void invokeCopyBaselinePostMoe<__nv_bfloat16>(
    __nv_bfloat16 const* in1,
    __nv_bfloat16 const* in2,
    __nv_bfloat16* out,
    int const tokens,
    int const hidden_dim,
    cudaStream_t stream);
#endif

template void invokeCopyBaselinePostMoe<half>(
    half const* in1,
    half const* in2,
    half* out,
    int const tokens,
    int const hidden_dim,
    cudaStream_t stream);

template void invokeCopyBaselinePostMoe<float>(
    float const* in1,
    float const* in2,
    float* out,
    int const tokens,
    int const hidden_dim,
    cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
