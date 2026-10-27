#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace kolibri
{
namespace fusions
{

// Align to 16-byte boundary for 128-bit LDG.E.128 and STG.E.128 instructions
template <typename T>
struct alignas(16) Vec128_t;

// Vector specialization for BF16 (8 elements = 16 bytes)
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
            float const f = __bfloat162float(data[k]);
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
            float const val_f = __bfloat162float(data[k]);
            float const g_f   = __bfloat162float(gamma.data[k]);
            out.data[k]       = __float2bfloat16(val_f * s_var * g_f);
        }
        return out;
    }

    __device__ inline Vec128_t<__nv_bfloat16> add(Vec128_t<__nv_bfloat16> const& b) const
    {
        Vec128_t<__nv_bfloat16> out;
        #pragma unroll
        for (int k = 0; k < 8; ++k)
        {
            out.data[k] = __float2bfloat16(__bfloat162float(data[k]) + __bfloat162float(b.data[k]));
        }
        return out;
    }
};

// Vector specialization for FP16 (8 elements = 16 bytes)
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
            float const f = __half2float(data[k]);
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
            float const val_f = __half2float(data[k]);
            float const g_f   = __half2float(gamma.data[k]);
            out.data[k]       = __float2half(val_f * s_var * g_f);
        }
        return out;
    }

    __device__ inline Vec128_t<half> add(Vec128_t<half> const& b) const
    {
        Vec128_t<half> out;
        #pragma unroll
        for (int k = 0; k < 8; ++k)
        {
            out.data[k] = __float2half(__half2float(data[k]) + __half2float(b.data[k]));
        }
        return out;
    }
};

// Vector specialization for FP32 (4 elements = 16 bytes)
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

// Host launcher function
template <typename T>
void invokeFusedSandwichNormReg(
    T const* attn_out,
    T const* residual_in,
    T const* gamma1,
    T const* gamma2,
    T* updated_residual_out,
    T* moe_in_out,
    int const tokens,
    int const hidden_dim,
    float const eps,
    cudaStream_t stream);

} // namespace fusions
} // namespace kolibri
