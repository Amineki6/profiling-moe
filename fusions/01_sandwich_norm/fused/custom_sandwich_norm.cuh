#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace kolibri
{
namespace fusions
{

// Conversion helpers: scalar T to float
template <typename T>
__device__ __forceinline__ float to_float(T val);

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 val)
{
    return __bfloat162float(val);
}

template <>
__device__ __forceinline__ float to_float<half>(half val)
{
    return __half2float(val);
}

template <>
__device__ __forceinline__ float to_float<float>(float val)
{
    return val;
}

// Conversion helpers: float to scalar T
template <typename T>
__device__ __forceinline__ T from_float(float val);

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float val)
{
    return __float2bfloat16(val);
}

template <>
__device__ __forceinline__ half from_float<half>(float val)
{
    return __float2half(val);
}

template <>
__device__ __forceinline__ float from_float<float>(float val)
{
    return val;
}

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
