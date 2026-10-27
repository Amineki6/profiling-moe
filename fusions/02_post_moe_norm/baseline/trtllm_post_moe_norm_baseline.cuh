#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace kolibri
{
namespace baseline
{

template <typename T>
void invokeTrtllmRmsNorm(
    T* out,
    T const* in,
    T const* gamma,
    int const tokens,
    int const hidden_dim,
    float const eps,
    cudaStream_t stream);

} // namespace baseline
} // namespace kolibri
