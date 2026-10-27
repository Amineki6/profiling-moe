#include "trtllm_post_moe_norm_baseline.cuh"

// Native TensorRT-LLM RMSNorm kernel header from /home/mkina/profiling/TensorRT-LLM/cpp/
#include "tensorrt_llm/common/cudaFp8Utils.h"
#include "tensorrt_llm/kernels/rmsnormKernels.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace kolibri
{
namespace baseline
{

// =================================================================================================
// Native TRT-LLM RMSNorm Kernel Invoker
//
// Directly calls native tensorrt_llm::kernels::invokeGeneralRmsNorm from TRT-LLM cpp source:
//   /home/mkina/profiling/TensorRT-LLM/cpp/tensorrt_llm/kernels/rmsnormKernels.cu
// =================================================================================================
template <typename T>
void invokeTrtllmRmsNorm(
    T* out,
    T const* in,
    T const* gamma,
    int const tokens,
    int const hidden_dim,
    float const eps,
    cudaStream_t stream)
{
    tensorrt_llm::kernels::invokeGeneralRmsNorm<T, int8_t>(
        out,
        in,
        gamma,
        nullptr, // beta (RMSNorm has no bias)
        eps,
        tokens,
        hidden_dim,
        tensorrt_llm::common::QuantMode(),
        stream
    );
}

// Explicit template instantiations
template void invokeTrtllmRmsNorm<__nv_bfloat16>(
    __nv_bfloat16* out,
    __nv_bfloat16 const* in,
    __nv_bfloat16 const* gamma,
    int const tokens,
    int const hidden_dim,
    float const eps,
    cudaStream_t stream);

template void invokeTrtllmRmsNorm<half>(
    half* out,
    half const* in,
    half const* gamma,
    int const tokens,
    int const hidden_dim,
    float const eps,
    cudaStream_t stream);

} // namespace baseline
} // namespace kolibri
