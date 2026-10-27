#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace kolibri
{
namespace baseline
{

// Stage 1: Native TRT-LLM Routing (from kolibriRoutingKernels.cu)
void invokeTrtllmRouting(
    float const* scores_in,
    float const* bias,
    float* out_weights,
    int* out_indices,
    int const num_tokens,
    cudaStream_t stream);

// Stage 2: Native TRT-LLM Build Expert Maps
// Dispatches:
//   - fusedBuildExpertMapsSortFirstTokenKernel (single-block CUB RadixRank) for N <= 32
//   - threeStepBuildExpertMapsSortFirstToken (3-kernel prefix sum + merge pipeline) for N > 32
void invokeTrtllmBuildExpertMaps(
    int const* token_selected_experts,
    int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row,
    int64_t* expert_first_token_offset,
    void* scratch,
    int const num_tokens,
    cudaStream_t stream);

// Stage 3: Native TRT-LLM Activation Row Expansion (expandInputRowsKernel from moe_kernels.cu)
template <typename T>
void invokeTrtllmActivationExpansion(
    T const* unpermuted_input,
    T* permuted_output,
    int const* permuted_row_to_unpermuted_row,
    int const num_expanded_rows,
    int const hidden_size,
    cudaStream_t stream);

} // namespace baseline
} // namespace kolibri
