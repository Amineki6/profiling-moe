#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace kolibri
{
namespace fusions
{

// Architectural constants for Kolibri-5E
static constexpr int NUM_EXPERTS = 384;
static constexpr int TOP_K = 6;
static constexpr int HYBRID_TIPPING_POINT = 512; // Tokens below 512 use single-block; >= 512 use 3-stage multi-block

// =================================================================================================
// 1. Router Entrypoint (Stage 1)
// =================================================================================================
void invokeKolibriRouter(
    float const* scores_in,
    float const* bias,
    float* out_weights,
    int* out_indices,
    int const num_tokens,
    cudaStream_t stream);

// =================================================================================================
// 2. Dispatch Map Entrypoints (Stage 2: Dynamic Hybrid Dispatcher)
//
// Dynamically routes based on sequence length:
//   - N < 512: Single-Block 256-Thread Kernel (3.0 KB SRAM, zero launch overhead)
//   - N >= 512: 3-Stage Multi-Block Pipeline (128 SM concurrency, linear scaling up to N=98k)
//
// Uses pre-allocated scratch memory from PyTorch caching allocator (0 µs overhead, no cudaMallocAsync).
// =================================================================================================
void invokeSingleBlockBuildExpertMaps(
    int const* token_selected_experts,
    int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row,
    int64_t* expert_first_token_offset,
    void* scratch,
    int const num_tokens,
    cudaStream_t stream);

void invokeMultiBlockBuildExpertMaps(
    int const* token_selected_experts,
    int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row,
    int64_t* expert_first_token_offset,
    void* scratch,
    int const num_tokens,
    cudaStream_t stream);

void invokeDynamicBuildExpertMaps(
    int const* token_selected_experts,
    int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row,
    int64_t* expert_first_token_offset,
    void* scratch,
    int const num_tokens,
    cudaStream_t stream,
    bool force_single_block = false,
    bool force_multi_block = false);

// =================================================================================================
// 3. Activation Expansion Entrypoint (Stage 3)
// =================================================================================================
template <typename T>
void invokeKolibriActivationExpansion(
    T const* unpermuted_input,
    T* permuted_output,
    int const* permuted_row_to_unpermuted_row,
    int const num_expanded_rows,
    int const hidden_size,
    cudaStream_t stream);

// =================================================================================================
// 4. End-to-End Prologue Pipeline Launcher
// =================================================================================================
template <typename T>
void invokeFusedMoePrologue(
    T const* unpermuted_input,
    float const* router_logits,
    float const* router_bias,
    T* permuted_output,
    float* routing_weights,
    int* selected_experts,
    int* permuted_row_to_unpermuted,
    int* unpermuted_to_permuted,
    int64_t* expert_offsets,
    void* scratch,
    int const num_tokens,
    int const hidden_size,
    cudaStream_t stream);

} // namespace fusions
} // namespace kolibri
