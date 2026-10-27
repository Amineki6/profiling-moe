#include "trtllm_moe_prologue_baseline.cuh"

// Native TensorRT-LLM routing kernel header from /home/mkina/profiling/TensorRT-LLM/cpp/
#include "tensorrt_llm/kernels/moe/routing/kolibriRoutingKernels.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cassert>
#include <stdexcept>
#include <string>
#include <algorithm>

namespace kolibri
{
namespace baseline
{

static constexpr int NUM_EXPERTS = 384;
static constexpr int TOP_K = 6;

static inline int64_t ceilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

static inline int64_t computeNumTokensPerBlock(int64_t const num_tokens, int64_t const num_experts_per_node)
{
    for (int64_t num_tokens_per_block = 32; num_tokens_per_block <= 1024; num_tokens_per_block *= 2)
    {
        int64_t const num_blocks_per_seq = ceilDiv(num_tokens, num_tokens_per_block);
        if (num_blocks_per_seq * num_experts_per_node <= num_tokens_per_block)
        {
            return num_tokens_per_block;
        }
    }
    return 1024;
}

// =================================================================================================
// 1. Stage 1: Native TensorRT-LLM Routing
//
// Directly calls invokeKolibriRouting from TRT-LLM cpp source:
//   /home/mkina/profiling/TensorRT-LLM/cpp/tensorrt_llm/kernels/moe/routing/kolibriRoutingKernels.cu
// =================================================================================================
void invokeTrtllmRouting(
    float const* scores_in,
    float const* bias,
    float* out_weights,
    int* out_indices,
    int const num_tokens,
    cudaStream_t stream)
{
    tensorrt_llm::kernels::invokeKolibriRouting<float, float>(
        scores_in,
        bias,
        out_weights,
        out_indices,
        num_tokens,
        NUM_EXPERTS,
        TOP_K,
        true, // norm_topk_prob
        stream
    );
}

// =================================================================================================
// 2A. Stage 2A: Native TRT-LLM Single-Block CUB RadixRank Build Expert Maps (for N <= 32)
//
// Matches fusedBuildExpertMapsSortFirstTokenKernel from TRT-LLM moe_kernels.cu line 480.
// Single-block CUB RadixRank with 32.9 KB shared memory footprint.
// =================================================================================================
template <int BLOCK_SIZE = 32, int EXPERTS_PER_TOKEN = 6, int LOG2_NUM_EXPERTS = 9>
__global__ void trtllm_build_expert_maps_kernel(
    int const* const token_selected_experts,
    int* const permuted_row_to_unpermuted_row,
    int* const unpermuted_row_to_permuted_row,
    int64_t* const expert_first_token_offset,
    int64_t const num_tokens,
    int const num_experts)
{
    assert(gridDim.x == 1);

    int const token = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    bool const is_valid_token = token < num_tokens;

    int local_token_selected_experts[EXPERTS_PER_TOKEN];
    int local_token_permuted_indices[EXPERTS_PER_TOKEN];

    #pragma unroll
    for (int i = 0; i < EXPERTS_PER_TOKEN; i++)
    {
        int const expert = is_valid_token ? token_selected_experts[token * EXPERTS_PER_TOKEN + i] : num_experts;
        bool const is_valid_expert = (expert >= 0 && expert < num_experts);
        local_token_selected_experts[i] = !is_valid_token ? (num_experts + 1)
            : is_valid_expert                             ? expert
                                                          : num_experts;
    }

    using BlockRadixRank = cub::BlockRadixRank<BLOCK_SIZE, LOG2_NUM_EXPERTS, false>;
    extern __shared__ unsigned char temp_storage[];
    auto& sort_temp = *reinterpret_cast<typename BlockRadixRank::TempStorage*>(temp_storage);

    int local_expert_first_token_offset[BlockRadixRank::BINS_TRACKED_PER_THREAD];
    cub::BFEDigitExtractor<int> extractor(0, LOG2_NUM_EXPERTS);

    BlockRadixRank(sort_temp).RankKeys(
        local_token_selected_experts, local_token_permuted_indices, extractor, local_expert_first_token_offset);

    if (is_valid_token)
    {
        #pragma unroll
        for (int i = 0; i < EXPERTS_PER_TOKEN; i++)
        {
            int const unpermuted_row = i * num_tokens + token;
            int const permuted_row = local_token_permuted_indices[i];
            permuted_row_to_unpermuted_row[permuted_row] = unpermuted_row;
            if (unpermuted_row_to_permuted_row != nullptr)
            {
                unpermuted_row_to_permuted_row[unpermuted_row] = permuted_row;
            }
        }
    }

    #pragma unroll
    for (int expert_id = 0; expert_id < BlockRadixRank::BINS_TRACKED_PER_THREAD; expert_id++)
    {
        int out_expert_id = expert_id + token * BlockRadixRank::BINS_TRACKED_PER_THREAD;
        if (out_expert_id < num_experts + 1)
        {
            expert_first_token_offset[out_expert_id] = local_expert_first_token_offset[expert_id];
        }
    }
}

template <int BLOCK_SIZE>
static void launchTrtllmSingleBlockBuildExpertMaps(
    int const* token_selected_experts,
    int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row,
    int64_t* expert_first_token_offset,
    int const num_tokens,
    cudaStream_t stream)
{
    constexpr int LOG2_NUM_EXPERTS = 9; // 2^9 = 512 >= 384
    using BlockRadixRank = cub::BlockRadixRank<BLOCK_SIZE, LOG2_NUM_EXPERTS, false>;
    size_t const shared_size = sizeof(typename BlockRadixRank::TempStorage);

    auto kernel = &trtllm_build_expert_maps_kernel<BLOCK_SIZE, 6, LOG2_NUM_EXPERTS>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_size));

    dim3 const grid(1);
    dim3 const block(BLOCK_SIZE);

    kernel<<<grid, block, shared_size, stream>>>(
        token_selected_experts,
        permuted_row_to_unpermuted_row,
        unpermuted_row_to_permuted_row,
        expert_first_token_offset,
        num_tokens,
        NUM_EXPERTS);
}

// =================================================================================================
// 2B. Stage 2B: Native TRT-LLM 3-Kernel Build Expert Maps Pipeline (for N > 32)
//
// Matches threeStepBuildExpertMapsSortFirstToken from TRT-LLM moe_kernels.cu line 908:
//   1. blockExpertPrefixSumKernel (block scan per expert)
//   2. globalExpertPrefixSumKernel / globalExpertPrefixSumLargeKernel (prefix sum across blocks)
//   3. mergeExpertPrefixSumKernel (scatter to permuted_row_to_unpermuted_row)
// =================================================================================================
template <int kNumTokensPerBlock>
__global__ void blockExpertPrefixSumKernel(
    int const* token_selected_experts,
    int* blocked_expert_counts,
    int* blocked_row_to_unpermuted_row,
    int64_t const num_tokens,
    int64_t const num_experts_per_token,
    int const start_expert_id)
{
    using BlockScan = cub::BlockScan<int, kNumTokensPerBlock>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    int const target_expert_id = blockIdx.x;
    int const block_id = blockIdx.y;
    int const num_blocks_per_seq = gridDim.y;
    int const token_id = block_id * kNumTokensPerBlock + threadIdx.x;

    int expanded_token_id = -1;
    if (token_id < num_tokens)
    {
        for (int i = 0; i < num_experts_per_token; i++)
        {
            int const expert_id = token_selected_experts[token_id * num_experts_per_token + i] - start_expert_id;
            if (expert_id == target_expert_id)
            {
                expanded_token_id = i * num_tokens + token_id;
                break;
            }
        }
    }

    int const has_matched = (expanded_token_id >= 0) ? 1 : 0;
    int index = 0;
    BlockScan(temp_storage).ExclusiveSum(has_matched, index);

    if (has_matched)
    {
        blocked_row_to_unpermuted_row[target_expert_id * num_tokens + block_id * kNumTokensPerBlock + index]
            = expanded_token_id;
    }
    if (threadIdx.x == kNumTokensPerBlock - 1)
    {
        blocked_expert_counts[target_expert_id * num_blocks_per_seq + block_id] = index + has_matched;
    }
}

template <int kNumThreadsPerBlock>
__global__ void globalExpertPrefixSumKernel(
    int const* blocked_expert_counts,
    int* blocked_expert_counts_cumsum,
    int64_t* expert_first_token_offset,
    int64_t const num_experts_per_node,
    int64_t const num_blocks_per_seq)
{
    using BlockScan = cub::BlockScan<int, kNumThreadsPerBlock>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    int const cnt = threadIdx.x < num_experts_per_node * num_blocks_per_seq ? blocked_expert_counts[threadIdx.x] : 0;
    int cumsum = 0;
    BlockScan(temp_storage).ExclusiveSum(cnt, cumsum);

    if (threadIdx.x < num_experts_per_node * num_blocks_per_seq)
    {
        blocked_expert_counts_cumsum[threadIdx.x] = cumsum;
        if (threadIdx.x % num_blocks_per_seq == 0)
        {
            expert_first_token_offset[threadIdx.x / num_blocks_per_seq] = cumsum;
        }
        if (threadIdx.x == num_experts_per_node * num_blocks_per_seq - 1)
        {
            expert_first_token_offset[num_experts_per_node] = cumsum + cnt;
        }
    }
}

template <int kNumThreadsPerBlock>
__global__ void globalExpertPrefixSumLargeKernel(
    int const* blocked_expert_counts,
    int* blocked_expert_counts_cumsum,
    int64_t* expert_first_token_offset,
    int64_t const num_experts_per_node,
    int64_t const num_blocks_per_seq,
    int64_t const num_elem_per_thread)
{
    using BlockScan = cub::BlockScan<int, kNumThreadsPerBlock>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    int offset = threadIdx.x * num_elem_per_thread;
    int cnt = 0;

    for (int i = 0; i < num_elem_per_thread; i++)
    {
        if (offset + i < num_experts_per_node * num_blocks_per_seq)
        {
            cnt += blocked_expert_counts[offset + i];
        }
    }

    int cumsum = 0;
    BlockScan(temp_storage).ExclusiveSum(cnt, cumsum);

    for (int i = 0; i < num_elem_per_thread; i++)
    {
        if (offset + i < num_experts_per_node * num_blocks_per_seq)
        {
            blocked_expert_counts_cumsum[offset + i] = cumsum;
            if ((offset + i) % num_blocks_per_seq == 0)
            {
                expert_first_token_offset[(offset + i) / num_blocks_per_seq] = cumsum;
            }
            cumsum += blocked_expert_counts[offset + i];
            if ((offset + i) == num_experts_per_node * num_blocks_per_seq - 1)
            {
                expert_first_token_offset[num_experts_per_node] = cumsum;
            }
        }
    }
}

__global__ void mergeExpertPrefixSumKernel(
    int const* blocked_expert_counts,
    int const* blocked_expert_counts_cumsum,
    int const* blocked_row_to_unpermuted_row,
    int* permuted_token_selected_experts,
    int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row,
    int const num_tokens)
{
    int const target_expert_id = blockIdx.x;
    int const block_id = blockIdx.y;
    int const num_blocks_per_seq = gridDim.y;
    int const token_id = block_id * blockDim.x + threadIdx.x;

    int const cnt = blocked_expert_counts[target_expert_id * num_blocks_per_seq + block_id];
    int const offset = blocked_expert_counts_cumsum[target_expert_id * num_blocks_per_seq + block_id];
    if (threadIdx.x < cnt)
    {
        int const unpermuted_row = blocked_row_to_unpermuted_row[target_expert_id * num_tokens + token_id];
        int const permuted_row = offset + threadIdx.x;
        permuted_row_to_unpermuted_row[permuted_row] = unpermuted_row;
        if (permuted_token_selected_experts != nullptr)
        {
            permuted_token_selected_experts[permuted_row] = target_expert_id;
        }
        if (unpermuted_row_to_permuted_row != nullptr)
        {
            unpermuted_row_to_permuted_row[unpermuted_row] = permuted_row;
        }
    }
}

static void threeStepBuildExpertMapsSortFirstToken(
    int const* token_selected_experts,
    int* permuted_token_selected_experts,
    int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row,
    int64_t* expert_first_token_offset,
    void* scratch,
    int64_t const num_tokens,
    int64_t const num_experts_per_node,
    int64_t const num_experts_per_token,
    int const start_expert_id,
    cudaStream_t stream)
{
    int64_t const num_tokens_per_block = computeNumTokensPerBlock(num_tokens, num_experts_per_node);
    int64_t const num_blocks_per_seq = ceilDiv(num_tokens, num_tokens_per_block);

    int* scratch_i = reinterpret_cast<int*>(scratch);
    int* blocked_expert_counts = scratch_i;
    int* blocked_expert_counts_cumsum = blocked_expert_counts + num_experts_per_node * num_blocks_per_seq;
    int* blocked_row_to_unpermuted_row = blocked_expert_counts_cumsum + num_experts_per_node * num_blocks_per_seq;

    // Step 1: blockExpertPrefixSum
    dim3 const b_grid(num_experts_per_node, num_blocks_per_seq);
    dim3 const b_threads(num_tokens_per_block);

    if (num_tokens_per_block <= 32)
    {
        blockExpertPrefixSumKernel<32><<<b_grid, b_threads, 0, stream>>>(
            token_selected_experts, blocked_expert_counts, blocked_row_to_unpermuted_row,
            num_tokens, num_experts_per_token, start_expert_id);
    }
    else if (num_tokens_per_block <= 64)
    {
        blockExpertPrefixSumKernel<64><<<b_grid, b_threads, 0, stream>>>(
            token_selected_experts, blocked_expert_counts, blocked_row_to_unpermuted_row,
            num_tokens, num_experts_per_token, start_expert_id);
    }
    else if (num_tokens_per_block <= 128)
    {
        blockExpertPrefixSumKernel<128><<<b_grid, b_threads, 0, stream>>>(
            token_selected_experts, blocked_expert_counts, blocked_row_to_unpermuted_row,
            num_tokens, num_experts_per_token, start_expert_id);
    }
    else if (num_tokens_per_block <= 256)
    {
        blockExpertPrefixSumKernel<256><<<b_grid, b_threads, 0, stream>>>(
            token_selected_experts, blocked_expert_counts, blocked_row_to_unpermuted_row,
            num_tokens, num_experts_per_token, start_expert_id);
    }
    else if (num_tokens_per_block <= 512)
    {
        blockExpertPrefixSumKernel<512><<<b_grid, b_threads, 0, stream>>>(
            token_selected_experts, blocked_expert_counts, blocked_row_to_unpermuted_row,
            num_tokens, num_experts_per_token, start_expert_id);
    }
    else
    {
        blockExpertPrefixSumKernel<1024><<<b_grid, b_threads, 0, stream>>>(
            token_selected_experts, blocked_expert_counts, blocked_row_to_unpermuted_row,
            num_tokens, num_experts_per_token, start_expert_id);
    }

    // Step 2: globalExpertPrefixSum
    int64_t const num_elements = num_experts_per_node * num_blocks_per_seq;
    if (num_elements <= 1024)
    {
        int const threads = std::max(32, static_cast<int>(num_elements));
        globalExpertPrefixSumKernel<1024><<<1, 1024, 0, stream>>>(
            blocked_expert_counts, blocked_expert_counts_cumsum,
            expert_first_token_offset, num_experts_per_node, num_blocks_per_seq);
    }
    else
    {
        int64_t const num_elem_per_thread = ceilDiv(num_elements, 1024);
        globalExpertPrefixSumLargeKernel<1024><<<1, 1024, 0, stream>>>(
            blocked_expert_counts, blocked_expert_counts_cumsum,
            expert_first_token_offset, num_experts_per_node, num_blocks_per_seq, num_elem_per_thread);
    }

    // Step 3: mergeExpertPrefixSum
    mergeExpertPrefixSumKernel<<<b_grid, b_threads, 0, stream>>>(
        blocked_expert_counts, blocked_expert_counts_cumsum,
        blocked_row_to_unpermuted_row, permuted_token_selected_experts,
        permuted_row_to_unpermuted_row, unpermuted_row_to_permuted_row,
        num_tokens);
}

// Master Stage 2 Dispatcher
void invokeTrtllmBuildExpertMaps(
    int const* token_selected_experts,
    int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row,
    int64_t* expert_first_token_offset,
    void* scratch,
    int const num_tokens,
    cudaStream_t stream)
{
    // TRT-LLM single-block CUB radix rank only runs for N <= 32 due to smem capacity limit (<100 KB)
    if (num_tokens <= 32)
    {
        launchTrtllmSingleBlockBuildExpertMaps<32>(
            token_selected_experts,
            permuted_row_to_unpermuted_row,
            unpermuted_row_to_permuted_row,
            expert_first_token_offset,
            num_tokens,
            stream);
    }
    else
    {
        // For N > 32, TRT-LLM falls back to 3-step prefix sum and merge pipeline
        threeStepBuildExpertMapsSortFirstToken(
            token_selected_experts,
            nullptr,
            permuted_row_to_unpermuted_row,
            unpermuted_row_to_permuted_row,
            expert_first_token_offset,
            scratch,
            num_tokens,
            NUM_EXPERTS,
            TOP_K,
            0,
            stream);
    }
}

// =================================================================================================
// 3. Stage 3: Native TRT-LLM Activation Row Expansion
//
// Matches expandInputRowsKernel from TRT-LLM moe_kernels.cu line 1601.
// Standard scalar memory loads without 128-bit vectorization.
// =================================================================================================
template <typename T>
__global__ void trtllm_expand_input_rows_kernel(
    T const* __restrict__ unpermuted_input,
    T* __restrict__ permuted_output,
    int const* __restrict__ permuted_row_to_unpermuted,
    int const num_tokens,
    int const num_expanded_rows,
    int const hidden_size)
{
    for (int dst_row = blockIdx.x; dst_row < num_expanded_rows; dst_row += gridDim.x)
    {
        int const unpermuted_row = permuted_row_to_unpermuted[dst_row];
        int const src_token_id   = unpermuted_row % num_tokens;

        T const* src = unpermuted_input + src_token_id * hidden_size;
        T* dst       = permuted_output + dst_row * hidden_size;

        for (int i = threadIdx.x; i < hidden_size; i += blockDim.x)
        {
            dst[i] = src[i];
        }
    }
}

template <typename T>
void invokeTrtllmActivationExpansion(
    T const* unpermuted_input,
    T* permuted_output,
    int const* permuted_row_to_unpermuted_row,
    int const num_expanded_rows,
    int const hidden_size,
    cudaStream_t stream)
{
    constexpr int BLOCK_SIZE = 256;
    int const num_tokens = num_expanded_rows / TOP_K;
    int const grid_size = std::min(num_expanded_rows, 768);

    trtllm_expand_input_rows_kernel<T><<<grid_size, BLOCK_SIZE, 0, stream>>>(
        unpermuted_input,
        permuted_output,
        permuted_row_to_unpermuted_row,
        num_tokens,
        num_expanded_rows,
        hidden_size
    );
}

template void invokeTrtllmActivationExpansion<__nv_bfloat16>(
    __nv_bfloat16 const* unpermuted_input,
    __nv_bfloat16* permuted_output,
    int const* permuted_row_to_unpermuted_row,
    int const num_expanded_rows,
    int const hidden_size,
    cudaStream_t stream);

template void invokeTrtllmActivationExpansion<half>(
    half const* unpermuted_input,
    half* permuted_output,
    int const* permuted_row_to_unpermuted_row,
    int const num_expanded_rows,
    int const hidden_size,
    cudaStream_t stream);

} // namespace baseline
} // namespace kolibri
