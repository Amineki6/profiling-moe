#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/kernels/archCondition.h"
#include "tensorrt_llm/kernels/moeTopKFuncs.cuh"
#include <climits>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda/std/limits>
#include <math.h>
#include <stdexcept>
#include <string>

#undef TLLM_CHECK_WITH_INFO
#define TLLM_CHECK_WITH_INFO(cond, msg, ...) do { if (!(cond)) throw std::runtime_error(std::string("Assertion failed: ") + (msg)); } while(0)

#undef sync_check_cuda_error
#define sync_check_cuda_error(stream) do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err)); \
} while(0)

namespace cg = cooperative_groups;

TRTLLM_NAMESPACE_BEGIN
namespace kernels
{

static constexpr int WARP_SIZE = 32;
static constexpr int kTOPK = 6;

// Edited gate forward kernel with register-kept scores (Variant B)
template <int nExperts, int topK, bool hash>
__global__ void gate_forward_kernel_vB(
    float const* __restrict__ scores_in,
    float const* __restrict__ bias,
    int const* __restrict__ input_ids,
    int const* __restrict__ tid2eid,
    float* __restrict__ out_weights,
    int* __restrict__ out_indices,
    int batch_size, float route_scale)
{
    constexpr int kExpertsPerThread = nExperts / WARP_SIZE;
    constexpr int kWarpsPerBlock = 4;

    __shared__ float smem_scores[kWarpsPerBlock][nExperts];

    int const global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int const local_warp_id = (threadIdx.x / WARP_SIZE) % kWarpsPerBlock;
    int const lane_id = threadIdx.x % WARP_SIZE;

    if (global_warp_id >= batch_size)
        return;

    auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
    float* my_smem = smem_scores[local_warp_id];
    float const* scores_row = scores_in + global_warp_id * nExperts;

    // Array holding scores in registers
    float my_scores[kExpertsPerThread];

#pragma unroll
    for (int e = 0; e < kExpertsPerThread; ++e)
    {
        int expert_id = lane_id + e * WARP_SIZE;
        float s = scores_row[expert_id];
        float sp = log1pf(expf(s));
        float score = sqrtf(sp);
        my_scores[e] = score; // Keep score in thread register
        my_smem[expert_id] = score;
    }
    __syncwarp();

    float my_topk_value = 0.0f;
    int my_topk_index = 0;

    if constexpr (hash)
    {
        int token_id = input_ids[global_warp_id];
        int const* expert_ids = tid2eid + token_id * topK;
        if (lane_id < topK)
        {
            int expert_id = expert_ids[lane_id];
            my_topk_index = expert_id;
            my_topk_value = my_smem[expert_id];
        }
    }
    else
    {
        float scores[kExpertsPerThread];
        int indices[kExpertsPerThread];

#pragma unroll
        for (int e = 0; e < kExpertsPerThread; ++e)
        {
            int expert_id = lane_id + e * WARP_SIZE;
            indices[e] = expert_id;
            scores[e] = my_scores[e] + bias[expert_id]; // Read from register array
        }

        float topk_values[topK];
        int32_t topk_indices[topK];
        constexpr float minValue = -1e30f;
        reduce_topk::reduceTopK<topK, float, kExpertsPerThread>(
            warp, topk_values, topk_indices, scores, indices, minValue, topK);

        if (lane_id < topK)
        {
            int expert_id = topk_indices[lane_id];
            my_topk_index = expert_id;
            my_topk_value = my_smem[expert_id];
        }
    }

    float weight_sum = cg::reduce(warp, my_topk_value, cg::plus<float>{});

    if (lane_id < topK)
    {
        out_weights[global_warp_id * topK + lane_id] = (my_topk_value / weight_sum) * route_scale;
        out_indices[global_warp_id * topK + lane_id] = my_topk_index;
    }
}

template <int nExperts, bool hash>
void launch_gate_forward_kernel_vB(float* scores_in, float* bias, int* input_ids, int* tid2eid, float* out_weights,
    int* out_indices, int batch_size, float route_scale, cudaStream_t stream)
{
    constexpr int warps_per_block = 4;
    constexpr int threads_per_block = warps_per_block * WARP_SIZE;
    int const blocks = (batch_size + warps_per_block - 1) / warps_per_block;

    gate_forward_kernel_vB<nExperts, kTOPK, hash><<<blocks, threads_per_block, 0, stream>>>(
        scores_in, bias, input_ids, tid2eid, out_weights, out_indices, batch_size, route_scale);
}

void gate_forward_vB(void* scores_in, void* bias, void* input_ids, void* tid2eid, void* out_weights,
    void* out_indices, int batch_size, int n_experts, float route_scale, bool is_hash, cudaStream_t stream)
{
    auto* scores = static_cast<float*>(scores_in);
    auto* bias_ptr = static_cast<float*>(bias);
    auto* input_ids_ptr = static_cast<int*>(input_ids);
    auto* tid2eid_ptr = static_cast<int*>(tid2eid);
    auto* weights = static_cast<float*>(out_weights);
    auto* indices = static_cast<int*>(out_indices);

    switch (n_experts)
    {
    case 256:
        if (is_hash)
            launch_gate_forward_kernel_vB<256, true>(scores, nullptr, input_ids_ptr, tid2eid_ptr, weights, indices, batch_size, route_scale, stream);
        else
            launch_gate_forward_kernel_vB<256, false>(scores, bias_ptr, nullptr, nullptr, weights, indices, batch_size, route_scale, stream);
        break;
    case 384:
        if (is_hash)
            launch_gate_forward_kernel_vB<384, true>(scores, nullptr, input_ids_ptr, tid2eid_ptr, weights, indices, batch_size, route_scale, stream);
        else
            launch_gate_forward_kernel_vB<384, false>(scores, bias_ptr, nullptr, nullptr, weights, indices, batch_size, route_scale, stream);
        break;
    default:
        TLLM_CHECK_WITH_INFO(false, "gate_forward_vB only supports n_experts 256 or 384");
    }
    sync_check_cuda_error(stream);
}

} // namespace kernels
TRTLLM_NAMESPACE_END
