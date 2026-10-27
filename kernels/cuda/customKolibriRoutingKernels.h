#pragma once

#include "tensorrt_llm/common/config.h"
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN
namespace kernels
{

// Production Kolibri MoE gate forward function
void kolibri_gate_forward(void* scores_in, void* bias, void* input_ids, void* tid2eid, void* out_weights,
    void* out_indices, int batch_size, int n_experts, float route_scale, bool is_hash, cudaStream_t stream);

// Standard TRT-LLM drop-in interface
void gate_forward(void* scores_in, void* bias, void* input_ids, void* tid2eid, void* out_weights,
    void* out_indices, int batch_size, int n_experts, float route_scale, bool is_hash, cudaStream_t stream);

void gate_forward_vFloat4(void* scores_in, void* bias, void* input_ids, void* tid2eid, void* out_weights,
    void* out_indices, int batch_size, int n_experts, float route_scale, bool is_hash, cudaStream_t stream);

} // namespace kernels
TRTLLM_NAMESPACE_END
