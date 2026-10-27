#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "customKolibriMultiBlockBuildExpertMapsKernels.h"
#include <vector>
#include <algorithm>

namespace th = torch;

std::vector<th::Tensor> kolibri_multi_block_build_expert_maps(
    th::Tensor token_selected_experts, // [num_tokens, top_k] (int32)
    int64_t num_experts = 384,
    int64_t max_blocks = 128,
    bool return_inverse_map = true
) {
    TORCH_CHECK(token_selected_experts.is_cuda(), "token_selected_experts must be a CUDA tensor");
    TORCH_CHECK(token_selected_experts.is_contiguous(), "token_selected_experts must be contiguous");
    TORCH_CHECK(token_selected_experts.scalar_type() == th::kInt32, "token_selected_experts must be int32");

    int const num_tokens = token_selected_experts.size(0);
    int const top_k = token_selected_experts.size(1);
    int const total_selections = num_tokens * top_k;

    auto permuted_row_to_unpermuted_row = th::empty({total_selections}, th::TensorOptions().dtype(th::kInt32).device(token_selected_experts.device()));
    auto unpermuted_row_to_permuted_row = return_inverse_map
        ? th::empty({total_selections}, th::TensorOptions().dtype(th::kInt32).device(token_selected_experts.device()))
        : th::empty({0}, th::TensorOptions().dtype(th::kInt32).device(token_selected_experts.device()));
    auto expert_first_token_offset = th::empty({num_experts + 1}, th::TensorOptions().dtype(th::kInt64).device(token_selected_experts.device()));

    int num_blocks = 128;
    if (max_blocks < 0)
    {
        num_blocks = -max_blocks;
    }
    else
    {
        int const max_b = max_blocks > 0 ? static_cast<int>(max_blocks) : 128;
        num_blocks = std::max(1, std::min(max_b, (total_selections + 2048 - 1) / 2048));
    }

    // Allocate scratch for block_expert_counts + block_expert_offsets
    auto scratch = th::empty({2 * num_blocks * static_cast<int>(num_experts)}, th::TensorOptions().dtype(th::kInt32).device(token_selected_experts.device()));

    auto stream = at::cuda::getCurrentCUDAStream(token_selected_experts.device().index());

    int* unpermuted_ptr = return_inverse_map ? reinterpret_cast<int*>(unpermuted_row_to_permuted_row.data_ptr()) : nullptr;

    tensorrt_llm::kernels::invokeKolibriMultiBlockBuildExpertMaps(
        reinterpret_cast<int const*>(token_selected_experts.data_ptr()),
        reinterpret_cast<int*>(permuted_row_to_unpermuted_row.data_ptr()),
        unpermuted_ptr,
        reinterpret_cast<int64_t*>(expert_first_token_offset.data_ptr()),
        reinterpret_cast<int*>(scratch.data_ptr()),
        num_tokens,
        static_cast<int>(num_experts),
        top_k,
        max_blocks,
        stream.stream()
    );

    return {permuted_row_to_unpermuted_row, unpermuted_row_to_permuted_row, expert_first_token_offset};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "kolibri_multi_block_build_expert_maps",
        &kolibri_multi_block_build_expert_maps,
        "Kolibri Multi-Block 3-Stage Dispatch Mapping Kernel (CUDA)",
        py::arg("token_selected_experts"),
        py::arg("num_experts") = 384,
        py::arg("max_blocks") = 128,
        py::arg("return_inverse_map") = true
    );
}
