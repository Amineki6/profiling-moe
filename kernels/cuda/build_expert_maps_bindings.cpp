#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <c10/cuda/CUDAStream.h>
#include "customKolibriBuildExpertMapsKernel.h"
#include <vector>

namespace th = torch;

std::vector<th::Tensor> kolibri_build_expert_maps(
    th::Tensor token_selected_experts, // [num_tokens, top_k] (int32)
    int64_t num_experts = 384
) {
    TORCH_CHECK(token_selected_experts.is_cuda(), "token_selected_experts must be a CUDA tensor");
    TORCH_CHECK(token_selected_experts.is_contiguous(), "token_selected_experts must be contiguous");
    TORCH_CHECK(token_selected_experts.scalar_type() == th::kInt32, "token_selected_experts must be int32");

    int const num_tokens = token_selected_experts.size(0);
    int const top_k = token_selected_experts.size(1);

    auto permuted_row_to_unpermuted_row = th::empty({num_tokens * top_k}, th::TensorOptions().dtype(th::kInt32).device(token_selected_experts.device()));
    auto unpermuted_row_to_permuted_row = th::empty({num_tokens * top_k}, th::TensorOptions().dtype(th::kInt32).device(token_selected_experts.device()));
    auto expert_first_token_offset = th::empty({num_experts + 1}, th::TensorOptions().dtype(th::kInt64).device(token_selected_experts.device()));

    auto stream = at::cuda::getCurrentCUDAStream(token_selected_experts.device().index());

    tensorrt_llm::kernels::invokeKolibriBuildExpertMaps(
        reinterpret_cast<int const*>(token_selected_experts.data_ptr()),
        reinterpret_cast<int*>(permuted_row_to_unpermuted_row.data_ptr()),
        reinterpret_cast<int*>(unpermuted_row_to_permuted_row.data_ptr()),
        reinterpret_cast<int64_t*>(expert_first_token_offset.data_ptr()),
        num_tokens,
        static_cast<int>(num_experts),
        top_k,
        stream.stream()
    );

    return {permuted_row_to_unpermuted_row, unpermuted_row_to_permuted_row, expert_first_token_offset};
}

std::vector<th::Tensor> trtllm_build_expert_maps_b32(
    th::Tensor token_selected_experts, // [num_tokens, top_k] (int32)
    int64_t num_experts = 384
) {
    TORCH_CHECK(token_selected_experts.is_cuda(), "token_selected_experts must be a CUDA tensor");
    TORCH_CHECK(token_selected_experts.is_contiguous(), "token_selected_experts must be contiguous");
    TORCH_CHECK(token_selected_experts.scalar_type() == th::kInt32, "token_selected_experts must be int32");

    int const num_tokens = token_selected_experts.size(0);
    int const top_k = token_selected_experts.size(1);

    auto permuted_row_to_unpermuted_row = th::empty({num_tokens * top_k}, th::TensorOptions().dtype(th::kInt32).device(token_selected_experts.device()));
    auto unpermuted_row_to_permuted_row = th::empty({num_tokens * top_k}, th::TensorOptions().dtype(th::kInt32).device(token_selected_experts.device()));
    auto expert_first_token_offset = th::empty({num_experts + 1}, th::TensorOptions().dtype(th::kInt64).device(token_selected_experts.device()));

    auto stream = at::cuda::getCurrentCUDAStream(token_selected_experts.device().index());

    tensorrt_llm::kernels::invokeTrtllmBuildExpertMapsB32(
        reinterpret_cast<int const*>(token_selected_experts.data_ptr()),
        reinterpret_cast<int*>(permuted_row_to_unpermuted_row.data_ptr()),
        reinterpret_cast<int*>(unpermuted_row_to_permuted_row.data_ptr()),
        reinterpret_cast<int64_t*>(expert_first_token_offset.data_ptr()),
        num_tokens,
        static_cast<int>(num_experts),
        top_k,
        stream.stream()
    );

    return {permuted_row_to_unpermuted_row, unpermuted_row_to_permuted_row, expert_first_token_offset};
}

th::Tensor kolibri_activation_expansion(
    th::Tensor unpermuted_input,             // [num_tokens, hidden_size] (FP16 or BF16)
    th::Tensor permuted_row_to_unpermuted_row // [num_tokens * top_k] (int32)
) {
    TORCH_CHECK(unpermuted_input.is_cuda(), "unpermuted_input must be a CUDA tensor");
    TORCH_CHECK(permuted_row_to_unpermuted_row.is_cuda(), "permuted_row_to_unpermuted_row must be a CUDA tensor");
    TORCH_CHECK(unpermuted_input.is_contiguous(), "unpermuted_input must be contiguous");
    TORCH_CHECK(permuted_row_to_unpermuted_row.is_contiguous(), "permuted_row_to_unpermuted_row must be contiguous");

    int const num_tokens = unpermuted_input.size(0);
    int const hidden_size = unpermuted_input.size(1);
    int const num_expanded_rows = permuted_row_to_unpermuted_row.size(0);

    auto permuted_output = th::empty({num_expanded_rows, hidden_size}, unpermuted_input.options());
    auto stream = at::cuda::getCurrentCUDAStream(unpermuted_input.device().index());

    if (unpermuted_input.scalar_type() == th::kHalf) {
        tensorrt_llm::kernels::invokeKolibriActivationExpansionFromMap<half>(
            reinterpret_cast<half const*>(unpermuted_input.data_ptr()),
            reinterpret_cast<half*>(permuted_output.data_ptr()),
            reinterpret_cast<int const*>(permuted_row_to_unpermuted_row.data_ptr()),
            num_tokens,
            num_expanded_rows,
            hidden_size,
            stream.stream()
        );
    } else if (unpermuted_input.scalar_type() == th::kBFloat16) {
#ifdef ENABLE_BF16
        tensorrt_llm::kernels::invokeKolibriActivationExpansionFromMap<__nv_bfloat16>(
            reinterpret_cast<__nv_bfloat16 const*>(unpermuted_input.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(permuted_output.data_ptr()),
            reinterpret_cast<int const*>(permuted_row_to_unpermuted_row.data_ptr()),
            num_tokens,
            num_expanded_rows,
            hidden_size,
            stream.stream()
        );
#else
        TORCH_CHECK(false, "BF16 support not enabled");
#endif
    } else if (unpermuted_input.scalar_type() == at::kFloat8_e4m3fn) {
#ifdef ENABLE_FP8
        tensorrt_llm::kernels::invokeKolibriActivationExpansionFromMap<__nv_fp8_e4m3>(
            reinterpret_cast<__nv_fp8_e4m3 const*>(unpermuted_input.data_ptr()),
            reinterpret_cast<__nv_fp8_e4m3*>(permuted_output.data_ptr()),
            reinterpret_cast<int const*>(permuted_row_to_unpermuted_row.data_ptr()),
            num_tokens,
            num_expanded_rows,
            hidden_size,
            stream.stream()
        );
#else
        TORCH_CHECK(false, "FP8 support not enabled");
#endif
    } else if (unpermuted_input.scalar_type() == at::kFloat8_e5m2) {
#ifdef ENABLE_FP8
        tensorrt_llm::kernels::invokeKolibriActivationExpansionFromMap<__nv_fp8_e5m2>(
            reinterpret_cast<__nv_fp8_e5m2 const*>(unpermuted_input.data_ptr()),
            reinterpret_cast<__nv_fp8_e5m2*>(permuted_output.data_ptr()),
            reinterpret_cast<int const*>(permuted_row_to_unpermuted_row.data_ptr()),
            num_tokens,
            num_expanded_rows,
            hidden_size,
            stream.stream()
        );
#else
        TORCH_CHECK(false, "FP8 support not enabled");
#endif
    } else {
        TORCH_CHECK(false, "Unsupported dtype: only float16, bfloat16, float8_e4m3fn, and float8_e5m2 are supported");
    }

    return permuted_output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kolibri_build_expert_maps", &kolibri_build_expert_maps, "Custom Kolibri Build Expert Maps (256 threads, 3 KB SRAM)");
    m.def("trtllm_build_expert_maps_b32", &trtllm_build_expert_maps_b32, "TRT-LLM Baseline Build Expert Maps (CUB B32)");
    m.def("kolibri_activation_expansion", &kolibri_activation_expansion, "Custom Kolibri Activation Expansion (Multi-Block uint4)");
}
