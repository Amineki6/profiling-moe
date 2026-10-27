#include "trtllm_moe_prologue_baseline.cuh"

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>

std::vector<at::Tensor> trtllm_router(
    at::Tensor const& scores_in,
    c10::optional<at::Tensor> const& bias_opt)
{
    TORCH_CHECK(scores_in.is_cuda(), "scores_in must be a CUDA tensor");
    TORCH_CHECK(scores_in.is_contiguous(), "scores_in must be contiguous");
    TORCH_CHECK(scores_in.scalar_type() == at::ScalarType::Float, "scores_in must be float32");

    int const num_tokens = scores_in.size(0);
    int const num_experts = scores_in.size(1);
    TORCH_CHECK(num_experts == 384, "num_experts must be 384 for Kolibri");

    float const* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt->defined())
    {
        TORCH_CHECK(bias_opt->is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias_opt->is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias_opt->scalar_type() == at::ScalarType::Float, "bias must be float32");
        bias_ptr = bias_opt->data_ptr<float>();
    }

    auto out_weights = torch::empty({num_tokens, 6}, scores_in.options());
    auto out_indices = torch::empty({num_tokens, 6}, scores_in.options().dtype(torch::kInt32));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    kolibri::baseline::invokeTrtllmRouting(
        scores_in.data_ptr<float>(),
        bias_ptr,
        out_weights.data_ptr<float>(),
        out_indices.data_ptr<int>(),
        num_tokens,
        stream
    );

    return {out_weights, out_indices};
}

std::vector<at::Tensor> trtllm_build_expert_maps(
    at::Tensor const& token_selected_experts,
    int64_t num_experts = 384)
{
    TORCH_CHECK(token_selected_experts.is_cuda(), "token_selected_experts must be a CUDA tensor");
    TORCH_CHECK(token_selected_experts.is_contiguous(), "token_selected_experts must be contiguous");
    TORCH_CHECK(token_selected_experts.scalar_type() == at::ScalarType::Int, "token_selected_experts must be int32");

    int const num_tokens = token_selected_experts.size(0);
    int const top_k = token_selected_experts.size(1);
    TORCH_CHECK(top_k == 6, "top_k must be 6 for Kolibri");

    int const total_selections = num_tokens * top_k;
    auto permuted_row_to_unpermuted = torch::empty({total_selections}, token_selected_experts.options());
    auto unpermuted_to_permuted = torch::empty({total_selections}, token_selected_experts.options());
    auto expert_offsets = torch::empty({num_experts + 1}, token_selected_experts.options().dtype(torch::kInt64));

    // Allocate scratch for 3-kernel pipeline: blocked counts (2x) + blocked row mappings
    int64_t const num_blocks_per_seq = (num_tokens + 31) / 32;
    int64_t const scratch_ints = 2 * num_experts * num_blocks_per_seq + num_experts * num_tokens;
    auto scratch = torch::empty({scratch_ints}, token_selected_experts.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    kolibri::baseline::invokeTrtllmBuildExpertMaps(
        token_selected_experts.data_ptr<int>(),
        permuted_row_to_unpermuted.data_ptr<int>(),
        unpermuted_to_permuted.data_ptr<int>(),
        expert_offsets.data_ptr<int64_t>(),
        scratch.data_ptr(),
        num_tokens,
        stream
    );

    return {permuted_row_to_unpermuted, unpermuted_to_permuted, expert_offsets};
}

at::Tensor trtllm_activation_expansion(
    at::Tensor const& unpermuted_input,
    at::Tensor const& permuted_row_to_unpermuted)
{
    TORCH_CHECK(unpermuted_input.is_cuda(), "unpermuted_input must be a CUDA tensor");
    TORCH_CHECK(permuted_row_to_unpermuted.is_cuda(), "permuted_row_to_unpermuted must be a CUDA tensor");
    TORCH_CHECK(unpermuted_input.is_contiguous(), "unpermuted_input must be contiguous");
    TORCH_CHECK(permuted_row_to_unpermuted.is_contiguous(), "permuted_row_to_unpermuted must be contiguous");

    int const num_expanded_rows = permuted_row_to_unpermuted.size(0);
    int const hidden_size = unpermuted_input.size(-1);

    auto permuted_output = torch::empty({num_expanded_rows, hidden_size}, unpermuted_input.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    auto const scalar_type = unpermuted_input.scalar_type();
    if (scalar_type == at::ScalarType::BFloat16)
    {
        kolibri::baseline::invokeTrtllmActivationExpansion<__nv_bfloat16>(
            reinterpret_cast<__nv_bfloat16 const*>(unpermuted_input.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(permuted_output.data_ptr()),
            permuted_row_to_unpermuted.data_ptr<int>(),
            num_expanded_rows,
            hidden_size,
            stream
        );
    }
    else if (scalar_type == at::ScalarType::Half)
    {
        kolibri::baseline::invokeTrtllmActivationExpansion<half>(
            reinterpret_cast<half const*>(unpermuted_input.data_ptr()),
            reinterpret_cast<half*>(permuted_output.data_ptr()),
            permuted_row_to_unpermuted.data_ptr<int>(),
            num_expanded_rows,
            hidden_size,
            stream
        );
    }
    else
    {
        TORCH_CHECK(false, "Unsupported dtype for activation expansion. Supported: BF16, FP16.");
    }

    return permuted_output;
}

std::vector<at::Tensor> trtllm_moe_prologue(
    at::Tensor const& unpermuted_input,
    at::Tensor const& router_logits,
    c10::optional<at::Tensor> const& router_bias_opt)
{
    auto router_res = trtllm_router(router_logits, router_bias_opt);
    auto routing_weights = router_res[0];
    auto selected_experts = router_res[1];

    auto map_res = trtllm_build_expert_maps(selected_experts, 384);
    auto permuted_row_to_unpermuted = map_res[0];
    auto unpermuted_to_permuted = map_res[1];
    auto expert_offsets = map_res[2];

    auto permuted_output = trtllm_activation_expansion(unpermuted_input, permuted_row_to_unpermuted);

    return {permuted_output, routing_weights, selected_experts, expert_offsets, permuted_row_to_unpermuted, unpermuted_to_permuted};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "Native TensorRT-LLM MoE Prologue Baseline";
    m.def("trtllm_router", &trtllm_router, "TRT-LLM Router (scores, bias=None)");
    m.def("trtllm_build_expert_maps", &trtllm_build_expert_maps, "TRT-LLM CUB Build Expert Maps (token_selected_experts, num_experts=384)");
    m.def("trtllm_activation_expansion", &trtllm_activation_expansion, "TRT-LLM Activation Expansion (unpermuted_input, permuted_row_to_unpermuted)");
    m.def("trtllm_moe_prologue", &trtllm_moe_prologue, "End-to-End TRT-LLM Prologue (unpermuted_input, router_logits, router_bias=None)");
}
