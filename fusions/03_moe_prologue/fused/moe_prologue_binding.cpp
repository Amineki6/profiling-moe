#include "custom_moe_prologue.cuh"

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>

std::vector<at::Tensor> kolibri_router(
    at::Tensor const& scores_in,
    c10::optional<at::Tensor> const& bias_opt)
{
    TORCH_CHECK(scores_in.is_cuda(), "scores_in must be a CUDA tensor");
    TORCH_CHECK(scores_in.is_contiguous(), "scores_in must be contiguous");
    TORCH_CHECK(scores_in.scalar_type() == at::ScalarType::Float, "scores_in must be float32");

    int const num_tokens = scores_in.size(0);
    int const num_experts = scores_in.size(1);
    TORCH_CHECK(num_experts == kolibri::fusions::NUM_EXPERTS, "num_experts must be 384 for Kolibri");

    float const* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt->defined())
    {
        TORCH_CHECK(bias_opt->is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias_opt->is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias_opt->scalar_type() == at::ScalarType::Float, "bias must be float32");
        bias_ptr = bias_opt->data_ptr<float>();
    }

    auto out_weights = torch::empty({num_tokens, kolibri::fusions::TOP_K}, scores_in.options());
    auto out_indices = torch::empty({num_tokens, kolibri::fusions::TOP_K}, scores_in.options().dtype(torch::kInt32));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    kolibri::fusions::invokeKolibriRouter(
        scores_in.data_ptr<float>(),
        bias_ptr,
        out_weights.data_ptr<float>(),
        out_indices.data_ptr<int>(),
        num_tokens,
        stream
    );

    return {out_weights, out_indices};
}

std::vector<at::Tensor> kolibri_dispatch_map(
    at::Tensor const& token_selected_experts,
    int64_t num_experts = 384,
    bool force_single_block = false,
    bool force_multi_block = false)
{
    TORCH_CHECK(token_selected_experts.is_cuda(), "token_selected_experts must be a CUDA tensor");
    TORCH_CHECK(token_selected_experts.is_contiguous(), "token_selected_experts must be contiguous");
    TORCH_CHECK(token_selected_experts.scalar_type() == at::ScalarType::Int, "token_selected_experts must be int32");

    int const num_tokens = token_selected_experts.size(0);
    int const top_k = token_selected_experts.size(1);
    TORCH_CHECK(top_k == kolibri::fusions::TOP_K, "top_k must be 6 for Kolibri");
    TORCH_CHECK(num_experts == kolibri::fusions::NUM_EXPERTS, "num_experts must be 384 for Kolibri");

    int const total_selections = num_tokens * top_k;
    auto permuted_row_to_unpermuted = torch::empty({total_selections}, token_selected_experts.options());
    auto unpermuted_to_permuted = torch::empty({total_selections}, token_selected_experts.options());
    auto expert_offsets = torch::empty({num_experts + 1}, token_selected_experts.options().dtype(torch::kInt64));

    int const scratch_elems = std::max(num_tokens * 12, 2 * 128 * kolibri::fusions::NUM_EXPERTS);
    auto scratch = torch::empty({scratch_elems}, token_selected_experts.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    kolibri::fusions::invokeDynamicBuildExpertMaps(
        token_selected_experts.data_ptr<int>(),
        permuted_row_to_unpermuted.data_ptr<int>(),
        unpermuted_to_permuted.data_ptr<int>(),
        expert_offsets.data_ptr<int64_t>(),
        scratch.data_ptr(),
        num_tokens,
        stream,
        force_single_block,
        force_multi_block
    );

    return {permuted_row_to_unpermuted, unpermuted_to_permuted, expert_offsets};
}

at::Tensor kolibri_activation_expansion(
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
        kolibri::fusions::invokeKolibriActivationExpansion<__nv_bfloat16>(
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
        kolibri::fusions::invokeKolibriActivationExpansion<half>(
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

std::vector<at::Tensor> fused_moe_prologue(
    at::Tensor const& unpermuted_input,
    at::Tensor const& router_logits,
    c10::optional<at::Tensor> const& router_bias_opt)
{
    TORCH_CHECK(unpermuted_input.is_cuda(), "unpermuted_input must be a CUDA tensor");
    TORCH_CHECK(router_logits.is_cuda(), "router_logits must be a CUDA tensor");
    TORCH_CHECK(unpermuted_input.is_contiguous(), "unpermuted_input must be contiguous");
    TORCH_CHECK(router_logits.is_contiguous(), "router_logits must be contiguous");

    int const num_tokens = unpermuted_input.size(0);
    int const hidden_size = unpermuted_input.size(-1);
    int const total_selections = num_tokens * kolibri::fusions::TOP_K;

    float const* bias_ptr = nullptr;
    if (router_bias_opt.has_value() && router_bias_opt->defined())
    {
        bias_ptr = router_bias_opt->data_ptr<float>();
    }

    auto permuted_output = torch::empty({total_selections, hidden_size}, unpermuted_input.options());
    auto routing_weights = torch::empty({num_tokens, kolibri::fusions::TOP_K}, router_logits.options());
    auto selected_experts = torch::empty({num_tokens, kolibri::fusions::TOP_K}, router_logits.options().dtype(torch::kInt32));
    auto permuted_row_to_unpermuted = torch::empty({total_selections}, router_logits.options().dtype(torch::kInt32));
    auto unpermuted_to_permuted = torch::empty({total_selections}, router_logits.options().dtype(torch::kInt32));
    auto expert_offsets = torch::empty({kolibri::fusions::NUM_EXPERTS + 1}, router_logits.options().dtype(torch::kInt64));

    int const scratch_elems = std::max(num_tokens * 12, 2 * 128 * kolibri::fusions::NUM_EXPERTS);
    auto scratch = torch::empty({scratch_elems}, router_logits.options().dtype(torch::kInt32));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    auto const scalar_type = unpermuted_input.scalar_type();
    if (scalar_type == at::ScalarType::BFloat16)
    {
        kolibri::fusions::invokeFusedMoePrologue<__nv_bfloat16>(
            reinterpret_cast<__nv_bfloat16 const*>(unpermuted_input.data_ptr()),
            router_logits.data_ptr<float>(),
            bias_ptr,
            reinterpret_cast<__nv_bfloat16*>(permuted_output.data_ptr()),
            routing_weights.data_ptr<float>(),
            selected_experts.data_ptr<int>(),
            permuted_row_to_unpermuted.data_ptr<int>(),
            unpermuted_to_permuted.data_ptr<int>(),
            expert_offsets.data_ptr<int64_t>(),
            scratch.data_ptr(),
            num_tokens,
            hidden_size,
            stream
        );
    }
    else if (scalar_type == at::ScalarType::Half)
    {
        kolibri::fusions::invokeFusedMoePrologue<half>(
            reinterpret_cast<half const*>(unpermuted_input.data_ptr()),
            router_logits.data_ptr<float>(),
            bias_ptr,
            reinterpret_cast<half*>(permuted_output.data_ptr()),
            routing_weights.data_ptr<float>(),
            selected_experts.data_ptr<int>(),
            permuted_row_to_unpermuted.data_ptr<int>(),
            unpermuted_to_permuted.data_ptr<int>(),
            expert_offsets.data_ptr<int64_t>(),
            scratch.data_ptr(),
            num_tokens,
            hidden_size,
            stream
        );
    }
    else
    {
        TORCH_CHECK(false, "Unsupported dtype for fused_moe_prologue. Supported: BF16, FP16.");
    }

    return {permuted_output, routing_weights, selected_experts, expert_offsets, permuted_row_to_unpermuted, unpermuted_to_permuted};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "Kolibri Fused MoE Prologue (Router + Dynamic Hybrid Dispatch Map + Expansion)";
    m.def("kolibri_router", &kolibri_router, "Top-6 Router (scores, bias=None)");
    m.def("kolibri_dispatch_map", &kolibri_dispatch_map, "Dynamic Hybrid Dispatch Map (selected_experts, num_experts=384, force_single_block=False, force_multi_block=False)");
    m.def("kolibri_activation_expansion", &kolibri_activation_expansion, "Vectorized 128-bit Activation Expansion (unpermuted_input, permuted_row_to_unpermuted)");
    m.def("fused_moe_prologue", &fused_moe_prologue, "End-to-End Prologue (unpermuted_input, router_logits, router_bias=None)");
}
