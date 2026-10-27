#include "trtllm_post_moe_norm_baseline.cuh"

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

at::Tensor trtllm_post_moe_norm_baseline(
    at::Tensor const& routed_out,
    at::Tensor const& shared_out,
    at::Tensor const& gamma,
    double const eps = 1e-5)
{
    TORCH_CHECK(routed_out.is_cuda(), "routed_out must be a CUDA tensor");
    TORCH_CHECK(shared_out.is_cuda(), "shared_out must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be a CUDA tensor");

    TORCH_CHECK(routed_out.is_contiguous(), "routed_out must be contiguous");
    TORCH_CHECK(shared_out.is_contiguous(), "shared_out must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");

    TORCH_CHECK(routed_out.sizes() == shared_out.sizes(), "routed_out and shared_out must have identical shapes");
    TORCH_CHECK(routed_out.dtype() == shared_out.dtype(), "routed_out and shared_out must have identical dtypes");
    TORCH_CHECK(routed_out.dtype() == gamma.dtype(), "routed_out and gamma must have identical dtypes");

    int const hidden_dim = routed_out.size(-1);
    TORCH_CHECK(gamma.numel() == hidden_dim, "gamma must have size equal to hidden_dim");

    int const tokens = routed_out.numel() / hidden_dim;

    // =========================================================================================
    // Step 1: TRT-LLM Model Vector Addition (Launch 1)
    // Matches exact forward pass in TRT-LLM (modeling_kolibri.py:303: out = routed_out + shared_out)
    // Executes: at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add>
    // =========================================================================================
    auto combined = at::add(routed_out, shared_out);

    // =========================================================================================
    // Step 2: Native TRT-LLM RMSNorm (Launch 2)
    // Directly calls tensorrt_llm::kernels::invokeGeneralRmsNorm from TRT-LLM rmsnormKernels.cu
    // Executes: tensorrt_llm::_v1::kernels::generalRmsNorm
    // =========================================================================================
    auto moe_out = torch::empty_like(routed_out);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    auto const scalar_type = routed_out.scalar_type();
    if (scalar_type == at::ScalarType::BFloat16)
    {
        kolibri::baseline::invokeTrtllmRmsNorm<__nv_bfloat16>(
            reinterpret_cast<__nv_bfloat16*>(moe_out.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(combined.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(gamma.data_ptr()),
            tokens,
            hidden_dim,
            static_cast<float>(eps),
            stream
        );
    }
    else if (scalar_type == at::ScalarType::Half)
    {
        kolibri::baseline::invokeTrtllmRmsNorm<half>(
            reinterpret_cast<half*>(moe_out.data_ptr()),
            reinterpret_cast<half const*>(combined.data_ptr()),
            reinterpret_cast<half const*>(gamma.data_ptr()),
            tokens,
            hidden_dim,
            static_cast<float>(eps),
            stream
        );
    }
    else
    {
        TORCH_CHECK(false, "Unsupported scalar type for trtllm_post_moe_norm_baseline. Supported: BF16, FP16.");
    }

    return moe_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "Native TensorRT-LLM Post-MoE Norm Baseline PyTorch Extension";
    m.def(
        "trtllm_post_moe_norm_baseline",
        &trtllm_post_moe_norm_baseline,
        "Native TRT-LLM Post-MoE Norm Baseline (routed_out, shared_out, gamma, eps=1e-5)"
    );
}
