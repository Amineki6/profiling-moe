#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>

#include "trtllm_sandwich_norm_baseline.cuh"

namespace th = torch;

std::vector<th::Tensor> trtllm_sandwich_norm_baseline(
    th::Tensor attn_out,     // [tokens, hidden_dim]
    th::Tensor residual_in, // [tokens, hidden_dim]
    th::Tensor gamma1,      // [hidden_dim]
    th::Tensor gamma2,      // [hidden_dim]
    double eps = 1e-5
) {
    TORCH_CHECK(attn_out.is_cuda(), "attn_out must be a CUDA tensor");
    TORCH_CHECK(residual_in.is_cuda(), "residual_in must be a CUDA tensor");
    TORCH_CHECK(gamma1.is_cuda(), "gamma1 must be a CUDA tensor");
    TORCH_CHECK(gamma2.is_cuda(), "gamma2 must be a CUDA tensor");

    TORCH_CHECK(attn_out.is_contiguous(), "attn_out must be contiguous");
    TORCH_CHECK(residual_in.is_contiguous(), "residual_in must be contiguous");
    TORCH_CHECK(gamma1.is_contiguous(), "gamma1 must be contiguous");
    TORCH_CHECK(gamma2.is_contiguous(), "gamma2 must be contiguous");

    int const hidden_dim = attn_out.size(-1);
    int const tokens = attn_out.numel() / hidden_dim;

    auto temp_post_attn = th::empty_like(attn_out);
    auto moe_in_out = th::empty_like(attn_out);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(attn_out.device().index()).stream();

    // =========================================================================================
    // Launch 1: Native TRT-LLM RMSNorm (Post-Attention Sandwich Norm)
    // Directly calls tensorrt_llm::kernels::invokeGeneralRmsNorm from TRT-LLM rmsnormKernels.cu
    // =========================================================================================
    if (attn_out.scalar_type() == th::kBFloat16) {
        kolibri::baseline::invokeTrtllmRmsNorm<__nv_bfloat16>(
            reinterpret_cast<__nv_bfloat16*>(temp_post_attn.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(attn_out.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(gamma1.data_ptr()),
            tokens, hidden_dim, static_cast<float>(eps), stream
        );
    } else if (attn_out.scalar_type() == th::kHalf) {
        kolibri::baseline::invokeTrtllmRmsNorm<half>(
            reinterpret_cast<half*>(temp_post_attn.data_ptr()),
            reinterpret_cast<half const*>(attn_out.data_ptr()),
            reinterpret_cast<half const*>(gamma1.data_ptr()),
            tokens, hidden_dim, static_cast<float>(eps), stream
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype: only bfloat16 and float16 are supported");
    }

    // =========================================================================================
    // Launch 2: TRT-LLM Model Vector Addition (residual_in + temp_post_attn)
    // Matches exact forward pass in TRT-LLM (modeling_kolibri.py:378)
    // Executes: at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add>
    // =========================================================================================
    auto updated_residual_out = at::add(residual_in, temp_post_attn);

    // =========================================================================================
    // Launch 3: Native TRT-LLM RMSNorm (Pre-MoE Norm)
    // Directly calls tensorrt_llm::kernels::invokeGeneralRmsNorm from TRT-LLM rmsnormKernels.cu
    // =========================================================================================
    if (attn_out.scalar_type() == th::kBFloat16) {
        kolibri::baseline::invokeTrtllmRmsNorm<__nv_bfloat16>(
            reinterpret_cast<__nv_bfloat16*>(moe_in_out.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(updated_residual_out.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(gamma2.data_ptr()),
            tokens, hidden_dim, static_cast<float>(eps), stream
        );
    } else if (attn_out.scalar_type() == th::kHalf) {
        kolibri::baseline::invokeTrtllmRmsNorm<half>(
            reinterpret_cast<half*>(moe_in_out.data_ptr()),
            reinterpret_cast<half const*>(updated_residual_out.data_ptr()),
            reinterpret_cast<half const*>(gamma2.data_ptr()),
            tokens, hidden_dim, static_cast<float>(eps), stream
        );
    }

    return {updated_residual_out, moe_in_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Native TRT-LLM Sandwich Norm Baseline Ops";
    m.def(
        "trtllm_sandwich_norm_baseline",
        &trtllm_sandwich_norm_baseline,
        "Native TRT-LLM Sandwich Norm Baseline"
    );
}
