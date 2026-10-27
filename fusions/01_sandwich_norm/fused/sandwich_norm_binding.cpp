#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>

#include "custom_sandwich_norm.cuh"

namespace th = torch;

std::vector<th::Tensor> fused_sandwich_norm(
    th::Tensor attn_out,     // [tokens, hidden_dim]
    th::Tensor residual_in, // [tokens, hidden_dim]
    th::Tensor gamma1,      // [hidden_dim] (Post-Attention Sandwich Norm weight)
    th::Tensor gamma2,      // [hidden_dim] (Pre-MoE Norm weight)
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

    auto updated_residual_out = th::empty_like(residual_in);
    auto moe_in_out = th::empty_like(attn_out);

    auto stream = at::cuda::getCurrentCUDAStream(attn_out.device().index());

    if (attn_out.scalar_type() == th::kBFloat16) {
        kolibri::fusions::invokeFusedSandwichNormReg<__nv_bfloat16>(
            reinterpret_cast<__nv_bfloat16 const*>(attn_out.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(residual_in.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(gamma1.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(gamma2.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(updated_residual_out.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(moe_in_out.data_ptr()),
            tokens, hidden_dim, static_cast<float>(eps), stream.stream()
        );
    } else if (attn_out.scalar_type() == th::kHalf) {
        kolibri::fusions::invokeFusedSandwichNormReg<half>(
            reinterpret_cast<half const*>(attn_out.data_ptr()),
            reinterpret_cast<half const*>(residual_in.data_ptr()),
            reinterpret_cast<half const*>(gamma1.data_ptr()),
            reinterpret_cast<half const*>(gamma2.data_ptr()),
            reinterpret_cast<half*>(updated_residual_out.data_ptr()),
            reinterpret_cast<half*>(moe_in_out.data_ptr()),
            tokens, hidden_dim, static_cast<float>(eps), stream.stream()
        );
    } else if (attn_out.scalar_type() == th::kFloat) {
        kolibri::fusions::invokeFusedSandwichNormReg<float>(
            reinterpret_cast<float const*>(attn_out.data_ptr()),
            reinterpret_cast<float const*>(residual_in.data_ptr()),
            reinterpret_cast<float const*>(gamma1.data_ptr()),
            reinterpret_cast<float const*>(gamma2.data_ptr()),
            reinterpret_cast<float*>(updated_residual_out.data_ptr()),
            reinterpret_cast<float*>(moe_in_out.data_ptr()),
            tokens, hidden_dim, static_cast<float>(eps), stream.stream()
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype: only bfloat16, float16, and float32 are supported");
    }

    return {updated_residual_out, moe_in_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_sandwich_norm", &fused_sandwich_norm, "Fused Post-Attention Sandwich Norm + Residual Add + Pre-MoE Norm");
}
