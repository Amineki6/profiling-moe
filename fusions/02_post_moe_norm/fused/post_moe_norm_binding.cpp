#include "custom_post_moe_norm.cuh"

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

at::Tensor fused_add_post_moe_norm(
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

    TORCH_CHECK(routed_out.sizes() == shared_out.sizes(), "routed_out and shared_out must have the same shape");
    TORCH_CHECK(routed_out.dtype() == shared_out.dtype(), "routed_out and shared_out must have the same dtype");
    TORCH_CHECK(routed_out.dtype() == gamma.dtype(), "routed_out and gamma must have the same dtype");

    int const hidden_dim = routed_out.size(-1);
    TORCH_CHECK(gamma.numel() == hidden_dim, "gamma must have size equal to hidden_dim");

    int const tokens = routed_out.numel() / hidden_dim;

    auto moe_out = torch::empty_like(routed_out);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    auto const scalar_type = routed_out.scalar_type();
    if (scalar_type == at::ScalarType::BFloat16)
    {
        kolibri::fusions::invokeFusedAddPostMoeNorm<__nv_bfloat16>(
            reinterpret_cast<__nv_bfloat16 const*>(routed_out.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(shared_out.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(gamma.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(moe_out.data_ptr()),
            tokens,
            hidden_dim,
            static_cast<float>(eps),
            stream
        );
    }
    else if (scalar_type == at::ScalarType::Half)
    {
        kolibri::fusions::invokeFusedAddPostMoeNorm<half>(
            reinterpret_cast<half const*>(routed_out.data_ptr()),
            reinterpret_cast<half const*>(shared_out.data_ptr()),
            reinterpret_cast<half const*>(gamma.data_ptr()),
            reinterpret_cast<half*>(moe_out.data_ptr()),
            tokens,
            hidden_dim,
            static_cast<float>(eps),
            stream
        );
    }
    else if (scalar_type == at::ScalarType::Float)
    {
        kolibri::fusions::invokeFusedAddPostMoeNorm<float>(
            reinterpret_cast<float const*>(routed_out.data_ptr()),
            reinterpret_cast<float const*>(shared_out.data_ptr()),
            reinterpret_cast<float const*>(gamma.data_ptr()),
            reinterpret_cast<float*>(moe_out.data_ptr()),
            tokens,
            hidden_dim,
            static_cast<float>(eps),
            stream
        );
    }
    else
    {
        TORCH_CHECK(false, "Unsupported scalar type for fused_add_post_moe_norm. Supported: BF16, FP16, FP32.");
    }

    return moe_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "Kolibri Fused Add + Post-MoE Norm Production CUDA Kernel";
    m.def(
        "fused_add_post_moe_norm",
        &fused_add_post_moe_norm,
        "Fused Add + Post-MoE Norm (routed_out, shared_out, gamma, eps=1e-5)"
    );
}
