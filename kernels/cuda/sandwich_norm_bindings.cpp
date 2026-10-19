#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/stringUtils.h"
#include "tensorrt_llm/common/tllmException.h"
#include "tensorrt_llm/common/logger.h"
#include "customKolibriSandwichNormKernels.h"
#include <cstdarg>
#include <cstdio>
#include <cstring>

TRTLLM_NAMESPACE_BEGIN
namespace common {
    void fmtstr_(char const* format, fmtstr_allocator alloc, void* target, va_list args) {
        va_list args0;
        va_copy(args0, args);
        size_t constexpr init_size = 2048;
        char fixed_buffer[init_size];
        auto const size = std::vsnprintf(fixed_buffer, init_size, format, args0);
        va_end(args0);
        if (size <= 0) return;
        auto* memory = alloc(target, size);
        if (static_cast<size_t>(size) < init_size) {
            std::memcpy(memory, fixed_buffer, size + 1);
        } else {
            std::vsnprintf(memory, size + 1, format, args);
        }
    }

    TllmException::TllmException(char const* file, std::size_t line, char const* msg)
        : std::runtime_error(std::string(msg ? msg : "") + " (" + file + ":" + std::to_string(line) + ")"), mNbFrames(0) {}
    TllmException::~TllmException() noexcept = default;
    std::string TllmException::getTrace() const { return ""; }
    std::string TllmException::demangle(char const* name) { return name ? std::string(name) : ""; }
    RequestSpecificException::RequestSpecificException(
        std::string const& file, std::size_t line, char const* msg, uint64_t requestID, RequestErrorCode errorCode)
        : std::runtime_error(msg ? msg : ""), mRequestID(requestID), mErrorCode(errorCode) {}
    RequestSpecificException::~RequestSpecificException() noexcept = default;
    uint64_t RequestSpecificException::getRequestId() const noexcept { return mRequestID; }
    RequestErrorCode RequestSpecificException::getErrorCode() const noexcept { return mErrorCode; }

    Logger::Logger() : level_(INFO), rank_(0) {}
    Logger* Logger::getLogger() {
        static Logger instance;
        return &instance;
    }
    bool getEnvEnablePDL() { return false; }
}
TRTLLM_NAMESPACE_END

namespace th = torch;

std::tuple<th::Tensor, th::Tensor> fused_sandwich_norm(
    th::Tensor attn_out,
    th::Tensor residual_in,
    th::Tensor gamma1,
    th::Tensor gamma2,
    double eps
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

    if (attn_out.scalar_type() == th::kHalf) {
        tensorrt_llm::kernels::invokeFusedSandwichResidualNorm<half>(
            reinterpret_cast<half const*>(attn_out.data_ptr()),
            reinterpret_cast<half const*>(residual_in.data_ptr()),
            reinterpret_cast<half const*>(gamma1.data_ptr()),
            reinterpret_cast<half const*>(gamma2.data_ptr()),
            reinterpret_cast<half*>(updated_residual_out.data_ptr()),
            reinterpret_cast<half*>(moe_in_out.data_ptr()),
            static_cast<float>(eps),
            tokens,
            hidden_dim,
            stream.stream()
        );
    } else if (attn_out.scalar_type() == th::kBFloat16) {
#ifdef ENABLE_BF16
        tensorrt_llm::kernels::invokeFusedSandwichResidualNorm<__nv_bfloat16>(
            reinterpret_cast<__nv_bfloat16 const*>(attn_out.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(residual_in.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(gamma1.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(gamma2.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(updated_residual_out.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(moe_in_out.data_ptr()),
            static_cast<float>(eps),
            tokens,
            hidden_dim,
            stream.stream()
        );
#else
        TORCH_CHECK(false, "BF16 support not enabled during compilation");
#endif
    } else if (attn_out.scalar_type() == th::kFloat32) {
        tensorrt_llm::kernels::invokeFusedSandwichResidualNorm<float>(
            reinterpret_cast<float const*>(attn_out.data_ptr()),
            reinterpret_cast<float const*>(residual_in.data_ptr()),
            reinterpret_cast<float const*>(gamma1.data_ptr()),
            reinterpret_cast<float const*>(gamma2.data_ptr()),
            reinterpret_cast<float*>(updated_residual_out.data_ptr()),
            reinterpret_cast<float*>(moe_in_out.data_ptr()),
            static_cast<float>(eps),
            tokens,
            hidden_dim,
            stream.stream()
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype: only float16, bfloat16, and float32 are supported");
    }

    return std::make_tuple(updated_residual_out, moe_in_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_sandwich_norm", &fused_sandwich_norm, "Custom Kolibri Fused Sandwich Norm (CUDA)");
}
