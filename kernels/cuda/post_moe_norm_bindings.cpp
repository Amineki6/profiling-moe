#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/stringUtils.h"
#include "tensorrt_llm/common/tllmException.h"
#include "tensorrt_llm/common/logger.h"
#include "customKolibriPostMoeNormKernels.h"
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

th::Tensor fused_add_post_moe_norm(
    th::Tensor routed_out,
    th::Tensor shared_out,
    th::Tensor gamma,
    double eps
) {
    TORCH_CHECK(routed_out.is_cuda(), "routed_out must be a CUDA tensor");
    TORCH_CHECK(shared_out.is_cuda(), "shared_out must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be a CUDA tensor");

    TORCH_CHECK(routed_out.is_contiguous(), "routed_out must be contiguous");
    TORCH_CHECK(shared_out.is_contiguous(), "shared_out must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");

    int const hidden_dim = routed_out.size(-1);
    int const tokens = routed_out.numel() / hidden_dim;

    auto moe_out = th::empty_like(routed_out);
    auto stream = at::cuda::getCurrentCUDAStream(routed_out.device().index());

    if (routed_out.scalar_type() == th::kHalf) {
        tensorrt_llm::kernels::invokeFusedAddPostMoeNorm<half>(
            reinterpret_cast<half const*>(routed_out.data_ptr()),
            reinterpret_cast<half const*>(shared_out.data_ptr()),
            reinterpret_cast<half const*>(gamma.data_ptr()),
            reinterpret_cast<half*>(moe_out.data_ptr()),
            static_cast<float>(eps),
            tokens,
            hidden_dim,
            stream.stream()
        );
    } else if (routed_out.scalar_type() == th::kBFloat16) {
#ifdef ENABLE_BF16
        tensorrt_llm::kernels::invokeFusedAddPostMoeNorm<__nv_bfloat16>(
            reinterpret_cast<__nv_bfloat16 const*>(routed_out.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(shared_out.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(gamma.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(moe_out.data_ptr()),
            static_cast<float>(eps),
            tokens,
            hidden_dim,
            stream.stream()
        );
#else
        TORCH_CHECK(false, "BF16 support not enabled during compilation");
#endif
    } else if (routed_out.scalar_type() == th::kFloat32) {
        tensorrt_llm::kernels::invokeFusedAddPostMoeNorm<float>(
            reinterpret_cast<float const*>(routed_out.data_ptr()),
            reinterpret_cast<float const*>(shared_out.data_ptr()),
            reinterpret_cast<float const*>(gamma.data_ptr()),
            reinterpret_cast<float*>(moe_out.data_ptr()),
            static_cast<float>(eps),
            tokens,
            hidden_dim,
            stream.stream()
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype: only float16, bfloat16, and float32 are supported");
    }

    return moe_out;
}

th::Tensor copy_baseline(
    th::Tensor in1,
    th::Tensor in2
) {
    TORCH_CHECK(in1.is_cuda() && in2.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(in1.is_contiguous() && in2.is_contiguous(), "Inputs must be contiguous");
    TORCH_CHECK(in1.sizes() == in2.sizes(), "Input sizes must match");

    int const hidden_dim = in1.size(-1);
    int const tokens = in1.numel() / hidden_dim;

    auto out = th::empty_like(in1);
    auto stream = at::cuda::getCurrentCUDAStream(in1.device().index());

    if (in1.scalar_type() == th::kBFloat16) {
#ifdef ENABLE_BF16
        tensorrt_llm::kernels::invokeCopyBaselinePostMoe<__nv_bfloat16>(
            reinterpret_cast<__nv_bfloat16 const*>(in1.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(in2.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            tokens,
            hidden_dim,
            stream.stream()
        );
#else
        TORCH_CHECK(false, "BF16 support not enabled during compilation");
#endif
    } else if (in1.scalar_type() == th::kHalf) {
        tensorrt_llm::kernels::invokeCopyBaselinePostMoe<half>(
            reinterpret_cast<half const*>(in1.data_ptr()),
            reinterpret_cast<half const*>(in2.data_ptr()),
            reinterpret_cast<half*>(out.data_ptr()),
            tokens,
            hidden_dim,
            stream.stream()
        );
    } else if (in1.scalar_type() == th::kFloat32) {
        tensorrt_llm::kernels::invokeCopyBaselinePostMoe<float>(
            reinterpret_cast<float const*>(in1.data_ptr()),
            reinterpret_cast<float const*>(in2.data_ptr()),
            reinterpret_cast<float*>(out.data_ptr()),
            tokens,
            hidden_dim,
            stream.stream()
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype");
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_add_post_moe_norm", &fused_add_post_moe_norm, "Custom Kolibri Fused Add + Post-MoE Norm (CUDA)");
    m.def("copy_baseline", &copy_baseline, "Pure 2-in 1-out 128-bit Copy Baseline (CUDA)");
}
