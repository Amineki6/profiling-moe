#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/stringUtils.h"
#include "tensorrt_llm/common/tllmException.h"
#include "tensorrt_llm/common/logger.h"
#include "customKolibriRouterDispatchKernels.h"
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <vector>

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

std::vector<th::Tensor> fused_kolibri_router_dispatch(
    th::Tensor unpermuted_input,  // [num_tokens, hidden_size] (FP16 or BF16)
    th::Tensor scores_in,         // [num_tokens, num_experts] (float32)
    th::Tensor bias               // [num_experts] (float32)
) {
    TORCH_CHECK(unpermuted_input.is_cuda(), "unpermuted_input must be a CUDA tensor");
    TORCH_CHECK(scores_in.is_cuda(), "scores_in must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");

    TORCH_CHECK(unpermuted_input.is_contiguous(), "unpermuted_input must be contiguous");
    TORCH_CHECK(scores_in.is_contiguous(), "scores_in must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    int const num_tokens = unpermuted_input.size(0);
    int const hidden_size = unpermuted_input.size(1);
    int const num_experts = scores_in.size(1);
    constexpr int top_k = 6;

    TORCH_CHECK(num_experts == 384, "Kolibri router requires 384 experts");
    TORCH_CHECK(bias.size(0) == num_experts, "bias size must match num_experts");

    // Output tensors
    auto permuted_output = th::empty({num_tokens * top_k, hidden_size}, unpermuted_input.options());
    auto permuted_scales = th::empty({num_tokens * top_k}, scores_in.options());
    auto permuted_row_to_unpermuted_row = th::empty({num_tokens * top_k}, th::TensorOptions().dtype(th::kInt32).device(unpermuted_input.device()));
    auto expert_first_token_offset = th::empty({num_experts + 1}, th::TensorOptions().dtype(th::kInt64).device(unpermuted_input.device()));
    auto permuted_token_ids = th::empty({num_tokens * top_k}, th::TensorOptions().dtype(th::kInt32).device(unpermuted_input.device()));

    auto stream = at::cuda::getCurrentCUDAStream(unpermuted_input.device().index());

    if (unpermuted_input.scalar_type() == th::kHalf) {
        tensorrt_llm::kernels::invokeFusedKolibriRouterDispatch<half>(
            reinterpret_cast<half const*>(unpermuted_input.data_ptr()),
            reinterpret_cast<float const*>(scores_in.data_ptr()),
            reinterpret_cast<float const*>(bias.data_ptr()),
            reinterpret_cast<half*>(permuted_output.data_ptr()),
            reinterpret_cast<float*>(permuted_scales.data_ptr()),
            reinterpret_cast<int*>(permuted_row_to_unpermuted_row.data_ptr()),
            reinterpret_cast<int64_t*>(expert_first_token_offset.data_ptr()),
            reinterpret_cast<int*>(permuted_token_ids.data_ptr()),
            num_tokens,
            hidden_size,
            num_experts,
            top_k,
            stream.stream()
        );
    } else if (unpermuted_input.scalar_type() == th::kBFloat16) {
#ifdef ENABLE_BF16
        tensorrt_llm::kernels::invokeFusedKolibriRouterDispatch<__nv_bfloat16>(
            reinterpret_cast<__nv_bfloat16 const*>(unpermuted_input.data_ptr()),
            reinterpret_cast<float const*>(scores_in.data_ptr()),
            reinterpret_cast<float const*>(bias.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(permuted_output.data_ptr()),
            reinterpret_cast<float*>(permuted_scales.data_ptr()),
            reinterpret_cast<int*>(permuted_row_to_unpermuted_row.data_ptr()),
            reinterpret_cast<int64_t*>(expert_first_token_offset.data_ptr()),
            reinterpret_cast<int*>(permuted_token_ids.data_ptr()),
            num_tokens,
            hidden_size,
            num_experts,
            top_k,
            stream.stream()
        );
#else
        TORCH_CHECK(false, "BF16 support not enabled during compilation");
#endif
    } else {
        TORCH_CHECK(false, "Unsupported dtype: only float16 and bfloat16 are supported");
    }

    return {permuted_output, permuted_scales, permuted_row_to_unpermuted_row, expert_first_token_offset};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_kolibri_router_dispatch", &fused_kolibri_router_dispatch, "Fused Kolibri Router + Dispatch + Expansion (CUDA)");
}
