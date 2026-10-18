#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/stringUtils.h"
#include "tensorrt_llm/common/tllmException.h"
#include "tensorrt_llm/common/logger.h"
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

namespace kernels {
    void gate_forward(void* scores_in, void* bias, void* input_ids, void* tid2eid, void* out_weights, void* out_indices, int batch_size, int n_experts, float route_scale, bool is_hash, cudaStream_t stream);
    void gate_forward_vB(void* scores_in, void* bias, void* input_ids, void* tid2eid, void* out_weights, void* out_indices, int batch_size, int n_experts, float route_scale, bool is_hash, cudaStream_t stream);
    void gate_forward_vFloat4(void* scores_in, void* bias, void* input_ids, void* tid2eid, void* out_weights, void* out_indices, int batch_size, int n_experts, float route_scale, bool is_hash, cudaStream_t stream);
}

TRTLLM_NAMESPACE_END

namespace th = torch;

static std::tuple<th::Tensor, th::Tensor> run_gate_forward_helper(
    void (*fn)(void*, void*, void*, void*, void*, void*, int, int, float, bool, cudaStream_t),
    th::Tensor logits,
    th::Tensor bias,
    int64_t topk
) {
    TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
    TORCH_CHECK(logits.dim() == 2, "logits must be 2D [batch_size, n_experts]");
    auto const batch_size = logits.size(0);
    auto const n_experts = logits.size(1);
    TORCH_CHECK(n_experts == 256 || n_experts == 384, "n_experts must be 256 or 384 (got ", n_experts, ")");
    TORCH_CHECK(topk == 6, "Currently gate_forward kernel only supports topk=6 (got ", topk, ")");

    auto logits_f32 = logits.scalar_type() == th::kFloat32 ? logits.contiguous() : logits.to(th::kFloat32).contiguous();
    th::Tensor bias_f32;
    if (bias.defined() && bias.numel() > 0) {
        bias_f32 = bias.scalar_type() == th::kFloat32 ? bias.contiguous() : bias.to(th::kFloat32).contiguous();
    } else {
        bias_f32 = th::zeros({n_experts}, logits_f32.options());
    }

    auto out_weights = th::empty({batch_size, topk}, logits_f32.options());
    auto out_indices = th::empty({batch_size, topk}, th::TensorOptions().dtype(th::kInt32).device(logits.device()));

    auto stream = at::cuda::getCurrentCUDAStream(logits.device().index());

    fn(
        logits_f32.data_ptr<float>(),
        bias_f32.data_ptr<float>(),
        nullptr,
        nullptr,
        out_weights.data_ptr<float>(),
        out_indices.data_ptr<int>(),
        static_cast<int>(batch_size),
        static_cast<int>(n_experts),
        1.0f,
        false,
        stream.stream()
    );

    if (logits.scalar_type() != th::kFloat32) {
        return {out_indices, out_weights.to(logits.scalar_type())};
    }
    return {out_indices, out_weights};
}

std::tuple<th::Tensor, th::Tensor> forward_vA(th::Tensor logits, th::Tensor bias, int64_t topk) {
    return run_gate_forward_helper(&tensorrt_llm::kernels::gate_forward, logits, bias, topk);
}

std::tuple<th::Tensor, th::Tensor> forward_vB(th::Tensor logits, th::Tensor bias, int64_t topk) {
    return run_gate_forward_helper(&tensorrt_llm::kernels::gate_forward_vB, logits, bias, topk);
}

std::tuple<th::Tensor, th::Tensor> forward_vFloat4(th::Tensor logits, th::Tensor bias, int64_t topk) {
    return run_gate_forward_helper(&tensorrt_llm::kernels::gate_forward_vFloat4, logits, bias, topk);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_vA", &forward_vA, "Kolibri MoE Router CUDA vA (Original)");
    m.def("forward_vB", &forward_vB, "Kolibri MoE Router CUDA vB (Edited Registers)");
    m.def("forward_vFloat4", &forward_vFloat4, "Kolibri MoE Router CUDA vFloat4 (Vectorized 128-bit Loads)");
}
