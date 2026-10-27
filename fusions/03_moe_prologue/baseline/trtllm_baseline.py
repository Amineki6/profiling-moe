import os
import sys
import torch
from torch.utils.cpp_extension import load

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

python_bin = os.path.dirname(sys.executable)
cuda_bin = "/usr/local/cuda/bin"
os.environ["PATH"] = f"{python_bin}:{cuda_bin}:{os.environ.get('PATH', '')}"

TRT_LLM_CPP = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", "..", "TensorRT-LLM", "cpp"))

_baseline_ops = None


def get_baseline_ops():
    global _baseline_ops
    if _baseline_ops is None:
        _baseline_ops = load(
            name="kolibri_trtllm_native_moe_prologue_baseline_ops",
            sources=[
                os.path.join(CURRENT_DIR, "trtllm_moe_prologue_baseline.cu"),
                os.path.join(CURRENT_DIR, "baseline_binding.cpp"),
                os.path.join(TRT_LLM_CPP, "tensorrt_llm/kernels/moe/routing/kolibriRoutingKernels.cu"),
                os.path.join(TRT_LLM_CPP, "tensorrt_llm/common/tllmException.cpp"),
                os.path.join(TRT_LLM_CPP, "tensorrt_llm/common/stringUtils.cpp"),
                os.path.join(TRT_LLM_CPP, "tensorrt_llm/common/logger.cpp"),
            ],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-lineinfo",
                "-std=c++20",
                "-arch=sm_89",
                "-DENABLE_BF16",
                "-DENABLE_FP8",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
            ],
            extra_cflags=[
                "-std=c++20",
                "-O3",
                "-DENABLE_BF16",
                "-DENABLE_FP8",
            ],
            extra_include_paths=[
                CURRENT_DIR,
                TRT_LLM_CPP,
                os.path.join(TRT_LLM_CPP, "include"),
            ],
            verbose=False,
        )
    return _baseline_ops


def trtllm_router(scores: torch.Tensor, bias: torch.Tensor = None):
    ops = get_baseline_ops()
    return ops.trtllm_router(scores, bias)


def trtllm_build_expert_maps(token_selected_experts: torch.Tensor, num_experts: int = 384):
    ops = get_baseline_ops()
    return ops.trtllm_build_expert_maps(token_selected_experts, num_experts)


def trtllm_activation_expansion(unpermuted_input: torch.Tensor, permuted_row_to_unpermuted: torch.Tensor):
    ops = get_baseline_ops()
    return ops.trtllm_activation_expansion(unpermuted_input, permuted_row_to_unpermuted)


def trtllm_moe_prologue(unpermuted_input: torch.Tensor, router_logits: torch.Tensor, router_bias: torch.Tensor = None):
    ops = get_baseline_ops()
    return ops.trtllm_moe_prologue(unpermuted_input, router_logits, router_bias)
