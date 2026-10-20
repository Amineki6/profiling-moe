import os
import sys
import torch
from torch.utils.cpp_extension import load

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRT_LLM_CPP = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", "TensorRT-LLM", "cpp"))

# Ensure CUDA compiler and conda bin (ninja) are in PATH
python_bin = os.path.dirname(sys.executable)
cuda_bin = "/usr/local/cuda/bin"
os.environ["PATH"] = f"{python_bin}:{cuda_bin}:{os.environ.get('PATH', '')}"

# JIT-compile custom Kolibri fused router + dispatch kernel
kolibri_router_dispatch_ops = load(
    name="kolibri_router_dispatch_ops",
    sources=[
        os.path.join(CURRENT_DIR, "customKolibriRouterDispatchMapKernels.cu"),
        os.path.join(CURRENT_DIR, "customKolibriActivationExpansionKernels.cu"),
        os.path.join(CURRENT_DIR, "customKolibriRouterDispatchKernels.cu"),
        os.path.join(CURRENT_DIR, "router_dispatch_bindings.cpp"),
    ],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-lineinfo",
        "-std=c++20",
        "-DENABLE_BF16",
        "-DENABLE_FP8",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
    ],
    extra_cflags=[
        "-std=c++20",
        "-DENABLE_BF16",
        "-DENABLE_FP8",
    ],
    extra_include_paths=[
        CURRENT_DIR,
        os.path.join(TRT_LLM_CPP, "include"),
        TRT_LLM_CPP,
        os.path.join(TRT_LLM_CPP, "tensorrt_llm"),
    ],
    verbose=True,
)

def fused_kolibri_router_dispatch(unpermuted_input: torch.Tensor, scores_in: torch.Tensor, bias: torch.Tensor):
    """
    Fused Kolibri Router Top-K + CUTLASS Expert Dispatch Mapping + Activation Expansion.

    Inputs:
        unpermuted_input: [num_tokens, hidden_size] (FP16 or BF16)
        scores_in:        [num_tokens, num_experts=384] (FP32)
        bias:             [num_experts=384] (FP32)

    Returns:
        permuted_output:                [num_tokens * 6, hidden_size]
        permuted_scales:                [num_tokens * 6]
        permuted_row_to_unpermuted_row: [num_tokens * 6]
        expert_first_token_offset:      [num_experts + 1]
    """
    return kolibri_router_dispatch_ops.fused_kolibri_router_dispatch(unpermuted_input, scores_in, bias)
