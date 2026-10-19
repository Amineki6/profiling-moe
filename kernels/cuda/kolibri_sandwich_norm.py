import os
import torch
from torch.utils.cpp_extension import load

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRT_LLM_CPP = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", "TensorRT-LLM", "cpp"))

import sys

# Ensure CUDA compiler and conda bin (ninja) are in PATH
python_bin = os.path.dirname(sys.executable)
cuda_bin = "/usr/local/cuda/bin"
os.environ["PATH"] = f"{python_bin}:{cuda_bin}:{os.environ.get('PATH', '')}"

# JIT-compile custom Kolibri fused sandwich norm kernel
kolibri_sandwich_norm_ops = load(
    name="kolibri_sandwich_norm_ops",
    sources=[
        os.path.join(CURRENT_DIR, "customKolibriSandwichNormKernels.cu"),
        os.path.join(CURRENT_DIR, "sandwich_norm_bindings.cpp"),
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
    verbose=False,
)

def fused_sandwich_norm(attn_out: torch.Tensor, residual_in: torch.Tensor, gamma1: torch.Tensor, gamma2: torch.Tensor, eps: float = 1e-5):
    """
    Fused Sandwich Norm + Residual Add + Pre-MoE Norm.
    
    Args:
        attn_out: [..., hidden_dim] Tensor
        residual_in: [..., hidden_dim] Tensor
        gamma1: [hidden_dim] Tensor (Sandwich norm weights)
        gamma2: [hidden_dim] Tensor (Pre-MoE norm weights)
        eps: epsilon for RMSNorm
        
    Returns:
        (updated_residual_out, moe_in_out)
    """
    return kolibri_sandwich_norm_ops.fused_sandwich_norm(attn_out, residual_in, gamma1, gamma2, eps)
