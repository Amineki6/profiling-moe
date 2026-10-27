import os
import sys
import torch
from torch.utils.cpp_extension import load

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure CUDA compiler and ninja are in PATH
python_bin = os.path.dirname(sys.executable)
cuda_bin = "/usr/local/cuda/bin"
os.environ["PATH"] = f"{python_bin}:{cuda_bin}:{os.environ.get('PATH', '')}"

_fused_ops = None


def get_fused_sandwich_norm_ops():
    global _fused_ops
    if _fused_ops is None:
        _fused_ops = load(
            name="kolibri_fused_sandwich_norm_ops",
            sources=[
                os.path.join(CURRENT_DIR, "custom_sandwich_norm.cu"),
                os.path.join(CURRENT_DIR, "sandwich_norm_binding.cpp"),
            ],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-lineinfo",
                "-std=c++20",
                "-arch=sm_89",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
            ],
            extra_cflags=[
                "-std=c++20",
                "-O3",
            ],
            extra_include_paths=[CURRENT_DIR],
            verbose=False,
        )
    return _fused_ops


def fused_sandwich_norm(
    attn_out: torch.Tensor,
    residual_in: torch.Tensor,
    gamma1: torch.Tensor,
    gamma2: torch.Tensor,
    eps: float = 1e-5,
):
    """
    Fused Sandwich Norm + Residual Add + Pre-MoE Norm (Camera-Ready Production Kernel).

    Mathematically computes in a single CUDA pass (100% in-register, 0 intermediate DRAM writes):
        post_attn = RMSNorm(attn_out, gamma1)
        updated_residual = residual_in + post_attn
        moe_in = RMSNorm(updated_residual, gamma2)

    Args:
        attn_out: [..., hidden_dim] Tensor (BF16/FP16)
        residual_in: [..., hidden_dim] Tensor (BF16/FP16)
        gamma1: [hidden_dim] Tensor (Post-Attention Sandwich Norm weight)
        gamma2: [hidden_dim] Tensor (Pre-MoE Norm weight)
        eps: epsilon for RMSNorm

    Returns:
        tuple (updated_residual, moe_in)
    """
    ops = get_fused_sandwich_norm_ops()
    return ops.fused_sandwich_norm(attn_out, residual_in, gamma1, gamma2, eps)
