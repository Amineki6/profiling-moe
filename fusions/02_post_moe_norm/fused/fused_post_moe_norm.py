import os
import sys
import torch
from torch.utils.cpp_extension import load

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure active conda environment and system CUDA bin are prioritized in PATH for nvcc
python_bin = os.path.dirname(sys.executable)
cuda_bin = "/usr/local/cuda/bin"
os.environ["PATH"] = f"{python_bin}:{cuda_bin}:{os.environ.get('PATH', '')}"

_fused_ops = None


def get_fused_ops():
    """
    Lazy JIT compilation of the production Fused Add + Post-MoE Norm CUDA extension.
    Builds once and caches in PyTorch extensions directory.
    """
    global _fused_ops
    if _fused_ops is None:
        _fused_ops = load(
            name="kolibri_fused_post_moe_norm_ops",
            sources=[
                os.path.join(CURRENT_DIR, "custom_post_moe_norm.cu"),
                os.path.join(CURRENT_DIR, "post_moe_norm_binding.cpp"),
            ],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-lineinfo",
                "-std=c++20",
                "-arch=sm_89",
                "-DENABLE_BF16",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
            ],
            extra_cflags=[
                "-std=c++20",
                "-O3",
                "-DENABLE_BF16",
            ],
            extra_include_paths=[
                CURRENT_DIR,
            ],
            verbose=False,
        )
    return _fused_ops


def fused_add_post_moe_norm(
    routed_out: torch.Tensor,
    shared_out: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Production Fused Add + Post-MoE Norm:
        moe_out = RMSNorm(routed_out + shared_out, gamma, eps=eps)

    Executed completely in registers within a single CTA per token row.
    Zero intermediate DRAM writes, zero dynamic shared memory.

    Args:
        routed_out: Tensor of shape [..., hidden_dim] (BF16, FP16, or FP32).
        shared_out: Tensor of shape [..., hidden_dim] (matching routed_out).
        gamma:      Post-MoE norm weight tensor of shape [hidden_dim].
        eps:        Numerical stability epsilon (default: 1e-5).

    Returns:
        moe_out:    Normalized output tensor of shape [..., hidden_dim].
    """
    ops = get_fused_ops()
    return ops.fused_add_post_moe_norm(routed_out, shared_out, gamma, eps)
