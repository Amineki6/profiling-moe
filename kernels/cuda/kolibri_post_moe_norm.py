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

# JIT-compile custom Kolibri fused add + post-MoE norm kernel
kolibri_post_moe_norm_ops = load(
    name="kolibri_post_moe_norm_ops",
    sources=[
        os.path.join(CURRENT_DIR, "customKolibriPostMoeNormKernels.cu"),
        os.path.join(CURRENT_DIR, "post_moe_norm_bindings.cpp"),
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

def fused_add_post_moe_norm(routed_out: torch.Tensor, shared_out: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-5):
    """
    Fused Routed + Shared Output Addition + Post-MoE Sandwich Norm.

    Computes:
        moe_out = RMSNorm(routed_out + shared_out, gamma)
    """
    return kolibri_post_moe_norm_ops.fused_add_post_moe_norm(routed_out, shared_out, gamma, eps)

def copy_baseline(routed_out: torch.Tensor, shared_out: torch.Tensor):
    """
    Pure 2-in 1-out 128-bit Copy Baseline (Upper-bound DRAM/L2 speed-of-light test).
    Reads routed_out + shared_out (11.8 MB at n=1152) and writes moe_out (5.9 MB).
    """
    return kolibri_post_moe_norm_ops.copy_baseline(routed_out, shared_out)
