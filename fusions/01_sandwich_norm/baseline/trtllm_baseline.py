import os
import sys
import torch
import torch.nn.functional as F
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
            name="kolibri_trtllm_native_sandwich_norm_baseline_ops",
            sources=[
                os.path.join(CURRENT_DIR, "trtllm_sandwich_norm_baseline.cu"),
                os.path.join(CURRENT_DIR, "baseline_binding.cpp"),
                os.path.join(TRT_LLM_CPP, "tensorrt_llm/kernels/rmsnormKernels.cu"),
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
                "-include", "tensorrt_llm/common/cudaFp8Utils.h",
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


def trtllm_cuda_baseline(
    attn_out: torch.Tensor,
    residual_in: torch.Tensor,
    gamma1: torch.Tensor,
    gamma2: torch.Tensor,
    eps: float = 1e-5,
):
    """
    TRT-LLM 3-Kernel Unfused CUDA Baseline:
      1. Launch 1: RMSNorm(attn_out, gamma1) -> writes temp_post_attn to global DRAM.
      2. Launch 2: VectorAdd(residual_in, temp_post_attn) -> writes updated_residual to DRAM.
      3. Launch 3: RMSNorm(updated_residual, gamma2) -> writes moe_in to DRAM.
    """
    ops = get_baseline_ops()
    return ops.trtllm_sandwich_norm_baseline(attn_out, residual_in, gamma1, gamma2, eps)


def eager_pytorch_baseline(
    attn_out: torch.Tensor,
    residual_in: torch.Tensor,
    gamma1: torch.Tensor,
    gamma2: torch.Tensor,
    eps: float = 1e-5,
):
    """
    Reference Eager PyTorch Baseline (Unfused):
        post_attn = F.rms_norm(attn_out, (d,), gamma1, eps=eps)
        updated_residual = residual_in + post_attn
        moe_in = F.rms_norm(updated_residual, (d,), gamma2, eps=eps)
    """
    d = attn_out.shape[-1]
    post_attn = F.rms_norm(attn_out, (d,), gamma1, eps=eps)
    updated_residual = residual_in + post_attn
    moe_in = F.rms_norm(updated_residual, (d,), gamma2, eps=eps)
    return updated_residual, moe_in
