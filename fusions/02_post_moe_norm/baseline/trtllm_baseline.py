import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure active conda environment and system CUDA bin are prioritized in PATH for nvcc
python_bin = os.path.dirname(sys.executable)
cuda_bin = "/usr/local/cuda/bin"
os.environ["PATH"] = f"{python_bin}:{cuda_bin}:{os.environ.get('PATH', '')}"

TRT_LLM_CPP = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", "..", "TensorRT-LLM", "cpp"))

_baseline_ops = None


def get_baseline_ops():
    global _baseline_ops
    if _baseline_ops is None:
        _baseline_ops = load(
            name="kolibri_trtllm_native_post_moe_norm_baseline_ops",
            sources=[
                os.path.join(CURRENT_DIR, "trtllm_post_moe_norm_baseline.cu"),
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
    routed_out: torch.Tensor,
    shared_out: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    TRT-LLM 2-Kernel Unfused CUDA Baseline:
      1. Launch 1: VectorAdd(routed_out, shared_out) -> writes temp_add to DRAM.
      2. Launch 2: Native TRT-LLM invokeGeneralRmsNorm(temp_add, gamma) -> writes moe_out to DRAM.
    """
    ops = get_baseline_ops()
    return ops.trtllm_post_moe_norm_baseline(routed_out, shared_out, gamma, eps)


def eager_pytorch_baseline(
    routed_out: torch.Tensor,
    shared_out: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Reference Eager PyTorch Baseline (Unfused):
        combined = routed_out + shared_out
        moe_out = F.rms_norm(combined, (d,), gamma, eps=eps)
    """
    d = routed_out.shape[-1]
    combined = routed_out + shared_out
    return F.rms_norm(combined, (d,), gamma, eps=eps)
