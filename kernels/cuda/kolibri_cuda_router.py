import os
import torch
from torch.utils.cpp_extension import load

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Repo root is 3 levels up: kernels/cuda/ -> kernels/ -> profiling-moe/ -> profiling/
TRT_LLM_CPP = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", "TensorRT-LLM", "cpp"))

import sys

# Ensure CUDA compiler and conda bin (ninja) are in PATH
python_bin = os.path.dirname(sys.executable)
cuda_bin = "/usr/local/cuda/bin"
os.environ["PATH"] = f"{python_bin}:{cuda_bin}:{os.environ.get('PATH', '')}"

# JIT-compile standard Kolibri MoE routing kernel (with 128-bit float4 loads & 100% in-register scores)
kolibri_cuda_ops = load(
    name="kolibri_cuda_ops",
    sources=[
        os.path.join(CURRENT_DIR, "customKolibriRoutingKernels.cu"),
        os.path.join(CURRENT_DIR, "router_bindings.cpp"),
    ],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-lineinfo",
        "-std=c++20",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
    ],
    extra_cflags=[
        "-std=c++20",
    ],
    extra_include_paths=[
        CURRENT_DIR,
        os.path.join(TRT_LLM_CPP, "include"),
        TRT_LLM_CPP,
        os.path.join(TRT_LLM_CPP, "tensorrt_llm"),
    ],
    verbose=True,
)
