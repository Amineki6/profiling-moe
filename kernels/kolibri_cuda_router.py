import os
import torch
from torch.utils.cpp_extension import load

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRT_LLM_CPP = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "TensorRT-LLM", "cpp"))

# Compiles on first import; subsequent imports are instant cache hits
kolibri_cuda_ops = load(
    name="kolibri_cuda_ops",
    sources=[
        os.path.join(CURRENT_DIR, "fused_kolibri_router.cu"),
        os.path.join(CURRENT_DIR, "router_bindings.cpp"),
    ],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-lineinfo",  # Essential for Nsight Compute source code mapping
        "-std=c++17",
    ],
    extra_include_paths=[
        CURRENT_DIR,
        TRT_LLM_CPP,
        os.path.join(TRT_LLM_CPP, "tensorrt_llm"),
    ],
    verbose=True,
)
