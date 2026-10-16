import os
import torch
from torch.utils.cpp_extension import load

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Compiles on first import; subsequent imports are instant cache hits
kolibri_cuda_ops = load(
    name="kolibri_cuda_ops",
    sources=[
        os.path.join(CURRENT_DIR, "customMoeRoutingKernels.cu"),
        os.path.join(CURRENT_DIR, "router_bindings.cpp"),
    ],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-lineinfo",  # Essential for Nsight Compute source code mapping
        "-std=c++17",
    ],
    extra_include_paths=[CURRENT_DIR],
    verbose=True,
)
