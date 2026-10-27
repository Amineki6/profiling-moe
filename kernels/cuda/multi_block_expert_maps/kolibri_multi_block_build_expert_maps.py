import os
import sys
import torch
from torch.utils.cpp_extension import load

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_CUDA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
TRT_LLM_CPP = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", "..", "TensorRT-LLM", "cpp"))

python_bin = os.path.dirname(sys.executable)
cuda_bin = "/usr/local/cuda/bin"
os.environ["PATH"] = f"{python_bin}:{cuda_bin}:{os.environ.get('PATH', '')}"

kolibri_multi_block_build_expert_maps_ops = load(
    name="kolibri_multi_block_build_expert_maps_ops",
    sources=[
        os.path.join(CURRENT_DIR, "customKolibriMultiBlockBuildExpertMapsKernels.cu"),
        os.path.join(CURRENT_DIR, "multi_block_build_expert_maps_bindings.cpp"),
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
        PARENT_CUDA_DIR,
        os.path.join(TRT_LLM_CPP, "include"),
        TRT_LLM_CPP,
        os.path.join(TRT_LLM_CPP, "tensorrt_llm"),
    ],
    verbose=True,
)

def kolibri_multi_block_build_expert_maps(
    token_selected_experts: torch.Tensor,
    num_experts: int = 384,
    max_blocks: int = 128,
    return_inverse_map: bool = True
):
    """
    Kolibri Multi-Block 3-Stage Dispatch Mapping Pipeline:
      Kernel 1: Local expert histograms across G blocks in shared memory.
      Kernel 2: Global prefix sum across blocks and experts.
      Kernel 3: Intra-block ranking and direct global scatter.
    """
    return kolibri_multi_block_build_expert_maps_ops.kolibri_multi_block_build_expert_maps(
        token_selected_experts, num_experts, max_blocks, return_inverse_map
    )
