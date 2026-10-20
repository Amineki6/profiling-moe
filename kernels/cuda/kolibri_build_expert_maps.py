import os
import sys
import torch
from torch.utils.cpp_extension import load

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRT_LLM_CPP = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", "TensorRT-LLM", "cpp"))

python_bin = os.path.dirname(sys.executable)
cuda_bin = "/usr/local/cuda/bin"
os.environ["PATH"] = f"{python_bin}:{cuda_bin}:{os.environ.get('PATH', '')}"

kolibri_build_expert_maps_ops = load(
    name="kolibri_build_expert_maps_ops",
    sources=[
        os.path.join(CURRENT_DIR, "customKolibriBuildExpertMapsKernel.cu"),
        os.path.join(CURRENT_DIR, "build_expert_maps_bindings.cpp"),
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
    verbose=True,
)

def kolibri_build_expert_maps(token_selected_experts: torch.Tensor, num_experts: int = 384):
    """
    Custom Kolibri Dispatch Mapping Kernel (256 threads, 3 KB SRAM).
    """
    return kolibri_build_expert_maps_ops.kolibri_build_expert_maps(token_selected_experts, num_experts)

def trtllm_build_expert_maps_b32(token_selected_experts: torch.Tensor, num_experts: int = 384):
    """
    TRT-LLM Baseline Dispatch Mapping Kernel (CUB BlockRadixRank B32).
    """
    return kolibri_build_expert_maps_ops.trtllm_build_expert_maps_b32(token_selected_experts, num_experts)

def kolibri_activation_expansion(unpermuted_input: torch.Tensor, permuted_row_to_unpermuted_row: torch.Tensor):
    """
    Custom Multi-Block Vectorized Activation Row Expansion (128-bit uint4).
    """
    return kolibri_build_expert_maps_ops.kolibri_activation_expansion(unpermuted_input, permuted_row_to_unpermuted_row)
