import os
import sys
import torch
from torch.utils.cpp_extension import load

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

python_bin = os.path.dirname(sys.executable)
cuda_bin = "/usr/local/cuda/bin"
os.environ["PATH"] = f"{python_bin}:{cuda_bin}:{os.environ.get('PATH', '')}"

_fused_ops = None


def get_fused_ops():
    """
    Lazy JIT compilation of the production Fused MoE Prologue CUDA extension.
    Compiles cleanly without external dependencies.
    """
    global _fused_ops
    if _fused_ops is None:
        _fused_ops = load(
            name="kolibri_fused_moe_prologue_ops",
            sources=[
                os.path.join(CURRENT_DIR, "custom_moe_prologue.cu"),
                os.path.join(CURRENT_DIR, "moe_prologue_binding.cpp"),
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
                os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", "..", "TensorRT-LLM", "cpp")),
                os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", "..", "TensorRT-LLM", "cpp", "include")),
            ],
            verbose=False,
        )
    return _fused_ops


def kolibri_router(scores: torch.Tensor, bias: torch.Tensor = None):
    """
    Fast Top-6 MoE Router:
        scores: [num_tokens, 384] float32
        bias:   [384] float32 (optional)
    Returns:
        routing_weights:  [num_tokens, 6] float32
        selected_experts: [num_tokens, 6] int32
    """
    ops = get_fused_ops()
    return ops.kolibri_router(scores, bias)


def kolibri_dispatch_map(
    token_selected_experts: torch.Tensor,
    num_experts: int = 384,
    force_single_block: bool = False,
    force_multi_block: bool = False,
):
    """
    Dynamic Hybrid Dispatch Mapping:
        Automatically selects:
          - Single-block 256-th kernel when num_tokens < 512 (Decode, 3 KB smem, 0 launch overhead)
          - 3-stage multi-block pipeline when num_tokens >= 512 (Prefill, 128 SM concurrency)
    Returns:
        (permuted_row_to_unpermuted, unpermuted_to_permuted, expert_first_token_offset)
    """
    ops = get_fused_ops()
    return ops.kolibri_dispatch_map(token_selected_experts, num_experts, force_single_block, force_multi_block)


def kolibri_activation_expansion(
    unpermuted_input: torch.Tensor,
    permuted_row_to_unpermuted: torch.Tensor,
):
    """
    128-bit Vectorized Activation Row Expansion (uint4 memory copy):
        unpermuted_input:           [num_tokens, hidden_size] (BF16 or FP16)
        permuted_row_to_unpermuted: [num_tokens * 6] (int32)
    Returns:
        permuted_output:            [num_tokens * 6, hidden_size]
    """
    ops = get_fused_ops()
    return ops.kolibri_activation_expansion(unpermuted_input, permuted_row_to_unpermuted)


def fused_moe_prologue(
    unpermuted_input: torch.Tensor,
    router_logits: torch.Tensor,
    router_bias: torch.Tensor = None,
):
    """
    End-to-End Fused MoE Prologue:
      1. Top-6 router reduction
      2. Dynamic hybrid dispatch mapping (Single-Block < 512, Multi-Block >= 512)
      3. Vectorized activation row expansion
    Returns:
      (permuted_output, routing_weights, selected_experts, expert_offsets, permuted_row_to_unpermuted, unpermuted_to_permuted)
    """
    ops = get_fused_ops()
    return ops.fused_moe_prologue(unpermuted_input, router_logits, router_bias)
