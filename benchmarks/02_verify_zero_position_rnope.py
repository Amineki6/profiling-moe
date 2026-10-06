#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verification script for Option 2: Zero-Position Workaround for RNoPE layers.

Mathematically proves and numerically verifies that passing position_ids = 0 to
TensorRT-LLM's fused C++ QK-Norm+RoPE kernel (fusedQKNormRopeKernel) produces
exact numerical equivalence with a pure per-head RMSNorm (i.e., perfect RNoPE bypass).
"""

import sys
from pathlib import Path
import torch

from tensorrt_llm._torch.configs.kolibri import Kolibri1Config
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_kolibri import Kolibri1Attention
from tensorrt_llm._torch.attention.backends.interface import PositionalEmbeddingParams, RopeParams
from tensorrt_llm.functional import PositionEmbeddingType


def load_kolibri_config() -> ModelConfig[Kolibri1Config]:
    config_path = Path(__file__).resolve().parent.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    pretrained_config = Kolibri1Config.from_json_file(str(config_path))
    return ModelConfig(pretrained_config=pretrained_config)


def create_mock_attention_layer(
    model_config: ModelConfig[Kolibri1Config],
    layer_idx: int,
    fuse_qk_norm_rope: bool,
    skip_rope: bool = False,
) -> Kolibri1Attention:
    config = model_config.pretrained_config
    layer_types = getattr(config, "layer_types", [])
    is_full_attn = (
        layer_idx is not None
        and layer_idx < len(layer_types)
        and layer_types[layer_idx] == "full_attention"
    )

    pos_embd_params = PositionalEmbeddingParams(
        type=PositionEmbeddingType.rope_gpt_neox,
        rope=RopeParams.from_config(config),
    )

    layer = Kolibri1Attention.__new__(Kolibri1Attention)
    layer.layer_idx = layer_idx
    layer.is_full_attention = is_full_attn
    layer.attention_window_size = None if is_full_attn else getattr(config, "sliding_window", None)

    super(Kolibri1Attention, layer).__init__(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        max_position_embeddings=config.max_position_embeddings,
        bias=False,
        pos_embd_params=pos_embd_params,
        skip_rope=skip_rope,
        fuse_qk_norm_rope=fuse_qk_norm_rope,
        rope_fusion=False,
        layer_idx=layer_idx,
        dtype=config.torch_dtype,
        config=model_config,
    )
    layer.to(device="cuda", dtype=torch.bfloat16)
    return layer


def verify_zero_position_rnope(model_config: ModelConfig[Kolibri1Config]):
    print("=" * 80)
    print("VERIFICATION: Zero-Position Workaround Equivalence with Pure RMSNorm (RNoPE)")
    print("=" * 80)

    cfg = model_config.pretrained_config
    num_heads_q = cfg.num_attention_heads       # 48
    num_heads_kv = cfg.num_key_value_heads      # 4
    head_dim = cfg.head_dim                    # 128
    qkv_dim = (num_heads_q + 2 * num_heads_kv) * head_dim  # 7168

    # 1. Unfused reference layer configured for RNoPE (skip_rope=True, pure RMSNorm)
    layer_unfused = create_mock_attention_layer(
        model_config, layer_idx=4, fuse_qk_norm_rope=False, skip_rope=True
    )

    # 2. Fused CUDA kernel layer (fuse_qk_norm_rope=True, skip_rope=False, but fed position_ids=0)
    layer_fused = create_mock_attention_layer(
        model_config, layer_idx=4, fuse_qk_norm_rope=True, skip_rope=False
    )

    # Ensure identical RMSNorm weights
    with torch.no_grad():
        layer_fused.q_norm.weight.copy_(layer_unfused.q_norm.weight)
        layer_fused.k_norm.weight.copy_(layer_unfused.k_norm.weight)

    print(f"\nModel Configuration:")
    print(f"  Head Dim: {head_dim}, Q Heads: {num_heads_q}, KV Heads: {num_heads_kv}")
    print(f"  Total QKV Hidden Dim: {qkv_dim}")
    print(f"  RMSNorm Epsilon: {cfg.rms_norm_eps}")

    batch_test_sizes = [1, 4, 16, 64]
    print(f"\nTesting batch sizes: {batch_test_sizes}\n")
    print(f"{'Batch':<8} | {'Max Diff Q':<14} | {'Max Diff K':<14} | {'Max Diff V':<14} | {'Status':<10}")
    print("-" * 70)

    for b in batch_test_sizes:
        torch.manual_seed(42 + b)
        qkv_in = torch.randn(b, qkv_dim, dtype=torch.bfloat16, device="cuda")

        # Reference: Unfused pure PyTorch RMSNorm on Q and K, V untouched
        q = qkv_in[:, :num_heads_q * head_dim].clone()
        k = qkv_in[:, num_heads_q * head_dim:(num_heads_q + num_heads_kv) * head_dim].clone()
        v = qkv_in[:, (num_heads_q + num_heads_kv) * head_dim:].clone()

        # Pure RMSNorm ground truth
        ref_q = layer_unfused.q_norm(q.reshape(-1, head_dim)).reshape(b, -1)
        ref_k = layer_unfused.k_norm(k.reshape(-1, head_dim)).reshape(b, -1)
        ref_v = v

        # Option 2: Fused CUDA kernel executed with position_ids = 0
        zero_position_ids = torch.zeros(b, dtype=torch.int32, device="cuda")
        fused_qkv = qkv_in.clone()
        out_fused_qkv, _, _ = layer_fused.apply_rope(fused_qkv, None, None, zero_position_ids)

        fused_q = out_fused_qkv[:, :num_heads_q * head_dim]
        fused_k = out_fused_qkv[:, num_heads_q * head_dim:(num_heads_q + num_heads_kv) * head_dim]
        fused_v = out_fused_qkv[:, (num_heads_q + num_heads_kv) * head_dim:]

        # Differences
        diff_q = (fused_q - ref_q).abs().max().item()
        diff_k = (fused_k - ref_k).abs().max().item()
        diff_v = (fused_v - ref_v).abs().max().item()
        norm_effect_q = (fused_q - q).abs().max().item()

        if b == 1:
            print(f"  Debug [Sample Token, first 4 Query dims]:")
            print(f"    Raw input Q       : {[round(x, 4) for x in q[0, :4].tolist()]}")
            print(f"    Pure RMSNorm Q    : {[round(x, 4) for x in ref_q[0, :4].tolist()]}")
            print(f"    Fused (pos=0) Q   : {[round(x, 4) for x in fused_q[0, :4].tolist()]}")
            print(f"    Norm effect magnitude: {norm_effect_q:.4f}\n")

        # Strict checks (in bfloat16, CUDA rsqrt intrinsics vs PyTorch rsqrt have ~1e-2 precision bounds)
        torch.testing.assert_close(fused_q, ref_q, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(fused_k, ref_k, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(fused_v, ref_v, atol=1e-5, rtol=1e-5)

        # Contrast check: verify that non-zero positions WOULD rotate and diverge
        if b > 1:
            nonzero_pos = torch.arange(b, dtype=torch.int32, device="cuda") + 10
            rotated_qkv, _, _ = layer_fused.apply_rope(qkv_in.clone(), None, None, nonzero_pos)
            rot_q = rotated_qkv[:, :num_heads_q * head_dim]
            rot_divergence = (rot_q - ref_q).abs().max().item()
            assert rot_divergence > 0.1, f"Expected non-zero position to rotate and diverge, got {rot_divergence}"

        print(f"{b:<8} | {diff_q:<14.6f} | {diff_k:<14.6f} | {diff_v:<14.6f} | {'PASSED':<10}")

    print("-" * 70)
    print("\nMATHEMATICAL PROOF & CONCLUSION:")
    print("  When position_ids = 0:")
    print("    theta = position_ids * freq = 0")
    print("    cos(theta) = 1.0,  sin(theta) = 0.0")
    print("    RoPE(x) = x * cos(0) + x_rotated * sin(0) = x * 1.0 + 0 = x")
    print("  Therefore:")
    print("    fusedQKNormRopeKernel(qkv, pos=0) == RMSNorm(qkv)")
    print("  Numerical equivalence is CONFIRMED with zero rotation artifact across all batch sizes.")
    print("=" * 80)


if __name__ == "__main__":
    cfg = load_kolibri_config()
    verify_zero_position_rnope(cfg)
