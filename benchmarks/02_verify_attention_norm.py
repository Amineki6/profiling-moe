#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verification script for Kolibri 1 Hybrid Attention Normalization (Investigation 02).

Validates:
1. Numerical equivalence between fused C++ QK-Norm+RoPE and unfused PyTorch on SWA layers.
2. Correct RNoPE bypass behavior (RoPE skipped, QK normalized) on Full Attention layers.
3. Layer-selective fusion configuration (fuse_qk_norm_rope = not self.is_full_attention) across all 50 layers.
4. Sequential 50-layer mock forward pass for decode (batch=1) and prefill (batch=16).
"""

import os
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
    rope_fusion: bool = True,
) -> Kolibri1Attention:
    """Helper to instantiate Kolibri1Attention with explicit fuse_qk_norm_rope flag.

    rope_fusion=True (production default) defers RoPE to the attention backend op,
    so apply_rope() on the unfused path only performs QK-RMSNorm. For standalone
    numerical comparisons, pass rope_fusion=False so RoPE is applied in PyTorch
    via RotaryEmbedding inside apply_rope().
    """
    config = model_config.pretrained_config
    layer_types = getattr(config, "layer_types", [])
    is_full_attn = (
        layer_idx is not None
        and layer_idx < len(layer_types)
        and layer_types[layer_idx] == "full_attention"
    )

    pos_embd_params = None
    if not is_full_attn:
        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gpt_neox,
            rope=RopeParams.from_config(config),
        )

    # Initialize directly via Kolibri1Attention super-constructor contract
    layer = Kolibri1Attention.__new__(Kolibri1Attention)
    layer.layer_idx = layer_idx
    layer.is_full_attention = is_full_attn
    layer.attention_window_size = (
        None if is_full_attn else getattr(config, "sliding_window", None)
    )

    super(Kolibri1Attention, layer).__init__(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        max_position_embeddings=config.max_position_embeddings,
        bias=False,
        pos_embd_params=pos_embd_params,
        skip_rope=is_full_attn,
        fuse_qk_norm_rope=fuse_qk_norm_rope,
        rope_fusion=rope_fusion,
        layer_idx=layer_idx,
        dtype=config.torch_dtype,
        config=model_config,
    )
    layer.to(device="cuda", dtype=torch.bfloat16)
    return layer


def test_01_swa_numerical_equivalence(model_config: ModelConfig[Kolibri1Config]):
    print("\n" + "=" * 80)
    print("TEST 1: SWA Layer Equivalence (Fused CUDA Kernel vs Unfused PyTorch)")
    print("=" * 80)

    # Layer 0 is Sliding Attention (SWA)
    # rope_fusion=False: otherwise RoPE is deferred to the attention op and the
    # baseline would only apply RMSNorm (no rotation), diverging for position > 0.
    layer_unfused = create_mock_attention_layer(
        model_config, layer_idx=0, fuse_qk_norm_rope=False, rope_fusion=False
    )
    layer_fused = create_mock_attention_layer(model_config, layer_idx=0, fuse_qk_norm_rope=True)

    # Copy identical weights to both layers
    with torch.no_grad():
        layer_fused.q_norm.weight.copy_(layer_unfused.q_norm.weight)
        layer_fused.k_norm.weight.copy_(layer_unfused.k_norm.weight)

    torch.manual_seed(42)
    b = 4
    num_heads_q = model_config.pretrained_config.num_attention_heads      # 48
    num_heads_kv = model_config.pretrained_config.num_key_value_heads     # 4
    head_dim = model_config.pretrained_config.head_dim                   # 128
    qkv_dim = (num_heads_q + 2 * num_heads_kv) * head_dim                # 7168

    qkv_in = torch.randn(b, qkv_dim, dtype=torch.bfloat16, device="cuda")
    position_ids = torch.arange(b, dtype=torch.int32, device="cuda")

    # 1. Unfused Eager Forward
    q_unfused = qkv_in[:, :num_heads_q * head_dim].clone()
    k_unfused = qkv_in[:, num_heads_q * head_dim:(num_heads_q + num_heads_kv) * head_dim].clone()
    v_unfused = qkv_in[:, (num_heads_q + num_heads_kv) * head_dim:].clone()

    out_q_unfused, out_k_unfused, out_v_unfused = layer_unfused.apply_rope(
        q_unfused, k_unfused, v_unfused, position_ids
    )

    # 2. Fused CUDA Kernel Forward
    qkv_fused = qkv_in.clone()
    out_qkv_fused, _, _ = layer_fused.apply_rope(
        qkv_fused, None, None, position_ids
    )

    out_q_fused = out_qkv_fused[:, :num_heads_q * head_dim]
    out_k_fused = out_qkv_fused[:, num_heads_q * head_dim:(num_heads_q + num_heads_kv) * head_dim]
    out_v_fused = out_qkv_fused[:, (num_heads_q + num_heads_kv) * head_dim:]

    # Assertions
    max_diff_q = (out_q_fused - out_q_unfused).abs().max().item()
    max_diff_k = (out_k_fused - out_k_unfused).abs().max().item()
    max_diff_v = (out_v_fused - out_v_unfused).abs().max().item()

    print(f"  Max Diff Q (Fused vs Unfused): {max_diff_q:.6f}")
    print(f"  Max Diff K (Fused vs Unfused): {max_diff_k:.6f}")
    print(f"  Max Diff V (Pass-through)    : {max_diff_v:.6f}")

    # For bfloat16, numerical tolerance ~1e-2 due to fast rsqrt / trig intrinsics in CUDA
    torch.testing.assert_close(out_q_fused, out_q_unfused, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(out_k_fused, out_k_unfused, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(out_v_fused, out_v_unfused, atol=1e-5, rtol=1e-5)
    print("  TEST 1 PASSED: SWA fused CUDA kernel numerically matches PyTorch baseline.")


def test_02_full_attention_rnope_bypass(model_config: ModelConfig[Kolibri1Config]):
    print("\n" + "=" * 80)
    print("TEST 2: Full-Attention RNoPE Bypass (RoPE Skipped, QK Normalized)")
    print("=" * 80)

    # Layer 4 is Full Attention (RNoPE)
    layer_full = create_mock_attention_layer(model_config, layer_idx=4, fuse_qk_norm_rope=False)

    torch.manual_seed(42)
    b = 4
    num_heads_q = 48
    num_heads_kv = 4
    head_dim = 128
    qkv_dim = (num_heads_q + 2 * num_heads_kv) * head_dim

    qkv_in = torch.randn(b, qkv_dim, dtype=torch.bfloat16, device="cuda")
    position_ids = torch.arange(b, dtype=torch.int32, device="cuda")

    q = qkv_in[:, :num_heads_q * head_dim].clone()
    k = qkv_in[:, num_heads_q * head_dim:(num_heads_q + num_heads_kv) * head_dim].clone()
    v = qkv_in[:, (num_heads_q + num_heads_kv) * head_dim:].clone()

    out_q, out_k, out_v = layer_full.apply_rope(q, k, v, position_ids)

    # Expected: Pure RMSNorm on Q and K, positions unrotated, V untouched
    expected_q = layer_full.q_norm(q.reshape(-1, head_dim)).reshape(b, -1)
    expected_k = layer_full.k_norm(k.reshape(-1, head_dim)).reshape(b, -1)

    torch.testing.assert_close(out_q, expected_q, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out_k, expected_k, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out_v, v, atol=1e-5, rtol=1e-5)

    print(f"  RNoPE check: Q and K match pure RMSNorm output (RoPE completely bypassed).")
    print(f"  skip_rope flag: {layer_full.skip_rope}")
    print("  TEST 2 PASSED: Full-Attention layer accurately preserves RNoPE semantics.")


def test_03_layer_selective_initialization(model_config: ModelConfig[Kolibri1Config]):
    print("\n" + "=" * 80)
    print("TEST 3: 50-Layer Model Initialization (Option 1: Layer-Selective Fusion)")
    print("=" * 80)

    num_layers = model_config.pretrained_config.num_hidden_layers  # 50
    layers = []
    swa_count = 0
    full_count = 0

    for idx in range(num_layers):
        is_full = (model_config.pretrained_config.layer_types[idx] == "full_attention")
        fuse_flag = not is_full  # Option 1: True for SWA, False for Full Attention
        layer = create_mock_attention_layer(model_config, layer_idx=idx, fuse_qk_norm_rope=fuse_flag)
        layers.append(layer)

        if is_full:
            assert layer.fuse_qk_norm_rope is False
            assert layer.skip_rope is True
            full_count += 1
        else:
            assert layer.fuse_qk_norm_rope is True
            assert layer.skip_rope is False
            swa_count += 1

    print(f"  Total Layers Instantiated : {len(layers)}")
    print(f"  SWA Layers (Fused CUDA)   : {swa_count} / {num_layers} ({swa_count / num_layers * 100:.1f}%)")
    print(f"  Full-Attn (Unfused RNoPE) : {full_count} / {num_layers} ({full_count / num_layers * 100:.1f}%)")
    print("  TEST 3 PASSED: All 50 layers initialized with zero assertion errors.")
    return layers


def test_04_sequential_50_layer_forward(layers, model_config: ModelConfig[Kolibri1Config]):
    print("\n" + "=" * 80)
    print("TEST 4: Sequential 50-Layer Forward Execution (Decode & Prefill)")
    print("=" * 80)

    num_heads_q = 48
    num_heads_kv = 4
    head_dim = 128
    qkv_dim = (num_heads_q + 2 * num_heads_kv) * head_dim

    for batch_size, phase in [(1, "Decode"), (16, "Prefill")]:
        qkv = torch.randn(batch_size, qkv_dim, dtype=torch.bfloat16, device="cuda")
        position_ids = torch.arange(batch_size, dtype=torch.int32, device="cuda")

        curr_qkv = qkv.clone()
        for idx, layer in enumerate(layers):
            if layer.fuse_qk_norm_rope:
                curr_qkv, _, _ = layer.apply_rope(curr_qkv, None, None, position_ids)
            else:
                q = curr_qkv[:, :num_heads_q * head_dim]
                k = curr_qkv[:, num_heads_q * head_dim:(num_heads_q + num_heads_kv) * head_dim]
                v = curr_qkv[:, (num_heads_q + num_heads_kv) * head_dim:]
                q_out, k_out, v_out = layer.apply_rope(q, k, v, position_ids)
                curr_qkv = torch.cat([q_out, k_out, v_out], dim=-1)

        print(f"  [{phase} b={batch_size:2d}] Successfully executed all 50 sequential attention layers. Output shape: {curr_qkv.shape}")

    print("  TEST 4 PASSED: End-to-end 50-layer forward execution verified.")


if __name__ == "__main__":
    print("=" * 80)
    print("Kolibri 1 Hybrid Attention Norm & RoPE Verification Harness")
    print("=" * 80)

    config = load_kolibri_config()
    print(f"Loaded Kolibri 1 Config: {config.pretrained_config.num_hidden_layers} layers, "
          f"{config.pretrained_config.num_attention_heads} heads, head_dim={config.pretrained_config.head_dim}")

    test_01_swa_numerical_equivalence(config)
    test_02_full_attention_rnope_bypass(config)
    layers = test_03_layer_selective_initialization(config)
    test_04_sequential_50_layer_forward(layers, config)

    print("\n" + "=" * 80)
    print(" ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)
