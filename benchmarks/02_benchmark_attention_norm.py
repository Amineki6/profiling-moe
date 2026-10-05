#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Microbenchmark for Kolibri 1 Hybrid Attention Normalization & RoPE Fusion.

Measures:
1. Isolated SWA layer latency: Unfused vs Fused CUDA vs torch.compile across batch sizes.
2. Isolated Full-Attention layer latency: Unfused vs torch.compile across batch sizes.
3. Full 50-layer model stack decode latency (40 SWA + 10 Full-Attention) in sequence.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import triton.testing



from tensorrt_llm._torch.configs.kolibri import Kolibri1Config
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_kolibri import Kolibri1Attention
from tensorrt_llm._torch.attention.backends.interface import PositionalEmbeddingParams, RopeParams
from tensorrt_llm.functional import PositionEmbeddingType


def load_kolibri_config() -> ModelConfig[Kolibri1Config]:
    config_path = Path(__file__).resolve().parent.parent / "config.json"
    pretrained_config = Kolibri1Config.from_json_file(str(config_path))
    return ModelConfig(pretrained_config=pretrained_config)


def create_mock_attention_layer(
    model_config: ModelConfig[Kolibri1Config],
    layer_idx: int,
    fuse_qk_norm_rope: bool,
    rope_fusion: bool = True,
) -> Kolibri1Attention:
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


@torch.inference_mode()
def run_benchmarks():
    print("=" * 80)
    print("Kolibri 1 Hybrid Attention Norm & RoPE Microbenchmark (RTX 4090)")
    print("=" * 80)

    model_config = load_kolibri_config()
    cfg = model_config.pretrained_config
    num_heads_q = cfg.num_attention_heads      # 48
    num_heads_kv = cfg.num_key_value_heads     # 4
    head_dim = cfg.head_dim                   # 128
    qkv_dim = (num_heads_q + 2 * num_heads_kv) * head_dim  # 7168

    batch_sizes = [1, 4, 8, 16, 32, 64, 128]

    # Instantiate isolated benchmark layers
    swa_unfused = create_mock_attention_layer(model_config, layer_idx=0, fuse_qk_norm_rope=False, rope_fusion=False)
    swa_fused = create_mock_attention_layer(model_config, layer_idx=0, fuse_qk_norm_rope=True)
    full_unfused = create_mock_attention_layer(model_config, layer_idx=4, fuse_qk_norm_rope=False)

    # 1. Warmup and compile wrappers
    def unfused_swa_op(qkv, pos_ids):
        q = qkv[:, :num_heads_q * head_dim]
        k = qkv[:, num_heads_q * head_dim:(num_heads_q + num_heads_kv) * head_dim]
        v = qkv[:, (num_heads_q + num_heads_kv) * head_dim:]
        return swa_unfused.apply_rope(q, k, v, pos_ids)

    def fused_swa_op(qkv, pos_ids):
        return swa_fused.apply_rope(qkv, None, None, pos_ids)

    def unfused_full_op(qkv, pos_ids):
        q = qkv[:, :num_heads_q * head_dim]
        k = qkv[:, num_heads_q * head_dim:(num_heads_q + num_heads_kv) * head_dim]
        v = qkv[:, (num_heads_q + num_heads_kv) * head_dim:]
        return full_unfused.apply_rope(q, k, v, pos_ids)

    print("\nTracing and warming up torch.compile...")
    compiled_swa = torch.compile(unfused_swa_op, mode="reduce-overhead")
    compiled_full = torch.compile(unfused_full_op, mode="reduce-overhead")

    # Dry run compile
    dummy_qkv = torch.randn(1, qkv_dim, dtype=torch.bfloat16, device="cuda")
    dummy_pos = torch.zeros(1, dtype=torch.int32, device="cuda")
    compiled_swa(dummy_qkv, dummy_pos)
    compiled_full(dummy_qkv, dummy_pos)
    torch.cuda.synchronize()

    # --- SECTION A: SWA Microbenchmark ---
    print("\n" + "-" * 80)
    print("SECTION A: Sliding Window Attention (SWA) Layer (40 layers / 80% of model)")
    print(f"{'Batch':<8} | {'Unfused (µs)':<14} | {'torch.compile':<14} | {'Fused CUDA (µs)':<16} | {'Speedup':<10}")
    print("-" * 80)

    swa_results = []
    for b in batch_sizes:
        qkv = torch.randn(b, qkv_dim, dtype=torch.bfloat16, device="cuda")
        pos_ids = torch.arange(b, dtype=torch.int32, device="cuda")

        t_unfused = triton.testing.do_bench(lambda: unfused_swa_op(qkv, pos_ids), warmup=25, rep=100, return_mode="median") * 1000.0
        t_comp = triton.testing.do_bench(lambda: compiled_swa(qkv, pos_ids), warmup=25, rep=100, return_mode="median") * 1000.0
        t_fused = triton.testing.do_bench(lambda: fused_swa_op(qkv, pos_ids), warmup=25, rep=100, return_mode="median") * 1000.0

        speedup = t_unfused / t_fused
        print(f"{b:<8} | {t_unfused:<14.2f} | {t_comp:<14.2f} | {t_fused:<16.2f} | {speedup:<9.2f}x")

        swa_results.append({
            "batch_size": b,
            "unfused_us": round(t_unfused, 2),
            "compiled_us": round(t_comp, 2),
            "fused_cuda_us": round(t_fused, 2),
            "speedup": f"{speedup:.2f}x",
        })

    # --- SECTION B: Full-Attention Microbenchmark ---
    print("\n" + "-" * 80)
    print("SECTION B: Full-Attention Layer (RNoPE) (10 layers / 20% of model)")
    print(f"{'Batch':<8} | {'Unfused (µs)':<14} | {'torch.compile (µs)':<20} | {'Workaround Fused (µs)':<22} | {'Workaround Speedup':<20}")
    print("-" * 80)

    full_results = []
    for b in batch_sizes:
        qkv = torch.randn(b, qkv_dim, dtype=torch.bfloat16, device="cuda")
        pos_ids = torch.arange(b, dtype=torch.int32, device="cuda")

        zeros_pos = torch.zeros_like(pos_ids)

        t_unfused = triton.testing.do_bench(lambda: unfused_full_op(qkv, pos_ids), warmup=25, rep=100, return_mode="median") * 1000.0
        t_comp = triton.testing.do_bench(lambda: compiled_full(qkv, pos_ids), warmup=25, rep=100, return_mode="median") * 1000.0
        t_workaround = triton.testing.do_bench(lambda: fused_swa_op(qkv, zeros_pos), warmup=25, rep=100, return_mode="median") * 1000.0

        workaround_speedup = t_unfused / t_workaround
        print(f"{b:<8} | {t_unfused:<14.2f} | {t_comp:<20.2f} | {t_workaround:<22.2f} | {workaround_speedup:<18.2f}x")

        full_results.append({
            "batch_size": b,
            "unfused_us": round(t_unfused, 2),
            "compiled_us": round(t_comp, 2),
            "workaround_fused_us": round(t_workaround, 2),
            "workaround_speedup": f"{workaround_speedup:.2f}x",
        })

    # --- SECTION C: Full 50-Layer Model Sequential Execution ---
    print("\n" + "-" * 80)
    print("SECTION C: Full 50-Layer Attention Pipeline Sequential Execution (40 SWA + 10 Full)")
    print("-" * 80)

    # Instantiate all 50 layers
    baseline_layers = [create_mock_attention_layer(model_config, i, fuse_qk_norm_rope=False) for i in range(50)]
    option1_layers = [
        create_mock_attention_layer(
            model_config, i,
            fuse_qk_norm_rope=(cfg.layer_types[i] != "full_attention")
        ) for i in range(50)
    ]

    qkv_dec = torch.randn(1, qkv_dim, dtype=torch.bfloat16, device="cuda")
    pos_dec = torch.zeros(1, dtype=torch.int32, device="cuda")

    def run_50_layers(layers_stack, qkv_tensor, pos_tensor):
        curr = qkv_tensor.clone()
        for lyr in layers_stack:
            if lyr.fuse_qk_norm_rope:
                curr, _, _ = lyr.apply_rope(curr, None, None, pos_tensor)
            else:
                q = curr[:, :num_heads_q * head_dim]
                k = curr[:, num_heads_q * head_dim:(num_heads_q + num_heads_kv) * head_dim]
                v = curr[:, (num_heads_q + num_heads_kv) * head_dim:]
                qo, ko, vo = lyr.apply_rope(q, k, v, pos_tensor)
                curr = torch.cat([qo, ko, vo], dim=-1)
        return curr

    t_50_baseline = triton.testing.do_bench(lambda: run_50_layers(baseline_layers, qkv_dec, pos_dec), warmup=15, rep=50, return_mode="median") * 1000.0
    t_50_option1 = triton.testing.do_bench(lambda: run_50_layers(option1_layers, qkv_dec, pos_dec), warmup=15, rep=50, return_mode="median") * 1000.0

    savings_us = t_50_baseline - t_50_option1
    speedup_50 = t_50_baseline / t_50_option1

    print(f"  Baseline 50-Layer Latency (All Unfused) : {t_50_baseline:.2f} µs ({t_50_baseline / 1000.0:.3f} ms)")
    print(f"  Option 1 50-Layer Latency (Layer-Selective): {t_50_option1:.2f} µs ({t_50_option1 / 1000.0:.3f} ms)")
    print(f"  Decode Latency Saved per Generated Token: {savings_us:.2f} µs ({savings_us / 1000.0:.3f} ms)")
    print(f"  Overall 50-Layer Speedup                : {speedup_50:.2f}x")

    # Export machine-readable results
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "02_attention_results.json"

    data = {
        "benchmark": "02_hybrid_attention_microbenchmark",
        "timestamp": datetime.now().isoformat(),
        "device": torch.cuda.get_device_name(0),
        "layers": 50,
        "swa_layers": 40,
        "full_layers": 10,
        "swa_results": swa_results,
        "full_attention_results": full_results,
        "full_50_layer_decode": {
            "baseline_all_unfused_us": round(t_50_baseline, 2),
            "option1_layer_selective_us": round(t_50_option1, 2),
            "latency_saved_us": round(savings_us, 2),
            "latency_saved_ms_per_token": round(savings_us / 1000.0, 4),
            "speedup": f"{speedup_50:.2f}x",
        }
    }
    with open(out_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved benchmark results to {out_file}")


if __name__ == "__main__":
    run_benchmarks()
