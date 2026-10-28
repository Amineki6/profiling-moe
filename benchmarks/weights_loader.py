#!/usr/bin/env python3
"""
Weight loader and extractor for Kolibri 1 production blocks.

Downloads only shards 1 & 2 (~9.32 GB) from `Aleph-Alpha/Kolibri-1-BF16` on Hugging Face,
extracts and packs layers 0 and 1 into a compact local checkpoint file
(`checkpoints/kolibri_2blocks_real.pt`, ~6.08 GB), and returns them directly to the benchmark harness.
"""

import os
import shutil
import sys
import torch
from safetensors.torch import load_file

REPO_ID = "Aleph-Alpha/Kolibri-1-BF16"
SHARDS = [
    "model-00001-of-00032.safetensors",
    "model-00002-of-00032.safetensors",
]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CHECKPOINT_PATH = os.path.join(ROOT, "checkpoints", "kolibri_2blocks_real.pt")


def download_and_extract_2blocks(
    output_path: str = DEFAULT_CHECKPOINT_PATH,
    clean_hf_cache: bool = True,
    device: str = "cpu",
) -> list:
    """
    Downloads shards 1 & 2 from HuggingFace (if checkpoint doesn't exist),
    assembles Layers 0 and 1 into the production format, and saves locally.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        print(f"[*] Found existing Kolibri real weights at: {output_path}")
        print("[*] Loading real weights into memory...")
        layers = torch.load(output_path, map_location=device)
        print(f"[+] Successfully loaded {len(layers)} production blocks from local checkpoint.\n")
        return layers

    print("=" * 80)
    print("  DOWNLOADING & EXTRACTING KOLIBRI 1 REAL WEIGHTS (2 BLOCKS)")
    print("=" * 80)
    print(f"[*] Source Repository:   {REPO_ID}")
    print(f"[*] Target Shards:       {SHARDS} (~9.32 GB total)")
    print(f"[*] Target Checkpoint:   {output_path} (~6.08 GB)")
    print("=" * 80)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[ERROR] huggingface_hub is required. Install with: pip install huggingface_hub")
        sys.exit(1)

    shard_files = []
    for shard in SHARDS:
        print(f"\n[*] Downloading shard: {shard}...")
        file_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=shard,
            resume_download=True,
        )
        shard_files.append(file_path)
        print(f"[+] Downloaded: {file_path}")

    print("\n[*] Parsing safetensors shards across layers 0 and 1...")
    raw_weights = {}
    for shard_path in shard_files:
        print(f"  Reading {os.path.basename(shard_path)}...")
        shard_dict = load_file(shard_path, device="cpu")
        for k, v in shard_dict.items():
            if "layers.0." in k or "layers.1." in k:
                raw_weights[k] = v

    print(f"[+] Collected {len(raw_weights)} relevant weight tensors for layers 0 & 1.")

    H, NQ, NKV, HD = 2560, 48, 4, 128
    E, I_MOE, I_SH = 384, 512, 512
    DT = torch.bfloat16

    layers = []
    for l_idx in range(2):
        prefix = f"model.layers.{l_idx}."
        print(f"\n[*] Assembling Layer {l_idx} weights...")

        # 1. Normalization weights
        input_ln = raw_weights[f"{prefix}input_layernorm.weight"].to(DT)
        post_attn_ln = raw_weights[f"{prefix}post_attn_norm.weight"].to(DT)
        pre_moe_ln = raw_weights[f"{prefix}post_attention_layernorm.weight"].to(DT)
        post_ffn_norm_key = f"{prefix}post_ffn_norm.weight"
        post_ffn_ln = raw_weights[post_ffn_norm_key].to(DT)

        q_norm = raw_weights[f"{prefix}self_attn.q_norm.weight"].to(DT)
        k_norm = raw_weights[f"{prefix}self_attn.k_norm.weight"].to(DT)

        # 2. Attention Projections
        # QKV: [H, (NQ + 2*NKV)*HD]
        q_proj_t = raw_weights[f"{prefix}self_attn.q_proj.weight"].t().to(DT)  # [H, NQ*HD]
        k_proj_t = raw_weights[f"{prefix}self_attn.k_proj.weight"].t().to(DT)  # [H, NKV*HD]
        v_proj_t = raw_weights[f"{prefix}self_attn.v_proj.weight"].t().to(DT)  # [H, NKV*HD]
        w_qkv = torch.cat([q_proj_t, k_proj_t, v_proj_t], dim=-1).contiguous()

        # O_proj: [NQ*HD, H]
        w_o = raw_weights[f"{prefix}self_attn.o_proj.weight"].t().to(DT).contiguous()

        # 3. Router Gate & Bias
        # gate: [H, E]
        w_gate = raw_weights[f"{prefix}mlp.gate.weight"].t().to(DT).contiguous()
        # bias: [E] float32
        bias_key = f"{prefix}moe.router.expert_bias"
        if bias_key not in raw_weights:
            bias_key = f"{prefix}mlp.gate.expert_bias"
        gate_bias = raw_weights[bias_key].float().contiguous()

        # 4. Routed Experts
        # w13: [E, H, 2 * I_MOE], w2: [E, I_MOE, H]
        print(f"  Assembling 384 routed experts for Layer {l_idx}...")
        w13_list, w2_list = [], []
        for e in range(E):
            g_t = raw_weights[f"{prefix}mlp.experts.{e}.gate_proj.weight"].t().to(DT)  # [H, I_MOE]
            u_t = raw_weights[f"{prefix}mlp.experts.{e}.up_proj.weight"].t().to(DT)    # [H, I_MOE]
            d_t = raw_weights[f"{prefix}mlp.experts.{e}.down_proj.weight"].t().to(DT)  # [I_MOE, H]
            gu_t = torch.cat([g_t, u_t], dim=-1)                                       # [H, 2*I_MOE]
            w13_list.append(gu_t)
            w2_list.append(d_t)

        w13 = torch.stack(w13_list, dim=0).contiguous()  # [384, H, 2*I_MOE]
        w2 = torch.stack(w2_list, dim=0).contiguous()    # [384, I_MOE, H]

        # 5. Shared Expert
        sh_g_t = raw_weights[f"{prefix}mlp.shared_experts.gate_proj.weight"].t().to(DT)
        sh_u_t = raw_weights[f"{prefix}mlp.shared_experts.up_proj.weight"].t().to(DT)
        sh_d_t = raw_weights[f"{prefix}mlp.shared_experts.down_proj.weight"].t().to(DT)
        sh_w13 = torch.cat([sh_g_t, sh_u_t], dim=-1).contiguous()  # [H, 2*I_SH]
        sh_w2 = sh_d_t.contiguous()                                 # [I_SH, H]

        layer_dict = {
            "input_ln": input_ln,
            "post_attn_ln": post_attn_ln,
            "pre_moe_ln": pre_moe_ln,
            "post_ffn_ln": post_ffn_ln,
            "q_norm": q_norm,
            "k_norm": k_norm,
            "w_qkv": w_qkv,
            "w_o": w_o,
            "w_gate": w_gate,
            "gate_bias": gate_bias,
            "w13": w13,
            "w2": w2,
            "sh_w13": sh_w13,
            "sh_w2": sh_w2,
        }
        layers.append(layer_dict)

    print(f"\n[*] Saving packed 2-block checkpoint to: {output_path}...")
    torch.save(layers, output_path)
    file_size_gb = os.path.getsize(output_path) / (1024**3)
    print(f"[+] Successfully created compact checkpoint ({file_size_gb:.2f} GB).\n")

    if clean_hf_cache:
        print("[*] Reclaiming disk space by cleaning up raw HuggingFace cached shards...")
        for shard_path in shard_files:
            try:
                if os.path.exists(shard_path):
                    # Check if it's a symlink or file
                    real_path = os.path.realpath(shard_path)
                    if os.path.exists(real_path):
                        os.remove(real_path)
                    if os.path.islink(shard_path):
                        os.remove(shard_path)
            except Exception as e:
                print(f"  Warning: could not delete {shard_path}: {e}")
        print("[+] Raw shards cleaned up. Checkpoint retained.\n")

    return layers


def get_kolibri_2block_weights(
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    device: str = "cuda",
) -> list:
    """
    Returns the 2 Kolibri blocks mapped to `device`.
    Automatically downloads and extracts if the checkpoint is missing.
    """
    layers_cpu = download_and_extract_2blocks(output_path=checkpoint_path, device="cpu")
    print(f"[*] Transferring 2 blocks to {device}...")
    layers = []
    for L in layers_cpu:
        layers.append({k: v.to(device) for k, v in L.items()})
    print(f"[+] Transfer to {device} complete.\n")
    return layers


if __name__ == "__main__":
    layers = get_kolibri_2block_weights(device="cpu")
    print("[SUCCESS] Verified weights structure!")
    for idx, L in enumerate(layers):
        print(f"Layer {idx} keys: {list(L.keys())}")
        print(f"  w13 shape: {L['w13'].shape}, w2 shape: {L['w2'].shape}")
