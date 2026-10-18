#!/usr/bin/env python3
"""
Test full forward pass of Kolibri 1 Causal LM using TensorRT-LLM with the
native CUDA Kolibri Router kernel.

This script tests the end-to-end model pipeline:
  Token Embedding -> SWA/Full Attention -> Sandwich RMSNorm ->
  Kolibri MoE (Native CUDA Router + 384 Experts + Shared Expert) ->
  Post-FFN Norm -> LM Head Logits.
"""

import sys
import time
import torch

try:
    import tensorrt_llm
    from tensorrt_llm._torch.configs.kolibri import Kolibri1Config
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_kolibri import Kolibri1ForCausalLM
    from tensorrt_llm._torch.moe.custom_ops.kolibri_router_custom_op import (
        is_kolibri_router_available,
    )
    from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
    from tensorrt_llm.bindings.executor import KvCacheConfig
    from tensorrt_llm.mapping import Mapping
except ImportError as e:
    print(f"[ERROR] Failed to import TensorRT-LLM components: {e}")
    print("Ensure you are running with the conda 'thesis' environment:")
    print("  /home/mkina/miniconda3/envs/thesis/bin/python benchmarks/04_test_trtllm_full_forward.py")
    sys.exit(1)


def main():
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available. This test requires an NVIDIA GPU.")
        sys.exit(1)

    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(device)
    print(f"[*] Running on device: {device_name}")
    print(f"[*] PyTorch version: {torch.__version__}")
    print(f"[*] TensorRT-LLM version: {tensorrt_llm.__version__}")

    num_experts = 384
    top_k = 6
    if not is_kolibri_router_available(num_experts, top_k):
        print(f"[ERROR] Native CUDA router is NOT available for num_experts={num_experts}, top_k={top_k}")
        sys.exit(1)
    print(f"[+] Native CUDA Kolibri Router is available and verified for E={num_experts}, K={top_k}")

    # 1. Model Configuration
    vocab_size = 1000
    hidden_size = 256
    num_layers = 2
    dtype = "bfloat16"

    print("\n[*] Initializing Kolibri1Config...")
    config = Kolibri1Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=128,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        sliding_window=64,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        dtype=dtype,
    )

    model_config = ModelConfig(pretrained_config=config)
    print("[*] Instantiating Kolibri1ForCausalLM model on GPU...")
    model = Kolibri1ForCausalLM(model_config).to(device)
    model.eval()

    # TRT-LLM leaves parameter tensors uninitialized (torch.empty) until checkpoint loading.
    # Initialize with small normal weights so synthetic forward pass produces valid numerical values.
    with torch.no_grad():
        for p in model.parameters():
            if p.is_floating_point():
                torch.nn.init.normal_(p, mean=0.0, std=0.02)

    # 2. Setup Input Tokens and KV Cache Manager
    input_ids = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int32, device=device)
    seq_len = input_ids.shape[0]
    position_ids = torch.arange(0, seq_len, device=device).unsqueeze(0)
    seq_lens = [seq_len]

    num_blocks = 10
    tokens_per_block = 64
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    kv_cache_config = KvCacheConfig(max_tokens=num_blocks * tokens_per_block)

    kv_cache_manager = KVCacheManager(
        kv_cache_config,
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=num_blocks * tokens_per_block,
        max_batch_size=1,
        mapping=mapping,
        dtype=tensorrt_llm.bindings.DataType.BF16,
    )
    kv_cache_manager.add_dummy_requests([0], seq_lens)

    # 3. Setup Attention Metadata
    metadata_cls = get_attention_backend(model_config.attn_backend).Metadata
    attn_metadata = metadata_cls(
        seq_lens=torch.tensor(seq_lens, dtype=torch.int),
        num_contexts=1,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[0],
        ),
        kv_cache_manager=kv_cache_manager,
        request_ids=[0],
        prompt_lens=seq_lens,
        max_num_requests=1,
        max_num_tokens=1024,
    )

    # 4. Execute Warmup and Timed Forward Pass
    print("[*] Executing forward pass...")
    with torch.inference_mode():
        attn_metadata.prepare()
        logits = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attn_metadata=attn_metadata,
            return_context_logits=True,
        )

    # 5. Validate Output
    expected_shape = torch.Size([seq_len, vocab_size])
    assert logits.shape == expected_shape, f"Expected shape {expected_shape}, got {logits.shape}"
    assert not torch.isnan(logits).any(), "Output logits contain NaN!"
    assert not torch.isinf(logits).any(), "Output logits contain Inf!"

    # 6. Benchmark Forward Pass Latency
    num_runs = 20
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.inference_mode():
        for _ in range(num_runs):
            _ = model(
                input_ids=input_ids,
                position_ids=position_ids,
                attn_metadata=attn_metadata,
                return_context_logits=True,
            )
    torch.cuda.synchronize()
    avg_latency_ms = ((time.perf_counter() - start_time) / num_runs) * 1000

    print("\n" + "=" * 60)
    print("  TRT-LLM KOLIBRI FULL FORWARD PASS TEST: SUCCESS")
    print("=" * 60)
    print(f"  Input Tokens:        {seq_len}")
    print(f"  Routed Experts:      {num_experts} (top-{top_k})")
    print(f"  Output Logits Shape: {list(logits.shape)}")
    print(f"  Output Logits Dtype: {logits.dtype}")
    print(f"  Average Latency:     {avg_latency_ms:.2f} ms / forward pass ({num_runs} runs)")
    print("=" * 60)


if __name__ == "__main__":
    main()
