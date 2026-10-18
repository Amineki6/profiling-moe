#!/usr/bin/env python3
"""
Test full forward pass of Kolibri 1 using full un-shrunk production dimensions
(hidden_size=2560, moe_intermediate_size=512, 384 routed experts, 48 attention heads),
isolating a configurable number of full blocks that fit comfortably in single-GPU VRAM.

Supported modes:
  --mode model : Full Causal LM forward pass (Embedding -> SWA Attention -> Sandwich Norm -> MoE -> LM Head)
  --mode moe   : Isolated full production MoE block (Gate -> CUDA Router -> 384 Experts + Shared Expert)
"""

import argparse
import sys
import time
import torch

try:
    import tensorrt_llm
    from tensorrt_llm._torch.configs.kolibri import Kolibri1Config
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_kolibri import (
        Kolibri1ForCausalLM,
        Kolibri1MoE,
    )
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
    print("Run with the 'thesis' conda environment:")
    print("  /home/mkina/miniconda3/envs/thesis/bin/python benchmarks/05_test_trtllm_full_blocks.py")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Kolibri 1 Full Production Dimension Forward Pass Test"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of full production blocks to allocate (default: 2, ~6.4 GB VRAM)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=16,
        help="Input sequence length in tokens (default: 16)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["model", "moe"],
        default="model",
        help="'model' for full Causal LM, 'moe' for isolated MoE block (default: model)",
    )
    parser.add_argument(
        "--use-cuda-graph",
        action="store_true",
        help="Capture and replay execution using CUDA Graphs to eliminate CPU launch overhead",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of benchmark iterations (default: 10)",
    )
    return parser.parse_args()


def test_isolated_moe(args, device):
    print("\n" + "=" * 65)
    print("  MODE: ISOLATED FULL PRODUCTION MoE BLOCK (Kolibri1MoE)")
    if args.use_cuda_graph:
        print("  ACCELERATION: CUDA Graphs Enabled (Zero CPU Dispatch)")
    else:
        print("  ACCELERATION: Eager Mode (Sequential CPU Launch)")
    print("=" * 65)

    config = Kolibri1Config(
        hidden_size=2560,
        num_experts=384,
        num_experts_per_tok=6,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        dtype="bfloat16",
    )
    model_config = ModelConfig(pretrained_config=config)

    torch.cuda.reset_peak_memory_stats()
    print("[*] Instantiating 1 full production MoE block (384 experts, hidden=2560)...")
    moe = Kolibri1MoE(model_config=model_config, aux_stream_dict={}, layer_idx=0).to(device)
    moe.eval()

    with torch.no_grad():
        for p in moe.parameters():
            if p.is_floating_point():
                torch.nn.init.normal_(p, mean=0.0, std=0.02)

    total_params = sum(p.numel() for p in moe.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in moe.parameters())

    print(f"  MoE Parameters:      {total_params / 1e9:.2f} B ({total_params:,})")
    print(f"  Weights VRAM:        {param_bytes / (1024**3):.2f} GB")

    from types import SimpleNamespace
    tokens = args.seq_len
    x = torch.randn(tokens, config.hidden_size, device=device, dtype=torch.bfloat16)
    metadata = SimpleNamespace(all_rank_num_tokens=None)

    print(f"[*] Executing MoE forward pass for {tokens} tokens...")
    with torch.inference_mode():
        out = moe(x, attn_metadata=metadata)

    assert out.shape == (tokens, config.hidden_size)
    assert not torch.isnan(out).any(), "Output contains NaN!"

    if args.use_cuda_graph:
        print("[*] Warming up and capturing MoE block into CUDA Graph...")
        static_x = x.clone()
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                static_out = moe(static_x, attn_metadata=metadata)
        torch.cuda.current_stream().wait_stream(capture_stream)

        torch.cuda.cudart().cudaProfilerStart()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            static_out = moe(static_x, attn_metadata=metadata)

        print(f"[*] Replaying CUDA Graph ({args.runs} runs)...")
        start_time = time.perf_counter()
        for i in range(args.runs):
            with torch.cuda.nvtx.range(f"moe_graph_replay_iter_{i}"):
                static_x.copy_(x)
                graph.replay()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        avg_latency_ms = ((time.perf_counter() - start_time) / args.runs) * 1000
    else:
        # Warmup
        torch.cuda.synchronize()
        with torch.inference_mode():
            for _ in range(2):
                _ = moe(x, attn_metadata=metadata)
        torch.cuda.synchronize()

        # Profiled iterations with NVTX
        torch.cuda.cudart().cudaProfilerStart()
        start_time = time.perf_counter()
        with torch.inference_mode():
            for i in range(args.runs):
                with torch.cuda.nvtx.range(f"moe_block_iter_{i}"):
                    _ = moe(x, attn_metadata=metadata)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        avg_latency_ms = ((time.perf_counter() - start_time) / args.runs) * 1000

    peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
    peak_res = torch.cuda.max_memory_reserved() / (1024**3)

    print("\n" + "=" * 65)
    print("  ISOLATED MoE BLOCK TEST: SUCCESS")
    print("=" * 65)
    print(f"  Hidden Size:         2,560 (full production)")
    print(f"  Routed Experts:      384 (top-6, intermediate=512)")
    print(f"  Shared Expert:       1 (intermediate=512)")
    print(f"  Tokens Processed:    {tokens}")
    print(f"  Peak Allocated VRAM: {peak_alloc:.2f} GB")
    print(f"  Peak Reserved VRAM:  {peak_res:.2f} GB")
    print(f"  Average Latency:     {avg_latency_ms:.2f} ms / pass ({args.runs} runs)")
    print("=" * 65)


def test_full_model(args, device):
    print("\n" + "=" * 65)
    print(f"  MODE: FULL CAUSAL LM WITH {args.num_layers} FULL PRODUCTION BLOCKS")
    print("=" * 65)

    # Full production dimensions from config.json
    vocab_size = 32000
    hidden_size = 2560
    moe_intermediate = 512
    shared_intermediate = 512
    num_heads = 48
    num_kv_heads = 4
    head_dim = 128
    sliding_window = 513
    num_experts = 384
    top_k = 6

    torch.cuda.reset_peak_memory_stats()
    print(f"[*] Initializing Kolibri1Config with {args.num_layers} full layers (hidden={hidden_size})...")
    config = Kolibri1Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        sliding_window=sliding_window,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        moe_intermediate_size=moe_intermediate,
        shared_expert_intermediate_size=shared_intermediate,
        dtype="bfloat16",
    )

    model_config = ModelConfig(pretrained_config=config)
    print(f"[*] Allocating {args.num_layers} full blocks on GPU...")
    model = Kolibri1ForCausalLM(model_config).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"  Total Parameters:    {total_params / 1e9:.2f} B ({total_params:,})")
    print(f"  Weights VRAM:        {param_bytes / (1024**3):.2f} GB")

    with torch.no_grad():
        for p in model.parameters():
            if p.is_floating_point():
                torch.nn.init.normal_(p, mean=0.0, std=0.02)

    seq_len = args.seq_len
    input_ids = torch.randint(0, 1000, (seq_len,), dtype=torch.int32, device=device)
    position_ids = torch.arange(0, seq_len, device=device).unsqueeze(0)
    seq_lens = [seq_len]

    tokens_per_block = 64
    num_blocks = 10
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

    metadata_cls = get_attention_backend(model_config.attn_backend).Metadata
    attn_metadata = metadata_cls(
        seq_lens=torch.tensor(seq_lens, dtype=torch.int),
        num_contexts=1,
        kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0]),
        kv_cache_manager=kv_cache_manager,
        request_ids=[0],
        prompt_lens=seq_lens,
        max_num_requests=1,
        max_num_tokens=1024,
    )

    print(f"[*] Executing full forward pass for {seq_len} tokens...")
    with torch.inference_mode():
        attn_metadata.prepare()
        logits = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attn_metadata=attn_metadata,
            return_context_logits=True,
        )

    expected_shape = torch.Size([seq_len, vocab_size])
    assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
    assert not torch.isnan(logits).any(), "Output logits contain NaN!"

    # Warmup
    torch.cuda.synchronize()
    with torch.inference_mode():
        for _ in range(2):
            _ = model(
                input_ids=input_ids,
                position_ids=position_ids,
                attn_metadata=attn_metadata,
                return_context_logits=True,
            )
    torch.cuda.synchronize()

    # Profiled iterations with NVTX markers
    torch.cuda.cudart().cudaProfilerStart()
    start_time = time.perf_counter()
    with torch.inference_mode():
        for i in range(args.runs):
            with torch.cuda.nvtx.range(f"forward_pass_iter_{i}"):
                _ = model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attn_metadata=attn_metadata,
                    return_context_logits=True,
                )
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    avg_latency_ms = ((time.perf_counter() - start_time) / args.runs) * 1000

    peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
    peak_res = torch.cuda.max_memory_reserved() / (1024**3)

    print("\n" + "=" * 65)
    print(f"  FULL FORWARD PASS ({args.num_layers} PRODUCTION BLOCKS): SUCCESS")
    print("=" * 65)
    print(f"  Hidden Size:         {hidden_size} (100% full scale)")
    print(f"  Attention Heads:     {num_heads} query, {num_kv_heads} KV, head_dim={head_dim}")
    print(f"  Routed Experts:      {num_experts} (top-{top_k}, intermediate={moe_intermediate})")
    print(f"  Full Layers Tested:  {args.num_layers}")
    print(f"  Input Tokens:        {seq_len}")
    print(f"  Output Logits Shape: {list(logits.shape)}")
    print(f"  Peak Allocated VRAM: {peak_alloc:.2f} GB / 24 GB")
    print(f"  Peak Reserved VRAM:  {peak_res:.2f} GB / 24 GB")
    print(f"  Average Latency:     {avg_latency_ms:.2f} ms / forward pass ({args.runs} runs)")
    print("=" * 65)


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available.")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"[*] GPU: {torch.cuda.get_device_name(device)}")
    print(f"[*] Total VRAM: {torch.cuda.get_device_properties(device).total_memory / (1024**3):.1f} GB")

    if not is_kolibri_router_available(384, 6):
        print("[ERROR] Native CUDA router kernel is not available!")
        sys.exit(1)
    print("[+] Native CUDA Kolibri Router verified for 384 experts, top-6")

    if args.mode == "moe":
        test_isolated_moe(args, device)
    else:
        test_full_model(args, device)


if __name__ == "__main__":
    main()
