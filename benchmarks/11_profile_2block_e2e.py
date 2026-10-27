#!/usr/bin/env python3
"""
End-to-end 2-block Kolibri decoder profiling: TRT-LLM baseline kernels vs. custom fused kernels.

Mirrors `Kolibri1DecoderLayer.forward` (TensorRT-LLM/tensorrt_llm/_torch/models/modeling_kolibri.py)
at full un-shrunk production dimensions (hidden=2560, 48 Q / 4 KV heads, head_dim=128, 384 experts top-6,
moe_intermediate=512, shared_intermediate=512, ~6.08 GB VRAM across 2 full blocks).

The three fusion regions are swapped between variants:
  R1  post_attn_norm + residual add + post_attention_layernorm
        baseline: fusions/01_sandwich_norm/baseline (native TRT-LLM invokeGeneralRmsNorm x2 + at::add)
        fused:    fusions/01_sandwich_norm/fused    (fusedSandwichNormRegKernel, 1 launch, 100% in-register)
  R2  MoE prologue: router top-6 + expert dispatch map + activation expansion
        baseline: fusions/03_moe_prologue/baseline  (kolibriRoutingKernels + TRT-LLM CUB/3-step map + expand)
        fused:    fusions/03_moe_prologue/fused     (reduceTopK router + dynamic hybrid map + uint4 expand)
  R3  routed + shared add + post_ffn_norm
        baseline: fusions/02_post_moe_norm/baseline (at::add + native TRT-LLM invokeGeneralRmsNorm)
        fused:    fusions/02_post_moe_norm/fused    (fusedAddPostMoeNormRegKernel, 1 launch, 100% in-register)

All other surrounding operations (attention, gate GEMM, expert GEMMs, shared expert, finalize)
are 100% IDENTICAL PyTorch operations in both variants, ensuring that any delta in execution time
and memory traffic stems directly from the three kernel fusions.

Usage:
  # Sweep both variants across decode & prefill sequence lengths:
  python benchmarks/11_profile_2block_e2e.py

  # Target a specific GPU:
  python benchmarks/11_profile_2block_e2e.py --device cuda:0

  # Nsight Systems profiling:
  nsys profile -t cuda,nvtx,osrt --capture-range=cudaProfilerApi --capture-range-end=stop -f true \
      -o traces/2block_e2e_fused python benchmarks/11_profile_2block_e2e.py --variant fused --seq-lens 64 --nsys
"""

import argparse
import importlib.util
import os
import sys
from collections import defaultdict

# Suppress ninja/arch warning by default if not set
if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUSIONS = os.path.join(ROOT, "fusions")


def _load(name, rel):
    path = os.path.join(FUSIONS, rel)
    sys.path.insert(0, os.path.dirname(path))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path.pop(0)
    return mod


sn_fused = _load("sn_fused", "01_sandwich_norm/fused/fused_sandwich_norm.py")
sn_base = _load("sn_base", "01_sandwich_norm/baseline/trtllm_baseline.py")
pm_fused = _load("pm_fused", "02_post_moe_norm/fused/fused_post_moe_norm.py")
pm_base = _load("pm_base", "02_post_moe_norm/baseline/trtllm_baseline.py")
pr_fused = _load("pr_fused", "03_moe_prologue/fused/fused_moe_prologue.py")
pr_base = _load("pr_base", "03_moe_prologue/baseline/trtllm_baseline.py")

KERNELS = {
    "baseline": dict(
        sandwich=sn_base.trtllm_cuda_baseline,
        prologue=pr_base.trtllm_moe_prologue,
        post_moe=pm_base.trtllm_cuda_baseline,
    ),
    "fused": dict(
        sandwich=sn_fused.fused_sandwich_norm,
        prologue=pr_fused.fused_moe_prologue,
        post_moe=pm_fused.fused_add_post_moe_norm,
    ),
}

# Full un-shrunk Kolibri 1 production dimensions
H, NQ, NKV, HD = 2560, 48, 4, 128
E, TOPK, I_MOE, I_SH = 384, 6, 512, 512
EPS = 1e-5
DT = torch.bfloat16


class Timer:
    """Per-region CUDA-event timing (no host synchronization inside the forward pass)."""

    def __init__(self):
        self.enabled = False
        self.pending = []
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)

    def region(self, name):
        timer = self

        class _R:
            def __enter__(self):
                torch.cuda.nvtx.range_push(name)
                if timer.enabled:
                    self.s = torch.cuda.Event(enable_timing=True)
                    self.e = torch.cuda.Event(enable_timing=True)
                    self.s.record()

            def __exit__(self, *a):
                if timer.enabled:
                    self.e.record()
                    timer.pending.append((name, self.s, self.e))
                torch.cuda.nvtx.range_pop()

        return _R()

    def flush(self, dev):
        torch.cuda.synchronize(dev)
        for name, s, e in self.pending:
            self.totals[name] += s.elapsed_time(e) * 1e3  # microseconds
            self.counts[name] += 1
        self.pending.clear()

    def reset(self):
        self.pending.clear()
        self.totals.clear()
        self.counts.clear()


def make_layer(gen, dev):
    """Instantiate independent, unshared weights for one full Kolibri production block."""
    def w(*shape, std=0.02):
        t = torch.empty(*shape, device=dev, dtype=DT)
        t.normal_(0.0, std, generator=gen)
        return t

    def g():
        t = torch.empty(H, device=dev, dtype=DT)
        t.normal_(1.0, 0.1, generator=gen)
        return t

    return dict(
        input_ln=g(),
        post_attn_ln=g(),
        pre_moe_ln=g(),
        post_ffn_ln=g(),
        q_norm=torch.empty(HD, device=dev, dtype=DT).normal_(1.0, 0.1, generator=gen),
        k_norm=torch.empty(HD, device=dev, dtype=DT).normal_(1.0, 0.1, generator=gen),
        w_qkv=w(H, (NQ + 2 * NKV) * HD),
        w_o=w(NQ * HD, H),
        w_gate=w(H, E),
        gate_bias=torch.empty(E, device=dev, dtype=torch.float32).normal_(0.0, 0.01, generator=gen),
        w13=w(E, H, 2 * I_MOE),
        w2=w(E, I_MOE, H),
        sh_w13=w(H, 2 * I_SH),
        sh_w2=w(I_SH, H),
    )


def attention(x, L):
    """SWA / GQA Attention sublayer."""
    n = x.shape[0]
    qkv = x @ L["w_qkv"]
    q, k, v = qkv.split([NQ * HD, NKV * HD, NKV * HD], dim=-1)
    q = F.rms_norm(q.view(n, NQ, HD), (HD,), L["q_norm"], EPS).transpose(0, 1)
    k = F.rms_norm(k.view(n, NKV, HD), (HD,), L["k_norm"], EPS).transpose(0, 1)
    v = v.view(n, NKV, HD).transpose(0, 1)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    return o.transpose(0, 1).reshape(n, NQ * HD) @ L["w_o"]


def experts(permuted, offsets, L):
    """Grouped expert MLP over contiguous expert segments (100% identical in both variants)."""
    out = torch.empty_like(permuted)
    offs = offsets.tolist()
    for e in range(E):
        s, t = offs[e], offs[e + 1]
        if t > s:
            gu = permuted[s:t] @ L["w13"][e]
            gt, up = gu.chunk(2, dim=-1)
            out[s:t] = (F.silu(gt) * up) @ L["w2"][e]
    return out


def decoder_layer(x, residual, L, K, T):
    """One Kolibri Decoder Layer forward pass with swappable fusion regions."""
    n = x.shape[0]

    # Pre-attention norm
    with T.region("input_layernorm"):
        if residual is None:
            residual = x
            h = F.rms_norm(x, (H,), L["input_ln"], EPS)
        else:
            residual = x + residual
            h = F.rms_norm(residual, (H,), L["input_ln"], EPS)

    # Attention sublayer
    with T.region("attention"):
        attn_out = attention(h, L)

    # R1: Sandwich Norm (Post-attention RMSNorm + Residual Add + Pre-MoE RMSNorm)
    with T.region("R1_sandwich_norm"):
        residual, moe_in = K["sandwich"](attn_out, residual, L["post_attn_ln"], L["pre_moe_ln"], EPS)

    # Gate projection
    with T.region("gate_gemm"):
        logits = (moe_in @ L["w_gate"]).float()

    # R2: MoE Prologue (Router Top-6 + Dispatch Mapping + Vectorized Activation Expansion)
    with T.region("R2_moe_prologue"):
        permuted, weights, sel, offsets, p2u, u2p = K["prologue"](moe_in, logits, L["gate_bias"])

    # Routed experts GEMMs
    with T.region("expert_gemms"):
        expert_out = experts(permuted, offsets, L)

    # Finalize MoE reduction
    with T.region("finalize"):
        perm = u2p.view(TOPK, n).long()
        routed = (expert_out[perm].float() * weights.float().t().unsqueeze(-1)).sum(0).to(DT)

    # Shared expert
    with T.region("shared_expert"):
        gt, up = (moe_in @ L["sh_w13"]).chunk(2, dim=-1)
        shared = (F.silu(gt) * up) @ L["sh_w2"]

    # R3: Post-MoE Norm (Routed + Shared Add + Post-FFN RMSNorm)
    with T.region("R3_post_moe_norm"):
        out = K["post_moe"](routed, shared, L["post_ffn_ln"], EPS)

    return out, residual, sel


def forward(x, layers, K, T):
    residual, sels = None, []
    for L in layers:
        x, residual, sel = decoder_layer(x, residual, L, K, T)
        sels.append(sel)
    return x + residual, sels


def run(variant, n, layers, args, T, dev):
    K = KERNELS[variant]
    gen = torch.Generator(device=dev).manual_seed(1234 + n)
    x = torch.empty(n, H, device=dev, dtype=DT).normal_(0.0, 1.0, generator=gen)

    with torch.inference_mode():
        # Warmup
        T.enabled = False
        for _ in range(args.warmup):
            forward(x, layers, K, T)
        torch.cuda.synchronize(dev)
        T.reset()

        # Profiler capture (e.g. for Nsight Systems)
        if args.nsys:
            torch.cuda.cudart().cudaProfilerStart()

        # (a) Uninstrumented wall-clock e2e timing (0 CUDA-event overhead)
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for i in range(args.runs):
            torch.cuda.nvtx.range_push(f"{variant}_N{n}_iter{i}")
            out, sels = forward(x, layers, K, T)
            torch.cuda.nvtx.range_pop()
        e.record()
        torch.cuda.synchronize(dev)

        if args.nsys:
            torch.cuda.cudart().cudaProfilerStop()

        e2e_us = s.elapsed_time(e) * 1e3 / args.runs

        # (b) Per-region breakdown using fine-grained CUDA events
        T.enabled = True
        for _ in range(args.runs):
            forward(x, layers, K, T)
        T.flush(dev)
        T.enabled = False

    regions = {k: T.totals[k] / args.runs for k in T.totals}
    return out, sels, e2e_us, regions


def main():
    p = argparse.ArgumentParser(
        description="2-Block Kolibri Decoder E2E Profiling: TRT-LLM Baseline vs. Custom Fused Kernels"
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Target CUDA device (e.g. cuda:0, cuda:1)",
    )
    p.add_argument(
        "--variant",
        choices=["baseline", "fused", "both"],
        default="both",
        help="Execution variant: 'baseline', 'fused', or 'both'",
    )
    p.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[1, 16, 64, 256, 512, 1024, 4096],
        help="Sequence lengths to benchmark",
    )
    p.add_argument(
        "--layers",
        type=int,
        default=2,
        help="Number of full production decoder layers (default: 2, ~6.08 GB VRAM)",
    )
    p.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Number of benchmark iterations",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations",
    )
    p.add_argument(
        "--nsys",
        action="store_true",
        help="Bracket benchmark iterations with cudaProfilerStart/Stop for Nsight Systems",
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is required to run this benchmark.")
        sys.exit(1)

    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    print("=" * 80)
    print("  KOLIBRI 1 DECODER 2-BLOCK E2E PROFILING HARNESS")
    print("=" * 80)
    print(f"[*] Target Device:       {torch.cuda.get_device_name(dev)} ({dev})")
    print(f"[*] Total VRAM:          {torch.cuda.get_device_properties(dev).total_memory / (1024**3):.2f} GB")
    print(f"[*] Free VRAM:           {torch.cuda.mem_get_info(dev)[0] / (1024**3):.2f} GB")
    print(f"[*] Decoder Blocks:      {args.layers} full un-shrunk production blocks")
    print(f"[*] Model Dimensions:    hidden={H}, Q_heads={NQ}, KV_heads={NKV}, head_dim={HD}")
    print(f"[*] MoE Structure:       {E} routed experts (top-{TOPK}, inter={I_MOE}) + 1 shared expert (inter={I_SH})")
    print(f"[*] Benchmark Regimes:   N = {args.seq_lens} tokens")
    print("=" * 80)

    print("\n[*] Initializing compiled extension kernels...")
    for mod, fn in [
        (sn_fused, "get_fused_sandwich_norm_ops"),
        (sn_base, "get_baseline_ops"),
        (pm_fused, "get_fused_ops"),
        (pm_base, "get_baseline_ops"),
        (pr_fused, "get_fused_ops"),
        (pr_base, "get_baseline_ops"),
    ]:
        getattr(mod, fn)()
    print("[+] All fusion and baseline extension ops successfully loaded.\n")

    print(f"[*] Allocating {args.layers} independent production blocks on {dev}...")
    torch.cuda.reset_peak_memory_stats(dev)
    gen = torch.Generator(device=dev).manual_seed(0)
    layers = [make_layer(gen, dev) for _ in range(args.layers)]

    seen, wbytes = set(), 0
    for L in layers:
        for t in L.values():
            if t.data_ptr() not in seen:
                seen.add(t.data_ptr())
                wbytes += t.numel() * t.element_size()
    print(f"[+] Allocated {wbytes / (1024**3):.2f} GB of independent weights across {args.layers} blocks.\n")

    variants = ["baseline", "fused"] if args.variant == "both" else [args.variant]
    fusion_regions = ["R1_sandwich_norm", "R2_moe_prologue", "R3_post_moe_norm"]
    summary = []

    for n in args.seq_lens:
        res = {}
        for v in variants:
            T = Timer()
            res[v] = run(v, n, layers, args, T, dev)

        print(f"=== N = {n:4d} tokens ({args.layers} blocks) " + "=" * 50)
        names = list(res[variants[0]][3].keys())
        hdr = f"  {'region (sum over 2 layers)':<28}" + "".join(f"{v:>15}" for v in variants)
        if len(variants) == 2:
            hdr += f"{'speedup':>12}"
        print(hdr)

        for r in names:
            line = f"  {r:<28}" + "".join(f"{res[v][3][r]:>12.2f} us" for v in variants)
            if len(variants) == 2:
                b, f = res["baseline"][3][r], res["fused"][3][r]
                if r in fusion_regions and f > 0:
                    line += f"{b / f:>11.2f}x"
            print(line)

        fr = {v: sum(res[v][3][r] for r in fusion_regions) for v in variants}
        print("  " + "-" * 68)
        print(
            f"  {'FUSION REGIONS TOTAL (R1+R2+R3)':<28}"
            + "".join(f"{fr[v]:>12.2f} us" for v in variants)
            + (f"{fr['baseline'] / fr['fused']:>11.2f}x" if len(variants) == 2 else "")
        )
        print(
            f"  {'END-TO-END 2-BLOCK FORWARD':<28}"
            + "".join(f"{res[v][2]:>12.2f} us" for v in variants)
            + (f"{res['baseline'][2] / res['fused'][2]:>11.2f}x" if len(variants) == 2 else "")
        )

        if len(variants) == 2:
            ob, sb = res["baseline"][0], res["baseline"][1]
            of, sf = res["fused"][0], res["fused"][1]
            sel_match = all(torch.equal(a.sort(-1).values, b.sort(-1).values) for a, b in zip(sb, sf))
            diff = (ob.float() - of.float()).abs()
            denom = ob.float().abs().max().item() + 1e-8
            rel = diff.max().item() / denom
            status = "PASS" if sel_match and rel < 5e-2 else "FAIL"
            print(
                f"  parity check: [{status}] (expert selection identical={sel_match}, "
                f"max|diff|={diff.max().item():.4e}, rel={rel:.2e}, mean|diff|={diff.mean().item():.2e})"
            )
            summary.append((n, fr["baseline"], fr["fused"], res["baseline"][2], res["fused"][2], sel_match, rel))
        print()

    if summary:
        print("=" * 105)
        print("  EXECUTIVE SUMMARY: 2-BLOCK FORWARD PASS (TRT-LLM BASELINE vs. FUSED KERNELS)")
        print("=" * 105)
        print(
            f"  {'N':>6} | {'Fusion Base':>13} {'Fusion Fused':>14} {'Fusion X':>10} | "
            f"{'E2E Base':>12} {'E2E Fused':>12} {'Saved (us)':>11} {'E2E X':>8} | {'Parity':>8}"
        )
        print("  " + "-" * 101)
        for n, fb, ff, eb, ef, sm, rel in summary:
            saved = eb - ef
            parity = "PASS" if sm and rel < 5e-2 else "FAIL"
            print(
                f"  {n:>6} | {fb:>10.2f} us {ff:>11.2f} us {fb / ff:>9.2f}x | "
                f"{eb:>9.1f} us {ef:>9.1f} us {saved:>8.1f} us {eb / ef:>7.3f}x | {parity:>8}"
            )
        print("=" * 105)


if __name__ == "__main__":
    main()
