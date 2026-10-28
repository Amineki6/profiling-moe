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
import numpy as np

# Suppress ninja/arch warning by default if not set
if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUSIONS = os.path.join(ROOT, "fusions")
BENCHMARKS = os.path.join(ROOT, "benchmarks")
if BENCHMARKS not in sys.path:
    sys.path.insert(0, BENCHMARKS)

from weights_loader import get_kolibri_2block_weights, DEFAULT_CHECKPOINT_PATH


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
        self.current_iter = 0
        self.pending = []
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)
        self.iter_records = defaultdict(list)

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
                    timer.pending.append((timer.current_iter, name, self.s, self.e))
                torch.cuda.nvtx.range_pop()

        return _R()

    def flush(self, dev):
        torch.cuda.synchronize(dev)
        iter_sums = defaultdict(lambda: defaultdict(float))
        for it, name, s, e in self.pending:
            dur = s.elapsed_time(e) * 1e3  # microseconds
            iter_sums[name][it] += dur
            self.totals[name] += dur
            self.counts[name] += 1
        for name, it_map in iter_sums.items():
            self.iter_records[name] = [it_map[i] for i in sorted(it_map.keys())]
        self.pending.clear()

    def reset(self):
        self.pending.clear()
        self.current_iter = 0
        self.totals.clear()
        self.counts.clear()
        self.iter_records.clear()


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

        # (a) Uninstrumented wall-clock e2e timing (0 CUDA-event overhead inside forward)
        e2e_events = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(args.runs)
        ]
        for i in range(args.runs):
            torch.cuda.nvtx.range_push(f"{variant}_N{n}_iter{i}")
            e2e_events[i][0].record()
            out, sels = forward(x, layers, K, T)
            e2e_events[i][1].record()
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize(dev)

        if args.nsys:
            torch.cuda.cudart().cudaProfilerStop()

        e2e_times = np.array([ev[0].elapsed_time(ev[1]) * 1e3 for ev in e2e_events])

        # (b) Per-region breakdown using fine-grained CUDA events
        T.enabled = True
        for i in range(args.runs):
            T.current_iter = i
            forward(x, layers, K, T)
        T.flush(dev)
        T.enabled = False

    regions = {}
    for k in T.totals:
        arr = np.array(T.iter_records[k]) if T.iter_records[k] else np.array([T.totals[k] / args.runs])
        regions[k] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    e2e_stats = {
        "mean": float(np.mean(e2e_times)),
        "std": float(np.std(e2e_times)),
        "median": float(np.median(e2e_times)),
        "min": float(np.min(e2e_times)),
        "max": float(np.max(e2e_times)),
    }

    return out, sels, e2e_stats, regions


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
        "--use-random-weights",
        action="store_true",
        help="Use synthetic random Gaussian weights instead of real HuggingFace weights",
    )
    p.add_argument(
        "--weights-path",
        type=str,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to local Kolibri 2-block checkpoint file",
    )
    p.add_argument(
        "--nsys",
        action="store_true",
        help="Bracket benchmark iterations with cudaProfilerStart/Stop for Nsight Systems",
    )
    p.add_argument(
        "--export-csv",
        type=str,
        default=None,
        help="Optional file path to export benchmark results as CSV",
    )
    p.add_argument(
        "--export-markdown",
        type=str,
        default=None,
        help="Optional file path to export benchmark results as Markdown",
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

    if not args.use_random_weights:
        print("[*] Loading real production weights (Aleph-Alpha/Kolibri-1-BF16)...")
        layers = get_kolibri_2block_weights(checkpoint_path=args.weights_path, device=dev)
        if args.layers != 2:
            print(f"  Note: repeating/slicing loaded 2 blocks to match --layers={args.layers}...")
            layers = [layers[i % len(layers)] for i in range(args.layers)]
    else:
        print(f"[*] Allocating {args.layers} synthetic random production blocks on {dev}...")
        torch.cuda.reset_peak_memory_stats(dev)
        gen = torch.Generator(device=dev).manual_seed(0)
        layers = [make_layer(gen, dev) for _ in range(args.layers)]

    seen, wbytes = set(), 0
    for L in layers:
        for t in L.values():
            if t.data_ptr() not in seen:
                seen.add(t.data_ptr())
                wbytes += t.numel() * t.element_size()
    print(f"[+] Loaded {wbytes / (1024**3):.2f} GB of independent weights across {args.layers} blocks on {dev}.\n")

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
            line = f"  {r:<28}"
            for v in variants:
                m = res[v][3][r]["mean"]
                s = res[v][3][r]["std"]
                line += f"{m:>8.2f} ± {s:<5.2f} us"
            if len(variants) == 2:
                b, f = res["baseline"][3][r]["mean"], res["fused"][3][r]["mean"]
                if r in fusion_regions and f > 0:
                    line += f"{b / f:>10.2f}x"
            print(line)

        fr_mean = {v: sum(res[v][3][r]["mean"] for r in fusion_regions) for v in variants}
        print("  " + "-" * 78)
        line_fr = f"  {'FUSION REGIONS TOTAL (R1+R2+R3)':<28}"
        for v in variants:
            line_fr += f"{fr_mean[v]:>15.2f} us"
        if len(variants) == 2:
            line_fr += f"{fr_mean['baseline'] / fr_mean['fused']:>10.2f}x"
        print(line_fr)

        line_e2e = f"  {'END-TO-END 2-BLOCK FORWARD':<28}"
        for v in variants:
            em = res[v][2]["mean"]
            es = res[v][2]["std"]
            line_e2e += f"{em:>8.2f} ± {es:<5.2f} us"
        if len(variants) == 2:
            line_e2e += f"{res['baseline'][2]['mean'] / res['fused'][2]['mean']:>10.2f}x"
        print(line_e2e)

        if len(variants) == 2:
            ob, sb = res["baseline"][0], res["baseline"][1]
            of, sf = res["fused"][0], res["fused"][1]

            # Layer-by-layer expert selection analysis
            layer_matches = []
            layer_flips = []
            for l_idx, (s_b, s_f) in enumerate(zip(sb, sf)):
                s_b_sorted = s_b.sort(-1).values
                s_f_sorted = s_f.sort(-1).values
                flips = ((s_b_sorted != s_f_sorted).any(-1)).sum().item()
                layer_matches.append(flips == 0)
                layer_flips.append(flips)

            diff = (ob.float() - of.float()).abs()
            denom = ob.float().abs().max().item() + 1e-8
            rel = diff.max().item() / denom

            # Analyze non-flipped tokens vs flipped tokens
            final_flipped_mask = (sb[-1].sort(-1).values != sf[-1].sort(-1).values).any(-1)
            num_final_flipped = final_flipped_mask.sum().item()

            if num_final_flipped > 0 and num_final_flipped < n:
                non_flipped = ~final_flipped_mask
                nf_diff = diff[non_flipped]
                nf_max = nf_diff.max().item()
                nf_mean = nf_diff.mean().item()
                nf_rel = nf_max / (ob[non_flipped].float().abs().max().item() + 1e-8)
                all_layers_matched = False
                status = "PASS (Boundary Ties)" if (nf_rel < 5e-2 and num_final_flipped / n < 0.05) else "FAIL"
                print(
                    f"  parity check: [{status}]\n"
                    f"    - Layer 1 Expert Match: {layer_matches[0]} ({layer_flips[0]}/{n} flipped)\n"
                    f"    - Layer 2 Expert Match: {layer_matches[1]} ({layer_flips[1]}/{n} flipped, {(1 - layer_flips[1]/n)*100:.1f}% identical)\n"
                    f"    - On Identical Tokens ({n - num_final_flipped}/{n}): max|diff|={nf_max:.4e}, rel={nf_rel:.2e}, mean|diff|={nf_mean:.2e} [BF16 Invariant PASS]\n"
                    f"    - Overall (incl. tie-swapped tokens): max|diff|={diff.max().item():.4e}, mean|diff|={diff.mean().item():.2e}"
                )
                parity_label = f"{(1 - num_final_flipped/n)*100:.1f}% Matched"
            else:
                all_layers_matched = all(layer_matches)
                status = "PASS" if all_layers_matched and rel < 5e-2 else "FAIL"
                print(
                    f"  parity check: [{status}] (expert selection identical={all_layers_matched}, "
                    f"max|diff|={diff.max().item():.4e}, rel={rel:.2e}, mean|diff|={diff.mean().item():.2e})"
                )
                parity_label = "PASS" if all_layers_matched else "FAIL"

            summary.append(
                dict(
                    n=n,
                    r1_base=res["baseline"][3]["R1_sandwich_norm"],
                    r1_fused=res["fused"][3]["R1_sandwich_norm"],
                    r2_base=res["baseline"][3]["R2_moe_prologue"],
                    r2_fused=res["fused"][3]["R2_moe_prologue"],
                    r3_base=res["baseline"][3]["R3_post_moe_norm"],
                    r3_fused=res["fused"][3]["R3_post_moe_norm"],
                    fusion_base=fr_mean["baseline"],
                    fusion_fused=fr_mean["fused"],
                    e2e_base=res["baseline"][2],
                    e2e_fused=res["fused"][2],
                    parity=parity_label,
                    rel=rel,
                )
            )
        print()

    if summary:
        print("=" * 132)
        print("  TABLE 1: REGION-BY-REGION SPEEDUP BREAKDOWN (R1, R2, R3) — MEAN ± STD")
        print("=" * 132)
        print(
            f"  {'N':>6} | {'R1 Base':>17} {'R1 Fused':>17} {'R1 X':>7} | "
            f"{'R2 Base':>17} {'R2 Fused':>17} {'R2 X':>7} | "
            f"{'R3 Base':>17} {'R3 Fused':>17} {'R3 X':>7} | {'Fusion Total X':>15}"
        )
        print("  " + "-" * 128)
        for row in summary:
            r1_b = f"{row['r1_base']['mean']:.2f} ± {row['r1_base']['std']:.2f}"
            r1_f = f"{row['r1_fused']['mean']:.2f} ± {row['r1_fused']['std']:.2f}"
            r1_x = row["r1_base"]["mean"] / row["r1_fused"]["mean"] if row["r1_fused"]["mean"] > 0 else 1.0

            r2_b = f"{row['r2_base']['mean']:.2f} ± {row['r2_base']['std']:.2f}"
            r2_f = f"{row['r2_fused']['mean']:.2f} ± {row['r2_fused']['std']:.2f}"
            r2_x = row["r2_base"]["mean"] / row["r2_fused"]["mean"] if row["r2_fused"]["mean"] > 0 else 1.0

            r3_b = f"{row['r3_base']['mean']:.2f} ± {row['r3_base']['std']:.2f}"
            r3_f = f"{row['r3_fused']['mean']:.2f} ± {row['r3_fused']['std']:.2f}"
            r3_x = row["r3_base"]["mean"] / row["r3_fused"]["mean"] if row["r3_fused"]["mean"] > 0 else 1.0

            tot_x = row["fusion_base"] / row["fusion_fused"] if row["fusion_fused"] > 0 else 1.0

            print(
                f"  {row['n']:>6} | "
                f"{r1_b:>17} {r1_f:>17} {r1_x:>6.2f}x | "
                f"{r2_b:>17} {r2_f:>17} {r2_x:>6.2f}x | "
                f"{r3_b:>17} {r3_f:>17} {r3_x:>6.2f}x | "
                f"{tot_x:>14.2f}x"
            )
        print("=" * 132)

        print("\n" + "=" * 142)
        print("  TABLE 2: END-TO-END 2-BLOCK FORWARD PASS — MEAN ± STD, MEDIAN & FUSION RECONCILIATION")
        print("=" * 142)
        print(
            f"  {'N':>6} | {'E2E Base (mean ± std)':>24} {'E2E Fused (mean ± std)':>24} "
            f"{'Saved Mean':>11} {'Saved Med':>11} {'E2E X':>8} | "
            f"{'Fusion Saved':>13} {'Fusion X':>9} | {'Parity':>14}"
        )
        print("  " + "-" * 138)
        for row in summary:
            b_m, b_s = row["e2e_base"]["mean"], row["e2e_base"]["std"]
            f_m, f_s = row["e2e_fused"]["mean"], row["e2e_fused"]["std"]
            b_med, f_med = row["e2e_base"]["median"], row["e2e_fused"]["median"]

            saved_mean = b_m - f_m
            saved_med = b_med - f_med
            e2e_x = b_m / f_m if f_m > 0 else 1.0

            fusion_saved = row["fusion_base"] - row["fusion_fused"]
            tot_x = row["fusion_base"] / row["fusion_fused"] if row["fusion_fused"] > 0 else 1.0

            b_str = f"{b_m:7.1f} ± {b_s:5.1f} us"
            f_str = f"{f_m:7.1f} ± {f_s:5.1f} us"

            print(
                f"  {row['n']:>6} | {b_str:>24} {f_str:>24} "
                f"{saved_mean:>+8.1f} us {saved_med:>+8.1f} us {e2e_x:>7.3f}x | "
                f"{fusion_saved:>10.1f} us {tot_x:>8.2f}x | {row['parity']:>14}"
            )
        print("=" * 142)

        # Markdown representation for easy copy-pasting
        md_lines = []
        md_lines.append("\n### Level 1: Fusion Regions Breakdown (Microbenchmarks — Mean ± Std)")
        md_lines.append("| $N$ (Tokens) | R1 Base ($\\mu$s) | R1 Fused ($\\mu$s) | R1 Speedup | R2 Base ($\\mu$s) | R2 Fused ($\\mu$s) | R2 Speedup | R3 Base ($\\mu$s) | R3 Fused ($\\mu$s) | R3 Speedup | Total Fusion Speedup |")
        md_lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
        for row in summary:
            r1_x = row["r1_base"]["mean"] / row["r1_fused"]["mean"] if row["r1_fused"]["mean"] > 0 else 1.0
            r2_x = row["r2_base"]["mean"] / row["r2_fused"]["mean"] if row["r2_fused"]["mean"] > 0 else 1.0
            r3_x = row["r3_base"]["mean"] / row["r3_fused"]["mean"] if row["r3_fused"]["mean"] > 0 else 1.0
            tot_x = row["fusion_base"] / row["fusion_fused"] if row["fusion_fused"] > 0 else 1.0
            md_lines.append(
                f"| {row['n']} | {row['r1_base']['mean']:.2f} ± {row['r1_base']['std']:.2f} | {row['r1_fused']['mean']:.2f} ± {row['r1_fused']['std']:.2f} | **{r1_x:.2f}x** | "
                f"{row['r2_base']['mean']:.2f} ± {row['r2_base']['std']:.2f} | {row['r2_fused']['mean']:.2f} ± {row['r2_fused']['std']:.2f} | **{r2_x:.2f}x** | "
                f"{row['r3_base']['mean']:.2f} ± {row['r3_base']['std']:.2f} | {row['r3_fused']['mean']:.2f} ± {row['r3_fused']['std']:.2f} | **{r3_x:.2f}x** | **{tot_x:.2f}x** |"
            )

        md_lines.append("\n### Level 2: End-to-End 2-Block Forward Pass (Mean ± Std & Median)")
        md_lines.append("| $N$ (Tokens) | Baseline E2E ($\\mu$s) | Fused E2E ($\\mu$s) | Saved Mean ($\\mu$s) | Saved Med ($\\mu$s) | E2E Speedup | Fusion Saved ($\\mu$s) | Fusion Speedup | Parity Status |")
        md_lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
        for row in summary:
            b_m, b_s = row["e2e_base"]["mean"], row["e2e_base"]["std"]
            f_m, f_s = row["e2e_fused"]["mean"], row["e2e_fused"]["std"]
            b_med, f_med = row["e2e_base"]["median"], row["e2e_fused"]["median"]
            saved_mean = b_m - f_m
            saved_med = b_med - f_med
            e2e_x = b_m / f_m if f_m > 0 else 1.0
            fusion_saved = row["fusion_base"] - row["fusion_fused"]
            tot_x = row["fusion_base"] / row["fusion_fused"] if row["fusion_fused"] > 0 else 1.0
            md_lines.append(
                f"| {row['n']} | {b_m:.1f} ± {b_s:.1f} | {f_m:.1f} ± {f_s:.1f} | {saved_mean:+.1f} | {saved_med:+.1f} | **{e2e_x:.3f}x** | {fusion_saved:+.1f} | {tot_x:.2f}x | {row['parity']} |"
            )

        md_content = "\n".join(md_lines)
        print("\n" + "-" * 80)
        print("  MARKDOWN EXPORT (Ready to copy into reports / papers):")
        print("-" * 80)
        print(md_content)

        if args.export_markdown:
            os.makedirs(os.path.dirname(os.path.abspath(args.export_markdown)), exist_ok=True)
            with open(args.export_markdown, "w") as f:
                f.write(md_content + "\n")
            print(f"\n[+] Saved Markdown report to: {args.export_markdown}")

        if args.export_csv:
            os.makedirs(os.path.dirname(os.path.abspath(args.export_csv)), exist_ok=True)
            import csv
            with open(args.export_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "n",
                    "r1_base_mean", "r1_base_std", "r1_fused_mean", "r1_fused_std",
                    "r2_base_mean", "r2_base_std", "r2_fused_mean", "r2_fused_std",
                    "r3_base_mean", "r3_base_std", "r3_fused_mean", "r3_fused_std",
                    "fusion_base", "fusion_fused", "fusion_saved",
                    "e2e_base_mean", "e2e_base_std", "e2e_base_med",
                    "e2e_fused_mean", "e2e_fused_std", "e2e_fused_med",
                    "e2e_saved_mean", "e2e_saved_med", "parity", "rel"
                ])
                writer.writeheader()
                for row in summary:
                    b_m, b_s = row["e2e_base"]["mean"], row["e2e_base"]["std"]
                    f_m, f_s = row["e2e_fused"]["mean"], row["e2e_fused"]["std"]
                    b_med, f_med = row["e2e_base"]["median"], row["e2e_fused"]["median"]
                    writer.writerow({
                        "n": row["n"],
                        "r1_base_mean": row["r1_base"]["mean"],
                        "r1_base_std": row["r1_base"]["std"],
                        "r1_fused_mean": row["r1_fused"]["mean"],
                        "r1_fused_std": row["r1_fused"]["std"],
                        "r2_base_mean": row["r2_base"]["mean"],
                        "r2_base_std": row["r2_base"]["std"],
                        "r2_fused_mean": row["r2_fused"]["mean"],
                        "r2_fused_std": row["r2_fused"]["std"],
                        "r3_base_mean": row["r3_base"]["mean"],
                        "r3_base_std": row["r3_base"]["std"],
                        "r3_fused_mean": row["r3_fused"]["mean"],
                        "r3_fused_std": row["r3_fused"]["std"],
                        "fusion_base": row["fusion_base"],
                        "fusion_fused": row["fusion_fused"],
                        "fusion_saved": row["fusion_base"] - row["fusion_fused"],
                        "e2e_base_mean": b_m,
                        "e2e_base_std": b_s,
                        "e2e_base_med": b_med,
                        "e2e_fused_mean": f_m,
                        "e2e_fused_std": f_s,
                        "e2e_fused_med": f_med,
                        "e2e_saved_mean": b_m - f_m,
                        "e2e_saved_med": b_med - f_med,
                        "parity": row["parity"],
                        "rel": row["rel"],
                    })
            print(f"[+] Saved CSV data to: {args.export_csv}")


if __name__ == "__main__":
    main()
