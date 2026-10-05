import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
import importlib

v = importlib.import_module("02_verify_attention_norm")

cfg = v.load_kolibri_config()
pc = cfg.pretrained_config
print("rope_scaling:", getattr(pc, "rope_scaling", None))
print("partial_rotary_factor:", getattr(pc, "partial_rotary_factor", None))

lu = v.create_mock_attention_layer(cfg, 0, fuse_qk_norm_rope=False)
lf = v.create_mock_attention_layer(cfg, 0, fuse_qk_norm_rope=True)
print("unfused: rope_fusion =", lu.rope_fusion, "| rotary_emb =", type(lu.rotary_emb).__name__)
print("fused  : rope_fusion =", lf.rope_fusion, "| rotary_emb =", type(lf.rotary_emb).__name__)
print("rope params:", lf.pos_embd_params.rope.theta, lf.pos_embd_params.rope.dim,
      lf.pos_embd_params.rope.scale_type, "is_neox =", lf.pos_embd_params.is_neox)

# Hand-written reference: RMSNorm per head, then NeoX RoPE (theta=10000, no scaling)
H, KV, D = pc.num_attention_heads, pc.num_key_value_heads, pc.head_dim
b = 4
torch.manual_seed(42)
qkv = torch.randn(b, (H + 2 * KV) * D, dtype=torch.bfloat16, device="cuda")
pos = torch.arange(b, dtype=torch.int32, device="cuda")


def ref(x, nh, w):
    x = x.float().view(b, nh, D)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + pc.rms_norm_eps) * w.float()
    inv = 1.0 / (pc.rope_theta ** (torch.arange(0, D, 2, device="cuda").float() / D))
    ang = pos.float()[:, None] * inv[None]  # [b, D/2]
    cos, sin = ang.cos()[:, None], ang.sin()[:, None]
    x1, x2 = x[..., : D // 2], x[..., D // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).view(b, -1).bfloat16()


q, k = qkv[:, : H * D], qkv[:, H * D:(H + KV) * D]
rq, rk = ref(q, H, lf.q_norm.weight), ref(k, KV, lf.k_norm.weight)

fo, _, _ = lf.apply_rope(qkv.clone(), None, None, pos)
uq, uk, _ = lu.apply_rope(q.clone(), k.clone(), qkv[:, (H + KV) * D:].clone(), pos)

print("fused   vs ref  maxdiff Q/K:", (fo[:, :H * D] - rq).abs().max().item(),
      (fo[:, H * D:(H + KV) * D] - rk).abs().max().item())
print("unfused vs ref  maxdiff Q/K:", (uq - rq).abs().max().item(), (uk - rk).abs().max().item())
