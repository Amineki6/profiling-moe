import torch
import sys
import os

# Ensure import of the fused kernel
sys.path.insert(0, "/home/mkina/profiling/profiling-moe/fusions/01_sandwich_norm/fused")
from fused_sandwich_norm import fused_sandwich_norm

# 128 SMs * 9 blocks/SM (1 full wave = 1152 blocks)
# 16 full SM waves = 16 * 1152 = 18432 blocks (tokens)
NUM_TOKENS = 18432
HIDDEN_DIM = 2560
EPS = 1e-5
DTYPE = torch.bfloat16
DEVICE = "cuda:0"

print(f"Allocating tensors: tokens={NUM_TOKENS}, hidden_dim={HIDDEN_DIM}, dtype={DTYPE}...")
attn_out = torch.randn(NUM_TOKENS, HIDDEN_DIM, device=DEVICE, dtype=DTYPE)
residual_in = torch.randn(NUM_TOKENS, HIDDEN_DIM, device=DEVICE, dtype=DTYPE)
gamma1 = torch.ones(HIDDEN_DIM, device=DEVICE, dtype=DTYPE)
gamma2 = torch.ones(HIDDEN_DIM, device=DEVICE, dtype=DTYPE)

# Warmup
for _ in range(3):
    fused_sandwich_norm(attn_out, residual_in, gamma1, gamma2, EPS)
torch.cuda.synchronize()

print("Triggering profiled kernel...")
# Profiled invocation
torch.cuda.nvtx.range_push("profile_target")
fused_sandwich_norm(attn_out, residual_in, gamma1, gamma2, EPS)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()

print("Done!")
