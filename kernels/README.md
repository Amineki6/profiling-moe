# Kolibri MoE Kernels Architecture

This directory organizes the various router implementations for Kolibri 1 MoE (384 experts, top-6):

```
kernels/
├── triton/
│   ├── __init__.py
│   └── fused_kolibri_router.py          # Single-pass SRAM/register fused Triton router
├── cuda/
│   ├── customKolibriRoutingKernels.cu       # Standard Production Kolibri Router (kolibri_gate_forward)
│   ├── customKolibriRoutingKernels.h        # Standard Kolibri Router header
│   ├── customKolibriRoutingKernelsFloat4.cu # Vectorized 128-bit Float4 variant
│   ├── router_bindings.cpp                  # PyTorch C++ bindings exposing standard forward
│   ├── kolibri_cuda_router.py               # PyTorch JIT compilation loader
│   ├── moeTopKFuncs.cuh                     # Warp-level top-k bitonic reduction functions
│   └── moeGateOp.cpp                        # TRT-LLM PyTorch op reference
├── fused_kolibri_router.py              # Backward-compatibility re-export shim
├── kolibri_cuda_router.py               # Backward-compatibility re-export shim
└── __init__.py                          # Package export
```

### Usage

```python
# Triton kernel:
from kernels import fused_kolibri_router
# or: from kernels.triton import fused_kolibri_router

# CUDA kernels (JIT compiled on import):
from kernels.cuda import kolibri_cuda_ops
# kolibri_cuda_ops.forward(logits, bias, k=6) -> Standard production Kolibri router
# kolibri_cuda_ops.forward_vFloat4(logits, bias, k=6) -> Vectorized 128-bit load variant
```
