# Kolibri MoE Kernels Architecture

This directory organizes the various router implementations for Kolibri 1 MoE (384 experts, top-6):

```
kernels/
├── triton/
│   ├── __init__.py
│   └── fused_kolibri_router.py     # Single-pass SRAM/register fused Triton router
├── cuda/
│   ├── customMoeRoutingKernels.cu         # Variant A: Pristine baseline from TensorRT-LLM (gate_forward)
│   ├── customMoeRoutingKernelsVariantB.cu # Variant B: Modified kernel with score register caching (gate_forward_vB)
│   ├── router_bindings.cpp         # PyTorch C++ bindings exposing forward_vA & forward_vB
│   ├── kolibri_cuda_router.py      # PyTorch JIT compilation loader
│   ├── customMoeRoutingKernels.h   # TRT-LLM kernel header
│   ├── moeTopKFuncs.cuh            # Warp-level top-k bitonic reduction functions
│   └── moeGateOp.cpp               # TRT-LLM PyTorch op reference
├── fused_kolibri_router.py         # Backward-compatibility re-export shim
├── kolibri_cuda_router.py          # Backward-compatibility re-export shim
└── __init__.py                     # Package export
```

### Usage

```python
# Triton kernel:
from kernels import fused_kolibri_router
# or: from kernels.triton import fused_kolibri_router

# CUDA kernels (JIT compiled on import):
from kernels.cuda import kolibri_cuda_ops
# kolibri_cuda_ops.forward_vA(logits, bias, k=6) -> Original TRT-LLM baseline
# kolibri_cuda_ops.forward_vB(logits, bias, k=6) -> Edited register-cached kernel
```
