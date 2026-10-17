# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Experimental FP64 global L2 and Torch2.10 fused-Adam compatibility math.

The norm is a global FP64 sum followed by FP32 sqrt. It differs from DeepSpeed
partition-dependent intermediate rounding and is not a universal native-norm
implementation. The Adam formula matches measured Torch2.10 CUDA fused AdamW
endpoints for FP32 state, WD0 and the declared hyperparameters. No checkpoints,
captured tensors or reference gradients enter computation.
"""

import math
from functools import lru_cache
from typing import Any

import torch
import torch.distributed as dist


def global_norm_fp64(grads: list[torch.Tensor], *, process_group: Any) -> torch.Tensor:

    if process_group is None or not grads:
        raise ValueError("Explicit process group and nonempty owned gradients required")
    if not all(
        g.is_cuda and g.dtype == torch.float32 and g.is_contiguous() and g.device == grads[0].device for g in grads
    ):
        raise ValueError("Contiguous CUDA FP32 owned gradients required")
    squared = sum(
        (g.double().square().sum() for g in grads), torch.zeros((), device=grads[0].device, dtype=torch.float64)
    )
    dist.all_reduce(squared, op=dist.ReduceOp.SUM, group=process_group)
    return squared.float().sqrt()


@lru_cache(maxsize=1)
def _kernel() -> Any:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def adam(P, G, M, V, OP, OM, OV, C, N: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = i < N
        p = tl.load(P + i, valid, other=0)
        g = tl.load(G + i, valid, other=0)
        m = tl.load(M + i, valid, other=0)
        v = tl.load(V + i, valid, other=0)
        b1, b2 = tl.load(C), tl.load(C + 1)
        lr, eps, step = tl.load(C + 2), tl.load(C + 3), tl.load(C + 4)
        gd, md, vd = g.to(tl.float64), m.to(tl.float64), v.to(tl.float64)
        nm = libdevice.fma(b1, md, (1.0 - b1) * gd).to(tl.float32)
        nv = libdevice.fma(b2, vd, ((1.0 - b2) * gd) * gd).to(tl.float32)
        bc1 = (1.0 - libdevice.pow(b1, step)).to(tl.float32)
        bc2 = libdevice.sqrt(1.0 - libdevice.pow(b2, step)).to(tl.float32)
        size = libdevice.div_rn(lr, bc1.to(tl.float64)).to(tl.float32)
        denom = (libdevice.div_rn(libdevice.sqrt_rn(nv), bc2).to(tl.float64) + eps).to(tl.float32)
        np = p - libdevice.div_rn(size * nm, denom)
        tl.store(OP + i, np, valid)
        tl.store(OM + i, nm, valid)
        tl.store(OV + i, nv, valid)

    return adam


def adam_step(
    master: torch.Tensor,
    gradient: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    *,
    lr: float,
    step: int,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import triton

    values = (master, gradient, exp_avg, exp_avg_sq)
    if not all(
        t.is_cuda
        and t.dtype == torch.float32
        and t.is_contiguous()
        and t.shape == master.shape
        and t.device == master.device
        for t in values
    ):
        raise ValueError("Same-shape contiguous CUDA FP32 state required")
    if (
        type(step) is not int
        or not 1 <= step <= 2**24
        or tuple(betas) != (0.9, 0.999)
        or eps != 1e-8
        or type(lr) not in (int, float)
        or not math.isfinite(lr)
        or lr < 0
    ):
        raise ValueError("Unsupported Adam configuration")
    constants = torch.tensor([*betas, lr, eps, float(step)], device=master.device, dtype=torch.float64)
    output = tuple(torch.empty_like(master) for _ in range(3))
    _kernel()[(triton.cdiv(master.numel(), 256),)](
        *values, *output, constants, master.numel(), 256, enable_fp_fusion=False
    )
    return output
