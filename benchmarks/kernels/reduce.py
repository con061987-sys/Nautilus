"""Reduction benchmark kernel — Triton row-wise sum/max.

Reduces along the last axis of a 2-D tensor. Mode (sum vs max)
is selected at compile time via a constexpr flag.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def reduce_kernel(
    X_ptr, Y_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    MODE: tl.constexpr,
):
    """One program per row. MODE=0 -> sum, MODE=1 -> max."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    x = tl.load(X_ptr + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
    if MODE == 0:
        y = tl.sum(x, axis=0)
    else:
        y = tl.max(x, axis=0)
    tl.store(Y_ptr + row, y)


def reference_reduce(x: torch.Tensor, mode: str = "sum") -> torch.Tensor:
    """Torch reference: sum or max along the last dim."""
    if mode == "sum":
        return x.sum(dim=-1)
    if mode == "max":
        return x.max(dim=-1).values
    raise ValueError(f"Unknown mode: {mode!r}; expected 'sum' or 'max'")
