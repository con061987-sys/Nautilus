"""Softmax benchmark kernel — row-wise Triton softmax.

Computes a numerically-stable softmax along the last axis of a
2-D tensor. The torch reference uses `torch.softmax` for an exact
floating-point match within tolerance.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    X_ptr, Y_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """One Triton program per row. Mask-safe for non-pow2 widths."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    x = tl.load(X_ptr + row * n_cols + offs, mask=mask, other=-float("inf"))
    # Triton provides a fused numerically-stable row-wise softmax
    y = tl.softmax(x)
    tl.store(Y_ptr + row * n_cols + offs, y, mask=mask)


def reference_softmax(x: torch.Tensor) -> torch.Tensor:
    """Torch reference: standard softmax along the last dim."""
    return torch.softmax(x, dim=-1)
