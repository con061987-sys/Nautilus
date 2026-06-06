"""GELU benchmark kernel — Triton elementwise GELU (exact erf form).

Computes GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2))).
Operates on a flat 1-D view of the input. Tanh-approximation is
deliberately not used so the kernel exercises the `erf` intrinsic.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(X_ptr, Y_ptr, n_elems, BLOCK_SIZE: tl.constexpr):
    """One program per BLOCK_SIZE chunk of the flat input."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elems
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    y = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    tl.store(Y_ptr + offs, y, mask=mask)


def reference_gelu(x: torch.Tensor) -> torch.Tensor:
    """Torch reference: exact GELU (approximate='none')."""
    return torch.nn.functional.gelu(x, approximate="none")
