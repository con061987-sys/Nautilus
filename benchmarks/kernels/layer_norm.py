"""Layer-norm benchmark kernel — Triton row-wise LayerNorm.

Computes y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
per row. Loads each row into a single program (BLOCK_N must be
>= the feature dimension).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """One program per row. Computes mean/var in fp32 for stability."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    x = tl.load(X_ptr + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / n_cols
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)

    y = (x - mean) * rstd * w + b
    tl.store(Y_ptr + row * n_cols + offs, y, mask=mask)


def reference_layer_norm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Torch reference: standard LayerNorm over the last dim."""
    return torch.nn.functional.layer_norm(x, gamma.shape, gamma, beta, eps)
