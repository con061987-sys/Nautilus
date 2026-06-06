"""Embedding lookup benchmark kernel — Triton gather.

For each (row, col) in the indices tensor, copies weight[indices[row, col]]
into the output. This is a memory-bound gather, not a matmul.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def embedding_kernel(
    W_ptr, I_ptr, Y_ptr,
    n_rows, n_cols, D,
    BLOCK_D: tl.constexpr,
):
    """One program per (row, col, block_d_chunk) — flat over total elements."""
    pid = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    # Decode pid -> (row, col)
    rc = pid // tl.cdiv(D, BLOCK_D)
    row = rc // n_cols
    col = rc % n_cols
    idx = tl.load(I_ptr + row * n_cols + col)
    w_off = idx * D + offs_d
    y_off = (row * n_cols + col) * D + offs_d
    val = tl.load(W_ptr + w_off, mask=mask_d, other=0.0)
    tl.store(Y_ptr + y_off, val, mask=mask_d)


def reference_embedding(
    weight: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    """Torch reference: standard embedding lookup."""
    return torch.nn.functional.embedding(indices, weight)
