"""Prefix-scan benchmark kernel — Triton row-wise cumulative sum.

Two-pass approach: pass 1 computes the total of each row block and
writes a per-block offset; pass 2 adds the running prefix to each
element. This avoids the complexity of a full Hillis-Steele
work-efficient scan inside a single program.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _scan_block_sums(
    X_ptr, S_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Per-row block sums, written back as one fp32 value per block."""
    row = tl.program_id(0)
    pid = tl.program_id(1)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    x = tl.load(X_ptr + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.sum(x, axis=0)
    tl.store(S_ptr + row * tl.num_programs(1) + pid, s)


@triton.jit
def _scan_write_offsets(
    S_ptr, O_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    N_BLOCKS: tl.constexpr,
):
    """Single-block exclusive scan over per-block sums (small N_BLOCKS)."""
    pid = tl.program_id(0)
    if pid != 0:
        return
    offs = tl.arange(0, N_BLOCKS)
    mask = offs < tl.num_programs(1)
    s = tl.load(S_ptr + offs, mask=mask, other=0.0)
    # Inclusive -> exclusive by shifting right
    exclusive = tl.cumsum(s, axis=0) - s
    tl.store(O_ptr + offs, exclusive, mask=mask)


@triton.jit
def _scan_apply(
    X_ptr, O_ptr, Y_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Add the per-block offset, then write the block's prefix sum."""
    row = tl.program_id(0)
    pid = tl.program_id(1)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    base = tl.load(O_ptr + row * tl.num_programs(1) + pid).to(tl.float32)
    x = tl.load(X_ptr + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
    # Inclusive prefix sum within the block
    local = tl.cumsum(x, axis=0)
    y = local + base
    tl.store(Y_ptr + row * n_cols + offs, y, mask=mask)


def reference_scan(x: torch.Tensor) -> torch.Tensor:
    """Torch reference: inclusive cumulative sum along the last dim."""
    return torch.cumsum(x, dim=-1)
