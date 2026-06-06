"""Scaled dot-product attention benchmark kernel — Triton flash-style.

Computes Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
for non-batched single-head inputs. Uses online softmax accumulation
to keep memory bounded; the full (M, N) score matrix is materialized
in registers/shared memory per program.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    M, N, D,
    sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Flash-style attention: one program per BLOCK_M query block."""
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)

    # Load Q block
    q_mask = (offs_m[:, None] < M) & (offs_d[None, :] < D)
    q = tl.load(Q_ptr + offs_m[:, None] * D + offs_d[None, :], mask=q_mask, other=0.0)

    # Online-softmax accumulators
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for start_n in range(0, N, BLOCK_N):
        cur_n = start_n + offs_n
        n_mask = (cur_n[None, :] < N) & (offs_d[:, None] < D)
        # K is laid out (N, D) — load transposed
        k = tl.load(
            K_ptr + cur_n[None, :] * D + offs_d[:, None],
            mask=n_mask,
            other=0.0,
        )
        # qk: (BLOCK_M, BLOCK_D) @ (BLOCK_D, BLOCK_N) -> (BLOCK_M, BLOCK_N)
        qk = tl.dot(q, k)
        qk = qk * sm_scale
        qk = tl.where(cur_n[None, :] < N, qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_ij)
        p = tl.exp(qk - m_ij[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_mask = (cur_n[:, None] < N) & (offs_d[None, :] < D)
        v = tl.load(
            V_ptr + cur_n[:, None] * D + offs_d[None, :],
            mask=v_mask,
            other=0.0,
        )
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    out = acc / l_i[:, None]
    o_mask = (offs_m[:, None] < M) & (offs_d[None, :] < D)
    tl.store(O_ptr + offs_m[:, None] * D + offs_d[None, :], out, mask=o_mask)


def reference_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Torch reference: exact scaled dot-product attention."""
    d_k = q.shape[-1]
    return torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(d_k), dim=-1) @ v
