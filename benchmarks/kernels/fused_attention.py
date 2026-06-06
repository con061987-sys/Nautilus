"""Fused QKV-projection + attention benchmark kernel.

Combines the linear projection Q=X@Wq, K=X@Wk, V=X@Wv with the
attention computation in a single kernel. The projections are
written as Triton matmuls sharing the same input load, then the
flash-style attention kernel from `attention.py` is reused via
an inlined definition to keep the benchmark self-contained.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def fused_attention_kernel(
    X_ptr, WQ_ptr, WK_ptr, WV_ptr, O_ptr,
    M, D_in, D, N,
    sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Fused: project to Q, K, V, then flash-attention.

    This implementation computes Q, K, V on the fly per program
    and uses online softmax for the attention accumulation. The
    K and V projections are unrolled across the N dimension.
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)

    # Q = X @ WQ  (D_in -> D)
    offs_k = tl.arange(0, BLOCK_K)
    q_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for kk in range(0, D_in, BLOCK_K):
        x = tl.load(
            X_ptr + offs_m[:, None] * D_in + (kk + offs_k)[None, :],
            mask=(offs_m[:, None] < M) & ((kk + offs_k)[None, :] < D_in),
            other=0.0,
        )
        w = tl.load(
            WQ_ptr + (kk + offs_k)[:, None] * D + offs_d[None, :],
            mask=((kk + offs_k)[:, None] < D_in) & (offs_d[None, :] < D),
            other=0.0,
        )
        q_acc += tl.dot(x, w)
    q = q_acc

    # Online softmax accumulators
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    o_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for start_n in range(0, N, BLOCK_N):
        cur_n = start_n + offs_n
        # K block: gather K = X_n @ WK  -> (BLOCK_N, BLOCK_D)
        k_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
        for kk in range(0, D_in, BLOCK_K):
            x_n = tl.load(
                X_ptr + cur_n[:, None] * D_in + (kk + offs_k)[None, :],
                mask=(cur_n[:, None] < N) & ((kk + offs_k)[None, :] < D_in),
                other=0.0,
            )
            w = tl.load(
                WK_ptr + (kk + offs_k)[:, None] * D + offs_d[None, :],
                mask=((kk + offs_k)[:, None] < D_in) & (offs_d[None, :] < D),
                other=0.0,
            )
            k_acc += tl.dot(x_n, w)
        k = k_acc

        qk = tl.dot(q, tl.trans(k))
        qk = qk * sm_scale
        qk = tl.where(cur_n[None, :] < N, qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_ij)
        p = tl.exp(qk - m_ij[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        o_acc = o_acc * alpha[:, None]

        # V block: gather V = X_n @ WV
        v_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
        for kk in range(0, D_in, BLOCK_K):
            x_n = tl.load(
                X_ptr + cur_n[:, None] * D_in + (kk + offs_k)[None, :],
                mask=(cur_n[:, None] < N) & ((kk + offs_k)[None, :] < D_in),
                other=0.0,
            )
            w = tl.load(
                WV_ptr + (kk + offs_k)[:, None] * D + offs_d[None, :],
                mask=((kk + offs_k)[:, None] < D_in) & (offs_d[None, :] < D),
                other=0.0,
            )
            v_acc += tl.dot(x_n, w)
        v = v_acc

        o_acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    out = o_acc / l_i[:, None]
    o_mask = (offs_m[:, None] < M) & (offs_d[None, :] < D)
    tl.store(O_ptr + offs_m[:, None] * D + offs_d[None, :], out, mask=o_mask)


def reference_fused_attention(
    x: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
) -> torch.Tensor:
    """Torch reference: project then attend. Self-attention (N == M)."""
    q = x @ wq
    k = x @ wk
    v = x @ wv
    d_k = q.shape[-1]
    return torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(d_k), dim=-1) @ v
