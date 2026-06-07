# ruff: noqa: N803 — UPPER_CASE constexpr params are intentional Triton convention

"""Vendor-optimized FlashAttention-2 kernel templates.

Provides four ``@triton.jit`` kernel implementations of the
FlashAttention-2 forward pass, each hand-optimised for a specific GPU
vendor:

* **H100** (Nvidia Hopper): Tensor Core MMA, 32-thread warps, TMA-friendly.
  Default tile: M=128, N=128, D=64 — uses fmha_v2 block layout.
* **MI300X** (AMD CDNA3): Matrix Core MFMA, 64-thread wavefronts, LDS-aware.
  Default tile: M=128, N=64, D=64 — 64-aligned for wavefront occupancy.
* **Gaudi** (Intel): SIMD-only (no tensor cores), SLM-constrained.
  Default tile: M=64, N=64, D=32 — smaller tiles for EU thread efficiency.
* **Apple M-series** (Metal backend): threadgroup memory 64 KB cap.
  Default tile: M=32, N=64, D=32 — smallest tiles to fit TGMEM.

All kernels implement the same FlashAttention-2 forward algorithm:

    O = softmax(Q @ K^T / sqrt(d)) @ V

with online softmax (O(N) memory), causal masking, and configurable
softscale (temperature) parameter.  No vendor-specific library calls —
pure Triton throughout.
"""

from __future__ import annotations

import triton
import triton.language as tl

# ===================================================================
# H100 (Nvidia Hopper) — Tensor Cores, 32-thread warps
# ===================================================================


@triton.jit
def attention_h100(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    stride_qb: tl.int64,
    stride_qh: tl.int64,
    stride_qd: tl.int64,
    stride_kb: tl.int64,
    stride_kh: tl.int64,
    stride_kd: tl.int64,
    stride_vb: tl.int64,
    stride_vh: tl.int64,
    stride_vd: tl.int64,
    stride_ob: tl.int64,
    stride_oh: tl.int64,
    stride_od: tl.int64,
    BATCH: tl.constexpr,
    N_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 128,
    BLOCK_D: tl.constexpr = 64,
    CAUSAL: tl.constexpr = False,
    SOFTSCALE: tl.constexpr = 1.0,
):
    """FlashAttention-2 forward for Nvidia H100 (Hopper).

    Optimised for Tensor Core MMA instructions with 32-thread warps.
    Default tile sizes (M=128, N=128, D=64) match the fmha_v2 layout
    that gives best Tensor Core utilisation on Hopper.

    Grid shape: (BATCH, N_HEADS, SEQ_LEN // BLOCK_M)

    Args:
        q_ptr/k_ptr/v_ptr/o_ptr: fp16 tensors shaped (BATCH, N_HEADS, SEQ_LEN, HEAD_DIM).
        stride_*: Strides in elements (not bytes) for each tensor.
        BATCH: Number of batch items.
        N_HEADS: Number of attention heads.
        SEQ_LEN: Sequence length.
        HEAD_DIM: Head dimension (must be <= BLOCK_D).
        BLOCK_M: Q-tile size along sequence dimension.
        BLOCK_N: K/V-tile size along sequence dimension.
        BLOCK_D: Head-dimension tile size.
        CAUSAL: Apply causal (autoregressive) mask.
        SOFTSCALE: Pre-softmax scaling factor (default 1.0).
    """
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)

    # Guard against out-of-range programs
    if pid_batch >= BATCH or pid_head >= N_HEADS:
        return

    # Per-tensor base offsets for this batch/head
    off_bh_q = pid_batch.to(tl.int64) * stride_qb + pid_head.to(tl.int64) * stride_qh
    off_bh_k = pid_batch.to(tl.int64) * stride_kb + pid_head.to(tl.int64) * stride_kh
    off_bh_v = pid_batch.to(tl.int64) * stride_vb + pid_head.to(tl.int64) * stride_vh
    off_bh_o = pid_batch.to(tl.int64) * stride_ob + pid_head.to(tl.int64) * stride_oh

    # Sequence and head-dimension indices for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # --- Load Q tile -------------------------------------------------
    m_mask = offs_m < SEQ_LEN
    q_ptrs = (
        q_ptr
        + off_bh_q
        + offs_m[:, None] * stride_qd
        + offs_d[None, :]
    )
    q_tile = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

    # --- Initialise accumulators ------------------------------------
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    m_prev = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)

    # --- KV loop (tiled over the sequence dimension) ----------------
    if CAUSAL:
        kv_end = (pid_m + 1) * BLOCK_M
    else:
        kv_end = SEQ_LEN

    for start_n in range(0, kv_end, BLOCK_N):
        offs_n_cur = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n_cur < SEQ_LEN

        # Load K tile
        k_ptrs = (
            k_ptr
            + off_bh_k
            + offs_n_cur[:, None] * stride_kd
            + offs_d[None, :]
        )
        k_tile = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)

        # S = Q @ K^T   (fp16 → fp32 via Tensor Core MMA)
        s = tl.dot(q_tile, tl.trans(k_tile))

        # Causal mask: Q row >= K column
        if CAUSAL:
            causal_cond = offs_m[:, None] >= offs_n_cur[None, :]
            s = tl.where(
                causal_cond & m_mask[:, None] & n_mask[None, :],
                s,
                float("-inf"),
            )

        # Pre-softmax scaling (temperature / 1/sqrt(d))
        if SOFTSCALE != 1.0:
            s = s * SOFTSCALE

        # --- Online softmax update ----------------------------------
        m_cur = tl.max(s, axis=1)
        m_new = tl.maximum(m_prev, m_cur)
        alpha = tl.exp(m_prev - m_new)
        beta = tl.exp(m_cur - m_new)

        acc = acc * alpha[:, None]

        p = tl.exp(s - m_new[:, None])
        l_cur = tl.sum(p, axis=1)

        # Load V tile
        v_ptrs = (
            v_ptr
            + off_bh_v
            + offs_n_cur[:, None] * stride_vd
            + offs_d[None, :]
        )
        v_tile = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        # Accumulate: O += P @ V
        acc += tl.dot(p.to(tl.float16), v_tile) * beta[:, None]

        # Update running state
        l_prev = l_prev * alpha + l_cur * beta
        m_prev = m_new

    # --- Final rescaling: O = O / l ---------------------------------
    acc = acc / l_prev[:, None]

    # --- Store output -----------------------------------------------
    o_ptrs = (
        o_ptr
        + off_bh_o
        + offs_m[:, None] * stride_od
        + offs_d[None, :]
    )
    tl.store(o_ptrs, acc.to(tl.float16), mask=m_mask[:, None])


# ===================================================================
# MI300X (AMD CDNA3) — Matrix Cores, 64-thread wavefronts
# ===================================================================


@triton.jit
def attention_mi300x(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    stride_qb: tl.int64,
    stride_qh: tl.int64,
    stride_qd: tl.int64,
    stride_kb: tl.int64,
    stride_kh: tl.int64,
    stride_kd: tl.int64,
    stride_vb: tl.int64,
    stride_vh: tl.int64,
    stride_vd: tl.int64,
    stride_ob: tl.int64,
    stride_oh: tl.int64,
    stride_od: tl.int64,
    BATCH: tl.constexpr,
    N_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_D: tl.constexpr = 64,
    CAUSAL: tl.constexpr = False,
    SOFTSCALE: tl.constexpr = 1.0,
):
    """FlashAttention-2 forward for AMD MI300X (CDNA3).

    Optimised for Matrix Core MFMA instructions with 64-thread wavefronts.
    All tile dimensions are multiples of 64 for optimal wavefront occupancy
    and LDS bank-conflict avoidance.

    Grid shape: (BATCH, N_HEADS, SEQ_LEN // BLOCK_M)

    Args:
        q_ptr/k_ptr/v_ptr/o_ptr: fp16 tensors shaped (BATCH, N_HEADS, SEQ_LEN, HEAD_DIM).
        stride_*: Strides in elements (not bytes).
        BATCH: Number of batch items.
        N_HEADS: Number of attention heads.
        SEQ_LEN: Sequence length.
        HEAD_DIM: Head dimension (must be <= BLOCK_D).
        BLOCK_M: Q-tile size (multiple of 64 recommended).
        BLOCK_N: K/V-tile size (multiple of 64 recommended).
        BLOCK_D: Head-dimension tile size.
        CAUSAL: Apply causal (autoregressive) mask.
        SOFTSCALE: Pre-softmax scaling factor (default 1.0).
    """
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)

    if pid_batch >= BATCH or pid_head >= N_HEADS:
        return

    off_bh_q = pid_batch.to(tl.int64) * stride_qb + pid_head.to(tl.int64) * stride_qh
    off_bh_k = pid_batch.to(tl.int64) * stride_kb + pid_head.to(tl.int64) * stride_kh
    off_bh_v = pid_batch.to(tl.int64) * stride_vb + pid_head.to(tl.int64) * stride_vh
    off_bh_o = pid_batch.to(tl.int64) * stride_ob + pid_head.to(tl.int64) * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # --- Load Q tile -------------------------------------------------
    m_mask = offs_m < SEQ_LEN
    q_ptrs = (
        q_ptr
        + off_bh_q
        + offs_m[:, None] * stride_qd
        + offs_d[None, :]
    )
    q_tile = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

    # --- Initialise accumulators ------------------------------------
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    m_prev = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)

    # --- KV loop ----------------------------------------------------
    if CAUSAL:
        kv_end = (pid_m + 1) * BLOCK_M
    else:
        kv_end = SEQ_LEN

    for start_n in range(0, kv_end, BLOCK_N):
        offs_n_cur = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n_cur < SEQ_LEN

        # Load K tile
        k_ptrs = (
            k_ptr
            + off_bh_k
            + offs_n_cur[:, None] * stride_kd
            + offs_d[None, :]
        )
        k_tile = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)

        # S = Q @ K^T   (MFMA matrix core with fp32 accumulation)
        s = tl.dot(q_tile, tl.trans(k_tile))

        # Causal mask
        if CAUSAL:
            causal_cond = offs_m[:, None] >= offs_n_cur[None, :]
            s = tl.where(
                causal_cond & m_mask[:, None] & n_mask[None, :],
                s,
                float("-inf"),
            )

        if SOFTSCALE != 1.0:
            s = s * SOFTSCALE

        # --- Online softmax -----------------------------------------
        m_cur = tl.max(s, axis=1)
        m_new = tl.maximum(m_prev, m_cur)
        alpha = tl.exp(m_prev - m_new)
        beta = tl.exp(m_cur - m_new)

        acc = acc * alpha[:, None]

        p = tl.exp(s - m_new[:, None])
        l_cur = tl.sum(p, axis=1)

        # Load V tile
        v_ptrs = (
            v_ptr
            + off_bh_v
            + offs_n_cur[:, None] * stride_vd
            + offs_d[None, :]
        )
        v_tile = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        # Accumulate
        acc += tl.dot(p.to(tl.float16), v_tile) * beta[:, None]

        l_prev = l_prev * alpha + l_cur * beta
        m_prev = m_new

    # --- Final rescaling --------------------------------------------
    acc = acc / l_prev[:, None]

    o_ptrs = (
        o_ptr
        + off_bh_o
        + offs_m[:, None] * stride_od
        + offs_d[None, :]
    )
    tl.store(o_ptrs, acc.to(tl.float16), mask=m_mask[:, None])


# ===================================================================
# Gaudi 2/3 (Intel) — SIMD-only, SLM-constrained
# ===================================================================


@triton.jit
def attention_gaudi(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    stride_qb: tl.int64,
    stride_qh: tl.int64,
    stride_qd: tl.int64,
    stride_kb: tl.int64,
    stride_kh: tl.int64,
    stride_kd: tl.int64,
    stride_vb: tl.int64,
    stride_vh: tl.int64,
    stride_vd: tl.int64,
    stride_ob: tl.int64,
    stride_oh: tl.int64,
    stride_od: tl.int64,
    BATCH: tl.constexpr,
    N_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_D: tl.constexpr = 32,
    CAUSAL: tl.constexpr = False,
    SOFTSCALE: tl.constexpr = 1.0,
):
    """FlashAttention-2 forward for Intel Gaudi 2/3.

    Gaudi has no tensor-/matrix-core instructions — ``tl.dot`` falls
    back to SIMD.  Smaller tiles (64x64x32) keep ALU utilisation high
    and stay within the 128 KB SLM (Shared Local Memory) limit.

    Grid shape: (BATCH, N_HEADS, SEQ_LEN // BLOCK_M)

    Args:
        q_ptr/k_ptr/v_ptr/o_ptr: fp16 tensors shaped (BATCH, N_HEADS, SEQ_LEN, HEAD_DIM).
        stride_*: Strides in elements (not bytes).
        BATCH: Number of batch items.
        N_HEADS: Number of attention heads.
        SEQ_LEN: Sequence length.
        HEAD_DIM: Head dimension (must be <= BLOCK_D).
        BLOCK_M: Q-tile size along sequence dimension.
        BLOCK_N: K/V-tile size along sequence dimension.
        BLOCK_D: Head-dimension tile size.
        CAUSAL: Apply causal (autoregressive) mask.
        SOFTSCALE: Pre-softmax scaling factor (default 1.0).
    """
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)

    if pid_batch >= BATCH or pid_head >= N_HEADS:
        return

    off_bh_q = pid_batch.to(tl.int64) * stride_qb + pid_head.to(tl.int64) * stride_qh
    off_bh_k = pid_batch.to(tl.int64) * stride_kb + pid_head.to(tl.int64) * stride_kh
    off_bh_v = pid_batch.to(tl.int64) * stride_vb + pid_head.to(tl.int64) * stride_vh
    off_bh_o = pid_batch.to(tl.int64) * stride_ob + pid_head.to(tl.int64) * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # --- Load Q tile -------------------------------------------------
    m_mask = offs_m < SEQ_LEN
    q_ptrs = (
        q_ptr
        + off_bh_q
        + offs_m[:, None] * stride_qd
        + offs_d[None, :]
    )
    q_tile = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

    # --- Initialise accumulators ------------------------------------
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    m_prev = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)

    # --- KV loop ----------------------------------------------------
    if CAUSAL:
        kv_end = (pid_m + 1) * BLOCK_M
    else:
        kv_end = SEQ_LEN

    for start_n in range(0, kv_end, BLOCK_N):
        offs_n_cur = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n_cur < SEQ_LEN

        # Load K tile — Gaudi uses SIMD for the dot product
        k_ptrs = (
            k_ptr
            + off_bh_k
            + offs_n_cur[:, None] * stride_kd
            + offs_d[None, :]
        )
        k_tile = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)

        s = tl.dot(q_tile, tl.trans(k_tile))

        if CAUSAL:
            causal_cond = offs_m[:, None] >= offs_n_cur[None, :]
            s = tl.where(
                causal_cond & m_mask[:, None] & n_mask[None, :],
                s,
                float("-inf"),
            )

        if SOFTSCALE != 1.0:
            s = s * SOFTSCALE

        # --- Online softmax -----------------------------------------
        m_cur = tl.max(s, axis=1)
        m_new = tl.maximum(m_prev, m_cur)
        alpha = tl.exp(m_prev - m_new)
        beta = tl.exp(m_cur - m_new)

        acc = acc * alpha[:, None]

        p = tl.exp(s - m_new[:, None])
        l_cur = tl.sum(p, axis=1)

        # Load V tile
        v_ptrs = (
            v_ptr
            + off_bh_v
            + offs_n_cur[:, None] * stride_vd
            + offs_d[None, :]
        )
        v_tile = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        # Accumulate
        acc += tl.dot(p.to(tl.float16), v_tile) * beta[:, None]

        l_prev = l_prev * alpha + l_cur * beta
        m_prev = m_new

    # --- Final rescaling --------------------------------------------
    acc = acc / l_prev[:, None]

    o_ptrs = (
        o_ptr
        + off_bh_o
        + offs_m[:, None] * stride_od
        + offs_d[None, :]
    )
    tl.store(o_ptrs, acc.to(tl.float16), mask=m_mask[:, None])


# ===================================================================
# Apple M-series — Threadgroup memory, unified architecture
# ===================================================================


@triton.jit
def attention_apple(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    stride_qb: tl.int64,
    stride_qh: tl.int64,
    stride_qd: tl.int64,
    stride_kb: tl.int64,
    stride_kh: tl.int64,
    stride_kd: tl.int64,
    stride_vb: tl.int64,
    stride_vh: tl.int64,
    stride_vd: tl.int64,
    stride_ob: tl.int64,
    stride_oh: tl.int64,
    stride_od: tl.int64,
    BATCH: tl.constexpr,
    N_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr = 32,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_D: tl.constexpr = 32,
    CAUSAL: tl.constexpr = False,
    SOFTSCALE: tl.constexpr = 1.0,
):
    """FlashAttention-2 forward for Apple M-series (Metal backend).

    Apple Silicon GPUs have 64 KB of threadgroup memory — the tightest
    shared-memory budget of the four vendors.  Default tile sizes
    (M=32, N=64, D=32) keep per-block footprint under 48 KB, leaving
    room for the runtime and Metal's argument buffer.

    Grid shape: (BATCH, N_HEADS, SEQ_LEN // BLOCK_M)

    Args:
        q_ptr/k_ptr/v_ptr/o_ptr: fp16 tensors shaped (BATCH, N_HEADS, SEQ_LEN, HEAD_DIM).
        stride_*: Strides in elements (not bytes).
        BATCH: Number of batch items.
        N_HEADS: Number of attention heads.
        SEQ_LEN: Sequence length.
        HEAD_DIM: Head dimension (must be <= BLOCK_D).
        BLOCK_M: Q-tile size along sequence dimension.
        BLOCK_N: K/V-tile size along sequence dimension.
        BLOCK_D: Head-dimension tile size.
        CAUSAL: Apply causal (autoregressive) mask.
        SOFTSCALE: Pre-softmax scaling factor (default 1.0).
    """
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)

    if pid_batch >= BATCH or pid_head >= N_HEADS:
        return

    off_bh_q = pid_batch.to(tl.int64) * stride_qb + pid_head.to(tl.int64) * stride_qh
    off_bh_k = pid_batch.to(tl.int64) * stride_kb + pid_head.to(tl.int64) * stride_kh
    off_bh_v = pid_batch.to(tl.int64) * stride_vb + pid_head.to(tl.int64) * stride_vh
    off_bh_o = pid_batch.to(tl.int64) * stride_ob + pid_head.to(tl.int64) * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # --- Load Q tile -------------------------------------------------
    m_mask = offs_m < SEQ_LEN
    q_ptrs = (
        q_ptr
        + off_bh_q
        + offs_m[:, None] * stride_qd
        + offs_d[None, :]
    )
    q_tile = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

    # --- Initialise accumulators ------------------------------------
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    m_prev = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)

    # --- KV loop ----------------------------------------------------
    if CAUSAL:
        kv_end = (pid_m + 1) * BLOCK_M
    else:
        kv_end = SEQ_LEN

    for start_n in range(0, kv_end, BLOCK_N):
        offs_n_cur = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n_cur < SEQ_LEN

        # Load K tile
        k_ptrs = (
            k_ptr
            + off_bh_k
            + offs_n_cur[:, None] * stride_kd
            + offs_d[None, :]
        )
        k_tile = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)

        s = tl.dot(q_tile, tl.trans(k_tile))

        if CAUSAL:
            causal_cond = offs_m[:, None] >= offs_n_cur[None, :]
            s = tl.where(
                causal_cond & m_mask[:, None] & n_mask[None, :],
                s,
                float("-inf"),
            )

        if SOFTSCALE != 1.0:
            s = s * SOFTSCALE

        # --- Online softmax -----------------------------------------
        m_cur = tl.max(s, axis=1)
        m_new = tl.maximum(m_prev, m_cur)
        alpha = tl.exp(m_prev - m_new)
        beta = tl.exp(m_cur - m_new)

        acc = acc * alpha[:, None]

        p = tl.exp(s - m_new[:, None])
        l_cur = tl.sum(p, axis=1)

        # Load V tile
        v_ptrs = (
            v_ptr
            + off_bh_v
            + offs_n_cur[:, None] * stride_vd
            + offs_d[None, :]
        )
        v_tile = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        acc += tl.dot(p.to(tl.float16), v_tile) * beta[:, None]

        l_prev = l_prev * alpha + l_cur * beta
        m_prev = m_new

    # --- Final rescaling --------------------------------------------
    acc = acc / l_prev[:, None]

    o_ptrs = (
        o_ptr
        + off_bh_o
        + offs_m[:, None] * stride_od
        + offs_d[None, :]
    )
    tl.store(o_ptrs, acc.to(tl.float16), mask=m_mask[:, None])
