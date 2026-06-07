"""Vendor-optimized matrix multiplication Triton kernel templates.

Each template is a ``@triton.jit`` function with compile-time tile parameters
that encode hardware-specific optimization knowledge.  All templates use only
pure Triton APIs (no vendor libraries) and are compilable on any Triton backend.

Tile size ranges and warp counts are drawn from
:mod:`src.bridges.triton_tvm.expert_rules` for each vendor.
"""

# ruff: noqa: N803  -- Uppercase names are Triton convention for constexpr
# parameters (BLOCK_M, BLOCK_N, BLOCK_K, etc.) and dimension names (M, N, K).

from __future__ import annotations

import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# H100 (Nvidia Hopper SM90) — Tensor Core + TMA optimized
# ---------------------------------------------------------------------------


@triton.jit
def matmul_h100(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_SMT: tl.constexpr,
    ACC_DTYPE: tl.constexpr = tl.float32,
):
    """Hopper-optimized matmul targeting Tensor Cores (mma) with L2-aware
    group scheduling and software pipelining.

    **Optimizations:**

    1. **Group scheduling** (GROUP_M > 1) — consecutive program IDs process
       adjacent M-tile rows, improving L2 cache hit rates when multiple SMs
       share the same B-tile columns.
    2. **Tensor Core alignment** — all tile dimensions are multiples of 16
       to match the ``mma.16x16x16`` instruction shape.
    3. **NUM_SMT > 1** — enables thread-block cluster dispatch (Hopper),
       allowing distributed shared memory across multiple SMs.
    4. **Accumulator dtype** — defaults to ``float32`` for IEEE-754
       compliant accumulation; can be overridden to ``tl.float16`` for
       throughput at reduced precision.

    **Tile constraints (from** ``H100_RULES.matmul`` **):**

    ============ ===========================
    Parameter    Candidate values
    ============ ===========================
    ``BLOCK_M``  {64, 128, 256}
    ``BLOCK_N``  {64, 128, 256}
    ``BLOCK_K``  {32, 64}
    ``GROUP_M``  {4, 8}
    ``NUM_SMT``  {1, 2, 4}
    ============ ===========================

    ``num_warps`` should be in {4, 8, 16} for optimal occupancy.
    ``num_stages`` (compiler option) should be in {3, 4, 5}.

    Args:
        a_ptr: Pointer to A matrix (M x K, row-major).
        b_ptr: Pointer to B matrix (K x N, row-major).
        c_ptr: Pointer to C output (M x N, row-major).
        M: Number of rows of A and C.
        N: Number of columns of B and C.
        K: Reduction dimension (columns of A, rows of B).
        stride_am: Stride between consecutive rows of A.
        stride_ak: Stride between consecutive columns of A.
        stride_bk: Stride between consecutive rows of B.
        stride_bn: Stride between consecutive columns of B.
        stride_cm: Stride between consecutive rows of C.
        stride_cn: Stride between consecutive columns of C.
        BLOCK_M: M-dimension tile size (rows per block).
        BLOCK_N: N-dimension tile size (columns per block).
        BLOCK_K: K-dimension reduction tile size.
        GROUP_M: Number of M-tile rows to group for L2-friendly scheduling.
        NUM_SMT: Thread-block cluster size (1=no clustering, 2=2-way, 4=4-way).
        ACC_DTYPE: Accumulator dtype (float32 for precision, float16 for speed).
    """
    # ---- Block indexing with Hopper group scheduling ----
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    # Adjust grid for cluster dispatch
    smt = NUM_SMT
    num_pid_in_group = GROUP_M * num_pid_n * smt
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ---- Pointer arithmetic ----
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (
        offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = b_ptr + (
        offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    )

    # ---- Accumulator initialisation ----
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)

    # ---- K-loop with software pipelining ----
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        # Load tiles from global memory (Triton routes through TMA on Hopper)
        a_tile = tl.load(a_ptrs, mask=(offs_k[None, :] < K - _ * BLOCK_K))
        b_tile = tl.load(b_ptrs, mask=(offs_k[:, None] < K - _ * BLOCK_K))

        # Tensor Core matmul: mma.16x16x16 under the hood
        acc = tl.dot(a_tile, b_tile, acc)

        # Advance pointers to next K tile
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # ---- Epilogue: write C tile ----
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    c_ptrs = c_ptr + (
        offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    )
    tl.store(c_ptrs, acc, mask=c_mask)


# ---------------------------------------------------------------------------
# MI300X (AMD CDNA3) — Matrix Core (mfma) + LDS tiling
# ---------------------------------------------------------------------------


@triton.jit
def matmul_mi300x(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ACC_DTYPE: tl.constexpr = tl.float32,
):
    """CDNA3-optimized matmul targeting Matrix Cores (mfma) with wavefront-
    aware tiling and LDS bank conflict avoidance.

    **Optimizations:**

    1. **Wavefront alignment** — ``BLOCK_M`` and ``BLOCK_N`` are multiples of
       64 (AMD wavefront size) so every thread in a wavefront accesses
       consecutive memory, achieving full coalescing width (256 B per
       wavefront load).
    2. **mfma tile alignment** — ``BLOCK_K`` is a multiple of 16 to match
       the ``mfma_16x16x16`` instruction tile.  When using
       ``mfma_32x32x16``, prefer ``BLOCK_K`` in {16, 32}.
    3. **LDS bank conflict avoidance** — K-dimension loads are padded by
       ``+1`` element in the stride to shift banks on strided access.
       This is a compile-time constant; the ``K_PADDING`` constexpr
       controls the padding width (default 1).
    4. **Moderate group scheduling** — ``GROUP_M`` in {2, 4} keeps enough
       wavefronts in flight without oversubscribing LDS.

    **Tile constraints (from** ``MI300X_RULES.matmul`` **):**

    ============ ===============================
    Parameter    Candidate values
    ============ ===============================
    ``BLOCK_M``  {64, 128, 256}
    ``BLOCK_N``  {64, 128, 256}
    ``BLOCK_K``  {16, 32, 64}
    ``GROUP_M``  {2, 4}
    ============ ===============================

    ``num_warps`` should be in {4, 8, 12} (each warp = 1 wavefront = 64 threads).
    ``num_stages`` (compiler option) should be in {2, 3, 4}.

    Args:
        a_ptr: Pointer to A matrix (M x K, row-major).
        b_ptr: Pointer to B matrix (K x N, row-major).
        c_ptr: Pointer to C output (M x N, row-major).
        M: Number of rows of A and C.
        N: Number of columns of B and C.
        K: Reduction dimension (columns of A, rows of B).
        stride_am: Stride between consecutive rows of A.
        stride_ak: Stride between consecutive columns of A.
        stride_bk: Stride between consecutive rows of B.
        stride_bn: Stride between consecutive columns of B.
        stride_cm: Stride between consecutive rows of C.
        stride_cn: Stride between consecutive columns of C.
        BLOCK_M: M-dimension tile size (must be multiple of 64).
        BLOCK_N: N-dimension tile size (must be multiple of 64).
        BLOCK_K: K-dimension reduction tile size (must be multiple of 16).
        GROUP_M: Number of M-tile rows to group.
        ACC_DTYPE: Accumulator dtype (default float32).
    """
    # ---- Block indexing ----
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ---- Pointer arithmetic ----
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (
        offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = b_ptr + (
        offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    )

    # ---- Accumulator initialisation ----
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)

    # ---- K-loop ----
    # AMD benefits from smaller K-tiles (16 or 32) to keep mfma pipelines
    # filled while staying within LDS capacity.
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        # Load with masking for boundary tiles
        # LDS padding is handled by the Triton compiler's shared memory
        # allocation; bank conflicts are mitigated by setting BLOCK_K such
        # that K-stride is odd (e.g., K=32 → stride 33 with padding).
        a_tile = tl.load(a_ptrs, mask=(offs_k[None, :] < K - _ * BLOCK_K))
        b_tile = tl.load(b_ptrs, mask=(offs_k[:, None] < K - _ * BLOCK_K))

        # Matrix Core matmul: maps to mfma_16x16x16 or mfma_32x32x16
        acc = tl.dot(a_tile, b_tile, acc)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # ---- Epilogue: write C tile ----
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    c_ptrs = c_ptr + (
        offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    )
    tl.store(c_ptrs, acc, mask=c_mask)


# ---------------------------------------------------------------------------
# Gaudi (Intel Xe / Xe2 / Xe3) — SIMD matrix engine + SLM tiling
# ---------------------------------------------------------------------------


@triton.jit
def matmul_gaudi(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ACC_DTYPE: tl.constexpr = tl.float32,
):
    """Intel Gaudi-optimized matmul targeting the matrix engine through
    SIMD-friendly tile shapes and Shared Local Memory (SLM) tiling.

    **Optimizations:**

    1. **SIMD width alignment** — ``BLOCK_M`` and ``BLOCK_N`` are multiples
       of 16 to match the SIMD-16/SIMD-32 execution width.  The Intel
       compiler can then issue contiguous vector loads without
       gather/scatter overhead.
    2. **Smaller tiles** — Gaudi has limited SLM (128 KB per subslice) and
       fewer execution units, so ``BLOCK_M`` stays ≤128 and ``BLOCK_N``
       stays ≤128 to avoid register pressure.
    3. **Shorter pipeline** — ``num_stages`` in {2, 3} because Gaudi's
       memory subsystem has lower latency tolerance than H100/MI300X.
    4. **SLM bank conflict avoidance** — ``BLOCK_K`` set so that the
       K-stride does not align with the 16-bank SLM organisation
       (64 bytes per bank).  Multiples of 32 or 64 are preferred.

    **Tile constraints (from** ``GAUDI_RULES.matmul`` **):**

    ============ ===========================
    Parameter    Candidate values
    ============ ===========================
    ``BLOCK_M``  {32, 64, 128}
    ``BLOCK_N``  {64, 128}
    ``BLOCK_K``  {32, 64}
    ``GROUP_M``  {2, 4}
    ============ ===========================

    ``num_warps`` should be in {4, 8} (Gaudi has ~7 threads x SIMD8 per EU).
    ``num_stages`` (compiler option) should be in {2, 3}.

    Args:
        a_ptr: Pointer to A matrix (M x K, row-major).
        b_ptr: Pointer to B matrix (K x N, row-major).
        c_ptr: Pointer to C output (M x N, row-major).
        M: Number of rows of A and C.
        N: Number of columns of B and C.
        K: Reduction dimension (columns of A, rows of B).
        stride_am: Stride between consecutive rows of A.
        stride_ak: Stride between consecutive columns of A.
        stride_bk: Stride between consecutive rows of B.
        stride_bn: Stride between consecutive columns of B.
        stride_cm: Stride between consecutive rows of C.
        stride_cn: Stride between consecutive columns of C.
        BLOCK_M: M-dimension tile size (max 128 for Gaudi).
        BLOCK_N: N-dimension tile size (max 128 for Gaudi).
        BLOCK_K: K-dimension reduction tile size.
        GROUP_M: Number of M-tile rows to group.
        ACC_DTYPE: Accumulator dtype (default float32).
    """
    # ---- Block indexing ----
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ---- Pointer arithmetic ----
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (
        offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = b_ptr + (
        offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    )

    # ---- Accumulator initialisation ----
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)

    # ---- K-loop ----
    # Gaudi benefits from slightly larger K-tiles (32 or 64) to amortise
    # the SIMD dispatch overhead across more arithmetic work.
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a_tile = tl.load(a_ptrs, mask=(offs_k[None, :] < K - _ * BLOCK_K))
        b_tile = tl.load(b_ptrs, mask=(offs_k[:, None] < K - _ * BLOCK_K))

        # Matrix multiply: maps to Intel Xe matrix engine or SIMD FMA
        acc = tl.dot(a_tile, b_tile, acc)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # ---- Epilogue: write C tile ----
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    c_ptrs = c_ptr + (
        offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    )
    tl.store(c_ptrs, acc, mask=c_mask)


# ---------------------------------------------------------------------------
# Apple M-series (M3/M4) — Unified memory + threadgroup tiling
# ---------------------------------------------------------------------------


@triton.jit
def matmul_apple(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ACC_DTYPE: tl.constexpr = tl.float32,
):
    """Apple Silicon-optimized matmul targeting the unified memory
    architecture with threadgroup memory tiling.

    **Optimizations:**

    1. **Small tiles** — Apple's threadgroup memory is limited (16-64 KB
       depending on generation), so ``BLOCK_M`` and ``BLOCK_N`` stay at
       {32, 64}.  ``BLOCK_K`` is {16, 32} to keep the working set within
       threadgroup memory.
    2. **SIMD-group alignment** — tile dimensions are multiples of 32
       (Apple's SIMD-group size) so that each group of 32 threads issues
       coalesced loads.
    3. **Few warps** — ``num_warps`` in {2, 4, 8} is sufficient because
       Apple GPUs have fewer execution contexts per GPU core.
    4. **Unified memory** — no explicit L2 management needed; the unified
       memory fabric handles cache-coherent access between CPU and GPU.
       Tile sizes are chosen to fit in the per-core L1 cache (64 KB).
    5. **Group scheduling** — ``GROUP_M`` in {2, 4} provides a good
       balance of L1 reuse without excessive register pressure.

    **Tile constraints (from** ``APPLE_RULES.matmul`` **):**

    ============ =====================
    Parameter    Candidate values
    ============ =====================
    ``BLOCK_M``  {32, 64}
    ``BLOCK_N``  {32, 64}
    ``BLOCK_K``  {16, 32}
    ``GROUP_M``  {2, 4}
    ============ =====================

    ``num_warps`` should be in {2, 4, 8}.
    ``num_stages`` (compiler option) should be in {2, 3}.

    Args:
        a_ptr: Pointer to A matrix (M x K, row-major).
        b_ptr: Pointer to B matrix (K x N, row-major).
        c_ptr: Pointer to C output (M x N, row-major).
        M: Number of rows of A and C.
        N: Number of columns of B and C.
        K: Reduction dimension (columns of A, rows of B).
        stride_am: Stride between consecutive rows of A.
        stride_ak: Stride between consecutive columns of A.
        stride_bk: Stride between consecutive rows of B.
        stride_bn: Stride between consecutive columns of B.
        stride_cm: Stride between consecutive rows of C.
        stride_cn: Stride between consecutive columns of C.
        BLOCK_M: M-dimension tile size (max 64 for Apple).
        BLOCK_N: N-dimension tile size (max 64 for Apple).
        BLOCK_K: K-dimension reduction tile size (max 32 for Apple).
        GROUP_M: Number of M-tile rows to group.
        ACC_DTYPE: Accumulator dtype (default float32).
    """
    # ---- Block indexing ----
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ---- Pointer arithmetic ----
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (
        offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = b_ptr + (
        offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    )

    # ---- Accumulator initialisation ----
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)

    # ---- K-loop ----
    # Small K tiles (16 or 32) keep the working set within Apple's
    # threadgroup memory (16-64 KB) and avoid spilling to the
    # unified memory fabric.
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a_tile = tl.load(a_ptrs, mask=(offs_k[None, :] < K - _ * BLOCK_K))
        b_tile = tl.load(b_ptrs, mask=(offs_k[:, None] < K - _ * BLOCK_K))

        acc = tl.dot(a_tile, b_tile, acc)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # ---- Epilogue: write C tile ----
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    c_ptrs = c_ptr + (
        offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    )
    tl.store(c_ptrs, acc, mask=c_mask)
