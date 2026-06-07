"""Attention benchmark — flash-style scaled dot-product attention.

Targets the same ``Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V``
pattern as the existing ``benchmarks/kernels/attention.py`` kernel,
but exposes it through the unified benchmark protocol so the runner
can compare it across vendors and detect regressions.

Targets
-------
  Same vendor matrix as the matmul benchmark. The CPU path uses
  pure-numpy online softmax so the benchmark runs even without GPUs.
"""

from __future__ import annotations

import math
import time
from typing import Any

from benchmarks.runner import RawRun, RunContext, time_callable
from src.common.logging import get_logger

log = get_logger("nautilus.bench.attention")


DEFAULT_SHAPE: dict[str, int] = {
    "batch": 1,
    "M": 512,       # query rows
    "N": 512,       # key/value rows
    "D": 64,        # head dim
}
DEFAULT_BLOCK_M = 64
DEFAULT_BLOCK_N = 64


class AttentionBenchmark:
    """Flash-style attention benchmark.

    Uses a small (M, N, D) shape so the kernel fits in a single
    Triton program. Larger shapes are possible but the per-run
    compile time dominates; CI is uninterested in 30s+ attention
    benchmarks.
    """

    def __init__(
        self,
        benchmark_name: str = "kernels/attention",
        shape: dict[str, int] | None = None,
        block_m: int = DEFAULT_BLOCK_M,
        block_n: int = DEFAULT_BLOCK_N,
    ) -> None:
        self._name = benchmark_name
        self.shape = dict(shape) if shape is not None else dict(DEFAULT_SHAPE)
        self.block_m = block_m
        self.block_n = block_n

    # --- BenchmarkProtocol ---
    def name(self) -> str:
        return self._name

    def targets(self) -> list[str]:
        return ["nvidia/sm_90", "amd/gfx942", "intel/xe_hpg", "cpu"]

    def run(self, target: str, ctx: RunContext) -> RawRun:
        M, N, D = self.shape["M"], self.shape["N"], self.shape["D"]
        if target == "cpu":
            return self._run_cpu(M, N, D, ctx)
        if target.startswith("nvidia"):
            return self._run_triton(M, N, D, vendor="cuda", ctx=ctx)
        if target.startswith("amd"):
            return self._run_triton(M, N, D, vendor="rocm", ctx=ctx)
        if target.startswith("intel"):
            return self._run_triton(M, N, D, vendor="xpu", ctx=ctx)
        return {"status": "skipped", "error": f"unknown target {target!r}"}

    # --- Backends ---

    def _run_cpu(self, M: int, N: int, D: int, ctx: RunContext) -> RawRun:
        """Numpy reference attention."""
        import numpy as np

        rng = np.random.default_rng(0)
        Q = rng.standard_normal((M, D)).astype(np.float32)
        K = rng.standard_normal((N, D)).astype(np.float32)
        V = rng.standard_normal((N, D)).astype(np.float32)
        scale = 1.0 / math.sqrt(D)

        def attn(Q: Any, K: Any, V: Any) -> Any:
            return _numpy_attention(Q, K, V, scale)

        median, samples = time_callable(
            attn, args=(Q, K, V), trials=ctx.effective_trials(), warmup=ctx.warmup,
        )
        return {
            "status": "ok",
            "compile_time_s": 0.0,
            "exec_time_s": median,
            "exec_time_samples": samples,
            "memory_mb": _peak_rss_mb(),
            "binary_size_b": 0,
            "params": {"M": M, "N": N, "D": D, "backend": "numpy"},
            "extras": {},
        }

    def _run_triton(self, M: int, N: int, D: int, *,
                    vendor: str, ctx: RunContext) -> RawRun:
        try:
            import torch
            import triton
            import triton.language as tl
        except ImportError as exc:
            return {"status": "skipped", "error": f"torch/triton missing: {exc}"}

        device = _torch_device_for(vendor)
        if device is None:
            return {"status": "skipped", "error": f"{vendor} not available"}

        @triton.jit
        def attention_kernel(
            Q_ptr, K_ptr, V_ptr, O_ptr,
            M, N, D,
            sm_scale,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
        ):
            pid_m = tl.program_id(0)
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_d = tl.arange(0, BLOCK_D)
            offs_n = tl.arange(0, BLOCK_N)
            q_mask = (offs_m[:, None] < M) & (offs_d[None, :] < D)
            q = tl.load(
                Q_ptr + offs_m[:, None] * D + offs_d[None, :],
                mask=q_mask, other=0.0,
            )
            m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
            l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
            acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
            for start_n in range(0, N, BLOCK_N):
                cur_n = start_n + offs_n
                n_mask = (cur_n[None, :] < N) & (offs_d[:, None] < D)
                k = tl.load(
                    K_ptr + cur_n[None, :] * D + offs_d[:, None],
                    mask=n_mask, other=0.0,
                )
                qk = tl.dot(q, k) * sm_scale
                qk = tl.where(cur_n[None, :] < N, qk, -float("inf"))
                m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
                alpha = tl.exp(m_i - m_ij)
                p = tl.exp(qk - m_ij[:, None])
                l_i = l_i * alpha + tl.sum(p, axis=1)
                acc = acc * alpha[:, None]
                v_mask = (cur_n[:, None] < N) & (offs_d[None, :] < D)
                v = tl.load(
                    V_ptr + cur_n[:, None] * D + offs_d[None, :],
                    mask=v_mask, other=0.0,
                )
                acc += tl.dot(p.to(v.dtype), v)
                m_i = m_ij
            out = acc / l_i[:, None]
            o_mask = (offs_m[:, None] < M) & (offs_d[None, :] < D)
            tl.store(
                O_ptr + offs_m[:, None] * D + offs_d[None, :], out, mask=o_mask,
            )

        Q = torch.randn((M, D), device=device, dtype=torch.float16)
        K = torch.randn((N, D), device=device, dtype=torch.float16)
        V = torch.randn((N, D), device=device, dtype=torch.float16)
        Out = torch.empty((M, D), device=device, dtype=torch.float16)
        sm_scale = 1.0 / math.sqrt(D)
        grid = (triton.cdiv(M, self.block_m),)

        t0 = time.perf_counter()
        try:
            compiled = attention_kernel.warmup(
                Q, K, V, Out, M, N, D, sm_scale,
                BLOCK_M=self.block_m, BLOCK_N=self.block_n, BLOCK_D=D,
                grid=grid,
            )
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": f"compile failed: {exc}"}
        compile_time_s = time.perf_counter() - t0

        def launch() -> None:
            attention_kernel[grid](
                Q, K, V, Out, M, N, D, sm_scale,
                BLOCK_M=self.block_m, BLOCK_N=self.block_n, BLOCK_D=D,
            )

        try:
            median, samples = time_callable(
                launch, trials=ctx.effective_trials(), warmup=ctx.warmup,
            )
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": f"exec failed: {exc}"}

        binary_size_b = _try_binary_size(compiled)
        return {
            "status": "ok",
            "compile_time_s": compile_time_s,
            "exec_time_s": median,
            "exec_time_samples": samples,
            "memory_mb": _peak_rss_mb(),
            "binary_size_b": binary_size_b,
            "params": {
                "M": M, "N": N, "D": D,
                "block_m": self.block_m, "block_n": self.block_n,
                "vendor": vendor,
            },
            "extras": {"grid": list(grid)},
        }


# ---------------------------------------------------------------------------
# Helpers (duplicated from matmul_bench to keep modules self-contained
# — we don't want a benchmarks._vendor_utils import chain).
# ---------------------------------------------------------------------------


def _numpy_attention(Q: Any, K: Any, V: Any, scale: float) -> Any:
    """Reference scaled-dot-product attention in pure numpy."""
    scores = Q @ K.T
    scores = scores * scale
    # Stable softmax
    scores = scores - scores.max(axis=-1, keepdims=True)
    exp = __import__("numpy").exp(scores)
    p = exp / exp.sum(axis=-1, keepdims=True)
    return p @ V


def _torch_device_for(vendor: str) -> str | None:
    try:
        import torch
    except ImportError:
        return None
    if vendor == "cuda":
        return "cuda" if torch.cuda.is_available() else None
    if vendor == "rocm":
        if hasattr(torch, "hip") and torch.hip.is_available():
            return "cuda"
        if torch.cuda.is_available() and torch.version.hip is not None:
            return "cuda"
        return None
    if vendor == "xpu":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        return None
    return None


def _peak_rss_mb() -> float:
    try:
        import resource
        import sys
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return rss / (1024.0 * 1024.0)
        return rss / 1024.0
    except (ImportError, OSError, ValueError):
        return 0.0


def _try_binary_size(compiled: Any) -> int | None:
    for attr in ("asm", "ptx", "cubin", "hsaco", "spv", "binary", "bytes"):
        v = getattr(compiled, attr, None)
        if isinstance(v, (bytes, bytearray)):
            return len(v)
        if isinstance(v, str) and v and attr in {"ptx", "asm"}:
            return len(v.encode("utf-8"))
    return None


BENCHMARK = AttentionBenchmark()
