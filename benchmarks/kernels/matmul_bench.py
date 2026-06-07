"""Matmul benchmark — measures compile / exec / memory / binary size.

Targets
-------
  - "nvidia/<arch>"  : uses Triton on CUDA if available
  - "amd/<arch>"     : uses Triton on ROCm if available (else skipped)
  - "intel/<arch>"   : uses Triton on Intel GPU/XPU if available
  - "cpu"            : always-available numpy fallback (no compile)

What it measures
----------------
  - compile_time_s    : time to build the per-vendor fat-binary section
  - exec_time_s      : median wall-clock time of ``trials`` runs
  - memory_mb        : peak RSS during execution
  - binary_size_b    : size of the per-vendor binary blob
  - extras.tflops    : achieved TFLOPs (2 * M * N * K / time)
"""

from __future__ import annotations

import time
from typing import Any

from benchmarks.runner import RawRun, RunContext, time_callable
from src.common.logging import get_logger

log = get_logger("nautilus.bench.matmul")


# Default shape: square 1024^3 matmul = ~2.15 GFLOPs at fp16. Matches
# the canonical "matmul benchmark" used across the Triton tutorials.
DEFAULT_SHAPE: dict[str, int] = {"M": 1024, "N": 1024, "K": 1024}


class MatmulBenchmark:
    """A single (shape, dtype) instance of a matmul benchmark.

    Multiple instances can be created with different shapes for
    coverage, but only one is exposed as ``BENCHMARK`` by default.

    Note: not a ``@dataclass`` because the ``name()`` method would
    collide with a field named ``name``.
    """

    def __init__(
        self,
        benchmark_name: str = "kernels/matmul",
        shape: dict[str, int] | None = None,
        dtype: str = "float16",
        block_m: int = 128,
        block_n: int = 128,
        block_k: int = 32,
        num_warps: int = 8,
    ) -> None:
        self._name = benchmark_name
        self.shape = dict(shape) if shape is not None else dict(DEFAULT_SHAPE)
        self.dtype = dtype
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.num_warps = num_warps

    # --- BenchmarkProtocol ---
    def name(self) -> str:
        return self._name

    def targets(self) -> list[str]:
        # The runner filters these by CLI --target; the union of all
        # supported vendors is fine. The CPU fallback is always supported.
        return ["nvidia/sm_90", "amd/gfx942", "intel/xe_hpg", "cpu"]

    def run(self, target: str, ctx: RunContext) -> RawRun:
        M, N, K = self.shape["M"], self.shape["N"], self.shape["K"]
        if target == "cpu":
            return self._run_cpu(M, N, K, ctx)
        if target.startswith("nvidia"):
            return self._run_triton(M, N, K, vendor="cuda", ctx=ctx)
        if target.startswith("amd"):
            return self._run_triton(M, N, K, vendor="rocm", ctx=ctx)
        if target.startswith("intel"):
            return self._run_triton(M, N, K, vendor="xpu", ctx=ctx)
        return {"status": "skipped", "error": f"unknown target {target!r}"}

    # --- Backend impls ---

    def _run_cpu(self, M: int, N: int, K: int, ctx: RunContext) -> RawRun:
        """Always-available numpy baseline.

        No compile step, no GPU memory — just FLOPs as a sanity check.
        """
        import numpy as np

        rng = np.random.default_rng(0)
        A = rng.standard_normal((M, K)).astype(np.float32)
        B = rng.standard_normal((K, N)).astype(np.float32)

        def matmul(A: Any, B: Any) -> Any:
            return A @ B

        median, samples = time_callable(
            matmul, args=(A, B), trials=ctx.effective_trials(), warmup=ctx.warmup,
        )
        flops = 2.0 * M * N * K
        return {
            "status": "ok",
            "compile_time_s": 0.0,
            "exec_time_s": median,
            "exec_time_samples": samples,
            "memory_mb": _peak_rss_mb(),
            "binary_size_b": 0,
            "params": {"M": M, "N": N, "K": K, "dtype": "float32", "backend": "numpy"},
            "extras": {"gflops": flops / median / 1e9 if median > 0 else 0.0},
        }

    def _run_triton(self, M: int, N: int, K: int, *,
                    vendor: str, ctx: RunContext) -> RawRun:
        """Triton-based matmul; switches torch device per vendor."""
        try:
            import torch
            import triton
            import triton.language as tl
        except ImportError as exc:
            return {
                "status": "skipped",
                "error": f"torch/triton not installed: {exc}",
            }

        device = _torch_device_for(vendor)
        if device is None:
            return {
                "status": "skipped",
                "error": f"{vendor} device not available on this host",
            }
        # dtype comes from the benchmark instance.
        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                       "float32": torch.float32}[self.dtype]

        @triton.jit
        def matmul_kernel(
            A_ptr, B_ptr, C_ptr,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        ):
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)
            a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in range(0, K, BLOCK_K):
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                acc += tl.dot(a, b)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
            c = acc.to(tl.float16)
            c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            tl.store(c_ptrs, c)

        A = torch.randn((M, K), device=device, dtype=torch_dtype)
        B = torch.randn((K, N), device=device, dtype=torch_dtype)
        C = torch.empty((M, N), device=device, dtype=torch_dtype)

        grid = (triton.cdiv(M, self.block_m), triton.cdiv(N, self.block_n))

        # --- compile step (timed) ---
        t0 = time.perf_counter()
        try:
            compiled = matmul_kernel.warmup(
                A, B, C, M, N, K,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                C.stride(0), C.stride(1),
                BLOCK_M=self.block_m, BLOCK_N=self.block_n, BLOCK_K=self.block_k,
                num_warps=self.num_warps, grid=grid,
            )
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": f"compile failed: {exc}"}
        compile_time_s = time.perf_counter() - t0

        # --- exec step ---
        def launch() -> None:
            matmul_kernel[grid](
                A, B, C, M, N, K,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                C.stride(0), C.stride(1),
                BLOCK_M=self.block_m, BLOCK_N=self.block_n, BLOCK_K=self.block_k,
                num_warps=self.num_warps,
            )

        try:
            median, samples = time_callable(
                launch, trials=ctx.effective_trials(), warmup=ctx.warmup,
            )
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": f"exec failed: {exc}"}

        # --- binary size: best-effort, depends on vendor ---
        binary_size_b = _try_binary_size(compiled)
        flops = 2.0 * M * N * K
        return {
            "status": "ok",
            "compile_time_s": compile_time_s,
            "exec_time_s": median,
            "exec_time_samples": samples,
            "memory_mb": _peak_rss_mb(),
            "binary_size_b": binary_size_b,
            "params": {
                "M": M, "N": N, "K": K, "dtype": self.dtype,
                "block_m": self.block_m, "block_n": self.block_n,
                "block_k": self.block_k, "num_warps": self.num_warps,
                "vendor": vendor,
            },
            "extras": {
                "tflops": flops / median / 1e12 if median > 0 else 0.0,
                "grid": list(grid),
            },
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _torch_device_for(vendor: str) -> str | None:
    """Return the torch device string for a vendor, or None if unavailable."""
    try:
        import torch
    except ImportError:
        return None
    if vendor == "cuda":
        return "cuda" if torch.cuda.is_available() else None
    if vendor == "rocm":
        # ROCm looks like CUDA to torch.
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
    """Best-effort peak RSS in MiB. Returns 0.0 on any failure."""
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if _is_macos():
            return rss / (1024.0 * 1024.0)
        return rss / 1024.0
    except (ImportError, OSError, ValueError):
        return 0.0


def _is_macos() -> bool:
    import sys
    return sys.platform == "darwin"


def _try_binary_size(compiled: Any) -> int | None:
    """Return the size of the compiled artifact in bytes, if accessible.

    Triton's compiled objects differ across versions. We poke at common
    attribute names and return ``len(...)`` for the first one that
    yields bytes. Returns None if nothing works — this is best-effort
    and a None MUST NOT crash the benchmark.
    """
    for attr in ("asm", "ptx", "cubin", "hsaco", "spv", "binary", "bytes"):
        v = getattr(compiled, attr, None)
        if isinstance(v, (bytes, bytearray)):
            return len(v)
        if isinstance(v, str) and v and attr in {"ptx", "asm"}:
            return len(v.encode("utf-8"))
    return None


# Default export for the runner's ``discover()``.
BENCHMARK = MatmulBenchmark()
