"""Layer-norm benchmark — row-wise LayerNorm with affine parameters.

Computes ``y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta`` per
row. One Triton program handles a single row so the kernel scales
linearly with the number of rows.

Targets
-------
  Same vendor matrix as the matmul benchmark. The CPU path is
  pure-numpy so the suite always produces a result.
"""

from __future__ import annotations

import time
from typing import Any

from benchmarks.runner import RawRun, RunContext, time_callable
from src.common.logging import get_logger

log = get_logger("nautilus.bench.layer_norm")


# Default: a transformer-FFN-shaped (rows, hidden) tensor.
DEFAULT_SHAPE: dict[str, int] = {
    "rows": 1024,
    "cols": 4096,
}
DEFAULT_EPS = 1e-5
DEFAULT_BLOCK_SIZE = 4096


class LayerNormBenchmark:
    """Row-wise LayerNorm benchmark.

    BLOCK_SIZE must be >= cols; the kernel does a single tl.arange()
    per program. If you change cols you must change BLOCK_SIZE.
    """

    def __init__(
        self,
        benchmark_name: str = "kernels/layer_norm",
        shape: dict[str, int] | None = None,
        eps: float = DEFAULT_EPS,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ) -> None:
        self._name = benchmark_name
        self.shape = dict(shape) if shape is not None else dict(DEFAULT_SHAPE)
        self.eps = eps
        # Clamp block_size up to cols.
        self.block_size = max(block_size, self.shape["cols"])

    # --- BenchmarkProtocol ---
    def name(self) -> str:
        return self._name

    def targets(self) -> list[str]:
        return ["nvidia/sm_90", "amd/gfx942", "intel/xe_hpg", "cpu"]

    def run(self, target: str, ctx: RunContext) -> RawRun:
        rows, cols = self.shape["rows"], self.shape["cols"]
        if target == "cpu":
            return self._run_cpu(rows, cols, ctx)
        if target.startswith("nvidia"):
            return self._run_triton(rows, cols, vendor="cuda", ctx=ctx)
        if target.startswith("amd"):
            return self._run_triton(rows, cols, vendor="rocm", ctx=ctx)
        if target.startswith("intel"):
            return self._run_triton(rows, cols, vendor="xpu", ctx=ctx)
        return {"status": "skipped", "error": f"unknown target {target!r}"}

    # --- Backends ---

    def _run_cpu(self, rows: int, cols: int, ctx: RunContext) -> RawRun:
        import numpy as np

        rng = np.random.default_rng(0)
        x = rng.standard_normal((rows, cols)).astype(np.float32)
        gamma = rng.standard_normal((cols,)).astype(np.float32)
        beta = rng.standard_normal((cols,)).astype(np.float32)

        def ln(x: Any, gamma: Any, beta: Any) -> Any:
            return _numpy_layer_norm(x, gamma, beta, self.eps)

        median, samples = time_callable(
            ln, args=(x, gamma, beta), trials=ctx.effective_trials(), warmup=ctx.warmup,
        )
        # Bandwidth estimate: read x, gamma, beta; write y.
        bytes_moved = (rows * cols * 4) * 3 + (rows * cols * 4)
        return {
            "status": "ok",
            "compile_time_s": 0.0,
            "exec_time_s": median,
            "exec_time_samples": samples,
            "memory_mb": _peak_rss_mb(),
            "binary_size_b": 0,
            "params": {
                "rows": rows, "cols": cols, "eps": self.eps,
                "backend": "numpy",
            },
            "extras": {
                "gbs": bytes_moved / median / 1e9 if median > 0 else 0.0,
            },
        }

    def _run_triton(self, rows: int, cols: int, *,
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
        def layer_norm_kernel(
            X_ptr, W_ptr, B_ptr, Y_ptr,
            n_cols,
            eps,
            BLOCK_SIZE: tl.constexpr,
        ):
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

        x = torch.randn((rows, cols), device=device, dtype=torch.float16)
        gamma = torch.randn((cols,), device=device, dtype=torch.float16)
        beta = torch.randn((cols,), device=device, dtype=torch.float16)
        y = torch.empty_like(x)
        grid = (rows,)

        t0 = time.perf_counter()
        try:
            compiled = layer_norm_kernel.warmup(
                x, gamma, beta, y, cols, self.eps,
                BLOCK_SIZE=self.block_size, grid=grid,
            )
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": f"compile failed: {exc}"}
        compile_time_s = time.perf_counter() - t0

        def launch() -> None:
            layer_norm_kernel[grid](
                x, gamma, beta, y, cols, self.eps,
                BLOCK_SIZE=self.block_size,
            )

        try:
            median, samples = time_callable(
                launch, trials=ctx.effective_trials(), warmup=ctx.warmup,
            )
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": f"exec failed: {exc}"}

        binary_size_b = _try_binary_size(compiled)
        bytes_moved = (rows * cols * 2) * 3 + (rows * cols * 2)
        return {
            "status": "ok",
            "compile_time_s": compile_time_s,
            "exec_time_s": median,
            "exec_time_samples": samples,
            "memory_mb": _peak_rss_mb(),
            "binary_size_b": binary_size_b,
            "params": {
                "rows": rows, "cols": cols, "eps": self.eps,
                "block_size": self.block_size, "vendor": vendor,
            },
            "extras": {
                "gbs": bytes_moved / median / 1e9 if median > 0 else 0.0,
            },
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _numpy_layer_norm(x: Any, gamma: Any, beta: Any, eps: float) -> Any:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / __import__("numpy").sqrt(var + eps) * gamma + beta


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


BENCHMARK = LayerNormBenchmark()
