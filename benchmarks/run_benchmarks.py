"""Benchmark suite driver.

Runs each kernel across the available targets and measures
baseline vs. tuned performance.

Usage:
    python benchmarks/run_benchmarks.py --kernel matmul
    python benchmarks/run_benchmarks.py --output results.json
"""

from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Any

# Make src importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common.logging import get_logger, configure_logging, span as span_context
from src.common.errors import DependencyMissingError

log = get_logger("nautilus.benchmark")


KERNELS = {
    "matmul": "Matrix multiplication (SGEMM)",
    "softmax": "Softmax activation",
    "gelu": "GELU activation",
    "reduce": "Sum reduction",
    "layer_norm": "Layer normalization",
    "embedding": "Embedding lookup",
}


def benchmark_matmul(M: int = 1024, N: int = 1024, K: int = 1024, trials: int = 5) -> dict[str, Any]:
    """Run matmul benchmark. Returns baseline vs tuned TFLOPs."""
    try:
        import torch
        import triton
        import triton.language as tl
    except ImportError as exc:
        raise DependencyMissingError(
            "torch and triton required for benchmarking; install with: "
            "pip install torch triton",
        ) from exc

    if not torch.cuda.is_available():
        return {
            "shape": {"M": M, "N": N, "K": K},
            "results": {"_note": "CUDA not available; benchmark skipped"},
        }

    @triton.jit
    def matmul_kernel(A, B, C, M, N, K,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
        b_ptrs = B + offs_k[:, None] * N + offs_n[None, :]
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * N
        c_ptrs = C + offs_m[:, None] * N + offs_n[None, :]
        tl.store(c_ptrs, acc.to(tl.float16))

    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)

    def run_with(block_m: int, block_n: int, block_k: int, num_warps: int) -> float:
        grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
        # Warmup
        for _ in range(3):
            matmul_kernel[grid](A, B, C, M, N, K, block_m, block_n, block_k, num_warps=num_warps)
        torch.cuda.synchronize()
        # Timed
        start = time.perf_counter()
        for _ in range(trials):
            matmul_kernel[grid](A, B, C, M, N, K, block_m, block_n, block_k, num_warps=num_warps)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / trials
        # TFLOPs for SGEMM: 2*M*N*K
        tflops = 2 * M * N * K / elapsed / 1e12
        return tflops

    # Baseline: default Triton config
    baseline = run_with(64, 64, 32, 4)
    # Tuned: from MetaSchedule (here a hand-tuned reasonable config)
    tuned = run_with(128, 128, 32, 8)
    return {
        "shape": {"M": M, "N": N, "K": K},
        "results": {
            "nvidia/cuda": {
                "baseline_tflops": baseline,
                "tuned_tflops": tuned,
                "speedup": tuned / baseline if baseline > 0 else 0,
            },
        },
    }


def benchmark_softmax(N: int = 1024 * 1024, trials: int = 10) -> dict[str, Any]:
    try:
        import torch
        import triton
        import triton.language as tl
    except ImportError as exc:
        raise DependencyMissingError("torch and triton required") from exc
    if not torch.cuda.is_available():
        return {"shape": {"N": N}, "results": {"_note": "CUDA not available"}}
    @triton.jit
    def softmax_kernel(X, Y, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(X + offs, mask=mask, other=-float("inf"))
        y = tl.softmax(x)
        tl.store(Y + offs, y, mask=mask)
    X = torch.randn(N, device="cuda", dtype=torch.float32)
    Y = torch.empty_like(X)
    def run(block, num_warps):
        for _ in range(3): softmax_kernel[(triton.cdiv(N, block),)](X, Y, N, block, num_warps=num_warps)
        torch.cuda.synchronize()
        s = time.perf_counter()
        for _ in range(trials): softmax_kernel[(triton.cdiv(N, block),)](X, Y, N, block, num_warps=num_warps)
        torch.cuda.synchronize()
        return N / (time.perf_counter() - s) / 1e9  # GB/s
    base = run(1024, 4)
    tuned = run(4096, 8)
    return {
        "shape": {"N": N},
        "results": {"nvidia/cuda": {"baseline_gbps": base, "tuned_gbps": tuned, "speedup": tuned/base if base > 0 else 0}},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", choices=list(KERNELS), help="Run a single kernel")
    parser.add_argument("--output", type=Path, help="Write results JSON here")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()
    configure_logging(level=args.log_level, json=True)

    runners = {
        "matmul": benchmark_matmul,
        "softmax": benchmark_softmax,
        # gelu, reduce, layer_norm, embedding use the same pattern
    }
    kernels = [args.kernel] if args.kernel else list(runners.keys())

    all_results: dict[str, Any] = {}
    for kernel in kernels:
        log.info("Benchmarking", kernel=kernel)
        try:
            with span_context("benchmark", kernel=kernel) as sp:
                result = runners[kernel](trials=args.trials)
                sp.set(completed=True)
                all_results[kernel] = result
        except DependencyMissingError as exc:
            log.warning("Benchmark skipped", kernel=kernel, error=str(exc))
            all_results[kernel] = {"skipped": str(exc)}
        except Exception as exc:
            log.error("Benchmark failed", kernel=kernel, error=str(exc))
            all_results[kernel] = {"error": str(exc)}

    if args.output:
        args.output.write_text(json.dumps(all_results, indent=2))
        log.info("Results written", path=str(args.output))
    else:
        print(json.dumps(all_results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
