"""ResNet-50 end-to-end benchmark.

Measures:
  - capture_time_s : torch.compile / FX graph capture time
  - compile_time_s : fat-binary build time
  - exec_time_s    : median forward-pass latency
  - memory_mb      : peak RSS during the forward pass
  - binary_size_b  : total bytes of the per-vendor compiled artifacts

The eager baseline is recorded in ``extras.eager_exec_time_s`` so the
result row carries a direct speedup ratio.

Targets
-------
  - "nvidia/<arch>"  : CUDA + cuDNN
  - "amd/<arch>"     : ROCm
  - "cpu"            : always-available eager torch

Vision deps (``torchvision``) are optional; if missing the benchmark
is recorded as "skipped" so the rest of the suite still runs.
"""

from __future__ import annotations

import time

from benchmarks.models import _have_module, _skip
from benchmarks.runner import RawRun, RunContext, time_callable
from src.common.logging import get_logger

log = get_logger("nautilus.bench.resnet50")


# ResNet-50 input: ImageNet 224x224 RGB, batch size 1.
DEFAULT_BATCH_SIZE = 1
DEFAULT_SHAPE = (3, 224, 224)
DEFAULT_TRIALS = 5  # fewer than kernel benchmarks; each pass is heavier


class ResNet50Benchmark:
    """ResNet-50 forward-pass benchmark.

    Wraps ``torchvision.models.resnet50(weights=None)`` so we don't
    pull weights from the network (offline CI). The model is created
    in eval mode and frozen with ``requires_grad_(False)``.
    """

    def __init__(
        self,
        benchmark_name: str = "models/resnet50",
        batch_size: int = DEFAULT_BATCH_SIZE,
        shape: tuple[int, int, int] = DEFAULT_SHAPE,
        trials: int = DEFAULT_TRIALS,
    ) -> None:
        self._name = benchmark_name
        self.batch_size = batch_size
        self.shape = shape
        self.trials_override = trials

    # --- BenchmarkProtocol ---
    def name(self) -> str:
        return self._name

    def targets(self) -> list[str]:
        return ["nvidia/sm_90", "amd/gfx942", "cpu"]

    def run(self, target: str, ctx: RunContext) -> RawRun:
        if not _have_module("torch"):
            return _skip("nvidia", ctx)
        if not _have_module("torchvision"):
            return _skip("nvidia", ctx)

        import torch
        import torchvision  # noqa: F401 — checked above

        if target.startswith("nvidia"):
            device = _torch_device_for("cuda")
            if device is None:
                return {"status": "skipped", "error": "cuda not available"}
        elif target.startswith("amd"):
            device = _torch_device_for("rocm")
            if device is None:
                return {"status": "skipped", "error": "rocm not available"}
        elif target == "cpu":
            device = "cpu"
        else:
            return {"status": "skipped", "error": f"unknown target {target!r}"}

        trials = self.trials_override or ctx.effective_trials()
        try:
            model = torchvision.models.resnet50(weights=None)
            model = model.to(device).eval()
            for p in model.parameters():
                p.requires_grad_(False)
            x = torch.randn((self.batch_size, *self.shape), device=device)
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": f"model setup failed: {exc}"}

        with torch.inference_mode():
            # 1. Eager baseline (always runnable, sets the speedup bar).
            try:
                eager_median, eager_samples = time_callable(
                    model, args=(x,),
                    trials=trials, warmup=ctx.warmup,
                )
            except Exception as exc:  # noqa: BLE001
                return {"status": "error", "error": f"eager exec failed: {exc}"}

            # 2. Compiled path: torch.compile + (optional) Nautilus
            #    bridge. This is what we want to track for regressions.
            t_capture = time.perf_counter()
            try:
                compiled = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            except Exception as exc:  # noqa: BLE001
                return {
                    "status": "ok",  # eager path still measured
                    "exec_time_s": eager_median,
                    "exec_time_samples": eager_samples,
                    "memory_mb": _peak_rss_mb(),
                    "binary_size_b": 0,
                    "compile_time_s": 0.0,
                    "extras": {
                        "eager_exec_time_s": eager_median,
                        "torch_compile_error": f"{type(exc).__name__}: {exc}",
                    },
                    "params": {"batch": self.batch_size, "device": device},
                }
            capture_time_s = time.perf_counter() - t_capture

            t_compile = time.perf_counter()
            try:
                # Warmup triggers the actual compile. We don't time the
                # user-visible warmup but we time the python-side call
                # to ``compiled``.
                compiled(x)
            except Exception as exc:  # noqa: BLE001
                log.warning("torch.compile failed", error=str(exc))
                return {
                    "status": "ok",
                    "exec_time_s": eager_median,
                    "exec_time_samples": eager_samples,
                    "memory_mb": _peak_rss_mb(),
                    "binary_size_b": 0,
                    "compile_time_s": capture_time_s,
                    "extras": {
                        "eager_exec_time_s": eager_median,
                        "torch_compile_error": f"{type(exc).__name__}: {exc}",
                    },
                    "params": {"batch": self.batch_size, "device": device},
                }
            compile_time_s = (time.perf_counter() - t_compile) + capture_time_s

            try:
                median, samples = time_callable(
                    compiled, args=(x,), trials=trials, warmup=ctx.warmup,
                )
            except Exception as exc:  # noqa: BLE001
                return {"status": "error", "error": f"compiled exec failed: {exc}"}

        speedup = (eager_median / median) if median > 0 else 0.0
        return {
            "status": "ok",
            "compile_time_s": compile_time_s,
            "exec_time_s": median,
            "exec_time_samples": samples,
            "memory_mb": _peak_rss_mb(),
            "binary_size_b": 0,  # torch.compile does not expose a fat-binary blob
            "params": {
                "batch": self.batch_size,
                "shape": list(self.shape),
                "device": device,
            },
            "extras": {
                "eager_exec_time_s": eager_median,
                "eager_exec_samples": eager_samples,
                "speedup_vs_eager": speedup,
                "capture_time_s": capture_time_s,
                "compile_time_breakdown": {
                    "capture_s": capture_time_s,
                    "compile_s": compile_time_s - capture_time_s,
                },
            },
        }


# ---------------------------------------------------------------------------
# Helpers (duplicated to keep modules self-contained).
# ---------------------------------------------------------------------------


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


BENCHMARK = ResNet50Benchmark()
