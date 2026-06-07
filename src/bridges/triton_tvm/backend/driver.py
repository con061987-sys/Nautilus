"""TVMDriver — Triton driver implementation for the bridge.

Drivers in Triton manage device-side resources: memory allocation,
kernel dispatch, and stream management. Our driver delegates to
Triton's active driver for allocation but routes kernel launches
through the bridge orchestrator so that TVM-tuned configs are
applied before each launch.

This is the runtime half of the bridge. The compiler half is
TVMBackend; together they form the full out-of-tree backend.
"""

from __future__ import annotations

from typing import Any

from src.common.logging import get_logger

try:
    from triton.runtime.driver import DriverBase

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

    class DriverBase:  # type: ignore
        def __init__(self, *a: Any, **kw: Any) -> None: ...


logger = get_logger(__name__)


class TVMDriver(DriverBase):
    """Driver that routes through the Triton ↔ TVM bridge orchestrator.

    Lifecycle:
      1. User calls: triton.runtime.driver.set_active(TVMDriver())
      2. Subsequent kernel launches go through this driver
      3. The driver queries the bridge for the best config for the kernel
      4. Launches the kernel via the underlying CUDA/HIP/Metal driver
    """

    # Class-level singleton — Triton expects one active driver at a time
    _instance: TVMDriver | None = None

    def __init__(self) -> None:
        if not TRITON_AVAILABLE:
            raise RuntimeError(
                "Triton is required to use TVMDriver. Install with: pip install triton"
            )
        super().__init__()
        # Lazy import to avoid circular dependency
        from ..bridge_orchestrator import TritonTVMBridge

        self._bridge = TritonTVMBridge(
            cache_dir="/tmp/nvindia_cud_driver_cache",
            enable_tvm=True,
        )
        logger.info("TVMDriver initialised; bridge is active")

    @classmethod
    def instance(cls) -> TVMDriver:
        """Return the singleton instance, creating if needed."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_active_torch_device(self) -> Any:
        """Return the active torch device, falling back to default."""
        try:
            import torch

            if torch.cuda.is_available():
                return torch.device("cuda", torch.cuda.current_device())
        except ImportError:
            pass
        return None

    def get_benchmarker(self) -> Any:
        """Return the benchmark function used by @triton.autotune.

        We delegate to Triton's CUDA benchmarker for now. In future
        versions this can be replaced with a TVM-backed benchmarker
        that uses MetaSchedule's cost model.
        """
        # Fall back to Triton's CUDA benchmarker
        try:
            from triton.testing import do_bench

            return do_bench
        except ImportError:
            logger.warning("triton.testing.do_bench not available; using time.perf_counter")
            import time

            def fallback_bench(fn: Any, **kwargs: Any) -> float:
                start = time.perf_counter()
                fn()
                return (time.perf_counter() - start) * 1000

            return fallback_bench

    def get_current_device(self) -> int:
        """Return the current device index."""
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.current_device()
        except ImportError:
            pass
        return 0

    def get_current_stream(self, device: Any) -> Any:
        """Return the current stream for a device."""
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.current_stream(device)
        except ImportError:
            pass
        return None

    def set_current_stream(self, device: Any, stream: Any) -> None:
        """Set the current stream for a device."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.set_stream(stream)
        except ImportError:
            pass

    def get_current_target(self) -> Any:
        """Return the current GPUTarget.

        Used by Triton to determine the backend for compilation.
        """
        try:
            import torch
            from triton.backends.compiler import GPUTarget

            if torch.cuda.is_available():
                cap = torch.cuda.get_device_capability()
                arch = cap[0] * 10 + cap[1]
                return GPUTarget(backend="cuda", arch=arch, warp_size=32)
        except Exception:
            pass
        # Fallback
        try:
            from triton.backends.compiler import GPUTarget

            return GPUTarget(backend="cuda", arch=80, warp_size=32)
        except Exception:
            return None

    def __del__(self) -> None:
        """Clean up singleton reference on driver teardown."""
        TVMDriver._instance = None
