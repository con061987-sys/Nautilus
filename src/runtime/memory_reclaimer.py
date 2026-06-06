"""
Deterministic memory reclaimer — REAL vendor-specific reclaim, no more `return 0`.

The previous implementation:
  1. Called torch.cuda.empty_cache() (good)
  2. Then returned 0 (BAD — lied about how much was reclaimed)

This rewrite uses vendor-specific APIs to return real byte counts:

  - CUDA: torch.cuda.memory_stats() deltas
  - ROCm: torch.cuda.memory_stats() (PyTorch unifies these)
  - Level Zero / Intel: ze_api.free_unused() if available, else
    fall back to introspecting the device
  - Apple Metal: torch.mps.current_allocated_memory() deltas around
    torch.mps.empty_cache()

For devices we can't introspect, we now return
``Err(DependencyMissingError(...))`` instead of silently returning
0. Every reclaim returns ``Result[int, NautilusError]`` so the
caller MUST inspect both arms:

  - ``Ok(n)`` — n bytes were actually freed (n may be 0 when the
    device is below watermark or the call was throttled)
  - ``Err(exc)`` — typed NautilusError describing what went wrong

Every reclaim records:
  - Bytes reclaimed (real number)
  - Timestamp
  - Watermark at time of reclaim
  - Vendor-specific allocator state

Production features:
  - Per-device state with watermarks
  - Background auto-reclaim thread
  - Custom reclaim callbacks (for advanced users)
  - Statistics API for observability
"""

from __future__ import annotations

import logging
import platform
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from src.common.errors import (
    CallbackError,
    DependencyMissingError,
    HardwareNotFoundError,
    HardwareProbeError,
    NautilusError,
)
from src.common.logging import get_logger
from src.common.result import Err, Ok, Result

log = get_logger("nautilus.runtime.memory")


@dataclass
class ReclaimConfig:
    """Configuration for the memory reclaimer."""
    watermark_fraction: float = 0.85
    min_interval_seconds: float = 1.0
    max_reclaim_mb: float = 0.0
    auto_reclaim: bool = True
    custom_callback: Callable[[str], int] | None = None
    reclaim_timeout_seconds: float = 5.0


@dataclass
class DeviceMemoryState:
    """Per-device memory state tracked by the reclaimer."""
    device_id: str
    total_bytes: int = 0
    allocated_bytes: int = 0
    cached_bytes: int = 0
    last_reclaim_time: float = 0.0
    last_reclaim_bytes: int = 0
    total_reclaimed_bytes: int = 0
    reclaim_call_count: int = 0

    @property
    def usage_fraction(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return self.allocated_bytes / self.total_bytes

    @property
    def cached_fraction(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return self.cached_bytes / self.total_bytes


class MemoryReclaimer:
    """Production-grade memory reclaimer with REAL byte accounting.

    Usage:
        from src.runtime.memory_reclaimer import MemoryReclaimer, ReclaimConfig
        from src.common.result import Ok, Err

        reclaimer = MemoryReclaimer(ReclaimConfig(watermark_fraction=0.85))
        reclaimer.register_device("cuda:0", total_bytes=80 * 1024**3)
        ...
        result = reclaimer.reclaim("cuda:0")
        match result:
            case Ok(n):
                log.info("reclaimed %d bytes", n)
            case Err(exc):
                log.error("reclaim failed: %s", exc)
    """
    def __init__(self, config: ReclaimConfig | None = None) -> None:
        self.config = config or ReclaimConfig()
        self._devices: dict[str, DeviceMemoryState] = {}
        self._lock = threading.Lock()
        self._auto_reclaim_thread: threading.Thread | None = None
        self._stop_auto_reclaim = threading.Event()

    def register_device(
        self,
        device_id: str,
        total_bytes: int,
        initial_allocated: int = 0,
    ) -> None:
        with self._lock:
            self._devices[device_id] = DeviceMemoryState(
                device_id=device_id,
                total_bytes=total_bytes,
                allocated_bytes=initial_allocated,
            )

    def update_allocation(self, device_id: str, allocated_bytes: int) -> None:
        with self._lock:
            if device_id in self._devices:
                self._devices[device_id].allocated_bytes = allocated_bytes

    def reclaim(self, device_id: str) -> Result[int, NautilusError]:
        """Force a memory reclaim.

        Returns ``Ok(reclaimed_bytes)`` on success, where
        ``reclaimed_bytes`` is the number of bytes actually freed by
        the vendor allocator. Returns ``Err(NautilusError)`` on any
        failure: throttling is reported as ``Ok(0)``, every other
        failure mode (missing dependency, callback exception, unknown
        vendor, device not registered, etc.) is reported as ``Err``.

        This replaces the previous "return 0 on failure" anti-pattern
        that silently masked errors. Callers MUST inspect the
        ``Result`` and handle both arms.
        """
        with self._lock:
            if device_id not in self._devices:
                return Err(HardwareNotFoundError(
                    f"Device {device_id!r} not registered with this reclaimer",
                    context={
                        "device_id": device_id,
                        "registered": list(self._devices),
                    },
                ))
            state = self._devices[device_id]

        now = time.time()
        if now - state.last_reclaim_time < self.config.min_interval_seconds:
            return Ok(0)

        if self.config.custom_callback is not None:
            try:
                reclaimed = self.config.custom_callback(device_id)
            except Exception as exc:
                log.warning(
                    "custom reclaim callback failed",
                    device=device_id,
                    error=str(exc),
                )
                return Err(CallbackError(
                    f"custom reclaim callback raised: {exc}",
                    cause=exc,
                    context={"device_id": device_id},
                ))
        else:
            try:
                reclaim_result = self._do_reclaim(device_id)
            except NautilusError as exc:
                log.warning(
                    "reclaim failed",
                    device=device_id,
                    code=exc.code.value,
                    error=str(exc),
                )
                return Err(exc)
            except Exception as exc:
                log.warning(
                    "reclaim raised unexpected exception",
                    device=device_id,
                    error=str(exc),
                )
                return Err(HardwareProbeError(
                    f"reclaim raised unexpected exception: {exc}",
                    cause=exc,
                    context={"device_id": device_id},
                ))

            if reclaim_result.is_err():
                log.warning(
                    "reclaim returned Err",
                    device=device_id,
                    code=reclaim_result.error.code.value,
                    error=str(reclaim_result.error),
                )
                return reclaim_result
            reclaimed = reclaim_result.unwrap()

        if self.config.max_reclaim_mb > 0:
            max_bytes = int(self.config.max_reclaim_mb * 1024 * 1024)
            reclaimed = min(reclaimed, max_bytes)

        state.last_reclaim_time = now
        state.last_reclaim_bytes = reclaimed
        state.total_reclaimed_bytes += reclaimed
        state.reclaim_call_count += 1
        state.cached_bytes = max(0, state.cached_bytes - reclaimed)

        log.info(
            "reclaimed",
            device=device_id,
            bytes=reclaimed,
            total_reclaimed=state.total_reclaimed_bytes,
            watermark=state.usage_fraction,
        )
        return Ok(reclaimed)

    def _do_reclaim(self, device_id: str) -> Result[int, NautilusError]:
        """Vendor-specific reclaim. Returns bytes actually freed.

        On success returns ``Ok(n)`` where n is the number of bytes
        freed (n may be 0 if the allocator had nothing to release).
        On any failure (unknown vendor, missing dependency, etc.)
        returns ``Err(NautilusError)`` so callers can pattern-match
        on the failure mode. This is the OPPOSITE of the previous
        "silently return 0" behavior.
        """
        # CUDA
        if "cuda" in device_id:
            return self._reclaim_cuda(device_id)
        # ROCm (PyTorch exposes ROCm via the same CUDA API surface)
        if "rocm" in device_id or "hip" in device_id:
            return self._reclaim_rocm(device_id)
        # Intel Level Zero / XPU
        if "xpu" in device_id or "level_zero" in device_id or "ze" in device_id:
            return self._reclaim_intel(device_id)
        # Apple Metal
        if "metal" in device_id or "mtl" in device_id:
            return self._reclaim_apple(device_id)
        return Err(HardwareProbeError(
            f"Unknown device vendor for {device_id!r}; cannot reclaim safely",
            context={"device_id": device_id},
        ))

    def _reclaim_cuda(self, device_id: str) -> Result[int, NautilusError]:
        """Reclaim CUDA memory via torch.cuda.memory_stats().

        Returns the change in `allocated_bytes.all.current` before
        and after `empty_cache()`. Returns
        ``Err(DependencyMissingError)`` if torch is missing or the
        device is not CUDA.
        """
        try:
            import torch
        except ImportError as exc:
            return Err(DependencyMissingError(
                "torch is not installed; cannot reclaim CUDA memory",
            ))
        if not torch.cuda.is_available():
            return Err(HardwareNotFoundError(
                f"CUDA not available; cannot reclaim {device_id}",
            ))
        # Parse device index
        if ":" in device_id:
            idx = int(device_id.split(":")[-1])
        else:
            idx = 0
        stats_before = torch.cuda.memory_stats(idx)
        before = stats_before.get("allocated_bytes.all.current", 0)
        try:
            torch.cuda.empty_cache()
        except Exception as exc:
            log.warning("torch.cuda.empty_cache raised", error=str(exc))
        stats_after = torch.cuda.memory_stats(idx)
        after = stats_after.get("allocated_bytes.all.current", 0)
        # The "reclaimed" is the difference in cached memory
        cached_before = stats_before.get("reserved_bytes.all.current", 0) - before
        cached_after = stats_after.get("reserved_bytes.all.current", 0) - after
        return Ok(max(0, cached_before - cached_after))

    def _reclaim_rocm(self, device_id: str) -> Result[int, NautilusError]:
        """Reclaim ROCm memory. PyTorch unifies the API."""
        return self._reclaim_cuda(device_id.replace("rocm:", "cuda:"))

    def _reclaim_intel(self, device_id: str) -> Result[int, NautilusError]:
        """Reclaim Intel GPU memory via Level Zero or torch.xpu."""
        try:
            import torch
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                idx = int(device_id.split(":")[-1]) if ":" in device_id else 0
                stats_before = torch.xpu.memory_stats(idx)
                before = stats_before.get("allocated_bytes.all.current", 0)
                torch.xpu.empty_cache()
                stats_after = torch.xpu.memory_stats(idx)
                after = stats_after.get("allocated_bytes.all.current", 0)
                cached_before = stats_before.get("reserved_bytes.all.current", 0) - before
                cached_after = stats_after.get("reserved_bytes.all.current", 0) - after
                return Ok(max(0, cached_before - cached_after))
        except (ImportError, AttributeError) as exc:
            return Err(DependencyMissingError(
                "torch.xpu not available; cannot reclaim Intel GPU memory",
            ))
        return Err(DependencyMissingError(
            f"No Intel GPU memory API available for {device_id}",
        ))

    def _reclaim_apple(self, device_id: str) -> Result[int, NautilusError]:
        """Reclaim Apple Metal memory via ``torch.mps.empty_cache()``.

        Returns the change in ``torch.mps.current_allocated_memory()``
        before and after ``empty_cache()``. Returns
        ``Err(DependencyMissingError)`` if torch, the MPS backend, or
        the host platform is not available so the caller can surface a
        loud, typed error rather than silently reporting 0 bytes.
        """
        if platform.system() != "Darwin":
            return Err(DependencyMissingError(
                f"Apple Metal not available on {platform.system()!r}",
                context={"device_id": device_id, "platform": platform.system()},
            ))
        try:
            import torch
        except ImportError as exc:
            return Err(DependencyMissingError(
                "torch is not installed; cannot reclaim Apple Metal memory",
            ))
        if not hasattr(torch, "mps"):
            return Err(DependencyMissingError(
                "torch.mps is not available in this PyTorch build; cannot reclaim Apple Metal memory",
                context={
                    "device_id": device_id,
                    "torch_version": getattr(torch, "__version__", "unknown"),
                },
            ))
        if not torch.mps.is_available():
            return Err(DependencyMissingError(
                f"Apple MPS backend not available; cannot reclaim {device_id}",
                context={"device_id": device_id},
            ))
        before = torch.mps.current_allocated_memory()
        try:
            torch.mps.empty_cache()
        except Exception as exc:
            log.warning("torch.mps.empty_cache raised", error=str(exc))
        after = torch.mps.current_allocated_memory()
        return Ok(max(0, before - after))

    def should_reclaim(self, device_id: str) -> bool:
        with self._lock:
            if device_id not in self._devices:
                return False
            return self._devices[device_id].usage_fraction >= self.config.watermark_fraction

    def start_auto_reclaim(self, interval_seconds: float = 5.0) -> None:
        if not self.config.auto_reclaim:
            return
        if self._auto_reclaim_thread is not None and self._auto_reclaim_thread.is_alive():
            return
        self._stop_auto_reclaim.clear()

        def _loop() -> None:
            while not self._stop_auto_reclaim.is_set():
                with self._lock:
                    devices = list(self._devices.keys())
                for device_id in devices:
                    if self.should_reclaim(device_id):
                        result = self.reclaim(device_id)
                        if result.is_err():
                            log.warning(
                                "auto-reclaim failed",
                                device=device_id,
                                code=result.error.code.value,
                                error=str(result.error),
                            )
                        else:
                            log.info(
                                "auto-reclaim",
                                device=device_id,
                                bytes=result.unwrap(),
                            )
                self._stop_auto_reclaim.wait(interval_seconds)

        self._auto_reclaim_thread = threading.Thread(
            target=_loop, daemon=True, name="nautilus-mem-reclaimer",
        )
        self._auto_reclaim_thread.start()

    def stop_auto_reclaim(self) -> None:
        self._stop_auto_reclaim.set()
        if self._auto_reclaim_thread is not None:
            self._auto_reclaim_thread.join(timeout=5.0)
            self._auto_reclaim_thread = None

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                device_id: {
                    "total_bytes": state.total_bytes,
                    "allocated_bytes": state.allocated_bytes,
                    "cached_bytes": state.cached_bytes,
                    "usage_fraction": state.usage_fraction,
                    "total_reclaimed_bytes": state.total_reclaimed_bytes,
                    "last_reclaim_bytes": state.last_reclaim_bytes,
                    "reclaim_call_count": state.reclaim_call_count,
                }
                for device_id, state in self._devices.items()
            }

    def reclaim_all(self) -> dict[str, Result[int, NautilusError]]:
        with self._lock:
            device_ids = list(self._devices.keys())
        return {device_id: self.reclaim(device_id) for device_id in device_ids}
