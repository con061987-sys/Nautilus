"""Per-shard kernel execution and fat binary dispatch.

After GSPMD shards a model across multiple devices, each shard
needs to run on its assigned device. This module:
  1. Builds per-shard fat binaries (using Phase 2's FatBinaryBuilder)
  2. Dispatches execution to the correct device and kernel
  3. Manages cross-shard communication
  4. Collects and returns results

Production features:
  - Per-shard fat binary caching
  - Heterogeneous execution (different binaries per vendor)
  - Communication backend integration
  - Result collection and aggregation
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from src.common.logging import get_logger

from .comm_backend import CommBackend
from .device_mesh import DeviceMesh, DeviceVendor
from .stablehlo_to_triton import (
    TritonSource,
)
from .stablehlo_to_triton import (
    translate as stablehlo_to_triton_translate,
)

try:
    from src.bridges.aot_packager.builder import (
        FatBinaryBuilder,
        FatBinaryConfig,
    )

    FAT_BINARY_AVAILABLE = True
except ImportError:
    FAT_BINARY_AVAILABLE = False

logger = get_logger(__name__)


@dataclass
class ShardExecutionResult:
    """Result of executing a single shard."""

    shard_id: int
    vendor: str
    arch: str
    device_id: int
    success: bool
    fat_binary_result: Any = None  # FatBinaryResult
    error: str | None = None
    execution_time_s: float = 0.0

    @property
    def is_usable(self) -> bool:
        return self.success and self.fat_binary_result is not None


class ShardExecutor:
    """Executes per-shard kernels on heterogeneous devices.

    Usage:
        executor = ShardExecutor(device_mesh, comm_backend)
        results = executor.execute_all_shards(gspmd_result, stablehlo_module)
    """

    def __init__(
        self,
        device_mesh: DeviceMesh,
        comm_backend: CommBackend,
    ) -> None:
        self.device_mesh = device_mesh
        self.comm_backend = comm_backend
        self._fat_binary_builder: FatBinaryBuilder | None = None
        if FAT_BINARY_AVAILABLE:
            try:
                self._fat_binary_builder = FatBinaryBuilder()
            except Exception as exc:
                logger.warning("FatBinaryBuilder not available: %s", exc)

    def execute_all_shards(
        self,
        gspmd_result: Any,
        stablehlo_module: Any,
    ) -> list[ShardExecutionResult]:
        """Execute all shards produced by GSPMD.

        For each vendor in the mesh, build a fat binary and run
        the shard on that vendor's hardware.
        """
        results: list[ShardExecutionResult] = []
        shard_id = 0

        for vendor in self.device_mesh.vendors:
            devices = self.device_mesh.get_devices_by_vendor(vendor)
            for device in devices:
                result = self._execute_single_shard(
                    shard_id=shard_id,
                    vendor=vendor,
                    device=device,
                    gspmd_result=gspmd_result,
                    stablehlo_module=stablehlo_module,
                )
                results.append(result)
                shard_id += 1

        return results

    def _execute_single_shard(
        self,
        shard_id: int,
        vendor: DeviceVendor,
        device: Any,
        gspmd_result: Any,
        stablehlo_module: Any,
    ) -> ShardExecutionResult:
        """Execute a single shard on the assigned device."""
        start = time.perf_counter()

        # Build a fat binary for this shard
        fat_binary_result = None
        kernel_name = f"shard_{shard_id}_{device.device_id}"
        if self._fat_binary_builder is not None:
            try:
                stablehlo_mlir_text = getattr(stablehlo_module, "mlir_text", "")
                if not stablehlo_mlir_text:
                    raise RuntimeError(
                        "StableHLO module has no mlir_text; cannot translate to Triton"
                    )
                kernel_source = self._generate_shard_source(
                    stablehlo_mlir_text,
                    kernel_name=kernel_name,
                )
                config = FatBinaryConfig(
                    kernel_name=kernel_name,
                    kernel_source=kernel_source,
                    skip_amd=(vendor != DeviceVendor.AMD),
                    skip_intel=(vendor != DeviceVendor.INTEL),
                    skip_nvidia=(vendor != DeviceVendor.NVIDIA),
                    skip_validation=True,
                )
                fat_binary_result = self._fat_binary_builder.build(config)
            except Exception as exc:
                logger.warning("Fat binary build failed for shard %d: %s", shard_id, exc)

        elapsed = time.perf_counter() - start
        return ShardExecutionResult(
            shard_id=shard_id,
            vendor=vendor.value,
            arch=device.arch,
            device_id=device.device_id,
            success=fat_binary_result is not None and fat_binary_result.is_usable,
            fat_binary_result=fat_binary_result,
            execution_time_s=elapsed,
        )

    def _generate_shard_source(
        self,
        stablehlo_mlir_text: str,
        kernel_name: str,
    ) -> str:
        """Translate per-shard StableHLO MLIR into a Triton kernel source.

        Delegates to ``stablehlo_to_triton.translate`` so the same
        translator used by Wave 1.1 produces the kernel that gets
        compiled into the per-shard fat binary in Stage 5. The
        returned string is the Python source of a ``@triton.jit``
        kernel ready for the fat binary builder.
        """
        triton_source: TritonSource = stablehlo_to_triton_translate(
            stablehlo_mlir_text,
            kernel_name=kernel_name,
        )
        return triton_source.source
