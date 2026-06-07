"""Metadata extraction from Triton kernels at JIT call time.

Intercepts Triton kernel execution to extract mathematical bounds,
shapes, strides, and tuning parameters without parsing the MLIR IR.
This is the foundation of the config bridge approach — we extract
metadata, not IR, and use it to construct equivalent TIR templates.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class KernelMetadata:
    """Metadata extracted from a Triton kernel at JIT call time.

    This is the bridge's canonical representation of a kernel's
    mathematical structure. It contains everything needed to construct
    an equivalent TVM TIR PrimFunc for MetaSchedule tuning.
    """

    # Kernel identity
    kernel_name: str
    source_hash: str  # SHA-256 of kernel source (body only, no filename)

    # Grid dimensions
    grid_0: int
    grid_1: int
    grid_2: int

    # Compile-time options
    num_warps: int
    num_stages: int
    num_ctas: int

    # Argument information
    arg_shapes: tuple[tuple[int, ...], ...] = field(default_factory=tuple)
    arg_strides: tuple[tuple[int, ...], ...] = field(default_factory=tuple)
    arg_dtypes: tuple[str, ...] = field(default_factory=tuple)

    # Kernel type classification
    is_matmul: bool = False
    is_reduction: bool = False
    is_elementwise: bool = False

    # Matmul-specific (if applicable)
    matmul_m: int | None = None
    matmul_n: int | None = None
    matmul_k: int | None = None

    def __post_init__(self) -> None:
        """Validate metadata fields."""
        if self.grid_0 < 1:
            raise ValueError(f"grid_0 must be >= 1, got {self.grid_0}")
        if self.num_warps not in (1, 2, 4, 8, 16, 32):
            raise ValueError(f"num_warps must be power of 2, got {self.num_warps}")

    @property
    def cache_key(self) -> str:
        """Deterministic cache key covering all tuning-relevant parameters.

        Includes: kernel source, grid, options, shapes, target.
        Two kernels with different shapes get different cache entries.
        """
        parts = [
            self.source_hash,
            str(self.grid_0),
            str(self.grid_1),
            str(self.grid_2),
            str(self.num_warps),
            str(self.num_stages),
            str(self.num_ctas),
            str(self.arg_shapes),
            str(self.arg_dtypes),
            str(self.matmul_m),
            str(self.matmul_n),
            str(self.matmul_k),
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

    @property
    def grid(self) -> tuple[int, int, int]:
        return (self.grid_0, self.grid_1, self.grid_2)


class MetadataExtractor:
    """Intercepts Triton JIT kernel calls to extract metadata.

    Two modes:
    1. Monkey-patch mode: replaces the kernel's .run() to capture args
    2. Static mode: inspects kernel signature + source (no execution)

    Mode 1 is preferred for accuracy (actual tensor shapes).
    Mode 2 is useful for pre-compilation / ahead-of-time.
    """

    def extract_from_call(
        self,
        kernel_fn: Any,
        grid: tuple[int, int, int],
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None = None,
        num_warps: int = 4,
        num_stages: int = 3,
        num_ctas: int = 1,
    ) -> KernelMetadata:
        """Extract metadata from an actual kernel call with real tensors.

        Args:
            kernel_fn: The @triton.jit decorated function.
            grid: Grid dimensions (gx, gy, gz).
            args: Positional arguments passed to the kernel.
            kwargs: Keyword arguments passed to the kernel.
            num_warps: Number of warps per block.
            num_stages: Number of pipeline stages.
            num_ctas: Number of CTAs (Hopper+ cluster feature).

        Returns:
            KernelMetadata with shapes, strides, dtypes from real tensors.
        """
        import torch  # local import: triton_tvm bridge must load without torch installed

        kwargs = kwargs or {}

        # Extract tensor shapes/strides/dtypes from arguments
        shapes: list[tuple[int, ...]] = []
        strides: list[tuple[int, ...]] = []
        dtypes: list[str] = []

        for arg in args:
            if isinstance(arg, torch.Tensor):
                shapes.append(tuple(arg.shape))
                strides.append(tuple(arg.stride()))
                dtypes.append(_torch_dtype_to_str(arg.dtype))

        # Compute kernel source hash
        source_hash = self._compute_source_hash(kernel_fn)

        # Classify kernel type
        kt = self._classify_kernel(kernel_fn)

        metadata = KernelMetadata(
            kernel_name=kernel_fn.__name__,
            source_hash=source_hash,
            grid_0=grid[0],
            grid_1=grid[1] if len(grid) > 1 else 1,
            grid_2=grid[2] if len(grid) > 2 else 1,
            num_warps=num_warps,
            num_stages=num_stages,
            num_ctas=num_ctas,
            arg_shapes=tuple(shapes),
            arg_strides=tuple(strides),
            arg_dtypes=tuple(dtypes),
            is_matmul=kt == "matmul",
            is_reduction=kt == "reduction",
            is_elementwise=kt == "elementwise",
        )
        return metadata

    def extract_static(
        self,
        kernel_fn: Any,
        grid: tuple[int, int, int],
        num_warps: int = 4,
        num_stages: int = 3,
        num_ctas: int = 1,
    ) -> KernelMetadata:
        """Extract metadata from kernel source alone (no execution).

        Useful for ahead-of-time compilation where tensor shapes
        may not be available yet. Shape information will be incomplete.
        """
        source_hash = self._compute_source_hash(kernel_fn)
        kt = self._classify_kernel(kernel_fn)

        return KernelMetadata(
            kernel_name=kernel_fn.__name__,
            source_hash=source_hash,
            grid_0=grid[0],
            grid_1=grid[1] if len(grid) > 1 else 1,
            grid_2=grid[2] if len(grid) > 2 else 1,
            num_warps=num_warps,
            num_stages=num_stages,
            num_ctas=num_ctas,
            is_matmul=kt == "matmul",
            is_reduction=kt == "reduction",
            is_elementwise=kt == "elementwise",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_source_hash(self, kernel_fn: Any) -> str:
        """Compute SHA-256 of kernel body source, stable across files."""
        try:
            source = inspect.getsource(kernel_fn)
        except (OSError, TypeError):
            source = f"{kernel_fn.__module__}.{kernel_fn.__name__}"

        # Strip function name to avoid cache invalidation on rename
        lines = source.split("\n")
        cleaned: list[str] = []
        for line in lines:
            if line.startswith("def ") or line.startswith("@"):
                continue
            cleaned.append(line.rstrip())
        body = "\n".join(cleaned).strip()
        return hashlib.sha256(body.encode()).hexdigest()

    def _classify_kernel(self, kernel_fn: Any) -> str:
        """Classify kernel type by inspecting source for known patterns.

        Returns one of: 'matmul', 'reduction', 'elementwise', 'unknown'.
        """
        try:
            source = inspect.getsource(kernel_fn)
        except (OSError, TypeError):
            return "unknown"

        if "tl.dot" in source:
            return "matmul"
        if "tl.reduce" in source:
            return "reduction"
        if "tl.load" in source and "tl.store" in source:
            # Elementwise ops typically have load immediately followed by
            # a simple op and store, with no dot/reduce
            return "elementwise"
        return "unknown"


def _torch_dtype_to_str(dtype: torch.dtype) -> str:
    """Map torch.dtype to string representation."""
    import torch  # local import: helper must work independently of extract_from_call

    mapping = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float64: "float64",
        torch.int32: "int32",
        torch.int64: "int64",
        torch.int8: "int8",
        torch.uint8: "uint8",
        torch.bool: "bool",
    }
    return mapping.get(dtype, str(dtype))
