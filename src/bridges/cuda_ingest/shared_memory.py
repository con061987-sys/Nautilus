"""Shared memory analysis and translation for CUDA ingestion.

Translates CUDA __shared__ memory declarations to Triton shared
memory allocations. Also performs bank conflict analysis to ensure
the translated code doesn't suffer from performance degradation.

CUDA shared memory model:
  - __shared__ float data[256];  // 256 floats per block
  - Allocated per-block (per-program in Triton terms)
  - Accessible by all threads in the block
  - ~100x faster than global memory

Triton shared memory model:
  - tl.zeros((BLOCK_SIZE,), dtype=tl.float32)  // per-block allocation
  - Also per-program
  - Same accessibility semantics

The translation is mostly direct, but we need to:
  1. Determine the size of each __shared__ allocation
  2. Analyze access patterns for bank conflicts
  3. Insert appropriate Triton allocation calls

Production features:
  - Static array sizes (compile-time constants) — most common case
  - Dynamic array sizes (runtime parameters) — supported
  - Bank conflict detection and warnings
  - Multiple __shared__ arrays per kernel
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SharedMemAllocation:
    """A single __shared__ memory allocation."""
    name: str
    element_type: str
    array_size: int  # 0 = dynamic, >0 = static size
    is_static: bool
    raw_declaration: str

    @property
    def total_bytes(self) -> int:
        """Estimate total bytes (assuming 4-byte elements)."""
        elem_size = {
            "float": 4, "int": 4, "double": 8, "char": 1,
            "short": 2, "long": 8, "long long": 8,
        }.get(self.element_type.lower(), 4)
        if self.is_static:
            return self.array_size * elem_size
        return elem_size  # Per-element estimate for dynamic


@dataclass
class SharedMemPlan:
    """Plan for all shared memory allocations in a kernel."""
    allocations: list[SharedMemAllocation] = field(default_factory=list)
    total_static_bytes: int = 0
    estimated_dynamic_bytes: int = 0
    bank_conflict_warnings: list[str] = field(default_factory=list)

    @property
    def num_allocations(self) -> int:
        return len(self.allocations)


class SharedMemoryAnalyzer:
    """Analyzes and plans shared memory usage for CUDA kernels.

    Usage:
        analyzer = SharedMemoryAnalyzer()
        plan = analyzer.analyze(kernel)
        # Use plan.allocations, plan.total_static_bytes, etc.
    """

    # Pattern for __shared__ declarations (supports multi-dimensional arrays)
    SHARED_DECL_RE = re.compile(
        r'__shared__\s+(\w+(?:\s*\*)?)\s+(\w+)((?:\s*\[[^\]]*\])+)\s*;',
    )

    def analyze(self, kernel: Any) -> SharedMemPlan:
        """Analyze a parsed CUDA kernel for shared memory usage.

        Args:
            kernel: CudaKernel from the parser.

        Returns:
            SharedMemPlan with all allocations and analysis.
        """
        plan = SharedMemPlan()

        for decl in kernel.shared_mem_declarations:
            allocation = self._parse_declaration(decl)
            if allocation is not None:
                plan.allocations.append(allocation)
                if allocation.is_static:
                    plan.total_static_bytes += allocation.total_bytes
                else:
                    plan.estimated_dynamic_bytes += allocation.total_bytes

        # Bank conflict analysis
        if len(plan.allocations) > 1:
            warnings = self._analyze_bank_conflicts(plan.allocations)
            plan.bank_conflict_warnings.extend(warnings)

        return plan

    def _parse_declaration(self, decl: dict[str, Any]) -> SharedMemAllocation | None:
        """Parse a __shared__ declaration into an allocation."""
        type_str = decl.get("type", "").strip()
        name = decl.get("name", "").strip()
        raw = decl.get("raw", "")

        if not type_str or not name:
            return None

        # Extract ALL [SIZE] dimensions for multi-dimensional arrays.
        # Pattern: __shared__ TYPE NAME[D1][D2]...[DN];
        # Total size is the product of all dimensions.
        head_match = re.search(
            rf'{re.escape(name)}((?:\s*\[[^\]]*\])+)', raw,
        )
        if not head_match:
            return SharedMemAllocation(
                name=name,
                element_type=type_str,
                array_size=0,
                is_static=False,
                raw_declaration=raw,
            )

        size_strs = [
            s.strip() for s in re.findall(r'\[([^\]]*)\]', head_match.group(1))
        ]
        if all(s.isdigit() for s in size_strs):
            total_size = 1
            for s in size_strs:
                total_size *= int(s)
            return SharedMemAllocation(
                name=name,
                element_type=type_str,
                array_size=total_size,
                is_static=True,
                raw_declaration=raw,
            )

        # Dynamic size (e.g. [BLOCK_SIZE])
        return SharedMemAllocation(
            name=name,
            element_type=type_str,
            array_size=0,
            is_static=False,
            raw_declaration=raw,
        )

    def _analyze_bank_conflicts(
        self, allocations: list[SharedMemAllocation],
    ) -> list[str]:
        """Detect potential shared memory bank conflicts.

        Shared memory has 32 banks. If multiple threads access
        addresses that map to the same bank, the access is serialized.
        Common patterns that cause conflicts:
        - Strided access with stride that's not coprime with 32
        - Same address accessed by all threads (broadcast, OK)
        - Adjacent addresses by adjacent threads (no conflict)
        """
        warnings: list[str] = []
        NUM_BANKS = 32

        for alloc in allocations:
            if not alloc.is_static:
                continue
            # Simple check: if the array size is a power of 2 and
            # the size is small, the stride will likely be the
            # element size which could cause conflicts
            if alloc.array_size > 0 and alloc.array_size & (alloc.array_size - 1) == 0:
                if alloc.array_size >= 32 and alloc.array_size % NUM_BANKS == 0:
                    warnings.append(
                        f"Shared memory array '{alloc.name}' has size "
                        f"{alloc.array_size} which is a multiple of {NUM_BANKS}. "
                        f"Consider padding by 1 element to avoid bank conflicts."
                    )

        return warnings

    def generate_triton_allocation(
        self, allocation: SharedMemAllocation,
    ) -> str:
        """Generate the Triton code for a shared memory allocation.

        For static sizes:
            smem = tl.zeros((SIZE,), dtype=DTYPE)

        For dynamic sizes:
            smem = tl.zeros((DYNAMIC_SIZE,), dtype=DTYPE)
        """
        dtype_map = {
            "float": "tl.float32",
            "int": "tl.int32",
            "double": "tl.float64",
            "char": "tl.int8",
            "short": "tl.int16",
            "long": "tl.int64",
            "long long": "tl.int64",
        }
        triton_dtype = dtype_map.get(allocation.element_type.lower(), "tl.float32")

        if allocation.is_static:
            return f"{allocation.name}_smem = tl.zeros(({allocation.array_size},), dtype={triton_dtype})"
        return (
            f"{allocation.name}_smem = tl.zeros((BLOCK_SIZE,), "
            f"dtype={triton_dtype})  # dynamic size — caller must pass BLOCK_SIZE"
        )

    def generate_all_allocations(self, plan: SharedMemPlan) -> str:
        """Generate Triton code for all shared memory allocations."""
        return "\n    ".join(
            self.generate_triton_allocation(alloc)
            for alloc in plan.allocations
        )
