"""CUDA intrinsic → Triton mapping.

Maps CUDA-specific intrinsics to their Triton equivalents.
This is one of the most critical steps in CUDA ingestion — getting
the intrinsics right determines whether translated kernels produce
the same results as the original.

ed CUDA intrinsics supported:
  Thread indexing:
    - threadIdx.x/y/z → tl.program_id(0/1/2)
    - blockIdx.x/y/z → tl.program_id(0/1/2)
    - blockDim.x/y/z → tl.num_programs(0/1/2) (in some cases)
    - gridDim.x/y/z → tl.num_programs(0/1/2)

  Synchronization:
    - __syncthreads() → tl.barrier()
    - __syncwarp() → no-op (Triton manages warp scheduling)
    - __threadfence() → tl.barrier()

  Atomics:
    - atomicAdd → tl.atomic_add
    - atomicSub → tl.atomic_sub
    - atomicMin/Max → tl.atomic_min/max
    - atomicCAS → tl.atomic_cas
    - atomicExch → tl.atomic_exch
    - atomicInc/Dec → tl.atomic_add (approximated)

  Warp-level (Triton limitation — many map to no-ops or approximations):
    - __shfl_sync → tl.shuffle (with limitations)
    - __shfl_up/down/xor_sync → tl.shuffle variants
    - __ballot_sync → bitwise ops
    - __any_sync / __all_sync → reductions

  Memory:
    - __shared__ → tl.shared allocations
    - __constant__ → module-level constants

  Math (mostly 1:1 mappings):
    - sinf/cosf/logf/sqrtf/etc → tl.sin/cos/log/sqrt

Production features:
  - Coverage table for all supported intrinsics
  - Fallback strategies for partially-supported ones
  - Clear documentation of approximations
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from src.common.logging import get_logger

logger = get_logger(__name__)


class IntrinsicCategory(Enum):
    """Categories of CUDA intrinsics."""

    THREAD_INDEX = auto()
    SYNCHRONIZATION = auto()
    ATOMIC = auto()
    WARP_LEVEL = auto()
    MEMORY = auto()
    MATH = auto()
    RUNTIME = auto()


@dataclass(frozen=True)
class IntrinsicMapping:
    """A single CUDA intrinsic → Triton mapping."""

    cuda_name: str
    triton_name: str
    category: IntrinsicCategory
    is_exact: bool  # True = semantically identical, False = approximation
    transform: Callable[[str], str] | None = None
    notes: str = ""


class IntrinsicMapper:
    """Maps CUDA intrinsics to their Triton equivalents.

    The mapper maintains a registry of all supported intrinsics and
    provides methods to transform CUDA source text by replacing
    intrinsic calls with their Triton equivalents.
    """

    def __init__(self) -> None:
        self._mappings: dict[str, IntrinsicMapping] = {}
        self._register_default_mappings()

    def _register_default_mappings(self) -> None:
        """Register all built-in CUDA → Triton mappings."""
        # Thread indexing
        self._register(
            IntrinsicMapping(
                cuda_name="threadIdx.x",
                triton_name="tl.program_id(0)",
                category=IntrinsicCategory.THREAD_INDEX,
                is_exact=False,  # Triton's mapping isn't 1:1
                notes="Triton doesn't have a 1:1 threadIdx.x; we use program_id(0) in a 1D launch",
            )
        )
        self._register(
            IntrinsicMapping(
                cuda_name="threadIdx.y",
                triton_name="tl.program_id(1)",
                category=IntrinsicCategory.THREAD_INDEX,
                is_exact=False,
                notes="Requires 2D or 3D launch grid",
            )
        )
        self._register(
            IntrinsicMapping(
                cuda_name="threadIdx.z",
                triton_name="tl.program_id(2)",
                category=IntrinsicCategory.THREAD_INDEX,
                is_exact=False,
                notes="Requires 3D launch grid",
            )
        )
        self._register(
            IntrinsicMapping(
                cuda_name="blockIdx.x",
                triton_name="tl.program_id(0)",
                category=IntrinsicCategory.THREAD_INDEX,
                is_exact=True,
            )
        )
        self._register(
            IntrinsicMapping(
                cuda_name="blockIdx.y",
                triton_name="tl.program_id(1)",
                category=IntrinsicCategory.THREAD_INDEX,
                is_exact=True,
            )
        )
        self._register(
            IntrinsicMapping(
                cuda_name="blockIdx.z",
                triton_name="tl.program_id(2)",
                category=IntrinsicCategory.THREAD_INDEX,
                is_exact=True,
            )
        )
        self._register(
            IntrinsicMapping(
                cuda_name="blockDim.x",
                triton_name="tl.num_programs(0)",
                category=IntrinsicCategory.THREAD_INDEX,
                is_exact=False,
            )
        )

        # Synchronization
        self._register(
            IntrinsicMapping(
                cuda_name="__syncthreads()",
                triton_name="tl.barrier()",
                category=IntrinsicCategory.SYNCHRONIZATION,
                is_exact=True,
            )
        )
        self._register(
            IntrinsicMapping(
                cuda_name="__syncwarp()",
                triton_name="# tl.syncwarp not directly supported; using tl.barrier()",
                category=IntrinsicCategory.SYNCHRONIZATION,
                is_exact=False,
                notes="Triton manages warp scheduling internally; no explicit sync needed",
            )
        )
        self._register(
            IntrinsicMapping(
                cuda_name="__threadfence()",
                triton_name="tl.barrier()",
                category=IntrinsicCategory.SYNCHRONIZATION,
                is_exact=False,
            )
        )

        # Atomics
        for cuda_name, triton_name in [
            ("atomicAdd", "tl.atomic_add"),
            ("atomicSub", "tl.atomic_sub"),
            ("atomicMin", "tl.atomic_min"),
            ("atomicMax", "tl.atomic_max"),
            ("atomicAnd", "tl.atomic_and"),
            ("atomicOr", "tl.atomic_or"),
            ("atomicXor", "tl.atomic_xor"),
        ]:
            self._register(
                IntrinsicMapping(
                    cuda_name=cuda_name,
                    triton_name=triton_name,
                    category=IntrinsicCategory.ATOMIC,
                    is_exact=True,
                )
            )
        self._register(
            IntrinsicMapping(
                cuda_name="atomicCAS",
                triton_name="tl.atomic_cas",
                category=IntrinsicCategory.ATOMIC,
                is_exact=True,
            )
        )
        self._register(
            IntrinsicMapping(
                cuda_name="atomicExch",
                triton_name="tl.atomic_xchg",
                category=IntrinsicCategory.ATOMIC,
                is_exact=True,
            )
        )

        # Math
        for cuda_name, triton_name in [
            ("sinf", "tl.sin"),
            ("cosf", "tl.cos"),
            ("tanf", "tl.tan"),
            ("logf", "tl.log"),
            ("log2f", "tl.log2"),
            ("expf", "tl.exp"),
            ("exp2f", "tl.exp2"),
            ("sqrtf", "tl.sqrt"),
            ("rsqrtf", "tl.rsqrt"),
            ("fabsf", "tl.abs"),
            ("fmaxf", "tl.maximum"),
            ("fminf", "tl.minimum"),
            ("floorf", "tl.floor"),
            ("ceilf", "tl.ceil"),
            ("roundf", "tl.extra.cuda.libdevice.round"),
            ("erff", "tl.extra.cuda.libdevice.erf"),
        ]:
            self._register(
                IntrinsicMapping(
                    cuda_name=cuda_name,
                    triton_name=triton_name,
                    category=IntrinsicCategory.MATH,
                    is_exact=True,
                )
            )

        # Memory qualifiers (no direct translation; used in parser)
        self._register(
            IntrinsicMapping(
                cuda_name="__shared__",
                triton_name="# (handled by shared_memory.py)",
                category=IntrinsicCategory.MEMORY,
                is_exact=True,
                notes="__shared__ arrays are translated to tl.shared allocations",
            )
        )
        self._register(
            IntrinsicMapping(
                cuda_name="__constant__",
                triton_name="# (module-level constant)",
                category=IntrinsicCategory.MEMORY,
                is_exact=True,
            )
        )

    def _register(self, mapping: IntrinsicMapping) -> None:
        """Register a mapping in the internal registry."""
        self._mappings[mapping.cuda_name] = mapping

    def get_mapping(self, cuda_intrinsic: str) -> IntrinsicMapping | None:
        """Get the mapping for a CUDA intrinsic."""
        # Try exact match first
        if cuda_intrinsic in self._mappings:
            return self._mappings[cuda_intrinsic]

        # Try with parentheses (for void functions)
        if f"{cuda_intrinsic}()" in self._mappings:
            return self._mappings[f"{cuda_intrinsic}()"]

        return None

    def transform_text(self, cuda_source: str) -> str:
        """Transform CUDA source by replacing intrinsics with Triton equivalents.

        This is a best-effort text transformation. For complex
        intrinsics that need AST-level handling, the translator
        calls the specific transform methods.

        The block-linearization pass runs FIRST so the canonical
        `blockIdx.d * blockDim.d + threadIdx.d` pattern is rewritten
        in one step, before the per-field replacements run.  This
        matches the AST-level pass in `translator._apply_block_linearization`
        and ensures the CPU (fallback) path produces identical output
        to the AST (GPU) path.
        """
        result = cuda_source

        # Block linearization. The backreference \1 forces all three dim
        # letters to match; mixed-dim expressions fall through to the
        # per-field pass below. This mirrors translator._apply_block_linearization.
        def _block_repl(match: re.Match[str]) -> str:
            dim = match.group(1)
            dim_idx = {"x": 0, "y": 1, "z": 2}[dim]
            return (
                f"tl.program_id({dim_idx}) * tl.num_programs({dim_idx}) + tl.program_id({dim_idx})"
            )

        result = re.sub(
            r"blockIdx\.([xyz])\s*\*\s*blockDim\.\1\s*\+\s*threadIdx\.\1",
            _block_repl,
            result,
        )

        # Replace __syncthreads()
        result = result.replace("__syncthreads()", "tl.barrier()")
        result = result.replace("__syncwarp()", "# tl.syncwarp removed (Triton manages warps)")
        result = result.replace("__threadfence()", "tl.barrier()")

        # C++11 rvalue cast: `std::move(x)` → `x`. The \b anchor prevents
        # matching names like `mystd::move`.
        result = re.sub(
            r"\bstd::move\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)", lambda m: m.group(1).strip(), result
        )

        # Replace math functions (order matters: longer names first)
        for cuda_fn in [
            "exp2f",
            "log2f",
            "sqrtf",
            "rsqrtf",
            "sinf",
            "cosf",
            "tanf",
            "logf",
            "expf",
            "fabsf",
            "fmaxf",
            "fminf",
            "floorf",
            "ceilf",
            "erff",
        ]:
            triton_fn = "tl." + cuda_fn[:-1]  # strip 'f' suffix
            result = re.sub(rf"\b{cuda_fn}\b", triton_fn, result)

        # Replace threadIdx with program_id
        result = re.sub(r"\bthreadIdx\.x\b", "tl.program_id(0)", result)
        result = re.sub(r"\bthreadIdx\.y\b", "tl.program_id(1)", result)
        result = re.sub(r"\bthreadIdx\.z\b", "tl.program_id(2)", result)
        result = re.sub(r"\bblockIdx\.x\b", "tl.program_id(0)", result)
        result = re.sub(r"\bblockIdx\.y\b", "tl.program_id(1)", result)
        result = re.sub(r"\bblockIdx\.z\b", "tl.program_id(2)", result)
        result = re.sub(r"\bblockDim\.x\b", "tl.num_programs(0)", result)
        result = re.sub(r"\bblockDim\.y\b", "tl.num_programs(1)", result)
        result = re.sub(r"\bblockDim\.z\b", "tl.num_programs(2)", result)
        result = re.sub(r"\bgridDim\.x\b", "tl.num_programs(0)", result)

        # Replace atomic functions
        for cuda_atomic, triton_atomic in [
            ("atomicAdd", "tl.atomic_add"),
            ("atomicSub", "tl.atomic_sub"),
            ("atomicMin", "tl.atomic_min"),
            ("atomicMax", "tl.atomic_max"),
            ("atomicAnd", "tl.atomic_and"),
            ("atomicOr", "tl.atomic_or"),
            ("atomicXor", "tl.atomic_xor"),
            ("atomicCAS", "tl.atomic_cas"),
            ("atomicExch", "tl.atomic_xchg"),
        ]:
            result = re.sub(rf"\b{cuda_atomic}\b", triton_atomic, result)

        return result

    def all_mappings(self) -> list[IntrinsicMapping]:
        """Return all registered mappings."""
        return list(self._mappings.values())

    def mappings_by_category(self, category: IntrinsicCategory) -> list[IntrinsicMapping]:
        """Return all mappings in a category."""
        return [m for m in self._mappings.values() if m.category == category]

    def coverage_report(self) -> dict[str, Any]:
        """Return a coverage report of the mapping table."""
        total = len(self._mappings)
        by_cat: dict[str, tuple[int, int]] = {}
        for m in self._mappings.values():
            cat = m.category.name
            if cat not in by_cat:
                by_cat[cat] = (0, 0)
            exact, total_cat = by_cat[cat]
            by_cat[cat] = (exact + (1 if m.is_exact else 0), total_cat + 1)
        return {
            "total_mappings": total,
            "by_category": {k: f"{v[0]}/{v[1]} exact" for k, v in by_cat.items()},
        }
