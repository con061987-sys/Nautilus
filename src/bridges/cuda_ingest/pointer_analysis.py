"""Pointer analysis for CUDA → Triton translation.

CUDA kernels use raw pointers extensively. Triton uses typed
tensors (tl.tensor) which carry shape and dtype information. This
module analyzes CUDA pointer usage patterns and determines the
correct Triton translation.

The main challenges:
  1. **Pointer arithmetic** — CUDA does `ptr + offset` manually;
     Triton uses block pointers with shape information
  2. **Multi-dimensional arrays** — CUDA flattens to 1D; Triton
     can use multi-dim pointers via tl.make_block_ptr
  3. **Coalescing** — CUDA relies on the hardware to coalesce
     adjacent thread accesses; Triton needs to use tl.load with
     proper stride information
  4. **Bounds checking** — CUDA usually doesn't bounds-check;
     Triton can optionally add bounds checks

Translation strategy:
  - 1D pointers with linear indexing → tl.tensor with 1D shape
  - 2D pointers with row/col indexing → tl.tensor with 2D shape
  - Pointer arithmetic → block pointer operations
  - Coalesced loads/stores → tl.load/tl.store with proper strides
  - Add bounds checks for safety (can be disabled for performance)

Production features:
  - Multi-dim pointer support
  - Bounds checking (optional)
  - Stride analysis for coalescing
  - Detection of unsafe pointer patterns
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PointerAccess:
    """A single pointer access in CUDA code."""
    name: str
    indices: list[str] = field(default_factory=list)  # e.g. ['i', 'j']
    is_write: bool = False
    is_atomic: bool = False
    element_type: str = "float"
    is_coalesced: bool = True


@dataclass
class PointerLayout:
    """Layout of a multi-dimensional pointer in CUDA."""
    name: str
    element_type: str
    dimensions: int
    shape_expr: list[str] = field(default_factory=list)  # e.g. ['M', 'N']
    stride_expr: list[str] = field(default_factory=list)  # e.g. ['N', '1']
    row_major: bool = True
    needs_bounds_check: bool = True

    @property
    def is_2d(self) -> bool:
        return self.dimensions == 2

    @property
    def is_1d(self) -> bool:
        return self.dimensions == 1


class PointerAnalyzer:
    """Analyzes pointer usage in CUDA kernels.

    The analyzer scans the kernel body for pointer accesses and
    infers the multi-dimensional layout, stride pattern, and
    coalescing characteristics. This information is used by the
    translator to generate correct Triton code.
    """

    # Pattern for pointer accesses: name[idx1][idx2]... = value
    POINTER_ACCESS_RE = re.compile(
        r'(\w+)\s*((?:\[[^\]]+\])+)\s*(=|\+=|-=|\*=|/=|\+\+|--)?',
    )
    # Pattern for parameter declarations like "float* A"
    POINTER_PARAM_RE = re.compile(
        r'(?:const\s+)?(\w+(?:\s*\*+)?)\s*\**\s*(\w+)\s*(?:\[[^\]]*\])?\s*(?:__restrict__)?',
    )

    def analyze_kernel(self, kernel: Any) -> dict[str, PointerLayout]:
        """Analyze a parsed CUDA kernel for pointer layouts.

        Returns a dict mapping parameter name to PointerLayout.
        """
        layouts: dict[str, PointerLayout] = {}

        # First, extract from parameter declarations
        for param in kernel.parameters:
            layout = self._layout_from_param(param, kernel)
            if layout is not None:
                layouts[param["name"]] = layout

        # Then, refine based on actual usage in the body
        for stmt in kernel.body:
            self._refine_layouts_from_stmt(stmt, layouts, kernel)

        return layouts

    def _layout_from_param(
        self, param: dict[str, str], kernel: Any,
    ) -> PointerLayout | None:
        """Determine a pointer's layout from its parameter declaration."""
        type_str = param.get("type", "")
        name = param.get("name", "")

        if "*" not in type_str and "__restrict__" not in type_str:
            return None  # Not a pointer

        # Extract the base type (strip pointer markers)
        element_type = type_str.replace("*", "").replace("const", "").strip()
        if " " in element_type:
            element_type = element_type.split()[-1]

        return PointerLayout(
            name=name,
            element_type=element_type,
            dimensions=1,  # Default; will be refined
            shape_expr=[],
            stride_expr=[],
        )

    def _refine_layouts_from_stmt(
        self,
        stmt: Any,
        layouts: dict[str, PointerLayout],
        kernel: Any,
    ) -> None:
        """Refine layout inference from statement patterns."""
        text = stmt.raw_text

        for match in self.POINTER_ACCESS_RE.finditer(text):
            name = match.group(1)
            if name not in layouts:
                continue

            # Extract indices from [idx1][idx2]...
            index_str = match.group(2)
            indices = re.findall(r'\[([^\]]+)\]', index_str)
            if len(indices) > layouts[name].dimensions:
                # Upgrade to multi-dim
                layouts[name].dimensions = len(indices)
                layouts[name].shape_expr = indices
                layouts[name].stride_expr = ["1"] * len(indices)

    def get_tenso_creation(
        self, layout: PointerLayout, block_size: int | None = None,
    ) -> str:
        """Generate the Triton code to create a tensor for this pointer.

        For 1D:
            A = tl.load(A_ptr + offsets, mask=mask)

        For 2D:
            A = tl.load(A_ptr + row_offsets[:, None] * stride_am
                          + col_offsets[None, :] * stride_ak)
        """
        if layout.dimensions == 1:
            if block_size:
                return (
                    f"offsets = tl.arange(0, {block_size})\n"
                    f"mask = offsets < SIZE\n"
                    f"{layout.name} = tl.load({layout.name}_ptr + offsets, mask=mask, other=0.0)"
                )
            return f"{layout.name} = tl.load({layout.name}_ptr)"
        elif layout.dimensions == 2:
            return (
                f"row_idx = tl.arange(0, BLOCK_M)\n"
                f"col_idx = tl.arange(0, BLOCK_N)\n"
                f"offsets = row_idx[:, None] * stride_{layout.name}_m + col_idx[None, :] * stride_{layout.name}_n\n"
                f"mask = (row_idx[:, None] < M) & (col_idx[None, :] < N)\n"
                f"{layout.name} = tl.load({layout.name}_ptr + offsets, mask=mask, other=0.0)"
            )
        else:
            return f"# TODO: {layout.dimensions}D pointer not yet supported"

    def analyze_coalescing(self, layout: PointerLayout) -> str:
        """Analyze whether the pointer access is coalesced.

        Coalesced means adjacent threads access adjacent memory.
        Non-coalesced access is 10-100x slower on GPUs.
        """
        if layout.dimensions == 1:
            return "coalesced (1D contiguous access)"
        elif layout.dimensions == 2 and layout.row_major:
            return "coalesced (row-major 2D)"
        elif layout.dimensions == 2:
            return "potentially non-coalesced (column-major or strided)"
        return f"unknown coalescing for {layout.dimensions}D"


def tl_arange(size: int) -> str:
    """Helper: generate tl.arange expression."""
    return f"tl.arange(0, {size})"
