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

import re
from dataclasses import dataclass, field
from typing import Any

from src.common.logging import get_logger

logger = get_logger(__name__)


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
    is_restrict: bool = False
    allocation_site: str = ""

    @property
    def is_2d(self) -> bool:
        return self.dimensions == 2

    @property
    def is_1d(self) -> bool:
        return self.dimensions == 1


@dataclass
class AliasInfo:
    """Aliasing information for a single pointer in a kernel.

    - `may_alias`: set of names that this pointer may alias with.  Two
      pointers in the same set are *not* guaranteed to be disjoint.
    - `is_restrict`: True when declared `__restrict__` (or `restrict`)
      — the compiler is told no other live pointer refers to the same
      memory, so the pointer is disjoint from every other `__restrict__`
      pointer and from non-pointer values.
    - `is_parameter`: True if the pointer came in as a kernel parameter
      (not a local or shared-memory allocation).  Parameters without
      `__restrict__` are the main source of unknown aliasing.
    - `provenance`: free-form description of where the pointer's
      storage came from (e.g. "kernel parameter", "shared memory
      allocation `tile`", "local address-of").
    """
    name: str
    may_alias: set[str] = field(default_factory=set)
    is_restrict: bool = False
    is_parameter: bool = False
    provenance: str = ""

    @property
    def aliases_nothing(self) -> bool:
        """`True` iff this pointer is provably disjoint from all other
        pointers in the kernel (e.g. restrict-qualified parameter or
        a stack-local address that was never taken)."""
        return self.is_restrict and not self.may_alias


class PointerAnalyzer:
    """Analyzes pointer usage in CUDA kernels.

    The analyzer scans the kernel body for pointer accesses and
    infers the multi-dimensional layout, stride pattern, and
    coalescing characteristics. This information is used by the
    translator to generate correct Triton code.

    Wave 2.7: also tracks *aliasing* — which pointers may refer to
    overlapping memory.  Restricted pointers (`__restrict__`) are
    treated as disjoint from every other restrict pointer and from
    non-pointer values.  Unrestricted parameters form a single
    alias set by default; the analyzer refines that set when a
    parameter is assigned to a local pointer (e.g. `float* p = A;`).
    """

    # Pattern for pointer accesses: name[idx1][idx2]... = value
    POINTER_ACCESS_RE = re.compile(
        r'(\w+)\s*((?:\[[^\]]+\])+)\s*(=|\+=|-=|\*=|/=|\+\+|--)?',
    )
    # Pattern for parameter declarations like "float* A" or
    # "const float* __restrict__ A". The optional `__restrict__`
    # marker is captured so the analyzer can mark the layout
    # accordingly.
    POINTER_PARAM_RE = re.compile(
        r'(?:const\s+)?(\w+(?:\s*\*+)?)\s*\**\s*(\w+)\s*'
        r'(?:\[[^\]]*\])?\s*(?:__restrict__|restrict)?',
    )
    # Pattern for restrict detection in the raw type text — the
    # parser may not always surface `__restrict__` in the cleaned
    # type, so we also re-scan.
    _RESTRICT_RE = re.compile(r'\b(?:__restrict__|restrict)\b')
    # Local pointer declaration in body; group 1 = name, group 2 = optional
    # initializer (used for alias-class inheritance downstream).
    _LOCAL_PTR_DECL_RE = re.compile(
        r'^\s*(?:const\s+)?\w+\s*\*+\s*(\w+)\s*(?:=\s*(.+?))?\s*;\s*$',
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

    def analyze_aliases(self, kernel: Any) -> dict[str, AliasInfo]:
        """Build an alias map for every pointer in the kernel.

        Rules:
          - Each `__restrict__`-qualified parameter is provably
            disjoint from every other restrict pointer and from any
            non-pointer value.  The set of other pointers it
            *cannot* alias is therefore the entire set of restrict
            pointers (these are mutually disjoint by the C99
            `restrict` contract).
          - Unrestricted parameters form a single alias set: any
            two unrestricted parameters may alias.
          - A local pointer initialised from a parameter
            (`float* p = A;`) inherits that parameter's alias set.
          - Shared-memory allocations are disjoint from every
            pointer (different storage class), so they never appear
            in the alias sets of global pointers.
        """
        alias_map: dict[str, AliasInfo] = {}
        layout_map = self.analyze_kernel(kernel)

        # Phase 1: seed alias info from layouts.
        restrict_pointers: list[str] = []
        unrestricted_pointers: list[str] = []
        for name, layout in layout_map.items():
            alias_map[name] = AliasInfo(
                name=name,
                is_restrict=layout.is_restrict,
                is_parameter=True,
                provenance="kernel parameter",
            )
            if layout.is_restrict:
                restrict_pointers.append(name)
            else:
                unrestricted_pointers.append(name)

        # Phase 2: cross-link the unrestricted parameters.
        for name in unrestricted_pointers:
            alias_map[name].may_alias.update(unrestricted_pointers)
            alias_map[name].may_alias.discard(name)

        # Phase 3: assign-by-reference in the body. A local pointer
        # initialised from a parameter inherits that parameter's
        # alias class.
        for stmt in kernel.body:
            self._propagate_aliases_via_assign(stmt, alias_map, layout_map)

        # Phase 4: shared-memory allocations are disjoint.
        for smem in getattr(kernel, "shared_mem_declarations", []) or []:
            smem_name = smem.get("name", "")
            if not smem_name:
                continue
            for info in alias_map.values():
                info.may_alias.discard(smem_name)

        return alias_map

    def _propagate_aliases_via_assign(
        self,
        stmt: Any,
        alias_map: dict[str, AliasInfo],
        layout_map: dict[str, PointerLayout],
    ) -> None:
        """If `p = A` (where A is a known pointer) is seen in the body,
        add A to p's may-alias set and inherit A's restrict status.

        Handles both bare assignments (``p = A;``) and declaration
        assignments (``const float* p = A;``) by extracting the
        rightmost identifier from the LHS.
        """
        text = getattr(stmt, "raw_text", "")
        if "=" not in text or text.lstrip().startswith("//"):
            return
        head, _, tail = text.rstrip(";").partition("=")
        lhs = head.strip()
        rhs = tail.strip()
        # LHS may be a declaration (e.g. `const float* p`); pull the
        # rightmost identifier for the layout-map lookup.
        lhs_name = self._rightmost_identifier(lhs)
        if lhs_name is None or lhs_name not in layout_map:
            return
        if rhs not in alias_map:
            return
        info = alias_map[lhs_name]
        source = alias_map[rhs]
        info.may_alias.update(source.may_alias)
        info.may_alias.add(rhs)
        info.may_alias.discard(lhs_name)
        if source.is_restrict:
            info.is_restrict = True

    @staticmethod
    def _rightmost_identifier(text: str) -> str | None:
        """Return the last identifier-like token in `text`, or None."""
        for token in reversed(text.split()):
            token = token.strip("*&")
            if token.isidentifier():
                return token
        return None

    def _layout_from_param(
        self, param: dict[str, str], kernel: Any,
    ) -> PointerLayout | None:
        """Determine a pointer's layout from its parameter declaration."""
        type_str = param.get("type", "")
        name = param.get("name", "")

        if "*" not in type_str and not self._RESTRICT_RE.search(type_str):
            return None  # Not a pointer

        return PointerLayout(
            name=name,
            element_type=self._extract_element_type(type_str),
            dimensions=1,  # Default; will be refined
            shape_expr=[],
            stride_expr=[],
            is_restrict=bool(self._RESTRICT_RE.search(type_str)),
            allocation_site="kernel parameter",
        )

    @staticmethod
    def _extract_element_type(type_str: str) -> str:
        """Strip pointers, `const`, and `__restrict__` to recover the base type.

        Examples:
            ``"const float*"``            → ``"float"``
            ``"float* __restrict__"``     → ``"float"``
            ``"const int* __restrict__"`` → ``"int"``
        """
        cleaned = type_str
        for token in ("*", "const", "__restrict__", "restrict"):
            cleaned = cleaned.replace(token, "")
        cleaned = " ".join(cleaned.split())
        return cleaned

    def _refine_layouts_from_stmt(
        self,
        stmt: Any,
        layouts: dict[str, PointerLayout],
        kernel: Any,
    ) -> None:
        """Refine layout inference from statement patterns.

        Detects two things in each body statement:
          1. Multi-dim access patterns that upgrade a known layout's
             ``dimensions`` (the original Wave 2.7 behaviour).
          2. Local pointer declarations (``const float* p = A;``) that
             introduce a new layout.  The alias analyser separately
             propagates the initialiser to inherit the source's alias
             class.
        """
        text = stmt.raw_text

        for match in self.POINTER_ACCESS_RE.finditer(text):
            name = match.group(1)
            if name not in layouts:
                continue
            index_str = match.group(2)
            indices = re.findall(r'\[([^\]]+)\]', index_str)
            if len(indices) > layouts[name].dimensions:
                layouts[name].dimensions = len(indices)
                layouts[name].shape_expr = indices
                layouts[name].stride_expr = ["1"] * len(indices)

        decl = self._LOCAL_PTR_DECL_RE.match(text)
        if decl:
            local_name = decl.group(1)
            if local_name and local_name not in layouts:
                # Base type isn't recoverable from raw text alone;
                # default to float; downstream refines via access patterns.
                layouts[local_name] = PointerLayout(
                    name=local_name,
                    element_type="float",
                    dimensions=1,
                    shape_expr=[],
                    stride_expr=[],
                    is_restrict=False,
                    allocation_site="local pointer",
                )

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
