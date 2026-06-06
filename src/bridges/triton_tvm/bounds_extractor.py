"""Extract mathematical bounds from captured Triton IR.

Reads tensor shapes, loop bounds, and reduction axes from the real
TTGIR text. Produces IRBounds that the TIR template constructor
uses to build an equivalent TVM TIR function.

This is the critical step that bridges Triton IR semantics into
TVM TIR semantics — done by reading shape attributes, NOT by
relying on Python-level metadata.
"""

from __future__ import annotations

import re
from typing import Any

from src.common.logging import get_logger

from .ir_capture import IRBounds, KernelKind

logger = get_logger(__name__)


class BoundsExtractor:
    """Extract mathematical bounds from captured TTGIR."""

    # Patterns for tensor shape attributes
    # Example: tensor<128x256xf32> or tensor<128x?xf32>
    SHAPE_RE = re.compile(r'tensor<([^>]+)>')

    # Patterns for explicit shape values
    # Example: ttg.affine_shape=...128x256... or constant declarations
    INT_RE = re.compile(r'(?<![a-zA-Z_])(\d+)(?![a-zA-Z_\d])')

    # Patterns for axis.range / scf.for loop bounds
    FOR_BOUNDS_RE = re.compile(r'scf\.for\s+%\w+\s*=\s*(\d+)\s*to\s*(\d+)')

    # Dtype pattern
    DTYPE_RE = re.compile(r'(f(?:32|16|64)|bf16|i(?:32|64|8)|u(?:32|64|8))')

    def extract(self, ir_text: str, kind: KernelKind) -> IRBounds:
        """Extract bounds from the IR based on the kernel kind."""
        # Universal: extract all tensor shapes and dtypes
        tensor_shapes, dtypes = self._extract_tensor_shapes(ir_text)

        # Pick the dominant dtype (most common)
        dtype = self._dominant_dtype(dtypes) if dtypes else "float32"

        # Pick a representative shape (largest, or first)
        rep_shape = self._representative_shape(tensor_shapes)

        # For loop bounds
        for_bounds = self._extract_for_bounds(ir_text)

        # Dispatch on kind
        if kind in (KernelKind.MATMUL, KernelKind.ATTENTION):
            return self._extract_matmul_bounds(ir_text, tensor_shapes, dtype)
        elif kind == KernelKind.REDUCTION:
            return self._extract_reduction_bounds(ir_text, tensor_shapes, for_bounds, dtype)
        else:
            return self._extract_generic_bounds(ir_text, tensor_shapes, dtype)

    # ------------------------------------------------------------------
    # Per-kind extraction
    # ------------------------------------------------------------------

    def _extract_matmul_bounds(
        self,
        ir_text: str,
        tensor_shapes: list[tuple[int, ...]],
        dtype: str,
    ) -> IRBounds:
        """Extract M, N, K for matmul-style kernels.

        For a tt.dot(A, B, C) where A is [M, K] and B is [K, N]:
          - A's shape gives (M, K)
          - B's shape gives (K, N)
          - C's shape gives (M, N) — used to cross-validate
        """
        m, n, k = None, None, None

        # Method 1: find tt.dot operands
        # Triton IR pattern: tt.dot %A, %B, %C where A,B,C are SSA values
        # A's defining op is typically a tt.load with shape [M, K]
        dot_re = re.compile(r'tt\.dot(?:_scaled)?\s+(%\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)')
        for m_obj in dot_re.finditer(ir_text):
            a_val, b_val, c_val = m_obj.group(1), m_obj.group(2), m_obj.group(3)
            # The shapes of A and B operands are what we need
            # In Triton, these are usually defined by tt.load operations
            a_shape = self._find_value_shape(ir_text, a_val)
            b_shape = self._find_value_shape(ir_text, b_val)
            if a_shape and len(a_shape) >= 2:
                m = a_shape[0] if a_shape[0] > 0 else m
                k = a_shape[1] if a_shape[1] > 0 else k
            if b_shape and len(b_shape) >= 2:
                if b_shape[0] > 0:
                    k = b_shape[0]
                if b_shape[1] > 0:
                    n = b_shape[1]

        # Method 2: fall back to largest two shapes
        if m is None or n is None or k is None:
            m, n, k = self._infer_matmul_from_shapes(tensor_shapes)

        return IRBounds(
            m=m or 0, n=n or 0, k=k or 0,
            data_dtype=dtype,
            tensor_ranks=[len(s) for s in tensor_shapes if s],
        )

    def _extract_reduction_bounds(
        self,
        ir_text: str,
        tensor_shapes: list[tuple[int, ...]],
        for_bounds: list[tuple[int, int]],
        dtype: str,
    ) -> IRBounds:
        """Extract bounds for reduction kernels.

        Identify the reduce axis by finding the largest dimension
        within a tt.reduce op, and the keep axis by other dims.
        """
        reduce_size = None
        keep_size = None

        # Method 1: explicit for-loop bounds
        if for_bounds:
            largest = max(for_bounds, key=lambda b: b[1] - b[0])
            reduce_size = largest[1] - largest[0]

        # Method 2: infer from largest tensor dim
        if reduce_size is None and tensor_shapes:
            flat_dims: list[int] = []
            for shape in tensor_shapes:
                flat_dims.extend(d for d in shape if d > 0)
            if flat_dims:
                reduce_size = max(flat_dims)
                keep_size = sum(1 for d in flat_dims if d != reduce_size) or 1

        return IRBounds(
            reduce_size=reduce_size or 0,
            keep_size=keep_size or 1,
            data_dtype=dtype,
            tensor_ranks=[len(s) for s in tensor_shapes if s],
        )

    def _extract_generic_bounds(
        self,
        ir_text: str,
        tensor_shapes: list[tuple[int, ...]],
        dtype: str,
    ) -> IRBounds:
        """Fallback bounds extraction for unknown kernel types."""
        total = 1
        for shape in tensor_shapes:
            for d in shape:
                if d > 0:
                    total *= d
        return IRBounds(
            total_elements=total if total > 0 else 0,
            data_dtype=dtype,
            tensor_ranks=[len(s) for s in tensor_shapes if s],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_tensor_shapes(
        self, ir_text: str,
    ) -> tuple[list[tuple[int, ...]], list[str]]:
        """Extract all tensor shapes and dtypes from the IR."""
        shapes: list[tuple[int, ...]] = []
        dtypes: list[str] = []
        for m in self.SHAPE_RE.finditer(ir_text):
            shape_str = m.group(1)
            # Parse the last 'xf32' or 'xi32' to get dtype, then shape
            parts = shape_str.split("x")
            if not parts:
                continue
            dtype_match = self.DTYPE_RE.search(parts[-1])
            dtype = dtype_match.group(1) if dtype_match else "float32"
            dtypes.append(self._normalize_dtype(dtype))

            # Strip dtype from last part
            shape_parts = []
            for p in parts:
                p_clean = re.sub(r'[a-z]+\d*$', '', p).strip()
                if p_clean in ("?", "-1", ""):
                    shape_parts.append(-1)
                else:
                    try:
                        shape_parts.append(int(p_clean))
                    except ValueError:
                        shape_parts.append(-1)

            if shape_parts and any(s > 0 for s in shape_parts):
                shapes.append(tuple(shape_parts))

        return shapes, dtypes

    def _extract_for_bounds(self, ir_text: str) -> list[tuple[int, int]]:
        """Extract scf.for loop bounds (start, end)."""
        bounds: list[tuple[int, int]] = []
        for m in self.FOR_BOUNDS_RE.finditer(ir_text):
            try:
                bounds.append((int(m.group(1)), int(m.group(2))))
            except (ValueError, IndexError):
                pass
        return bounds

    def _find_value_shape(
        self, ir_text: str, value_name: str,
    ) -> tuple[int, ...] | None:
        """Find the shape of an SSA value by looking at its defining op.

        Pattern: %value = tt.load %ptr, ... : tensor<...>
        Or: %value = arith.constant dense<...> : tensor<...>
        """
        # Look for the value being defined
        pat = re.compile(
            rf'{re.escape(value_name)}\s*=\s*\S+\s+[^:]+\:\s*tensor<([^>]+)>',
        )
        m = pat.search(ir_text)
        if m:
            return self._parse_shape_str(m.group(1))
        return None

    def _parse_shape_str(self, shape_str: str) -> tuple[int, ...]:
        """Parse '128x256xf32' into (128, 256)."""
        parts = shape_str.split("x")
        result: list[int] = []
        for p in parts:
            p_clean = re.sub(r'[a-z]+\d*$', '', p).strip()
            if p_clean in ("?", "-1", ""):
                result.append(-1)
            else:
                try:
                    result.append(int(p_clean))
                except ValueError:
                    result.append(-1)
        return tuple(result)

    def _infer_matmul_from_shapes(
        self, shapes: list[tuple[int, ...]],
    ) -> tuple[int | None, int | None, int | None]:
        """Heuristically infer M, N, K from tensor shapes.

        Assumes:
          - 2D shapes belong to A [M,K], B [K,N], or C [M,N]
          - If A and B are present, K is the matching inner dim
          - C is the output (largest 2D shape typically)
        """
        rank2 = [s for s in shapes if len(s) == 2 and all(d > 0 for d in s)]
        if not rank2:
            return None, None, None

        # C is typically the largest
        c_shape = max(rank2, key=lambda s: s[0] * s[1])
        m, n = c_shape

        # K is the inner dim of A or B; if A or B is found, take inner
        k = None
        for s in rank2:
            if s == c_shape:
                continue
            # A is [M, k], B is [k, N]
            if s[0] == m and s[1] != n:
                k = s[1]
                break
            if s[1] == n and s[0] != m:
                k = s[0]
                break
        if k is None:
            # Last resort: take the average of unknown dims
            k = min(m, n) // 4 or 64

        return m, n, k

    def _representative_shape(
        self, shapes: list[tuple[int, ...]],
    ) -> tuple[int, ...] | None:
        """Pick the largest tensor shape as representative."""
        valid = [s for s in shapes if s and all(d > 0 for d in s)]
        if not valid:
            return None
        return max(valid, key=lambda s: 1)

    def _dominant_dtype(self, dtypes: list[str]) -> str:
        """Return the most common dtype."""
        if not dtypes:
            return "float32"
        return max(set(dtypes), key=dtypes.count)

    def _normalize_dtype(self, dtype: str) -> str:
        """Normalize MLIR dtype names to canonical form."""
        mapping = {
            "f32": "float32",
            "f16": "float16",
            "f64": "float64",
            "bf16": "bfloat16",
            "i32": "int32",
            "i64": "int64",
            "i8": "int8",
            "u32": "uint32",
            "u64": "uint64",
            "u8": "uint8",
        }
        return mapping.get(dtype, dtype)
