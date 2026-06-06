"""IR classification — pattern-matches Triton ops in captured TTGIR.

Determines what kind of kernel we're looking at by inspecting the ops
in the captured IR. This drives which TIR template we construct
and how we map the MetaSchedule result back to Triton.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .ir_capture import KernelKind


class IRClassifier:
    """Classify captured TTGIR by inspecting its op structure."""

    # Op regex patterns
    OP_RE = re.compile(r'tt\.(\w+)|ttg\.(\w+)|arith\.(\w+)|math\.(\w+)|scf\.(\w+)|gpu\.(\w+)')
    TENSOR_TYPE_RE = re.compile(
        r'tensor<([^>]*),\s*#ttg\.(\w+)|memref<([^>]*),\s*#ttg\.(\w+)|'
        r'tensor<([^>]+)>'
    )

    # Op signatures
    MATMUL_OPS = {"dot", "dot_scaled"}
    REDUCTION_OPS = {"reduce", "sum", "max", "min", "argmax", "argmin"}
    ELEMENTWISE_OPS = {
        "addf", "subf", "mulf", "divf", "fma", "fmul", "fadd",
        "addi", "subi", "muli",
        "exp", "log", "sqrt", "rsqrt", "abs", "neg", "maxf", "minf",
        "sigmoid", "tanh", "cos", "sin",
    }
    MEMORY_OPS = {"load", "store", "atomic_rmw", "atomic_cas"}
    SCAN_OPS = {"scan"}
    PERSISTENT_OPS = {"while", "for"}

    def classify(self, ir_text: str) -> KernelKind:
        """Determine the kernel kind from the captured IR text.

        Returns KernelKind.UNKNOWN if the IR doesn't match any
        well-known pattern.
        """
        ops = self.collect_ops(ir_text)

        # Check for matmul first (most specific)
        if any(op in self.MATMUL_OPS for op in ops):
            # Check for attention pattern: dot, then reduce (softmax), then dot
            has_softmax = any(op in self.REDUCTION_OPS for op in ops)
            has_2nd_dot = self._count_op(ops, "dot") >= 2
            if has_softmax and has_2nd_dot:
                return KernelKind.ATTENTION
            return KernelKind.MATMUL

        # Check for scan
        if any(op in self.SCAN_OPS for op in ops):
            return KernelKind.SCAN

        # Check for persistent kernel (scf.while with no exit)
        if "while" in ops:
            return KernelKind.PERSISTENT

        # Check for reduction-dominant
        reduction_count = sum(1 for op in ops if op in self.REDUCTION_OPS)
        if reduction_count >= 1 and reduction_count >= len(ops) * 0.3:
            return KernelKind.REDUCTION

        # Check for elementwise (memory ops with arith, no reduction)
        if "load" in ops and "store" in ops and not reduction_count:
            return KernelKind.ELEMENTWISE

        return KernelKind.UNKNOWN

    def collect_ops(self, ir_text: str) -> list[str]:
        """Collect all op names from the IR in order of appearance.

        Each occurrence is preserved so that callers can count repeats
        (e.g. attention kernels have multiple tt.dot ops). Use
        `collect_op_counts()` for deduplicated counts.
        """
        ops: list[str] = []
        for m in self.OP_RE.finditer(ir_text):
            op = m.group(1) or m.group(2) or m.group(3) or m.group(4) or m.group(5) or m.group(6)
            if op:
                ops.append(op)
        return ops

    def collect_op_counts(self, ir_text: str) -> Counter:
        """Count occurrences of each op."""
        return Counter(self.collect_ops(ir_text))

    def collect_tensor_types(self, ir_text: str) -> list[tuple[tuple[int, ...], str]]:
        """Extract all tensor type definitions and their shapes.

        Returns list of (shape, dtype) tuples.
        """
        results: list[tuple[tuple[int, ...], str]] = []
        for m in self.TENSOR_TYPE_RE.finditer(ir_text):
            shape_str = m.group(1) or m.group(3) or m.group(5) or ""
            shape = self._parse_shape(shape_str)
            dtype = "float32"  # default; extract from encoding if available
            results.append((shape, dtype))
        return results

    def _count_op(self, ops: list[str], op_name: str) -> int:
        """Count occurrences of an op (including re-entry into patterns)."""
        return sum(1 for o in ops if o == op_name)

    def _parse_shape(self, shape_str: str) -> tuple[int, ...]:
        """Parse a shape string like '128x256' or '?x?' into a tuple.

        Unknown dimensions ('?') are returned as -1.
        """
        shape_str = shape_str.strip()
        if not shape_str:
            return ()
        parts = shape_str.split("x")
        result: list[int] = []
        for p in parts:
            p = p.strip()
            if p in ("?", "-1"):
                result.append(-1)
            else:
                try:
                    result.append(int(p))
                except ValueError:
                    result.append(-1)
        return tuple(result)
