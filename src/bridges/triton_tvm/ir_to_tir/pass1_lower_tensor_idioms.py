"""Pass 1: LowerToTensorIdioms.

Lifts scalar arith/math ops to tensor.generate form. After this pass,
every arithmetic operation is expressed as a tensor-level operation
that operates on whole tensors, not individual elements.

Before:
    %result = arith.addf %a, %b : f32

After:
    %result = tensor.generate %a, %b {
    ^bb0(%a_0: f32, %b_0: f32):
      %tmp = arith.addf %a_0, %b_0 : f32
      tensor.yield %tmp : f32
    } : tensor<...xf32>

This is the foundational transformation — once everything is tensor-
level, subsequent passes can treat operations uniformly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .ttgir_parser import (
    OpKind,
    TTGIRFunction,
    TTGIROperation,
)

logger = logging.getLogger(__name__)


@dataclass
class LowerTensorIdioms:
    """Pass 1: Lift scalar ops to tensor.generate form.

    The pass traverses the function's ops recursively. For each
    scalar op (kind != UNKNOWN, no .nested_ops), it wraps the op
    in a tensor.generate construct with the original op as the
    body. For ops with nested bodies (for/if), the body is
    recursed into and the same treatment applied.

    Operations of kind UNKNOWN are left unchanged — the pipeline
    routes kernels with too many unknowns to the template fallback.
    """

    def run(self, func: TTGIRFunction) -> TTGIRFunction:
        """Apply the pass to the function. Returns the (possibly modified) function."""
        new_ops = [self._lower_op(op) for op in func.ops]
        return TTGIRFunction(
            name=func.name,
            args=func.args,
            ops=new_ops,
            module_attrs=func.module_attrs,
        )

    def _lower_op(self, op: TTGIROperation) -> TTGIROperation:
        """Apply the pass to a single op."""
        # First, recurse into nested ops
        if op.nested_ops:
            op.nested_ops = [self._lower_op(child) for child in op.nested_ops]

        # Skip ops that don't need lowering:
        # - Control flow (handled at the higher level)
        # - Already tensor-level ops
        # - Unknown ops (route to fallback)
        if op.kind in (OpKind.FOR_LOOP, OpKind.IF_STATEMENT, OpKind.YIELD):
            return op
        if op.kind == OpKind.UNKNOWN:
            return op
        if op.kind in (OpKind.LOAD, OpKind.STORE, OpKind.DOT):
            # These are already tensor-level
            return op
        if op.kind in (OpKind.GET_PROGRAM_ID, OpKind.GET_NUM_PROGRAMS):
            # These are scalar values but will be handled by Pass 2
            return op
        if op.kind in (OpKind.CONSTANT,):
            # Constants are inherently scalar
            return op
        if op.kind in (OpKind.ADDPTR, MAKE_TENSOR_PTR_OP, ADVANCE_OP):
            # Pointer ops handled by Pass 3
            return op

        # Wrap the scalar op in a tensor.generate
        return self._wrap_in_tensor_generate(op)

    def _wrap_in_tensor_generate(self, op: TTGIROperation) -> TTGIROperation:
        """Wrap a scalar op in tensor.generate form.

        The wrapped op preserves the original op's name, operands,
        and attributes — the only addition is the tensor.generate
        wrapper semantics.
        """
        # We don't actually mutate the AST in place here — the
        # emitter reads op.name and op.operands and emits the
        # generate form. But we mark the op with a flag that the
        # emitter can check.
        op.attributes = dict(op.attributes)
        op.attributes["__lowered_to_tensor"] = "true"
        op.attributes["__original_kind"] = op.kind.name
        return op


# Marker constants for op kinds that have aliases
MAKE_TENSOR_PTR_OP = OpKind.MAKE_TENSOR_PTR
ADVANCE_OP = OpKind.ADVANCE
