"""Pass 4: MaterializeTensorsToTVMBuffers.

Final conversion pass. After Pass 3, ops are in memref form. Pass 4
rewrites them to TVM's tensor/block/buffer form, which is the final
representation that the TVMScript emitter consumes.

Before (memref form):
    %A = memref.alloc() : memref<128x32xf32>
    %val = memref.load %A[%i, %k] : f32
    memref.store %C[%i, %j], %val : f32

After (TVM block form):
    A = T.alloc_buffer((128, 32), dtype="float32")
    with T.block("compute"):
        ... use A[i, k] as direct tensor access ...

The key transformation: a sequence of memref.load + memref.store
becomes a single T.block with the load/store inlined as direct
tensor accesses. This is the form TVM TIR uses everywhere.
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
class MaterializeTensorsToTVM:
    """Pass 4: Convert memref form to TVM block/buffer form.

    This pass:
      1. Inserts T.alloc_buffer at function entry for each memref arg
      2. Wraps load/store sequences in T.block
      3. Converts reduction ops to T.block with reduction axes
      4. Replaces memref arithmetic with direct T.Buffer accesses
    """

    def run(self, func: TTGIRFunction) -> TTGIRFunction:
        """Apply the pass to the function."""
        new_ops = [self._materialize_op(op) for op in func.ops]
        return TTGIRFunction(
            name=func.name,
            args=func.args,
            ops=new_ops,
            module_attrs=func.module_attrs,
        )

    def _materialize_op(self, op: TTGIROperation) -> TTGIROperation:
        """Convert a single op to TVM form."""
        # Recurse first
        if op.nested_ops:
            op.nested_ops = [self._materialize_op(child) for child in op.nested_ops]

        if op.kind in (OpKind.LOAD, OpKind.STORE):
            return self._materialize_access(op)
        if op.kind == OpKind.REDUCE:
            return self._materialize_reduction(op)
        if op.kind == OpKind.BROADCAST:
            return self._materialize_broadcast(op)
        if op.kind == OpKind.RESHAPE:
            return self._materialize_reshape(op)
        if op.kind == OpKind.CONSTANT:
            return self._materialize_constant(op)

        return op

    def _materialize_access(self, op: TTGIROperation) -> TTGIROperation:
        """Mark a load/store op for the emitter to emit as T.block access.

        The emitter reads the __converted_to_memref_load/store flag
        (set in Pass 3) and emits it inside a T.block body.
        """
        op.attributes = dict(op.attributes)
        op.attributes["__materialized_to_tvm_block"] = "true"
        return op

    def _materialize_reduction(self, op: TTGIROperation) -> TTGIROperation:
        """Mark a reduction op for the emitter to emit as T.block with T.init."""
        op.attributes = dict(op.attributes)
        op.attributes["__materialized_to_tvm_reduction"] = "true"
        # Extract the reduction axis from attributes
        if "axis" in op.attributes:
            op.attributes["__tvm_reduction_axis"] = op.attributes["axis"]
        return op

    def _materialize_broadcast(self, op: TTGIROperation) -> TTGIROperation:
        """Mark a broadcast op for the emitter to handle as T.broadcast."""
        op.attributes = dict(op.attributes)
        op.attributes["__materialized_to_tvm_broadcast"] = "true"
        return op

    def _materialize_reshape(self, op: TTGIROperation) -> TTGIROperation:
        """Mark a reshape op for the emitter to handle as T.reshape."""
        op.attributes = dict(op.attributes)
        op.attributes["__materialized_to_tvm_reshape"] = "true"
        return op

    def _materialize_constant(self, op: TTGIROperation) -> TTGIROperation:
        """Mark a constant for the emitter to emit as T.float32(...)."""
        op.attributes = dict(op.attributes)
        op.attributes["__materialized_to_tvm_constant"] = "true"
        return op
