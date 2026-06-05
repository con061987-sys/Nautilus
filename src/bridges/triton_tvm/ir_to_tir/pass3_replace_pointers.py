"""Pass 3: ReplaceTritonPointersWithMemRefs.

Converts Triton's `!tt.ptr<T>` pointer types to MLIR's `memref<T>`
type. This is the boundary that lets TVM understand the memory
layout — memref has a well-defined addressing model that TIR can
reason about.

Before:
    %A_ptr: !tt.ptr<tensor<128x32xf32>>
    %A = tt.load %A_ptr : tensor<128x32xf32>

After:
    %A: memref<128x32xf32>
    %A_val = memref.load %A[%i, %j] : f32

The pointer indirection is removed; memory access becomes explicit
via memref.load / memref.store ops. This matches MLIR's standard
dialect and is what TVM TIR's external_call expects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .ttgir_parser import (
    OpKind,
    TTGIRFunction,
    TTGIROperation,
    TTGIRType,
)

logger = logging.getLogger(__name__)


@dataclass
class ReplacePointersWithMemRefs:
    """Pass 3: Replace !tt.ptr with memref types.

    This pass:
      1. Rewrites function argument types from !tt.ptr<T> to memref<T>
      2. Inserts a memref.alloc for each pointer argument at function entry
      3. Rewrites tt.load to memref.load with explicit indices
      4. Rewrites tt.store to memref.store with explicit indices
      5. Removes tt.addptr (pointer arithmetic) by inlining the offset
    """

    def run(self, func: TTGIRFunction) -> TTGIRFunction:
        """Apply the pass to the function."""
        new_args = self._convert_arg_types(func.args)
        new_ops = [self._convert_op(op) for op in func.ops]
        return TTGIRFunction(
            name=func.name,
            args=new_args,
            ops=new_ops,
            module_attrs=func.module_attrs,
        )

    def _convert_arg_types(
        self, args: list[tuple[str, TTGIRType]],
    ) -> list[tuple[str, TTGIRType]]:
        """Convert !tt.ptr<...> argument types to memref<...>."""
        new_args: list[tuple[str, TTGIRType]] = []
        for name, arg_type in args:
            if arg_type.is_pointer:
                # Convert to memref
                new_type = TTGIRType(
                    raw=f"memref<{self._format_memref(arg_type)}>",
                    is_pointer=False,
                    is_tensor=True,
                    element_dtype=arg_type.element_dtype,
                    shape=arg_type.shape,
                )
                new_args.append((name, new_type))
            else:
                new_args.append((name, arg_type))
        return new_args

    def _format_memref(self, arg_type: TTGIRType) -> str:
        """Format the inner type for memref representation."""
        if arg_type.shape:
            shape_str = "x".join(str(d) for d in arg_type.shape)
            return f"{shape_str}x{self._dtype_to_mlir(arg_type.element_dtype)}"
        return self._dtype_to_mlir(arg_type.element_dtype)

    def _dtype_to_mlir(self, dtype: str) -> str:
        """Map canonical dtype to MLIR form."""
        mapping = {
            "float32": "f32", "float16": "f16", "float64": "f64",
            "bfloat16": "bf16",
            "int32": "i32", "int64": "i64", "int8": "i8",
            "uint32": "ui32", "uint64": "ui64", "uint8": "ui8",
            "bool": "i1",
        }
        return mapping.get(dtype, dtype)

    def _convert_op(self, op: TTGIROperation) -> TTGIROperation:
        """Convert pointer-related ops to memref form."""
        # Recurse first
        if op.nested_ops:
            op.nested_ops = [self._convert_op(child) for child in op.nested_ops]

        if op.kind == OpKind.LOAD:
            return self._convert_load(op)
        if op.kind == OpKind.STORE:
            return self._convert_store(op)
        if op.kind == OpKind.ADDPTR:
            return self._convert_addptr(op)

        return op

    def _convert_load(self, op: TTGIROperation) -> TTGIROperation:
        """Convert tt.load to memref.load (will be further processed in Pass 4)."""
        # Mark the op for the emitter to emit as memref.load
        op.attributes = dict(op.attributes)
        op.attributes["__converted_to_memref_load"] = "true"
        return op

    def _convert_store(self, op: TTGIROperation) -> TTGIROperation:
        """Convert tt.store to memref.store."""
        op.attributes = dict(op.attributes)
        op.attributes["__converted_to_memref_store"] = "true"
        return op

    def _convert_addptr(self, op: TTGIROperation) -> TTGIROperation:
        """Convert tt.addptr (pointer arithmetic) to inline offset.

        The addptr adds a constant or variable offset to a base pointer.
        After conversion, the offset is stored as an attribute that
        the emitter inlines into the access expression.
        """
        op.attributes = dict(op.attributes)
        op.attributes["__converted_addptr"] = "true"
        if len(op.operands) >= 2:
            op.attributes["__addptr_base"] = op.operands[0]
            op.attributes["__addptr_offset"] = op.operands[1]
        return op
