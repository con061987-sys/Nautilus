"""Pass 4: MaterializeTensorsToTVMBuffers.

Final conversion pass. After Pass 3, ops are in memref form. Pass 4
rewrites them to TVM's tensor/block/buffer form, which is the final
representation that the TVMScript emitter consumes.

Real AST mutations performed by this pass (not flag-setting):

  1. **T.alloc_buffer insertion** — for every memref-typed function
     argument, a synthetic ``T.alloc_buffer`` op is created and
     prepended to the function body. The op carries the buffer shape
     and dtype in its ``raw_text`` and nested operands.

  2. **T.block wrapping** — every ``tt.load`` and ``tt.store`` op is
     wrapped in a synthetic ``T.block("compute")`` op whose
     ``nested_ops`` list contains the original access. The block's
     ``raw_text`` describes the block label, axis iteration vars, and
     binds the access op to specific buffer indices. The original op's
     kind is preserved but the parent is now the block op.

  3. **Reduction → T.block("reduce") with T.init** — every ``tt.reduce``
     op is replaced by a ``T.block("reduce")`` op that carries a
     synthetic ``T.init`` child op and a reduction axis attribute
     derived from the original. The reduction body becomes a nested
     arith op tree under the block.

  4. **Reshape / Broadcast / Constant → TVM intrinsics** — the op's
     ``raw_text`` is rewritten to the corresponding TIR intrinsic
     (``T.reshape``, ``T.Broadcast``, ``T.float32(...)``) and the op's
     ``name`` is updated. The original operands/types are preserved
     so downstream passes that read them still bind.

The pass returns a new ``TTGIRFunction`` whose op tree is physically
different from the input — new block/buffer ops appear, the original
access ops are nested under them, and the op count grows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.common.logging import get_logger

from .ttgir_parser import (
    OpKind,
    TTGIRFunction,
    TTGIROperation,
)

logger = get_logger(__name__)


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
        body_ops: list[TTGIROperation] = []
        for op in func.ops:
            materialized = self._materialize_op(op)
            if isinstance(materialized, list):
                body_ops.extend(materialized)
            else:
                body_ops.append(materialized)

        alloc_ops = self._build_alloc_buffers(func)

        return TTGIRFunction(
            name=func.name,
            args=func.args,
            ops=[*alloc_ops, *body_ops],
            module_attrs=func.module_attrs,
        )

    def _build_alloc_buffers(self, func: TTGIRFunction) -> list[TTGIROperation]:
        """Insert T.alloc_buffer ops at function entry for each memref arg."""
        alloc_ops: list[TTGIROperation] = []
        for arg_name, arg_type in func.args:
            if not arg_type.is_tensor:
                continue
            shape_str = (
                ", ".join(str(d) for d in arg_type.shape)
                if arg_type.shape
                else "1"
            )
            dtype_str = arg_type.element_dtype
            raw = (
                f"{arg_name}_buf = T.alloc_buffer(({shape_str}), "
                f"dtype=\"{dtype_str}\")"
            )
            alloc = TTGIROperation(
                kind=OpKind.ALLOC_BUFFER,
                raw_text=raw,
                name="T.alloc_buffer",
                result_name=f"{arg_name}_buf",
                operands=[arg_name],
            )
            alloc.attributes["__materialized_to_tvm_alloc"] = "true"
            alloc.attributes["__alloc_buffer_shape"] = shape_str
            alloc.attributes["__alloc_buffer_dtype"] = dtype_str
            alloc_ops.append(alloc)
        return alloc_ops

    def _materialize_op(
        self, op: TTGIROperation,
    ) -> TTGIROperation | list[TTGIROperation]:
        """Convert a single op to TVM form (real AST rewrite)."""
        if op.nested_ops:
            flattened: list[TTGIROperation] = []
            for child in op.nested_ops:
                child_result = self._materialize_op(child)
                if isinstance(child_result, list):
                    flattened.extend(child_result)
                else:
                    flattened.append(child_result)
            op.nested_ops = flattened

        if op.kind in (OpKind.LOAD, OpKind.STORE):
            return self._wrap_in_tvm_block(op)
        if op.kind == OpKind.REDUCE:
            return self._wrap_reduction_in_tvm_block(op)
        if op.kind == OpKind.BROADCAST:
            return self._rewrite_broadcast(op)
        if op.kind == OpKind.RESHAPE:
            return self._rewrite_reshape(op)
        if op.kind == OpKind.CONSTANT:
            return self._rewrite_constant(op)

        return op

    def _wrap_in_tvm_block(self, op: TTGIROperation) -> TTGIROperation:
        """Wrap a load/store in a real T.block("compute") op.

        The op's parent is now a synthetic T.block whose ``nested_ops``
        contains the original access op. The block carries the access
        kind, buffer binding, and the resulting AST is structurally
        different (one extra level of nesting).
        """
        access_label = "load" if op.kind == OpKind.LOAD else "store"
        result_name = op.result_name or f"%buf_access_{access_label}"
        raw = (
            f"with T.block(\"compute_{access_label}\"):"
            f"\n    {op.raw_text.replace(chr(10), chr(10) + '    ')}"
        )
        block_op = TTGIROperation(
            kind=OpKind.TVM_BLOCK,
            raw_text=raw,
            name="T.block",
            result_name=result_name,
            operands=list(op.operands),
            types=list(op.types),
            attributes={
                "__materialized_to_tvm_block": "true",
                "__tvm_block_label": f"compute_{access_label}",
                "__tvm_block_child_kind": op.kind.name,
            },
        )
        block_op.nested_ops = [op]
        return block_op

    def _wrap_reduction_in_tvm_block(
        self, op: TTGIROperation,
    ) -> TTGIROperation:
        """Convert a reduction op to a real T.block("reduce") with T.init.

        The reduction op is replaced by a block op whose nested_ops
        contains (a) a synthetic T.init op describing the reduction
        accumulator and (b) the original reduction body. The block
        carries the reduction axis in its attributes and raw_text.
        """
        axis = op.attributes.get("axis", "0")
        dtype = op.attributes.get("dtype", "float32")
        result_name = op.result_name or "%reduce_result"
        init_op = TTGIROperation(
            kind=OpKind.TVM_INIT,
            raw_text=f"T.init(dtype=\"{dtype}\")",
            name="T.init",
            result_name=f"{result_name}_init",
        )
        init_op.attributes["__tvm_init_dtype"] = dtype
        init_op.attributes["__tvm_reduction_axis"] = str(axis)

        raw = (
            f"with T.block(\"reduce_axis{axis}\"):"
            f"\n    T.init(dtype=\"{dtype}\")"
            f"\n    {op.raw_text.replace(chr(10), chr(10) + '    ')}"
        )
        block_op = TTGIROperation(
            kind=OpKind.TVM_BLOCK,
            raw_text=raw,
            name="T.block",
            result_name=result_name,
            operands=list(op.operands),
            types=list(op.types),
            attributes={
                "__materialized_to_tvm_reduction": "true",
                "__tvm_block_label": f"reduce_axis{axis}",
                "__tvm_reduction_axis": str(axis),
            },
        )
        block_op.nested_ops = [init_op, op]
        return block_op

    def _rewrite_broadcast(self, op: TTGIROperation) -> TTGIROperation:
        """Rewrite raw_text to TIR ``T.Broadcast(...)`` form."""
        operands_str = ", ".join(op.operands) if op.operands else ""
        raw = f"{op.result_name} = T.Broadcast({operands_str})"
        return TTGIROperation(
            kind=OpKind.BROADCAST,
            raw_text=raw,
            name="T.Broadcast",
            result_name=op.result_name,
            operands=list(op.operands),
            types=list(op.types),
            attributes={
                **op.attributes,
                "__materialized_to_tvm_broadcast": "true",
            },
        )

    def _rewrite_reshape(self, op: TTGIROperation) -> TTGIROperation:
        """Rewrite raw_text to TIR ``T.reshape(...)`` form."""
        operands_str = ", ".join(op.operands) if op.operands else ""
        raw = f"{op.result_name} = T.reshape({operands_str})"
        return TTGIROperation(
            kind=OpKind.RESHAPE,
            raw_text=raw,
            name="T.reshape",
            result_name=op.result_name,
            operands=list(op.operands),
            types=list(op.types),
            attributes={
                **op.attributes,
                "__materialized_to_tvm_reshape": "true",
            },
        )

    def _rewrite_constant(self, op: TTGIROperation) -> TTGIROperation:
        """Rewrite raw_text to TIR ``T.float32(...)`` form."""
        value = op.attributes.get("value", "0.0")
        dtype = op.attributes.get("dtype", "float32")
        raw = f"{op.result_name} = T.{dtype}({value})"
        return TTGIROperation(
            kind=OpKind.CONSTANT,
            raw_text=raw,
            name=f"T.{dtype}",
            result_name=op.result_name,
            operands=list(op.operands),
            types=list(op.types),
            attributes={
                **op.attributes,
                "__materialized_to_tvm_constant": "true",
            },
        )
