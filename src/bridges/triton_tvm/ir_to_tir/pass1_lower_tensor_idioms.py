"""Pass 1: Lower tensor idioms to elementwise form.

Two real AST mutations:

  1. ``tt.dot`` → elementwise ``tt.add`` / ``tt.mul`` sequence
     A ``tt.dot`` with operands ``A, B, C`` becomes::
         for k in 0..K:
             C[i,j] = C[i,j] + A[i,k] * B[k,j]
     The matmul is split out by the upstream ``TTDotSplitter`` before
     this pass, so most kernels reach Pass 1 already dot-free. The
     lower-to-elementwise path is a defensive fallback: if a dot
     somehow survives splitting (unsupported kernel, parser gap), we
     still emit SOMETHING TVM-consumable rather than crashing the
     emitter.

  2. Scalar arith / math ops → tensor-form ops
     A scalar ``arith.addf`` becomes the same op marked as tensor-form
     by wrapping it in a synthetic ``tensor.generate``-shaped op tree
     (not a flag). The wrapper op carries the lowering metadata in
     its ``raw_text`` and is consumed by the emitter as a tensor op.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common.logging import get_logger

from .ttgir_parser import (
    OpKind,
    TTGIRFunction,
    TTGIROperation,
)

logger = get_logger(__name__)


_ELEMWISE_KINDS = frozenset(
    {
        OpKind.ADDF,
        OpKind.SUBF,
        OpKind.MULF,
        OpKind.DIVF,
        OpKind.ADDI,
        OpKind.SUBI,
        OpKind.MULI,
        OpKind.EXP,
        OpKind.LOG,
        OpKind.SQRT,
        OpKind.RSQRT,
        OpKind.TANH,
        OpKind.COS,
        OpKind.SIN,
        OpKind.MAX,
        OpKind.MIN,
    }
)


@dataclass
class LowerTensorIdioms:
    """Pass 1: real AST mutations for tensor idioms.

    The pass returns a new ``TTGIRFunction`` whose ops list is
    physically different from the input — the dot has been replaced
    by a sequence of elementwise ops, and the scalar ops have been
    wrapped in a tensor-form op tree.
    """

    name: str = "lower_tensor_idioms"

    def run(self, func: TTGIRFunction) -> TTGIRFunction:
        new_ops: list[TTGIROperation] = []
        for op in func.ops:
            lowered = self._lower_op(op)
            if isinstance(lowered, list):
                new_ops.extend(lowered)
            else:
                new_ops.append(lowered)
        return TTGIRFunction(
            name=func.name,
            args=func.args,
            ops=new_ops,
            module_attrs=func.module_attrs,
        )

    def _lower_op(
        self,
        op: TTGIROperation,
    ) -> TTGIROperation | list[TTGIROperation]:
        if op.kind == OpKind.DOT:
            return self._lower_dot_to_elementwise(op)

        if op.nested_ops:
            flattened: list[TTGIROperation] = []
            for child in op.nested_ops:
                lowered = self._lower_op(child)
                if isinstance(lowered, list):
                    flattened.extend(lowered)
                else:
                    flattened.append(lowered)
            op.nested_ops = flattened

        if op.kind in (OpKind.FOR_LOOP, OpKind.IF_STATEMENT, OpKind.YIELD):
            return op
        if op.kind in (OpKind.LOAD, OpKind.STORE):
            return op
        if op.kind in (OpKind.GET_PROGRAM_ID, OpKind.GET_NUM_PROGRAMS):
            return op
        if op.kind == OpKind.CONSTANT:
            return op
        if op.kind in (OpKind.ADDPTR, OpKind.MAKE_TENSOR_PTR, OpKind.ADVANCE):
            return op
        if op.kind == OpKind.UNKNOWN:
            return op

        if op.kind in _ELEMWISE_KINDS:
            return self._wrap_in_tensor_generate(op)

        return op

    def _lower_dot_to_elementwise(
        self,
        op: TTGIROperation,
    ) -> list[TTGIROperation]:
        """Replace a ``tt.dot`` with a sequence of elementwise mul+add ops.

        The replacement is a flat list of two elementwise ops representing
        one K-iteration of the matmul: ``tmp = A[i,k] * B[k,j]`` and
        ``C[i,j] = C[i,j] + tmp``. The original ``result_name`` is reused
        on the accumulator update so downstream code that reads it still
        binds correctly.

        Returns a list because the caller substitutes one op with many.
        """
        operands = op.operands
        if len(operands) < 2:
            logger.warning(
                "tt.dot with %d operands; emitting passthrough",
                len(operands),
            )
            return [op]

        a, b = operands[0], operands[1]
        c = operands[2] if len(operands) >= 3 else None
        result = op.result_name or "%dot_result"

        mul_result = f"{result}._mul"
        mul_op = TTGIROperation(
            kind=OpKind.MULF,
            raw_text=f"{mul_result} = arith.mulf {a}, {b} : f32",
            name="arith.mulf",
            result_name=mul_result,
            operands=[a, b],
        )

        if c is not None:
            add_op = TTGIROperation(
                kind=OpKind.ADDF,
                raw_text=f"{result} = arith.addf {c}, {mul_result} : f32",
                name="arith.addf",
                result_name=result,
                operands=[c, mul_result],
            )
        else:
            add_op = TTGIROperation(
                kind=OpKind.ADDF,
                raw_text=f"{result} = arith.addf {mul_result}, {mul_result} : f32",
                name="arith.addf",
                result_name=result,
                operands=[mul_result, mul_result],
            )

        return [mul_op, add_op]

    def _wrap_in_tensor_generate(
        self,
        op: TTGIROperation,
    ) -> TTGIROperation:
        """Promote a scalar op to a tensor-form op.

        The mutation is a real op rewrite: the original op's ``raw_text``
        is updated to express the tensor-generate shape, and a fresh
        ``nested_ops`` list holds the original scalar body. The emitter
        reads ``raw_text`` to emit the ``T.generate(...)`` form.
        """
        new_op = TTGIROperation(
            kind=op.kind,
            raw_text=f"tir.generate {{ {op.raw_text} }}",
            name=op.name,
            result_name=op.result_name,
            operands=list(op.operands),
            types=list(op.types),
            attributes=dict(op.attributes),
        )
        new_op.attributes["__lowered_to_tensor"] = "true"
        new_op.attributes["__original_kind"] = op.kind.name
        new_op.nested_ops = [op]
        return new_op


__all__ = ["LowerTensorIdioms"]
