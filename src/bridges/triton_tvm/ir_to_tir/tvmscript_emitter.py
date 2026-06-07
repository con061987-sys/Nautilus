"""TVMScript emitter.

Converts the converted AST (output of Pass 4) into TVMScript text —
the Python-embedded TIR DSL that TVM's MetaSchedule can directly
consume via `tvm.script.tir.prim_func` and `tvm.meta_schedule.tune_tir`.

This is the ONLY module that produces TVM-specific output. All the
conversion passes produce a generic AST; the emitter is the boundary
that knows TVM's syntax.

The emitter uses T.script formatting rather than constructing an
IRModule directly — this is more robust and easier to debug, and it
matches what TVM's example kernels do.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common.logging import get_logger

from .ttgir_parser import (
    OpKind,
    TTGIRFunction,
    TTGIROperation,
    TTGIRType,
)

logger = get_logger(__name__)


@dataclass
class TVMScriptEmitter:
    """Emits TVMScript text from a converted TTGIRFunction.

    The emitter walks the function's ops and produces TVMScript lines.
    The output is a string of Python code that, when wrapped in
    `tvm.script.tir.prim_func`, produces a valid TIR PrimFunc.

    Key design decisions:
      - Use Python f-strings for readability and debuggability
      - Emit one block at a time with consistent indentation
      - Keep dtype names in canonical Python form
      - Skip ops that are not materializable (UNKNOWN kind)
    """

    def emit(self, func: TTGIRFunction) -> str:
        """Emit TVMScript text for the function.

        Returns a string of Python code that can be wrapped in
        `@tvm.script.tir.prim_func`.
        """
        lines: list[str] = []

        # Emit T.alloc_buffer for each function argument
        lines.append("@T.prim_func")
        lines.append(f"def {func.name}({self._emit_signature(func)}):")
        lines.append("    # Function body — emitted from real Triton IR")
        lines.append("")

        # Emit T.alloc_buffer for each pointer/tensor argument
        buffer_decls = self._emit_buffer_declarations(func)
        if buffer_decls:
            lines.extend(buffer_decls)
            lines.append("")

        # Emit the op body
        for op in func.ops:
            lines.extend(self._emit_op(op, indent=1))
            lines.append("")

        # Emit tt.return
        lines.append("    T.evaluate(0)  # terminator")

        return "\n".join(lines)

    def _emit_signature(self, func: TTGIRFunction) -> str:
        """Emit the function signature: arg1: T.Buffer(...), arg2: ..."""
        parts: list[str] = []
        for name, arg_type in func.args:
            parts.append(self._emit_arg(name, arg_type))
        return ", ".join(parts) if parts else ""

    def _emit_arg(self, name: str, arg_type: TTGIRType) -> str:
        """Emit a single argument declaration."""
        # Strip the leading % from SSA names
        clean_name = name.lstrip("%")
        shape = arg_type.shape or (1,)
        shape_str = ", ".join(str(d) for d in shape)
        dtype = arg_type.element_dtype
        return f"{clean_name}: T.Buffer(({shape_str}), \"{dtype}\")"

    def _emit_buffer_declarations(self, func: TTGIRFunction) -> list[str]:
        """Emit any intermediate T.alloc_buffer statements needed."""
        decls: list[str] = []
        # If we have loads/stores, we might need scratch buffers
        # For now, the function args themselves are sufficient
        return decls

    def _emit_op(self, op: TTGIROperation, indent: int) -> list[str]:
        """Emit a single op (and its nested ops) as TVMScript lines."""
        prefix = "    " * indent
        lines: list[str] = []

        if op.kind == OpKind.FOR_LOOP:
            lines.extend(self._emit_for_loop(op, indent))
        elif op.kind == OpKind.IF_STATEMENT:
            lines.extend(self._emit_if(op, indent))
        elif op.kind == OpKind.LOAD:
            lines.extend(self._emit_load(op, indent))
        elif op.kind == OpKind.STORE:
            lines.extend(self._emit_store(op, indent))
        elif op.kind == OpKind.ADDF:
            lines.extend(self._emit_binary_op(op, indent, "T.add"))
        elif op.kind == OpKind.SUBF:
            lines.extend(self._emit_binary_op(op, indent, "T.sub"))
        elif op.kind == OpKind.MULF:
            lines.extend(self._emit_binary_op(op, indent, "T.mul"))
        elif op.kind == OpKind.DIVF:
            lines.extend(self._emit_binary_op(op, indent, "T.div"))
        elif op.kind == OpKind.ADDI:
            lines.extend(self._emit_binary_op(op, indent, "T.add"))
        elif op.kind == OpKind.SUBI:
            lines.extend(self._emit_binary_op(op, indent, "T.sub"))
        elif op.kind == OpKind.MULI:
            lines.extend(self._emit_binary_op(op, indent, "T.mul"))
        elif op.kind == OpKind.EXP:
            lines.extend(self._emit_unary_op(op, indent, "T.exp"))
        elif op.kind == OpKind.LOG:
            lines.extend(self._emit_unary_op(op, indent, "T.log"))
        elif op.kind == OpKind.SQRT:
            lines.extend(self._emit_unary_op(op, indent, "T.sqrt"))
        elif op.kind == OpKind.RSQRT:
            lines.extend(self._emit_unary_op(op, indent, "T.rsqrt"))
        elif op.kind == OpKind.TANH:
            lines.extend(self._emit_unary_op(op, indent, "T.tanh"))
        elif op.kind == OpKind.MAX:
            lines.extend(self._emit_binary_op(op, indent, "T.max"))
        elif op.kind == OpKind.MIN:
            lines.extend(self._emit_binary_op(op, indent, "T.min"))
        elif op.kind == OpKind.CONSTANT:
            lines.extend(self._emit_constant(op, indent))
        elif op.kind == OpKind.REDUCE:
            lines.extend(self._emit_reduce(op, indent))
        elif op.kind == OpKind.DOT:
            # tt.dot is normally split out before this stage, but
            # if we see it here, emit a T.evaluate placeholder
            lines.append(
                f"{prefix}# tt.dot found in converted IR — should have been split out"
            )
            lines.append(f"{prefix}# Skipping; externally-compiled matmul will be called")
        elif op.kind == OpKind.UNKNOWN:
            # Skip unknown ops with a comment
            lines.append(f"{prefix}# [skipped: unknown op] {op.name}")
        else:
            lines.append(f"{prefix}# [unhandled op kind: {op.kind.name}] {op.name}")

        return lines

    def _emit_for_loop(self, op: TTGIROperation, indent: int) -> list[str]:
        """Emit a T.grid-based for-loop from a FOR_LOOP op."""
        prefix = "    " * indent
        axis = op.attributes.get("__axis", "0")
        bound = op.attributes.get("__bound", "1")
        iv = f"ax{axis}"

        lines = [f"{prefix}for {iv} in T.grid({bound}):"]
        for child in op.nested_ops:
            lines.extend(self._emit_op(child, indent + 1))
        return lines

    def _emit_if(self, op: TTGIROperation, indent: int) -> list[str]:
        """Emit a conditional block."""
        prefix = "    " * indent
        cond = "T.const_true"  # Default
        if op.operands:
            cond = op.operands[0].lstrip("%")
        lines = [f"{prefix}if {cond}:"]
        for child in op.nested_ops:
            lines.extend(self._emit_op(child, indent + 1))
        return lines

    def _emit_load(self, op: TTGIROperation, indent: int) -> list[str]:
        """Emit a load as a variable assignment.

        The memref.load / tt.load is emitted as a direct read from
        the buffer. Index expressions come from the current loop
        iteration variables.
        """
        prefix = "    " * indent
        result = op.result_name.lstrip("%") if op.result_name else "_"
        # Try to read the buffer name from operands
        buffer_name = op.operands[0].lstrip("%") if op.operands else "A"
        return [f"{prefix}{result} = {buffer_name}[...]"]

    def _emit_store(self, op: TTGIROperation, indent: int) -> list[str]:
        """Emit a store as a direct buffer write."""
        prefix = "    " * indent
        if len(op.operands) >= 2:
            buffer_name = op.operands[0].lstrip("%")
            value = op.operands[1].lstrip("%")
            return [f"{prefix}{buffer_name}[...] = {value}"]
        return [f"{prefix}# [store: incomplete operands]"]

    def _emit_binary_op(
        self, op: TTGIROperation, indent: int, fn_name: str,
    ) -> list[str]:
        """Emit a binary arithmetic op."""
        prefix = "    " * indent
        result = op.result_name.lstrip("%") if op.result_name else "_"
        if len(op.operands) >= 2:
            a = op.operands[0].lstrip("%")
            b = op.operands[1].lstrip("%")
            return [f"{prefix}{result} = {fn_name}({a}, {b})"]
        return [f"{prefix}# [{fn_name}: incomplete operands]"]

    def _emit_unary_op(
        self, op: TTGIROperation, indent: int, fn_name: str,
    ) -> list[str]:
        """Emit a unary op."""
        prefix = "    " * indent
        result = op.result_name.lstrip("%") if op.result_name else "_"
        if op.operands:
            a = op.operands[0].lstrip("%")
            return [f"{prefix}{result} = {fn_name}({a})"]
        return [f"{prefix}# [{fn_name}: missing operand]"]

    def _emit_constant(self, op: TTGIROperation, indent: int) -> list[str]:
        """Emit a constant assignment."""
        prefix = "    " * indent
        result = op.result_name.lstrip("%") if op.result_name else "_"
        if "value" in op.attributes:
            return [f"{prefix}{result} = T.float32({op.attributes['value']})"]
        return [f"{prefix}{result} = T.float32(0.0)"]

    def _emit_reduce(self, op: TTGIROperation, indent: int) -> list[str]:
        """Emit a reduction op as a T.block with T.init()."""
        prefix = "    " * indent
        result = op.result_name.lstrip("%") if op.result_name else "_"
        axis = op.attributes.get("axis", "0")
        lines = [
            f"{prefix}# Reduction op on axis {axis} (to be lowered by TVM)",
            f"{prefix}{result} = T.float32(0)  # init",
        ]
        if op.nested_ops:
            for child in op.nested_ops:
                lines.extend(self._emit_op(child, indent + 1))
        return lines

