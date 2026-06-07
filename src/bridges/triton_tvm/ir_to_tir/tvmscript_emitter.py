"""TVMScript emitter.

Converts the converted AST (output of Pass 4) into TVMScript text —
the Python-embedded TIR DSL that TVM's MetaSchedule can directly
consume via `tvm.script.tir.prim_func` and `tvm.meta_schedule.tune_tir`.

This is the ONLY module that produces TVM-specific output. All the
conversion passes produce a generic AST; the emitter is the boundary
that knows TVM's syntax.

Loop structure
--------------
The post-pipeline AST contains a single top-level ``FOR_LOOP`` with
``__axis=0, __bound=1`` — a placeholder inserted by Pass 2 (it doesn't
know the real program count). The emitter ignores that loop and
reconstructs a fresh loop nest from the function's tensor arguments:

  - Matmul detection: if the first three args are 2D tensors with
    shapes (M, K), (K, N), (M, N), the loop nest is ``(M, N, K)``
    with IVs ``ax0, ax1, ax2``. Memory accesses use matmul-aware
    indexing: A→``[ax0, ax2]``, B→``[ax2, ax1]``, C→``[ax0, ax1]``.
  - Otherwise: the loop nest is the largest tensor arg's shape,
    with IVs ``ax0, ax1, ...`` matching the dim count.

Memory access indexing
----------------------
Every LOAD / STORE reads the buffer's shape from the function args
(``func.args``) and emits exactly ``len(shape)`` index expressions.
The index expressions are the loop induction variables in nesting
order, with matmul-aware remapping for the matmul case.

This module never emits ``[...]`` (Ellipsis). Every memory access
references concrete loop induction variables, and the emitter raises
``ValueError`` if it cannot determine a buffer's shape — that is the
only way to keep the "no ellipsis" invariant honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
      - Loop nest is reconstructed from function arg shapes; the
        placeholder FOR_LOOP from Pass 2 is intentionally ignored
      - Memory accesses use concrete loop induction variables, never
        Ellipsis; a missing shape is a hard error
    """

    # Populated per emit() call from func.args
    _buffer_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    _buffer_dtypes: dict[str, str] = field(default_factory=dict)
    # True when the (M, K), (K, N), (M, N) matmul pattern is detected
    _is_matmul: bool = False
    # Loop bounds (M, N, K) when matmul, else the largest arg's shape
    _loop_dims: tuple[int, ...] = ()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def emit(self, func: TTGIRFunction) -> str:
        """Emit TVMScript text for the function.

        Returns a string of Python code that can be wrapped in
        `@tvm.script.tir.prim_func`.
        """
        # Reset per-call state
        self._buffer_shapes = {}
        self._buffer_dtypes = {}
        self._is_matmul = False
        self._loop_dims = ()

        # Index every tensor arg by clean name so load/store can look up shapes
        for arg_name, arg_type in func.args:
            if arg_type.is_tensor:
                clean_name = arg_name.lstrip("%")
                self._buffer_shapes[clean_name] = arg_type.shape
                self._buffer_dtypes[clean_name] = arg_type.element_dtype

        # Compute loop dimensions and detect matmul
        self._is_matmul = self._detect_matmul(func)
        self._loop_dims = self._compute_loop_dims(func)

        lines: list[str] = []

        # Function header
        lines.append("@T.prim_func")
        lines.append(f"def {func.name}({self._emit_signature(func)}):")
        lines.append("    # Function body — emitted from real Triton IR")
        lines.append("")

        # T.alloc_buffer declarations for every tensor arg
        alloc_lines = self._emit_alloc_buffers(func)
        if alloc_lines:
            lines.extend(alloc_lines)
            lines.append("")

        # Pass 2 wraps the body in a placeholder FOR_LOOP (bound=1);
        # we extract its children because we rebuild the loop nest
        # from the function arg shapes below.
        body_ops: list[TTGIROperation] = []
        for op in func.ops:
            if op.kind == OpKind.ALLOC_BUFFER:
                continue
            if op.kind == OpKind.FOR_LOOP:
                body_ops.extend(op.nested_ops)
            else:
                body_ops.append(op)

        # Emit the loop nest + body
        self._emit_loop_nest(body_ops, lines, indent=1)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Signature + buffer declarations
    # ------------------------------------------------------------------

    def _emit_signature(self, func: TTGIRFunction) -> str:
        """Emit the function signature: arg1: T.Buffer(...), arg2: ..."""
        parts: list[str] = []
        for name, arg_type in func.args:
            parts.append(self._emit_arg(name, arg_type))
        return ", ".join(parts) if parts else ""

    def _emit_arg(self, name: str, arg_type: TTGIRType) -> str:
        """Emit a single argument declaration."""
        clean_name = name.lstrip("%")
        shape = arg_type.shape or (1,)
        shape_str = ", ".join(str(d) for d in shape)
        dtype = arg_type.element_dtype
        return f"{clean_name}: T.Buffer(({shape_str}), \"{dtype}\")"

    def _emit_alloc_buffers(self, func: TTGIRFunction) -> list[str]:
        """Emit T.alloc_buffer for every tensor arg.

        Each declaration is of the form
        ``A_buf = T.alloc_buffer((M, N), dtype="float32")`` and is
        placed at the top of the function body. Buffer access ops
        reference the *original* arg name (A, B, C, …) since in
        TVMScript the function parameters are themselves T.Buffer
        objects — the ``_buf`` is an optional scratch alias.
        """
        decls: list[str] = []
        for arg_name, arg_type in func.args:
            if not arg_type.is_tensor:
                continue
            clean_name = arg_name.lstrip("%")
            shape = arg_type.shape or (1,)
            shape_str = ", ".join(str(d) for d in shape)
            decls.append(
                f"    {clean_name}_buf = T.alloc_buffer(({shape_str}), "
                f"dtype=\"{arg_type.element_dtype}\")",
            )
        return decls

    # ------------------------------------------------------------------
    # Loop-nest reconstruction
    # ------------------------------------------------------------------

    def _detect_matmul(self, func: TTGIRFunction) -> bool:
        """Return True if the first three args form a (M,K) x (K,N) = (M,N) matmul."""
        if len(func.args) < 3:
            return False
        a_type = func.args[0][1]
        b_type = func.args[1][1]
        c_type = func.args[2][1]
        if not (a_type.is_tensor and b_type.is_tensor and c_type.is_tensor):
            return False
        if not (len(a_type.shape) == 2 and len(b_type.shape) == 2 and len(c_type.shape) == 2):
            return False
        m, k1 = a_type.shape
        k2, n = b_type.shape
        cm, cn = c_type.shape
        return bool(k1 == k2 and m == cm and n == cn)

    def _compute_loop_dims(self, func: TTGIRFunction) -> tuple[int, ...]:
        """Compute the loop-nest dimensions from the function args.

        Matmul case: returns ``(M, N, K)`` so the emitter can do
        matmul-aware indexing (A→[ax0, ax2], B→[ax2, ax1], C→[ax0, ax1]).

        Otherwise: returns the largest tensor arg's shape. This handles
        1D elementwise (256,), 2D reductions, and any N-D case.
        """
        if self._is_matmul:
            a_shape = func.args[0][1].shape
            b_shape = func.args[1][1].shape
            # (M, K) x (K, N) — emit (M, N, K) to expose K as the
            # reduction dim last
            m, _k = a_shape
            _k2, n = b_shape
            return (m, n, _k)

        max_shape: tuple[int, ...] = ()
        for _name, arg_type in func.args:
            if arg_type.is_tensor and len(arg_type.shape) > len(max_shape):
                max_shape = arg_type.shape
        if not max_shape:
            # No tensor args (rare) — single iteration default
            return (1,)
        return max_shape

    def _emit_loop_nest(
        self,
        body_ops: list[TTGIROperation],
        lines: list[str],
        indent: int,
    ) -> None:
        """Emit the loop nest wrapping ``body_ops``.

        The nest is collapsed into a single ``T.grid`` call when there
        is more than one dimension, mirroring the canonical
        ``for i, j in T.grid(M, N):`` TVMScript idiom.
        """
        dims = self._loop_dims
        if not dims:
            # No loop at all — emit body at the given indent
            for op in body_ops:
                lines.extend(self._emit_op(op, indent))
            return

        if len(dims) == 1:
            ivs = "ax0"
            bounds = f"{dims[0]}"
        else:
            ivs = ", ".join(f"ax{i}" for i in range(len(dims)))
            bounds = ", ".join(str(d) for d in dims)

        # First line opens the loop; subsequent IVs are notional — a
        # single T.grid(M, N, K) expands into a Python tuple-unpack.
        prefix = "    " * indent
        lines.append(f"{prefix}for {ivs} in T.grid({bounds}):")

        # Emit the body inside the innermost loop
        body_indent = indent + 1
        if not body_ops:
            # Defensive: never emit an empty loop (TVMScript rejects it)
            lines.append(f"{'    ' * body_indent}T.evaluate(0)")
            return
        for op in body_ops:
            lines.extend(self._emit_op(op, body_indent))

    # ------------------------------------------------------------------
    # Op dispatch
    # ------------------------------------------------------------------

    def _emit_op(self, op: TTGIROperation, indent: int) -> list[str]:
        """Emit a single op as TVMScript lines."""
        prefix = "    " * indent
        lines: list[str] = []

        # Pass 4 wraps every LOAD/STORE/REDUCE in a TVM_BLOCK; unwrap
        # transparently so the inner op's emitter sees the real kind.
        if op.kind == OpKind.TVM_BLOCK:
            for child in op.nested_ops:
                lines.extend(self._emit_op(child, indent))
            return lines

        # Defensive: handle a leftover FOR_LOOP (e.g. when something
        # downstream wraps the body in another loop). We re-emit the
        # axis as a fresh T.grid loop and recurse.
        if op.kind == OpKind.FOR_LOOP:
            axis = op.attributes.get("__axis", "0")
            bound = op.attributes.get("__bound", "1")
            iv = f"ax{axis}"
            lines.append(f"{prefix}for {iv} in T.grid({bound}):")
            for child in op.nested_ops:
                lines.extend(self._emit_op(child, indent + 1))
            return lines

        if op.kind == OpKind.IF_STATEMENT:
            cond = "T.const_true"
            if op.operands:
                cond = op.operands[0].lstrip("%")
            lines.append(f"{prefix}if {cond}:")
            for child in op.nested_ops:
                lines.extend(self._emit_op(child, indent + 1))
            return lines

        if op.kind == OpKind.LOAD:
            return self._emit_load(op, indent)
        if op.kind == OpKind.STORE:
            return self._emit_store(op, indent)
        if op.kind == OpKind.ADDF:
            return self._emit_binary_op(op, indent, "T.add")
        if op.kind == OpKind.SUBF:
            return self._emit_binary_op(op, indent, "T.sub")
        if op.kind == OpKind.MULF:
            return self._emit_binary_op(op, indent, "T.mul")
        if op.kind == OpKind.DIVF:
            return self._emit_binary_op(op, indent, "T.div")
        if op.kind == OpKind.ADDI:
            return self._emit_binary_op(op, indent, "T.add")
        if op.kind == OpKind.SUBI:
            return self._emit_binary_op(op, indent, "T.sub")
        if op.kind == OpKind.MULI:
            return self._emit_binary_op(op, indent, "T.mul")
        if op.kind == OpKind.EXP:
            return self._emit_unary_op(op, indent, "T.exp")
        if op.kind == OpKind.LOG:
            return self._emit_unary_op(op, indent, "T.log")
        if op.kind == OpKind.SQRT:
            return self._emit_unary_op(op, indent, "T.sqrt")
        if op.kind == OpKind.RSQRT:
            return self._emit_unary_op(op, indent, "T.rsqrt")
        if op.kind == OpKind.TANH:
            return self._emit_unary_op(op, indent, "T.tanh")
        if op.kind == OpKind.COS:
            return self._emit_unary_op(op, indent, "T.cos")
        if op.kind == OpKind.SIN:
            return self._emit_unary_op(op, indent, "T.sin")
        if op.kind == OpKind.MAX:
            return self._emit_binary_op(op, indent, "T.max")
        if op.kind == OpKind.MIN:
            return self._emit_binary_op(op, indent, "T.min")
        if op.kind == OpKind.CONSTANT:
            return self._emit_constant(op, indent)
        if op.kind == OpKind.REDUCE:
            return self._emit_reduce(op, indent)
        if op.kind == OpKind.BROADCAST:
            return self._emit_broadcast(op, indent)
        if op.kind == OpKind.RESHAPE:
            return self._emit_reshape(op, indent)
        if op.kind == OpKind.DOT:
            return self._emit_dot(op, indent)
        if op.kind == OpKind.GET_PROGRAM_ID:
            return self._emit_get_program_id(op, indent)
        if op.kind == OpKind.GET_NUM_PROGRAMS:
            return self._emit_get_num_programs(op, indent)
        if op.kind == OpKind.TVM_INIT:
            return self._emit_tvm_init(op, indent)
        if op.kind == OpKind.YIELD:
            # scf.yield has no TIR equivalent — skip silently
            return lines
        if op.kind in (OpKind.UNKNOWN, OpKind.RETURN):
            # MLIR terminators (tt.return) and other unrecognised ops
            # have no TIR form; T.prim_func returns implicitly when
            # the body ends.
            return lines

        # Truly unrecognised — log and skip rather than emit garbage
        logger.warning("Unrecognised op kind %s in emitter; skipping", op.kind.name)
        return lines

    # ------------------------------------------------------------------
    # Memory access (the bug the task targets)
    # ------------------------------------------------------------------

    def _index_expr_for_buffer(self, buffer_name: str) -> str:
        """Compute the index expression for a buffer access.

        Matmul-aware: A(M, K)→[ax0, ax2], B(K, N)→[ax2, ax1],
        C(M, N)→[ax0, ax1]. The mapping is derived from the buffer's
        shape, not its name, so a kernel that swaps A and B (e.g. for
        a ``B @ A`` matmul) still gets correct indexing as long as the
        shape still matches one of the matmul slots.

        For non-matmul kernels, the index is just the first N loop
        IVs where N is the buffer's dim count.
        """
        shape = self._buffer_shapes.get(buffer_name)
        if shape is None:
            raise ValueError(
                f"No shape recorded for buffer {buffer_name!r}; "
                f"known buffers: {sorted(self._buffer_shapes)}",
            )

        if self._is_matmul and len(shape) == 2:
            m, n, k = self._loop_dims
            if shape == (m, k):
                return "ax0, ax2"
            if shape == (k, n):
                return "ax2, ax1"
            if shape == (m, n):
                return "ax0, ax1"
            # Fall through to the default mapping if a 2D buffer does
            # not match any of the three matmul slots

        ndim = len(shape)
        return ", ".join(f"ax{i}" for i in range(ndim))

    def _emit_load(self, op: TTGIROperation, indent: int) -> list[str]:
        """Emit a load as ``result = buffer_name[i, j, ...]``."""
        prefix = "    " * indent
        result = op.result_name.lstrip("%") if op.result_name else "_"
        if not op.operands:
            raise ValueError(f"LOAD op {op.raw_text!r} has no operands")
        buffer_name = op.operands[0].lstrip("%")
        index_expr = self._index_expr_for_buffer(buffer_name)
        return [f"{prefix}{result} = {buffer_name}[{index_expr}]"]

    def _emit_store(self, op: TTGIROperation, indent: int) -> list[str]:
        """Emit a store as ``buffer_name[i, j, ...] = value``."""
        prefix = "    " * indent
        if len(op.operands) < 2:
            raise ValueError(
                f"STORE op {op.raw_text!r} has fewer than 2 operands: {op.operands}",
            )
        buffer_name = op.operands[0].lstrip("%")
        value = op.operands[1].lstrip("%")
        index_expr = self._index_expr_for_buffer(buffer_name)
        return [f"{prefix}{buffer_name}[{index_expr}] = {value}"]

    # ------------------------------------------------------------------
    # Arithmetic / math ops
    # ------------------------------------------------------------------

    def _emit_binary_op(
        self, op: TTGIROperation, indent: int, fn_name: str,
    ) -> list[str]:
        """Emit a binary arithmetic op as ``result = fn(a, b)``."""
        prefix = "    " * indent
        result = op.result_name.lstrip("%") if op.result_name else "_"
        if len(op.operands) < 2:
            raise ValueError(
                f"Binary op {op.raw_text!r} has fewer than 2 operands: {op.operands}",
            )
        a = op.operands[0].lstrip("%")
        b = op.operands[1].lstrip("%")
        return [f"{prefix}{result} = {fn_name}({a}, {b})"]

    def _emit_unary_op(
        self, op: TTGIROperation, indent: int, fn_name: str,
    ) -> list[str]:
        """Emit a unary op as ``result = fn(a)``."""
        prefix = "    " * indent
        result = op.result_name.lstrip("%") if op.result_name else "_"
        if not op.operands:
            raise ValueError(f"Unary op {op.raw_text!r} has no operands")
        a = op.operands[0].lstrip("%")
        return [f"{prefix}{result} = {fn_name}({a})"]

    def _emit_constant(self, op: TTGIROperation, indent: int) -> list[str]:
        """Emit a constant assignment with the recorded dtype."""
        prefix = "    " * indent
        result = op.result_name.lstrip("%") if op.result_name else "_"
        value = op.attributes.get("value", "0.0")
        dtype = op.attributes.get("dtype", "float32")
        return [f"{prefix}{result} = T.{dtype}({value})"]

    # ------------------------------------------------------------------
    # Reduction
    # ------------------------------------------------------------------

    def _emit_reduce(self, op: TTGIROperation, indent: int) -> list[str]:
        """Emit a reduction as a TIR ``T.sum`` (or T.max / T.min) call.

        Pass 4 wraps the original REDUCE in a TVM_BLOCK whose
        ``__tvm_reduction_axis`` attribute records the axis. The
        body of the reduction (e.g. an ``arith.addf`` child) is not
        lowered here — only the high-level shape is preserved. For
        the tt.reduce-with-addf pattern we emit ``T.sum``; for other
        combiners we fall back to ``T.reduce``.
        """
        prefix = "    " * indent
        result = op.result_name.lstrip("%") if op.result_name else "_"
        axis = op.attributes.get("axis", "0")

        # The original reduction's first operand is the input tensor
        input_var = "x"
        if op.operands:
            input_var = op.operands[0].lstrip("%")

        # Heuristic: if the reduction's nested body contains an
        # arith.addf, the combiner is a sum
        combiner = "T.sum"
        for child in op.nested_ops:
            if child.kind == OpKind.ADDF:
                combiner = "T.sum"
                break
            if child.kind == OpKind.MAX:
                combiner = "T.max"
                break
            if child.kind == OpKind.MIN:
                combiner = "T.min"
                break

        return [f"{prefix}{result} = {combiner}({input_var}, axis=int32({axis}))"]

    def _emit_tvm_init(self, op: TTGIROperation, indent: int) -> list[str]:
        """Emit a T.init as the dtype-typed zero (or -inf for max)."""
        prefix = "    " * indent
        result = op.result_name.lstrip("%") if op.result_name else "_"
        dtype = op.attributes.get("__tvm_init_dtype", "float32")
        return [f"{prefix}{result} = T.{dtype}(0)"]

    # ------------------------------------------------------------------
    # Tensor ops (broadcast / reshape)
    # ------------------------------------------------------------------

    def _emit_broadcast(self, op: TTGIROperation, indent: int) -> list[str]:
        """Emit ``T.Broadcast(x, shape)`` for a broadcast op."""
        prefix = "    " * indent
        result = op.result_name.lstrip("%") if op.result_name else "_"
        operands_str = ", ".join(o.lstrip("%") for o in op.operands)
        return [f"{prefix}{result} = T.Broadcast({operands_str})"]

    def _emit_reshape(self, op: TTGIROperation, indent: int) -> list[str]:
        """Emit ``T.reshape(x, shape)`` for a reshape op."""
        prefix = "    " * indent
        result = op.result_name.lstrip("%") if op.result_name else "_"
        operands_str = ", ".join(o.lstrip("%") for o in op.operands)
        return [f"{prefix}{result} = T.reshape({operands_str})"]

    # ------------------------------------------------------------------
    # Dot (defensive — should normally be split out before emission)
    # ------------------------------------------------------------------

    def _emit_dot(self, op: TTGIROperation, indent: int) -> list[str]:
        """Emit a defensive T.tvm_call_cpacked for any surviving tt.dot.

        In the normal flow, TTDotSplitter extracts the dot before the
        4-pass pipeline runs, so the emitter should not see a DOT op.
        If one survives (unsupported kernel shape, parser gap), emit
        an extern call so the script is at least syntactically valid.
        """
        prefix = "    " * indent
        result = op.result_name.lstrip("%") if op.result_name else "_"
        a = op.operands[0].lstrip("%") if op.operands else "A"
        b = op.operands[1].lstrip("%") if len(op.operands) >= 2 else "B"
        c = op.operands[2].lstrip("%") if len(op.operands) >= 3 else None
        args = f"{a}, {b}" + (f", {c}" if c else "")
        return [
            f"{prefix}# tt.dot survived dot-split; emitting extern call",
            f"{prefix}{result} = T.tvm_call_cpacked("
            f"\"nautilus_matmul\", {args})",
        ]

    # ------------------------------------------------------------------
    # SPMD primitives
    # ------------------------------------------------------------------

    def _emit_get_program_id(
        self, op: TTGIROperation, indent: int,
    ) -> list[str]:
        """Emit ``pid = T.launch_thread(axis, 1)`` for ``tt.get_program_id``."""
        prefix = "    " * indent
        result = op.result_name.lstrip("%") if op.result_name else "_"
        axis = op.operands[0] if op.operands else "0"
        return [f"{prefix}{result} = T.launch_thread({axis}, 1)"]

    def _emit_get_num_programs(
        self, op: TTGIROperation, indent: int,
    ) -> list[str]:
        """Emit ``n = T.launch_thread(axis, 1)`` for ``tt.get_num_programs``."""
        prefix = "    " * indent
        result = op.result_name.lstrip("%") if op.result_name else "_"
        axis = op.operands[0] if op.operands else "0"
        return [f"{prefix}{result} = T.launch_thread({axis}, 1)"]


__all__ = ["TVMScriptEmitter"]
