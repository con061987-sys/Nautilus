"""Extract mathematical bounds from captured Triton IR.

Reads tensor shapes, loop bounds, and reduction axes from real
TTGIR text by walking the structured AST produced by
:class:`TTGIRParser`. Produces :class:`IRBounds` that the TIR template
constructor uses to build an equivalent TVM TIR function.

This is the critical step that bridges Triton IR semantics into
TVM TIR semantics — done by analysing AST nodes (op kinds, types,
shapes, attributes), NOT by scanning the whole IR text with regex.

The previous regex-based implementation could not resolve the shapes
of ``tt.dot`` operands (it returned ``m=n=k=0`` for real TTGIR). This
module rebuilds a ``value → TTGIRType`` map from the parsed function
arguments and from each op's result type annotation in its raw text,
then resolves dot/reduce/loop operands against that map.

Supported op families
---------------------
- 2-D matmul: ``tt.dot``, ``tt.dot_scaled``, ``tt.matmul``
  Operands A:(M,K), B:(K,N) → result:(M,N)
- 3-D batched matmul: ``tt.bmm``
  Operands A:(B,M,K), B:(B,K,N) → result:(B,M,N)
- Reductions: ``tt.reduce`` with ``axis = N`` attribute
- Loops: ``scf.for %i = LB to UB { ... }`` — start/end captured as
  ``block_size`` on :class:`IRBounds`

Errors
------
If the IR is malformed, the requested op family is absent, or a
required shape cannot be resolved, :class:`BoundsExtractionError` is
raised. Callers that need to fall back to a template-based path
should catch this exception.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Final

from src.common.logging import get_logger

from .ir_capture import IRBounds, KernelKind
from .ir_to_tir.ttgir_parser import (
    OpKind,
    TTGIRFunction,
    TTGIROperation,
    TTGIRParser,
    TTGIRType,
)

logger = get_logger(__name__)


class BoundsExtractionError(Exception):
    """Raised when bounds cannot be extracted from the TTGIR.

    This is the single, canonical failure signal for the
    :class:`BoundsExtractor`. It is raised when:

    - the IR text cannot be parsed into a TTGIRFunction;
    - the kernel kind (e.g. MATMUL) requires an op (e.g. ``tt.dot``)
      that is not present;
    - a required operand SSA value has no resolvable tensor type;
    - a contracted dimension is mismatched or non-positive;
    - a reduction ``axis`` attribute is missing or out of range.

    Callers that need to fall back to a template-based path
    (``TIRTemplateBuilder``) should catch this exception and let the
    downstream code pick defaults. Callers that need hard guarantees
    (e.g. the MetaSchedule adapter) should let it propagate.
    """


# Op names that are matmul-family. ``tt.dot_scaled`` is the FP8
# variant of ``tt.dot`` and shares the same shape contract. The
# ``tt.matmul`` / ``tt.bmm`` names are not in the upstream Triton
# dialect but appear in vendor forks and downstream dialects; they
# are handled identically.
_MATMUL_OPS: Final[frozenset[str]] = frozenset(
    {
        "tt.dot",
        "tt.dot_scaled",
        "tt.matmul",
        "tt.bmm",
    }
)

# scf.for header: ``scf.for %i = <lb> to <ub> { ... }``. The two
# integer groups are the loop lower and upper bound. This is the
# ONLY regex used in the module and it is applied to a single
# ``scf.for`` op's raw text (a single line), NOT to the whole IR.
# The whole-IR scan that the previous regex-based implementation
# used has been removed.
_SCF_FOR_BOUNDS_RE: Final[re.Pattern[str]] = re.compile(
    r"scf\.for\s+%\w+\s*=\s*(-?\d+)\s*to\s*(-?\d+)",
)

# Matches an explicit return-type annotation ``-> tensor<...>`` that
# appears in ops like ``tt.reduce`` whose textual form is
# ``(tensor<...>) -> tensor<...>``.
_ARROW_TENSOR_RE: Final[re.Pattern[str]] = re.compile(
    r"->\s*tensor<([^>]+)>",
)


class BoundsExtractor:
    """Extract mathematical bounds from captured TTGIR using the AST.

    All extraction goes through :class:`TTGIRParser`. The whole-IR
    regex scan that the previous implementation relied on is gone —
    shapes, dtypes, loop bounds, and reduction axes are resolved by
    walking the AST nodes.
    """

    def __init__(self) -> None:
        # One parser per extractor; the parser is stateless so this is
        # purely a stylistic choice to make the dependency explicit.
        self._parser: TTGIRParser = TTGIRParser()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def extract(self, ir_text: str, kind: KernelKind) -> IRBounds:
        """Extract bounds from the IR based on the kernel kind.

        Args:
            ir_text: The TTGIR text captured from Triton's pipeline.
            kind:    Classification of the kernel — drives which
                     extraction strategy is used.

        Returns:
            A fully-populated :class:`IRBounds` instance.

        Raises:
            BoundsExtractionError: If the IR is malformed, the
                expected op family is missing, or any required
                dimension cannot be resolved.
        """
        func = self._parse(ir_text)
        value_types = self._build_value_type_map(func)
        loop_bounds = self._extract_loop_bounds(func)

        if kind in (KernelKind.MATMUL, KernelKind.ATTENTION):
            return self._extract_matmul_bounds(func, value_types, loop_bounds)
        if kind == KernelKind.REDUCTION:
            return self._extract_reduction_bounds(func, value_types, loop_bounds)
        if kind == KernelKind.ELEMENTWISE:
            return self._extract_elementwise_bounds(func, value_types, loop_bounds)
        # SCAN / PERSISTENT / UNKNOWN: best-effort generic extraction
        return self._extract_generic_bounds(func, value_types, loop_bounds)

    # ------------------------------------------------------------------
    # AST construction
    # ------------------------------------------------------------------

    def _parse(self, ir_text: str) -> TTGIRFunction:
        """Parse IR text into a :class:`TTGIRFunction`.

        All exceptions from the parser are wrapped in
        :class:`BoundsExtractionError` with a clear, single-line
        message so callers don't have to know about the parser's
        internal exception types.
        """
        try:
            return self._parser.parse(ir_text)
        except Exception as exc:
            raise BoundsExtractionError(f"Failed to parse TTGIR: {exc}") from exc

    def _build_value_type_map(
        self,
        func: TTGIRFunction,
    ) -> dict[str, TTGIRType]:
        """Build SSA value → :class:`TTGIRType` map from args and ops.

        Function arguments (e.g. ``%A_ptr: !tt.ptr<tensor<128x32xf32>>``)
        are seeded first — the parser already gives us their full
        type. For every other op, the result type is extracted from
        the op's ``raw_text`` (the part after the trailing ``:`` for
        ops like ``tt.load`` / ``tt.dot`` / ``arith.addf``, or the
        part after ``->`` for ops like ``tt.reduce``).
        """
        value_types: dict[str, TTGIRType] = {}

        # Function args are typically ``!tt.ptr<tensor<...>>``; the
        # inner tensor<...> carries the shape. The parser already
        # collapses this to a TTGIRType with shape and dtype set.
        for name, ttype in func.args:
            value_types[name] = ttype

        # Walk all ops (including those nested in loops / ifs) and
        # extract the result type from each op's raw text.
        for op in func.iter_all_ops():
            if not op.result_name:
                # Statement-style ops (e.g. tt.store, tt.return) have
                # no SSA result and therefore no result type.
                continue
            result_type = self._extract_result_type(op.raw_text)
            if result_type is not None:
                # First definition wins — SSA means a value is
                # defined exactly once, so there is no ambiguity.
                value_types.setdefault(op.result_name, result_type)

        return value_types

    def _extract_result_type(self, raw_text: str) -> TTGIRType | None:
        """Extract the result :class:`TTGIRType` from a single op's text.

        Strategy:

        1. If the op has an explicit ``-> tensor<...>`` return-type
           annotation (used by ``tt.reduce`` and similar ops), use it.
        2. Otherwise, take the **last** ``tensor<...>`` annotation
           in the raw text. For ``tt.load``, ``tt.store``, ``tt.dot``,
           ``arith.addf`` etc. the result type is the only tensor
           annotation, so the last one is the result.
        3. If neither is present, return ``None`` (caller skips the
           op — likely an op that has no tensor result, such as
           ``tt.get_program_id`` returning ``i32``).
        """
        # Strategy 1: explicit return-type form
        arrow_match = _ARROW_TENSOR_RE.search(raw_text)
        if arrow_match is not None:
            return self._parser._parse_type(
                f"tensor<{arrow_match.group(1)}>",
            )

        # Strategy 2: last ``tensor<...>`` in the op text
        matches = list(TTGIRParser.TENSOR_TYPE_RE.finditer(raw_text))
        if not matches:
            return None
        return self._parser._parse_type(
            f"tensor<{matches[-1].group(1)}>",
        )

    # ------------------------------------------------------------------
    # Per-kind extraction
    # ------------------------------------------------------------------

    def _extract_matmul_bounds(
        self,
        func: TTGIRFunction,
        value_types: dict[str, TTGIRType],
        loop_bounds: list[tuple[int, int]],
    ) -> IRBounds:
        """Extract M, N, K for matmul-style kernels.

        The first matmul-family op with resolvable shapes is used.
        Dimensions are derived from the *operand* tensor types
        (A, B) and cross-validated against the *result* tensor type
        when the result type is available.
        """
        op_names = [op.name for op in func.iter_all_ops()]
        for op in func.iter_all_ops():
            if op.name not in _MATMUL_OPS:
                continue
            if len(op.operands) < 2:
                raise BoundsExtractionError(
                    f"{op.name!r} has {len(op.operands)} operand(s); "
                    f"need at least 2 (A, B): {op.raw_text!r}",
                )

            a_name, b_name = op.operands[0], op.operands[1]
            a_type = value_types.get(a_name)
            b_type = value_types.get(b_name)
            result_type = value_types.get(op.result_name) if op.result_name else None

            self._assert_tensor_shape(a_type, a_name, op.name)
            self._assert_tensor_shape(b_type, b_name, op.name)

            m, n, k = self._matmul_dims(op.name, a_type, b_type, result_type)

            if m <= 0 or n <= 0 or k <= 0:
                raise BoundsExtractionError(
                    f"Non-positive dim in {op.name}: "
                    f"M={m}, N={n}, K={k} (A={a_type.shape}, B={b_type.shape})",
                )

            dtype = a_type.element_dtype
            ranks = sorted({len(t.shape) for t in value_types.values() if t.shape})
            return IRBounds(
                m=m,
                n=n,
                k=k,
                data_dtype=dtype,
                block_size=tuple(loop_bounds[0]) if loop_bounds else (),
                tensor_ranks=ranks,
            )

        raise BoundsExtractionError(
            f"IR classified as MATMUL/ATTENTION but no matmul-family op "
            f"({sorted(_MATMUL_OPS)}) found; ops present: {op_names}",
        )

    def _matmul_dims(
        self,
        op_name: str,
        a_type: TTGIRType,
        b_type: TTGIRType,
        result_type: TTGIRType | None,
    ) -> tuple[int, int, int]:
        """Compute (M, N, K) from operand and result tensor types.

        Different matmul families have different shape contracts:

        - ``tt.dot`` / ``tt.dot_scaled`` / ``tt.matmul`` (2-D):
          A:(M,K), B:(K,N), result:(M,N)
        - ``tt.bmm`` (3-D batched):
          A:(B,M,K), B:(B,K,N), result:(B,M,N)

        Returns (M, N, K). Cross-validates against ``result_type``
        if it is provided and has a known shape.
        """
        if op_name == "tt.bmm":
            return self._bmm_dims(a_type, b_type, result_type)
        return self._dot_dims(a_type, b_type, result_type)

    def _dot_dims(
        self,
        a_type: TTGIRType,
        b_type: TTGIRType,
        result_type: TTGIRType | None,
    ) -> tuple[int, int, int]:
        """Compute (M, N, K) for a 2-D dot product."""
        if len(a_type.shape) != 2:
            raise BoundsExtractionError(
                f"2-D dot operand A has rank {len(a_type.shape)} "
                f"(shape {a_type.shape}); expected rank 2",
            )
        if len(b_type.shape) != 2:
            raise BoundsExtractionError(
                f"2-D dot operand B has rank {len(b_type.shape)} "
                f"(shape {b_type.shape}); expected rank 2",
            )

        m, k_a = a_type.shape
        k_b, n = b_type.shape
        if k_a != k_b:
            raise BoundsExtractionError(
                f"Contracted-dim mismatch: A has K={k_a} (shape {a_type.shape}), "
                f"B has K={k_b} (shape {b_type.shape})",
            )

        if result_type is not None and result_type.shape and len(result_type.shape) == 2:
            r_m, r_n = result_type.shape
            if r_m != m or r_n != n:
                raise BoundsExtractionError(
                    f"Result shape {result_type.shape} inconsistent with AxB dims (M={m}, N={n})",
                )

        return m, n, k_a

    def _bmm_dims(
        self,
        a_type: TTGIRType,
        b_type: TTGIRType,
        result_type: TTGIRType | None,
    ) -> tuple[int, int, int]:
        """Compute (M, N, K) for a 3-D batched dot product.

        For ``tt.bmm`` A:(B,M,K), B:(B,K,N), result:(B,M,N), the
        batch dim B is dropped from the returned (M, N, K) since
        :class:`IRBounds` does not currently expose a batch dim.
        """
        if len(a_type.shape) != 3:
            raise BoundsExtractionError(
                f"tt.bmm operand A has rank {len(a_type.shape)} "
                f"(shape {a_type.shape}); expected rank 3",
            )
        if len(b_type.shape) != 3:
            raise BoundsExtractionError(
                f"tt.bmm operand B has rank {len(b_type.shape)} "
                f"(shape {b_type.shape}); expected rank 3",
            )

        batch_a, m, k_a = a_type.shape
        batch_b, k_b, n = b_type.shape
        if k_a != k_b:
            raise BoundsExtractionError(
                f"tt.bmm contracted-dim mismatch: A has K={k_a} "
                f"(shape {a_type.shape}), B has K={k_b} (shape {b_type.shape})",
            )
        if batch_a != batch_b:
            raise BoundsExtractionError(
                f"tt.bmm batch-dim mismatch: A has B={batch_a}, B has B={batch_b}",
            )

        if result_type is not None and result_type.shape:
            if len(result_type.shape) != 3:
                raise BoundsExtractionError(
                    f"tt.bmm result must be 3-D, got rank {len(result_type.shape)} "
                    f"(shape {result_type.shape})",
                )
            r_batch, r_m, r_n = result_type.shape
            if r_batch != batch_a:
                raise BoundsExtractionError(
                    f"tt.bmm result batch dim {r_batch} != A's batch dim {batch_a}",
                )
            if r_m != m or r_n != n:
                raise BoundsExtractionError(
                    f"tt.bmm result shape {result_type.shape} inconsistent with "
                    f"AxB dims (M={m}, N={n}, B={batch_a})",
                )

        return m, n, k_a

    def _extract_reduction_bounds(
        self,
        func: TTGIRFunction,
        value_types: dict[str, TTGIRType],
        loop_bounds: list[tuple[int, int]],
    ) -> IRBounds:
        """Extract bounds for reduction kernels.

        The first ``tt.reduce`` op is used. ``reduce_size`` is the
        size of the input tensor along the reduce axis, and
        ``keep_size`` is the product of the remaining (kept) dims.
        """
        for op in func.iter_all_ops():
            if op.name != "tt.reduce":
                continue
            if not op.operands:
                raise BoundsExtractionError(
                    f"tt.reduce has no operands: {op.raw_text!r}",
                )

            input_name = op.operands[0]
            input_type = value_types.get(input_name)
            self._assert_tensor_shape(input_type, input_name, "tt.reduce")

            axis = self._parse_reduce_axis(op)
            ndim = len(input_type.shape)
            if axis < 0 or axis >= ndim:
                raise BoundsExtractionError(
                    f"tt.reduce axis={axis} out of range for input shape {input_type.shape}",
                )

            reduce_size = input_type.shape[axis]
            if reduce_size <= 0:
                raise BoundsExtractionError(
                    f"tt.reduce axis {axis} has non-positive size {reduce_size} "
                    f"(input shape {input_type.shape})",
                )

            keep_size = 1
            for i, d in enumerate(input_type.shape):
                if i == axis:
                    continue
                if d <= 0:
                    # Unknown keep dim — leave keep_size at 1 rather
                    # than failing the entire extraction.
                    continue
                keep_size *= d
            if keep_size <= 0:
                keep_size = 1

            ranks = sorted({len(t.shape) for t in value_types.values() if t.shape})
            return IRBounds(
                reduce_size=reduce_size,
                keep_size=keep_size,
                data_dtype=input_type.element_dtype,
                block_size=tuple(loop_bounds[0]) if loop_bounds else (),
                tensor_ranks=ranks,
            )

        op_names = [op.name for op in func.iter_all_ops()]
        raise BoundsExtractionError(
            f"IR classified as REDUCTION but no tt.reduce op found; ops present: {op_names}",
        )

    def _parse_reduce_axis(self, op: TTGIROperation) -> int:
        """Parse the ``axis`` attribute of a ``tt.reduce`` op.

        The parser already extracts attributes into ``op.attributes``
        as a ``dict[str, str]``. The axis value is the textual
        representation of an integer (possibly with an ``: i32``
        suffix that the parser strips).
        """
        if "axis" not in op.attributes:
            raise BoundsExtractionError(
                f"tt.reduce missing 'axis' attribute: {op.raw_text!r}",
            )
        axis_text = op.attributes["axis"].strip()
        # Strip an optional ``: i32`` / ``: index`` suffix that the
        # parser may have left attached.
        for suffix in (": i32", ": index", ": i64"):
            if axis_text.endswith(suffix):
                axis_text = axis_text[: -len(suffix)]
                break
        try:
            return int(axis_text)
        except ValueError as exc:
            raise BoundsExtractionError(
                f"tt.reduce 'axis' attribute is not an integer: {op.attributes['axis']!r}",
            ) from exc

    def _extract_elementwise_bounds(
        self,
        func: TTGIRFunction,
        value_types: dict[str, TTGIRType],
        loop_bounds: list[tuple[int, int]],
    ) -> IRBounds:
        """Extract bounds for elementwise kernels.

        For elementwise kernels, ``total_elements`` is the product of
        the first resolvable load operand's shape. If no load is
        present, any other SSA value with a fully-known shape is
        used as a fallback.
        """
        # Prefer a load operand — it represents the dominant input.
        for op in func.iter_all_ops():
            if op.name != "tt.load" or not op.operands:
                continue
            arg_name = op.operands[0]
            ttype = value_types.get(arg_name)
            if self._has_known_shape(ttype):
                return self._build_elementwise_bounds(
                    ttype,
                    value_types,
                    loop_bounds,
                )

        # Fallback: any SSA value with a known shape.
        for ttype in value_types.values():
            if self._has_known_shape(ttype):
                return self._build_elementwise_bounds(
                    ttype,
                    value_types,
                    loop_bounds,
                )

        raise BoundsExtractionError(
            "No tensor shape resolvable for elementwise bounds "
            f"(value_types has {len(value_types)} entries)",
        )

    def _extract_generic_bounds(
        self,
        func: TTGIRFunction,
        value_types: dict[str, TTGIRType],
        loop_bounds: list[tuple[int, int]],
    ) -> IRBounds:
        """Generic bounds: sum of all tensor element counts.

        Used as a last-resort fallback for kernels whose kind is
        SCAN / PERSISTENT / UNKNOWN. Each value with a known shape
        contributes its element count to ``total_elements``.
        """
        total = 0
        dtypes: list[str] = []
        ranks_set: set[int] = set()
        for ttype in value_types.values():
            if not self._has_known_shape(ttype):
                continue
            prod = 1
            for d in ttype.shape:
                prod *= d
            total += prod
            dtypes.append(ttype.element_dtype)
            ranks_set.add(len(ttype.shape))

        if total <= 0:
            raise BoundsExtractionError(
                f"No tensor shapes resolvable for generic bounds "
                f"(value_types has {len(value_types)} entries)",
            )

        dtype = self._dominant_dtype(dtypes)
        ranks = sorted(ranks_set)
        return IRBounds(
            total_elements=total,
            data_dtype=dtype,
            block_size=tuple(loop_bounds[0]) if loop_bounds else (),
            tensor_ranks=ranks,
        )

    # ------------------------------------------------------------------
    # Loop bounds
    # ------------------------------------------------------------------

    def _extract_loop_bounds(
        self,
        func: TTGIRFunction,
    ) -> list[tuple[int, int]]:
        """Extract ``scf.for`` loop bounds (start, end) from FOR_LOOP ops.

        The regex is applied to a single ``scf.for`` op's raw text,
        not to the whole IR — this is per-AST-node analysis, not a
        whole-IR scan. The first scf.for bound is also exposed via
        :attr:`IRBounds.block_size` by the per-kind extractors.
        """
        bounds: list[tuple[int, int]] = []
        for op in func.iter_all_ops():
            if op.kind != OpKind.FOR_LOOP:
                continue
            m = _SCF_FOR_BOUNDS_RE.search(op.raw_text)
            if m is None:
                # FOR_LOOP node exists but the bounds are not static
                # integers (e.g. ``scf.for %i = %lb to %ub``). Skip
                # silently — dynamic loops can't be summarised as
                # a (lb, ub) pair.
                continue
            try:
                bounds.append((int(m.group(1)), int(m.group(2))))
            except (ValueError, IndexError) as exc:
                raise BoundsExtractionError(
                    f"Failed to parse scf.for bounds: {op.raw_text!r}: {exc}",
                ) from exc
        return bounds

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_known_shape(ttype: TTGIRType | None) -> bool:
        """True if the type has a non-empty shape with all-known dims."""
        if ttype is None:
            return False
        if not ttype.shape:
            return False
        return all(d > 0 for d in ttype.shape)

    @staticmethod
    def _assert_tensor_shape(
        ttype: TTGIRType | None,
        value_name: str,
        context_op: str,
    ) -> None:
        """Raise :class:`BoundsExtractionError` if shape cannot be resolved."""
        if ttype is None:
            raise BoundsExtractionError(
                f"Cannot resolve tensor type for {value_name!r} (operand of {context_op!r})",
            )
        if not ttype.shape:
            raise BoundsExtractionError(
                f"Operand {value_name!r} of {context_op!r} has no shape (type {ttype.raw!r})",
            )
        if any(d <= 0 for d in ttype.shape):
            raise BoundsExtractionError(
                f"Operand {value_name!r} of {context_op!r} has unknown "
                f"dim(s) in shape {ttype.shape}",
            )

    def _build_elementwise_bounds(
        self,
        ttype: TTGIRType,
        value_types: dict[str, TTGIRType],
        loop_bounds: list[tuple[int, int]],
    ) -> IRBounds:
        """Build an elementwise :class:`IRBounds` from a representative type."""
        total = 1
        for d in ttype.shape:
            total *= d
        ranks = sorted({len(t.shape) for t in value_types.values() if t.shape})
        return IRBounds(
            total_elements=total,
            data_dtype=ttype.element_dtype,
            block_size=tuple(loop_bounds[0]) if loop_bounds else (),
            tensor_ranks=ranks,
        )

    @staticmethod
    def _dominant_dtype(dtypes: list[str]) -> str:
        """Return the most common dtype, or ``float32`` for an empty list."""
        if not dtypes:
            return "float32"
        return Counter(dtypes).most_common(1)[0][0]
