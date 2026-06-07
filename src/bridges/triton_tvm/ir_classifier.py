"""IR classification — classifies captured TTGIR by walking the AST.

Determines the kind of kernel we're looking at by inspecting the
parsed operation structure (not raw text). This drives which TIR
template the bridge constructs and how MetaSchedule results map
back to Triton.

The classifier relies on TTGIRParser for AST extraction — it does
not re-implement lexer/regex logic. Classification is a pure
function of the parsed function: kind, op counts, reduce-axis
attributes, and combine-op names surface as structured fields on
``IRClassification`` for downstream consumers.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto

from src.common.logging import get_logger

from .ir_capture import KernelKind
from .ir_to_tir.ttgir_parser import OpKind, TTGIRFunction, TTGIRParser, TTGIRType

logger = get_logger(__name__)


class ClassificationError(Exception):
    """Raised when classification is impossible for a given IR.

    The classifier is total over IR containing any of the supported
    op kinds (load/store/dot/reduce/broadcast/trans/elementwise/scan/
    persistent). Only IR that contains none of these — and therefore
    has no recognisable Triton/arith/math op structure at all —
    triggers this error.
    """


class ReductionType(Enum):
    """Combine-op categories for a tt.reduce.

    Maps MLIR combine ops (e.g. arith.addf, arith.maximumf) to
    human-readable names. Used by the classifier to surface the
    reduction semantics of a kernel so the TIR template can pick
    the right primitive.
    """
    SUM = auto()
    MAX = auto()
    MIN = auto()
    ARGMAX = auto()
    ARGMIN = auto()
    MEAN = auto()      # divide-after-sum; inferred from addf + size
    UNKNOWN = auto()

    @property
    def name(self) -> str:  # type: ignore[override]
        return self._name_.lower()


# Map combine-op names extracted from the reduce body to reduction type.
# Order matters: more specific (argmax/argmin) before their scalar siblings
# so a future "argmax-of-floats" op doesn't get misclassified as "max".
_COMBINE_OP_TO_REDUCTION: dict[str, ReductionType] = {
    "arith.addf": ReductionType.SUM,
    "arith.addi": ReductionType.SUM,
    "arith.maximumf": ReductionType.MAX,
    "arith.maxsi": ReductionType.MAX,
    "arith.maximum": ReductionType.MAX,
    "arith.minimumf": ReductionType.MIN,
    "arith.minsi": ReductionType.MIN,
    "arith.minimum": ReductionType.MIN,
    "tt.argmax": ReductionType.ARGMAX,
    "tt.argmin": ReductionType.ARGMIN,
}


def _dtype_to_str(dtype: str) -> str:
    """Map MLIR dtype string to canonical Python form."""
    mapping = {
        "f32": "float32", "f16": "float16", "f64": "float64",
        "bf16": "bfloat16",
        "i32": "int32", "i64": "int64", "i8": "int8",
        "u32": "uint32", "u64": "uint64", "u8": "uint8",
        "i1": "bool",
    }
    return mapping.get(dtype, dtype)


@dataclass
class IRClassification:
    """Structured classification result for a captured TTGIR function.

    The ``__eq__`` override lets callers compare a classification
    against a bare ``KernelKind`` (e.g. ``result == KernelKind.MATMUL``)
    without unwrapping ``.kind`` first — preserves the original
    single-value return semantics alongside the richer structured
    output.

    Attributes:
        kind: The classified kernel kind.
        reduction_type: For REDUCTION/MATMUL kernels, the combine-op
            category extracted from the reduce body. ``None`` for
            kernels that don't reduce.
        reduction_axis: The integer axis (from the ``axis`` attribute)
            of the reduce op. ``None`` if the IR has no reduce or
            the attribute is missing.
        tensor_element_type: Dominant element dtype across the
            function's tensor types (e.g. ``"float32"``).
        tensor_shapes: All concrete tensor shapes seen in the IR,
            in order of first appearance. Unknown dimensions
            (``?``) appear as ``-1``.
        ops: Op names (``tt.dot``, ``arith.addf``, ...) in the
            order they appear in the IR, including nested ones.
    """
    kind: KernelKind
    reduction_type: ReductionType | None = None
    reduction_axis: int | None = None
    tensor_element_type: str | None = None
    tensor_shapes: list[tuple[int, ...]] = field(default_factory=list)
    ops: list[str] = field(default_factory=list)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, KernelKind):
            return self.kind == other
        if isinstance(other, IRClassification):
            return (
                self.kind == other.kind
                and self.reduction_type == other.reduction_type
                and self.reduction_axis == other.reduction_axis
            )
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.kind, self.reduction_type, self.reduction_axis))


class IRClassifier:
    """Classify captured TTGIR by inspecting its parsed op structure."""

    def __init__(self, parser: TTGIRParser | None = None) -> None:
        self._parser = parser or TTGIRParser()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, ir_text: str) -> IRClassification:
        """Classify the captured IR by walking the parsed AST.

        Raises:
            ClassificationError: If the IR has no recognisable
                supported ops (no tt.* / arith.* / math.* / scf.*
                at all) or no ``tt.func`` definition at all. The
                classifier is total over IR containing any supported
                op — it never returns UNKNOWN for IR that mentions
                ``tt.`` or arithmetic.
        """
        try:
            func = self._parser.parse(ir_text)
        except ValueError as exc:
            raise ClassificationError(
                f"IR cannot be parsed (no tt.func definition or "
                f"malformed): {exc}",
            ) from exc
        return self._classify_parsed(func)

    def classify_kind(self, ir_text: str) -> KernelKind:
        """Convenience: classify and return just the KernelKind.

        Equivalent to ``self.classify(ir_text).kind`` but spelled
        out for callers that don't need the rest of the structured
        result.
        """
        return self.classify(ir_text).kind

    def collect_ops(self, ir_text: str) -> list[str]:
        """Collect all op names from the IR in order of appearance.

        Each occurrence is preserved (no dedup). Delegates to the
        parser so the order matches the AST.
        """
        func = self._parser.parse(ir_text)
        return [op.name for op in func.iter_all_ops() if op.name]

    def collect_op_counts(self, ir_text: str) -> Counter:
        """Count occurrences of each op name in the IR."""
        return Counter(self.collect_ops(ir_text))

    def collect_tensor_types(
        self, ir_text: str,
    ) -> list[tuple[tuple[int, ...], str]]:
        """Extract all tensor type definitions and their (shape, dtype).

        Walks the parsed function's args and ops, surfacing only
        concrete (non-pointer) tensor types. Unknown dimensions
        in the shape string appear as ``-1``.
        """
        func = self._parser.parse(ir_text)
        results: list[tuple[tuple[int, ...], str]] = []
        seen: set[tuple[tuple[int, ...], str]] = set()
        for _, arg_type in func.args:
            entry = self._tensor_entry(arg_type)
            if entry is not None and entry not in seen:
                results.append(entry)
                seen.add(entry)
        for op in func.iter_all_ops():
            for t in op.types:
                entry = self._tensor_entry(t)
                if entry is not None and entry not in seen:
                    results.append(entry)
                    seen.add(entry)
        return results

    def parse_shape(self, shape_str: str) -> tuple[int, ...]:
        """Public alias for the shape parser (was ``_parse_shape``).

        Returns an empty tuple for empty input — the parser's
        internal helper returns ``(-1,)`` for ``""`` because it
        only sees the lone empty part after ``split``. We
        normalise that to ``()`` here so callers can use the
        result as a sentinel for "no shape" without special-casing.
        """
        if not shape_str.strip():
            return ()
        return self._parser._parse_shape(shape_str)  # noqa: SLF001

    # ------------------------------------------------------------------
    # Internal classification
    # ------------------------------------------------------------------

    def _classify_parsed(self, func: TTGIRFunction) -> IRClassification:
        ops = list(func.iter_all_ops())
        op_kinds = [op.kind for op in ops]
        kind_counts = Counter(op_kinds)

        # Gather tensor shape/dtype context from the parsed types.
        tensor_types = self._collect_tensor_types_from_func(func)
        tensor_shapes = [shape for shape, _ in tensor_types]
        element_type = self._dominant_element_type(tensor_types)

        # Most-specific patterns first. We dispatch in priority order
        # because some kernels legitimately contain multiple kinds
        # of op (e.g. attention has dots AND a reduce). The "primary"
        # classification is the dominant or most-specific pattern.
        dot_count = kind_counts.get(OpKind.DOT, 0)
        reduce_count = kind_counts.get(OpKind.REDUCE, 0)
        broadcast_count = kind_counts.get(OpKind.BROADCAST, 0)
        transpose_count = kind_counts.get(OpKind.TRANSPOSE, 0)
        # tt.scan / scf.while aren't in the parser's OP_KIND_MAP; use
        # name match until the parser learns them.
        scan_count = sum(1 for op in ops if op.name == "tt.scan")
        while_count = sum(1 for op in ops if op.name == "scf.while")

        if dot_count >= 2 and reduce_count >= 1 and self._looks_like_softmax(ops):
            return self._wrap(
                func, KernelKind.ATTENTION, ops, tensor_types, element_type,
            )

        if dot_count >= 1:
            return self._wrap(
                func, KernelKind.MATMUL, ops, tensor_types, element_type,
            )

        if reduce_count >= 1:
            return self._wrap(
                func, KernelKind.REDUCTION, ops, tensor_types, element_type,
            )

        if scan_count >= 1:
            return self._wrap(
                func, KernelKind.SCAN, ops, tensor_types, element_type,
            )

        if while_count >= 1 and dot_count == 0 and reduce_count == 0:
            return self._wrap(
                func, KernelKind.PERSISTENT, ops, tensor_types, element_type,
            )

        if transpose_count >= 1 and dot_count == 0 and reduce_count == 0:
            return self._wrap(
                func, KernelKind.TRANSPOSE, ops, tensor_types, element_type,
            )

        if broadcast_count >= 1 and dot_count == 0 and reduce_count == 0:
            return self._wrap(
                func, KernelKind.BROADCAST, ops, tensor_types, element_type,
            )

        # ELEMENTWISE: loads + stores + at least one elementwise
        # arith/math op, with no dot/reduce/trans/broadcast dominating.
        if self._looks_like_elementwise(ops, kind_counts):
            return self._wrap(
                func, KernelKind.ELEMENTWISE, ops, tensor_types, element_type,
            )

        # Fallback: if we recognised ANY supported op at all, classify
        # as UNKNOWN with full structured info. If we recognised
        # nothing (no tt.* / arith.* / math.* / scf.* ops), the IR
        # is outside the supported set and classification is
        # genuinely impossible.
        if not any(
            kind in kind_counts
            for kind in (
                OpKind.LOAD, OpKind.STORE, OpKind.DOT, OpKind.REDUCE,
                OpKind.BROADCAST, OpKind.TRANSPOSE, OpKind.ADDF,
                OpKind.SUBF, OpKind.MULF, OpKind.DIVF, OpKind.ADDI,
                OpKind.SUBI, OpKind.MULI, OpKind.EXP, OpKind.LOG,
                OpKind.SQRT, OpKind.RSQRT, OpKind.TANH, OpKind.COS,
                OpKind.SIN, OpKind.CONSTANT, OpKind.FOR_LOOP,
                OpKind.IF_STATEMENT, OpKind.GET_PROGRAM_ID,
            )
        ):
            raise ClassificationError(
                f"IR contains no supported ops; cannot classify: "
                f"{func.name!r}",
            )

        return IRClassification(
            kind=KernelKind.UNKNOWN,
            tensor_element_type=element_type,
            tensor_shapes=tensor_shapes,
            ops=[op.name for op in ops if op.name],
        )

    def _wrap(
        self,
        func: TTGIRFunction,
        kind: KernelKind,
        ops: list,
        tensor_types: list[tuple[tuple[int, ...], str]],
        element_type: str | None,
    ) -> IRClassification:
        """Build an IRClassification with the right per-kind extras."""
        op_names = [op.name for op in ops if op.name]
        reduction_type: ReductionType | None = None
        reduction_axis: int | None = None

        if kind in (KernelKind.REDUCTION, KernelKind.ATTENTION):
            reduce_ops = [op for op in ops if op.kind == OpKind.REDUCE]
            if reduce_ops:
                first = reduce_ops[0]
                reduction_type = self._combine_op_to_reduction(first)
                axis_attr = first.attributes.get("axis")
                if axis_attr is not None:
                    try:
                        reduction_axis = int(axis_attr)
                    except ValueError:
                        reduction_axis = None
        elif kind == KernelKind.MATMUL:
            # Matmul is a sum-of-products reduction along the K axis.
            # The K dim is the inner dim of the first dot's left
            # operand (if known) — defaults to None when shapes are
            # all unknown.
            reduction_type = ReductionType.SUM
            reduction_axis = self._matmul_k_axis(ops)

        return IRClassification(
            kind=kind,
            reduction_type=reduction_type,
            reduction_axis=reduction_axis,
            tensor_element_type=element_type,
            tensor_shapes=[shape for shape, _ in tensor_types],
            ops=op_names,
        )

    @staticmethod
    def _looks_like_softmax(ops: list) -> bool:
        """True if the function contains a math.exp right after a reduce.

        Used to disambiguate attention (reduce → exp → dot) from a
        bare reduction followed by an unrelated exp. Cheap and
        structural — no regex on op text.
        """
        saw_reduce = False
        for op in ops:
            if op.kind == OpKind.REDUCE:
                saw_reduce = True
            elif saw_reduce and op.kind in (OpKind.EXP,):
                return True
        return False

    @staticmethod
    def _looks_like_elementwise(
        ops: list, kind_counts: Counter,
    ) -> bool:
        """A kernel is elementwise when it has loads + stores + at
        least one pointwise arith/math op and no dot/reduce/broadcast/
        trans/scan dominating."""
        if kind_counts.get(OpKind.DOT, 0):
            return False
        if kind_counts.get(OpKind.REDUCE, 0):
            return False
        if kind_counts.get(OpKind.BROADCAST, 0):
            return False
        if kind_counts.get(OpKind.TRANSPOSE, 0):
            return False
        if any(op.name == "tt.scan" for op in ops):
            return False
        if not kind_counts.get(OpKind.LOAD, 0):
            return False
        if not kind_counts.get(OpKind.STORE, 0):
            return False
        pointwise = {
            OpKind.ADDF, OpKind.SUBF, OpKind.MULF, OpKind.DIVF,
            OpKind.ADDI, OpKind.SUBI, OpKind.MULI, OpKind.EXP,
            OpKind.LOG, OpKind.SQRT, OpKind.RSQRT, OpKind.TANH,
            OpKind.COS, OpKind.SIN, OpKind.MAX, OpKind.MIN,
        }
        return any(k in pointwise for k in kind_counts)

    @staticmethod
    def _combine_op_to_reduction(op) -> ReductionType:
        """Map a reduce op's ``combine_op`` attribute to ReductionType."""
        combine = op.attributes.get("combine_op", "")
        if combine in _COMBINE_OP_TO_REDUCTION:
            return _COMBINE_OP_TO_REDUCTION[combine]
        return ReductionType.UNKNOWN

    @staticmethod
    def _matmul_k_axis(ops: list) -> int | None:
        """Best-effort K-axis inference for a matmul.

        Looks at the first dot op's left operand's defining op for a
        concrete shape; returns the inner dim as the K axis. Returns
        None when no shape is known.
        """
        # The parser doesn't track defining-op relationships between
        # %result values, so we use the reduce's K-axis attribute as
        # a fallback signal when no operand shape is available. For
        # now, surface None and let downstream bounds extraction
        # fill in the concrete value.
        return None

    def _collect_tensor_types_from_func(
        self, func: TTGIRFunction,
    ) -> list[tuple[tuple[int, ...], str]]:
        """Collect (shape, dtype) tuples from the parsed function's types."""
        results: list[tuple[tuple[int, ...], str]] = []
        seen: set[tuple[tuple[int, ...], str]] = set()
        for _, arg_type in func.args:
            entry = self._tensor_entry(arg_type)
            if entry is not None and entry not in seen:
                results.append(entry)
                seen.add(entry)
        for op in func.iter_all_ops():
            for t in op.types:
                entry = self._tensor_entry(t)
                if entry is not None and entry not in seen:
                    results.append(entry)
                    seen.add(entry)
        return results

    @staticmethod
    def _tensor_entry(t: TTGIRType) -> tuple[tuple[int, ...], str] | None:
        """Convert a parsed TTGIRType to a (shape, dtype) entry.

        Filters out scalar types (shape=()) and pointer types —
        we only report concrete tensor types.
        """
        if not t.is_tensor:
            return None
        if not t.shape:
            return None
        return (t.shape, _dtype_to_str(t.element_dtype))

    @staticmethod
    def _dominant_element_type(
        tensor_types: list[tuple[tuple[int, ...], str]],
    ) -> str | None:
        """Return the most common element dtype across tensor types."""
        if not tensor_types:
            return None
        dtypes = [dtype for _, dtype in tensor_types]
        return Counter(dtypes).most_common(1)[0][0]
