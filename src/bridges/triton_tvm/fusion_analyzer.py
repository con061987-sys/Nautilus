"""Fusion analyzer — graph pattern matching engine for fusion opportunities.

Identifies sequences of ops within a parsed TTGIR function that are
candidates for kernel fusion (e.g. matmul+activation, elementwise
chains, reduction+elementwise). Each match includes a confidence
score derived from SSA data dependency verification between the
ops in the pattern.

Integration with ``ir_classifier.py``:
  The analyzer consumes a parsed ``TTGIRFunction`` (already available
  from the parser that the classifier uses). It does **not** call
  ``IRClassifier.classify()`` — that would classify the whole function
  as a single kind. Instead, it maps each *individual* operation's
  ``OpKind`` to a coarse ``KernelKind`` category and scans the
  resulting sequence for fusion pattern signatures.

Integration with ``kernel_fusion.py`` (Task 15):
  The return type is ``list[PatternMatch]``, which kernel_fusion
  should consume to drive actual fusion transformations. Each
  ``PatternMatch`` carries the function-level indices of the matched
  ops so the fusion pass knows exactly which ops to fuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from src.common.logging import get_logger

from .ir_capture import KernelKind
from .ir_to_tir.ttgir_parser import OpKind, TTGIRFunction, TTGIROperation

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Confidence levels
# ---------------------------------------------------------------------------


class Confidence(Enum):
    """Named confidence levels returned by data dependency verification.

    These are canonical tags returned by ``_verify_data_dependency``.
    Callers can compare against the enum or inspect ``.value`` for the
    float score.
    """

    NONE = auto()  # 0.0 — no data flow at all (pattern is spurious)
    STRUCTURAL = auto()  # 0.5 — kind-sequence matches but no dep verified
    PARTIAL = auto()  # 0.85 — some pairs verified
    FULL = auto()  # 1.0 — every adjacent pair has an SSA chain


_CONFIDENCE_VALUES: dict[Confidence, float] = {
    Confidence.NONE: 0.0,
    Confidence.STRUCTURAL: 0.5,
    Confidence.PARTIAL: 0.85,
    Confidence.FULL: 1.0,
}


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PatternMatch:
    """A single fusion pattern match within a parsed function.

    Attributes:
        pattern_name: Human-readable name
            (e.g. ``"matmul_activation"``).
        start_op: Index into the function's flat op list
            (``iter_all_ops()`` order) where the pattern starts.
        end_op: Index (inclusive) where the pattern ends.
        confidence: Float in ``[0, 1]`` indicating how likely this is
            a real fusion opportunity. ``1.0`` = confirmed SSA chain;
            ``0.5`` = structural match without verified data flow.
        matched_kinds: The ``KernelKind`` sequence that was matched.
    """

    pattern_name: str
    start_op: int
    end_op: int
    confidence: float
    matched_kinds: list[KernelKind]


# ---------------------------------------------------------------------------
# Fusion pattern definitions
# ---------------------------------------------------------------------------
#
# Each pattern is a sequence of KernelKind values.  The order matters —
# patterns are scanned left-to-right over the function's kernel-kind
# sequence (loads, stores, constants, control flow are filtered out).
#
# 5 required patterns:
#   1. matmul_activation       — tt.dot + arith/math (ReLU, tanh, …)
#   2. matmul_bias_activation  — tt.dot + bias add + activation
#   3. elementwise_chain       — three+ pointwise ops in sequence
#   4. reduction_elementwise   — tt.reduce + pointwise op
#   5. matmul_broadcast_add    — tt.dot + tt.broadcast + arith.addf

FUSION_PATTERNS: dict[str, list[KernelKind]] = {
    "matmul_activation": [KernelKind.MATMUL, KernelKind.ELEMENTWISE],
    "matmul_bias_activation": [
        KernelKind.MATMUL,
        KernelKind.ELEMENTWISE,
        KernelKind.ELEMENTWISE,
    ],
    "elementwise_chain": [
        KernelKind.ELEMENTWISE,
        KernelKind.ELEMENTWISE,
        KernelKind.ELEMENTWISE,
    ],
    "reduction_elementwise": [KernelKind.REDUCTION, KernelKind.ELEMENTWISE],
    "matmul_broadcast_add": [
        KernelKind.MATMUL,
        KernelKind.BROADCAST,
        KernelKind.ELEMENTWISE,
    ],
}

# -- Optional extra patterns -------------------------------------------------
# Beyond the required five, these patterns cover more fusion scenarios
# that real backends like TVM and XLA support.

FUSION_PATTERNS_EXTRA: dict[str, list[KernelKind]] = {
    "transpose_elementwise": [KernelKind.TRANSPOSE, KernelKind.ELEMENTWISE],
    "elementwise_reduction": [KernelKind.ELEMENTWISE, KernelKind.REDUCTION],
}

# ---------------------------------------------------------------------------
# OpKind → KernelKind mapping
# ---------------------------------------------------------------------------
#
# Every operation that can appear in a fusion pattern maps from its
# fine-grained OpKind to a coarse KernelKind category.  Ops not in
# this map (loads, stores, constants, control flow) are treated as
# infrastructure and excluded from pattern matching.

_KERNEL_KIND_FOR_OP: dict[OpKind, KernelKind] = {
    # Matmul
    OpKind.DOT: KernelKind.MATMUL,
    # Reduction
    OpKind.REDUCE: KernelKind.REDUCTION,
    # Layout
    OpKind.BROADCAST: KernelKind.BROADCAST,
    OpKind.TRANSPOSE: KernelKind.TRANSPOSE,
    # Elementwise — arithmetic
    OpKind.ADDF: KernelKind.ELEMENTWISE,
    OpKind.SUBF: KernelKind.ELEMENTWISE,
    OpKind.MULF: KernelKind.ELEMENTWISE,
    OpKind.DIVF: KernelKind.ELEMENTWISE,
    OpKind.ADDI: KernelKind.ELEMENTWISE,
    OpKind.SUBI: KernelKind.ELEMENTWISE,
    OpKind.MULI: KernelKind.ELEMENTWISE,
    OpKind.MAX: KernelKind.ELEMENTWISE,
    OpKind.MIN: KernelKind.ELEMENTWISE,
    # Elementwise — transcendental
    OpKind.EXP: KernelKind.ELEMENTWISE,
    OpKind.LOG: KernelKind.ELEMENTWISE,
    OpKind.SQRT: KernelKind.ELEMENTWISE,
    OpKind.RSQRT: KernelKind.ELEMENTWISE,
    OpKind.TANH: KernelKind.ELEMENTWISE,
    OpKind.COS: KernelKind.ELEMENTWISE,
    OpKind.SIN: KernelKind.ELEMENTWISE,
    # Elementwise — data movement / misc
    OpKind.RESHAPE: KernelKind.ELEMENTWISE,
    OpKind.GET_PROGRAM_ID: KernelKind.ELEMENTWISE,
    OpKind.GET_NUM_PROGRAMS: KernelKind.ELEMENTWISE,
    OpKind.ADDPTR: KernelKind.ELEMENTWISE,
    OpKind.MAKE_TENSOR_PTR: KernelKind.ELEMENTWISE,
    OpKind.ADVANCE: KernelKind.ELEMENTWISE,
    # Pass-4 materialisation targets (treated as elementwise for fusion)
    OpKind.TVM_BLOCK: KernelKind.ELEMENTWISE,
    OpKind.TVM_INIT: KernelKind.ELEMENTWISE,
    OpKind.ALLOC_BUFFER: KernelKind.ELEMENTWISE,
    # Triton return (no semantic weight)
    OpKind.RETURN: KernelKind.ELEMENTWISE,
}

# OpKind values that are infrastructure and skipped entirely.
_SKIPPED_OP_KINDS: frozenset[OpKind] = frozenset({
    OpKind.LOAD,
    OpKind.STORE,
    OpKind.CONSTANT,
    OpKind.FOR_LOOP,
    OpKind.IF_STATEMENT,
    OpKind.YIELD,
    OpKind.UNKNOWN,
})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _op_to_kernel_kind(op: TTGIROperation) -> KernelKind | None:
    """Map a parsed operation to its coarse kernel-kind category.

    Returns ``None`` for infrastructure ops (loads, stores, constants,
    control flow) that should be excluded from pattern matching.

    Ops whose ``OpKind`` is not in the mapping at all also return
    ``None`` — they are not part of any known fusion pattern.
    """
    if op.kind in _SKIPPED_OP_KINDS:
        return None
    return _KERNEL_KIND_FOR_OP.get(op.kind)


def _verify_data_dependency(
    ops_in_pattern: list[TTGIROperation],
) -> float:
    """Verify SSA data flow between ops that form a pattern match.

    This checks **only** the ops that contributed to the kernel-kind
    sequence (infrastructure ops between them are irrelevant to the
    data-flow check).

    For each adjacent pair ``(op_i, op_{i+1})`` in the matched
    sequence, we verify that ``op_i.result_name`` appears in
    ``op_{i+1}.operands``.

    Returns:
        A float confidence score:
        - ``1.0`` (FULL) — every adjacent pair has a direct SSA chain.
        - ``0.85`` (PARTIAL) — at least half the pairs verify.
        - ``0.5`` (STRUCTURAL) — no pairs verified, but the kind
          sequence matched.
        - ``0.0`` (NONE) — only one op (degenerate), or the kind
          sequence matched but all pairs failed verification.
    """
    total_pairs = len(ops_in_pattern) - 1
    if total_pairs <= 0:
        return _CONFIDENCE_VALUES[Confidence.NONE]

    chain_verified = 0
    for i in range(total_pairs):
        current = ops_in_pattern[i]
        next_op = ops_in_pattern[i + 1]
        if current.result_name and current.result_name in next_op.operands:
            chain_verified += 1

    if chain_verified == total_pairs:
        return _CONFIDENCE_VALUES[Confidence.FULL]
    if chain_verified > 0 and chain_verified >= total_pairs // 2:
        return _CONFIDENCE_VALUES[Confidence.PARTIAL]

    # Structural match only — composition matched but no SSA chain found.
    return _CONFIDENCE_VALUES[Confidence.STRUCTURAL]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class FusionAnalyzer:
    """Analyze a parsed TTGIR function for fusion opportunities.

    Usage::

        parser = TTGIRParser()
        func = parser.parse(ir_text)
        analyzer = FusionAnalyzer()
        matches = analyzer.analyze(func)

    Each ``PatternMatch`` in the returned list carries the indices
    into the function's flat op list and a confidence score so that
    ``kernel_fusion`` (Task 15) can decide which matches to act on.
    """

    def __init__(self, include_extra_patterns: bool = False) -> None:
        """Initialize the analyzer.

        Args:
            include_extra_patterns: If ``True``, also check the
                optional patterns defined in ``FUSION_PATTERNS_EXTRA``.
                Defaults to ``False``.
        """
        self._include_extra = include_extra_patterns

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, func: TTGIRFunction) -> list[PatternMatch]:
        """Analyze a parsed function and return fusion candidates.

        Walks the function's flat op list in depth-first order
        (``iter_all_ops()``), maps each non-infrastructure op to a
        ``KernelKind``, and scans for known fusion patterns using a
        sliding window over the resulting kind sequence.

        Args:
            func: A parsed ``TTGIRFunction`` from the TTGIR parser.

        Returns:
            A list of ``PatternMatch`` objects describing each fusion
            opportunity found, ordered by their position in the IR.
            Returns an empty list if no patterns match or the function
            has no ops after filtering infrastructure.

        Note:
            The matched ops' indices (``start_op`` / ``end_op``)
            refer to positions in **``list(func.iter_all_ops())``**,
            which includes nested ops from scf.for/scf.if bodies.
            Consumers should use ``list(func.iter_all_ops())[idx]``
            to retrieve the actual ``TTGIROperation``.
        """
        all_ops = list(func.iter_all_ops())
        if not all_ops:
            return []

        # Build the kernel-kind sequence: (original_index, op, kind).
        kind_seq: list[tuple[int, TTGIROperation, KernelKind]] = []
        for idx, op in enumerate(all_ops):
            kind = _op_to_kernel_kind(op)
            if kind is not None:
                kind_seq.append((idx, op, kind))

        if not kind_seq:
            return []

        matches: list[PatternMatch] = []
        patterns = dict(FUSION_PATTERNS)
        if self._include_extra:
            patterns.update(FUSION_PATTERNS_EXTRA)

        for pattern_name, pattern_kinds in patterns.items():
            self._scan_pattern(
                pattern_name,
                pattern_kinds,
                kind_seq,
                all_ops,
                matches,
            )

        # Sort by position in IR (start_op ascending).
        matches.sort(key=lambda m: m.start_op)
        return matches

    def analyze_text(
        self,
        ir_text: str,
        parser=None,
    ) -> list[PatternMatch]:
        """Convenience: parse IR text and analyze in one call.

        Args:
            ir_text: Raw TTGIR text.
            parser: A ``TTGIRParser`` instance. If ``None``, a new
                parser is created on each call.

        Returns:
            List of ``PatternMatch`` objects, same as ``analyze()``.
        """
        if parser is None:
            from .ir_to_tir.ttgir_parser import TTGIRParser as _Parser

            parser = _Parser()
        func = parser.parse(ir_text)
        return self.analyze(func)

    @staticmethod
    def has_fusion_opportunity(
        func_or_text: TTGIRFunction | str,
        parser=None,
    ) -> bool:
        """Quick check: does this function contain any fusion opportunity?

        Useful for fast-path filtering before running the full analysis.

        Args:
            func_or_text: A parsed ``TTGIRFunction`` or raw TTGIR text.
            parser: Required if ``func_or_text`` is a string.

        Returns:
            ``True`` if at least one pattern matched.
        """
        analyzer = FusionAnalyzer()
        if isinstance(func_or_text, TTGIRFunction):
            return len(analyzer.analyze(func_or_text)) > 0
        return len(analyzer.analyze_text(func_or_text, parser)) > 0

    @staticmethod
    def list_patterns(include_extra: bool = False) -> list[str]:
        """Return the names of all registered fusion patterns.

        Args:
            include_extra: Include optional extra patterns.

        Returns:
            List of pattern name strings.
        """
        names = list(FUSION_PATTERNS.keys())
        if include_extra:
            names.extend(FUSION_PATTERNS_EXTRA.keys())
        return names

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _scan_pattern(
        pattern_name: str,
        pattern_kinds: list[KernelKind],
        kind_seq: list[tuple[int, TTGIROperation, KernelKind]],
        all_ops: list[TTGIROperation],
        matches: list[PatternMatch],
    ) -> None:
        """Scan the kernel-kind sequence for a single pattern.

        Uses a sliding window over ``kind_seq``.  When a match is
        found, the data dependency verifier runs over the matched
        ops, and the result is appended to ``matches``.

        Args:
            pattern_name: Name of the pattern being scanned.
            pattern_kinds: The KernelKind sequence to match.
            kind_seq: The function's kernel-kind sequence.
            all_ops: The full (unfiltered) op list, used for
                dependency verification.
            matches: Accumulator list mutated in-place.
        """
        pattern_len = len(pattern_kinds)
        if len(kind_seq) < pattern_len:
            return

        for i in range(len(kind_seq) - pattern_len + 1):
            window = kind_seq[i : i + pattern_len]

            # Check if the window's kinds match the pattern.
            matched = True
            for j, (_, _, kind) in enumerate(window):
                if kind != pattern_kinds[j]:
                    matched = False
                    break

            if not matched:
                continue

            # Gather the matched ops for dependency verification.
            orig_start = window[0][0]
            orig_end = window[-1][0]
            ops_in_pattern = [entry[1] for entry in window]

            confidence = _verify_data_dependency(ops_in_pattern)

            # Skip patterns with no verified data flow at all.
            if confidence <= _CONFIDENCE_VALUES[Confidence.NONE]:
                logger.debug(
                    "Skipping pattern %r at ops [%d, %d] — no data flow",
                    pattern_name,
                    orig_start,
                    orig_end,
                )
                continue

            match = PatternMatch(
                pattern_name=pattern_name,
                start_op=orig_start,
                end_op=orig_end,
                confidence=confidence,
                matched_kinds=list(pattern_kinds),
            )
            matches.append(match)

            logger.debug(
                "Fusion match: %s at ops [%d, %d] confidence=%.2f",
                pattern_name,
                orig_start,
                orig_end,
                confidence,
            )
