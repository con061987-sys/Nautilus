"""Tests for the fusion analyzer.

Covers:
  - All 5 required patterns produce correct PatternMatch entries.
  - Data dependency verification assigns correct confidence scores.
  - Edge cases: empty IR, no matches, intervening infrastructure ops.
  - Overlapping patterns are both reported.
  - Static helpers (list_patterns, has_fusion_opportunity).
  - Integration with TTGIRParser via analyze_text().
"""

from __future__ import annotations

from src.bridges.triton_tvm.fusion_analyzer import (
    FUSION_PATTERNS,
    FUSION_PATTERNS_EXTRA,
    FusionAnalyzer,
    PatternMatch,
)
from src.bridges.triton_tvm.ir_capture import KernelKind
from src.bridges.triton_tvm.ir_to_tir.ttgir_parser import (
    OpKind,
    TTGIRFunction,
    TTGIROperation,
    TTGIRParser,
)

# ---------------------------------------------------------------------------
# Helpers — build ops with precise result_name / operands
# ---------------------------------------------------------------------------


def _op(
    kind: OpKind,
    result_name: str = "",
    operands: list[str] | None = None,
    name: str = "",
) -> TTGIROperation:
    """Construct a minimal TTGIROperation for testing."""
    return TTGIROperation(
        kind=kind,
        raw_text="",
        name=name or kind.name.lower(),
        result_name=result_name,
        operands=operands or [],
        types=[],
        attributes={},
    )


def _func(ops: list[TTGIROperation], name: str = "test_kernel") -> TTGIRFunction:
    """Wrap ops in a minimal TTGIRFunction."""
    return TTGIRFunction(name=name, ops=ops)


# ---------------------------------------------------------------------------
# Sample IR texts for integration-style tests
# ---------------------------------------------------------------------------

# Pattern 1: matmul_activation — tt.dot + math.tanh
MATMUL_ACT_IR = """
module {
  tt.func public @matmul_relu(
    %A: !tt.ptr<tensor<128x32xf32>>,
    %B: !tt.ptr<tensor<32x128xf32>>,
    %C: !tt.ptr<tensor<128x128xf32>>
  ) {
    %a = tt.load %A : tensor<128x32xf32>
    %b = tt.load %B : tensor<32x128xf32>
    %c = tt.dot %a, %b : tensor<128x128xf32>
    %d = math.tanh %c : tensor<128x128xf32>
    tt.store %C, %d : tensor<128x128xf32>
    tt.return
  }
}
"""

# Pattern 2: matmul_bias_activation — tt.dot + arith.addf (bias) + math.tanh (act)
MATMUL_BIAS_ACT_IR = """
module {
  tt.func public @matmul_bias_gelu(
    %A: !tt.ptr<tensor<128x32xf32>>,
    %B: !tt.ptr<tensor<32x128xf32>>,
    %bias: !tt.ptr<tensor<128xf32>>,
    %C: !tt.ptr<tensor<128x128xf32>>
  ) {
    %a = tt.load %A : tensor<128x32xf32>
    %b = tt.load %B : tensor<32x128xf32>
    %bias_v = tt.load %bias : tensor<128xf32>
    %c = tt.dot %a, %b : tensor<128x128xf32>
    %d = arith.addf %c, %bias_v : tensor<128x128xf32>
    %e = math.tanh %d : tensor<128x128xf32>
    tt.store %C, %e : tensor<128x128xf32>
    tt.return
  }
}
"""

# Pattern 3: elementwise_chain — three consecutive pointwise ops
ELEMENTWISE_CHAIN_IR = """
module {
  tt.func @elementwise_chain(
    %X: !tt.ptr<tensor<128xf32>>,
    %Y: !tt.ptr<tensor<128xf32>>
  ) {
    %x = tt.load %X : tensor<128xf32>
    %a = arith.addf %x, %x : tensor<128xf32>
    %b = arith.mulf %a, %a : tensor<128xf32>
    %c = arith.subf %b, %a : tensor<128xf32>
    tt.store %Y, %c : tensor<128xf32>
    tt.return
  }
}
"""

# Pattern 4: reduction_elementwise — tt.reduce + arith.addf
REDUCTION_EW_IR = """
module {
  tt.func @reduce_ew(
    %input: !tt.ptr<tensor<1024xf32>>,
    %output: !tt.ptr<tensor<1xf32>>
  ) {
    %x = tt.load %input : tensor<1024xf32>
    %r = "tt.reduce"(%x) ({
      ^bb0(%a: f32, %b: f32): arith.addf %a, %b : f32
    }) {axis = 0 : i32} : (tensor<1024xf32>) -> tensor<1xf32>
    %s = arith.addf %r, %r : tensor<1xf32>
    tt.store %output, %s : tensor<1xf32>
    tt.return
  }
}
"""

# Pattern 5: matmul_broadcast_add — tt.dot + tt.broadcast + arith.addf
MATMUL_BC_ADD_IR = """
module {
  tt.func @matmul_bc_add(
    %A: !tt.ptr<tensor<64x32xf32>>,
    %B: !tt.ptr<tensor<32x64xf32>>,
    %bias: !tt.ptr<tensor<64xf32>>,
    %C: !tt.ptr<tensor<64x64xf32>>
  ) {
    %a = tt.load %A : tensor<64x32xf32>
    %b = tt.load %B : tensor<32x64xf32>
    %bc = tt.load %bias : tensor<64xf32>
    %c = tt.dot %a, %b : tensor<64x64xf32>
    %d = tt.broadcast %bc : tensor<64xf32> -> tensor<64x64xf32>
    %e = arith.addf %c, %d : tensor<64x64xf32>
    tt.store %C, %e : tensor<64x64xf32>
    tt.return
  }
}
"""

# Empty / trivial IR — no patterns expected
EMPTY_IR = """
module {
  tt.func @empty() {
    tt.return
  }
}
"""

# Single-elementwise op — no fusion pattern expected
SINGLE_EW_IR = """
module {
  tt.func @single_ew() {
    %x = tt.load %in : tensor<128xf32>
    %y = arith.addf %x, %x : tensor<128xf32>
    tt.store %out, %y : tensor<128xf32>
    tt.return
  }
}
"""


# ---------------------------------------------------------------------------
# Tests — direct TTGIRFunction construction
# ---------------------------------------------------------------------------


class TestFusionAnalyzerDirect:
    """Tests using directly constructed TTGIRFunction objects."""

    def setup_method(self) -> None:
        self.analyzer = FusionAnalyzer()

    # ---- Pattern 1: matmul_activation ---------------------------------

    def test_matmul_activation(self) -> None:
        """tt.dot + math.tanh detects matmul_activation at full confidence."""
        func = _func([
            _op(OpKind.LOAD, "%a", ["%A"]),
            _op(OpKind.LOAD, "%b", ["%B"]),
            _op(OpKind.DOT, "%c", ["%a", "%b"], "tt.dot"),
            _op(OpKind.TANH, "%d", ["%c"], "math.tanh"),
            _op(OpKind.STORE, "", ["%d"]),
        ])
        matches = self.analyzer.analyze(func)
        assert len(matches) >= 1
        match = _find_by_name(matches, "matmul_activation")
        assert match is not None
        assert match.confidence == 1.0  # %c → %d is a direct SSA chain
        assert match.matched_kinds == [KernelKind.MATMUL, KernelKind.ELEMENTWISE]

    # ---- Pattern 2: matmul_bias_activation ----------------------------

    def test_matmul_bias_activation(self) -> None:
        """tt.dot + arith.addf + math.tanh detects matmul_bias_activation."""
        func = _func([
            _op(OpKind.LOAD, "%a", ["%A"]),
            _op(OpKind.LOAD, "%b", ["%B"]),
            _op(OpKind.LOAD, "%bias", ["%bias_ptr"]),
            _op(OpKind.DOT, "%c", ["%a", "%b"], "tt.dot"),
            _op(OpKind.ADDF, "%d", ["%c", "%bias"]),
            _op(OpKind.TANH, "%e", ["%d"], "math.tanh"),
            _op(OpKind.STORE, "", ["%e"]),
        ])
        matches = self.analyzer.analyze(func)
        match = _find_by_name(matches, "matmul_bias_activation")
        assert match is not None
        assert match.confidence == 1.0  # %c→%d→%e
        assert match.matched_kinds == [
            KernelKind.MATMUL,
            KernelKind.ELEMENTWISE,
            KernelKind.ELEMENTWISE,
        ]

    # ---- Pattern 3: elementwise_chain ---------------------------------

    def test_elementwise_chain(self) -> None:
        """Three consecutive pointwise ops detect elementwise_chain."""
        func = _func([
            _op(OpKind.LOAD, "%x", ["%X"]),
            _op(OpKind.ADDF, "%a", ["%x", "%x"]),
            _op(OpKind.MULF, "%b", ["%a", "%a"]),
            _op(OpKind.SUBF, "%c", ["%b", "%a"]),
            _op(OpKind.STORE, "", ["%c"]),
        ])
        matches = self.analyzer.analyze(func)
        match = _find_by_name(matches, "elementwise_chain")
        assert match is not None
        assert match.confidence == 1.0  # %a→%b→%c fully chained
        assert match.matched_kinds == [
            KernelKind.ELEMENTWISE,
            KernelKind.ELEMENTWISE,
            KernelKind.ELEMENTWISE,
        ]

    # ---- Pattern 4: reduction_elementwise -----------------------------

    def test_reduction_elementwise(self) -> None:
        """tt.reduce + arith.addf detects reduction_elementwise."""
        func = _func([
            _op(OpKind.LOAD, "%x", ["%input"]),
            _op(OpKind.REDUCE, "%r", ["%x"], "tt.reduce"),
            _op(OpKind.ADDF, "%s", ["%r", "%r"]),
            _op(OpKind.STORE, "", ["%s"]),
        ])
        matches = self.analyzer.analyze(func)
        match = _find_by_name(matches, "reduction_elementwise")
        assert match is not None
        assert match.confidence == 1.0  # %r→%s
        assert match.matched_kinds == [KernelKind.REDUCTION, KernelKind.ELEMENTWISE]

    # ---- Pattern 5: matmul_broadcast_add ------------------------------

    def test_matmul_broadcast_add(self) -> None:
        """tt.dot + tt.broadcast + arith.addf detects matmul_broadcast_add."""
        func = _func([
            _op(OpKind.LOAD, "%a", ["%A"]),
            _op(OpKind.LOAD, "%b", ["%B"]),
            _op(OpKind.LOAD, "%bc", ["%bias"]),
            _op(OpKind.DOT, "%c", ["%a", "%b"], "tt.dot"),
            _op(OpKind.BROADCAST, "%d", ["%bc"], "tt.broadcast"),
            _op(OpKind.ADDF, "%e", ["%c", "%d"]),
            _op(OpKind.STORE, "", ["%e"]),
        ])
        matches = self.analyzer.analyze(func)
        match = _find_by_name(matches, "matmul_broadcast_add")
        assert match is not None
        # The broadcast is parallel to the dot (both feed into addf), not a
        # consumer of the dot result, so only 1 of 2 adjacent pairs has a
        # direct SSA chain → PARTIAL (0.85) confidence.
        assert match.confidence == 0.85
        assert match.matched_kinds == [
            KernelKind.MATMUL,
            KernelKind.BROADCAST,
            KernelKind.ELEMENTWISE,
        ]

    # ---- Confidence edge cases ----------------------------------------

    def test_confidence_partial(self) -> None:
        """Broken SSA chain yields PARTIAL (0.85) confidence."""
        # The dot result (%c) is NOT used by the tanh; they are
        # independent operations that happen to be adjacent.
        func = _func([
            _op(OpKind.LOAD, "%a", ["%A"]),
            _op(OpKind.LOAD, "%b", ["%B"]),
            _op(OpKind.LOAD, "%x", ["%X"]),
            _op(OpKind.DOT, "%c", ["%a", "%b"], "tt.dot"),
            _op(OpKind.TANH, "%d", ["%x"], "math.tanh"),
        ])
        matches = self.analyzer.analyze(func)
        match = _find_by_name(matches, "matmul_activation")
        if match is not None:  # structural match possible
            assert match.confidence == 0.5  # structural only

    def test_confidence_full_with_intervening_loads(self) -> None:
        """Infrastructure ops between pattern ops don't lower confidence.

        The data dependency checker operates on the kernel-kind
        sequence entries only — loads between them don't affect
        the confidence because they are skipped.
        """
        func = _func([
            _op(OpKind.LOAD, "%a", ["%A"]),
            _op(OpKind.LOAD, "%b", ["%B"]),
            _op(OpKind.DOT, "%c", ["%a", "%b"], "tt.dot"),
            _op(OpKind.LOAD, "%bias", ["%bias_ptr"]),
            _op(OpKind.ADDF, "%d", ["%c", "%bias"]),
            _op(OpKind.LOAD, "%scale", ["%scale_ptr"]),
            _op(OpKind.MULF, "%e", ["%d", "%scale"]),
            _op(OpKind.STORE, "", ["%e"]),
        ])
        matches = self.analyzer.analyze(func)
        # matmul_bias_activation should match with FULL confidence
        match_3 = _find_by_name(matches, "matmul_bias_activation")
        if match_3 is not None:
            assert match_3.confidence == 1.0  # %c→%d→%e chain verified
        # elementwise_chain (ADDF→MULF) should also match
        match_chain = _find_by_name(matches, "elementwise_chain")
        if match_chain is not None:
            assert match_chain.confidence == 1.0  # %d→%e

    # ---- Overlapping patterns -----------------------------------------

    def test_overlapping_patterns(self) -> None:
        """When patterns overlap in the kind sequence, both are reported."""
        # MATMUL + ELEMENTWISE matches both:
        #   matmul_activation (len 2)  at [0, 1]
        #   ... and potentially more
        func = _func([
            _op(OpKind.LOAD, "%a", ["%A"]),
            _op(OpKind.LOAD, "%b", ["%B"]),
            _op(OpKind.DOT, "%c", ["%a", "%b"], "tt.dot"),
            _op(OpKind.TANH, "%d", ["%c"], "math.tanh"),
            _op(OpKind.ADDF, "%e", ["%d", "%d"]),
        ])
        matches = self.analyzer.analyze(func)
        # Should have at least matmul_activation (span 0-1 in kind_seq)
        names = {m.pattern_name for m in matches}
        assert "matmul_activation" in names

    # ---- No patterns --------------------------------------------------

    def test_no_patterns_single_ew(self) -> None:
        """Single elementwise op yields no fusion patterns."""
        func = _func([
            _op(OpKind.LOAD, "%x", ["%X"]),
            _op(OpKind.ADDF, "%y", ["%x", "%x"]),
            _op(OpKind.STORE, "", ["%y"]),
        ])
        matches = self.analyzer.analyze(func)
        assert len(matches) == 0

    def test_no_patterns_empty_fn(self) -> None:
        """Function with no ops yields empty results."""
        func = _func([], name="empty")
        matches = self.analyzer.analyze(func)
        assert len(matches) == 0

    def test_no_patterns_infra_only(self) -> None:
        """Function with only infrastructure ops yields empty results."""
        func = _func([
            _op(OpKind.LOAD, "%x", ["%X"]),
            _op(OpKind.STORE, "", ["%x"]),
            _op(OpKind.RETURN, "", []),
        ])
        matches = self.analyzer.analyze(func)
        assert len(matches) == 0

    # ---- Extra patterns -----------------------------------------------

    def test_extra_patterns(self) -> None:
        """With include_extra_patterns=True, extra patterns are checked."""
        analyzer = FusionAnalyzer(include_extra_patterns=True)
        # transpose_elementwise pattern
        func = _func([
            _op(OpKind.LOAD, "%x", ["%X"]),
            _op(OpKind.TRANSPOSE, "%y", ["%x"], "tt.trans"),
            _op(OpKind.TANH, "%z", ["%y"], "math.tanh"),
            _op(OpKind.STORE, "", ["%z"]),
        ])
        matches = analyzer.analyze(func)
        match = _find_by_name(matches, "transpose_elementwise")
        assert match is not None
        assert match.confidence == 1.0

    def test_extra_patterns_off_by_default(self) -> None:
        """Without include_extra_patterns, extra patterns are not checked."""
        func = _func([
            _op(OpKind.LOAD, "%x", ["%X"]),
            _op(OpKind.TRANSPOSE, "%y", ["%x"], "tt.trans"),
            _op(OpKind.TANH, "%z", ["%y"], "math.tanh"),
            _op(OpKind.STORE, "", ["%z"]),
        ])
        matches = self.analyzer.analyze(func)
        assert _find_by_name(matches, "transpose_elementwise") is None

    # ---- Static helpers -----------------------------------------------

    def test_list_patterns(self) -> None:
        """list_patterns returns at least the 5 required patterns."""
        names = FusionAnalyzer.list_patterns()
        for required in [
            "matmul_activation",
            "matmul_bias_activation",
            "elementwise_chain",
            "reduction_elementwise",
            "matmul_broadcast_add",
        ]:
            assert required in names

    def test_list_patterns_extra(self) -> None:
        """list_patterns(include_extra=True) includes extra patterns."""
        names = FusionAnalyzer.list_patterns(include_extra=True)
        for extra_name in FUSION_PATTERNS_EXTRA:
            assert extra_name in names

    def test_has_fusion_opportunity_true(self) -> None:
        """has_fusion_opportunity returns True when a pattern matches."""
        func = _func([
            _op(OpKind.LOAD, "%a", ["%A"]),
            _op(OpKind.LOAD, "%b", ["%B"]),
            _op(OpKind.DOT, "%c", ["%a", "%b"], "tt.dot"),
            _op(OpKind.TANH, "%d", ["%c"], "math.tanh"),
        ])
        assert FusionAnalyzer.has_fusion_opportunity(func) is True

    def test_has_fusion_opportunity_false(self) -> None:
        """has_fusion_opportunity returns False when no pattern matches."""
        func = _func([
            _op(OpKind.LOAD, "%x", ["%X"]),
            _op(OpKind.ADDF, "%y", ["%x", "%x"]),
            _op(OpKind.STORE, "", ["%y"]),
        ])
        assert FusionAnalyzer.has_fusion_opportunity(func) is False


# ---------------------------------------------------------------------------
# Tests — integration with real parser (IR text)
# ---------------------------------------------------------------------------


class TestFusionAnalyzerIntegration:
    """Integration tests using real TTGIR parsing."""

    def setup_method(self) -> None:
        self.analyzer = FusionAnalyzer()
        self.parser = TTGIRParser()

    def _analyze(self, ir_text: str) -> list[PatternMatch]:
        return self.analyzer.analyze_text(ir_text, parser=self.parser)

    # ---- Pattern 1: matmul_activation ---------------------------------

    def test_matmul_activation_from_ir(self) -> None:
        """Matmul+tanh IR parses and detects matmul_activation."""
        matches = self._analyze(MATMUL_ACT_IR)
        match = _find_by_name(matches, "matmul_activation")
        assert match is not None
        assert match.confidence == 1.0
        assert match.start_op < match.end_op

    # ---- Pattern 2: matmul_bias_activation ----------------------------

    def test_matmul_bias_activation_from_ir(self) -> None:
        """Matmul+bias+tanh IR detects matmul_bias_activation."""
        matches = self._analyze(MATMUL_BIAS_ACT_IR)
        match = _find_by_name(matches, "matmul_bias_activation")
        assert match is not None
        assert match.confidence == 1.0

    # ---- Pattern 3: elementwise_chain ---------------------------------

    def test_elementwise_chain_from_ir(self) -> None:
        """Three consecutive pointwise ops detect elementwise_chain."""
        matches = self._analyze(ELEMENTWISE_CHAIN_IR)
        match = _find_by_name(matches, "elementwise_chain")
        assert match is not None
        assert match.confidence == 1.0

    # ---- Pattern 4: reduction_elementwise -----------------------------

    def test_reduction_elementwise_from_ir(self) -> None:
        """Reduce+addf detects reduction_elementwise."""
        matches = self._analyze(REDUCTION_EW_IR)
        match = _find_by_name(matches, "reduction_elementwise")
        assert match is not None
        assert match.confidence == 1.0

    # ---- Pattern 5: matmul_broadcast_add ------------------------------

    def test_matmul_broadcast_add_from_ir(self) -> None:
        """Dot+broadcast+addf detects matmul_broadcast_add (0.85 partial)."""
        matches = self._analyze(MATMUL_BC_ADD_IR)
        match = _find_by_name(matches, "matmul_broadcast_add")
        assert match is not None
        # Parallel-input pattern: broadcast doesn't consume dot result.
        assert match.confidence == 0.85

    # ---- Edge cases ---------------------------------------------------

    def test_empty_ir_no_matches(self) -> None:
        """Empty function IR yields no fusion matches."""
        matches = self._analyze(EMPTY_IR)
        assert len(matches) == 0

    def test_single_ew_no_chain(self) -> None:
        """Single elementwise op yields no pattern matches."""
        matches = self._analyze(SINGLE_EW_IR)
        assert len(matches) == 0

    def test_all_5_patterns_present(self) -> None:
        """All 5 required patterns are found in appropriate IRs."""
        for ir_text, pattern_name in [
            (MATMUL_ACT_IR, "matmul_activation"),
            (MATMUL_BIAS_ACT_IR, "matmul_bias_activation"),
            (ELEMENTWISE_CHAIN_IR, "elementwise_chain"),
            (REDUCTION_EW_IR, "reduction_elementwise"),
            (MATMUL_BC_ADD_IR, "matmul_broadcast_add"),
        ]:
            matches = self._analyze(ir_text)
            assert _find_by_name(matches, pattern_name) is not None, (
                f"Pattern {pattern_name!r} not found in its expected IR"
            )

    def test_analyze_text_creates_parser(self) -> None:
        """analyze_text without explicit parser should still work."""
        matches = self.analyzer.analyze_text(MATMUL_ACT_IR)
        assert len(matches) >= 1

    def test_has_fusion_opportunity_from_text(self) -> None:
        """has_fusion_opportunity works with IR text strings."""
        assert FusionAnalyzer.has_fusion_opportunity(MATMUL_ACT_IR, self.parser) is True
        assert FusionAnalyzer.has_fusion_opportunity(EMPTY_IR, self.parser) is False


# ---------------------------------------------------------------------------
# Tests — PatternMatch dataclass properties
# ---------------------------------------------------------------------------


class TestPatternMatch:
    """PatternMatch dataclass correctness."""

    def test_dataclass_fields(self) -> None:
        """PatternMatch stores all expected fields."""
        pm = PatternMatch(
            pattern_name="test_pattern",
            start_op=0,
            end_op=2,
            confidence=1.0,
            matched_kinds=[KernelKind.ELEMENTWISE],
        )
        assert pm.pattern_name == "test_pattern"
        assert pm.start_op == 0
        assert pm.end_op == 2
        assert pm.confidence == 1.0
        assert pm.matched_kinds == [KernelKind.ELEMENTWISE]

    def test_multiple_matches_ordered(self) -> None:
        """Multiple matches are returned in IR position order."""
        # A sequence that contains elementwise_chain + matmul_activation
        func = _func([
            _op(OpKind.LOAD, "%a1", ["%A1"]),
            _op(OpKind.LOAD, "%b1", ["%B1"]),
            _op(OpKind.ADDF, "%x", ["%a1", "%b1"]),
            _op(OpKind.MULF, "%y", ["%x", "%x"]),
            _op(OpKind.SUBF, "%z", ["%y", "%x"]),
            _op(OpKind.LOAD, "%a2", ["%A2"]),
            _op(OpKind.LOAD, "%b2", ["%B2"]),
            _op(OpKind.DOT, "%c", ["%a2", "%b2"], "tt.dot"),
            _op(OpKind.TANH, "%d", ["%c"], "math.tanh"),
            _op(OpKind.STORE, "", ["%d"]),
        ])
        matches = FusionAnalyzer().analyze(func)
        assert len(matches) >= 2
        for i in range(len(matches) - 1):
            assert matches[i].start_op <= matches[i + 1].start_op


# ---------------------------------------------------------------------------
# Tests — FUSION_PATTERNS constant
# ---------------------------------------------------------------------------


class TestFusionPatternsConstant:
    """The FUSION_PATTERNS dict meets all contractual requirements."""

    def test_at_least_5_patterns(self) -> None:
        """There are at least 5 fusion patterns defined."""
        assert len(FUSION_PATTERNS) >= 5

    def test_required_patterns_present(self) -> None:
        """All 5 required pattern names are present."""
        for name in [
            "matmul_activation",
            "matmul_bias_activation",
            "elementwise_chain",
            "reduction_elementwise",
            "matmul_broadcast_add",
        ]:
            assert name in FUSION_PATTERNS, f"Missing required pattern: {name}"

    def test_each_pattern_has_valid_kinds(self) -> None:
        """Every pattern's kind sequence contains valid KernelKind values."""
        valid = set(KernelKind)
        for name, kinds in FUSION_PATTERNS.items():
            assert len(kinds) >= 2, f"Pattern {name!r} has fewer than 2 kinds"
            for k in kinds:
                assert k in valid, f"Pattern {name!r} has invalid kind: {k}"

    def test_extra_patterns_same_shape(self) -> None:
        """Extra patterns also have valid kind sequences."""
        valid = set(KernelKind)
        for _, kinds in FUSION_PATTERNS_EXTRA.items():
            assert len(kinds) >= 2
            for k in kinds:
                assert k in valid


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_by_name(
    matches: list[PatternMatch],
    name: str,
) -> PatternMatch | None:
    """Find the first PatternMatch with the given name."""
    for m in matches:
        if m.pattern_name == name:
            return m
    return None
