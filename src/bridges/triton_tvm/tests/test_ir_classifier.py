"""Tests for the IR classifier.

Covers the AST-based classification path:
  - Each supported op type classifies to the correct KernelKind.
  - Reduction bodies are inspected for the combine op (sum/max/min).
  - Tensor element types and shapes are surfaced in IRClassification.
  - ClassificationError is raised for IR with no recognisable ops.
  - Property-based tests verify structural invariants of the
    classifier over randomly generated IRs.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.bridges.triton_tvm.ir_capture import KernelKind
from src.bridges.triton_tvm.ir_classifier import (
    ClassificationError,
    IRClassification,
    IRClassifier,
    ReductionType,
)

# ---------------------------------------------------------------------------
# Sample IR texts
# ---------------------------------------------------------------------------

MATMUL_IR = """
module {
  tt.func public @matmul_kernel(
    %A: !tt.ptr<tensor<128x32xf32>>,
    %B: !tt.ptr<tensor<32x128xf32>>,
    %C: !tt.ptr<tensor<128x128xf32>>
  ) {
    %a = tt.load %A : tensor<128x32xf32>
    %b = tt.load %B : tensor<32x128xf32>
    %c = tt.dot %a, %b : tensor<128x128xf32>
    tt.store %C, %c : tensor<128x128xf32>
    tt.return
  }
}
"""

ATTENTION_IR = """
module {
  tt.func @attention_kernel() {
    %scores = tt.dot %Q, %K : tensor<64x64xf32>
    %max = "tt.reduce"(%scores) ({
      ^bb0(%a: f32, %b: f32): arith.maximumf %a, %b : f32
    }) {axis = 1 : i32} : (tensor<64x64xf32>) -> tensor<64xf32>
    %exp = math.exp %scores : tensor<64x64xf32>
    %out = tt.dot %exp, %V : tensor<64x64xf32>
    tt.return
  }
}
"""

REDUCTION_SUM_IR = """
module {
  tt.func @sum_kernel() {
    %x = tt.load %input : tensor<1024xf32>
    %sum = "tt.reduce"(%x) ({
      ^bb0(%a: f32, %b: f32): arith.addf %a, %b : f32
    }) {axis = 0 : i32} : (tensor<1024xf32>) -> tensor<1xf32>
    tt.store %output, %sum : tensor<1xf32>
    tt.return
  }
}
"""

REDUCTION_MAX_IR = """
module {
  tt.func @max_kernel() {
    %x = tt.load %input : tensor<1024xf32>
    %max = "tt.reduce"(%x) ({
      ^bb0(%a: f32, %b: f32): arith.maximumf %a, %b : f32
    }) {axis = 0 : i32} : (tensor<1024xf32>) -> tensor<1xf32>
    tt.store %output, %max : tensor<1xf32>
    tt.return
  }
}
"""

REDUCTION_MIN_IR = """
module {
  tt.func @min_kernel() {
    %x = tt.load %input : tensor<1024xf32>
    %min = "tt.reduce"(%x) ({
      ^bb0(%a: f32, %b: f32): arith.minimumf %a, %b : f32
    }) {axis = 0 : i32} : (tensor<1024xf32>) -> tensor<1xf32>
    tt.store %output, %min : tensor<1xf32>
    tt.return
  }
}
"""

ELEMENTWISE_IR = """
module {
  tt.func @add_kernel() {
    %a = tt.load %A : tensor<128xf32>
    %b = tt.load %B : tensor<128xf32>
    %sum = arith.addf %a, %b : tensor<128xf32>
    tt.store %C, %sum : tensor<128xf32>
    tt.return
  }
}
"""

BROADCAST_IR = """
module {
  tt.func @broadcast_kernel() {
    %x = tt.load %input : tensor<32xf32>
    %y = tt.broadcast %x : tensor<32xf32> -> tensor<128x32xf32>
    tt.store %output, %y : tensor<128x32xf32>
    tt.return
  }
}
"""

TRANSPOSE_IR = """
module {
  tt.func @transpose_kernel() {
    %x = tt.load %input : tensor<32x64xf32>
    %y = tt.trans %x : tensor<32x64xf32> -> tensor<64x32xf32>
    tt.store %output, %y : tensor<64x32xf32>
    tt.return
  }
}
"""

UNSUPPORTED_IR = """
module {
  %x = "my.custom.op"() : () -> tensor<128xf32>
  tt.return
}
"""

# Multiple dtypes — verifies dtype extraction is correct.
MATMUL_F16_IR = """
module {
  tt.func public @matmul_f16(
    %A: !tt.ptr<tensor<128x32xf16>>,
    %B: !tt.ptr<tensor<32x128xf16>>,
    %C: !tt.ptr<tensor<128x128xf16>>
  ) {
    %a = tt.load %A : tensor<128x32xf16>
    %b = tt.load %B : tensor<32x128xf16>
    %c = tt.dot %a, %b : tensor<128x128xf16>
    tt.store %C, %c : tensor<128x128xf16>
    tt.return
  }
}
"""

MATMUL_F64_IR = """
module {
  tt.func public @matmul_f64(
    %A: !tt.ptr<tensor<128x32xf64>>,
    %B: !tt.ptr<tensor<32x128xf64>>,
    %C: !tt.ptr<tensor<128x128xf64>>
  ) {
    %a = tt.load %A : tensor<128x32xf64>
    %b = tt.load %B : tensor<32x128xf64>
    %c = tt.dot %a, %b : tensor<128x128xf64>
    tt.store %C, %c : tensor<128x128xf64>
    tt.return
  }
}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wrap_module(body: str) -> str:
    """Wrap an inner function body in a minimal module."""
    return f"module {{\n{body}\n}}\n"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIRClassifier:
    """Tests for the IRClassifier class."""

    def setup_method(self) -> None:
        self.classifier = IRClassifier()

    # ---- kind classification -----------------------------------------

    def test_classify_matmul(self) -> None:
        """IR with tt.dot should classify as MATMUL."""
        assert self.classifier.classify(MATMUL_IR) == KernelKind.MATMUL

    def test_classify_attention(self) -> None:
        """IR with 2 dots + reduce + exp should classify as ATTENTION."""
        assert self.classifier.classify(ATTENTION_IR) == KernelKind.ATTENTION

    def test_classify_reduction_sum(self) -> None:
        """IR with tt.reduce using arith.addf should classify as REDUCTION."""
        result = self.classifier.classify(REDUCTION_SUM_IR)
        assert result == KernelKind.REDUCTION
        assert result.reduction_type == ReductionType.SUM
        assert result.reduction_axis == 0

    def test_classify_reduction_max(self) -> None:
        """IR with tt.reduce using arith.maximumf → MAX reduction."""
        result = self.classifier.classify(REDUCTION_MAX_IR)
        assert result == KernelKind.REDUCTION
        assert result.reduction_type == ReductionType.MAX

    def test_classify_reduction_min(self) -> None:
        """IR with tt.reduce using arith.minimumf → MIN reduction."""
        result = self.classifier.classify(REDUCTION_MIN_IR)
        assert result == KernelKind.REDUCTION
        assert result.reduction_type == ReductionType.MIN

    def test_classify_reduction_axis(self) -> None:
        """Reduction axis attribute is surfaced from the IR."""
        result = self.classifier.classify(REDUCTION_SUM_IR)
        assert result.reduction_axis == 0

    def test_classify_reduction_axis_nonzero(self) -> None:
        """Non-zero axis values are surfaced correctly."""
        ir = _wrap_module("""
            tt.func @k() {
              %x = tt.load %i : tensor<8x16xf32>
              %r = "tt.reduce"(%x) ({
                ^bb0(%a: f32, %b: f32): arith.addf %a, %b : f32
              }) {axis = 1 : i32} : (tensor<8x16xf32>) -> tensor<8xf32>
              tt.return
            }
        """)
        result = self.classifier.classify(ir)
        assert result.reduction_axis == 1

    def test_classify_elementwise(self) -> None:
        """IR with load+store+arith and no reduce classifies as ELEMENTWISE."""
        assert self.classifier.classify(ELEMENTWISE_IR) == KernelKind.ELEMENTWISE

    def test_classify_broadcast(self) -> None:
        """IR with tt.broadcast classifies as BROADCAST."""
        assert self.classifier.classify(BROADCAST_IR) == KernelKind.BROADCAST

    def test_classify_transpose(self) -> None:
        """IR with tt.trans classifies as TRANSPOSE."""
        assert self.classifier.classify(TRANSPOSE_IR) == KernelKind.TRANSPOSE

    def test_classify_matmul_reduction_type_sum(self) -> None:
        """Matmul classifies with SUM reduction type (K-axis sum-of-products)."""
        result = self.classifier.classify(MATMUL_IR)
        assert result.reduction_type == ReductionType.SUM

    def test_unsupported_ir_raises(self) -> None:
        """IR with no supported ops raises ClassificationError."""
        with pytest.raises(ClassificationError):
            self.classifier.classify(UNSUPPORTED_IR)

    def test_classify_neq_against_wrong_kind(self) -> None:
        """IRClassification is not == to an unrelated KernelKind."""
        result = self.classifier.classify(MATMUL_IR)
        assert (result == KernelKind.REDUCTION) is False

    # ---- tensor type extraction --------------------------------------

    def test_tensor_shapes_matmul(self) -> None:
        """Matmul IR surfaces the 2D input/output shapes."""
        result = self.classifier.classify(MATMUL_IR)
        assert (128, 32) in result.tensor_shapes
        assert (32, 128) in result.tensor_shapes
        assert (128, 128) in result.tensor_shapes

    def test_tensor_shapes_1d_reduction(self) -> None:
        """1D reduction input/output shapes are surfaced."""
        result = self.classifier.classify(REDUCTION_SUM_IR)
        assert (1024,) in result.tensor_shapes
        assert (1,) in result.tensor_shapes

    def test_element_type_f32(self) -> None:
        """f32 element type is extracted from the IR."""
        result = self.classifier.classify(MATMUL_IR)
        assert result.tensor_element_type == "float32"

    def test_element_type_f16(self) -> None:
        """f16 element type is extracted from the IR."""
        result = self.classifier.classify(MATMUL_F16_IR)
        assert result.tensor_element_type == "float16"

    def test_element_type_f64(self) -> None:
        """f64 element type is extracted from the IR."""
        result = self.classifier.classify(MATMUL_F64_IR)
        assert result.tensor_element_type == "float64"

    def test_collect_tensor_types_matmul(self) -> None:
        """collect_tensor_types returns (shape, dtype) tuples for matmul."""
        types = self.classifier.collect_tensor_types(MATMUL_IR)
        assert len(types) > 0
        shapes = [shape for shape, _ in types]
        assert any(len(s) == 2 and all(d > 0 for d in s) for s in shapes)
        dtypes = [dtype for _, dtype in types]
        assert "float32" in dtypes

    def test_collect_tensor_types_1d(self) -> None:
        """1D tensor types are correctly extracted."""
        types = self.classifier.collect_tensor_types(REDUCTION_SUM_IR)
        shapes = [shape for shape, _ in types]
        assert any(len(s) == 1 for s in shapes)

    # ---- ops collection ----------------------------------------------

    def test_collect_ops_matmul(self) -> None:
        """collect_ops returns all op names in order for matmul."""
        ops = self.classifier.collect_ops(MATMUL_IR)
        assert "tt.load" in ops
        assert "tt.dot" in ops
        assert "tt.store" in ops

    def test_collect_ops_preserves_order(self) -> None:
        """ops appear in AST traversal order."""
        ops = self.classifier.collect_ops(MATMUL_IR)
        load_idx = ops.index("tt.load")
        dot_idx = ops.index("tt.dot")
        store_idx = ops.index("tt.store")
        assert load_idx < dot_idx < store_idx

    def test_collect_op_counts(self) -> None:
        """op_counts gives accurate counts (deduplicated)."""
        from collections import Counter

        counts = self.classifier.collect_op_counts(ATTENTION_IR)
        assert isinstance(counts, Counter)
        assert counts["tt.dot"] == 2  # Attention has 2 dots
        assert counts["tt.reduce"] == 1

    # ---- shape parsing -----------------------------------------------

    def test_parse_shape_with_unknown(self) -> None:
        """Unknown dimensions (?) are parsed as -1."""
        result = self.classifier.parse_shape("128x?x256")
        assert result == (128, -1, 256)

    def test_parse_shape_empty(self) -> None:
        """Empty shape string returns empty tuple."""
        assert self.classifier.parse_shape("") == ()

    def test_parse_shape_single_dim(self) -> None:
        """Single-dim shape parses correctly."""
        assert self.classifier.parse_shape("1024") == (1024,)

    # ---- IRClassification dataclass ----------------------------------

    def test_ir_classification_dataclass_equality(self) -> None:
        """Two IRClassification instances compare by kind+reduction."""
        a = IRClassification(
            kind=KernelKind.REDUCTION, reduction_type=ReductionType.SUM, reduction_axis=0
        )
        b = IRClassification(
            kind=KernelKind.REDUCTION, reduction_type=ReductionType.SUM, reduction_axis=0
        )
        c = IRClassification(
            kind=KernelKind.REDUCTION, reduction_type=ReductionType.MAX, reduction_axis=0
        )
        assert a == b
        assert a != c

    def test_ir_classification_eq_kernelkind(self) -> None:
        """IRClassification can be compared to a bare KernelKind."""
        result = self.classifier.classify(MATMUL_IR)
        assert (result == KernelKind.MATMUL) is True
        assert (result == KernelKind.REDUCTION) is False


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------


class TestIRClassifierProperties:
    """Property-based tests for invariants of the AST classifier."""

    @given(axes=st.lists(st.integers(min_value=1, max_value=128), min_size=1, max_size=3))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_elementwise_kernel_with_random_shapes(self, axes: list[int]) -> None:
        """Random-shape elementwise kernels always classify as ELEMENTWISE."""
        shape = "x".join(str(a) for a in axes)
        ir = _wrap_module(f"""
            tt.func @k() {{
              %a = tt.load %A : tensor<{shape}xf32>
              %b = tt.load %B : tensor<{shape}xf32>
              %s = arith.addf %a, %b : tensor<{shape}xf32>
              tt.store %C, %s : tensor<{shape}xf32>
              tt.return
            }}
        """)
        result = IRClassifier().classify(ir)
        assert result == KernelKind.ELEMENTWISE

    @given(
        m=st.integers(min_value=1, max_value=128),
        n=st.integers(min_value=1, max_value=128),
        k=st.integers(min_value=1, max_value=128),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_matmul_kernel_with_random_dims(self, m: int, n: int, k: int) -> None:
        """Random-dim matmul kernels always classify as MATMUL."""
        ir = _wrap_module(f"""
            tt.func public @matmul(
              %A: !tt.ptr<tensor<{m}x{k}xf32>>,
              %B: !tt.ptr<tensor<{k}x{n}xf32>>,
              %C: !tt.ptr<tensor<{m}x{n}xf32>>
            ) {{
              %a = tt.load %A : tensor<{m}x{k}xf32>
              %b = tt.load %B : tensor<{k}x{n}xf32>
              %c = tt.dot %a, %b : tensor<{m}x{n}xf32>
              tt.store %C, %c : tensor<{m}x{n}xf32>
              tt.return
            }}
        """)
        result = IRClassifier().classify(ir)
        assert result == KernelKind.MATMUL
        assert result.reduction_type == ReductionType.SUM

    @given(
        size=st.integers(min_value=1, max_value=4096),
        axis=st.integers(min_value=0, max_value=2),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_reduction_kernel_random_size_and_axis(self, size: int, axis: int) -> None:
        """Random-size/axis sum reductions always classify as REDUCTION/SUM."""
        rank = axis + 1
        shape = "x".join(["32"] * rank)
        in_shape = "x".join([str(size)] + ["32"] * (rank - 1))
        ir = _wrap_module(f"""
            tt.func @sum() {{
              %x = tt.load %input : tensor<{in_shape}xf32>
              %r = "tt.reduce"(%x) ({{
                ^bb0(%a: f32, %b: f32): arith.addf %a, %b : f32
              }}) {{axis = {axis} : i32}} : (tensor<{in_shape}xf32>) -> tensor<{shape}xf32>
              tt.store %output, %r : tensor<{shape}xf32>
              tt.return
            }}
        """)
        result = IRClassifier().classify(ir)
        assert result == KernelKind.REDUCTION
        assert result.reduction_type == ReductionType.SUM
        assert result.reduction_axis == axis

    @given(
        width=st.integers(min_value=1, max_value=128),
        height=st.integers(min_value=1, max_value=128),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_broadcast_kernel_random_dims(self, width: int, height: int) -> None:
        """Random-dim broadcast kernels always classify as BROADCAST."""
        ir = _wrap_module(f"""
            tt.func @bc() {{
              %x = tt.load %in : tensor<{width}xf32>
              %y = tt.broadcast %x : tensor<{width}xf32> -> tensor<{height}x{width}xf32>
              tt.store %out, %y : tensor<{height}x{width}xf32>
              tt.return
            }}
        """)
        result = IRClassifier().classify(ir)
        assert result == KernelKind.BROADCAST

    @given(
        rows=st.integers(min_value=1, max_value=128),
        cols=st.integers(min_value=1, max_value=128),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_transpose_kernel_random_dims(self, rows: int, cols: int) -> None:
        """Random-dim transpose kernels always classify as TRANSPOSE."""
        ir = _wrap_module(f"""
            tt.func @tr() {{
              %x = tt.load %in : tensor<{rows}x{cols}xf32>
              %y = tt.trans %x : tensor<{rows}x{cols}xf32> -> tensor<{cols}x{rows}xf32>
              tt.store %out, %y : tensor<{cols}x{rows}xf32>
              tt.return
            }}
        """)
        result = IRClassifier().classify(ir)
        assert result == KernelKind.TRANSPOSE

    @given(
        text=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "Lu", "Ll", "Nd"),
                whitelist_characters="_. ",
            ),
            min_size=1,
            max_size=20,
        ).filter(lambda s: not any(op in s for op in ("tt.", "arith.", "math.", "scf.")))
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_no_supported_ops_raises(self, text: str) -> None:
        """IRs with no supported ops raise ClassificationError.

        Hypothesis generates random text; the .filter() strips out
        any case where a supported op prefix leaks in. The IR is
        still expected to be unparseable.
        """
        ir = _wrap_module(f"%x = {text}\n  tt.return")
        with pytest.raises(ClassificationError):
            IRClassifier().classify(ir)
