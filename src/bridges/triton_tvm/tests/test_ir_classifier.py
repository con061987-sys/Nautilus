"""Tests for the IR classifier."""

from __future__ import annotations

import pytest

from src.bridges.triton_tvm.ir_capture import KernelKind
from src.bridges.triton_tvm.ir_classifier import IRClassifier


# Sample IR texts for testing
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

REDUCTION_IR = """
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

UNKNOWN_IR = """
module {
  // Some custom op
  %x = "my.custom.op"() : () -> tensor<128xf32>
  tt.return
}
"""


class TestIRClassifier:
    """Tests for the IRClassifier class."""

    def setup_method(self) -> None:
        self.classifier = IRClassifier()

    def test_classify_matmul(self) -> None:
        """IR with tt.dot should be classified as MATMUL."""
        assert self.classifier.classify(MATMUL_IR) == KernelKind.MATMUL

    def test_classify_attention(self) -> None:
        """IR with 2 dots and a reduction should be ATTENTION."""
        assert self.classifier.classify(ATTENTION_IR) == KernelKind.ATTENTION

    def test_classify_reduction(self) -> None:
        """IR with tt.reduce should be REDUCTION."""
        assert self.classifier.classify(REDUCTION_IR) == KernelKind.REDUCTION

    def test_classify_elementwise(self) -> None:
        """IR with load/store + arith but no reduce should be ELEMENTWISE."""
        assert self.classifier.classify(ELEMENTWISE_IR) == KernelKind.ELEMENTWISE

    def test_classify_unknown(self) -> None:
        """IR with no recognizable ops should be UNKNOWN."""
        assert self.classifier.classify(UNKNOWN_IR) == KernelKind.UNKNOWN

    def test_collect_ops_matmul(self) -> None:
        """collect_ops should return all op names in order."""
        ops = self.classifier.collect_ops(MATMUL_IR)
        assert "load" in ops
        assert "dot" in ops
        assert "store" in ops

    def test_collect_ops_preserves_order(self) -> None:
        """ops should appear in the order they're first seen."""
        ops = self.classifier.collect_ops(MATMUL_IR)
        # load should come before dot which should come before store
        load_idx = ops.index("load")
        dot_idx = ops.index("dot")
        store_idx = ops.index("store")
        assert load_idx < dot_idx < store_idx

    def test_collect_op_counts(self) -> None:
        """op_counts should give accurate counts."""
        from collections import Counter
        counts = self.classifier.collect_op_counts(ATTENTION_IR)
        assert isinstance(counts, Counter)
        assert counts["dot"] == 2  # Attention has 2 dots
        assert counts["reduce"] == 1

    def test_collect_tensor_types_matmul(self) -> None:
        """Tensor types should be extracted from the IR."""
        types = self.classifier.collect_tensor_types(MATMUL_IR)
        assert len(types) > 0
        # Should contain at least one 2D shape
        shapes = [t[0] for t in types]
        assert any(len(s) == 2 and all(d > 0 for d in s) for s in shapes)

    def test_collect_tensor_types_1d(self) -> None:
        """1D tensor types should be correctly extracted."""
        types = self.classifier.collect_tensor_types(REDUCTION_IR)
        shapes = [t[0] for t in types]
        assert any(len(s) == 1 for s in shapes)

    def test_parse_shape_with_unknown(self) -> None:
        """Unknown dimensions (?) should be parsed as -1."""
        result = self.classifier._parse_shape("128x?x256")
        assert result == (128, -1, 256)

    def test_parse_shape_empty(self) -> None:
        """Empty shape should return empty tuple."""
        assert self.classifier._parse_shape("") == ()
