"""Tests for the bounds extractor."""

from __future__ import annotations

import pytest

from src.bridges.triton_tvm.bounds_extractor import BoundsExtractor
from src.bridges.triton_tvm.ir_capture import IRBounds, KernelKind


# Sample IR texts with clear bounds
MATMUL_IR_128 = """
module {
  tt.func @matmul_128() {
    %A = arith.constant dense<0.0> : tensor<128x128xf32>
    %B = arith.constant dense<0.0> : tensor<128x128xf32>
    %A_load = tt.load %A_ptr : tensor<128x128xf32>
    %B_load = tt.load %B_ptr : tensor<128x128xf32>
    %result = tt.dot %A_load, %B_load : tensor<128x128xf32>
    tt.store %C_ptr, %result : tensor<128x128xf32>
    tt.return
  }
}
"""

REDUCTION_IR_1024 = """
module {
  tt.func @sum_1024() {
    %input = tt.load %ptr : tensor<1024xf32>
    %sum = "tt.reduce"(%input) ({
      ^bb0(%a: f32, %b: f32): arith.addf %a, %b : f32
    }) {axis = 0 : i32} : (tensor<1024xf32>) -> tensor<1xf32>
    tt.store %out, %sum : tensor<1xf32>
    tt.return
  }
}
"""

ELEMENTWISE_IR = """
module {
  tt.func @elem() {
    %A = tt.load %A_ptr : tensor<256xf32>
    %B = tt.load %B_ptr : tensor<256xf32>
    %sum = arith.addf %A, %B : tensor<256xf32>
    tt.store %C_ptr, %sum : tensor<256xf32>
    tt.return
  }
}
"""

FP16_MATMUL = """
module {
  tt.func @matmul_fp16() {
    %A = tt.load %A_ptr : tensor<64x32xf16>
    %B = tt.load %B_ptr : tensor<32x64xf16>
    %result = tt.dot %A, %B : tensor<64x64xf16>
    tt.return
  }
}
"""


class TestBoundsExtractor:
    """Tests for the BoundsExtractor class."""

    def setup_method(self) -> None:
        self.extractor = BoundsExtractor()

    def test_extract_matmul_bounds(self) -> None:
        """Matmul IR should yield M, N, K bounds."""
        bounds = self.extractor.extract(MATMUL_IR_128, KernelKind.MATMUL)
        # M=128, N=128, K=128 should be inferable
        assert bounds.m is not None and bounds.m > 0
        assert bounds.n is not None and bounds.n > 0
        assert bounds.k is not None and bounds.k > 0
        assert bounds.data_dtype == "float32"

    def test_extract_reduction_bounds(self) -> None:
        """Reduction IR should yield reduce_size."""
        bounds = self.extractor.extract(REDUCTION_IR_1024, KernelKind.REDUCTION)
        # 1024 should appear as the largest dim
        assert bounds.reduce_size is not None and bounds.reduce_size >= 1024
        assert bounds.data_dtype == "float32"

    def test_extract_elementwise_bounds(self) -> None:
        """Elementwise IR should yield total_elements."""
        bounds = self.extractor.extract(ELEMENTWISE_IR, KernelKind.ELEMENTWISE)
        assert bounds.total_elements is not None and bounds.total_elements >= 256

    def test_extract_dtype_fp16(self) -> None:
        """FP16 dtypes should be detected as 'float16'."""
        bounds = self.extractor.extract(FP16_MATMUL, KernelKind.MATMUL)
        assert bounds.data_dtype == "float16"

    def test_extract_dtype_normalization(self) -> None:
        """MLIR dtype names should be normalized to canonical form."""
        assert self.extractor._normalize_dtype("f32") == "float32"
        assert self.extractor._normalize_dtype("f16") == "float16"
        assert self.extractor._normalize_dtype("bf16") == "bfloat16"
        assert self.extractor._normalize_dtype("i32") == "int32"
        assert self.extractor._normalize_dtype("unknown") == "unknown"

    def test_extract_tensor_shapes_2d(self) -> None:
        """2D shapes should be parsed correctly."""
        shapes, dtypes = self.extractor._extract_tensor_shapes(MATMUL_IR_128)
        # Should have at least one 2D shape
        assert any(len(s) == 2 and all(d > 0 for d in s) for s in shapes)

    def test_extract_tensor_shapes_1d(self) -> None:
        """1D shapes should be parsed correctly."""
        shapes, _ = self.extractor._extract_tensor_shapes(REDUCTION_IR_1024)
        assert any(len(s) == 1 and 1024 in s for s in shapes)

    def test_parse_shape_str(self) -> None:
        """Shape strings should be parsed to tuples."""
        assert self.extractor._parse_shape_str("128x256xf32") == (128, 256)
        assert self.extractor._parse_shape_str("64x?xf16") == (64, -1)
        assert self.extractor._parse_shape_str("") == ()

    def test_dominant_dtype(self) -> None:
        """The most common dtype should be returned as dominant."""
        assert self.extractor._dominant_dtype(["float32", "float32", "float16"]) == "float32"
        assert self.extractor._dominant_dtype([]) == "float32"

    def test_extract_for_bounds(self) -> None:
        """scf.for loop bounds should be extracted."""
        ir = """
        module {
          scf.for %i = 0 to 100 {
            scf.for %j = 0 to 50 {
              ...
            }
          }
        }
        """
        bounds = self.extractor._extract_for_bounds(ir)
        assert (0, 100) in bounds
        assert (0, 50) in bounds

    def test_infer_matmul_from_shapes(self) -> None:
        """M, N, K should be inferrable from 2D shapes."""
        shapes = [(128, 64), (64, 128), (128, 128)]
        m, n, k = self.extractor._infer_matmul_from_shapes(shapes)
        # C should be the largest, giving M=128, N=128
        # K should come from A or B's inner dim
        assert m == 128
        assert n == 128
        assert k == 64
