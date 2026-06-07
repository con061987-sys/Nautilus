"""Tests for the bounds extractor.

These tests verify the AST-based :class:`BoundsExtractor` walks the
:class:`TTGIRParser` AST (not whole-IR regex) to extract M/N/K for
matmul-family ops, reduce axes for ``tt.reduce``, and ``scf.for``
loop bounds. Property-based tests use Hypothesis to generate 100
random valid TTGIR snippets per kernel kind and assert that the
extractor returns non-zero values.

The ``IRClassifier`` is intentionally NOT used here — we feed each
``extract()`` call the *correct* :class:`KernelKind` so the test
isolates the bounds-extraction logic from the (separately-tested)
classification logic. The class name in the test module remains
``TestBoundsExtractor`` for back-compat with CI grep patterns.
"""

from __future__ import annotations

import string
import textwrap
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.bridges.triton_tvm.bounds_extractor import (
    BoundsExtractionError,
    BoundsExtractor,
)
from src.bridges.triton_tvm.ir_capture import KernelKind

# ---------------------------------------------------------------------------
# Test IR fixtures — realistic TTGIR shapes that exercise each code path.
# ---------------------------------------------------------------------------

MATMUL_IR_128 = """
module {
  tt.func @matmul_128() {
    %A_load = tt.load %A_ptr : tensor<128x128xf32>
    %B_load = tt.load %B_ptr : tensor<128x128xf32>
    %result = tt.dot %A_load, %B_load : tensor<128x128xf32>
    tt.store %C_ptr, %result : tensor<128x128xf32>
    tt.return
  }
}
"""

MATMUL_IR_ASYMMETRIC = """
module {
  tt.func @matmul_64x256x32() {
    %A_load = tt.load %A_ptr : tensor<64x32xf32>
    %B_load = tt.load %B_ptr : tensor<32x256xf32>
    %result = tt.dot %A_load, %B_load : tensor<64x256xf32>
    tt.store %C_ptr, %result : tensor<64x256xf32>
    tt.return
  }
}
"""

REDUCTION_IR_1024 = """
module {
  tt.func @sum_1024() {
    %input = tt.load %ptr : tensor<1024xf32>
    %sum = "tt.reduce"(%input) ({
      ^bb0(%a: f32, %b: f32):
        arith.addf %a, %b : f32
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

# Batched matmul — 3-D operands, M=64, N=128, K=32, batch=4
BMM_IR = """
module {
  tt.func @bmm_kernel() {
    %A = tt.load %A_ptr : tensor<4x64x32xf32>
    %B = tt.load %B_ptr : tensor<4x32x128xf32>
    %result = tt.bmm %A, %B : tensor<4x64x128xf32>
    tt.return
  }
}
"""

# Matmul kernel with a loop that should populate ``block_size``
LOOP_MATMUL_IR = """
module {
  tt.func @loop_matmul(%A_ptr: !tt.ptr<tensor<128x128xf32>>, %B_ptr: !tt.ptr<tensor<128x128xf32>>) {
    scf.for %k = 0 to 16 {
      %A = tt.load %A_ptr : tensor<128x128xf32>
      %B = tt.load %B_ptr : tensor<128x128xf32>
      %C = tt.dot %A, %B : tensor<128x128xf32>
    }
    tt.return
  }
}
"""

# 2-D reduction over axis=1 of a 64x32 tensor
REDUCTION_2D_IR = """
module {
  tt.func @sum_2d() {
    %input = tt.load %ptr : tensor<64x32xf32>
    %sum = "tt.reduce"(%input) ({
      ^bb0(%a: f32, %b: f32):
        arith.addf %a, %b : f32
    }) {axis = 1 : i32} : (tensor<64x32xf32>) -> tensor<64x1xf32>
    tt.store %out, %sum : tensor<64x1xf32>
    tt.return
  }
}
"""


# ---------------------------------------------------------------------------
# Helpers shared by the property-based tests
# ---------------------------------------------------------------------------

# Power-of-two-positive strategy — TTGIR block sizes are typically
# powers of two, so this gives a realistic distribution without
# exploding into large primes that would slow the parser.
_POSITIVE_DIM = st.integers(min_value=1, max_value=512)
_DTYPE_TOKENS = ("f32", "f16", "bf16", "i32", "i64")


def _dtype() -> st.SearchStrategy[str]:
    return st.sampled_from(_DTYPE_TOKENS)


def _shape(rank: int) -> st.SearchStrategy[tuple[int, ...]]:
    return st.tuples(*([_POSITIVE_DIM] * rank))


def _render_shape(shape: tuple[int, ...]) -> str:
    return "x".join(str(d) for d in shape)


def _build_matmul_ir(
    m: int,
    k: int,
    n: int,
    dtype: str,
    op_name: str = "tt.dot",
) -> str:
    """Build a valid matmul IR snippet with the given M, N, K, dtype.

    ``op_name`` is one of ``tt.dot`` / ``tt.dot_scaled`` / ``tt.matmul``.
    For ``tt.bmm`` use :func:`_build_bmm_ir` instead.
    """
    a_shape = _render_shape((m, k))
    b_shape = _render_shape((k, n))
    c_shape = _render_shape((m, n))
    return textwrap.dedent(f"""
        module {{
          tt.func @kernel() {{
            %A = tt.load %A_ptr : tensor<{a_shape}x{dtype}>
            %B = tt.load %B_ptr : tensor<{b_shape}x{dtype}>
            %result = {op_name} %A, %B : tensor<{c_shape}x{dtype}>
            tt.return
          }}
        }}
    """)


def _build_bmm_ir(
    batch: int,
    m: int,
    k: int,
    n: int,
    dtype: str,
) -> str:
    """Build a valid batched matmul IR snippet (tt.bmm)."""
    a_shape = _render_shape((batch, m, k))
    b_shape = _render_shape((batch, k, n))
    c_shape = _render_shape((batch, m, n))
    return textwrap.dedent(f"""
        module {{
          tt.func @kernel() {{
            %A = tt.load %A_ptr : tensor<{a_shape}x{dtype}>
            %B = tt.load %B_ptr : tensor<{b_shape}x{dtype}>
            %result = tt.bmm %A, %B : tensor<{c_shape}x{dtype}>
            tt.return
          }}
        }}
    """)


def _build_reduction_ir(
    shape: tuple[int, ...],
    axis: int,
    dtype: str,
) -> str:
    """Build a valid ``tt.reduce`` IR snippet for the given shape/axis."""
    input_shape = _render_shape(shape)
    # Output keeps all dims except the reduce axis (which becomes 1)
    out_shape_parts: list[str] = []
    for i, d in enumerate(shape):
        out_shape_parts.append("1" if i == axis else str(d))
    out_shape = "x".join(out_shape_parts)
    return textwrap.dedent(f"""
        module {{
          tt.func @kernel() {{
            %input = tt.load %ptr : tensor<{input_shape}x{dtype}>
            %sum = "tt.reduce"({{%input}}) ({{
              ^bb0(%a: {dtype}, %b: {dtype}):
                arith.addf %a, %b : {dtype}
            }}) {{axis = {axis} : i32}} : (tensor<{input_shape}x{dtype}>) -> tensor<{out_shape}x{dtype}>
            tt.return
          }}
        }}
    """)


def _build_elementwise_ir(shape: tuple[int, ...], dtype: str) -> str:
    """Build a valid elementwise IR snippet with the given shape."""
    s = _render_shape(shape)
    return textwrap.dedent(f"""
        module {{
          tt.func @kernel() {{
            %A = tt.load %A_ptr : tensor<{s}x{dtype}>
            %B = tt.load %B_ptr : tensor<{s}x{dtype}>
            %sum = arith.addf %A, %B : tensor<{s}x{dtype}>
            tt.return
          }}
        }}
    """)


# ---------------------------------------------------------------------------
# Sanity tests
# ---------------------------------------------------------------------------


class TestBoundsExtractor:
    """Unit tests for the AST-based :class:`BoundsExtractor`."""

    def setup_method(self) -> None:
        self.extractor = BoundsExtractor()

    # -- matmul ---------------------------------------------------------

    def test_extract_matmul_bounds(self) -> None:
        """Matmul IR yields M, N, K bounds from the AST, not regex."""
        bounds = self.extractor.extract(MATMUL_IR_128, KernelKind.MATMUL)
        assert bounds.m == 128
        assert bounds.n == 128
        assert bounds.k == 128
        assert bounds.data_dtype == "float32"

    def test_extract_matmul_asymmetric(self) -> None:
        """Asymmetric matmul (M=64, N=256, K=32) is extracted correctly."""
        bounds = self.extractor.extract(
            MATMUL_IR_ASYMMETRIC,
            KernelKind.MATMUL,
        )
        assert bounds.m == 64
        assert bounds.n == 256
        assert bounds.k == 32
        assert bounds.data_dtype == "float32"

    def test_extract_bmm_bounds(self) -> None:
        """tt.bmm (3-D batched matmul) yields M, N, K without the batch dim."""
        bounds = self.extractor.extract(BMM_IR, KernelKind.MATMUL)
        assert bounds.m == 64
        assert bounds.n == 128
        assert bounds.k == 32
        assert 3 in bounds.tensor_ranks  # batched tensors are 3-D

    def test_extract_matmul_loop_block_size(self) -> None:
        """A matmul wrapped in scf.for propagates the loop bound into block_size."""
        bounds = self.extractor.extract(LOOP_MATMUL_IR, KernelKind.MATMUL)
        assert bounds.m == 128
        assert bounds.n == 128
        assert bounds.k == 128
        assert (0, 16) in {tuple(b) for b in [bounds.block_size]}
        assert bounds.block_size == (0, 16)

    def test_extract_matmul_missing_dot_raises(self) -> None:
        """If the IR has no matmul-family op, extraction raises."""
        no_dot = textwrap.dedent("""
            module {
              tt.func @kernel() {
                %A = tt.load %A_ptr : tensor<128x128xf32>
                tt.return
              }
            }
        """)
        with pytest.raises(BoundsExtractionError, match="no matmul-family op"):
            self.extractor.extract(no_dot, KernelKind.MATMUL)

    def test_extract_matmul_contracted_dim_mismatch_raises(self) -> None:
        """A: (M=64, K=32), B: (K=64, N=128) — K mismatch raises."""
        bad = textwrap.dedent("""
            module {
              tt.func @kernel() {
                %A = tt.load %A_ptr : tensor<64x32xf32>
                %B = tt.load %B_ptr : tensor<64x128xf32>
                %result = tt.dot %A, %B : tensor<64x128xf32>
                tt.return
              }
            }
        """)
        with pytest.raises(BoundsExtractionError, match="Contracted-dim mismatch"):
            self.extractor.extract(bad, KernelKind.MATMUL)

    # -- reduction ------------------------------------------------------

    def test_extract_reduction_bounds(self) -> None:
        """Reduction IR yields reduce_size = axis dim."""
        bounds = self.extractor.extract(REDUCTION_IR_1024, KernelKind.REDUCTION)
        assert bounds.reduce_size == 1024
        assert bounds.keep_size == 1
        assert bounds.data_dtype == "float32"

    def test_extract_reduction_2d_axis1(self) -> None:
        """2-D reduction over axis=1 of a 64x32 tensor."""
        bounds = self.extractor.extract(REDUCTION_2D_IR, KernelKind.REDUCTION)
        assert bounds.reduce_size == 32
        assert bounds.keep_size == 64

    def test_extract_reduction_missing_op_raises(self) -> None:
        """IR classified as REDUCTION but no tt.reduce present → error."""
        no_reduce = textwrap.dedent("""
            module {
              tt.func @kernel() {
                %A = tt.load %A_ptr : tensor<128xf32>
                tt.return
              }
            }
        """)
        with pytest.raises(BoundsExtractionError, match=r"no tt.reduce op"):
            self.extractor.extract(no_reduce, KernelKind.REDUCTION)

    # -- elementwise ----------------------------------------------------

    def test_extract_elementwise_bounds(self) -> None:
        """Elementwise IR yields total_elements = product of shape."""
        bounds = self.extractor.extract(ELEMENTWISE_IR, KernelKind.ELEMENTWISE)
        assert bounds.total_elements == 256
        assert bounds.data_dtype == "float32"

    # -- dtype ----------------------------------------------------------

    def test_extract_dtype_fp16(self) -> None:
        """FP16 dtypes are detected as 'float16'."""
        bounds = self.extractor.extract(FP16_MATMUL, KernelKind.MATMUL)
        assert bounds.data_dtype == "float16"

    # -- malformed IR ---------------------------------------------------

    def test_malformed_ir_raises(self) -> None:
        """Garbage IR raises :class:`BoundsExtractionError`."""
        with pytest.raises(BoundsExtractionError, match="Failed to parse TTGIR"):
            self.extractor.extract("not valid ttgir at all", KernelKind.MATMUL)

    def test_empty_ir_raises(self) -> None:
        """Empty string raises :class:`BoundsExtractionError`."""
        with pytest.raises(BoundsExtractionError, match="Failed to parse TTGIR"):
            self.extractor.extract("", KernelKind.MATMUL)

    # -- generic / unknown ----------------------------------------------

    def test_generic_extraction_unknown_kind(self) -> None:
        """An UNKNOWN-kind IR with shapes still yields total_elements."""
        ir = textwrap.dedent("""
            module {
              tt.func @kernel() {
                %A = tt.load %A_ptr : tensor<128x128xf32>
                tt.return
              }
            }
        """)
        bounds = self.extractor.extract(ir, KernelKind.UNKNOWN)
        assert bounds.total_elements == 128 * 128


# ---------------------------------------------------------------------------
# Property-based tests — 100 random valid TTGIR snippets per kind.
# These verify the extractor is shape-preserving for arbitrary
# dimension values, not just the 5 hardcoded cases above.
# ---------------------------------------------------------------------------


# Reusable custom identifiers — Hypothesis can generate many strings,
# and most will not be valid Python/MLIR identifiers, so we restrict
# the alphabet.
_IDENTIFIER_CHARS = string.ascii_letters + string.digits + "_"


@st.composite
def _matmul_params(draw: Any) -> dict[str, Any]:
    """Strategy that yields (M, K, N, dtype, op_name) for 2-D matmul."""
    m = draw(_POSITIVE_DIM)
    k = draw(_POSITIVE_DIM)
    n = draw(_POSITIVE_DIM)
    dtype = draw(_dtype())
    op_name = draw(
        st.sampled_from(("tt.dot", "tt.dot_scaled", "tt.matmul")),
    )
    return {"m": m, "k": k, "n": n, "dtype": dtype, "op_name": op_name}


@st.composite
def _bmm_params(draw: Any) -> dict[str, Any]:
    """Strategy that yields (batch, M, K, N, dtype) for 3-D batched matmul."""
    return {
        "batch": draw(st.integers(min_value=1, max_value=32)),
        "m": draw(_POSITIVE_DIM),
        "k": draw(_POSITIVE_DIM),
        "n": draw(_POSITIVE_DIM),
        "dtype": draw(_dtype()),
    }


@st.composite
def _reduction_params(draw: Any) -> dict[str, Any]:
    """Strategy that yields (shape, axis, dtype) for tt.reduce."""
    rank = draw(st.integers(min_value=1, max_value=4))
    shape = draw(_shape(rank))
    axis = draw(st.integers(min_value=0, max_value=rank - 1))
    dtype = draw(_dtype())
    return {"shape": shape, "axis": axis, "dtype": dtype}


@st.composite
def _elementwise_params(draw: Any) -> dict[str, Any]:
    """Strategy that yields (shape, dtype) for elementwise."""
    rank = draw(st.integers(min_value=1, max_value=3))
    return {"shape": draw(_shape(rank)), "dtype": draw(_dtype())}


# Common Hypothesis settings: 100 examples, suppress the slow-DB
# health check (the parser+extractor is fast but a misfire of
# ``too_slow`` would flake the run).
_PROP_SETTINGS = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)


class TestPropertyBasedExtraction:
    """Property-based tests with Hypothesis (100 examples per kind)."""

    def setup_method(self) -> None:
        self.extractor = BoundsExtractor()

    @given(params=_matmul_params())
    @_PROP_SETTINGS
    def test_property_matmul_extracts_correct_dims(self, params: dict) -> None:
        """For any valid (M, K, N, dtype) tuple, extraction returns them exactly."""
        ir_text = _build_matmul_ir(
            m=params["m"],
            k=params["k"],
            n=params["n"],
            dtype=params["dtype"],
            op_name=params["op_name"],
        )
        bounds = self.extractor.extract(ir_text, KernelKind.MATMUL)
        assert bounds.m == params["m"]
        assert bounds.n == params["n"]
        assert bounds.k == params["k"]
        assert bounds.m is not None and bounds.m > 0
        assert bounds.n is not None and bounds.n > 0
        assert bounds.k is not None and bounds.k > 0
        # Dtype round-trips through TTGIRParser._normalize_dtype.
        # All the _DTYPE_TOKENS values are already in canonical form.
        assert (
            bounds.data_dtype
            == {
                "f32": "float32",
                "f16": "float16",
                "bf16": "bfloat16",
                "i32": "int32",
                "i64": "int64",
            }[params["dtype"]]
        )

    @given(params=_bmm_params())
    @_PROP_SETTINGS
    def test_property_bmm_extracts_mnk(self, params: dict) -> None:
        """For any valid (batch, M, K, N) tuple, bmm extraction returns M, N, K."""
        ir_text = _build_bmm_ir(
            batch=params["batch"],
            m=params["m"],
            k=params["k"],
            n=params["n"],
            dtype=params["dtype"],
        )
        bounds = self.extractor.extract(ir_text, KernelKind.MATMUL)
        assert bounds.m == params["m"]
        assert bounds.n == params["n"]
        assert bounds.k == params["k"]
        assert 3 in bounds.tensor_ranks

    @given(params=_reduction_params())
    @_PROP_SETTINGS
    def test_property_reduction_extracts_axis_size(self, params: dict) -> None:
        """For any shape/axis, reduce_size == shape[axis]."""
        ir_text = _build_reduction_ir(
            shape=params["shape"],
            axis=params["axis"],
            dtype=params["dtype"],
        )
        bounds = self.extractor.extract(ir_text, KernelKind.REDUCTION)
        assert bounds.reduce_size == params["shape"][params["axis"]]
        assert bounds.reduce_size is not None and bounds.reduce_size > 0
        expected_keep = 1
        for i, d in enumerate(params["shape"]):
            if i != params["axis"]:
                expected_keep *= d
        assert bounds.keep_size == expected_keep

    @given(params=_elementwise_params())
    @_PROP_SETTINGS
    def test_property_elementwise_extracts_total(self, params: dict) -> None:
        """For any shape, total_elements == product of shape dims."""
        ir_text = _build_elementwise_ir(
            shape=params["shape"],
            dtype=params["dtype"],
        )
        bounds = self.extractor.extract(ir_text, KernelKind.ELEMENTWISE)
        expected = 1
        for d in params["shape"]:
            expected *= d
        assert bounds.total_elements == expected
        assert bounds.total_elements > 0


# ---------------------------------------------------------------------------
# AST-shape contract: verify the extractor really is AST-based, not regex.
# These tests would fail against a regex-based fallback (e.g. an IR where
# the shape digits are spread across a multi-line continuation) because
# the parser handles multi-line ops correctly but a naive whole-IR scan
# would miss them.
# ---------------------------------------------------------------------------


class TestASTRobustness:
    """Tests that verify the AST path is exercised, not a regex fallback."""

    def setup_method(self) -> None:
        self.extractor = BoundsExtractor()

    def test_multiline_dot_extracts_correctly(self) -> None:
        """A ``tt.dot`` whose operands span multiple lines is still parsed.

        The TTGIRParser's recursive-descent _parse_ops tracks brace
        depth, so a multi-line shape annotation is still part of the
        same op. A regex-based extractor would either miss it or
        extract only the first line's shape.
        """
        ir = textwrap.dedent("""
            module {
              tt.func @kernel() {
                %A = tt.load %A_ptr
                  : tensor<128x64xf32>
                %B = tt.load %B_ptr
                  : tensor<64x256xf32>
                %result = tt.dot %A, %B
                  : tensor<128x256xf32>
                tt.return
              }
            }
        """)
        bounds = self.extractor.extract(ir, KernelKind.MATMUL)
        assert bounds.m == 128
        assert bounds.n == 256
        assert bounds.k == 64

    def test_dot_inside_loop_extracts_correctly(self) -> None:
        """A ``tt.dot`` nested inside ``scf.for`` is found by ``iter_all_ops``."""
        ir = textwrap.dedent("""
            module {
              tt.func @kernel(%A_ptr: !tt.ptr<tensor<128x128xf32>>) {
                scf.for %i = 0 to 8 {
                  %A = tt.load %A_ptr : tensor<128x128xf32>
                  %B = tt.load %A_ptr : tensor<128x128xf32>
                  %C = tt.dot %A, %B : tensor<128x128xf32>
                }
                tt.return
              }
            }
        """)
        bounds = self.extractor.extract(ir, KernelKind.MATMUL)
        assert bounds.m == 128
        assert bounds.n == 128
        assert bounds.k == 128

    def test_dtype_extraction_via_ast_not_regex(self) -> None:
        """The dtype comes from the AST's TTGIRType, not a regex sweep.

        If a digit ('8') in the dtype could confuse a regex (e.g. the
        previous implementation's INT_RE), this test would extract the
        wrong dim. Here the dtype contains '8' (``f8`` is not in our
        list, but we use a legitimate dtype that the AST handles).
        """
        ir = textwrap.dedent("""
            module {
              tt.func @kernel() {
                %A = tt.load %A_ptr : tensor<128x64xi64>
                %B = tt.load %B_ptr : tensor<64x128xi64>
                %C = tt.dot %A, %B : tensor<128x128xi64>
                tt.return
              }
            }
        """)
        bounds = self.extractor.extract(ir, KernelKind.MATMUL)
        assert bounds.data_dtype == "int64"
        assert bounds.m == 128
        assert bounds.n == 128
        assert bounds.k == 64


# ---------------------------------------------------------------------------
# Exception-class contract
# ---------------------------------------------------------------------------


class TestBoundsExtractionError:
    """The :class:`BoundsExtractionError` type is the single failure signal."""

    def test_is_an_exception(self) -> None:
        """BoundsExtractionError is an Exception subclass."""
        assert issubclass(BoundsExtractionError, Exception)

    def test_raises_with_clear_message(self) -> None:
        """The error message contains useful diagnostic context."""
        with pytest.raises(BoundsExtractionError) as exc_info:
            BoundsExtractor().extract("garbage", KernelKind.MATMUL)
        assert "Failed to parse TTGIR" in str(exc_info.value)

    def test_can_be_caught_as_exception(self) -> None:
        """Callers can catch BoundsExtractionError as Exception."""
        with pytest.raises(BoundsExtractionError):
            BoundsExtractor().extract("garbage", KernelKind.MATMUL)
        assert isinstance(
            BoundsExtractionError("x"),
            Exception,
        )
