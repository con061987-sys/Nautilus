"""Tests for the CUDA to Triton translator."""

from __future__ import annotations

from typing import Any

from src.bridges.cuda_ingest.intrinsic_mapper import IntrinsicMapper
from src.bridges.cuda_ingest.parser import CudaParser
from src.bridges.cuda_ingest.translator import (
    CudaToTritonTranslator,
    _decompose_compound_assignment,
    _is_scalar_lhs,
)

SAMPLE_KERNEL = """
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
"""


SAMPLE_MATMUL_KERNEL = """
__global__ void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float tile[16][16];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
    __syncthreads();
}
"""


SAMPLE_DEVICE_FUNCTION = """
__device__ float helper(float x) {
    return x * 2.0f;
}
"""


SAMPLE_COMPOUND_KERNEL = """
__global__ void accumulate(float* x, float* y, int n) {
    float sum = 0.0f;
    sum += x[0];
    sum -= y[0];
    sum *= 2.0f;
    x[1] = sum;
}
"""


SAMPLE_AUTO_KERNEL = """
__global__ void use_auto(float* x, int n) {
    auto len = n;
    auto& ref = x[0];
    const auto* p = x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = ref + len;
    }
}
"""


SAMPLE_DECLTYPE_KERNEL = """
__global__ void use_decltype(float* x, int n) {
    decltype(x[0]) first = x[0];
    static const decltype(n) limit = 100;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = first;
    }
}
"""


SAMPLE_STD_MOVE_KERNEL = """
__global__ void use_move(float* x, int n) {
    float* p = std::move(x);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = p[i] + 1.0f;
    }
}
"""


SAMPLE_ATOMIC_UNSAFE_KERNEL = """
__global__ void compound_deref(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] += 1.0f;
    }
}
"""


def make_kernel(source: str) -> Any:
    """Parse a CUDA source and return the first kernel."""
    parser = CudaParser()
    kernels = parser.parse_source(source)
    return kernels[0] if kernels else None


class TestCudaToTritonTranslator:
    """Tests for the CudaToTritonTranslator class."""

    def test_translator_init(self) -> None:
        """CudaToTritonTranslator should initialise without error."""
        translator = CudaToTritonTranslator()
        assert translator is not None

    def test_translate_simple_kernel(self) -> None:
        """A simple vector_add kernel should translate successfully."""
        kernel = make_kernel(SAMPLE_KERNEL)
        translator = CudaToTritonTranslator()
        result = translator.translate(kernel)
        assert result.is_usable
        assert "vector_add" in result.triton_source
        assert "@triton.jit" in result.triton_source
        assert "tl.program_id" in result.triton_source

    def test_translate_matmul_kernel(self) -> None:
        """A matmul kernel should translate with shared memory."""
        kernel = make_kernel(SAMPLE_MATMUL_KERNEL)
        translator = CudaToTritonTranslator()
        result = translator.translate(kernel)
        assert result.is_usable
        # Should include tl.zeros for shared memory
        assert "tl.zeros" in result.triton_source
        # Should include tl.barrier for __syncthreads
        assert "tl.barrier" in result.triton_source

    def test_translate_rejects_device_function(self) -> None:
        """__device__ functions should not be translatable as kernels."""
        kernel = make_kernel(SAMPLE_DEVICE_FUNCTION)
        translator = CudaToTritonTranslator()
        result = translator.translate(kernel)
        assert not result.is_usable
        assert result.error is not None
        assert "device" in result.error.lower()

    def test_translation_preserves_kernel_name(self) -> None:
        """The translated function should have the same name as the CUDA kernel."""
        kernel = make_kernel(SAMPLE_KERNEL)
        translator = CudaToTritonTranslator()
        result = translator.translate(kernel)
        assert result.kernel_name == kernel.name
        assert f"def {kernel.name}" in result.triton_source

    def test_translation_records_pointer_layouts(self) -> None:
        """The translation should record the inferred pointer layouts."""
        kernel = make_kernel(SAMPLE_KERNEL)
        translator = CudaToTritonTranslator()
        result = translator.translate(kernel)
        assert "a" in result.pointer_layouts
        assert "b" in result.pointer_layouts
        assert "c" in result.pointer_layouts

    def test_translation_records_shared_mem_plan(self) -> None:
        """The translation should record the shared memory plan."""
        kernel = make_kernel(SAMPLE_MATMUL_KERNEL)
        translator = CudaToTritonTranslator()
        result = translator.translate(kernel)
        assert result.shared_mem_plan is not None
        assert result.shared_mem_plan.num_allocations >= 1

    def test_translation_collects_warnings(self) -> None:
        """Bank conflict warnings should be collected in the result."""
        kernel = make_kernel(SAMPLE_MATMUL_KERNEL)
        translator = CudaToTritonTranslator()
        result = translator.translate(kernel)
        # Warnings may or may not be present depending on shared memory analysis
        assert isinstance(result.warnings, list)


# ---------------------------------------------------------------------------
# Limitation-fix tests — one per documented limitation in
# docs/cuda_ingestion_architecture.md.
# ---------------------------------------------------------------------------


class TestBlockLinearizationPattern:
    """Limitation 2: `blockIdx.d * blockDim.d + threadIdx.d` idiom."""

    def test_gpu_path_rewrites_block_linearization(self) -> None:
        """AST-level (GPU) path rewrites the canonical pattern."""
        kernel = make_kernel(SAMPLE_KERNEL)
        result = CudaToTritonTranslator().translate(kernel)
        assert "tl.program_id(0) * tl.num_programs(0) + tl.program_id(0)" in result.triton_source

    def test_cpu_path_rewrites_block_linearization(self) -> None:
        """Text-level (CPU/fallback) path also rewrites the canonical pattern."""
        mapper = IntrinsicMapper()
        text = "int i = blockIdx.x * blockDim.x + threadIdx.x;"
        out = mapper.transform_text(text)
        assert "tl.program_id(0) * tl.num_programs(0) + tl.program_id(0)" in out

    def test_block_linearization_handles_y_dimension(self) -> None:
        """The pattern match works for .y/.z dims, not just .x."""
        source = """
__global__ void k(float* a, int N) {
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j < N) {
        a[j] = 1.0f;
    }
}
"""
        result = CudaToTritonTranslator().translate(make_kernel(source))
        assert "tl.program_id(1) * tl.num_programs(1) + tl.program_id(1)" in result.triton_source

    def test_mixed_dim_pattern_falls_through(self) -> None:
        """Mixed-dim expressions are NOT caught by the canonical pattern."""
        source = """
__global__ void k(float* a, int N) {
    int i = blockIdx.x * blockDim.y + threadIdx.x;
    if (i < N) {
        a[i] = 1.0f;
    }
}
"""
        result = CudaToTritonTranslator().translate(make_kernel(source))
        assert "* tl.num_programs(0) +" not in result.triton_source


class TestCompoundAssignmentDecomposition:
    """Limitation 4: `+=`, `-=`, etc. → load → modify → store sequence."""

    def test_decompose_simple_plus_equals(self) -> None:
        """`x += y` decomposes to the explicit 3-line form."""
        lines = _decompose_compound_assignment("x += y")
        assert lines == [
            "__tmp_x = x",
            "__tmp_x = __tmp_x + y",
            "x = __tmp_x",
        ]

    def test_decompose_minus_equals(self) -> None:
        """`-=` decomposes correctly."""
        assert _decompose_compound_assignment("sum -= delta") == [
            "__tmp_sum = sum",
            "__tmp_sum = __tmp_sum - delta",
            "sum = __tmp_sum",
        ]

    def test_decompose_all_compound_operators(self) -> None:
        """All 10 compound operators decompose to their plain binary form."""
        cases = {
            "+=": "+",
            "-=": "-",
            "*=": "*",
            "/=": "/",
            "%=": "%",
            "&=": "&",
            "|=": "|",
            "^=": "^",
            "<<=": "<<",
            ">>=": ">>",
        }
        for cuda_op, py_op in cases.items():
            lines = _decompose_compound_assignment(f"a {cuda_op} b")
            assert lines is not None, f"{cuda_op} did not decompose"
            assert lines[0] == "__tmp_a = a"
            assert lines[1] == f"__tmp_a = __tmp_a {py_op} b"
            assert lines[2] == "a = __tmp_a"

    def test_decompose_non_compound_returns_none(self) -> None:
        """A plain assignment returns None (no decomposition)."""
        assert _decompose_compound_assignment("x = y") is None
        assert _decompose_compound_assignment("x = x + y") is None

    def test_decompose_memory_deref_emits_review_comment(self) -> None:
        """`x[i] += y` (non-scalar LHS) emits a review comment instead of
        silently producing a racy non-atomic translation."""
        lines = _decompose_compound_assignment("x[i] += y")
        assert lines is not None
        assert len(lines) == 1
        assert lines[0].startswith("# (atomic-unsafe)")

    def test_decompose_struct_field_emits_review_comment(self) -> None:
        """`obj.field += y` (non-scalar LHS) also emits a review comment."""
        lines = _decompose_compound_assignment("obj.field += 1")
        assert lines is not None
        assert len(lines) == 1
        assert "# (atomic-unsafe)" in lines[0]

    def test_decompose_with_rhs_compound_expression(self) -> None:
        """`x += a + b * c` decomposes keeping the full RHS intact."""
        lines = _decompose_compound_assignment("x += a + b * c")
        assert lines == [
            "__tmp_x = x",
            "__tmp_x = __tmp_x + a + b * c",
            "x = __tmp_x",
        ]

    def test_is_scalar_lhs(self) -> None:
        """Scalar LHS detection covers the common cases."""
        assert _is_scalar_lhs("x") is True
        assert _is_scalar_lhs("sum") is True
        assert _is_scalar_lhs("x[i]") is False
        assert _is_scalar_lhs("obj.field") is False
        assert _is_scalar_lhs("p->next") is False

    def test_full_translation_emits_three_lines(self) -> None:
        """The full translator emits 3 lines for a scalar `+=`."""
        import re

        kernel = make_kernel(SAMPLE_COMPOUND_KERNEL)
        result = CudaToTritonTranslator().translate(kernel)
        body = result.triton_source
        # Count the load step (must be exactly 3 — one per compound op).
        load_count = body.count("__tmp_sum = sum\n")
        assert load_count == 3
        # Count the store step at the start of a line (with leading
        # indent).  The modify step also contains `sum = __tmp_sum`
        # as a substring, so a bare `.count(...)` would over-count.
        store_count = len(re.findall(r"^\s{4}sum = __tmp_sum$", body, re.M))
        assert store_count == 3

    def test_atomic_unsafe_top_level_compound(self) -> None:
        """`x[i] += y` at the top level (not nested in an if/for body)
        produces an atomic-unsafe review comment in the body.

        Note: the translator does not currently recurse into if/for
        bodies (pre-existing limitation), so this test only covers the
        top-level case.  The unit-level coverage for nested cases is
        in `test_decompose_memory_deref_emits_review_comment`.
        """
        source = """
__global__ void compound_deref(float* x) {
    x[0] += 1.0f;
}
"""
        result = CudaToTritonTranslator().translate(make_kernel(source))
        assert "# (atomic-unsafe)" in result.triton_source


class TestCpp11Features:
    """Limitation 5: `auto`, `decltype`, move semantics."""

    def test_auto_declaration_strips_type(self) -> None:
        """`auto x = expr;` translates to `x = expr;`."""
        kernel = make_kernel(SAMPLE_AUTO_KERNEL)
        result = CudaToTritonTranslator().translate(kernel)
        # Look at body lines (after the function signature) only, so
        # the kernel name "use_auto" doesn't false-match.
        body_section = result.triton_source.split(
            "    # Translated kernel body",
            1,
        )[-1]
        body_lines = [
            ln
            for ln in body_section.splitlines()
            if "auto" in ln and not ln.lstrip().startswith("#")
        ]
        assert body_lines == [], f"auto leaked into body: {body_lines}"

    def test_auto_reference_declaration_strips_type(self) -> None:
        """`auto& x = expr;` translates to `x = expr;`."""
        kernel = make_kernel(SAMPLE_AUTO_KERNEL)
        result = CudaToTritonTranslator().translate(kernel)
        assert "ref = x[0]" in result.triton_source

    def test_const_auto_pointer_declaration_strips_type(self) -> None:
        """`const auto* p = x;` translates to `p = x;`."""
        kernel = make_kernel(SAMPLE_AUTO_KERNEL)
        result = CudaToTritonTranslator().translate(kernel)
        assert "p = x" in result.triton_source

    def test_decltype_declaration_strips_type(self) -> None:
        """`decltype(x[0]) first = x[0];` translates to `first = x[0];`."""
        kernel = make_kernel(SAMPLE_DECLTYPE_KERNEL)
        result = CudaToTritonTranslator().translate(kernel)
        body_section = result.triton_source.split(
            "    # Translated kernel body",
            1,
        )[-1]
        body_lines = [
            ln
            for ln in body_section.splitlines()
            if "decltype" in ln and not ln.lstrip().startswith("#")
        ]
        assert body_lines == [], f"decltype leaked into body: {body_lines}"
        assert "first = x[0]" in result.triton_source

    def test_static_const_decltype_strips_type(self) -> None:
        """`static const decltype(n) limit = 100;` → `limit = 100;`."""
        kernel = make_kernel(SAMPLE_DECLTYPE_KERNEL)
        result = CudaToTritonTranslator().translate(kernel)
        assert "limit = 100" in result.triton_source

    def test_std_move_unwrapped(self) -> None:
        """`std::move(x)` is unwrapped to just `x` (rvalue cast is a no-op)."""
        kernel = make_kernel(SAMPLE_STD_MOVE_KERNEL)
        result = CudaToTritonTranslator().translate(kernel)
        assert "std::move" not in result.triton_source
        assert "p = x" in result.triton_source

    def test_std_move_does_not_match_substring(self) -> None:
        """`mystd::move` (user-defined namespace) must NOT be unwrapped."""
        source = """
__global__ void k(float* x, int n) {
    float* p = mystd::move(x);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { x[i] = p[i]; }
}
"""
        result = CudaToTritonTranslator().translate(make_kernel(source))
        assert "mystd::move" in result.triton_source


class TestSharedMemoryMultiDim:
    """Limitation 1: multi-dim `__shared__` arrays.

    Most of the surface area is covered in test_shared_memory.py; the
    end-to-end test here exercises the translator + analyzer together
    so a regression in either would be caught.
    """

    def test_translator_records_total_size_for_2d(self) -> None:
        """A 2D __shared__ array has total size = D1 * D2 in the plan."""
        kernel = make_kernel(SAMPLE_MATMUL_KERNEL)
        result = CudaToTritonTranslator().translate(kernel)
        plan = result.shared_mem_plan
        assert plan is not None
        tile_alloc = next(a for a in plan.allocations if a.name == "tile")
        assert tile_alloc.array_size == 16 * 16
        assert tile_alloc.is_static is True

    def test_translator_emits_tl_zeros_for_2d(self) -> None:
        """The emitted Triton code allocates the flattened 2D size."""
        kernel = make_kernel(SAMPLE_MATMUL_KERNEL)
        result = CudaToTritonTranslator().translate(kernel)
        assert "tl.zeros((256,)" in result.triton_source
