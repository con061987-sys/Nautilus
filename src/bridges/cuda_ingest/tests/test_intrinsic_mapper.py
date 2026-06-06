"""Tests for the CUDA intrinsic mapper."""

from __future__ import annotations

import pytest

from src.bridges.cuda_ingest.intrinsic_mapper import (
    IntrinsicCategory,
    IntrinsicMapper,
    IntrinsicMapping,
)


class TestIntrinsicMapper:
    """Tests for the IntrinsicMapper class."""

    def test_mapper_init(self) -> None:
        """IntrinsicMapper should register default mappings on init."""
        mapper = IntrinsicMapper()
        assert len(mapper.all_mappings()) > 0

    def test_thread_idx_mappings(self) -> None:
        """threadIdx.x/y/z should map to tl.program_id."""
        mapper = IntrinsicMapper()
        for name in ["threadIdx.x", "threadIdx.y", "threadIdx.z"]:
            mapping = mapper.get_mapping(name)
            assert mapping is not None
            assert "tl.program_id" in mapping.triton_name

    def test_block_idx_mappings(self) -> None:
        """blockIdx.x/y/z should map to tl.program_id."""
        mapper = IntrinsicMapper()
        for name in ["blockIdx.x", "blockIdx.y", "blockIdx.z"]:
            mapping = mapper.get_mapping(name)
            assert mapping is not None
            assert "tl.program_id" in mapping.triton_name

    def test_sync_threads_mapping(self) -> None:
        """__syncthreads() should map to tl.barrier()."""
        mapper = IntrinsicMapper()
        mapping = mapper.get_mapping("__syncthreads()")
        assert mapping is not None
        assert mapping.triton_name == "tl.barrier()"
        assert mapping.is_exact is True

    def test_atomic_mappings(self) -> None:
        """All atomic ops should map to their tl.atomic_* equivalents."""
        mapper = IntrinsicMapper()
        for cuda_atomic in ["atomicAdd", "atomicSub", "atomicMin",
                           "atomicMax", "atomicCAS", "atomicExch"]:
            mapping = mapper.get_mapping(cuda_atomic)
            assert mapping is not None
            assert "tl.atomic" in mapping.triton_name

    def test_math_mappings(self) -> None:
        """Math functions should map 1:1 to Triton."""
        mapper = IntrinsicMapper()
        for cuda_math in ["sinf", "cosf", "logf", "expf", "sqrtf"]:
            mapping = mapper.get_mapping(cuda_math)
            assert mapping is not None
            assert mapping.triton_name == f"tl.{cuda_math[:-1]}"  # strip 'f' suffix
            assert mapping.is_exact is True

    def test_transform_text_replaces_intrinsics(self) -> None:
        """transform_text should replace CUDA intrinsics with Triton equivalents."""
        mapper = IntrinsicMapper()
        cuda_code = '''
__global__ void test(int* x) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    __syncthreads();
    x[i] = sinf(x[i]) + cosf(x[i]);
    atomicAdd(&counter, 1);
}
'''
        triton_code = mapper.transform_text(cuda_code)
        assert "tl.program_id(0)" in triton_code
        assert "tl.barrier()" in triton_code
        assert "tl.sin(x[i])" in triton_code
        assert "tl.cos(x[i])" in triton_code
        assert "tl.atomic_add" in triton_code

    def test_transform_text_preserves_syntax(self) -> None:
        """transform_text should not break non-intrinsic syntax."""
        mapper = IntrinsicMapper()
        cuda_code = "int x = 5; int y = x + 3;"
        result = mapper.transform_text(cuda_code)
        assert "int x = 5" in result
        assert "int y = x + 3" in result

    def test_transform_text_handles_block_linearization(self) -> None:
        """The CPU (fallback) path must rewrite the canonical
        ``blockIdx.x * blockDim.x + threadIdx.x`` pattern in one pass.
        """
        mapper = IntrinsicMapper()
        text = "int i = blockIdx.x * blockDim.x + threadIdx.x;"
        result = mapper.transform_text(text)
        assert (
            "tl.program_id(0) * tl.num_programs(0) + tl.program_id(0)"
            in result
        )

    def test_transform_text_handles_block_linearization_yz(self) -> None:
        """CPU path block linearization works for .y and .z dims too."""
        mapper = IntrinsicMapper()
        for dim_letter, dim_idx in [("y", 1), ("z", 2)]:
            text = f"int i = blockIdx.{dim_letter} * blockDim.{dim_letter} + threadIdx.{dim_letter};"
            result = mapper.transform_text(text)
            expected = (
                f"tl.program_id({dim_idx}) * tl.num_programs({dim_idx})"
                f" + tl.program_id({dim_idx})"
            )
            assert expected in result

    def test_transform_text_unwraps_std_move(self) -> None:
        """The CPU (fallback) path must unwrap ``std::move(x)``."""
        mapper = IntrinsicMapper()
        text = "float* p = std::move(x);"
        result = mapper.transform_text(text)
        assert "std::move" not in result
        assert "p = x" in result

    def test_mappings_by_category(self) -> None:
        """mappings_by_category should return mappings in a category."""
        mapper = IntrinsicMapper()
        atomic_mappings = mapper.mappings_by_category(IntrinsicCategory.ATOMIC)
        assert len(atomic_mappings) > 0
        for m in atomic_mappings:
            assert m.category == IntrinsicCategory.ATOMIC

    def test_coverage_report(self) -> None:
        """coverage_report should return a useful summary."""
        mapper = IntrinsicMapper()
        report = mapper.coverage_report()
        assert "total_mappings" in report
        assert "by_category" in report
        assert report["total_mappings"] > 0
