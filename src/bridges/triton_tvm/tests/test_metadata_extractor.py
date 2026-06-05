"""Tests for the metadata extractor."""

from __future__ import annotations

from typing import Any

import pytest

from src.bridges.triton_tvm.metadata_extractor import (
    KernelMetadata,
    MetadataExtractor,
)


class TestKernelMetadata:
    """KernelMetadata validation and cache key."""

    def test_invalid_num_warps(self) -> None:
        """Non-power-of-2 num_warps should raise ValueError."""
        with pytest.raises(ValueError, match="num_warps"):
            KernelMetadata(
                kernel_name="test",
                source_hash="x",
                grid_0=1, grid_1=1, grid_2=1,
                num_warps=3,  # not power of 2
                num_stages=3, num_ctas=1,
            )

    def test_cache_key_deterministic(self) -> None:
        """Same metadata should produce the same cache key."""
        m1 = KernelMetadata(
            kernel_name="test", source_hash="abc",
            grid_0=4, grid_1=4, grid_2=1,
            num_warps=4, num_stages=3, num_ctas=1,
            arg_shapes=((128, 128), (128, 128), (128, 128)),
            arg_dtypes=("float32", "float32", "float32"),
            is_matmul=True, matmul_m=128, matmul_n=128, matmul_k=128,
        )
        m2 = KernelMetadata(
            kernel_name="test", source_hash="abc",
            grid_0=4, grid_1=4, grid_2=1,
            num_warps=4, num_stages=3, num_ctas=1,
            arg_shapes=((128, 128), (128, 128), (128, 128)),
            arg_dtypes=("float32", "float32", "float32"),
            is_matmul=True, matmul_m=128, matmul_n=128, matmul_k=128,
        )
        assert m1.cache_key == m2.cache_key
        assert len(m1.cache_key) == 64  # SHA-256 hex

    def test_cache_key_differs_on_shape(self) -> None:
        """Different shapes should produce different cache keys."""
        m_small = KernelMetadata(
            kernel_name="test", source_hash="abc",
            grid_0=1, grid_1=1, grid_2=1,
            num_warps=4, num_stages=3, num_ctas=1,
            arg_shapes=((64, 64), (64, 64)),
            arg_dtypes=("float32", "float32"),
        )
        m_large = KernelMetadata(
            kernel_name="test", source_hash="abc",
            grid_0=1, grid_1=1, grid_2=1,
            num_warps=4, num_stages=3, num_ctas=1,
            arg_shapes=((256, 256), (256, 256)),
            arg_dtypes=("float32", "float32"),
        )
        assert m_small.cache_key != m_large.cache_key

    def test_grid_property(self) -> None:
        """Grid property returns a tuple."""
        m = KernelMetadata(
            kernel_name="test", source_hash="x",
            grid_0=4, grid_1=8, grid_2=1,
            num_warps=4, num_stages=3, num_ctas=1,
        )
        assert m.grid == (4, 8, 1)


class TestMetadataExtractor:
    """Metadata extraction from kernel sources."""

    def test_classify_matmul(self) -> None:
        """Kernel with tl.dot should classify as matmul."""
        extractor = MetadataExtractor()

        class FakeMatmul:
            __name__ = "matmul_kernel"
            __module__ = "test"
            def __call__(self) -> None: pass

        # Monkey-patch getsource for testing
        import inspect
        original_getsource = inspect.getsource
        inspect.getsource = lambda fn: "def matmul_kernel:\n  tl.dot(a, b, c)"

        try:
            result = extractor._classify_kernel(FakeMatmul())
            assert result == "matmul"
        finally:
            inspect.getsource = original_getsource

    def test_classify_reduction(self) -> None:
        """Kernel with tl.reduce should classify as reduction."""
        extractor = MetadataExtractor()

        class FakeReduce:
            __name__ = "reduce_kernel"
            __module__ = "test"
            def __call__(self) -> None: pass

        import inspect
        original_getsource = inspect.getsource
        inspect.getsource = lambda fn: "def reduce_kernel:\n  tl.reduce(x, y, z)"

        try:
            result = extractor._classify_kernel(FakeReduce())
            assert result == "reduction"
        finally:
            inspect.getsource = original_getsource

    def test_classify_unknown(self) -> None:
        """Kernel with no recognizable ops should classify as unknown."""
        extractor = MetadataExtractor()

        class FakeUnknown:
            __name__ = "weird_kernel"
            __module__ = "test"
            def __call__(self) -> None: pass

        import inspect
        original_getsource = inspect.getsource
        inspect.getsource = lambda fn: "def weird_kernel:\n  some_other_op(x)"

        try:
            result = extractor._classify_kernel(FakeUnknown())
            assert result == "unknown"
        finally:
            inspect.getsource = original_getsource

    def test_compute_source_hash_stable(self) -> None:
        """Same source should produce same hash regardless of filename."""
        extractor = MetadataExtractor()

        class FakeA:
            __name__ = "kernel"
            __module__ = "test"

        class FakeB:
            __name__ = "kernel"
            __module__ = "test"

        import inspect
        source_code = "def kernel(a, b):\n  return a + b"
        original_getsource = inspect.getsource
        inspect.getsource = lambda fn: source_code

        try:
            hash_a = extractor._compute_source_hash(FakeA())
            hash_b = extractor._compute_source_hash(FakeB())
            assert hash_a == hash_b
        finally:
            inspect.getsource = original_getsource
