"""Test fixtures for the Triton ↔ TVM bridge."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def cache_dir(tmp_path: Path) -> str:
    """Temporary cache directory for testing."""
    d = tmp_path / "nvindia_cache"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


@pytest.fixture
def sample_matmul_metadata() -> Any:
    """Create a sample KernelMetadata for a matmul kernel."""
    from src.bridges.triton_tvm.metadata_extractor import KernelMetadata
    return KernelMetadata(
        kernel_name="matmul_kernel",
        source_hash="abc123",
        grid_0=4,
        grid_1=4,
        grid_2=1,
        num_warps=4,
        num_stages=3,
        num_ctas=1,
        arg_shapes=((1024, 1024), (1024, 1024), (1024, 1024)),
        arg_strides=((1024, 1), (1024, 1), (1024, 1)),
        arg_dtypes=("float32", "float32", "float32"),
        is_matmul=True,
        matmul_m=1024,
        matmul_n=1024,
        matmul_k=1024,
    )


@pytest.fixture
def sample_reduction_metadata() -> Any:
    """Create a sample KernelMetadata for a reduction kernel."""
    from src.bridges.triton_tvm.metadata_extractor import KernelMetadata
    return KernelMetadata(
        kernel_name="reduce_kernel",
        source_hash="def456",
        grid_0=1,
        grid_1=1,
        grid_2=1,
        num_warps=4,
        num_stages=3,
        num_ctas=1,
        arg_shapes=((1024,), (1024,)),
        arg_strides=((1,), (1,)),
        arg_dtypes=("float32", "float32"),
        is_reduction=True,
    )


@pytest.fixture
def sample_elementwise_metadata() -> Any:
    """Create a sample KernelMetadata for an elementwise kernel."""
    from src.bridges.triton_tvm.metadata_extractor import KernelMetadata
    return KernelMetadata(
        kernel_name="add_kernel",
        source_hash="ghi789",
        grid_0=8,
        grid_1=1,
        grid_2=1,
        num_warps=4,
        num_stages=3,
        num_ctas=1,
        arg_shapes=((2048,), (2048,), (2048,)),
        arg_strides=((1,), (1,), (1,)),
        arg_dtypes=("float32", "float32", "float32"),
        is_elementwise=True,
    )
