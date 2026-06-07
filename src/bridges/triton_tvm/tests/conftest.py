"""Test fixtures for the Triton ↔ TVM bridge."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

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


@pytest.fixture
def auto_tuning_bridge(cache_dir: str) -> Any:
    """Create a TritonTVMBridge with all external systems mocked.

    This fixture:
    - Patches ``TVM_AVAILABLE`` to ``True`` in both ``tir_template`` and
      ``metaschedule_adapter`` so the bridge initialises with TVM enabled.
    - Replaces ``TIRTemplateBuilder`` with a mock that returns a dummy
      IR module (no real TVM required).
    - Replaces ``MetaScheduleAdapter.tune`` with a mock that returns
      ``Ok(MappedTuningConfig(block_m=64, ...))`` by default.

    The returned bridge's ``tvm_adapter.tune`` can be further configured
    in individual tests to simulate failures, timeouts, or different
    tuning results.

    Usage::

        def test_something(auto_tuning_bridge):
            bridge = auto_tuning_bridge
            bridge.tvm_adapter.tune.return_value = Ok(...)
            result = bridge._tuning_chain(metadata, target)
    """
    import src.bridges.triton_tvm.bridge_orchestrator as bo_mod
    import src.bridges.triton_tvm.metaschedule_adapter as ms_mod

    from src.bridges.triton_tvm.bridge_orchestrator import TritonTVMBridge
    from src.bridges.triton_tvm.config_mapper import MappedTuningConfig
    from src.common.result import Ok

    with (
        patch.object(bo_mod, "TVM_AVAILABLE", True),
        patch.object(ms_mod, "TVM_AVAILABLE", True),
    ):
        bridge = TritonTVMBridge(cache_dir=cache_dir, enable_tvm=True)

        bridge._build_tir_template = MagicMock(return_value=MagicMock())

        bridge.tvm_adapter.tune = MagicMock(
            return_value=Ok(MappedTuningConfig(
                block_m=64, block_n=128, block_k=64,
                num_warps=8, num_stages=4,
            )),
        )

        return bridge
