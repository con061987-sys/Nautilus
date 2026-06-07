"""Tests for the bridge orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.bridges.triton_tvm.bridge_orchestrator import (
    FallbackTier,
    MappedTuningConfig,
    TritonTVMBridge,
)


class TestTritonTVMBridge:
    """Bridge orchestrator tests (no GPU required)."""

    def test_init_defaults(self) -> None:
        """Bridge should initialize with sensible defaults."""
        bridge = TritonTVMBridge(enable_tvm=False)
        assert bridge.max_trials == 64
        assert bridge.enable_tvm is False
        assert bridge._max_cache_entries == 256

    def test_init_custom_cache_dir(self, cache_dir: str) -> None:
        """Custom cache dir should be used."""
        bridge = TritonTVMBridge(cache_dir=cache_dir, enable_tvm=False)
        assert str(bridge.cache_dir) == cache_dir

    def test_resolve_grid_tuple(self) -> None:
        """Tuple grid should pass through."""
        result = TritonTVMBridge._resolve_grid((4, 4, 1))
        assert result == (4, 4, 1)

    def test_resolve_grid_partial_tuple(self) -> None:
        """1-tuple and 2-tuple should be extended to 3-tuple."""
        r1 = TritonTVMBridge._resolve_grid((8,))
        assert r1 == (8, 1, 1)
        r2 = TritonTVMBridge._resolve_grid((4, 2))
        assert r2 == (4, 2, 1)

    def test_resolve_grid_callable(self) -> None:
        """Lambda grid should be called and resolved."""
        grid = lambda meta: (4, 4, 1)  # noqa: E731
        result = TritonTVMBridge._resolve_grid(grid)
        assert result == (4, 4, 1)

    def test_resolve_grid_callable_error(self) -> None:
        """Broken callable should fall back to (1,1,1)."""

        def broken(_: Any) -> None:
            raise RuntimeError("boom")

        result = TritonTVMBridge._resolve_grid(broken)
        assert result == (1, 1, 1)

    def test_cache_set_and_get(self) -> None:
        """Set then get should return the same config."""
        bridge = TritonTVMBridge(enable_tvm=False)
        config = MappedTuningConfig(block_m=64, num_warps=8)
        bridge._set_cache("key123", "nvidia/nvidia-a100", config)
        retrieved = bridge._get_cached("key123", "nvidia/nvidia-a100")
        assert retrieved is not None
        assert retrieved.block_m == 64
        assert retrieved.num_warps == 8

    def test_cache_miss(self) -> None:
        """Unknown key should return None."""
        bridge = TritonTVMBridge(enable_tvm=False)
        result = bridge._get_cached("nonexistent", "nvidia/nvidia-a100")
        assert result is None

    def test_cache_lru_eviction(self, tmp_path: Path) -> None:
        """Exceeding max cache entries should evict oldest."""
        bridge = TritonTVMBridge(cache_dir=str(tmp_path / "cache"), enable_tvm=False)
        bridge._max_cache_entries = 3

        for i in range(5):
            bridge._set_cache(f"key{i}", "target", MappedTuningConfig(num_warps=i))

        # keys 0 and 1 should be evicted
        assert bridge._get_cached("key0", "target") is None
        assert bridge._get_cached("key1", "target") is None
        # key 4 should exist
        assert bridge._get_cached("key4", "target") is not None

    def test_cache_disk_persistence(self, cache_dir: str) -> None:
        """Disk cache should survive bridge recreation."""
        bridge1 = TritonTVMBridge(cache_dir=cache_dir, enable_tvm=False)
        bridge1._set_cache("persist", "target", MappedTuningConfig(block_m=256))

        bridge2 = TritonTVMBridge(cache_dir=cache_dir, enable_tvm=False)
        retrieved = bridge2._get_cached("persist", "target")
        assert retrieved is not None
        assert retrieved.block_m == 256

    def test_fallback_matmul(self, sample_matmul_metadata: Any) -> None:
        """Matmul fallback should use sensible defaults."""
        bridge = TritonTVMBridge(enable_tvm=False)
        result = bridge._fallback(sample_matmul_metadata, FallbackTier.L4_TRITON_DEFAULT)
        assert result.block_m == 128
        assert result.block_n == 128
        assert result.block_k == 32
        assert result.num_warps == 4  # min(metadata.num_warps=4, 8)

    def test_fallback_defaults(self, sample_elementwise_metadata: Any) -> None:
        """Non-matmul fallback should use all-defaults."""
        bridge = TritonTVMBridge(enable_tvm=False)
        result = bridge._fallback(sample_elementwise_metadata, FallbackTier.L5_SAFE_FALLBACK)
        assert result == MappedTuningConfig.defaults()


class TestAutotuneConfigs:
    """The autotune_configs convenience function."""

    def test_generates_configs_without_tvm(self, sample_matmul_metadata: Any) -> None:
        """Should generate configs even without TVM."""
        try:
            import importlib

            _ = importlib.util.find_spec("src.bridges.triton_tvm.bridge_orchestrator")
        except (ImportError, AttributeError):
            pytest.skip("triton not installed, skipping integration test")
