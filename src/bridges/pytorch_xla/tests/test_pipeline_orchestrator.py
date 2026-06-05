"""Tests for the pipeline orchestrator module."""

from __future__ import annotations

import pytest

from src.bridges.pytorch_xla.pipeline_orchestrator import (
    AutoShardingBridge,
    ShardingConfig,
    ShardingResult,
)
from src.bridges.pytorch_xla.gspmd_runner import ShardingStrategy
from src.bridges.pytorch_xla.device_mesh import (
    DeviceMesh,
    DeviceVendor,
    InterconnectType,
    MeshDevice,
)


def make_mesh() -> DeviceMesh:
    """Create a test mesh."""
    devices = [
        MeshDevice(i, DeviceVendor.NVIDIA, "sm_90", 80.0, 989.0, InterconnectType.NVLINK)
        for i in range(2)
    ]
    return DeviceMesh(devices=devices, mesh_shape=[2])


class TestShardingConfig:
    """Tests for the ShardingConfig class."""

    def test_config_defaults(self) -> None:
        """ShardingConfig should have sensible defaults."""
        config = ShardingConfig(
            model=None,
            example_inputs=(),
            device_mesh=make_mesh(),
        )
        assert config.sharding_strategy == ShardingStrategy.AUTO
        assert config.enable_cache is True
        assert config.timeout_seconds > 0


class TestAutoShardingBridge:
    """Tests for the AutoShardingBridge class."""

    def test_bridge_init(self) -> None:
        """AutoShardingBridge should initialise all stages."""
        bridge = AutoShardingBridge()
        assert bridge.graph_capture is not None
        assert bridge.stablehlo_exporter is not None
        assert bridge.gspmd_runner is not None

    def test_bridge_init_without_circuit_breakers(self) -> None:
        """AutoShardingBridge should support disabling circuit breakers."""
        bridge = AutoShardingBridge(enable_circuit_breakers=False)
        assert bridge.breakers is None
        assert bridge.timeout_manager is None

    def test_shard_with_no_real_model(self) -> None:
        """shard() should return a graceful failure when no model is provided."""
        bridge = AutoShardingBridge()
        mesh = make_mesh()
        config = ShardingConfig(
            model=None,
            example_inputs=(),
            device_mesh=mesh,
        )
        result = bridge.shard(
            model=None,
            example_inputs=(),
            device_mesh=mesh,
            config=config,
        )
        # The result should report failure
        assert result.success is False
        assert result.error is not None

    def test_shard_records_stage_durations(self) -> None:
        """shard() should record timing for each stage."""
        bridge = AutoShardingBridge()
        mesh = make_mesh()
        result = bridge.shard(
            model=None,
            example_inputs=(),
            device_mesh=mesh,
        )
        assert isinstance(result.stage_durations, dict)
        # At minimum, graph_capture should be recorded
        assert "graph_capture" in result.stage_durations

    def test_shard_to_dict(self) -> None:
        """ShardingResult.to_dict should serialise the result."""
        bridge = AutoShardingBridge()
        mesh = make_mesh()
        result = bridge.shard(
            model=None,
            example_inputs=(),
            device_mesh=mesh,
        )
        d = result.to_dict()
        assert "success" in d
        assert "total_duration_ms" in d
        assert "stage_durations" in d
        assert "captured" in d

    def test_shard_sets_comm_backend(self) -> None:
        """shard() should set up the comm backend for the mesh."""
        bridge = AutoShardingBridge()
        mesh = make_mesh()
        bridge.shard(
            model=None,
            example_inputs=(),
            device_mesh=mesh,
        )
        # After shard(), the comm backend should be set
        assert bridge.comm_backend is not None
        assert bridge.comm_backend.mesh is mesh

    def test_shard_sets_executor(self) -> None:
        """shard() should set up the shard executor."""
        bridge = AutoShardingBridge()
        mesh = make_mesh()
        bridge.shard(
            model=None,
            example_inputs=(),
            device_mesh=mesh,
        )
        assert bridge.executor is not None
