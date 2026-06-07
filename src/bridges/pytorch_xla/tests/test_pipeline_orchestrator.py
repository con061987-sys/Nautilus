"""Tests for the pipeline orchestrator module."""

from __future__ import annotations

import pytest

from src.bridges.pytorch_xla.device_mesh import (
    DeviceMesh,
    DeviceVendor,
    InterconnectType,
    MeshDevice,
)
from src.bridges.pytorch_xla.gspmd_runner import ShardingStrategy
from src.bridges.pytorch_xla.pipeline_orchestrator import (
    AutoShardingBridge,
    ShardingConfig,
    ShardingResult,
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


# ── Happy path with a real model + ShardingConfig ─────────────────────
#
# These tests use the actual `ShardingConfig(model=..., example_inputs=...,
# device_mesh=...)` form requested in the spec. The model is a tiny
# nn.Linear so PyTorch can capture it; the mesh is a 2-device NVLink
# mesh, which exercises the real pipeline. The test does not require
# torch_xla or TVM to be installed — the bridge must handle the
# unavailable-tier case as a graceful no-op, not a hard crash.

torch = pytest.importorskip("torch", reason="PyTorch is required for the pipeline happy-path test")
import torch.nn as nn  # noqa: E402


def _make_real_model() -> nn.Module:
    """A tiny linear model — exercises the full pipeline without GPU."""
    return nn.Linear(8, 4)


class TestPipelineHappyPath:
    """End-to-end pipeline using real ShardingConfig + real model."""

    def test_shard_config_holds_real_model_and_inputs(self) -> None:
        """A ShardingConfig must accept and store a real model + tensors."""
        model = _make_real_model()
        example_inputs = (torch.randn(2, 8),)
        mesh = make_mesh()
        config = ShardingConfig(
            model=model,
            example_inputs=example_inputs,
            device_mesh=mesh,
        )

        assert config.model is model
        assert config.example_inputs is example_inputs
        assert config.device_mesh is mesh
        # Optional fields keep their defaults
        assert config.sharding_strategy == ShardingStrategy.AUTO
        assert config.timeout_seconds > 0

    def test_shard_with_real_model_returns_sharding_result(self) -> None:
        """shard() must accept a real model and return a ShardingResult.

        The pipeline may legitimately fail in the absence of torch_xla /
        TVM (no export tier is available), in which case the result has
        success=False and a populated stage_durations map. Either way the
        call must NOT raise and the return type must be ShardingResult.
        """
        model = _make_real_model()
        example_inputs = (torch.randn(2, 8),)
        mesh = make_mesh()
        config = ShardingConfig(
            model=model,
            example_inputs=example_inputs,
            device_mesh=mesh,
            # Skip fat-binary build — that requires the Phase 2 stack.
            skip_fat_binary=True,
        )

        bridge = AutoShardingBridge(enable_circuit_breakers=False)
        result = bridge.shard(
            model=model,
            example_inputs=example_inputs,
            device_mesh=mesh,
            config=config,
        )

        assert isinstance(result, ShardingResult)
        # Stage timings must always be recorded, even on failure.
        assert "graph_capture" in result.stage_durations
        # Either capture succeeded (and downstream may or may not) or
        # capture itself failed — both are acceptable as long as no
        # exception escaped.
        if result.captured_graph is not None:
            assert result.captured_graph.is_usable is True
            # Capture must have produced a non-zero op count for a real model
            assert result.captured_graph.metadata.op_count > 0
            # Input shapes must match the example inputs
            assert (2, 8) in result.captured_graph.metadata.input_shapes

    def test_shard_with_skip_flags_returns_usable_result(self) -> None:
        """When sharding and DTensor are skipped, the pipeline must still
        complete end-to-end and produce a ShardingResult with the
        capture + export stages populated.
        """
        model = _make_real_model()
        example_inputs = (torch.randn(1, 8),)
        mesh = make_mesh()
        config = ShardingConfig(
            model=model,
            example_inputs=example_inputs,
            device_mesh=mesh,
            skip_sharding=True,
            skip_dtensor=True,
            skip_fat_binary=True,
        )

        bridge = AutoShardingBridge(enable_circuit_breakers=False)
        result = bridge.shard(
            model=model,
            example_inputs=example_inputs,
            device_mesh=mesh,
            config=config,
        )

        # With the heavy stages skipped, this is the only path that can
        # succeed without any of torch_xla/TVM installed.
        assert result.success is True
        assert result.captured_graph is not None
        assert result.captured_graph.is_usable is True
        # The dtensor plan is allowed to be un-usable since we skipped it,
        # but the captured graph must be present.
        assert "graph_capture" in result.stage_durations
        assert "stablehlo_export" in result.stage_durations

    def test_shard_records_total_duration(self) -> None:
        """total_duration_ms must always be populated."""
        model = _make_real_model()
        example_inputs = (torch.randn(1, 8),)
        mesh = make_mesh()
        config = ShardingConfig(
            model=model,
            example_inputs=example_inputs,
            device_mesh=mesh,
            skip_sharding=True,
            skip_dtensor=True,
            skip_fat_binary=True,
        )

        bridge = AutoShardingBridge(enable_circuit_breakers=False)
        result = bridge.shard(
            model=model,
            example_inputs=example_inputs,
            device_mesh=mesh,
            config=config,
        )
        assert result.total_duration_ms > 0.0

    def test_shard_with_real_model_sets_infrastructure(self) -> None:
        """After shard() with a real mesh, the comm backend and executor
        must be configured against that mesh.
        """
        model = _make_real_model()
        example_inputs = (torch.randn(1, 8),)
        mesh = make_mesh()
        config = ShardingConfig(
            model=model,
            example_inputs=example_inputs,
            device_mesh=mesh,
            skip_sharding=True,
            skip_dtensor=True,
            skip_fat_binary=True,
        )

        bridge = AutoShardingBridge(enable_circuit_breakers=False)
        bridge.shard(
            model=model,
            example_inputs=example_inputs,
            device_mesh=mesh,
            config=config,
        )

        assert bridge.comm_backend is not None
        assert bridge.comm_backend.mesh is mesh
        assert bridge.executor is not None

    def test_shard_to_dict_for_real_run(self) -> None:
        """to_dict() must include every public flag after a real run."""
        model = _make_real_model()
        example_inputs = (torch.randn(1, 8),)
        mesh = make_mesh()
        config = ShardingConfig(
            model=model,
            example_inputs=example_inputs,
            device_mesh=mesh,
            skip_sharding=True,
            skip_dtensor=True,
            skip_fat_binary=True,
        )

        bridge = AutoShardingBridge(enable_circuit_breakers=False)
        result = bridge.shard(
            model=model,
            example_inputs=example_inputs,
            device_mesh=mesh,
            config=config,
        )
        d = result.to_dict()
        # All five booleans must be present
        assert "success" in d
        assert "captured" in d
        assert "stablehlo" in d
        assert "gspmd" in d
        assert "dtensor" in d
        assert d["captured"] is True
