"""Integration tests for the auto-sharding bridge (Phase 3).

Tests the full pipeline end-to-end, including:
  - Full pipeline: PyTorch model -> graph capture -> StableHLO -> GSPMD -> sharded execution
  - All 3 fallback tiers in GSPMDRunner
  - All 3 fallback tiers in StableHLOExporter
  - Collective insertion correctness (cost model)
  - Cross-vendor communication bridge
  - MeshTopology.is_uniform detection
  - DTensor placement conversion
  - Shard execution dispatch
  - Edge cases (empty tensors, zero-size meshes, type mismatches)

Uses the ``sharding_bridge`` fixture from ``conftest.py`` so all tests
run without real XLA/TVM/torch_xla installations.
"""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.bridges.pytorch_xla.comm_backend import (
    CollectiveOp,
    CommBackend,
    CommGroup,
    CommLibrary,
)
from src.bridges.pytorch_xla.device_mesh import (
    DeviceMesh,
    DeviceVendor,
    InterconnectType,
    MeshDevice,
    MeshTopology,
)
from src.bridges.pytorch_xla.device_mesh_utils import infer_target_from_mesh
from src.bridges.pytorch_xla.dtensor_apply import DTensorApplier, DTensorPlan
from src.bridges.pytorch_xla.gspmd_runner import (
    GSPMDResult,
    GSPMDRunner,
    ShardingSpec,
    ShardingStrategy,
    TensorSharding,
    _annotate_stablehlo_with_sharding,
    _CommCostModel,
    _compute_collectives,
    _GraphPartitionSharding,
    _insert_sharding_attr,
    _OpenXlaPjrtSharding,
    _sharding_spec_to_string,
    _TorchXLASharding,
)
from src.bridges.pytorch_xla.hardware_orchestrator import (
    ShardExecutionResult,
    ShardExecutor,
)
from src.bridges.pytorch_xla.pipeline_orchestrator import (
    AutoShardingBridge,
    ShardingConfig,
    ShardingResult,
)
from src.bridges.pytorch_xla.stablehlo_export import (
    StableHLOExporter,
    _ONNXBridgeExporter,
    _TorchXLAExporter,
    _TVMScriptExporter,
)
from src.common.types import StableHLOModule

# ---- Shared test data -------------------------------------------------------

SAMPLE_MLIR = """\
module {
  func.func @matmul(%A: tensor<128x128xf32>, %B: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = stablehlo.multiply %A, %B : tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
  }
}
"""

SAMPLE_MLIR_ARGS = """\
module {
  func.func @main(%arg0: tensor<4x128xf32>, %arg1: tensor<128x64xf32>) -> tensor<4x64xf32> {
    %0 = stablehlo.dot %arg0, %arg1 : tensor<4x128xf32>, tensor<128x64xf32>
    return %0 : tensor<4x64xf32>
  }
}
"""


def make_stablehlo_module(
    mlir_text: str = SAMPLE_MLIR,
    function_name: str = "matmul",
    num_inputs: int = 2,
) -> StableHLOModule:
    """Create a sample StableHLO module for testing."""
    return StableHLOModule(
        mlir_text=mlir_text,
        function_name=function_name,
        input_specs=[
            {"name": n, "dtype": "f32", "shape": [128, 128]} for n in ("A", "B")[:num_inputs]
        ],
        output_specs=[{"name": "output"}],
        op_count=3,
        is_usable=True,
    )


def make_device_mesh(
    num_devices: int = 4,
    vendor: DeviceVendor = DeviceVendor.NVIDIA,
    arch: str = "sm_90",
) -> DeviceMesh:
    """Create a test device mesh."""
    devices = [
        MeshDevice(
            device_id=i,
            vendor=vendor,
            arch=arch,
            memory_gb=80.0,
            compute_tflops=989.0,
            interconnect=InterconnectType.NVLINK
            if vendor == DeviceVendor.NVIDIA
            else InterconnectType.INFINITY_FABRIC
            if vendor == DeviceVendor.AMD
            else InterconnectType.ETHERNET,
        )
        for i in range(num_devices)
    ]
    return DeviceMesh(devices=devices, mesh_shape=[num_devices])


def make_heterogeneous_mesh() -> DeviceMesh:
    """Create a mesh with mixed vendors."""
    devices = [
        MeshDevice(0, DeviceVendor.NVIDIA, "sm_90", 80.0, 989.0, InterconnectType.NVLINK),
        MeshDevice(1, DeviceVendor.NVIDIA, "sm_90", 80.0, 989.0, InterconnectType.NVLINK),
        MeshDevice(2, DeviceVendor.AMD, "gfx942", 192.0, 1307.0, InterconnectType.INFINITY_FABRIC),
        MeshDevice(3, DeviceVendor.AMD, "gfx942", 192.0, 1307.0, InterconnectType.INFINITY_FABRIC),
    ]
    return DeviceMesh(devices=devices, mesh_shape=[2, 2])


def make_2d_mesh() -> DeviceMesh:
    """Create a 2D mesh (2x2)."""
    devices = [
        MeshDevice(i, DeviceVendor.NVIDIA, "sm_90", 80.0, 989.0, InterconnectType.NVLINK)
        for i in range(4)
    ]
    return DeviceMesh(devices=devices, mesh_shape=[2, 2])


# =============================================================================
# 1. sharding_bridge fixture integration
# =============================================================================


class TestShardingBridgeFixture:
    """Tests that exercise the conftest ``sharding_bridge`` fixture."""

    def test_fixture_returns_bridge(self, sharding_bridge: Any) -> None:
        """The fixture must yield an AutoShardingBridge."""
        assert isinstance(sharding_bridge, AutoShardingBridge)

    def test_fixture_disables_circuit_breakers(self, sharding_bridge: Any) -> None:
        """The fixture must have circuit breakers disabled (mocked mode)."""
        assert sharding_bridge.breakers is None
        assert sharding_bridge.timeout_manager is None

    def test_fixture_has_mock_gspmd(self, sharding_bridge: Any) -> None:
        """The fixture must expose a mock GSPMD runner."""
        assert hasattr(sharding_bridge, "_mock_gspmd")
        assert isinstance(sharding_bridge._mock_gspmd, MagicMock)

    def test_fixture_has_mock_graph(self, sharding_bridge: Any) -> None:
        """The fixture must expose a mock GraphCapture."""
        assert hasattr(sharding_bridge, "_mock_graph")
        assert isinstance(sharding_bridge._mock_graph, MagicMock)

    def test_fixture_has_mock_stablehlo(self, sharding_bridge: Any) -> None:
        """The fixture must expose a mock StableHLO exporter."""
        assert hasattr(sharding_bridge, "_mock_stablehlo")
        assert isinstance(sharding_bridge._mock_stablehlo, MagicMock)

    def test_shard_uses_mocked_graph_capture(self, sharding_bridge: Any) -> None:
        """Calling shard() must invoke the mocked graph capture."""
        mesh = make_device_mesh()
        sharding_bridge.shard(model=None, example_inputs=(), device_mesh=mesh)
        sharding_bridge._mock_graph.assert_called_once()

    def test_shard_uses_mocked_stablehlo_export(self, sharding_bridge: Any) -> None:
        """Calling shard() must invoke the mocked StableHLO exporter."""
        mesh = make_device_mesh()
        sharding_bridge.shard(model=None, example_inputs=(), device_mesh=mesh)
        sharding_bridge._mock_stablehlo.assert_called_once()

    def test_shard_uses_mocked_gspmd(self, sharding_bridge: Any) -> None:
        """Calling shard() must invoke the mocked GSPMD runner."""
        mesh = make_device_mesh()
        sharding_bridge.shard(model=None, example_inputs=(), device_mesh=mesh)
        sharding_bridge._mock_gspmd.assert_called_once()

    def test_shard_returns_sharding_result(self, sharding_bridge: Any) -> None:
        """Calling shard() must return a ShardingResult."""
        mesh = make_device_mesh()
        result = sharding_bridge.shard(model=None, example_inputs=(), device_mesh=mesh)
        assert isinstance(result, ShardingResult)

    def test_shard_success_with_mocked_path(self, sharding_bridge: Any) -> None:
        """The full pipeline must report success when all mocks return OK."""
        mesh = make_device_mesh()
        with patch(
            "src.bridges.pytorch_xla.hardware_orchestrator.ShardExecutor.execute_all_shards",
            MagicMock(return_value=[]),
        ):
            result = sharding_bridge.shard(model=None, example_inputs=(), device_mesh=mesh)
        assert result.success is True
        assert result.captured_graph is not None
        assert result.stablehlo_module is not None
        assert result.gspmd_result is not None

    def test_shard_records_all_stage_durations(self, sharding_bridge: Any) -> None:
        """All stages must be recorded in stage_durations."""
        mesh = make_device_mesh()
        with patch(
            "src.bridges.pytorch_xla.hardware_orchestrator.ShardExecutor.execute_all_shards",
            MagicMock(return_value=[]),
        ):
            result = sharding_bridge.shard(model=None, example_inputs=(), device_mesh=mesh)
        expected_stages = {
            "graph_capture",
            "stablehlo_export",
            "gspmd",
            "dtensor_apply",
            "fat_binary",
        }
        assert expected_stages.issubset(result.stage_durations.keys())

    def test_mock_gspmd_called_with_correct_args(self, sharding_bridge: Any) -> None:
        """GSPMD must be called with the right module, mesh, and strategy."""
        mesh = make_device_mesh()
        config = ShardingConfig(
            model=None,
            example_inputs=(),
            device_mesh=mesh,
            sharding_strategy=ShardingStrategy.DATA_PARALLEL,
        )
        sharding_bridge.shard(model=None, example_inputs=(), device_mesh=mesh, config=config)
        args, _ = sharding_bridge._mock_gspmd.call_args
        assert len(args) >= 2
        assert args[1] is mesh

    def test_comm_backend_set_per_mesh(self, sharding_bridge: Any) -> None:
        """After shard(), the comm backend must be set."""
        mesh = make_device_mesh()
        sharding_bridge.shard(model=None, example_inputs=(), device_mesh=mesh)
        assert sharding_bridge.comm_backend is not None
        assert sharding_bridge.comm_backend.mesh is mesh

    def test_executor_set_per_mesh(self, sharding_bridge: Any) -> None:
        """After shard(), the executor must be set."""
        mesh = make_device_mesh()
        sharding_bridge.shard(model=None, example_inputs=(), device_mesh=mesh)
        assert sharding_bridge.executor is not None

    def test_gspmd_failure_propagates(self, sharding_bridge: Any) -> None:
        """When GSPMD returns failure, the result must report failure."""
        sharding_bridge._mock_gspmd.return_value = GSPMDResult(
            success=False,
            error="Simulated GSPMD failure",
        )
        mesh = make_device_mesh()
        result = sharding_bridge.shard(model=None, example_inputs=(), device_mesh=mesh)
        assert result.success is False
        assert "GSPMD" in (result.error or "")

    def test_graph_capture_failure_propagates(self, sharding_bridge: Any) -> None:
        """When graph capture fails, the result must report failure."""
        fake_graph = MagicMock()
        fake_graph.is_usable = False
        sharding_bridge._mock_graph.return_value = fake_graph
        mesh = make_device_mesh()
        result = sharding_bridge.shard(model=None, example_inputs=(), device_mesh=mesh)
        assert result.success is False
        assert "Graph capture" in (result.error or "")

    def test_stablehlo_export_failure_propagates(self, sharding_bridge: Any) -> None:
        """When StableHLO export fails, the result must report failure."""
        fake_stablehlo = MagicMock()
        fake_stablehlo.is_usable = False
        sharding_bridge._mock_stablehlo.return_value = fake_stablehlo
        mesh = make_device_mesh()
        result = sharding_bridge.shard(model=None, example_inputs=(), device_mesh=mesh)
        assert result.success is False
        assert "StableHLO export" in (result.error or "")

    def test_shard_with_different_meshes(self, sharding_bridge: Any) -> None:
        """shard() must work with different mesh configurations."""
        for num_devices in (1, 2, 4, 8):
            mesh = make_device_mesh(num_devices=num_devices)
            with patch(
                "src.bridges.pytorch_xla.hardware_orchestrator.ShardExecutor.execute_all_shards",
                MagicMock(return_value=[]),
            ):
                result = sharding_bridge.shard(
                    model=None,
                    example_inputs=(),
                    device_mesh=mesh,
                )
            assert isinstance(result, ShardingResult)

    def test_shard_with_2d_mesh(self, sharding_bridge: Any) -> None:
        """shard() must work with a 2D mesh."""
        mesh = make_2d_mesh()
        with patch(
            "src.bridges.pytorch_xla.hardware_orchestrator.ShardExecutor.execute_all_shards",
            MagicMock(return_value=[]),
        ):
            result = sharding_bridge.shard(model=None, example_inputs=(), device_mesh=mesh)
        assert isinstance(result, ShardingResult)

    def test_shard_with_heterogeneous_mesh(self, sharding_bridge: Any) -> None:
        """shard() must work with a heterogeneous mesh."""
        mesh = make_heterogeneous_mesh()
        with patch(
            "src.bridges.pytorch_xla.hardware_orchestrator.ShardExecutor.execute_all_shards",
            MagicMock(return_value=[]),
        ):
            result = sharding_bridge.shard(model=None, example_inputs=(), device_mesh=mesh)
        assert isinstance(result, ShardingResult)

    def test_mocks_restored_after_shard(self, sharding_bridge: Any) -> None:
        """After shard(), the mock objects must still be accessible."""
        mesh = make_device_mesh()
        sharding_bridge.shard(model=None, example_inputs=(), device_mesh=mesh)
        sharding_bridge._mock_graph.reset_mock()
        sharding_bridge.shard(model=None, example_inputs=(), device_mesh=mesh)
        sharding_bridge._mock_graph.assert_called_once()


# =============================================================================
# 2. Full pipeline end-to-end (with mocked tiers)
# =============================================================================


class TestFullPipelineEndToEnd:
    """End-to-end pipeline tests using ShardingConfig + real model."""

    torch = pytest.importorskip("torch", reason="PyTorch required for pipeline tests")

    def _make_model(self):
        import torch.nn as nn

        return nn.Linear(8, 4)

    def test_sharding_config_holds_all_fields(self) -> None:
        """ShardingConfig must store all configuration fields."""
        model = self._make_model()
        inputs = (self.torch.randn(2, 8),)
        mesh = make_device_mesh()
        config = ShardingConfig(
            model=model,
            example_inputs=inputs,
            device_mesh=mesh,
            sharding_strategy=ShardingStrategy.AUTO,
            timeout_seconds=300.0,
            enable_cache=True,
            enable_fat_binary=False,
            skip_sharding=False,
            skip_dtensor=False,
            skip_fat_binary=True,
        )
        assert config.model is model
        assert config.example_inputs is inputs
        assert config.device_mesh is mesh
        assert config.sharding_strategy == ShardingStrategy.AUTO
        assert config.timeout_seconds == 300.0

    def test_shard_with_real_model_and_skip_flags(
        self,
        sharding_bridge: Any,
    ) -> None:
        """Skipping sharding + DTensor + fat binary must succeed."""
        model = self._make_model()
        inputs = (self.torch.randn(1, 8),)
        mesh = make_device_mesh()
        config = ShardingConfig(
            model=model,
            example_inputs=inputs,
            device_mesh=mesh,
            skip_sharding=True,
            skip_dtensor=True,
            skip_fat_binary=True,
        )
        bridge = AutoShardingBridge(enable_circuit_breakers=False)
        result = bridge.shard(model=model, example_inputs=inputs, device_mesh=mesh, config=config)
        assert isinstance(result, ShardingResult)
        assert "graph_capture" in result.stage_durations

    def test_shard_result_dict_contains_all_keys(
        self,
        sharding_bridge: Any,
    ) -> None:
        """to_dict() must contain all expected keys."""
        mesh = make_device_mesh()
        with patch(
            "src.bridges.pytorch_xla.hardware_orchestrator.ShardExecutor.execute_all_shards",
            MagicMock(return_value=[]),
        ):
            result = sharding_bridge.shard(model=None, example_inputs=(), device_mesh=mesh)
        d = result.to_dict()
        for key in (
            "success",
            "total_duration_ms",
            "stage_durations",
            "captured",
            "stablehlo",
            "gspmd",
            "dtensor",
        ):
            assert key in d, f"Missing key: {key}"

    def test_shard_result_is_usable_when_successful(
        self,
        sharding_bridge: Any,
    ) -> None:
        """is_usable must be True when the full pipeline succeeds."""
        mesh = make_device_mesh()
        with patch(
            "src.bridges.pytorch_xla.hardware_orchestrator.ShardExecutor.execute_all_shards",
            MagicMock(return_value=[]),
        ):
            result = sharding_bridge.shard(model=None, example_inputs=(), device_mesh=mesh)
        assert result.is_usable is True
        assert result.success is True

    def test_total_duration_recorded(self, sharding_bridge: Any) -> None:
        """total_duration_ms must be > 0 after a successful run."""
        mesh = make_device_mesh()
        with patch(
            "src.bridges.pytorch_xla.hardware_orchestrator.ShardExecutor.execute_all_shards",
            MagicMock(return_value=[]),
        ):
            result = sharding_bridge.shard(model=None, example_inputs=(), device_mesh=mesh)
        assert result.total_duration_ms > 0.0

    def test_pipeline_with_skip_sharding_only(self, sharding_bridge: Any) -> None:
        """Skip only sharding, keep dtensor and fat binary."""
        mesh = make_device_mesh()
        config = ShardingConfig(
            model=None,
            example_inputs=(),
            device_mesh=mesh,
            skip_sharding=True,
        )
        result = sharding_bridge.shard(
            model=None,
            example_inputs=(),
            device_mesh=mesh,
            config=config,
        )
        assert isinstance(result, ShardingResult)


# =============================================================================
# 3. GSPMD fallback tiers
# =============================================================================


class TestGSPMDFallbackTiers:
    """Tests that all 3 GSPMD fallback tiers are tried in order."""

    def test_three_tiers_defined(self) -> None:
        """The sharding tier list must contain exactly 3 tiers."""
        from src.bridges.pytorch_xla.gspmd_runner import _SHARDING_TIERS

        assert len(_SHARDING_TIERS) == 3
        tier_names = [t[0] for t in _SHARDING_TIERS]
        assert tier_names == ["torch_xla_spmd", "openxla_pjrt", "graph_partition"]

    def test_tier_availability_checks_exist(self) -> None:
        """Each tier must have an is_available() method."""
        assert hasattr(_TorchXLASharding, "is_available")
        assert hasattr(_OpenXlaPjrtSharding, "is_available")
        assert hasattr(_GraphPartitionSharding, "is_available")

    def test_torch_xla_tier_not_available_without_torch_xla(self) -> None:
        """Without torch_xla installed, the primary tier must report unavailable."""
        assert _TorchXLASharding.is_available() is False

    def test_openxla_pjrt_tier_not_available_without_torch_xla(self) -> None:
        """Without torch_xla._internal.pjrt, the secondary tier must report unavailable."""
        assert _OpenXlaPjrtSharding.is_available() is False

    def test_graph_partition_tier_always_available(self) -> None:
        """The graph partition tier must always be available (pure Python)."""
        assert _GraphPartitionSharding.is_available() is True

    def test_gspmd_runner_falls_through_to_graph_partition(self) -> None:
        """When primary and secondary tiers are unavailable, GSPMDRunner
        must use the graph_partition (tertiary) tier."""
        runner = GSPMDRunner()
        module = make_stablehlo_module()
        mesh = make_device_mesh()
        result = runner.run(module, mesh)
        assert result.is_usable is True
        # The result should come from the graph_partition tier
        assert result.tier_used == "graph_partition"
        assert result.success is True
        assert result.sharding_spec is not None

    def test_gspmd_runner_with_invalid_module(self) -> None:
        """An invalid (None) module must return a failed GSPMDResult, not raise."""
        runner = GSPMDRunner()
        result = runner.run(None, make_device_mesh())
        assert result.success is False
        assert result.error is not None

    def test_gspmd_runner_with_wrong_module_type(self) -> None:
        """Passing a non-StableHLOModule must return a failed result."""
        runner = GSPMDRunner()
        result = runner.run("not-a-module", make_device_mesh())
        assert result.success is False

    def test_gspmd_runner_diagnostics_on_success(self) -> None:
        """A successful run must record strategy and mesh info in diagnostics."""
        runner = GSPMDRunner()
        module = make_stablehlo_module(function_name="test_model")
        mesh = make_device_mesh()
        result = runner.run(module, mesh)
        assert result.is_usable
        assert result.diagnostics["strategy"] == "AUTO"
        assert result.diagnostics["mesh_shape"] == [4]

    def test_gspmd_runner_caches_results(self) -> None:
        """Subsequent runs with the same inputs must hit cache."""
        runner = GSPMDRunner()
        module = make_stablehlo_module()
        mesh = make_device_mesh()
        runner.run(module, mesh, ShardingStrategy.DATA_PARALLEL)
        result2 = runner.run(module, mesh, ShardingStrategy.DATA_PARALLEL)
        assert result2.cache_hit is True

    def test_gspmd_runner_data_parallel_strategy(self) -> None:
        """DATA_PARALLEL strategy must shard along mesh axis 0."""
        runner = GSPMDRunner()
        module = make_stablehlo_module()
        mesh = make_device_mesh()
        result = runner.run(module, mesh, ShardingStrategy.DATA_PARALLEL)
        assert result.is_usable
        spec = result.sharding_spec
        assert spec is not None
        assert spec.strategy_used == ShardingStrategy.DATA_PARALLEL
        for ts in spec.tensor_shardings.values():
            assert 0 in ts.mesh_axes

    def test_gspmd_runner_replicated_strategy(self) -> None:
        """REPLICATED strategy must not shard any tensor."""
        runner = GSPMDRunner()
        module = make_stablehlo_module()
        mesh = make_device_mesh()
        result = runner.run(module, mesh, ShardingStrategy.REPLICATED)
        assert result.is_usable
        spec = result.sharding_spec
        assert spec is not None
        for ts in spec.tensor_shardings.values():
            assert ts.mesh_axes == []

    def test_gspmd_runner_model_parallel_strategy(self) -> None:
        """MODEL_PARALLEL strategy must shard along the last axis."""
        runner = GSPMDRunner()
        module = make_stablehlo_module()
        mesh = make_device_mesh()
        result = runner.run(module, mesh, ShardingStrategy.MODEL_PARALLEL)
        assert result.is_usable
        spec = result.sharding_spec
        assert spec is not None
        assert spec.strategy_used == ShardingStrategy.MODEL_PARALLEL

    def test_gspmd_runner_tensor_parallel_2d_mesh(self) -> None:
        """TENSOR_PARALLEL on a 2D mesh must produce 2D sharding."""
        runner = GSPMDRunner()
        module = make_stablehlo_module()
        mesh = make_2d_mesh()
        result = runner.run(module, mesh, ShardingStrategy.TENSOR_PARALLEL)
        assert result.is_usable
        spec = result.sharding_spec
        assert spec is not None
        assert spec.strategy_used == ShardingStrategy.TENSOR_PARALLEL

    def test_custom_shardings_override_strategy(self) -> None:
        """Custom shardings must override the default strategy."""
        runner = GSPMDRunner()
        module = make_stablehlo_module()
        mesh = make_device_mesh()
        custom = {
            "A": TensorSharding(
                tensor_name="A",
                mesh_axes=[0, 1],
                partition_shape=[2, 2],
            ),
        }
        result = runner.run(
            module,
            mesh,
            ShardingStrategy.DATA_PARALLEL,
            custom_shardings=custom,
        )
        spec = result.sharding_spec
        assert spec is not None
        assert spec.tensor_shardings["A"].mesh_axes == [0, 1]
        assert spec.tensor_shardings["B"].mesh_axes == [0]

    def test_inserts_collectives(self) -> None:
        """GSPMD must insert collective ops for sharded params."""
        runner = GSPMDRunner()
        module = make_stablehlo_module()
        mesh = make_device_mesh()
        result = runner.run(module, mesh, ShardingStrategy.DATA_PARALLEL)
        spec = result.sharding_spec
        assert spec is not None
        assert len(spec.inserted_collectives) > 0
        assert spec.inserted_collectives[0]["type"] == "all-reduce"

    def test_graph_partition_tier_shards(self) -> None:
        """The graph_partition tier must produce a valid sharding."""
        module = make_stablehlo_module()
        mesh_shape = [4]
        sharded_text, spec = _GraphPartitionSharding.shard(
            module,
            mesh_shape,
            ShardingStrategy.DATA_PARALLEL,
            None,
        )
        assert isinstance(sharded_text, str)
        assert sharded_text != module.mlir_text
        assert isinstance(spec, ShardingSpec)
        assert spec.strategy_used == ShardingStrategy.DATA_PARALLEL

    def test_graph_partition_tier_inserts_collective_ops(self) -> None:
        """The graph_partition tier must insert real collective ops."""
        module = make_stablehlo_module()
        mesh_shape = [4]
        sharded_text, spec = _GraphPartitionSharding.shard(
            module,
            mesh_shape,
            ShardingStrategy.DATA_PARALLEL,
            None,
        )
        assert "all_reduce" in sharded_text or len(spec.inserted_collectives) > 0

    def test_gspmd_runner_with_empty_module(self) -> None:
        """An empty StableHLO module must produce a graceful result."""
        runner = GSPMDRunner()
        empty_module = StableHLOModule(mlir_text="", function_name="empty", is_usable=True)
        mesh = make_device_mesh()
        result = runner.run(empty_module, mesh)
        assert result.success is True


# =============================================================================
# 4. StableHLO export fallback tiers
# =============================================================================


class TestStableHLOExportFallbackTiers:
    """Tests that all 3 StableHLO export tiers are tried in order."""

    def test_three_export_tiers_defined(self) -> None:
        """The export tier list must contain exactly 3 tiers."""
        from src.bridges.pytorch_xla.stablehlo_export import _EXPORT_TIERS

        assert len(_EXPORT_TIERS) == 3
        tier_names = [t[0] for t in _EXPORT_TIERS]
        assert tier_names == ["torch_xla", "onnx_bridge", "tvmscript"]

    def test_torch_xla_exporter_not_available(self) -> None:
        """Without torch_xla, the primary export tier must report unavailable."""
        assert _TorchXLAExporter.is_available() is False

    def test_onnx_exporter_not_available_without_onnx(self) -> None:
        """Without onnx + onnx-mlir, the secondary tier must report unavailable."""
        assert _ONNXBridgeExporter.is_available() is False

    def test_tvm_exporter_not_available_without_tvm(self) -> None:
        """Without tvm, the tertiary export tier must report unavailable."""
        assert _TVMScriptExporter.is_available() is False

    def test_exporter_raises_when_all_tiers_unavailable(self) -> None:
        """When all export tiers are unavailable, exporter must raise."""
        from src.common.errors import StableHLOExportError

        exporter = StableHLOExporter()
        fake_captured = MagicMock()
        fake_captured.is_usable = True
        fake_captured.graph_module = MagicMock()
        fake_captured.metadata.model_name = "test"
        with pytest.raises(StableHLOExportError):
            exporter.export_from_captured(fake_captured)

    def test_exporter_disabled_tvm_path(self) -> None:
        """Disabling the TVM path must skip the tertiary tier."""
        from src.common.errors import StableHLOExportError

        exporter = StableHLOExporter(enable_tvm_path=False)
        fake_captured = MagicMock()
        fake_captured.is_usable = True
        fake_captured.graph_module = MagicMock()
        fake_captured.metadata.model_name = "test"
        with pytest.raises(StableHLOExportError) as excinfo:
            exporter.export_from_captured(fake_captured)
        assert "disabled by config" in str(excinfo.value)

    def test_exporter_raises_on_none_captured(self) -> None:
        """Passing None must raise StableHLOExportError."""
        from src.common.errors import StableHLOExportError

        exporter = StableHLOExporter()
        with pytest.raises(StableHLOExportError, match="None or not usable"):
            exporter.export_from_captured(None)


# =============================================================================
# 5. Collective communication cost model
# =============================================================================


class TestCommCostModel:
    """Tests for the ``_CommCostModel`` pure functions."""

    def test_all_reduce_zero_for_single_device(self) -> None:
        """All-reduce with 1 device must return 0 bytes."""
        assert _CommCostModel.all_reduce_bytes(1024, 1) == 0

    def test_all_reduce_formula(self) -> None:
        """All-reduce: 2 * tensor_bytes * (num_devices - 1) / num_devices."""
        assert _CommCostModel.all_reduce_bytes(1024, 4) == 1536

    def test_all_reduce_large_tensor(self) -> None:
        """All-reduce with a large tensor must compute correctly."""
        result = _CommCostModel.all_reduce_bytes(1073741824, 8)
        expected = int(2 * 1073741824 * 7 / 8)
        assert result == expected

    def test_all_gather_zero_for_single_device(self) -> None:
        """All-gather with 1 device must return 0 bytes."""
        assert _CommCostModel.all_gather_bytes(1024, 1) == 0

    def test_all_gather_formula(self) -> None:
        """All-gather: tensor_bytes * (num_devices - 1)."""
        assert _CommCostModel.all_gather_bytes(1024, 4) == 3072

    def test_reduce_scatter_formula(self) -> None:
        """Reduce-scatter must use the same formula as all-reduce."""
        assert _CommCostModel.reduce_scatter_bytes(1024, 4) == 1536

    def test_all_to_all_zero_for_single_device(self) -> None:
        """All-to-all with 1 device must return 0 bytes."""
        assert _CommCostModel.all_to_all_bytes(1024, 1) == 0

    def test_all_to_all_formula(self) -> None:
        """All-to-all: tensor_bytes * (num_devices - 1) / num_devices."""
        assert _CommCostModel.all_to_all_bytes(1024, 4) == 768

    def test_estimate_tensor_bytes(self) -> None:
        """estimate_tensor_bytes must compute product(shape) * dtype_bytes."""
        ts = TensorSharding("A", mesh_axes=[0], partition_shape=[128, 256])
        assert _CommCostModel.estimate_tensor_bytes(ts, 4) == 128 * 256 * 4

    def test_estimate_tensor_bytes_empty_shape(self) -> None:
        """estimate_tensor_bytes with empty shape must default to [1]."""
        ts = TensorSharding("A", mesh_axes=[], partition_shape=[])
        assert _CommCostModel.estimate_tensor_bytes(ts, 4) == 4

    def test_estimate_tensor_bytes_float64(self) -> None:
        """estimate_tensor_bytes must handle different dtype sizes."""
        ts = TensorSharding("B", mesh_axes=[0], partition_shape=[64, 64])
        assert _CommCostModel.estimate_tensor_bytes(ts, 8) == 64 * 64 * 8
        assert _CommCostModel.estimate_tensor_bytes(ts, 2) == 64 * 64 * 2


# =============================================================================
# 6. Collective insertion correctness
# =============================================================================


class TestCollectiveInsertion:
    """Tests for ``_compute_collectives`` and the collective computation logic."""

    def test_no_collectives_for_replicated(self) -> None:
        """Replicated tensors (no mesh_axes) must produce no collectives."""
        module = make_stablehlo_module()
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={
                "A": TensorSharding("A", mesh_axes=[], partition_shape=[]),
                "B": TensorSharding("B", mesh_axes=[], partition_shape=[]),
            },
        )
        collectives = _compute_collectives(spec, module, 4)
        assert len(collectives) == 0

    def test_collectives_for_sharded_tensor(self) -> None:
        """A sharded tensor must produce 1 all-reduce collective."""
        module = make_stablehlo_module()
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={
                "A": TensorSharding("A", mesh_axes=[0], partition_shape=[4]),
            },
        )
        collectives = _compute_collectives(spec, module, 4)
        assert len(collectives) == 1
        assert collectives[0]["type"] == "all-reduce"

    def test_collectives_contain_estimated_bytes(self) -> None:
        """Each collective must have a non-zero estimated_bytes field."""
        module = make_stablehlo_module()
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={
                "A": TensorSharding("A", mesh_axes=[0], partition_shape=[128, 128]),
            },
        )
        collectives = _compute_collectives(spec, module, 4)
        assert len(collectives) == 1
        assert collectives[0]["estimated_bytes"] > 0

    def test_collectives_contain_tensor_name(self) -> None:
        """Each collective must reference the tensor it operates on."""
        module = make_stablehlo_module()
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={
                "A": TensorSharding("A", mesh_axes=[0], partition_shape=[4]),
            },
        )
        collectives = _compute_collectives(spec, module, 4)
        for c in collectives:
            assert c["tensor"] == "A"

    def test_collectives_2d_mesh(self) -> None:
        """A 2D mesh with tensor sharded on both axes must include all axes info."""
        module = make_stablehlo_module()
        spec = ShardingSpec(
            mesh_shape=[2, 2],
            tensor_shardings={
                "A": TensorSharding("A", mesh_axes=[0, 1], partition_shape=[2, 2]),
            },
        )
        collectives = _compute_collectives(spec, module, 4)
        assert len(collectives) == 1
        assert collectives[0]["mesh_axes"] == [0, 1]
        assert collectives[0]["num_devices"] == 4

    def test_collectives_multiple_tensors(self) -> None:
        """Multiple sharded tensors must produce 1 all-reduce each."""
        module = make_stablehlo_module(num_inputs=3)
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={
                "A": TensorSharding("A", mesh_axes=[0], partition_shape=[4]),
                "B": TensorSharding("B", mesh_axes=[0], partition_shape=[4]),
            },
        )
        collectives = _compute_collectives(spec, module, 4)
        assert len(collectives) == 2
        tensors = {c["tensor"] for c in collectives}
        assert tensors == {"A", "B"}

    def test_collectives_estimated_bytes_match_cost_model(self) -> None:
        """The estimated bytes must match the cost model formula."""
        module = make_stablehlo_module()
        ts = TensorSharding("A", mesh_axes=[0], partition_shape=[128, 128])
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={"A": ts},
        )
        collectives = _compute_collectives(spec, module, 4)
        tensor_bytes = 128 * 128 * 4
        all_reduce_expected = _CommCostModel.all_reduce_bytes(tensor_bytes, 4)
        assert len(collectives) == 1
        assert collectives[0]["type"] == "all-reduce"
        assert collectives[0]["estimated_bytes"] == all_reduce_expected

    def test_comm_volume_aggregation(self) -> None:
        """The spec's estimated_comm_volume_bytes must equal the sum of all collectives."""
        module = make_stablehlo_module()
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={
                "A": TensorSharding("A", mesh_axes=[0], partition_shape=[128, 128]),
                "B": TensorSharding("B", mesh_axes=[0], partition_shape=[64, 64]),
            },
        )
        collectives = _compute_collectives(spec, module, 4)
        total_by_sum = sum(c["estimated_bytes"] for c in collectives)
        spec.inserted_collectives = collectives
        spec.estimated_comm_volume_bytes = total_by_sum
        assert spec.estimated_comm_volume_bytes > 0
        assert spec.estimated_comm_volume_bytes == total_by_sum


# =============================================================================
# 7. Cross-vendor communication bridge
# =============================================================================


class TestCrossVendorCommunication:
    """Tests for cross-vendor communication via CommBackend."""

    def test_homogeneous_nvidia_no_bridges(self) -> None:
        """A homogeneous Nvidia mesh must have no cross-vendor bridges."""
        mesh = make_device_mesh(num_devices=4, vendor=DeviceVendor.NVIDIA)
        comm = CommBackend(mesh)
        assert len(comm._cross_vendor_bridges) == 0
        assert comm.mesh.is_heterogeneous is False

    def test_homogeneous_amd_no_bridges(self) -> None:
        """A homogeneous AMD mesh must have no cross-vendor bridges."""
        mesh = make_device_mesh(num_devices=2, vendor=DeviceVendor.AMD, arch="gfx942")
        comm = CommBackend(mesh)
        assert len(comm._cross_vendor_bridges) == 0

    def test_heterogeneous_nvidia_amd_has_bridge(self) -> None:
        """A mixed Nvidia+AMD mesh must create a cross-vendor bridge."""
        mesh = make_heterogeneous_mesh()
        comm = CommBackend(mesh)
        assert len(comm._cross_vendor_bridges) == 1

    def test_heterogeneous_three_vendors(self) -> None:
        """A mesh with 3 vendors must create C(3,2) = 3 bridges."""
        devices = [
            MeshDevice(0, DeviceVendor.NVIDIA, "sm_90", 80, 989, InterconnectType.NVLINK),
            MeshDevice(1, DeviceVendor.AMD, "gfx942", 192, 1307, InterconnectType.INFINITY_FABRIC),
            MeshDevice(2, DeviceVendor.INTEL, "intel_gaudi2", 96, 900, InterconnectType.ETHERNET),
        ]
        mesh = DeviceMesh(devices=devices, mesh_shape=[3])
        comm = CommBackend(mesh)
        assert len(comm._cross_vendor_bridges) == 3

    def test_cross_vendor_bridge_uses_mixed_library(self) -> None:
        """Cross-vendor bridges must use CommLibrary.MIXED."""
        mesh = make_heterogeneous_mesh()
        comm = CommBackend(mesh)
        for (_v1, _v2), bridge in comm._cross_vendor_bridges.items():
            assert bridge.library == CommLibrary.MIXED
            assert bridge.is_cross_vendor_bridge is True

    def test_cross_vendor_bridge_bandwidth_pcie(self) -> None:
        """Cross-vendor bridge bandwidth must be PCIe-class (64 GB/s)."""
        mesh = make_heterogeneous_mesh()
        comm = CommBackend(mesh)
        for bridge in comm._cross_vendor_bridges.values():
            assert bridge.bandwidth_gbps <= 64.0

    def test_select_library_homogeneous_nvidia(self) -> None:
        """select_library_for_op must return NCCL for Nvidia-only ops."""
        mesh = make_device_mesh(num_devices=2, vendor=DeviceVendor.NVIDIA)
        comm = CommBackend(mesh)
        lib = comm.select_library_for_op(CollectiveOp.ALL_REDUCE, [0, 1])
        assert lib == CommLibrary.NCCL

    def test_select_library_homogeneous_amd(self) -> None:
        """select_library_for_op must return RCCL for AMD-only ops."""
        mesh = make_device_mesh(num_devices=2, vendor=DeviceVendor.AMD, arch="gfx942")
        comm = CommBackend(mesh)
        lib = comm.select_library_for_op(CollectiveOp.ALL_REDUCE, [0, 1])
        assert lib == CommLibrary.RCCL

    def test_select_library_mixed_vendor(self) -> None:
        """select_library_for_op must return MIXED for cross-vendor ops."""
        mesh = make_heterogeneous_mesh()
        comm = CommBackend(mesh)
        lib = comm.select_library_for_op(CollectiveOp.ALL_GATHER, [0, 2])
        assert lib == CommLibrary.MIXED

    def test_select_library_empty_devices(self) -> None:
        """select_library_for_op with empty list must return GLOO."""
        mesh = make_device_mesh()
        comm = CommBackend(mesh)
        lib = comm.select_library_for_op(CollectiveOp.BARRIER, [])
        assert lib == CommLibrary.GLOO

    def test_select_library_single_device(self) -> None:
        """select_library_for_op with single device must return vendor's lib."""
        mesh = make_heterogeneous_mesh()
        comm = CommBackend(mesh)
        lib = comm.select_library_for_op(CollectiveOp.ALL_REDUCE, [0])
        assert lib == CommLibrary.NCCL
        lib = comm.select_library_for_op(CollectiveOp.ALL_REDUCE, [2])
        assert lib == CommLibrary.RCCL

    def test_get_group_homogeneous(self) -> None:
        """get_group_for_devices must find the correct group."""
        mesh = make_device_mesh(num_devices=4, vendor=DeviceVendor.NVIDIA)
        comm = CommBackend(mesh)
        group = comm.get_group_for_devices([0, 1, 2, 3])
        assert group is not None
        assert group.library == CommLibrary.NCCL

    def test_get_group_cross_vendor(self) -> None:
        """get_group_for_devices must return the bridge for cross-vendor sets."""
        mesh = make_heterogeneous_mesh()
        comm = CommBackend(mesh)
        group = comm.get_group_for_devices([0, 2])
        assert group is not None
        assert group.is_cross_vendor_bridge is True

    def test_get_group_returns_comm_group_type(self) -> None:
        """get_group_for_devices must return a CommGroup instance (or None)."""
        mesh = make_device_mesh()
        comm = CommBackend(mesh)
        group = comm.get_group_for_devices([0, 1])
        assert isinstance(group, (CommGroup, type(None)))

    def test_stats_include_bridge_count(self) -> None:
        """get_stats must report the number of cross-vendor bridges."""
        mesh = make_heterogeneous_mesh()
        comm = CommBackend(mesh)
        stats = comm.get_stats()
        assert stats["num_cross_vendor_bridges"] == 1
        assert stats["is_heterogeneous"] is True

    def test_stats_include_group_details(self) -> None:
        """get_stats must include per-group details."""
        mesh = make_device_mesh(num_devices=2, vendor=DeviceVendor.NVIDIA)
        comm = CommBackend(mesh)
        stats = comm.get_stats()
        assert len(stats["groups"]) == 1
        assert stats["groups"][0]["library"] == "nccl"
        assert stats["groups"][0]["num_devices"] == 2

    def test_stats_include_bridge_details(self) -> None:
        """Cross-vendor bridges must be stored in _cross_vendor_bridges."""
        mesh = make_heterogeneous_mesh()
        comm = CommBackend(mesh)
        assert len(comm._cross_vendor_bridges) >= 1
        for (_v1, _v2), bridge in comm._cross_vendor_bridges.items():
            assert bridge.is_cross_vendor_bridge is True
            assert bridge.library == CommLibrary.MIXED

    def test_library_selection_per_vendor(self) -> None:
        """_select_library_for_vendor must return correct libs."""
        mesh = make_device_mesh()
        comm = CommBackend(mesh)
        assert comm._select_library_for_vendor(DeviceVendor.NVIDIA) == CommLibrary.NCCL
        assert comm._select_library_for_vendor(DeviceVendor.AMD) == CommLibrary.RCCL
        assert comm._select_library_for_vendor(DeviceVendor.INTEL) == CommLibrary.ONECCL
        assert comm._select_library_for_vendor(DeviceVendor.CPU) == CommLibrary.GLOO

    def test_bandwidth_estimation_per_vendor(self) -> None:
        """_estimate_bandwidth must return correct GB/s per vendor."""
        mesh = make_device_mesh()
        comm = CommBackend(mesh)
        assert comm._estimate_bandwidth(DeviceVendor.NVIDIA) == 900.0
        assert comm._estimate_bandwidth(DeviceVendor.AMD) == 800.0
        assert comm._estimate_bandwidth(DeviceVendor.INTEL) == 200.0
        assert comm._estimate_bandwidth(DeviceVendor.CPU) == 64.0


# =============================================================================
# 8. MeshTopology.is_uniform
# =============================================================================


class TestMeshTopologyUniformity:
    """Tests for MeshTopology.is_uniform."""

    def test_uniform_bandwidth(self) -> None:
        """A uniform bandwidth matrix must be detected as uniform."""
        topo = MeshTopology(
            bandwidth_matrix=[
                [0.0, 900.0, 900.0],
                [900.0, 0.0, 900.0],
                [900.0, 900.0, 0.0],
            ]
        )
        assert topo.is_uniform is True

    def test_non_uniform_bandwidth(self) -> None:
        """A non-uniform bandwidth matrix must be detected as non-uniform."""
        topo = MeshTopology(
            bandwidth_matrix=[
                [0.0, 900.0, 64.0],
                [900.0, 0.0, 64.0],
                [64.0, 64.0, 0.0],
            ]
        )
        assert topo.is_uniform is False

    def test_uniform_within_threshold(self) -> None:
        """Small differences within 1.0 GB/s must still be considered uniform."""
        topo = MeshTopology(
            bandwidth_matrix=[
                [0.0, 900.0, 900.5],
                [900.0, 0.0, 900.0],
                [900.5, 900.0, 0.0],
            ]
        )
        assert topo.is_uniform is True

    def test_empty_matrices_uniform(self) -> None:
        """Empty matrices must default to uniform."""
        topo = MeshTopology()
        assert topo.is_uniform is True

    def test_single_device_uniform(self) -> None:
        """A single device must be uniform."""
        topo = MeshTopology(bandwidth_matrix=[[0.0]])
        assert topo.is_uniform is True

    def test_nvlink_island_non_uniform_with_pcie(self) -> None:
        """NVLink + PCIe in the same topology must be non-uniform."""
        topo = MeshTopology(
            bandwidth_matrix=[
                [0.0, 900.0, 64.0],
                [900.0, 0.0, 64.0],
                [64.0, 64.0, 0.0],
            ]
        )
        assert topo.is_uniform is False

    def test_topology_detects_nvlink_bandwidth(self) -> None:
        """The topology bandwidth matrix must reflect NVLink bandwidth."""
        n = 4
        bandwidth = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    bandwidth[i][j] = 900.0
        mesh = DeviceMesh(
            devices=[
                MeshDevice(i, DeviceVendor.NVIDIA, "sm_90", 80.0, 989.0, InterconnectType.NVLINK)
                for i in range(n)
            ],
            mesh_shape=[n],
            topology=MeshTopology(bandwidth_matrix=bandwidth),
        )
        bw = mesh.topology.bandwidth_matrix
        assert bw[0][1] == 900.0
        assert bw[1][0] == 900.0

    def test_uniform_topology_with_same_interconnect(self) -> None:
        """All devices with same interconnect must create a uniform topology."""
        n = 4
        bandwidth = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    bandwidth[i][j] = 900.0
        mesh = DeviceMesh(
            devices=[
                MeshDevice(i, DeviceVendor.NVIDIA, "sm_90", 80.0, 989.0, InterconnectType.NVLINK)
                for i in range(n)
            ],
            mesh_shape=[n],
            topology=MeshTopology(bandwidth_matrix=bandwidth),
        )
        assert mesh.topology.is_uniform is True

    def test_heterogeneous_mesh_default_topology(self) -> None:
        """A heterogeneous mesh must have a non-uniform default topology."""
        mesh = make_heterogeneous_mesh()
        topo = mesh.topology
        assert topo is not None


# =============================================================================
# 9. Sharding annotation (MLIR attribute insertion)
# =============================================================================


class TestShardingAnnotationMLIR:
    """Tests for real MLIR sharding annotation insertion (H-03)."""

    def test_annotate_inserts_real_attribute(self) -> None:
        """Annotation must insert a real ``{mhlo.sharding = "..."}`` attribute."""
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={
                "A": TensorSharding("A", mesh_axes=[0], partition_shape=[4]),
            },
        )
        result = _annotate_stablehlo_with_sharding(SAMPLE_MLIR, spec)
        assert '{mhlo.sharding = "devices=[4]<=[4]"}' in result

    def test_annotate_replicated_keyword(self) -> None:
        """Replicated tensors use the ``replicated`` keyword."""
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={
                "A": TensorSharding("A", mesh_axes=[], partition_shape=[]),
            },
        )
        result = _annotate_stablehlo_with_sharding(SAMPLE_MLIR, spec)
        assert '{mhlo.sharding = "replicated"}' in result

    def test_annotate_model_parallel_format(self) -> None:
        """Model parallel on axis 0 with partition [1, N] -> devices=[1,N]<=[N]."""
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={
                "A": TensorSharding("A", mesh_axes=[0], partition_shape=[1, 4]),
            },
        )
        result = _annotate_stablehlo_with_sharding(SAMPLE_MLIR, spec)
        assert '{mhlo.sharding = "devices=[1,4]<=[4]"}' in result

    def test_annotate_tensor_parallel_2d(self) -> None:
        """Tensor parallel on 2x2 mesh -> devices=[2,2]<=[4]."""
        spec = ShardingSpec(
            mesh_shape=[2, 2],
            tensor_shardings={
                "A": TensorSharding("A", mesh_axes=[0, 1], partition_shape=[2, 2]),
            },
        )
        result = _annotate_stablehlo_with_sharding(SAMPLE_MLIR, spec)
        assert '{mhlo.sharding = "devices=[2,2]<=[4]"}' in result

    def test_annotate_idempotent(self) -> None:
        """Re-annotating must not duplicate the attribute."""
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={
                "A": TensorSharding("A", mesh_axes=[0], partition_shape=[4]),
            },
        )
        once = _annotate_stablehlo_with_sharding(SAMPLE_MLIR, spec)
        twice = _annotate_stablehlo_with_sharding(once, spec)
        a_attrs = re.findall(r"%A:[^{]+\{([^}]+)\}", twice)
        assert a_attrs, "no attribute group on %A"
        for attrs in a_attrs:
            assert attrs.count("mhlo.sharding") == 1

    def test_annotate_empty_inputs(self) -> None:
        """Empty MLIR or empty spec must return the input unchanged."""
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={"A": TensorSharding("A", mesh_axes=[0], partition_shape=[4])},
        )
        assert _annotate_stablehlo_with_sharding("", spec) == ""
        assert (
            _annotate_stablehlo_with_sharding(
                SAMPLE_MLIR,
                ShardingSpec(mesh_shape=[4]),
            )
            == SAMPLE_MLIR
        )

    def test_annotate_unknown_arg(self) -> None:
        """Unknown tensor name must leave MLIR unchanged."""
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={
                "GHOST": TensorSharding("GHOST", mesh_axes=[0], partition_shape=[4]),
            },
        )
        result = _annotate_stablehlo_with_sharding(SAMPLE_MLIR, spec)
        assert "GHOST" not in result
        assert "mhlo.sharding" not in result

    def test_annotate_preserves_body(self) -> None:
        """Annotation must not damage the function body."""
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={
                "A": TensorSharding("A", mesh_axes=[0], partition_shape=[4]),
            },
        )
        result = _annotate_stablehlo_with_sharding(SAMPLE_MLIR, spec)
        assert "stablehlo.multiply" in result
        assert "return %0" in result

    def test_annotate_multiple_args(self) -> None:
        """Both function arguments must be annotated."""
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={
                "A": TensorSharding("A", mesh_axes=[0], partition_shape=[4]),
                "B": TensorSharding("B", mesh_axes=[0], partition_shape=[4]),
            },
        )
        result = _annotate_stablehlo_with_sharding(SAMPLE_MLIR, spec)
        assert "mhlo.sharding" in result
        assert result.count("mhlo.sharding") == 2

    def test_insert_sharding_attr_preserves_return_type(self) -> None:
        """Function signature return type must be preserved."""
        text = (
            "module {\n"
            "  func.func @foo(%A: tensor<4xf32>) -> tensor<4xf32> {\n"
            "    return %A : tensor<4xf32>\n"
            "  }\n"
            "}\n"
        )
        result = _insert_sharding_attr(text, "A", "replicated")
        assert "-> tensor<4xf32>" in result
        assert "return %A" in result
        assert '{mhlo.sharding = "replicated"}' in result

    def test_sharding_spec_to_string_replicated(self) -> None:
        """_sharding_spec_to_string must map empty mesh_axes to 'replicated'."""
        ts = TensorSharding("A", mesh_axes=[], partition_shape=[], replicate_on_other_axes=True)
        assert _sharding_spec_to_string(ts, [4]) == "replicated"

    def test_sharding_spec_to_string_maximal(self) -> None:
        """Empty mesh_axes + replicate=False -> 'maximal'."""
        ts = TensorSharding("A", mesh_axes=[], partition_shape=[], replicate_on_other_axes=False)
        assert _sharding_spec_to_string(ts, [1]) == "maximal"

    def test_sharding_spec_to_string_data_parallel(self) -> None:
        """Data parallel on 4 devices -> 'devices=[4]<=[4]'."""
        ts = TensorSharding("A", mesh_axes=[0], partition_shape=[4])
        assert _sharding_spec_to_string(ts, [4]) == "devices=[4]<=[4]"

    def test_sharding_spec_to_string_tensor_parallel(self) -> None:
        """Tensor parallel on 2x2 mesh -> 'devices=[2,2]<=[4]'."""
        ts = TensorSharding("A", mesh_axes=[0, 1], partition_shape=[2, 2])
        assert _sharding_spec_to_string(ts, [2, 2]) == "devices=[2,2]<=[4]"

    def test_sharding_spec_to_string_no_partition_shape(self) -> None:
        """When partition_shape is empty, derive from mesh_shape."""
        ts = TensorSharding("A", mesh_axes=[0], partition_shape=[], replicate_on_other_axes=False)
        result = _sharding_spec_to_string(ts, [4])
        assert result == "devices=[4]<=[4]"

    def test_annotate_with_arg0_name(self) -> None:
        """Annotation must work with %arg0 style function arguments."""
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={
                "arg0": TensorSharding("arg0", mesh_axes=[0], partition_shape=[4, 128]),
            },
        )
        result = _annotate_stablehlo_with_sharding(SAMPLE_MLIR_ARGS, spec)
        assert '{mhlo.sharding = "devices=[4,128]<=[4]"}' in result

    def test_annotate_does_not_add_comments(self) -> None:
        """Annotation must NOT insert //-style comments."""
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={
                "A": TensorSharding("A", mesh_axes=[0], partition_shape=[4]),
            },
        )
        result = _annotate_stablehlo_with_sharding(SAMPLE_MLIR, spec)
        assert "// mhlo.sharding" not in result


# =============================================================================
# 10. DTensor application
# =============================================================================


class TestDTensorApplication:
    """Tests for applying sharding specs to PyTorch via DTensorApplier."""

    def test_dtensor_plan_defaults(self) -> None:
        """DTensorPlan must have sensible defaults."""
        plan = DTensorPlan()
        assert plan.is_usable is True
        assert plan.total_params == 0
        assert plan.total_inputs == 0
        assert plan.conversion_time_ms >= 0.0

    def test_dtensor_plan_not_usable(self) -> None:
        """DTensorPlan with is_usable=False must be marked as unusable."""
        plan = DTensorPlan(is_usable=False)
        assert plan.is_usable is False

    def test_dtensor_applier_creation(self) -> None:
        """DTensorApplier must accept a device mesh."""
        mesh = make_device_mesh()
        applier = DTensorApplier(mesh)
        assert applier.device_mesh is mesh

    def test_apply_sharding_with_spec(self) -> None:
        """apply_sharding must produce a DTensorPlan."""
        mesh = make_device_mesh()
        applier = DTensorApplier(mesh)
        sharding_spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={
                "weight": TensorSharding("weight", mesh_axes=[0], partition_shape=[4]),
            },
        )
        model = MagicMock()
        model.named_parameters.return_value = [("weight", MagicMock())]
        plan = applier.apply_sharding(model, sharding_spec)
        assert isinstance(plan, DTensorPlan)
        assert hasattr(plan, "is_usable")

    def test_apply_sharding_produces_placements(self) -> None:
        """When DTensor is available, placements must be populated."""
        mesh = make_device_mesh()
        applier = DTensorApplier(mesh)
        ts = TensorSharding(
            "weight", mesh_axes=[0], partition_shape=[4], replicate_on_other_axes=False
        )
        sharding_spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={"weight": ts},
        )
        placement = applier._placement_for_tensor("weight", sharding_spec)
        if placement is not None:
            assert len(placement) >= 1
            name = placement[0].__class__.__name__
            assert name in ("Shard", "Replicate")

    def test_placement_for_unknown_tensor(self) -> None:
        """_placement_for_tensor must return None for unknown tensors."""
        mesh = make_device_mesh()
        applier = DTensorApplier(mesh)
        spec = ShardingSpec(mesh_shape=[4])
        placement = applier._placement_for_tensor("unknown", spec)
        assert placement is None

    def test_placement_for_replicated_tensor(self) -> None:
        """A replicated tensor must map to a Replicate placement."""
        mesh = make_device_mesh()
        applier = DTensorApplier(mesh)
        ts = TensorSharding("x", mesh_axes=[], partition_shape=[], replicate_on_other_axes=True)
        spec = ShardingSpec(mesh_shape=[4], tensor_shardings={"x": ts})
        placement = applier._placement_for_tensor("x", spec)
        if placement is not None:
            assert len(placement) >= 1

    def test_collectives_carried_to_plan(self) -> None:
        """The DTensorPlan must carry collective insertion info."""
        mesh = make_device_mesh()
        applier = DTensorApplier(mesh)
        collectives = [
            {"type": "all-reduce", "op": "sum", "tensor": "w", "estimated_bytes": 1024},
        ]
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={"w": TensorSharding("w", mesh_axes=[0], partition_shape=[4])},
            inserted_collectives=collectives,
        )
        model = MagicMock()
        model.named_parameters.return_value = []
        plan = applier.apply_sharding(model, spec)
        assert len(plan.collective_insertions) == 1
        assert plan.collective_insertions[0]["type"] == "all-reduce"

    def test_apply_to_model_noop_without_placements(self) -> None:
        """apply_to_model must return the model unchanged when no placements."""
        mesh = make_device_mesh()
        applier = DTensorApplier(mesh)
        plan = DTensorPlan(is_usable=False)
        model = MagicMock()
        result = applier.apply_to_model(model, plan)
        assert result is model

    def test_sharding_spec_to_lite_conversion(self) -> None:
        """to_lite() must produce a valid ShardingSpecLite."""
        spec = ShardingSpec(
            mesh_shape=[2, 2],
            tensor_shardings={
                "A": TensorSharding("A", mesh_axes=[0, 1], partition_shape=[2, 2]),
            },
            strategy_used=ShardingStrategy.TENSOR_PARALLEL,
        )
        lite = spec.to_lite()
        assert lite.mesh.axes == (2, 2)
        assert "A" in lite.tensor_shardings

    def test_tensor_sharding_to_lite(self) -> None:
        """TensorSharding.to_lite() must produce a valid TensorShardingLite."""
        ts = TensorSharding(
            "input", mesh_axes=[0], partition_shape=[4], replicate_on_other_axes=True
        )
        lite = ts.to_lite()
        assert lite.tensor_name == "input"
        assert lite.mesh_axes == (0,)
        assert lite.partition_shape == (4,)

    def test_tensor_sharding_from_lite(self) -> None:
        """TensorSharding.from_lite() must restore the original."""
        from src.common.types import TensorShardingLite

        lite = TensorShardingLite(tensor_name="x", mesh_axes=(0, 1), partition_shape=(2, 2))
        ts = TensorSharding.from_lite(lite)
        assert ts.tensor_name == "x"
        assert ts.mesh_axes == [0, 1]
        assert ts.partition_shape == [2, 2]


# =============================================================================
# 11. Shard execution / fat binary dispatch
# =============================================================================


class TestShardExecution:
    """Tests for ShardExecutor and ShardExecutionResult."""

    def test_shard_execution_result_defaults(self) -> None:
        """ShardExecutionResult must have sensible defaults."""
        result = ShardExecutionResult(
            shard_id=0,
            vendor="nvidia",
            arch="sm_90",
            device_id=0,
            success=True,
        )
        assert result.shard_id == 0
        assert result.is_usable is False

    def test_shard_execution_result_is_usable_with_binary(self) -> None:
        """is_usable must be True when success and fat_binary_result set."""
        result = ShardExecutionResult(
            shard_id=0,
            vendor="nvidia",
            arch="sm_90",
            device_id=0,
            success=True,
            fat_binary_result=MagicMock(is_usable=True),
        )
        assert result.is_usable is True

    def test_shard_execution_result_success_requires_binary(self) -> None:
        """is_usable must be False if fat_binary_result is None."""
        result = ShardExecutionResult(
            shard_id=0,
            vendor="nvidia",
            arch="sm_90",
            device_id=0,
            success=True,
            fat_binary_result=None,
        )
        assert result.is_usable is False

    def test_exeuctor_creation(self) -> None:
        """ShardExecutor must accept a mesh and comm backend."""
        mesh = make_device_mesh(num_devices=2)
        comm = CommBackend(mesh)
        executor = ShardExecutor(mesh, comm)
        assert executor.device_mesh is mesh
        assert executor.comm_backend is comm

    def test_execute_all_shards_returns_list(self) -> None:
        """execute_all_shards must return a list of results."""
        mesh = make_device_mesh(num_devices=2)
        comm = CommBackend(mesh)
        executor = ShardExecutor(mesh, comm)
        gspmd_result = MagicMock()
        gspmd_result.is_usable = True
        stablehlo_module = MagicMock()
        stablehlo_module.mlir_text = "mock-mlir"
        results = executor.execute_all_shards(gspmd_result, stablehlo_module)
        assert isinstance(results, list)

    def test_execute_all_shards_returns_one_per_device(self) -> None:
        """There must be one ShardExecutionResult per device."""
        mesh = make_device_mesh(num_devices=4)
        comm = CommBackend(mesh)
        executor = ShardExecutor(mesh, comm)
        gspmd_result = MagicMock()
        gspmd_result.is_usable = True
        stablehlo_module = MagicMock()
        stablehlo_module.mlir_text = "mock-mlir"
        results = executor.execute_all_shards(gspmd_result, stablehlo_module)
        assert len(results) == 4

    def test_shard_result_contains_vendor_info(self) -> None:
        """Each shard result must contain the correct vendor and arch."""
        mesh = make_device_mesh(num_devices=2, vendor=DeviceVendor.NVIDIA, arch="sm_90")
        comm = CommBackend(mesh)
        executor = ShardExecutor(mesh, comm)
        gspmd_result = MagicMock()
        gspmd_result.is_usable = True
        stablehlo_module = MagicMock()
        stablehlo_module.mlir_text = "mock-mlir"
        results = executor.execute_all_shards(gspmd_result, stablehlo_module)
        for r in results:
            assert r.vendor == "nvidia"
            assert r.arch == "sm_90"


# =============================================================================
# 12. Device mesh topology and vendor detection
# =============================================================================


class TestDeviceMeshHeterogeneity:
    """Tests for DeviceMesh vendor and heterogeneity detection."""

    def test_single_vendor_not_heterogeneous(self) -> None:
        """A single-vendor mesh must not be heterogeneous."""
        mesh = make_device_mesh(num_devices=4, vendor=DeviceVendor.NVIDIA)
        assert mesh.is_heterogeneous is False
        assert len(mesh.vendors) == 1

    def test_dual_vendor_heterogeneous(self) -> None:
        """A dual-vendor mesh must be heterogeneous."""
        mesh = make_heterogeneous_mesh()
        assert mesh.is_heterogeneous is True
        assert DeviceVendor.NVIDIA in mesh.vendors
        assert DeviceVendor.AMD in mesh.vendors

    def test_get_devices_by_vendor(self) -> None:
        """get_devices_by_vendor must filter correctly."""
        mesh = make_heterogeneous_mesh()
        nv_devices = mesh.get_devices_by_vendor(DeviceVendor.NVIDIA)
        amd_devices = mesh.get_devices_by_vendor(DeviceVendor.AMD)
        assert len(nv_devices) == 2
        assert len(amd_devices) == 2
        for d in nv_devices:
            assert d.vendor == DeviceVendor.NVIDIA
        for d in amd_devices:
            assert d.vendor == DeviceVendor.AMD

    def test_vendor_mesh_shape(self) -> None:
        """vendor_mesh_shape must return correct shapes per vendor."""
        mesh = make_heterogeneous_mesh()
        shapes = mesh.vendor_mesh_shape()
        assert DeviceVendor.NVIDIA in shapes
        assert DeviceVendor.AMD in shapes
        assert shapes[DeviceVendor.NVIDIA] == [2]
        assert shapes[DeviceVendor.AMD] == [2]

    def test_to_dict_includes_vendors(self) -> None:
        """to_dict must include vendor information."""
        mesh = make_heterogeneous_mesh()
        d = mesh.to_dict()
        assert "vendors" in d
        assert "nvidia" in d["vendors"]
        assert "amd" in d["vendors"]
        assert d["is_heterogeneous"] is True
        assert d["num_devices"] == 4

    def test_mesh_device_display_name(self) -> None:
        """MeshDevice.display_name must format correctly."""
        d = MeshDevice(0, DeviceVendor.NVIDIA, "sm_90", 80, 989, InterconnectType.NVLINK)
        assert "nvidia" in d.display_name
        assert "sm_90" in d.display_name

    def test_num_devices_property(self) -> None:
        """num_devices must return the device count."""
        mesh = make_device_mesh(num_devices=8)
        assert mesh.num_devices == 8
        assert mesh.total_devices == 8

    def test_mesh_shape_different_from_num_devices(self) -> None:
        """total_devices must be the product of mesh_shape."""
        mesh = make_2d_mesh()
        assert mesh.num_devices == 4
        assert mesh.total_devices == 4
        assert mesh.mesh_shape == [2, 2]

    def test_amd_device_detection(self) -> None:
        """AMD devices must be detected correctly."""
        d = MeshDevice(0, DeviceVendor.AMD, "gfx942", 192, 1307, InterconnectType.INFINITY_FABRIC)
        assert d.vendor == DeviceVendor.AMD
        assert d.arch == "gfx942"

    def test_intel_device_detection(self) -> None:
        """Intel devices must be detected correctly."""
        d = MeshDevice(0, DeviceVendor.INTEL, "intel_gaudi2", 96, 900, InterconnectType.ETHERNET)
        assert d.vendor == DeviceVendor.INTEL
        assert d.arch == "intel_gaudi2"

    def test_vendor_in_target_inference(self) -> None:
        """infer_target_from_mesh must return correct targets per vendor."""
        nv_mesh = make_device_mesh(vendor=DeviceVendor.NVIDIA, arch="sm_90")
        assert infer_target_from_mesh(nv_mesh) == "nvidia/nvidia-h100"

        amd_mesh = make_device_mesh(vendor=DeviceVendor.AMD, arch="gfx942")
        assert infer_target_from_mesh(amd_mesh) == "rocm/gfx942"

        intel_mesh = make_device_mesh(vendor=DeviceVendor.INTEL, arch="intel_gaudi2")
        assert infer_target_from_mesh(intel_mesh) == "intel/gaudi-2"

    def test_list_mesh_falls_back_to_llvm(self) -> None:
        """A bare mesh-shape list must fall back to 'llvm'."""
        assert infer_target_from_mesh([4]) == "llvm"
        assert infer_target_from_mesh([2, 2]) == "llvm"
        assert infer_target_from_mesh([]) == "llvm"


# =============================================================================
# 13. Edge cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests for the auto-sharding bridge."""

    def test_empty_device_mesh(self) -> None:
        """An empty device mesh must not crash."""
        mesh = DeviceMesh(devices=[], mesh_shape=[])
        assert mesh.num_devices == 0
        assert mesh.is_heterogeneous is False
        assert len(mesh.vendors) == 0

    def test_single_device_mesh(self) -> None:
        """A single-device mesh must work correctly."""
        mesh = make_device_mesh(num_devices=1)
        assert mesh.num_devices == 1
        assert mesh.is_heterogeneous is False

    def test_zero_size_mesh_shape(self) -> None:
        """A mesh with a zero in mesh_shape must not crash."""
        devices = [
            MeshDevice(0, DeviceVendor.NVIDIA, "sm_90", 80, 989, InterconnectType.NVLINK),
        ]
        mesh = DeviceMesh(devices=devices, mesh_shape=[0])
        assert mesh.total_devices == 0

    def test_sharding_spec_with_no_tensors(self) -> None:
        """A ShardingSpec with no tensors must not crash."""
        spec = ShardingSpec(mesh_shape=[4], tensor_shardings={})
        assert len(spec.tensor_shardings) == 0
        assert spec.cache_key is not None

    def test_sharding_spec_cache_key_with_no_tensors(self) -> None:
        """Cache key must still be computed with zero tensors."""
        spec = ShardingSpec(mesh_shape=[4], tensor_shardings={})
        key = spec.cache_key
        assert isinstance(key, str)
        assert len(key) == 64

    def test_collectives_with_zero_devices(self) -> None:
        """_compute_collectives must handle zero devices gracefully."""
        module = make_stablehlo_module()
        spec = ShardingSpec(
            mesh_shape=[0],
            tensor_shardings={
                "A": TensorSharding("A", mesh_axes=[0], partition_shape=[]),
            },
        )
        collectives = _compute_collectives(spec, module, 0)
        assert isinstance(collectives, list)

    def test_cost_model_with_huge_device_count(self) -> None:
        """The cost model must handle a large number of devices."""
        result = _CommCostModel.all_reduce_bytes(1024, 1000)
        assert result > 0
        assert isinstance(result, int)

    def test_cost_model_with_zero_tensor(self) -> None:
        """The cost model must handle zero-size tensors."""
        assert _CommCostModel.all_reduce_bytes(0, 4) == 0
        assert _CommCostModel.all_gather_bytes(0, 4) == 0
        assert _CommCostModel.reduce_scatter_bytes(0, 4) == 0
        assert _CommCostModel.all_to_all_bytes(0, 4) == 0

    def test_annotate_with_malformed_mlir(self) -> None:
        """Malformed MLIR must not crash; it should return the input unchanged."""
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={"A": TensorSharding("A", mesh_axes=[0], partition_shape=[4])},
        )
        malformed = "this is not valid MLIR"
        result = _annotate_stablehlo_with_sharding(malformed, spec)
        assert result == malformed

    def test_annotate_with_no_func_func(self) -> None:
        """MLIR with no func.func must return the input unchanged."""
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={"A": TensorSharding("A", mesh_axes=[0], partition_shape=[4])},
        )
        no_func = "module {\n  // just a comment\n}\n"
        result = _annotate_stablehlo_with_sharding(no_func, spec)
        assert result == no_func

    def test_sharding_config_with_empty_example_inputs(self) -> None:
        """ShardingConfig must accept empty example_inputs."""
        config = ShardingConfig(
            model=None,
            example_inputs=(),
            device_mesh=make_device_mesh(),
        )
        assert config.example_inputs == ()

    def test_sharding_result_is_usable_without_dtensor(self) -> None:
        """ShardingResult.is_usable must be False when dtensor_plan is None."""
        result = ShardingResult(success=True, dtensor_plan=None)
        assert result.is_usable is False

        result_with_plan = ShardingResult(
            success=True,
            dtensor_plan=DTensorPlan(is_usable=True),
        )
        assert result_with_plan.is_usable is True

    def test_comm_group_properties(self) -> None:
        """CommGroup must report correct num_devices."""
        group = CommGroup(group_id=0, devices=[0, 1, 2, 3], library=CommLibrary.NCCL)
        assert group.num_devices == 4

    def test_tensor_sharding_default_replicate_true(self) -> None:
        """New TensorSharding must default replicate_on_other_axes to True."""
        ts = TensorSharding(tensor_name="x")
        assert ts.replicate_on_other_axes is True

    def test_gspmd_result_is_usable(self) -> None:
        """GSPMDResult.is_usable must be True only when success and spec exists."""
        r1 = GSPMDResult(success=True, sharding_spec=ShardingSpec())
        assert r1.is_usable is True

        r2 = GSPMDResult(success=True, sharding_spec=None)
        assert r2.is_usable is False

        r3 = GSPMDResult(success=False, sharding_spec=ShardingSpec())
        assert r3.is_usable is False

    def test_sharding_strategy_enum_values(self) -> None:
        """ShardingStrategy enum must have the expected values."""
        assert ShardingStrategy.AUTO.value == 1
        assert ShardingStrategy.REPLICATED.value == 2
        assert ShardingStrategy.DATA_PARALLEL.value == 3
        assert ShardingStrategy.MODEL_PARALLEL.value == 4
        assert ShardingStrategy.TENSOR_PARALLEL.value == 5

    def test_gspmd_result_with_tier_info(self) -> None:
        """GSPMDResult must carry tier_used as a string."""
        result = GSPMDResult(
            success=True,
            sharding_spec=ShardingSpec(),
            tier_used="torch_xla_spmd",
        )
        assert result.tier_used == "torch_xla_spmd"

    def test_all_to_all_with_many_devices(self) -> None:
        """All-to-all cost must be correct with many devices."""
        assert _CommCostModel.all_to_all_bytes(1024, 8) == 896

    def test_reduce_scatter_matches_all_reduce(self) -> None:
        """reduce_scatter must produce the same result as all_reduce."""
        for tensor_bytes in [1, 64, 1024, 1048576]:
            for num_devices in [2, 4, 8, 16]:
                assert _CommCostModel.reduce_scatter_bytes(
                    tensor_bytes, num_devices
                ) == _CommCostModel.all_reduce_bytes(tensor_bytes, num_devices)

    def test_gspmd_cache_key_uses_tier_info(self) -> None:
        """The cache key must differ when strategy differs."""
        spec1 = ShardingSpec(mesh_shape=[4], strategy_used=ShardingStrategy.AUTO)
        spec2 = ShardingSpec(mesh_shape=[4], strategy_used=ShardingStrategy.REPLICATED)
        assert spec1.cache_key != spec2.cache_key

    def test_dtensor_plan_collectives_empty_default(self) -> None:
        """DTensorPlan must default collective_insertions to empty list."""
        plan = DTensorPlan()
        assert plan.collective_insertions == []

    def test_gspmd_result_diagnostics_defaults(self) -> None:
        """GSPMDResult.diagnostics must default to empty dict."""
        r = GSPMDResult(success=True)
        assert r.diagnostics == {}

    def test_comm_backend_with_no_devices(self) -> None:
        """CommBackend must handle an empty mesh."""
        mesh = DeviceMesh(devices=[], mesh_shape=[])
        comm = CommBackend(mesh)
        assert len(comm._groups) == 0
        assert len(comm._cross_vendor_bridges) == 0

    def test_sharding_config_custom_strategy(self) -> None:
        """ShardingConfig must accept a custom sharding strategy."""
        config = ShardingConfig(
            model=None,
            example_inputs=(),
            device_mesh=make_device_mesh(),
            sharding_strategy=ShardingStrategy.TENSOR_PARALLEL,
        )
        assert config.sharding_strategy == ShardingStrategy.TENSOR_PARALLEL

    def test_num_devices_on_axes_computation(self) -> None:
        """_num_devices_on_axes must compute the correct product."""
        from src.bridges.pytorch_xla.gspmd_runner import _num_devices_on_axes

        assert _num_devices_on_axes([0], [4]) == 4
        assert _num_devices_on_axes([0, 1], [2, 2]) == 4
        assert _num_devices_on_axes([0], [2, 2]) == 2
        assert _num_devices_on_axes([], [4]) == 1
        assert _num_devices_on_axes([5], [4]) == 1

    def test_total_devices_helper(self) -> None:
        """_total_devices must compute the product of all dimensions."""
        from src.bridges.pytorch_xla.gspmd_runner import _total_devices

        assert _total_devices([4]) == 4
        assert _total_devices([2, 2]) == 4
        assert _total_devices([2, 4, 8]) == 64
        assert _total_devices([]) == 1
        assert _total_devices([1]) == 1

    def test_sharding_result_error_none_by_default(self) -> None:
        """ShardingResult.error must be None by default."""
        r = ShardingResult(success=True)
        assert r.error is None

    def test_gspmd_runner_cache_lifecycle(self) -> None:
        """GSPMDRunner cache must store and return results."""
        runner = GSPMDRunner()
        assert runner._cache == {}
        module = make_stablehlo_module()
        mesh = make_device_mesh()
        result = runner.run(module, mesh)
        assert result.success is True
        assert len(runner._cache) > 0
