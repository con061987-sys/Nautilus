"""Tests for the GSPMD runner module."""

from __future__ import annotations

import pytest

from src.bridges.pytorch_xla.gspmd_runner import (
    GSPMDRunner,
    GSPMDResult,
    ShardingSpec,
    ShardingStrategy,
    TensorSharding,
)
from src.bridges.pytorch_xla.stablehlo_export import StableHLOModule
from src.bridges.pytorch_xla.device_mesh import (
    DeviceMesh,
    DeviceVendor,
    InterconnectType,
    MeshDevice,
)


SAMPLE_MLIR = """
module {
  func.func @matmul(%A: tensor<128x128xf32>, %B: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = stablehlo.multiply %A, %B : tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
  }
}
"""


def make_stablehlo_module() -> StableHLOModule:
    """Create a sample StableHLO module for testing."""
    return StableHLOModule(
        mlir_text=SAMPLE_MLIR,
        function_name="matmul",
        input_specs=[{"name": "A", "dtype": "f32"}, {"name": "B", "dtype": "f32"}],
        output_specs=[{"name": "output"}],
        op_count=3,
        is_usable=True,
    )


def make_device_mesh(num_devices: int = 4) -> DeviceMesh:
    """Create a test device mesh."""
    devices = [
        MeshDevice(
            device_id=i,
            vendor=DeviceVendor.NVIDIA,
            arch="sm_90",
            memory_gb=80.0,
            compute_tflops=989.0,
            interconnect=InterconnectType.NVLINK,
        )
        for i in range(num_devices)
    ]
    return DeviceMesh(devices=devices, mesh_shape=[num_devices])


class TestShardingSpec:
    """Tests for the ShardingSpec class."""

    def test_sharding_spec_creation(self) -> None:
        """ShardingSpec should aggregate all sharding info."""
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={
                "A": TensorSharding(tensor_name="A", mesh_axes=[0], partition_shape=[4]),
            },
            inserted_collectives=[{"type": "all-reduce", "op": "sum"}],
            estimated_comm_volume_bytes=1024,
            strategy_used=ShardingStrategy.DATA_PARALLEL,
        )
        assert spec.mesh_shape == [4]
        assert "A" in spec.tensor_shardings
        assert len(spec.inserted_collectives) == 1

    def test_cache_key_stable(self) -> None:
        """Same spec should produce the same cache key."""
        spec1 = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={"A": TensorSharding("A", [0], [4])},
            strategy_used=ShardingStrategy.AUTO,
        )
        spec2 = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={"A": TensorSharding("A", [0], [4])},
            strategy_used=ShardingStrategy.AUTO,
        )
        assert spec1.cache_key == spec2.cache_key

    def test_cache_key_differs_on_strategy(self) -> None:
        """Different strategies should produce different cache keys."""
        spec1 = ShardingSpec(mesh_shape=[4], strategy_used=ShardingStrategy.AUTO)
        spec2 = ShardingSpec(mesh_shape=[4], strategy_used=ShardingStrategy.DATA_PARALLEL)
        assert spec1.cache_key != spec2.cache_key


class TestTensorSharding:
    """Tests for the TensorSharding class."""

    def test_tensor_sharding_default(self) -> None:
        """Default TensorSharding should be replicated."""
        ts = TensorSharding(tensor_name="input")
        assert ts.tensor_name == "input"
        assert ts.replicate_on_other_axes is True

    def test_tensor_sharding_sharded(self) -> None:
        """A sharded tensor should specify axes and partition."""
        ts = TensorSharding(
            tensor_name="weight",
            mesh_axes=[0, 1],
            partition_shape=[2, 2],
            replicate_on_other_axes=False,
        )
        assert ts.mesh_axes == [0, 1]
        assert ts.partition_shape == [2, 2]


class TestGSPMDRunner:
    """Tests for the GSPMDRunner class."""

    def test_gspmd_runner_init(self) -> None:
        """GSPMDRunner should initialise with sensible defaults."""
        runner = GSPMDRunner()
        assert runner.default_strategy == ShardingStrategy.AUTO
        assert runner.timeout_seconds > 0

    def test_gspmd_runner_custom_strategy(self) -> None:
        """GSPMDRunner should accept a custom default strategy."""
        runner = GSPMDRunner(default_strategy=ShardingStrategy.DATA_PARALLEL)
        assert runner.default_strategy == ShardingStrategy.DATA_PARALLEL

    def test_run_with_data_parallel(self) -> None:
        """DATA_PARALLEL strategy should shard along mesh axis 0."""
        runner = GSPMDRunner()
        module = make_stablehlo_module()
        mesh = make_device_mesh(num_devices=4)
        result = runner.run(module, mesh, ShardingStrategy.DATA_PARALLEL)
        assert result.is_usable
        spec = result.sharding_spec
        assert spec is not None
        assert spec.strategy_used == ShardingStrategy.DATA_PARALLEL
        # All tensors should be sharded along axis 0
        for ts in spec.tensor_shardings.values():
            assert 0 in ts.mesh_axes

    def test_run_with_replicated(self) -> None:
        """REPLICATED strategy should not shard any tensor."""
        runner = GSPMDRunner()
        module = make_stablehlo_module()
        mesh = make_device_mesh(num_devices=4)
        result = runner.run(module, mesh, ShardingStrategy.REPLICATED)
        assert result.is_usable
        spec = result.sharding_spec
        assert spec is not None
        for ts in spec.tensor_shardings.values():
            assert ts.mesh_axes == []
            assert ts.replicate_on_other_axes is True

    def test_run_caches_results(self) -> None:
        """Subsequent runs with the same inputs should hit cache."""
        runner = GSPMDRunner()
        module = make_stablehlo_module()
        mesh = make_device_mesh(num_devices=4)
        result1 = runner.run(module, mesh, ShardingStrategy.DATA_PARALLEL)
        result2 = runner.run(module, mesh, ShardingStrategy.DATA_PARALLEL)
        assert result2.cache_hit is True

    def test_run_with_invalid_module(self) -> None:
        """An invalid module should return a failed result."""
        runner = GSPMDRunner()
        result = runner.run(None, make_device_mesh())
        assert result.success is False
        assert result.error is not None

    def test_run_records_diagnostics(self) -> None:
        """A successful run should record strategy and mesh info."""
        runner = GSPMDRunner()
        module = make_stablehlo_module()
        mesh = make_device_mesh(num_devices=4)
        result = runner.run(module, mesh, ShardingStrategy.DATA_PARALLEL)
        assert result.diagnostics["strategy"] == "DATA_PARALLEL"
        assert result.diagnostics["mesh_shape"] == [4]

    def test_custom_shardings_override(self) -> None:
        """Custom shardings should override the default strategy."""
        runner = GSPMDRunner()
        module = make_stablehlo_module()
        mesh = make_device_mesh(num_devices=4)
        custom = {
            "A": TensorSharding(
                tensor_name="A",
                mesh_axes=[0, 1],
                partition_shape=[2, 2],
            ),
        }
        result = runner.run(
            module, mesh, ShardingStrategy.DATA_PARALLEL, custom_shardings=custom,
        )
        spec = result.sharding_spec
        assert spec is not None
        assert spec.tensor_shardings["A"].mesh_axes == [0, 1]
        # B should still use the default
        assert spec.tensor_shardings["B"].mesh_axes == [0]

    def test_inserts_collectives(self) -> None:
        """GSPMD should insert collective ops for sharded params."""
        runner = GSPMDRunner()
        module = make_stablehlo_module()
        mesh = make_device_mesh(num_devices=4)
        result = runner.run(module, mesh, ShardingStrategy.DATA_PARALLEL)
        spec = result.sharding_spec
        assert spec is not None
        assert len(spec.inserted_collectives) > 0
        # The collective should be an all-reduce
        assert spec.inserted_collectives[0]["type"] == "all-reduce"
