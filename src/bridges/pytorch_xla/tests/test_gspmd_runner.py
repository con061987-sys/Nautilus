"""Tests for the GSPMD runner module."""

from __future__ import annotations

import re

import pytest

from src.bridges.pytorch_xla.device_mesh import (
    DeviceMesh,
    DeviceVendor,
    InterconnectType,
    MeshDevice,
)
from src.bridges.pytorch_xla.device_mesh_utils import infer_target_from_mesh
from src.bridges.pytorch_xla.gspmd_runner import (
    GSPMDResult,
    GSPMDRunner,
    ShardingSpec,
    ShardingStrategy,
    TensorSharding,
    _annotate_stablehlo_with_sharding,
    _insert_sharding_attr,
    _sharding_spec_to_string,
    _TVMMetaScheduleSharding,
)
from src.bridges.pytorch_xla.stablehlo_export import StableHLOModule

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


class TestShardingAnnotationRealMLIRAttrs:
    """Regression tests for H-03: sharding annotations must be REAL MLIR
    attributes, not ``// mhlo.sharding`` comments. The StableHLO pipeline
    only consumes the former."""

    def test_annotate_inserts_real_attribute_not_comment(self) -> None:
        """H-03: the result must contain a real ``{mhlo.sharding = "..."}``
        attribute on a function argument, not a comment line."""
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={
                "A": TensorSharding("A", mesh_axes=[0], partition_shape=[4]),
            },
        )
        result = _annotate_stablehlo_with_sharding(SAMPLE_MLIR, spec)
        assert '{mhlo.sharding = "devices=[4]<=[4]"}' in result
        assert "// tensor" not in result
        assert "// output_sharding" not in result
        assert "// sharding_spec" not in result
        assert "// GSPMD-sharded" not in result

    def test_annotate_replicated_uses_special_keyword(self) -> None:
        """Replicated tensors use the ``"replicated"`` keyword, not a
        custom device assignment."""
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={
                "A": TensorSharding(
                    "A", mesh_axes=[], partition_shape=[],
                    replicate_on_other_axes=True,
                ),
            },
        )
        result = _annotate_stablehlo_with_sharding(SAMPLE_MLIR, spec)
        assert '{mhlo.sharding = "replicated"}' in result

    def test_annotate_model_parallel_format(self) -> None:
        """Model parallel on axis 0 with partition [1, N] produces
        ``devices=[1,N]<=[N]``."""
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={
                "A": TensorSharding("A", mesh_axes=[0], partition_shape=[1, 4]),
            },
        )
        result = _annotate_stablehlo_with_sharding(SAMPLE_MLIR, spec)
        assert '{mhlo.sharding = "devices=[1,4]<=[4]"}' in result

    def test_annotate_tensor_parallel_2d_mesh(self) -> None:
        """Tensor parallel on a 2x2 mesh produces ``devices=[2,2]<=[4]``."""
        spec = ShardingSpec(
            mesh_shape=[2, 2],
            tensor_shardings={
                "A": TensorSharding(
                    "A", mesh_axes=[0, 1], partition_shape=[2, 2],
                ),
            },
        )
        result = _annotate_stablehlo_with_sharding(SAMPLE_MLIR, spec)
        assert '{mhlo.sharding = "devices=[2,2]<=[4]"}' in result

    def test_annotate_is_idempotent(self) -> None:
        """Re-annotating the same MLIR must not duplicate the attribute."""
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
            assert attrs.count("mhlo.sharding") == 1, (
                f"duplicate mhlo.sharding on %A: {attrs}"
            )

    def test_annotate_noop_on_empty_inputs(self) -> None:
        """Empty MLIR or empty spec must return the input unchanged."""
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={
                "A": TensorSharding("A", mesh_axes=[0], partition_shape=[4]),
            },
        )
        assert _annotate_stablehlo_with_sharding("", spec) == ""
        assert _annotate_stablehlo_with_sharding(SAMPLE_MLIR, ShardingSpec(mesh_shape=[4])) == SAMPLE_MLIR

    def test_annotate_no_match_for_unknown_arg(self) -> None:
        """If the tensor name doesn't match any function arg, the MLIR is
        returned unchanged (no spurious annotations)."""
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={
                "GHOST": TensorSharding("GHOST", mesh_axes=[0], partition_shape=[4]),
            },
        )
        result = _annotate_stablehlo_with_sharding(SAMPLE_MLIR, spec)
        assert "GHOST" not in result
        assert "mhlo.sharding" not in result

    def test_annotate_preserves_op_body(self) -> None:
        """Annotation must not damage the function body (ops, returns)."""
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={
                "A": TensorSharding("A", mesh_axes=[0], partition_shape=[4]),
            },
        )
        result = _annotate_stablehlo_with_sharding(SAMPLE_MLIR, spec)
        assert "stablehlo.multiply" in result
        assert "return %0" in result

    def test_sharding_spec_to_string_replicated(self) -> None:
        """_sharding_spec_to_string maps empty mesh_axes to ``replicated``."""
        ts = TensorSharding("A", mesh_axes=[], partition_shape=[], replicate_on_other_axes=True)
        assert _sharding_spec_to_string(ts, [4]) == "replicated"

    def test_sharding_spec_to_string_maximal(self) -> None:
        """Empty mesh_axes + replicate=False -> ``maximal`` (single device)."""
        ts = TensorSharding("A", mesh_axes=[], partition_shape=[], replicate_on_other_axes=False)
        assert _sharding_spec_to_string(ts, [1]) == "maximal"

    def test_insert_sharding_attr_preserves_return_type(self) -> None:
        """The function signature's return type and braces must be intact."""
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
        assert "func.func @foo" in result
        assert '{mhlo.sharding = "replicated"}' in result


class TestTVMInferTarget:
    """Regression tests for H-06: TVM MetaSchedule must use a real GPU
    target, not hardcoded ``Target("llvm")`` (which compiles for CPU)."""

    def test_mesh_shape_list_falls_back_to_llvm(self) -> None:
        """A bare mesh-shape list (no device info) must fall back to ``llvm``."""
        assert infer_target_from_mesh([4]) == "llvm"
        assert infer_target_from_mesh([2, 2]) == "llvm"
        assert infer_target_from_mesh([]) == "llvm"

    def test_nvidia_sm90_maps_to_h100(self) -> None:
        """Nvidia Hopper (sm_90) maps to ``nvidia/nvidia-h100``."""
        mesh = DeviceMesh(
            devices=[MeshDevice(
                device_id=0, vendor=DeviceVendor.NVIDIA, arch="sm_90",
                memory_gb=80, compute_tflops=989,
                interconnect=InterconnectType.NVLINK,
            )],
            mesh_shape=[1],
        )
        assert infer_target_from_mesh(mesh) == "nvidia/nvidia-h100"

    def test_nvidia_sm80_maps_to_a100(self) -> None:
        """Nvidia Ampere (sm_80) maps to ``nvidia/nvidia-a100``."""
        mesh = DeviceMesh(
            devices=[MeshDevice(
                device_id=0, vendor=DeviceVendor.NVIDIA, arch="sm_80",
                memory_gb=80, compute_tflops=312,
                interconnect=InterconnectType.NVLINK,
            )],
            mesh_shape=[1],
        )
        assert infer_target_from_mesh(mesh) == "nvidia/nvidia-a100"

    def test_amd_gfx942_maps_to_rocm(self) -> None:
        """AMD MI300X (gfx942) maps to ``rocm/gfx942``."""
        mesh = DeviceMesh(
            devices=[MeshDevice(
                device_id=0, vendor=DeviceVendor.AMD, arch="gfx942",
                memory_gb=192, compute_tflops=1300,
                interconnect=InterconnectType.INFINITY_FABRIC,
            )],
            mesh_shape=[1],
        )
        assert infer_target_from_mesh(mesh) == "rocm/gfx942"

    def test_intel_gaudi2_maps_to_intel(self) -> None:
        """Intel Gaudi 2 maps to ``intel/gaudi-2``."""
        mesh = DeviceMesh(
            devices=[MeshDevice(
                device_id=0, vendor=DeviceVendor.INTEL, arch="intel_gaudi2",
                memory_gb=96, compute_tflops=900,
                interconnect=InterconnectType.ETHERNET,
            )],
            mesh_shape=[1],
        )
        assert infer_target_from_mesh(mesh) == "intel/gaudi-2"

    def test_cpu_fallback(self) -> None:
        """CPU-only mesh falls back to ``llvm``."""
        mesh = DeviceMesh(
            devices=[MeshDevice(
                device_id=0, vendor=DeviceVendor.CPU, arch="x86_64",
                memory_gb=64, compute_tflops=0.5,
                interconnect=InterconnectType.PCIE,
            )],
            mesh_shape=[1],
        )
        assert infer_target_from_mesh(mesh) == "llvm"

    def test_vendor_fallback_for_unknown_arch(self) -> None:
        """A vendor we recognise but with an unknown arch falls back to the
        vendor's default target, not llvm."""
        mesh = DeviceMesh(
            devices=[MeshDevice(
                device_id=0, vendor=DeviceVendor.NVIDIA, arch="sm_999",
                memory_gb=80, compute_tflops=312,
                interconnect=InterconnectType.NVLINK,
            )],
            mesh_shape=[1],
        )
        assert infer_target_from_mesh(mesh) == "nvidia/nvidia-h100"

    def test_tvm_class_mesh_target_delegates_to_util(self) -> None:
        """_TVMMetaScheduleSharding._mesh_target must use the same mapping
        as infer_target_from_mesh."""
        nvidia_mesh = DeviceMesh(
            devices=[MeshDevice(
                device_id=0, vendor=DeviceVendor.NVIDIA, arch="sm_90",
                memory_gb=80, compute_tflops=989,
                interconnect=InterconnectType.NVLINK,
            )],
            mesh_shape=[1],
        )
        assert _TVMMetaScheduleSharding._mesh_target(nvidia_mesh) == "nvidia/nvidia-h100"
        assert _TVMMetaScheduleSharding._mesh_target([4]) == "llvm"
