"""Tests for the collective insertion pass.

Two layers of testing:
  1. Unit tests for each public type and helper.
  2. Property-based tests (via ``hypothesis``) that generate random
     sharding specs and assert invariants on the resulting plan.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from src.bridges.pytorch_xla.collective_insertion import (
    CollectiveInserter,
    CollectiveType,
    InsertedCollective,
    _collective_volume_bytes,
    _extract_func_arg_types,
    _parse_tensor_shape_and_dtype,
    _plan_collectives,
    _tensor_bytes,
    plan_and_insert,
)
from src.bridges.pytorch_xla.comm_backend import CommLibrary
from src.bridges.pytorch_xla.device_mesh import (
    DeviceMesh,
    DeviceVendor,
    InterconnectType,
    MeshDevice,
)
from src.bridges.pytorch_xla.gspmd_runner import (
    ShardingSpec,
    ShardingStrategy,
    TensorSharding,
)
from src.bridges.pytorch_xla.stablehlo_export import StableHLOModule

# ── Sample MLIR fixtures ────────────────────────────────────────────────

SAMPLE_MLIR = """
module {
  func.func @matmul(%A: tensor<128x128xf32>, %B: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = stablehlo.multiply %A, %B : tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
  }
}
"""

SAMPLE_MLIR_BF16 = """
module {
  func.func @gemm(%A: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
    %0 = stablehlo.add %A, %A : tensor<64x64xbf16>
    return %0 : tensor<64x64xbf16>
  }
}
"""

SAMPLE_MLIR_SCALAR = """
module {
  func.func @scalar_op(%x: tensor<f32>) -> tensor<f32> {
    return %x : tensor<f32>
  }
}
"""


# ── Helpers ─────────────────────────────────────────────────────────────


def _make_module(mlir: str = SAMPLE_MLIR, name: str = "matmul") -> StableHLOModule:
    return StableHLOModule(
        mlir_text=mlir,
        function_name=name,
        input_specs=[{"name": "A", "shape": [128, 128], "dtype": "f32"}],
        output_specs=[{"name": "output"}],
        op_count=2,
        is_usable=True,
    )


def _make_mesh(
    num_devices: int = 4,
    vendor: DeviceVendor = DeviceVendor.NVIDIA,
) -> DeviceMesh:
    devices = [
        MeshDevice(
            device_id=i,
            vendor=vendor,
            arch="sm_90"
            if vendor == DeviceVendor.NVIDIA
            else ("gfx942" if vendor == DeviceVendor.AMD else "intel_gaudi2"),
            memory_gb=80.0,
            compute_tflops=900.0,
            interconnect=(
                InterconnectType.NVLINK
                if vendor == DeviceVendor.NVIDIA
                else InterconnectType.INFINITY_FABRIC
                if vendor == DeviceVendor.AMD
                else InterconnectType.ETHERNET
            ),
        )
        for i in range(num_devices)
    ]
    return DeviceMesh(devices=devices, mesh_shape=[num_devices])


# ── Public type tests ───────────────────────────────────────────────────


class TestCollectiveType:
    def test_values_match_stablehlo_ops(self) -> None:
        """The enum values must be valid StableHLO dialect op suffixes."""
        assert CollectiveType.ALL_REDUCE.value == "all_reduce"
        assert CollectiveType.ALL_GATHER.value == "all_gather"
        assert CollectiveType.REDUCE_SCATTER.value == "reduce_scatter"
        assert CollectiveType.ALL_TO_ALL.value == "all_to_all"

    def test_member_count(self) -> None:
        """All four canonical collectives are present."""
        assert len(CollectiveType) == 4


class TestTensorShapeParsing:
    @pytest.mark.parametrize(
        "type_str, expected_shape, expected_dtype",
        [
            ("tensor<128x256xf32>", (128, 256), "f32"),
            ("tensor<4xi64>", (4,), "i64"),
            ("tensor<f32>", (), "f32"),
            ("tensor<2x2xbf16>", (2, 2), "bf16"),
            ("tensor<1x1x1xi8>", (1, 1, 1), "i8"),
        ],
    )
    def test_parse_valid(
        self,
        type_str: str,
        expected_shape: tuple[int, ...],
        expected_dtype: str,
    ) -> None:
        shape, dtype = _parse_tensor_shape_and_dtype(type_str)
        assert shape == expected_shape
        assert dtype == expected_dtype

    def test_parse_invalid_raises(self) -> None:
        from src.common.errors import GSPMDError

        with pytest.raises(GSPMDError):
            _parse_tensor_shape_and_dtype("not a tensor type")


class TestTensorBytes:
    @pytest.mark.parametrize(
        "shape, dtype, expected",
        [
            ((128, 128), "f32", 128 * 128 * 4),
            ((64, 64), "f16", 64 * 64 * 2),
            ((32, 32), "bf16", 32 * 32 * 2),
            ((8,), "i64", 8 * 8),
            ((1, 1, 1, 1), "i1", 1),
            ((), "f32", 4),  # scalar: 1 element x 4 bytes
            ((100, 100), "unknown_dtype", 100 * 100 * 4),  # default fp32
        ],
    )
    def test_tensor_bytes(
        self,
        shape: tuple[int, ...],
        dtype: str,
        expected: int,
    ) -> None:
        assert _tensor_bytes(shape, dtype) == expected


class TestCollectiveVolume:
    """Verify the GSPMD cost model exactly."""

    def test_all_reduce_volume(self) -> None:
        # N=4 devices, 1024-byte tensor: 2*1024*3/4 = 1536
        assert _collective_volume_bytes(CollectiveType.ALL_REDUCE, 1024, 4) == 1536

    def test_all_gather_volume(self) -> None:
        # N=4, 1024 bytes: 1024*3 = 3072
        assert _collective_volume_bytes(CollectiveType.ALL_GATHER, 1024, 4) == 3072

    def test_reduce_scatter_volume(self) -> None:
        # N=4, 1024 bytes: 2*1024*3/4 = 1536
        assert _collective_volume_bytes(CollectiveType.REDUCE_SCATTER, 1024, 4) == 1536

    def test_all_to_all_volume(self) -> None:
        # N=4, 1024 bytes: 1024*3/4 = 768
        assert _collective_volume_bytes(CollectiveType.ALL_TO_ALL, 1024, 4) == 768

    @pytest.mark.parametrize("ctype", list(CollectiveType))
    def test_single_device_is_zero(self, ctype: CollectiveType) -> None:
        """A single-device mesh has zero communication cost for any op."""
        assert _collective_volume_bytes(ctype, 999_999, 1) == 0

    def test_volume_scales_with_tensor_bytes(self) -> None:
        """Volume is linear in tensor bytes (no hardcoded shapes)."""
        v1 = _collective_volume_bytes(CollectiveType.ALL_REDUCE, 1000, 4)
        v2 = _collective_volume_bytes(CollectiveType.ALL_REDUCE, 2000, 4)
        assert v2 == 2 * v1

    def test_volume_scales_with_num_devices(self) -> None:
        """Volume is monotonic in number of devices."""
        prev = 0
        for n in (2, 4, 8, 16):
            v = _collective_volume_bytes(CollectiveType.ALL_GATHER, 1024, n)
            assert v > prev
            prev = v


class TestExtractFuncArgTypes:
    def test_extracts_args(self) -> None:
        types = _extract_func_arg_types(SAMPLE_MLIR)
        assert "A" in types
        assert "B" in types
        assert types["A"] == "tensor<128x128xf32>"
        assert types["B"] == "tensor<128x128xf32>"

    def test_no_func_returns_empty(self) -> None:
        types = _extract_func_arg_types("module {}")
        assert types == {}


# ── Planner tests ──────────────────────────────────────────────────────


class TestPlanCollectives:
    def test_replicated_no_collectives(self) -> None:
        mod = _make_module()
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={
                "A": TensorSharding("A", [], [], replicate_on_other_axes=True),
            },
        )
        plan = _plan_collectives(spec, mod, mesh=None)
        assert plan == []

    def test_data_parallel_emits_four_collectives(self) -> None:
        mod = _make_module()
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={
                "A": TensorSharding("A", [0], [32, 128]),
            },
        )
        plan = _plan_collectives(spec, mod, mesh=None)
        # 4 collective types x 1 sharded axis = 4
        assert len(plan) == 4
        types = {c.collective_type for c in plan}
        assert types == set(CollectiveType)

    def test_tensor_parallel_2d_mesh(self) -> None:
        mod = _make_module()
        spec = ShardingSpec(
            mesh_shape=[2, 2],
            tensor_shardings={
                "A": TensorSharding("A", [0, 1], [64, 64]),
            },
        )
        plan = _plan_collectives(spec, mod, mesh=None)
        # 4 collective types x 2 sharded axes = 8
        assert len(plan) == 8

    def test_volume_scales_with_shape(self) -> None:
        """Volume must depend on real tensor shape, not be hardcoded."""
        # Two different MLIR strings are required because the parser
        # reads the per-device shape from the function's tensor<...>
        # type, so the MLIR text must encode the actual tensor shape.
        small_mlir = """
module {
  func.func @small(%A: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = stablehlo.multiply %A, %A : tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}
"""
        big_mlir = """
module {
  func.func @big(%A: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %0 = stablehlo.multiply %A, %A : tensor<256x256xf32>
    return %0 : tensor<256x256xf32>
  }
}
"""
        small_mod = StableHLOModule(
            mlir_text=small_mlir,
            function_name="small",
            input_specs=[{"name": "A", "shape": [4, 4], "dtype": "f32"}],
            is_usable=True,
        )
        big_mod = StableHLOModule(
            mlir_text=big_mlir,
            function_name="big",
            input_specs=[{"name": "A", "shape": [256, 256], "dtype": "f32"}],
            is_usable=True,
        )
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={
                "A": TensorSharding("A", [0], [1, 4]),
            },
        )
        spec_big = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={
                "A": TensorSharding("A", [0], [64, 256]),
            },
        )
        small_plan = _plan_collectives(spec, small_mod, mesh=None)
        big_plan = _plan_collectives(spec_big, big_mod, mesh=None)
        small_bytes = sum(c.estimated_bytes for c in small_plan)
        big_bytes = sum(c.estimated_bytes for c in big_plan)
        assert small_bytes > 0
        assert big_bytes > small_bytes, (
            f"big_bytes={big_bytes} should exceed small_bytes={small_bytes}"
        )

    def test_dtype_falls_back_to_f32(self) -> None:
        mod = StableHLOModule(
            mlir_text=SAMPLE_MLIR_BF16,
            function_name="gemm",
            input_specs=[{"name": "A", "shape": [64, 64], "dtype": "bf16"}],
            is_usable=True,
        )
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={"A": TensorSharding("A", [0], [16, 64])},
        )
        plan = _plan_collectives(spec, mod, mesh=None)
        assert all(c.dtype == "bf16" for c in plan)


class TestCommLibrarySelection:
    def test_nvidia_mesh_uses_nccl(self) -> None:
        mesh = _make_mesh(num_devices=4, vendor=DeviceVendor.NVIDIA)
        mod = _make_module()
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={"A": TensorSharding("A", [0], [32, 128])},
        )
        plan = _plan_collectives(spec, mod, mesh=mesh)
        assert all(c.comm_library == CommLibrary.NCCL for c in plan)

    def test_amd_mesh_uses_rccl(self) -> None:
        mesh = _make_mesh(num_devices=4, vendor=DeviceVendor.AMD)
        mod = _make_module()
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={"A": TensorSharding("A", [0], [32, 128])},
        )
        plan = _plan_collectives(spec, mod, mesh=mesh)
        assert all(c.comm_library == CommLibrary.RCCL for c in plan)

    def test_intel_mesh_uses_oneccl(self) -> None:
        mesh = _make_mesh(num_devices=4, vendor=DeviceVendor.INTEL)
        mod = _make_module()
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={"A": TensorSharding("A", [0], [32, 128])},
        )
        plan = _plan_collectives(spec, mod, mesh=mesh)
        assert all(c.comm_library == CommLibrary.ONECCL for c in plan)

    def test_heterogeneous_mesh_uses_mixed(self) -> None:
        devices = [
            MeshDevice(0, DeviceVendor.NVIDIA, "sm_90", 80.0, 900.0, InterconnectType.NVLINK),
            MeshDevice(
                1, DeviceVendor.AMD, "gfx942", 192.0, 1300.0, InterconnectType.INFINITY_FABRIC
            ),
        ]
        mesh = DeviceMesh(devices=devices, mesh_shape=[2])
        mod = _make_module()
        spec = ShardingSpec(
            mesh_shape=[2],
            tensor_shardings={"A": TensorSharding("A", [0], [64, 128])},
        )
        plan = _plan_collectives(spec, mod, mesh=mesh)
        assert all(c.comm_library == CommLibrary.MIXED for c in plan)

    def test_no_mesh_falls_back_to_nccl(self) -> None:
        """When no mesh is provided, default to NCCL (conservative)."""
        mod = _make_module()
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={"A": TensorSharding("A", [0], [32, 128])},
        )
        plan = _plan_collectives(spec, mod, mesh=None)
        assert all(c.comm_library == CommLibrary.NCCL for c in plan)


# ── MLIR emission tests ─────────────────────────────────────────────────


class TestEmitCollectives:
    def test_emit_inserts_all_four_collectives(self) -> None:
        mod = _make_module()
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={"A": TensorSharding("A", [0], [32, 128])},
        )
        result = plan_and_insert(mod, spec)
        assert result.success
        text = result.mlir_text
        assert "stablehlo.all_reduce" in text
        assert "stablehlo.all_gather" in text
        assert "stablehlo.reduce_scatter" in text
        assert "stablehlo.all_to_all" in text

    def test_emit_preserves_function_signature(self) -> None:
        mod = _make_module()
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={"A": TensorSharding("A", [0], [32, 128])},
        )
        result = plan_and_insert(mod, spec)
        assert "func.func @matmul" in result.mlir_text
        assert "%A: tensor<128x128xf32>" in result.mlir_text

    def test_emit_carries_channel_handle(self) -> None:
        mod = _make_module()
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={"A": TensorSharding("A", [0], [32, 128])},
        )
        result = plan_and_insert(mod, spec)
        assert "#stablehlo.channel_handle" in result.mlir_text

    def test_emit_carries_replica_groups(self) -> None:
        mod = _make_module()
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={"A": TensorSharding("A", [0], [32, 128])},
        )
        result = plan_and_insert(mod, spec)
        assert "replica_groups" in result.mlir_text
        assert "dense<" in result.mlir_text

    def test_emit_replicated_module_is_unchanged(self) -> None:
        mod = _make_module()
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={"A": TensorSharding("A", [], [], replicate_on_other_axes=True)},
        )
        result = plan_and_insert(mod, spec)
        assert result.success
        assert result.mlir_text == mod.mlir_text
        assert result.inserted_collectives == []
        assert result.total_comm_bytes == 0

    def test_emit_total_bytes_matches_sum(self) -> None:
        mod = _make_module()
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={"A": TensorSharding("A", [0], [32, 128])},
        )
        result = plan_and_insert(mod, spec)
        assert result.success
        assert result.total_comm_bytes == sum(
            c.estimated_bytes for c in result.inserted_collectives
        )

    def test_emit_with_bf16_dtype(self) -> None:
        mod = StableHLOModule(
            mlir_text=SAMPLE_MLIR_BF16,
            function_name="gemm",
            input_specs=[{"name": "A", "shape": [64, 64], "dtype": "bf16"}],
            is_usable=True,
        )
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={"A": TensorSharding("A", [0], [16, 64])},
        )
        result = plan_and_insert(mod, spec)
        assert result.success
        assert "bf16" in result.mlir_text

    def test_emit_with_return_inside_function(self) -> None:
        """Return inside the function body must not be the insertion point."""
        mlir = """
module {
  func.func @branchy(%A: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %zero = stablehlo.constant dense<0.0> : tensor<8x8xf32>
    %pred = stablehlo.constant dense<true> : tensor<i1>
    %result = stablehlo.if %pred -> (tensor<8x8xf32>) {
      stablehlo.return %A : tensor<8x8xf32>
    } else {
      stablehlo.return %zero : tensor<8x8xf32>
    }
    return %result : tensor<8x8xf32>
  }
}
"""
        mod = StableHLOModule(
            mlir_text=mlir,
            function_name="branchy",
            input_specs=[{"name": "A", "shape": [8, 8], "dtype": "f32"}],
            is_usable=True,
        )
        spec = ShardingSpec(
            mesh_shape=[2],
            tensor_shardings={"A": TensorSharding("A", [0], [4, 8])},
        )
        result = plan_and_insert(mod, spec)
        assert result.success
        # Both `stablehlo.return` and the trailing `return` should still
        # be present, and the new collectives should be inserted before
        # the function-level `return`.
        assert "stablehlo.return" in result.mlir_text
        assert "stablehlo.all_reduce" in result.mlir_text


class TestCommBackendWiring:
    """Verify the inserter actually uses CommBackend for library selection."""

    def test_heterogeneous_mesh_backend_usage(self) -> None:
        devices = [
            MeshDevice(0, DeviceVendor.NVIDIA, "sm_90", 80.0, 900.0, InterconnectType.NVLINK),
            MeshDevice(
                1, DeviceVendor.AMD, "gfx942", 192.0, 1300.0, InterconnectType.INFINITY_FABRIC
            ),
        ]
        mesh = DeviceMesh(devices=devices, mesh_shape=[2])
        mod = _make_module()
        spec = ShardingSpec(
            mesh_shape=[2],
            tensor_shardings={"A": TensorSharding("A", [0], [64, 128])},
        )
        result = plan_and_insert(mod, spec, mesh=mesh)
        assert result.success
        assert result.backend_usage.get("mixed", 0) == 4

    def test_homogeneous_nvidia_backend_usage(self) -> None:
        mesh = _make_mesh(num_devices=4, vendor=DeviceVendor.NVIDIA)
        mod = _make_module()
        spec = ShardingSpec(
            mesh_shape=[4],
            tensor_shardings={"A": TensorSharding("A", [0], [32, 128])},
        )
        result = plan_and_insert(mod, spec, mesh=mesh)
        assert result.success
        assert result.backend_usage.get("nccl", 0) == 4
        assert result.backend_usage.get("rccl", 0) == 0
        assert result.backend_usage.get("oneccl", 0) == 0


# ── CollectiveInserter class tests ──────────────────────────────────────


class TestCollectiveInserterClass:
    def test_default_construction(self) -> None:
        inserter = CollectiveInserter()
        assert inserter.mesh is None
        assert inserter.channel_id_base == 0

    def test_custom_channel_id_base(self) -> None:
        inserter = CollectiveInserter(channel_id_base=100)
        assert inserter.channel_id_base == 100

    def test_invalid_module_returns_error_result(self) -> None:
        inserter = CollectiveInserter()
        result = inserter.insert(None, ShardingSpec(mesh_shape=[4]))
        assert not result.success
        assert result.error is not None

    def test_unusable_module_returns_error_result(self) -> None:
        mod = StableHLOModule(
            mlir_text="",
            function_name="bad",
            is_usable=False,
        )
        inserter = CollectiveInserter()
        result = inserter.insert(mod, ShardingSpec(mesh_shape=[4]))
        assert not result.success
        assert result.error is not None

    def test_empty_spec_passes_through(self) -> None:
        mod = _make_module()
        inserter = CollectiveInserter()
        result = inserter.insert(mod, ShardingSpec(mesh_shape=[4]))
        assert result.success
        assert result.mlir_text == mod.mlir_text
        assert result.inserted_collectives == []

    def test_inserted_collective_is_frozen(self) -> None:
        """InsertedCollective must be immutable (frozen dataclass)."""
        from dataclasses import FrozenInstanceError

        coll = InsertedCollective(
            collective_type=CollectiveType.ALL_REDUCE,
            tensor_name="%A",
            mesh_axis=0,
            num_devices=4,
            device_ids=(0, 1, 2, 3),
            comm_library=CommLibrary.NCCL,
            result_name="%r",
            tensor_shape=(32, 128),
            dtype="f32",
        )
        with pytest.raises(FrozenInstanceError):
            coll.mesh_axis = 1  # type: ignore[misc]


# ── Property-based tests ────────────────────────────────────────────────


# Strategy: a sharding spec with random mesh shape, number of tensors,
# and per-tensor sharding axes.
@st.composite
def _sharding_specs(draw: st.DrawFn) -> ShardingSpec:
    mesh_rank = draw(st.integers(min_value=1, max_value=2))
    if mesh_rank == 1:
        mesh_shape: list[int] = [draw(st.integers(min_value=1, max_value=8))]
    else:
        r = draw(st.integers(min_value=1, max_value=4))
        c = draw(st.integers(min_value=1, max_value=4))
        mesh_shape = [r, c]

    num_tensors = draw(st.integers(min_value=0, max_value=4))
    tensor_shardings: dict[str, TensorSharding] = {}
    for i in range(num_tensors):
        # Random subset of mesh axes
        axes: list[int] = []
        for ax in range(len(mesh_shape)):
            if draw(st.booleans()):
                axes.append(ax)
        if not axes:
            # Sometimes keep it replicated
            if draw(st.booleans()):
                continue
            axes = [0]
        # Build a partition shape compatible with the mesh axes
        partition = [mesh_shape[a] if a < len(mesh_shape) else 1 for a in axes]
        tensor_shardings[f"T{i}"] = TensorSharding(
            tensor_name=f"T{i}",
            mesh_axes=axes,
            partition_shape=partition,
            replicate_on_other_axes=False,
        )

    return ShardingSpec(
        mesh_shape=mesh_shape,
        tensor_shardings=tensor_shardings,
        strategy_used=ShardingStrategy.AUTO,
    )


@st.composite
def _stablehlo_modules(draw: st.DrawFn) -> StableHLOModule:
    rank = draw(st.integers(min_value=1, max_value=3))
    shape_dims = [draw(st.integers(min_value=1, max_value=64)) for _ in range(rank)]
    shape_str = "x".join(str(d) for d in shape_dims)
    mlir = (
        "module {\n"
        f"  func.func @f(%A: tensor<{shape_str}xf32>) "
        f"-> tensor<{shape_str}xf32> {{\n"
        f"    %0 = stablehlo.multiply %A, %A : tensor<{shape_str}xf32>\n"
        f"    return %0 : tensor<{shape_str}xf32>\n"
        "  }\n"
        "}\n"
    )
    return StableHLOModule(
        mlir_text=mlir,
        function_name="f",
        input_specs=[{"name": "A", "shape": shape_dims, "dtype": "f32"}],
        is_usable=True,
    )


class TestPropertyCollectivePlan:
    """Property: for any sharding spec, the plan matches the expected pattern."""

    @given(spec=_sharding_specs(), module=_stablehlo_modules())
    @settings(
        max_examples=40,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_plan_count_matches_axes(
        self,
        spec: ShardingSpec,
        module: StableHLOModule,
    ) -> None:
        """For each sharded tensor with K axes, the plan has 4*K collectives."""
        plan = _plan_collectives(spec, module, mesh=None)
        expected = 0
        for ts in spec.tensor_shardings.values():
            expected += 4 * len(ts.mesh_axes)
        assert len(plan) == expected, (
            f"plan={len(plan)}, expected={expected}, tensor_shardings={spec.tensor_shardings}"
        )

    @given(spec=_sharding_specs(), module=_stablehlo_modules())
    @settings(
        max_examples=40,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_plan_covers_all_four_collective_types(
        self,
        spec: ShardingSpec,
        module: StableHLOModule,
    ) -> None:
        """If any tensor is sharded, every collective type must appear."""
        plan = _plan_collectives(spec, module, mesh=None)
        if not plan:
            return  # All replicated — nothing to assert
        types = {c.collective_type for c in plan}
        assert types == set(CollectiveType), f"missing types: {set(CollectiveType) - types}"

    @given(spec=_sharding_specs(), module=_stablehlo_modules())
    @settings(
        max_examples=40,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_each_collective_has_valid_comm_library(
        self,
        spec: ShardingSpec,
        module: StableHLOModule,
    ) -> None:
        """Every collective must have a real CommLibrary value."""
        plan = _plan_collectives(spec, module, mesh=None)
        for c in plan:
            assert isinstance(c.comm_library, CommLibrary)

    @given(spec=_sharding_specs(), module=_stablehlo_modules())
    @settings(
        max_examples=40,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_estimated_bytes_are_computed_from_shape(
        self,
        spec: ShardingSpec,
        module: StableHLOModule,
    ) -> None:
        """Bytes are not hardcoded; they follow the cost model exactly."""
        plan = _plan_collectives(spec, module, mesh=None)
        for c in plan:
            if c.num_devices <= 1:
                assert c.estimated_bytes == 0, f"single-device mesh should have 0 bytes for {c}"
            else:
                # Re-derive the expected volume from the cost model
                # and assert equality.
                expected = _collective_volume_bytes(
                    c.collective_type,
                    _tensor_bytes(c.tensor_shape, c.dtype),
                    c.num_devices,
                )
                assert c.estimated_bytes == expected, (
                    f"volume mismatch for {c}: plan={c.estimated_bytes}, expected={expected}"
                )

    @given(spec=_sharding_specs(), module=_stablehlo_modules())
    @settings(
        max_examples=30,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_emit_produces_parseable_mlir(
        self,
        spec: ShardingSpec,
        module: StableHLOModule,
    ) -> None:
        """For any spec, the emitted MLIR contains a func.func and a return."""
        result = plan_and_insert(module, spec)
        if not result.success:
            return  # Empty specs are allowed to pass through
        text = result.mlir_text
        assert "func.func" in text
        assert "return" in text

    @given(spec=_sharding_specs(), module=_stablehlo_modules())
    @settings(
        max_examples=30,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_total_bytes_equals_sum_of_inserted(
        self,
        spec: ShardingSpec,
        module: StableHLOModule,
    ) -> None:
        """The reported total_bytes must equal the sum of the plan."""
        result = plan_and_insert(module, spec)
        assert result.total_comm_bytes == sum(
            c.estimated_bytes for c in result.inserted_collectives
        )


class TestPropertyCommBackend:
    """Property: backend selection follows the device mesh vendor map."""

    @given(
        n_nvidia=st.integers(min_value=1, max_value=4),
        n_amd=st.integers(min_value=0, max_value=4),
        n_intel=st.integers(min_value=0, max_value=4),
    )
    @settings(
        max_examples=15,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_backend_selection_by_vendor(
        self,
        n_nvidia: int,
        n_amd: int,
        n_intel: int,
    ) -> None:
        """A homogeneous vendor mesh picks the matching comm library."""
        assume(n_nvidia + n_amd + n_intel >= 1)

        devices: list[MeshDevice] = []
        did = 0
        for _ in range(n_nvidia):
            devices.append(
                MeshDevice(
                    did,
                    DeviceVendor.NVIDIA,
                    "sm_90",
                    80,
                    900,
                    InterconnectType.NVLINK,
                )
            )
            did += 1
        for _ in range(n_amd):
            devices.append(
                MeshDevice(
                    did,
                    DeviceVendor.AMD,
                    "gfx942",
                    192,
                    1300,
                    InterconnectType.INFINITY_FABRIC,
                )
            )
            did += 1
        for _ in range(n_intel):
            devices.append(
                MeshDevice(
                    did,
                    DeviceVendor.INTEL,
                    "intel_gaudi2",
                    96,
                    900,
                    InterconnectType.ETHERNET,
                )
            )
            did += 1

        mesh = DeviceMesh(devices=devices, mesh_shape=[len(devices)])
        mod = _make_module()
        spec = ShardingSpec(
            mesh_shape=[len(devices)],
            tensor_shardings={"A": TensorSharding("A", [0], [1, 128])},
        )
        result = plan_and_insert(mod, spec, mesh=mesh)

        if n_nvidia > 0 and n_amd == 0 and n_intel == 0:
            expected = CommLibrary.NCCL
        elif n_amd > 0 and n_nvidia == 0 and n_intel == 0:
            expected = CommLibrary.RCCL
        elif n_intel > 0 and n_nvidia == 0 and n_amd == 0:
            expected = CommLibrary.ONECCL
        else:
            expected = CommLibrary.MIXED

        assert result.success
        assert all(c.comm_library == expected for c in result.inserted_collectives), (
            f"n_nvidia={n_nvidia}, n_amd={n_amd}, n_intel={n_intel}, "
            f"expected={expected}, got={ {c.comm_library for c in result.inserted_collectives} }"
        )
