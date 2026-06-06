"""Tests for the DTensor apply module."""

from __future__ import annotations

import pytest

from src.bridges.pytorch_xla.device_mesh import (
    DeviceMesh,
    DeviceVendor,
    InterconnectType,
    MeshDevice,
)
from src.bridges.pytorch_xla.dtensor_apply import (
    DTensorApplier,
    DTensorPlan,
)
from src.bridges.pytorch_xla.gspmd_runner import (
    ShardingSpec,
    ShardingStrategy,
    TensorSharding,
)


def make_mesh() -> DeviceMesh:
    """Create a simple test mesh."""
    devices = [
        MeshDevice(i, DeviceVendor.NVIDIA, "sm_90", 80.0, 989.0, InterconnectType.NVLINK)
        for i in range(4)
    ]
    return DeviceMesh(devices=devices, mesh_shape=[4])


def make_sharding_spec() -> ShardingSpec:
    """Create a test sharding spec."""
    return ShardingSpec(
        mesh_shape=[4],
        tensor_shardings={
            "weight": TensorSharding("weight", [0], [4], False),
            "bias": TensorSharding("bias", [], [], True),
        },
        strategy_used=ShardingStrategy.DATA_PARALLEL,
    )


class TestDTensorPlan:
    """Tests for the DTensorPlan class."""

    def test_plan_creation(self) -> None:
        """DTensorPlan should store placements."""
        plan = DTensorPlan(
            parameter_placements={"weight": ("shard_0",)},
            input_placements={"input": ("shard_0",)},
        )
        assert plan.total_params == 1
        assert plan.total_inputs == 1

    def test_plan_default(self) -> None:
        """Default plan should be empty and usable."""
        plan = DTensorPlan()
        assert plan.total_params == 0
        assert plan.is_usable is True


class TestDTensorApplier:
    """Tests for the DTensorApplier class."""

    def test_applier_init(self) -> None:
        """DTensorApplier should accept a device mesh."""
        mesh = make_mesh()
        applier = DTensorApplier(mesh)
        assert applier.device_mesh is mesh

    def test_apply_sharding_creates_plan(self) -> None:
        """apply_sharding should produce a DTensorPlan."""
        mesh = make_mesh()
        applier = DTensorApplier(mesh)
        spec = make_sharding_spec()

        # Create a simple model-like object
        class FakeModel:
            def named_parameters(self):
                return [("weight", object()), ("bias", object())]

        model = FakeModel()
        plan = applier.apply_sharding(model, spec)
        # Plan should record placements for params that are in the spec
        assert "weight" in plan.parameter_placements
        assert "bias" in plan.parameter_placements

    def test_placement_for_sharded_tensor(self) -> None:
        """A sharded tensor should get a Shard placement."""
        mesh = make_mesh()
        applier = DTensorApplier(mesh)
        spec = make_sharding_spec()
        placement = applier._placement_for_tensor("weight", spec)
        assert placement is not None
        # The first element should be a Shard for axis 0
        assert len(placement) > 0

    def test_placement_for_replicated_tensor(self) -> None:
        """A replicated tensor should get a Replicate placement."""
        mesh = make_mesh()
        applier = DTensorApplier(mesh)
        spec = make_sharding_spec()
        placement = applier._placement_for_tensor("bias", spec)
        assert placement is not None
        # All placements should be Replicate
        for p in placement:
            assert type(p).__name__ == "Replicate"

    def test_placement_for_unknown_tensor(self) -> None:
        """An unknown tensor should return None."""
        mesh = make_mesh()
        applier = DTensorApplier(mesh)
        spec = make_sharding_spec()
        placement = applier._placement_for_tensor("nonexistent", spec)
        assert placement is None

    def test_plan_includes_collectives(self) -> None:
        """The plan should include the collectives from the sharding spec."""
        mesh = make_mesh()
        applier = DTensorApplier(mesh)
        spec = make_sharding_spec()
        spec.inserted_collectives = [{"type": "all-reduce", "op": "sum"}]

        class FakeModel:
            def named_parameters(self):
                return []

        plan = applier.apply_sharding(FakeModel(), spec)
        assert len(plan.collective_insertions) == 1
        assert plan.collective_insertions[0]["type"] == "all-reduce"
