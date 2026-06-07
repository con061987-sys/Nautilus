"""Sharding spec → PyTorch DTensor conversion.

After GSPMD produces a sharding specification, this module applies
it to the actual PyTorch model parameters and inputs, converting
them to PyTorch DTensors that are distributed across the device
mesh.

DTensor is PyTorch's native distributed tensor type (similar to
JAX's pjit). It provides:
  - Sharding annotations on each tensor
  - Automatic collective communication
  - Compatible with torch.distributed

This module bridges GSPMD's sharding output to DTensor's API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.common.logging import get_logger

try:
    from torch.distributed._tensor import (
        DTensor,
        Replicate,
        Shard,
    )

    DTENSOR_AVAILABLE = True
except ImportError:
    DTENSOR_AVAILABLE = False

logger = get_logger(__name__)


@dataclass
class DTensorPlan:
    """Plan for converting a model to DTensor-based distributed execution."""

    parameter_placements: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    input_placements: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    collective_insertions: list[dict[str, Any]] = field(default_factory=list)
    conversion_time_ms: float = 0.0
    is_usable: bool = True

    @property
    def total_params(self) -> int:
        return len(self.parameter_placements)

    @property
    def total_inputs(self) -> int:
        return len(self.input_placements)


class DTensorApplier:
    """Applies GSPMD sharding specs to PyTorch models via DTensor.

    Usage:
        applier = DTensorApplier(device_mesh)
        plan = applier.apply_sharding(model, sharding_spec)
        if plan.is_usable:
            # model is now DTensor-distributed across the mesh
            ...
    """

    def __init__(self, device_mesh: Any) -> None:
        self.device_mesh = device_mesh

    def apply_sharding(
        self,
        model: Any,
        sharding_spec: Any,
    ) -> DTensorPlan:
        """Apply a sharding spec to a PyTorch model.

        Args:
            model: The PyTorch model to shard.
            sharding_spec: ShardingSpec from GSPMDRunner.

        Returns:
            DTensorPlan with the per-tensor placements.
        """
        import time

        start = time.perf_counter()

        if not DTENSOR_AVAILABLE:
            return DTensorPlan(
                is_usable=False,
                conversion_time_ms=(time.perf_counter() - start) * 1000,
            )

        plan = DTensorPlan()

        # Apply sharding to each parameter
        try:
            for name, _param in model.named_parameters():
                placement = self._placement_for_tensor(
                    name,
                    sharding_spec,
                )
                if placement is not None:
                    plan.parameter_placements[name] = placement
        except Exception as exc:
            logger.warning("Parameter sharding failed: %s", exc)

        # Apply sharding to each input (placeholder for runtime)
        for input_spec in getattr(sharding_spec, "_input_specs", []):
            name = input_spec.get("name", "input")
            placement = self._placement_for_tensor(name, sharding_spec)
            if placement is not None:
                plan.input_placements[name] = placement

        # Add collective insertions
        if hasattr(sharding_spec, "inserted_collectives"):
            plan.collective_insertions = list(sharding_spec.inserted_collectives)

        plan.conversion_time_ms = (time.perf_counter() - start) * 1000
        return plan

    def _placement_for_tensor(
        self,
        tensor_name: str,
        sharding_spec: Any,
    ) -> tuple[Any, ...] | None:
        """Get the DTensor placement for a specific tensor.

        Returns a tuple of placements (one per mesh dim) or None
        if the tensor isn't in the sharding spec.
        """
        from .gspmd_runner import ShardingSpec, TensorSharding

        # Find the tensor in the sharding spec
        tensor_sharding: TensorSharding | None = None
        if isinstance(sharding_spec, ShardingSpec):
            tensor_sharding = sharding_spec.tensor_shardings.get(tensor_name)

        if tensor_sharding is None:
            return None

        # Build the DTensor placements
        placements: list[Shard | Replicate] = []
        for axis in tensor_sharding.mesh_axes:
            placements.append(Shard(axis))
        if tensor_sharding.replicate_on_other_axes and tensor_sharding.mesh_axes:
            # Add replicate placements for non-sharded axes
            all_axes = set(range(len(sharding_spec.mesh_shape)))
            sharded_axes = set(tensor_sharding.mesh_axes)
            for _axis in sorted(all_axes - sharded_axes):
                placements.append(Replicate())
        if not placements:
            return (Replicate(),)
        return tuple(placements)

    def apply_to_model(self, model: Any, plan: DTensorPlan) -> Any:
        """Actually apply the plan to the model (converting to DTensor).

        This is a more invasive operation than apply_sharding — it
        modifies the model's parameters in-place. Use with care.

        Args:
            model: The PyTorch model to modify.
            plan: DTensorPlan from apply_sharding.

        Returns:
            The modified model.
        """
        if not plan.is_usable or not DTENSOR_AVAILABLE:
            return model

        try:
            for name, param in model.named_parameters():
                if name not in plan.parameter_placements:
                    continue
                placement = plan.parameter_placements[name]
                # Convert parameter to DTensor
                dtensor = DTensor.from_local(
                    param.data,
                    self.device_mesh,
                    list(placement),
                    run_check=False,
                )
                param.data = dtensor
        except Exception as exc:
            logger.warning("apply_to_model failed: %s", exc)
        return model
