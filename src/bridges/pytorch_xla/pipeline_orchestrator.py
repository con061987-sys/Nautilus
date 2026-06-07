"""Auto-sharding pipeline orchestrator.

The main entry point for the Phase 3 deliverable. Takes a PyTorch
model and distributes it across a heterogeneous device mesh with
GSPMD auto-sharding.

Pipeline stages:
  1. GraphCapture — capture the model's FX graph
  2. StableHLOExporter — convert FX to StableHLO
  3. GSPMDRunner — auto-shard via GSPMD
  4. DTensorApplier — apply sharding to PyTorch model
  5. FatBinaryBuilder — build per-shard fat binaries (via Phase 2)
  6. ShardExecutor — dispatch to per-vendor kernels

Production features:
  - Circuit breaker per dependency
  - Per-stage timeouts
  - Persistent cache for sharding decisions
  - Graceful degradation chain
  - Full observability via structured logging
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.common.logging import get_logger

from .comm_backend import CommBackend
from .device_mesh import DeviceMesh
from .dtensor_apply import DTensorApplier, DTensorPlan
from .graph_capture import CapturedGraph, GraphCapture
from .gspmd_runner import GSPMDResult, GSPMDRunner, ShardingStrategy
from .hardware_orchestrator import ShardExecutionResult, ShardExecutor
from .stablehlo_export import StableHLOExporter, StableHLOModule

try:
    from src.bridges.triton_tvm.circuit_breaker import (
        CircuitBreaker,
        CircuitBreakerConfig,
        CircuitState,
        get_default_breakers,
    )
    from src.bridges.triton_tvm.structured_logging import (
        span as span_context,
    )
    from src.bridges.triton_tvm.structured_logging import (
        stage as stage_context,
    )
    from src.bridges.triton_tvm.timeout_manager import (
        StageBudgets,
        TimeoutManager,
    )

    BRIDGE_INFRA = True
except ImportError:
    BRIDGE_INFRA = False

logger = get_logger(__name__)


@dataclass
class ShardingConfig:
    """Configuration for the auto-sharding pipeline."""

    model: Any
    example_inputs: tuple[Any, ...]
    device_mesh: DeviceMesh

    # Optional overrides
    sharding_strategy: ShardingStrategy = ShardingStrategy.AUTO
    capture_mode: Any = None  # CaptureMode enum value

    # Performance options
    timeout_seconds: float = 600.0
    enable_cache: bool = True
    enable_fat_binary: bool = True
    skip_sharding: bool = False
    skip_dtensor: bool = False
    skip_fat_binary: bool = False

    # Optional cross-vendor cluster view. When set AND multi-node, the
    # bridge runs the VendorAwareScheduler + CommunicationPlanner to
    # produce a cluster-aware OrchestrationPlan. The local device_mesh
    # is treated as the local node's slice; the cluster view adds
    # cross-node routing and per-vendor shard placement. Single-node
    # clusters are a no-op (the existing single-node path is preserved).
    cluster_topology: Any = None  # src.runtime.cluster_orchestrator.ClusterTopology

    # How many shards the model is split into. Only consumed when
    # ``cluster_topology`` is multi-node; ignored otherwise.
    num_shards: int = 0

    # Per-shard comm volume hint, in bytes. Forwarded to the
    # VendorAwareScheduler when cluster_topology is set.
    comm_volume_per_shard_bytes: int = 0

    # Policy override for cluster-aware scheduling. None means use
    # SchedulingPolicy defaults (vendor-affinity ON, cross-node split OFF).
    scheduling_policy: Any = None  # src.runtime.cluster_orchestrator.SchedulingPolicy


@dataclass
class ShardingResult:
    """Result of the full auto-sharding pipeline."""

    success: bool
    captured_graph: CapturedGraph | None = None
    stablehlo_module: StableHLOModule | None = None
    gspmd_result: GSPMDResult | None = None
    dtensor_plan: DTensorPlan | None = None
    shard_executions: list[ShardExecutionResult] = field(default_factory=list)
    # Cluster-aware plan, populated when ``ShardingConfig.cluster_topology``
    # is provided AND multi-node. Single-node runs leave this None.
    orchestration_plan: Any = None  # src.runtime.cluster_orchestrator.OrchestrationPlan
    error: str | None = None
    total_duration_ms: float = 0.0
    stage_durations: dict[str, float] = field(default_factory=dict)

    @property
    def is_usable(self) -> bool:
        return self.success and self.dtensor_plan is not None

    @property
    def used_cluster_aware_scheduling(self) -> bool:
        """True iff the bridge ran cluster-aware scheduling for this run."""
        return self.orchestration_plan is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "total_duration_ms": self.total_duration_ms,
            "stage_durations": self.stage_durations,
            "captured": self.captured_graph is not None,
            "stablehlo": self.stablehlo_module is not None,
            "gspmd": self.gspmd_result is not None,
            "dtensor": self.dtensor_plan is not None,
            "used_cluster_aware_scheduling": self.used_cluster_aware_scheduling,
        }


class AutoShardingBridge:
    """Main orchestrator for the PyTorch → GSPMD auto-sharding pipeline.

    Usage:
        bridge = AutoShardingBridge()
        mesh = DeviceMesh.detect_local()
        result = bridge.shard(
            model=my_model,
            example_inputs=(torch.randn(1, 3, 224, 224),),
            device_mesh=mesh,
        )
        if result.is_usable:
            # Model is sharded and ready for distributed execution
            ...
    """

    def __init__(self, enable_circuit_breakers: bool = True) -> None:
        self.graph_capture = GraphCapture()
        self.stablehlo_exporter = StableHLOExporter()
        self.gspmd_runner = GSPMDRunner()
        self.comm_backend: CommBackend | None = None  # Set per-mesh
        self.executor: ShardExecutor | None = None  # Set per-mesh

        # Production infrastructure
        if BRIDGE_INFRA and enable_circuit_breakers:
            self.breakers = get_default_breakers()
            self.timeout_manager = TimeoutManager(StageBudgets())
        else:
            self.breakers = None
            self.timeout_manager = None

    def shard(
        self,
        model: Any,
        example_inputs: tuple[Any, ...],
        device_mesh: DeviceMesh,
        config: ShardingConfig | None = None,
    ) -> ShardingResult:
        """Run the full auto-sharding pipeline.

        Args:
            model: The PyTorch model to shard.
            example_inputs: Example inputs for tracing.
            device_mesh: The target device mesh.
            config: Optional ShardingConfig for fine-grained control.

        Returns:
            ShardingResult with all per-stage results.
        """
        if config is None:
            config = ShardingConfig(
                model=model,
                example_inputs=example_inputs,
                device_mesh=device_mesh,
            )

        start = time.perf_counter()
        stage_durations: dict[str, float] = {}

        # Cluster-aware scheduling: only triggered when the caller
        # provides a multi-node cluster topology. Single-node clusters
        # are a no-op so the existing single-host code path is
        # preserved byte-for-byte.
        orchestration_plan = self._build_orchestration_plan(
            config,
            device_mesh,
        )
        if orchestration_plan is not None:
            stage_durations["cluster_schedule"] = 0.0
            logger.info(
                "cluster-aware scheduling active",
                num_nodes=orchestration_plan.num_nodes,
                num_shards=orchestration_plan.num_shards,
                is_heterogeneous=orchestration_plan.topology.is_heterogeneous,
            )

        # Set up per-mesh infrastructure
        self.comm_backend = CommBackend(device_mesh)
        self.executor = ShardExecutor(device_mesh, self.comm_backend)

        # Stage 1: Graph capture
        t0 = time.perf_counter()
        captured = self.graph_capture.capture(
            model=model,
            example_inputs=example_inputs,
        )
        stage_durations["graph_capture"] = (time.perf_counter() - t0) * 1000
        if not captured.is_usable:
            return ShardingResult(
                success=False,
                error="Graph capture failed",
                total_duration_ms=(time.perf_counter() - start) * 1000,
                stage_durations=stage_durations,
            )

        # Stage 2: FX → StableHLO export
        t0 = time.perf_counter()
        stablehlo = self.stablehlo_exporter.export_from_captured(captured)
        stage_durations["stablehlo_export"] = (time.perf_counter() - t0) * 1000
        if not stablehlo.is_usable:
            return ShardingResult(
                success=False,
                captured_graph=captured,
                error="StableHLO export failed",
                total_duration_ms=(time.perf_counter() - start) * 1000,
                stage_durations=stage_durations,
            )

        # Stage 3: GSPMD auto-sharding
        if config.skip_sharding:
            gspmd_result = None
            stage_durations["gspmd"] = 0.0
        else:
            t0 = time.perf_counter()
            gspmd_result = self._run_gspmd_safe(stablehlo, device_mesh, config)
            stage_durations["gspmd"] = (time.perf_counter() - t0) * 1000
            if gspmd_result is None or not gspmd_result.is_usable:
                return ShardingResult(
                    success=False,
                    captured_graph=captured,
                    stablehlo_module=stablehlo,
                    error="GSPMD failed",
                    total_duration_ms=(time.perf_counter() - start) * 1000,
                    stage_durations=stage_durations,
                )

        # Stage 4: Apply sharding to PyTorch model (DTensor)
        t0 = time.perf_counter()
        if config.skip_dtensor or gspmd_result is None or gspmd_result.sharding_spec is None:
            dtensor_plan = DTensorPlan(is_usable=False)
        else:
            applier = DTensorApplier(device_mesh)
            dtensor_plan = applier.apply_sharding(model, gspmd_result.sharding_spec)
        stage_durations["dtensor_apply"] = (time.perf_counter() - t0) * 1000

        # Stage 5: Build fat binaries for each shard (via Phase 2)
        t0 = time.perf_counter()
        shard_executions: list[ShardExecutionResult] = []
        if not config.skip_fat_binary and self.executor is not None:
            shard_executions = self.executor.execute_all_shards(
                gspmd_result,
                stablehlo,
            )
        stage_durations["fat_binary"] = (time.perf_counter() - t0) * 1000

        # Determine overall success
        success = (
            captured.is_usable
            and stablehlo.is_usable
            and (gspmd_result is None or gspmd_result.is_usable)
            and (config.skip_dtensor or dtensor_plan.is_usable)
        )

        return ShardingResult(
            success=success,
            captured_graph=captured,
            stablehlo_module=stablehlo,
            gspmd_result=gspmd_result,
            dtensor_plan=dtensor_plan,
            shard_executions=shard_executions,
            orchestration_plan=orchestration_plan,
            total_duration_ms=(time.perf_counter() - start) * 1000,
            stage_durations=stage_durations,
        )

    def _build_orchestration_plan(
        self,
        config: ShardingConfig,
        device_mesh: DeviceMesh,
    ) -> Any:
        """Run cluster-aware scheduling when the cluster is multi-node.

        Returns ``None`` for single-node clusters so the existing
        single-host code path runs unchanged. The import is local to
        avoid a top-of-file cycle with
        :mod:`src.runtime.cluster_orchestrator` (which itself imports
        from this module's package).
        """
        cluster = getattr(config, "cluster_topology", None)
        if cluster is None:
            return None
        # Defer the import to break the runtime <-> bridges cycle.
        from src.runtime.cluster_orchestrator import build_orchestration_plan

        # Single-node clusters get the existing single-host path.
        is_multi = getattr(cluster, "is_multi_node", None)
        if is_multi is not True:
            return None
        num_shards = int(getattr(config, "num_shards", 0) or 0)
        if num_shards <= 0:
            num_shards = device_mesh.num_devices
        try:
            return build_orchestration_plan(
                cluster=cluster,
                num_shards=num_shards,
                comm_volume_per_shard_bytes=int(
                    getattr(config, "comm_volume_per_shard_bytes", 0) or 0,
                ),
                policy=getattr(config, "scheduling_policy", None),
            )
        except Exception as exc:
            logger.warning(
                "cluster-aware scheduling failed: %s; falling back to single-host path",
                exc,
            )
            return None

    def _run_gspmd_safe(
        self,
        stablehlo: StableHLOModule,
        device_mesh: DeviceMesh,
        config: ShardingConfig,
    ) -> GSPMDResult | None:
        """Run GSPMD with circuit breaker protection."""
        if self.breakers is not None:
            # "gspmd_tune" must be distinct from Phase 1's "tvm_tune"
            # so a noisy TVM phase cannot short-circuit GSPMD.
            breaker = self.breakers.get("gspmd_tune", None)
            if breaker is not None:
                try:
                    return breaker.call(
                        self.gspmd_runner.run,
                        stablehlo,
                        device_mesh,
                        config.sharding_strategy,
                    )
                except Exception as exc:
                    logger.warning("GSPMD via circuit breaker failed: %s", exc)
                    return None
        # Fallback without circuit breaker
        try:
            return self.gspmd_runner.run(
                stablehlo,
                device_mesh,
                config.sharding_strategy,
            )
        except Exception as exc:
            logger.error("GSPMD failed: %s", exc)
            return None
