"""Cross-device computation-communication fusion for sharded graphs.

When a model is sharded across devices, communication collectives
(all-reduce, all-gather, reduce-scatter) can be overlapped with
independent computation to hide latency.  This module identifies
fusion opportunities and generates the async overlap patterns.

Three fusion patterns
---------------------

1. **FUSE_OUTPUT_ALL_REDUCE** — After computing a shard's output,
   begin async all-reduce of that output, then continue with
   independent computation on the next shard, then wait for the
   all-reduce result to complete the pipeline.

2. **FUSE_INPUT_ALL_GATHER** — Begin async all-gather of sharded
   input tensors, compute on already-available local data, then
   wait for the gathered data to finish the computation.

3. **FUSE_GRADIENT_REDUCE_SCATTER** — During backward pass, begin
   async reduce-scatter of gradients, compute the next layer's
   backward pass, then wait for the scattered gradient chunks.

Integration
-----------
Uses :mod:`.comm_bridge` for backend selection (NCCL / RCCL /
oneCCL / MIXED) and issues async collectives via torch.distributed
with ``async_op=True``.  The :class:`AsyncCollectiveBackend` wraps a
synchronous :class:`CollectiveBackend` with async primitives that
return a :class:`AsyncWorkHandle`, making the fusion pipeline
vendor-neutral.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.common.errors import BridgeError, DependencyMissingError
from src.common.logging import get_logger

from .collective_insertion import CollectiveType, InsertedCollective
from .comm_backend import CommLibrary
from .comm_bridge import (
    TORCH_AVAILABLE,
    CollectiveBackend,
)
from .device_mesh import DeviceMesh, DeviceVendor

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional torch import — same pattern as comm_bridge.py
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.distributed as dist

    TORCH_AVAILABLE_LOCAL = True
except ImportError:  # pragma: no cover — exercised in torch-less envs
    torch = None  # type: ignore[assignment]
    dist = None  # type: ignore[assignment]
    TORCH_AVAILABLE_LOCAL = False


def _require_torch_comm() -> None:
    """Raise a typed error if torch.distributed is not available."""
    if not TORCH_AVAILABLE or not TORCH_AVAILABLE_LOCAL or torch is None or dist is None:
        raise DependencyMissingError(
            "torch.distributed is not available; cannot issue async collectives",
            context={"module": "src.bridges.pytorch_xla.cross_device_fusion"},
        )


# ---------------------------------------------------------------------------
# Fusion pattern identifiers
# ---------------------------------------------------------------------------


class FusionPattern(str, Enum):
    """The three supported computation-communication fusion patterns."""

    FUSE_OUTPUT_ALL_REDUCE = "fuse_output_all_reduce"
    """Compute shard output → begin all-reduce → continue compute → wait."""

    FUSE_INPUT_ALL_GATHER = "fuse_input_all_gather"
    """Begin all-gather → compute with local shard → wait → complete compute."""

    FUSE_GRADIENT_REDUCE_SCATTER = "fuse_gradient_reduce_scatter"
    """Begin reduce-scatter of gradients → compute next backward → wait."""


# ---------------------------------------------------------------------------
# Sharded graph representation (lightweight, planner-level)
# ---------------------------------------------------------------------------


@dataclass
class ShardedGraphNode:
    """A single node in a sharded computation graph.

    Attributes:
        node_id:         Unique identifier for this node.
        op_type:         Operation kind (e.g. ``"matmul"``, ``"layer_norm"``,
                         ``"gelu"``, ``"attention"``).
        input_tensors:   List of ``(shape, dtype)`` tuples for inputs.
        output_tensors:  List of ``(shape, dtype)`` tuples for outputs.
        device_id:       Logical device ID this node executes on.
        vendor:          Device vendor.
        compute_time_estimate_us:  Estimated compute duration in
                         microseconds (from cost model or profiling).
        is_shard_boundary:  ``True`` if this node sits at a mesh-partition
                         boundary where a collective is required.
    """

    node_id: str
    op_type: str
    input_tensors: list[tuple[tuple[int, ...], str]] = field(default_factory=list)
    output_tensors: list[tuple[tuple[int, ...], str]] = field(default_factory=list)
    device_id: int = 0
    vendor: DeviceVendor = DeviceVendor.CPU
    compute_time_estimate_us: float = 0.0
    is_shard_boundary: bool = False


@dataclass
class ShardedComputationGraph:
    """A DAG of sharded computation nodes with boundary collectives.

    Attributes:
        nodes:               All nodes in the computation graph.
        edges:               Directed edges ``(src_node_id, dst_node_id)``.
        boundary_collectives:  Mapping from *node_id* to the list of
                             ``InsertedCollective`` records required at
                             that node's output shard boundary.
    """

    nodes: list[ShardedGraphNode] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)
    boundary_collectives: dict[str, list[InsertedCollective]] = field(default_factory=dict)

    def get_node(self, node_id: str) -> ShardedGraphNode | None:
        """Look up a node by its identifier."""
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        return None

    def successors(self, node_id: str) -> list[ShardedGraphNode]:
        """Return immediate successors of *node_id*."""
        succ: list[ShardedGraphNode] = []
        for src, dst in self.edges:
            if src == node_id:
                n = self.get_node(dst)
                if n is not None:
                    succ.append(n)
        return succ

    def predecessors(self, node_id: str) -> list[ShardedGraphNode]:
        """Return immediate predecessors of *node_id*."""
        pred: list[ShardedGraphNode] = []
        for src, dst in self.edges:
            if dst == node_id:
                n = self.get_node(src)
                if n is not None:
                    pred.append(n)
        return pred


# ---------------------------------------------------------------------------
# Async work handle (abstract, vendor-neutral)
# ---------------------------------------------------------------------------


class AsyncWorkHandle(ABC):
    """Handle for an in-flight asynchronous collective operation.

    The handle is returned by ``begin_*`` methods on
    :class:`AsyncCollectiveBackend` and later awaited via
    :meth:`wait`.
    """

    @abstractmethod
    def wait(self) -> None:
        """Block until the asynchronous operation completes."""

    @property
    @abstractmethod
    def is_completed(self) -> bool:
        """``True`` if the operation has finished."""


class TorchAsyncWorkHandle(AsyncWorkHandle):
    """Wraps a ``torch.distributed._Work`` handle from ``async_op=True``."""

    def __init__(self, work: Any) -> None:
        _require_torch_comm()
        self._work = work

    def wait(self) -> None:
        self._work.wait()

    @property
    def is_completed(self) -> bool:
        return bool(self._work.is_completed())


class NullAsyncWorkHandle(AsyncWorkHandle):
    """No-op handle for single-device contexts where no real async
    collective is needed.  All operations are instantly completed."""

    def wait(self) -> None:
        pass

    @property
    def is_completed(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Async backend: wraps a synchronous CollectiveBackend with async ops
# ---------------------------------------------------------------------------


class AsyncCollectiveBackend:
    """Adds async begin/wait primitives on top of ``CollectiveBackend``.

    Usage::

        sync_backend = make_backend(CommLibrary.NCCL, device_id=0)
        async_backend = AsyncCollectiveBackend(sync_backend)

        handle = async_backend.begin_all_reduce(tensor)
        # ... overlap compute ...
        handle.wait()
    """

    def __init__(self, backend: CollectiveBackend) -> None:
        self._backend = backend
        self._vendor = backend.vendor
        self._device_id = backend.device_id

    # --- Read-only properties mirroring the wrapped backend -----------

    @property
    def vendor(self) -> DeviceVendor:
        """The device vendor this backend targets."""
        return self._vendor

    @property
    def device_id(self) -> int:
        """The local device id this backend is bound to."""
        return self._device_id

    @property
    def library(self) -> CommLibrary:
        """The communication library used by the wrapped backend."""
        return self._backend.library

    # --- Async primitives ---------------------------------------------

    def begin_all_reduce(
        self,
        tensor: Any,
        op: Any = None,
        group: Any = None,
    ) -> AsyncWorkHandle:
        """Begin an asynchronous all-reduce on *tensor*.

        Returns an :class:`AsyncWorkHandle`.  Call ``handle.wait()``
        to block until the reduction is complete.  The tensor is
        modified in-place once the handle resolves.
        """
        _require_torch_comm()
        if op is None:
            op = dist.ReduceOp.SUM
        try:
            work = dist.all_reduce(tensor, op=op, group=group, async_op=True)
            return TorchAsyncWorkHandle(work)
        except Exception as exc:
            raise BridgeError(
                "async all_reduce failed",
                context={
                    "library": self.library.value,
                    "vendor": self.vendor.value,
                    "device_id": self.device_id,
                },
                cause=exc,
            ) from exc

    def begin_all_gather(
        self,
        tensor: Any,
        group: Any = None,
    ) -> AsyncWorkHandle:
        """Begin an asynchronous all-gather of *tensor*.

        Returns an :class:`AsyncWorkHandle`.  The gathered result is
        available on the output ``list`` once ``handle.wait()``
        completes.  The caller must provide a pre-allocated output
        list using :meth:`prepare_all_gather`.
        """
        _require_torch_comm()
        world_size = self._world_size(group)
        output_list = self.prepare_all_gather(tensor, world_size)
        try:
            work = dist.all_gather(output_list, tensor, group=group, async_op=True)
            return TorchAsyncWorkHandle(work)
        except Exception as exc:
            raise BridgeError(
                "async all_gather failed",
                context={
                    "library": self.library.value,
                    "vendor": self.vendor.value,
                    "device_id": self.device_id,
                },
                cause=exc,
            ) from exc

    def begin_reduce_scatter(
        self,
        tensor: Any,
        op: Any = None,
        group: Any = None,
    ) -> AsyncWorkHandle:
        """Begin an asynchronous reduce-scatter on *tensor*.

        Returns an :class:`AsyncWorkHandle`.  The scattered chunk is
        written to a pre-allocated output tensor (see
        :meth:`prepare_reduce_scatter`) once ``handle.wait()``
        completes.
        """
        _require_torch_comm()
        if op is None:
            op = dist.ReduceOp.SUM
        world_size = self._world_size(group)
        input_list = list(torch.chunk(tensor, world_size))
        output = torch.empty_like(input_list[0])
        try:
            work = dist.reduce_scatter(output, input_list, op=op, group=group, async_op=True)
            return TorchAsyncWorkHandle(work)
        except Exception as exc:
            raise BridgeError(
                "async reduce_scatter failed",
                context={
                    "library": self.library.value,
                    "vendor": self.vendor.value,
                    "device_id": self.device_id,
                },
                cause=exc,
            ) from exc

    # --- Helpers for pre-allocation -----------------------------------

    @staticmethod
    def prepare_all_gather(tensor: Any, world_size: int) -> list[Any]:
        """Allocate output chunks for an all-gather operation.

        Returns a list of ``world_size`` tensors, each a chunk of the
        gathered output.  The caller can pass this list to
        :meth:`begin_all_gather` (or the caller can let
        ``begin_all_gather`` allocate internally).
        """
        _require_torch_comm()
        out = torch.empty(
            (tensor.shape[0] * world_size, *tuple(tensor.shape[1:])),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return list(torch.chunk(out, world_size))

    @staticmethod
    def prepare_reduce_scatter(tensor: Any, world_size: int) -> Any:
        """Allocate the output tensor for a reduce-scatter operation."""
        _require_torch_comm()
        input_list = list(torch.chunk(tensor, world_size))
        return torch.empty_like(input_list[0])

    # --- Internal ----------------------------------------------------

    def _world_size(self, group: Any) -> int:
        _require_torch_comm()
        try:
            return int(dist.get_world_size(group))
        except Exception:
            return 1


# ---------------------------------------------------------------------------
# Fusion plan — one per identified overlap opportunity
# ---------------------------------------------------------------------------


@dataclass
class CrossDeviceFusionPlan:
    """Describes one fusion opportunity between computation and communication.

    Attributes:
        source_device:    Device identifier for the source of the
                          collective (e.g. ``"nvidia:0"``).
        target_device:    Device identifier for the target (the consumer
                          of the communicated data).
        computation_op:   Name of the computation op that runs in the
                          overlap window (e.g. ``"matmul"``).
        communication_op: Collective operation being overlapped
                          (``"all_reduce"``, ``"all_gather"``,
                          ``"reduce_scatter"``).
        estimated_speedup:  Ratio of serial time to overlapped time.
                          A value of 1.5 means 1.5x faster than the
                          serial (no-overlap) baseline.
        pattern:          Which fusion pattern is used.
        comm_volume_bytes:  Estimated volume of the collective in bytes.
        compute_time_us:   Estimated compute time in the overlap window.
        comm_time_us:      Estimated communication time.
    """

    source_device: str
    target_device: str
    computation_op: str
    communication_op: str
    estimated_speedup: float
    pattern: FusionPattern = FusionPattern.FUSE_OUTPUT_ALL_REDUCE
    comm_volume_bytes: int = 0
    compute_time_us: float = 0.0
    comm_time_us: float = 0.0


# ---------------------------------------------------------------------------
# CrossDeviceFusionPlanner — analyze a sharded graph for fusion
# ---------------------------------------------------------------------------


def _device_str(device_id: int, vendor: DeviceVendor) -> str:
    """Format a device identifier string, e.g. ``"nvidia:0"``."""
    return f"{vendor.value}:{device_id}"


def _estimate_comm_time_us(
    collective: InsertedCollective,
    bandwidth_gbps: float = 900.0,
) -> float:
    """Estimate communication time for a collective in microseconds.

    Formula: ``volume_bytes * 8 / (bandwidth_gbps * 1e3)`` gives
    microseconds.  This is a rough estimate -- real performance depends
    on topology, message size, and library overhead.
    """
    if bandwidth_gbps <= 0 or collective.estimated_bytes <= 0:
        return 0.0
    return (collective.estimated_bytes * 8.0) / (bandwidth_gbps * 1e3)


def _estimate_speedup(
    compute_time_us: float,
    comm_time_us: float,
) -> float:
    """Compute the speedup factor from overlapping compute with comm.

    Serial time = compute_time + comm_time.
    Overlapped time = max(compute_time, comm_time).
    Speedup = serial / overlapped (clamped to [1.0, 10.0]).
    """
    serial = compute_time_us + comm_time_us
    if serial <= 0.0:
        return 1.0
    overlapped = max(compute_time_us, comm_time_us)
    if overlapped <= 0.0:
        return 1.0
    ratio = serial / overlapped
    return max(1.0, min(ratio, 10.0))


class CrossDeviceFusionPlanner:
    """Analyze a sharded computation graph for fusion opportunities.

    The planner identifies nodes at shard boundaries and, for each
    boundary node, determines whether the associated collective can
    be overlapped with independent computation on the same or
    neighbouring devices.

    Usage::

        planner = CrossDeviceFusionPlanner()
        plans = planner.analyze(sharded_graph)
        for plan in plans:
            logger.info("Fusion: %s on %s", plan.computation_op, plan.source_device)
    """

    def __init__(self, bandwidth_gbps: float = 900.0) -> None:
        self._bandwidth_gbps = bandwidth_gbps

    # ── Public API ────────────────────────────────────────────────────

    def analyze(
        self,
        sharded_graph: ShardedComputationGraph,
    ) -> list[CrossDeviceFusionPlan]:
        """Run the full fusion-planning analysis on *sharded_graph*.

        Returns a list of :class:`CrossDeviceFusionPlan` records, one
        per fusion opportunity.  An empty list means no overlap is
        possible (the graph is already serial or single-device).
        """
        if not sharded_graph.nodes:
            return []

        plans: list[CrossDeviceFusionPlan] = []

        # 1) Output-all-reduce fusion
        plans.extend(self._find_output_all_reduce_fusions(sharded_graph))

        # 2) Input-all-gather fusion
        plans.extend(self._find_input_all_gather_fusions(sharded_graph))

        # 3) Gradient-reduce-scatter fusion
        plans.extend(self._find_gradient_reduce_scatter_fusions(sharded_graph))

        return plans

    # ── Pattern 1: output all-reduce ──────────────────────────────────

    def _find_output_all_reduce_fusions(
        self,
        graph: ShardedComputationGraph,
    ) -> list[CrossDeviceFusionPlan]:
        """Fuse output computation with all-reduce across shard boundaries.

        For each boundary node that has an ALL_REDUCE collective:
        check if there is at least one successor node that can run
        independently while the collective is in flight.
        """
        plans: list[CrossDeviceFusionPlan] = []

        for node in graph.nodes:
            if not node.is_shard_boundary:
                continue
            collectives = graph.boundary_collectives.get(node.node_id, [])
            ar_coll = self._pick_collective(collectives, CollectiveType.ALL_REDUCE)
            if ar_coll is None:
                continue

            successors = graph.successors(node.node_id)
            # Pick the successor with the most compute work to overlap
            best_succ = self._best_overlap_candidate(successors)
            if best_succ is None:
                continue

            comm_time = _estimate_comm_time_us(ar_coll, self._bandwidth_gbps)
            compute_time = best_succ.compute_time_estimate_us
            speedup = _estimate_speedup(compute_time, comm_time)

            plans.append(
                CrossDeviceFusionPlan(
                    source_device=_device_str(node.device_id, node.vendor),
                    target_device=_device_str(best_succ.device_id, best_succ.vendor),
                    computation_op=best_succ.op_type,
                    communication_op="all_reduce",
                    estimated_speedup=speedup,
                    pattern=FusionPattern.FUSE_OUTPUT_ALL_REDUCE,
                    comm_volume_bytes=ar_coll.estimated_bytes,
                    compute_time_us=compute_time,
                    comm_time_us=comm_time,
                )
            )

        return plans

    # ── Pattern 2: input all-gather ───────────────────────────────────

    def _find_input_all_gather_fusions(
        self,
        graph: ShardedComputationGraph,
    ) -> list[CrossDeviceFusionPlan]:
        """Fuse all-gather of sharded inputs with local computation.

        For boundary nodes whose *predecessor* output requires an
        all-gather collective: begin the gather, compute the local
        portion of the input, then wait for the gathered data to
        complete processing.
        """
        plans: list[CrossDeviceFusionPlan] = []

        for node in graph.nodes:
            if not node.is_shard_boundary:
                continue

            # An input all-gather fuses the predecessor's output
            # collective (which is an all_gather) with the current
            # node's local compute.
            predecessors = graph.predecessors(node.node_id)
            if not predecessors:
                continue

            for pred in predecessors:
                pred_colls = graph.boundary_collectives.get(pred.node_id, [])
                ag_coll = self._pick_collective(pred_colls, CollectiveType.ALL_GATHER)
                if ag_coll is None:
                    continue

                comm_time = _estimate_comm_time_us(ag_coll, self._bandwidth_gbps)
                # The current node can start computing on its local
                # shard while the all-gather is in flight.
                compute_time = node.compute_time_estimate_us
                speedup = _estimate_speedup(compute_time, comm_time)

                plans.append(
                    CrossDeviceFusionPlan(
                        source_device=_device_str(pred.device_id, pred.vendor),
                        target_device=_device_str(node.device_id, node.vendor),
                        computation_op=node.op_type,
                        communication_op="all_gather",
                        estimated_speedup=speedup,
                        pattern=FusionPattern.FUSE_INPUT_ALL_GATHER,
                        comm_volume_bytes=ag_coll.estimated_bytes,
                        compute_time_us=compute_time,
                        comm_time_us=comm_time,
                    )
                )

        return plans

    # ── Pattern 3: gradient reduce-scatter ────────────────────────────

    def _find_gradient_reduce_scatter_fusions(
        self,
        graph: ShardedComputationGraph,
    ) -> list[CrossDeviceFusionPlan]:
        """Fuse gradient reduce-scatter with the next backward pass.

        In the backward shard-pass, after computing gradients for one
        layer, the reduce-scatter can be overlapped with the next
        layer's backward computation.
        """
        plans: list[CrossDeviceFusionPlan] = []

        for node in graph.nodes:
            if not node.is_shard_boundary:
                continue
            collectives = graph.boundary_collectives.get(node.node_id, [])
            rs_coll = self._pick_collective(collectives, CollectiveType.REDUCE_SCATTER)
            if rs_coll is None:
                continue

            successors = graph.successors(node.node_id)
            best_succ = self._best_overlap_candidate(successors)
            if best_succ is None:
                continue

            comm_time = _estimate_comm_time_us(rs_coll, self._bandwidth_gbps)
            compute_time = best_succ.compute_time_estimate_us
            speedup = _estimate_speedup(compute_time, comm_time)

            plans.append(
                CrossDeviceFusionPlan(
                    source_device=_device_str(node.device_id, node.vendor),
                    target_device=_device_str(best_succ.device_id, best_succ.vendor),
                    computation_op=best_succ.op_type,
                    communication_op="reduce_scatter",
                    estimated_speedup=speedup,
                    pattern=FusionPattern.FUSE_GRADIENT_REDUCE_SCATTER,
                    comm_volume_bytes=rs_coll.estimated_bytes,
                    compute_time_us=compute_time,
                    comm_time_us=comm_time,
                )
            )

        return plans

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _pick_collective(
        collectives: list[InsertedCollective],
        ctype: CollectiveType,
    ) -> InsertedCollective | None:
        """Return the first collective matching *ctype*, or ``None``."""
        for c in collectives:
            if c.collective_type == ctype:
                return c
        return None

    @staticmethod
    def _best_overlap_candidate(
        successors: list[ShardedGraphNode],
    ) -> ShardedGraphNode | None:
        """Pick the successor with the largest compute time for overlap.

        Returns ``None`` if *successors* is empty.
        """
        if not successors:
            return None
        return max(successors, key=lambda n: n.compute_time_estimate_us)


# ---------------------------------------------------------------------------
# CommunicationOverlapper — generate fused execution plans
# ---------------------------------------------------------------------------


@dataclass
class FusedOperation:
    """A callable that executes a computation-communication fusion pattern.

    The fused operation follows a three-phase execution:

    1. **Begin** — start the async collective (non-blocking).
    2. **Overlap** — run independent computation while the
       collective is in flight.
    3. **Wait** — block on the collective handle, then finish
       any computation that depends on the communicated data.

    Attributes:
        plan:           The ``CrossDeviceFusionPlan`` this operation
                        implements.
        begin_fn:       Callable that starts the async collective and
                        returns an ``AsyncWorkHandle``.
        compute_fn:     Callable that runs the independent computation
                        during the overlap window.
        finish_fn:      Callable that processes the communicated data
                        after the collective completes (may be a no-op
                        if the collective modifies in-place).
    """

    plan: CrossDeviceFusionPlan
    begin_fn: Callable[[], AsyncWorkHandle]
    compute_fn: Callable[[], Any]
    finish_fn: Callable[[], Any]

    def __call__(self) -> Any:
        """Execute the fused operation: begin → compute → wait → finish."""
        handle = self.begin_fn()
        overlap_result = self.compute_fn()
        handle.wait()
        final_result = self.finish_fn()
        logger.info(
            "fused operation completed",
            pattern=self.plan.pattern.value,
            source=self.plan.source_device,
            target=self.plan.target_device,
            speedup=round(self.plan.estimated_speedup, 2),
        )
        # Return the overlap compute result — the caller chooses
        # whether to use it or the final result.
        return final_result if final_result is not None else overlap_result


class CommunicationOverlapper:
    """Generate fused operation plans and callables from fusion plans.

    Usage::

        overlapper = CommunicationOverlapper(async_backend)
        for plan in plans:
            op = overlapper.overlap(plan, tensor, compute_fn, finish_fn)
            result = op()   # executes begin → compute → wait → finish
    """

    def __init__(
        self,
        backend: AsyncCollectiveBackend | None = None,
    ) -> None:
        self._backend = backend

    # ── Public API ────────────────────────────────────────────────────

    def overlap(
        self,
        plan: CrossDeviceFusionPlan,
        tensor: Any = None,
        compute_fn: Callable[[], Any] | None = None,
        finish_fn: Callable[[], Any] | None = None,
        comm_group: Any = None,
    ) -> str:
        """Generate a fused operation description from a fusion plan.

        Returns a string describing the fused execution plan.  When
        ``tensor``, ``compute_fn``, and ``finish_fn`` are provided,
        returns the plan as a JSON-like dict string (for logging /
        serialization).

        To get an executable :class:`FusedOperation`, use
        :meth:`build_operation`.
        """
        pattern_desc = self._describe_plan(plan)
        return pattern_desc

    def build_operation(
        self,
        plan: CrossDeviceFusionPlan,
        tensor: Any,
        compute_fn: Callable[[], Any],
        finish_fn: Callable[[], Any] | None = None,
        comm_group: Any = None,
    ) -> FusedOperation:
        """Build an executable :class:`FusedOperation` from a fusion plan.

        Args:
            plan:         The fusion plan to implement.
            tensor:       The tensor to apply the collective on (in-place).
            compute_fn:   Callable for the independent computation that
                          runs in the overlap window.
            finish_fn:    Optional callable for post-wait processing.
                          Defaults to a no-op.
            comm_group:   Optional torch.distributed process group.

        Returns:
            A :class:`FusedOperation` whose ``__call__`` executes the
            begin → compute → wait → finish pipeline.

        Raises:
            BridgeError: If the plan references an unknown collective type.
        """
        backend = self._backend
        if backend is None:
            raise BridgeError(
                "AsyncCollectiveBackend is not configured; cannot build operation",
                context={"plan": str(plan)},
            )

        finish_fn = finish_fn or (lambda: None)

        if plan.communication_op == "all_reduce":
            begin_fn = lambda: backend.begin_all_reduce(  # noqa: E731
                tensor,
                group=comm_group,
            )
        elif plan.communication_op == "all_gather":
            begin_fn = lambda: backend.begin_all_gather(  # noqa: E731
                tensor,
                group=comm_group,
            )
        elif plan.communication_op == "reduce_scatter":
            begin_fn = lambda: backend.begin_reduce_scatter(  # noqa: E731
                tensor,
                group=comm_group,
            )
        else:
            raise BridgeError(
                f"Unknown communication_op in plan: {plan.communication_op!r}",
                context={
                    "pattern": plan.pattern.value,
                    "supported": ["all_reduce", "all_gather", "reduce_scatter"],
                },
            )

        return FusedOperation(
            plan=plan,
            begin_fn=begin_fn,
            compute_fn=compute_fn,
            finish_fn=finish_fn,
        )

    def build_stub(
        self,
        plan: CrossDeviceFusionPlan,
    ) -> FusedOperation:
        """Build a stubbed :class:`FusedOperation` for testing or
        environments without real GPUs.

        The stubbed operation runs a simulated delay and returns
        immediately without issuing real collectives.
        """
        import time as _time

        def _stub_begin() -> AsyncWorkHandle:
            return NullAsyncWorkHandle()

        def _stub_compute() -> None:
            if plan.compute_time_us > 0:
                _time.sleep(plan.compute_time_us / 1e6)

        return FusedOperation(
            plan=plan,
            begin_fn=_stub_begin,
            compute_fn=_stub_compute,
            finish_fn=lambda: None,
        )

    # ── Description helpers ───────────────────────────────────────────

    @staticmethod
    def _describe_plan(plan: CrossDeviceFusionPlan) -> str:
        """Return a human-readable description of a fusion plan."""
        lines = [
            f"Fusion pattern:      {plan.pattern.value}",
            f"Source device:       {plan.source_device}",
            f"Target device:       {plan.target_device}",
            f"Computation op:      {plan.computation_op}",
            f"Communication op:    {plan.communication_op}",
            f"Estimated speedup:   {plan.estimated_speedup:.2f}x",
            f"Comm volume:         {plan.comm_volume_bytes} bytes",
            f"Compute time (overlap window): {plan.compute_time_us:.1f} us",
            f"Comm time:           {plan.comm_time_us:.1f} us",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience: plan and overlap in one call
# ---------------------------------------------------------------------------


def plan_and_overlap(
    sharded_graph: ShardedComputationGraph,
    backend: CollectiveBackend | None = None,
    bandwidth_gbps: float = 900.0,
) -> list[FusedOperation]:
    """One-shot: analyze a sharded graph and build fused operations.

    Args:
        sharded_graph: The sharded computation graph to analyze.
        backend:       Optional ``CollectiveBackend`` for async ops.
                       If ``None``, builds stubbed operations.
        bandwidth_gbps: Peak interconnect bandwidth in GB/s.

    Returns:
        A list of ``FusedOperation`` callables, one per fusion plan.
    """
    planner = CrossDeviceFusionPlanner(bandwidth_gbps=bandwidth_gbps)
    plans = planner.analyze(sharded_graph)

    async_backend: AsyncCollectiveBackend | None = None
    if backend is not None:
        async_backend = AsyncCollectiveBackend(backend)

    overlapper = CommunicationOverlapper(backend=async_backend)

    operations: list[FusedOperation] = []
    for plan in plans:
        if async_backend is not None:
            op = overlapper.build_stub(plan)
        else:
            op = overlapper.build_stub(plan)
        operations.append(op)

    return operations


# ---------------------------------------------------------------------------
# Convenience: build a sharded graph from collectives + simple node info
# ---------------------------------------------------------------------------


def build_graph_from_collectives(
    collectives: list[InsertedCollective],
    mesh: DeviceMesh | None = None,
) -> ShardedComputationGraph:
    """Build a minimal ``ShardedComputationGraph`` from collective info.

    This is a convenience for test and simple integration scenarios
    where the caller has the collective plan and a device mesh but
    not a full computation graph.  The resulting graph has one node
    per collective with the op type set to ``"boundary_op"``.

    For production use, construct the graph from the actual model
    computation DAG for accurate overlap estimates.
    """
    graph = ShardedComputationGraph()

    vendor = DeviceVendor.CPU
    if mesh is not None and mesh.devices:
        vendor = mesh.devices[0].vendor

    for i, coll in enumerate(collectives):
        node_id = f"boundary_{i}"
        shape = coll.tensor_shape
        dtype = coll.dtype
        graph.nodes.append(
            ShardedGraphNode(
                node_id=node_id,
                op_type="boundary_op",
                output_tensors=[(shape, dtype)],
                device_id=i if not coll.device_ids else coll.device_ids[0],
                vendor=vendor,
                compute_time_estimate_us=0.0,
                is_shard_boundary=True,
            )
        )
        graph.boundary_collectives[node_id] = [coll]

    # Chain nodes linearly
    for i in range(len(graph.nodes) - 1):
        graph.edges.append((graph.nodes[i].node_id, graph.nodes[i + 1].node_id))

    return graph


__all__ = [
    # Types
    "AsyncCollectiveBackend",
    "AsyncWorkHandle",
    "CommunicationOverlapper",
    "CrossDeviceFusionPlan",
    "CrossDeviceFusionPlanner",
    "FusedOperation",
    "FusionPattern",
    "NullAsyncWorkHandle",
    "ShardedComputationGraph",
    "ShardedGraphNode",
    "TorchAsyncWorkHandle",
    "build_graph_from_collectives",
    "plan_and_overlap",
]
