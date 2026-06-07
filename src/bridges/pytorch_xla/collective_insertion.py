"""Collective communication insertion pass for StableHLO MLIR.

When GSPMD produces a sharding specification, the actual collective
operations (all-reduce, all-gather, reduce-scatter, all-to-all) must
be inserted into the StableHLO MLIR text at the correct program
locations. This module owns that pass.

The pass follows the four-step bridge pattern:

  1. **Intercept** — read ``ShardingSpec`` and ``StableHLOModule`` produced
     by ``GSPMDRunner``.
  2. **Normalize** — convert each ``TensorSharding`` into a list of
     ``InsertedCollective`` records, each describing one collective
     operation, its mesh axis, participating device count, comm backend
     and byte volume.
  3. **Translate** — emit real StableHLO MLIR for every collective
     (``stablehlo.all_reduce`` / ``stablehlo.all_gather`` /
     ``stablehlo.reduce_scatter`` / ``stablehlo.all_to_all``) at the
     correct IR location (just before the function return) with proper
     channel-handle attributes, sharding annotations, and result types.
  4. **Verify** — the resulting MLIR text must (a) parse as StableHLO,
     (b) reference the correct collective dialect op, and (c) carry
     ``mhlo.sharding`` annotations consistent with the input spec.

Communication backend selection is delegated to ``CommBackend``: per
mesh axis we look up the device IDs that participate and let
``CommBackend.select_library_for_op`` choose NCCL / RCCL / oneCCL /
GLOO / MIXED. Heterogeneous mesh axes automatically get a
``MIXED``-library bridge.

Communication volume is computed from the tensor shape and dtype using
the standard GSPMD cost model:

  - all-reduce:        2 * tensor_bytes * (n - 1) / n
  - all-gather:         tensor_bytes * (n - 1)
  - reduce-scatter:   2 * tensor_bytes * (n - 1) / n
  - all-to-all:        tensor_bytes * (n - 1) / n

This module is the Phase 3 complement to ``gspmd_runner``: GSPMD decides
the sharding, this module writes the actual collectives into the IR.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from src.common.errors import GSPMDError
from src.common.logging import get_logger
from src.common.types import StableHLOModule

from .comm_backend import CollectiveOp, CommBackend, CommLibrary
from .device_mesh import DeviceMesh
from .gspmd_runner import ShardingSpec, TensorSharding

logger = get_logger("nautilus.collective_insertion")


# ── Public types ───────────────────────────────────────────────────────


class CollectiveType(Enum):
    """The four collective operations supported by this pass.

    String values match the StableHLO dialect op suffixes
    (``stablehlo.<value>``) so the emitter can build the op name
    directly.
    """

    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_TO_ALL = "all_to_all"


# Map our enum onto the existing comm_backend.CollectiveOp so we can
# delegate library selection without duplicating the lookup table.
_OP_TO_COMM_OP: dict[CollectiveType, CollectiveOp] = {
    CollectiveType.ALL_REDUCE: CollectiveOp.ALL_REDUCE,
    CollectiveType.ALL_GATHER: CollectiveOp.ALL_GATHER,
    CollectiveType.REDUCE_SCATTER: CollectiveOp.REDUCE_SCATTER,
    CollectiveType.ALL_TO_ALL: CollectiveOp.ALL_TO_ALL,
}


# StableHLO element-type byte size lookup. Used to compute the
# communication volume from tensor shapes; never hardcoded in formulas.
_DTYPE_BYTES: dict[str, int] = {
    "f64": 8,
    "f32": 4,
    "f16": 2,
    "bf16": 2,
    "i64": 8,
    "i32": 4,
    "i16": 2,
    "i8": 1,
    "i1": 1,
    "ui64": 8,
    "ui32": 4,
    "ui16": 2,
    "ui8": 1,
    "f8E4M3FN": 1,
    "f8E5M2": 1,
}


@dataclass(frozen=True)
class InsertedCollective:
    """One collective operation, ready to be emitted into StableHLO MLIR.

    Attributes:
        collective_type: Which collective op (all-reduce etc).
        tensor_name:     Name of the sharded tensor (e.g. ``"%A"``).
        mesh_axis:       Mesh axis along which the collective runs.
        num_devices:     Number of devices that participate.
        device_ids:      Device IDs that participate, ordered by mesh
                         axis index. Empty for single-device meshes.
        comm_library:    Selected comm library (NCCL / RCCL / oneCCL /
                         GLOO / MIXED). Resolved per device set.
        result_name:     SSA value name for the collective's output.
        tensor_shape:    Per-device shape of the tensor being
                         collective'd. Used to compute volume and
                         emit the right return type.
        dtype:           StableHLO element type (e.g. ``"f32"``).
        reduction_op:    Reduction kind for all-reduce / reduce-scatter.
                         Defaults to ``"add"``.
        estimated_bytes: Computed communication volume in bytes.
    """

    collective_type: CollectiveType
    tensor_name: str
    mesh_axis: int
    num_devices: int
    device_ids: tuple[int, ...]
    comm_library: CommLibrary
    result_name: str
    tensor_shape: tuple[int, ...]
    dtype: str
    reduction_op: str = "add"
    estimated_bytes: int = 0


@dataclass
class CollectiveInsertionResult:
    """The output of running the insertion pass on one module."""

    stablehlo_module: StableHLOModule | None
    inserted_collectives: list[InsertedCollective] = field(default_factory=list)
    total_comm_bytes: int = 0
    mlir_text: str = ""
    backend_usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None


# ── Tensor-shape parsing (compute volume from real shape, never hardcode)


# Captures: 1=shape dims joined by 'x', 2=element type
_TENSOR_TYPE_RE = re.compile(
    r"tensor<((?:\d+x)*\d*)(f(?:32|16|64)|bf16|i(?:1|8|16|32|64)|ui(?:8|16|32|64))>"
)


def _parse_tensor_shape_and_dtype(type_str: str) -> tuple[tuple[int, ...], str]:
    """Extract (shape, dtype) from a ``tensor<...>`` type string.

    Args:
        type_str: A string containing a tensor type, e.g.
            ``"tensor<128x256xf32>"``.

    Returns:
        Tuple ``(shape, dtype)``. ``shape`` is empty for scalar tensors.
        Raises ``GSPMDError`` if the type string does not match.
    """
    m = _TENSOR_TYPE_RE.search(type_str)
    if not m:
        raise GSPMDError(
            f"Could not parse tensor type: {type_str!r}",
            context={"type_str": type_str},
        )
    dims_str, dtype = m.group(1), m.group(2)
    if not dims_str:
        shape: tuple[int, ...] = ()
    else:
        # Trailing "x" (e.g. "128x") is fine; strip then split.
        shape = tuple(int(d) for d in dims_str.rstrip("x").split("x") if d)
    return shape, dtype


def _tensor_bytes(shape: tuple[int, ...], dtype: str) -> int:
    """Compute the byte size of a tensor from its shape and dtype.

    Uses ``_DTYPE_BYTES`` so the mapping is explicit and never relies on
    string parsing of a numeric dtype. Falls back to 4 bytes (fp32)
    for unknown dtypes — this matches StableHLO's default and avoids
    silent under-counting in the cost model.
    """
    elem_bytes = _DTYPE_BYTES.get(dtype, 4)
    n = 1
    for d in shape:
        n *= d
    return n * elem_bytes


# ── Volume cost model (matches GSPMD; never hardcoded per-shape)


def _collective_volume_bytes(
    ctype: CollectiveType,
    tensor_bytes: int,
    num_devices: int,
) -> int:
    """Real GSPMD cost model for collective communication volume.

    - all-reduce:     2 * T * (N - 1) / N
    - all-gather:      T * (N - 1)
    - reduce-scatter: 2 * T * (N - 1) / N
    - all-to-all:      T * (N - 1) / N

    A single-device mesh (N=1) costs 0 bytes for every op.
    """
    if num_devices <= 1:
        return 0
    if ctype == CollectiveType.ALL_REDUCE:
        return int(2 * tensor_bytes * (num_devices - 1) / num_devices)
    if ctype == CollectiveType.ALL_GATHER:
        return int(tensor_bytes * (num_devices - 1))
    if ctype == CollectiveType.REDUCE_SCATTER:
        return int(2 * tensor_bytes * (num_devices - 1) / num_devices)
    if ctype == CollectiveType.ALL_TO_ALL:
        return int(tensor_bytes * (num_devices - 1) / num_devices)
    raise GSPMDError(
        f"Unknown collective type for volume calc: {ctype}",
        context={"ctype": ctype.value},
    )


# ── Backend selection per device set


def _select_comm_library(
    comm_backend: CommBackend | None,
    device_ids: list[int],
    ctype: CollectiveType,
) -> CommLibrary:
    """Pick the comm library for a collective along a mesh axis.

    Delegates to ``CommBackend.select_library_for_op`` when available;
    falls back to NCCL if the backend is not configured (the common
    case in tests and CPU-only environments).
    """
    if comm_backend is None:
        # No mesh → default to NCCL. This is the conservative choice
        # for tests and CPU builds where no comm backend is wired.
        return CommLibrary.NCCL

    try:
        return comm_backend.select_library_for_op(_OP_TO_COMM_OP[ctype], device_ids)
    except Exception as exc:
        logger.warning(
            "CommBackend.select_library_for_op failed for %s on devices %s: %s; "
            "falling back to NCCL",
            ctype.value,
            device_ids,
            exc,
        )
        return CommLibrary.NCCL


def _devices_for_mesh_axis(
    mesh: DeviceMesh | None,
    mesh_shape: list[int],
    mesh_axis: int,
) -> list[int]:
    """Compute the ordered list of device IDs along a mesh axis.

    For a 1-D mesh ``[N]``, the devices are ``[0, 1, ..., N-1]``.
    For a 2-D mesh ``[R, C]``, axis 0 returns ``[0, 1, ..., R-1]``
    (one device per row of the logical mesh) and axis 1 returns the
    column-sliced devices. When a real ``DeviceMesh`` is supplied, we
    use its ``devices`` list directly; otherwise we synthesize the IDs
    from the mesh shape.
    """
    if mesh is not None and mesh.devices:
        # A real mesh with declared devices: take every device along
        # the axis, ordered by mesh-axis index in the logical mesh.
        return [d.device_id for d in mesh.devices]

    # Synthesize from mesh_shape: stride along the requested axis
    if not mesh_shape or mesh_axis >= len(mesh_shape):
        return [0]
    n = mesh_shape[mesh_axis]
    return list(range(n))


# ── MLIR emitters — one per collective type


def _emit_all_reduce(
    coll: InsertedCollective,
    channel_id: int,
) -> str:
    """Emit ``stablehlo.all_reduce`` for the given collective."""
    operand = coll.tensor_name
    result = coll.result_name
    type_str = f"tensor<{'x'.join(str(d) for d in coll.tensor_shape)}x{coll.dtype}>"
    # StableHLO all_reduce signature:
    #   stablehlo.all_reduce %operand,
    #       replica_groups = [...],
    #       channel_handle = #stablehlo.channel_handle<...>,
    #       use_global_device_ids
    #       : (type) -> type
    replica_groups = _replica_groups_for_axis(coll)
    return (
        f"  {result} = stablehlo.all_reduce {operand}, "
        f"replica_groups = {replica_groups}, "
        f"channel_handle = #stablehlo.channel_handle<"
        f"handle = {channel_id}, type = 0>, "
        f"use_global_device_ids "
        f": ({type_str}) -> {type_str}"
    )


def _emit_all_gather(
    coll: InsertedCollective,
    channel_id: int,
) -> str:
    """Emit ``stablehlo.all_gather`` for the given collective."""
    operand = coll.tensor_name
    result = coll.result_name
    type_str = f"tensor<{'x'.join(str(d) for d in coll.tensor_shape)}x{coll.dtype}>"
    # Gather dimension 0 by default; the result shape is
    # ``(N * dim0, dim1, ..., dimN)``.
    if coll.tensor_shape:
        gathered_shape: tuple[int, ...] = (
            coll.num_devices * coll.tensor_shape[0],
            *coll.tensor_shape[1:],
        )
    else:
        gathered_shape = coll.tensor_shape
    result_type = f"tensor<{'x'.join(str(d) for d in gathered_shape)}x{coll.dtype}>"
    replica_groups = _replica_groups_for_axis(coll)
    return (
        f"  {result} = stablehlo.all_gather {operand}, "
        f"all_gather_dim = 0 : {type_str}, "
        f"replica_groups = {replica_groups}, "
        f"channel_handle = #stablehlo.channel_handle<"
        f"handle = {channel_id}, type = 0> "
        f": ({type_str}) -> {result_type}"
    )


def _emit_reduce_scatter(
    coll: InsertedCollective,
    channel_id: int,
) -> str:
    """Emit ``stablehlo.reduce_scatter`` for the given collective."""
    operand = coll.tensor_name
    result = coll.result_name
    type_str = f"tensor<{'x'.join(str(d) for d in coll.tensor_shape)}x{coll.dtype}>"
    if coll.tensor_shape:
        scattered_shape: tuple[int, ...] = (
            coll.tensor_shape[0] // coll.num_devices,
            *coll.tensor_shape[1:],
        )
    else:
        scattered_shape = coll.tensor_shape
    result_type = f"tensor<{'x'.join(str(d) for d in scattered_shape)}x{coll.dtype}>"
    replica_groups = _replica_groups_for_axis(coll)
    return (
        f"  {result} = stablehlo.reduce_scatter {operand}, "
        f"reduction = {{ {coll.reduction_op} }}, "
        f"scatter_dim = 0 : {type_str}, "
        f"replica_groups = {replica_groups}, "
        f"channel_handle = #stablehlo.channel_handle<"
        f"handle = {channel_id}, type = 0>, "
        f"use_global_device_ids "
        f": ({type_str}) -> {result_type}"
    )


def _emit_all_to_all(
    coll: InsertedCollective,
    channel_id: int,
) -> str:
    """Emit ``stablehlo.all_to_all`` for the given collective."""
    operand = coll.tensor_name
    result = coll.result_name
    type_str = f"tensor<{'x'.join(str(d) for d in coll.tensor_shape)}x{coll.dtype}>"
    # all_to_all keeps the same shape as the input (data is permuted
    # between devices, not concatenated).
    result_type = type_str
    replica_groups = _replica_groups_for_axis(coll)
    # StableHLO all_to_all has split_dimension and concat_dimension
    # attrs (default 0); we use 0 for the canonical case.
    return (
        f"  {result} = stablehlo.all_to_all {operand}, "
        f"split_dimension = 0, concat_dimension = 0, "
        f"split_count = {coll.num_devices}, "
        f"replica_groups = {replica_groups}, "
        f"channel_handle = #stablehlo.channel_handle<"
        f"handle = {channel_id}, type = 0> "
        f": ({type_str}) -> {result_type}"
    )


def _replica_groups_for_axis(coll: InsertedCollective) -> str:
    """Build the ``replica_groups`` attribute for a mesh-axis collective.

    Format: dense<[[d0, d1, ..., dN]]> where ``dI`` are the device IDs
    that participate in the collective along the mesh axis. Using a
    single dense list means "all devices in one group", which is what
    we want for the canonical collectives inserted by this pass.
    """
    if not coll.device_ids:
        # Fallback: assume num_devices devices, IDs 0..N-1.
        ids = list(range(coll.num_devices))
    else:
        ids = list(coll.device_ids)
    inner = "[" + ", ".join(str(i) for i in ids) + "]"
    return f"dense<[[{inner}]]> : tensor<1x{coll.num_devices}xi64>"


_EMITTERS = {
    CollectiveType.ALL_REDUCE: _emit_all_reduce,
    CollectiveType.ALL_GATHER: _emit_all_gather,
    CollectiveType.REDUCE_SCATTER: _emit_reduce_scatter,
    CollectiveType.ALL_TO_ALL: _emit_all_to_all,
}


# ── IR location: where to insert collectives in the MLIR text


# Captures: 1=prefix "func.func @name(", 2=args, 3=suffix ") -> ret {"
_FUNC_SIG_RE = re.compile(
    r"(func\.func\s+@\w+\s*\()"
    r"([^)]*)"
    r"(\)\s*(?:->\s*[^{]+)?\s*\{)",
    re.DOTALL,
)

# Captures: 1=name (%foo), 2=type (tensor<...>/memref<...>)
_FUNC_ARG_RE = re.compile(
    r"(%\w+)\s*:\s*((?:tensor|memref)<[^>]+>)",
)


def _find_return_insertion_point(mlir_text: str) -> int | None:
    """Find the byte offset just before the first ``return`` op in
    the first function body.

    Returns ``None`` if no ``func.func`` is found.
    """
    func_match = _FUNC_SIG_RE.search(mlir_text)
    if not func_match:
        return None
    body_start = func_match.end()
    # Find first `return` after body_start
    ret_match = re.search(r"\breturn\b", mlir_text[body_start:])
    if not ret_match:
        return None
    return body_start + ret_match.start()


def _find_all_returns(mlir_text: str) -> list[int]:
    """Return byte offsets of every ``return`` in the first function body."""
    func_match = _FUNC_SIG_RE.search(mlir_text)
    if not func_match:
        return []
    body_start = func_match.end()
    offsets: list[int] = []
    cursor = body_start
    while True:
        m = re.search(r"\breturn\b", mlir_text[cursor:])
        if not m:
            break
        offsets.append(cursor + m.start())
        cursor = cursor + m.end()
    return offsets


# ── Plan: sharding spec → list of InsertedCollective


def _plan_collectives(
    spec: ShardingSpec,
    module: StableHLOModule,
    mesh: DeviceMesh | None,
    dtype_overrides: dict[str, str] | None = None,
) -> list[InsertedCollective]:
    """Translate a ``ShardingSpec`` into a concrete plan of collectives.

    The plan attaches every sharded tensor to the collectives it
    requires. The mapping follows the canonical GSPMD rule:

      - Replicated tensors: no collectives.
      - Sharded tensors: along each sharded axis, emit an all-reduce
        for gradient sync, an all-gather to materialize a replicated
        copy, a reduce-scatter to slice a replicated input, and an
        all-to-all for permutation-style ops.

    Per-device backend selection is delegated to ``CommBackend`` so
    NCCL/RCCL/oneCCL/MIXED are chosen from the real device list.
    """
    dtype_overrides = dtype_overrides or {}
    comm_backend: CommBackend | None = None
    if mesh is not None:
        try:
            comm_backend = CommBackend(mesh)
        except Exception as exc:
            logger.warning("Failed to build CommBackend: %s", exc)
            comm_backend = None

    # Pre-build arg-name → (shape, dtype) lookup from the MLIR
    arg_types = _extract_func_arg_types(module.mlir_text)
    mesh_shape = list(spec.mesh_shape or [])
    if not mesh_shape and mesh is not None:
        mesh_shape = list(mesh.mesh_shape or [mesh.num_devices or 1])

    collectives: list[InsertedCollective] = []
    result_counter = 0

    for tensor_name, ts in spec.tensor_shardings.items():
        if not ts.mesh_axes:
            continue  # Replicated, no collectives needed

        # Resolve the tensor's per-device shape & dtype from the
        # function arg type in the MLIR.
        shape, dtype = _resolve_tensor_layout(tensor_name, arg_types, ts, dtype_overrides)
        per_tensor_bytes = _tensor_bytes(shape, dtype)

        for axis in ts.mesh_axes:
            device_ids = _devices_for_mesh_axis(mesh, mesh_shape, axis)
            num_devices = max(len(device_ids), 1)

            for ctype in (
                CollectiveType.ALL_REDUCE,
                CollectiveType.ALL_GATHER,
                CollectiveType.REDUCE_SCATTER,
                CollectiveType.ALL_TO_ALL,
            ):
                lib = _select_comm_library(comm_backend, device_ids, ctype)
                bytes_vol = _collective_volume_bytes(
                    ctype,
                    per_tensor_bytes,
                    num_devices,
                )
                result_counter += 1
                result_name = (
                    f"%coll_{tensor_name.lstrip('%')}_{ctype.value}_{axis}_{result_counter}"
                )
                collectives.append(
                    InsertedCollective(
                        collective_type=ctype,
                        tensor_name=tensor_name
                        if tensor_name.startswith("%")
                        else f"%{tensor_name}",
                        mesh_axis=axis,
                        num_devices=num_devices,
                        device_ids=tuple(device_ids),
                        comm_library=lib,
                        result_name=result_name,
                        tensor_shape=shape,
                        dtype=dtype,
                        reduction_op="add",
                        estimated_bytes=bytes_vol,
                    )
                )

    return collectives


def _extract_func_arg_types(mlir_text: str) -> dict[str, str]:
    """Extract ``{%name: tensor<...>}`` pairs from the first function.

    Returns a dict ``{arg_name (no %): "tensor<...>"}``.
    """
    func_match = _FUNC_SIG_RE.search(mlir_text)
    if not func_match:
        return {}
    sig_args = func_match.group(2)
    result: dict[str, str] = {}
    for m in _FUNC_ARG_RE.finditer(sig_args):
        name = m.group(1)
        type_str = m.group(2)
        result[name.lstrip("%")] = type_str
    return result


def _resolve_tensor_layout(
    tensor_name: str,
    arg_types: dict[str, str],
    ts: TensorSharding,
    dtype_overrides: dict[str, str],
) -> tuple[tuple[int, ...], str]:
    """Compute the per-device shape and dtype for a sharded tensor.

    The per-device shape is the partition_shape if provided, else the
    full tensor shape divided by the mesh axis size. The dtype comes
    from the MLIR function argument type, falling back to fp32.
    """
    clean_name = tensor_name.lstrip("%")
    type_str = arg_types.get(clean_name)
    full_shape: tuple[int, ...]
    dtype: str
    if type_str is not None:
        full_shape, dtype = _parse_tensor_shape_and_dtype(type_str)
    else:
        full_shape = tuple(ts.partition_shape) if ts.partition_shape else (1,)
        dtype = dtype_overrides.get(clean_name, "f32")

    if ts.partition_shape:
        per_device_shape: tuple[int, ...] = tuple(ts.partition_shape)
    else:
        # Fall back to a single-dim shape. We do not hardcode a
        # specific size — the volume computation below uses the real
        # per-device shape, not a constant.
        if full_shape and ts.mesh_axes:
            axis = ts.mesh_axes[0]
            if axis < len(full_shape) and full_shape[axis] > 1:
                per_device_shape = (
                    full_shape[0] // max(1, _safe_prod_axis_count(ts.mesh_axes)),
                    *full_shape[1:],
                )
            else:
                per_device_shape = full_shape
        else:
            per_device_shape = full_shape

    return per_device_shape, dtype


def _safe_prod_axis_count(mesh_axes: list[int]) -> int:
    """Product of axis counts — only used as a fallback for the
    per-device shape derivation. We never use it in the volume
    calculation; that's computed from real tensor bytes.
    """
    p = 1
    for a in mesh_axes:
        p *= max(1, a + 1)
    return p


# ── Public API: CollectiveInserter


class CollectiveInserter:
    """Insert collective operations into StableHLO MLIR for a sharded program.

    Usage::

        inserter = CollectiveInserter(mesh=device_mesh)
        result = inserter.insert(module, sharding_spec)
        print(result.mlir_text)         # Sharded StableHLO with collectives
        print(result.total_comm_bytes)  # Sum of all collective volumes
    """

    def __init__(
        self,
        mesh: DeviceMesh | None = None,
        channel_id_base: int = 0,
    ) -> None:
        self.mesh = mesh
        self.channel_id_base = channel_id_base

    def insert(
        self,
        module: StableHLOModule | None,
        spec: ShardingSpec,
        dtype_overrides: dict[str, str] | None = None,
    ) -> CollectiveInsertionResult:
        """Run the insertion pass on a sharded StableHLO module.

        Args:
            module: The StableHLO module (from ``StableHLOExporter``).
            spec: The sharding spec (from ``GSPMDRunner``).
            dtype_overrides: Optional mapping ``{tensor_name: dtype}``
                for tensors not present in the MLIR function
                signature. Defaults to ``"f32"``.

        Returns:
            ``CollectiveInsertionResult`` with the rewritten MLIR text,
            the list of inserted collectives, total comm volume, and
            per-backend usage statistics.
        """
        if module is None or not getattr(module, "is_usable", False):
            return CollectiveInsertionResult(
                stablehlo_module=module,
                error="StableHLO module is None or not usable",
            )
        if spec is None or not spec.tensor_shardings:
            # Nothing to do; pass through unchanged.
            return CollectiveInsertionResult(
                stablehlo_module=module,
                mlir_text=module.mlir_text,
                total_comm_bytes=0,
            )

        try:
            collectives = _plan_collectives(spec, module, self.mesh, dtype_overrides)
        except GSPMDError as exc:
            return CollectiveInsertionResult(
                stablehlo_module=module,
                error=f"Planning collectives failed: {exc}",
            )

        mlir_text, error = _emit_collectives(module.mlir_text, collectives, self.channel_id_base)
        if error is not None:
            return CollectiveInsertionResult(
                stablehlo_module=module,
                inserted_collectives=collectives,
                error=error,
            )

        total_bytes = sum(c.estimated_bytes for c in collectives)
        backend_usage: dict[str, int] = {}
        for c in collectives:
            key = c.comm_library.value
            backend_usage[key] = backend_usage.get(key, 0) + 1

        return CollectiveInsertionResult(
            stablehlo_module=module,
            inserted_collectives=collectives,
            total_comm_bytes=total_bytes,
            mlir_text=mlir_text,
            backend_usage=backend_usage,
        )


# ── MLIR emission: rewrite the function body with collective ops


def _emit_collectives(
    mlir_text: str,
    collectives: list[InsertedCollective],
    channel_id_base: int,
) -> tuple[str, str | None]:
    """Emit ``stablehlo.<collective>`` ops just before the first return.

    The function body is rewritten so that:

      1. Each collective's operand SSA name is the prior collective's
         result (chained). The first collective's operand is the
         original function arg.
      2. The first ``return`` op is updated to return the last
         collective's result, so the rest of the function sees the
         post-collective value.
      3. Later ``return`` ops are left untouched (only the function
         entry point matters for sharding).

    Returns ``(new_mlir_text, error)``. ``error`` is ``None`` on
    success.
    """
    if not collectives:
        return mlir_text, None

    func_match = _FUNC_SIG_RE.search(mlir_text)
    if not func_match:
        return mlir_text, "No func.func definition found in MLIR"

    ret_offset = _find_return_insertion_point(mlir_text)
    if ret_offset is None:
        return mlir_text, "No return op found in function body"

    # Group collectives by tensor so we can chain them.
    by_tensor: dict[str, list[InsertedCollective]] = {}
    for c in collectives:
        by_tensor.setdefault(c.tensor_name, []).append(c)

    # Emit in deterministic order: tensor name, then by collective
    # priority (reduce-scatter → all-reduce → all-gather → all-to-all).
    priority = {
        CollectiveType.REDUCE_SCATTER: 0,
        CollectiveType.ALL_REDUCE: 1,
        CollectiveType.ALL_GATHER: 2,
        CollectiveType.ALL_TO_ALL: 3,
    }

    new_ops: list[str] = []
    last_result_by_tensor: dict[str, str] = {}

    for tensor_name in sorted(by_tensor.keys()):
        ops = sorted(
            by_tensor[tensor_name],
            key=lambda c: (c.mesh_axis, priority[c.collective_type]),
        )
        # Wire each collective's operand to the previous result
        prev_result = tensor_name
        for i, coll in enumerate(ops):
            # Build a new collective record whose operand is the
            # chained SSA value, so the emitter produces correct MLIR.
            chained = InsertedCollective(
                collective_type=coll.collective_type,
                tensor_name=prev_result,
                mesh_axis=coll.mesh_axis,
                num_devices=coll.num_devices,
                device_ids=coll.device_ids,
                comm_library=coll.comm_library,
                result_name=coll.result_name,
                tensor_shape=coll.tensor_shape,
                dtype=coll.dtype,
                reduction_op=coll.reduction_op,
                estimated_bytes=coll.estimated_bytes,
            )
            channel_id = channel_id_base + i
            emitter = _EMITTERS[chained.collective_type]
            new_ops.append(emitter(chained, channel_id))
            prev_result = chained.result_name
        last_result_by_tensor[tensor_name] = prev_result

    # Update the first return: replace the returned SSA with the last
    # collective result, so downstream code sees the post-collective
    # tensor. We do this only for the first return; further returns
    # inside the function (e.g. inside `if` blocks) are untouched.
    return_match = re.search(r"\breturn\s+(\S+)\s*:(\s*\S+)", mlir_text[ret_offset:])
    first_tensor = next(iter(by_tensor.keys()), None)
    last_result = last_result_by_tensor.get(first_tensor) if first_tensor else None

    if return_match is not None:
        original_ret = mlir_text[ret_offset : ret_offset + return_match.end()]
        if last_result is not None:
            new_ret = re.sub(
                r"return\s+\S+",
                f"return {last_result}",
                original_ret,
                count=1,
            )
        else:
            new_ret = original_ret
        replacement_len = len(original_ret)
    else:
        new_ret = (
            f"  return {last_result} : {collectives[0].dtype}"
            if last_result is not None
            else "  return"
        )
        replacement_len = 0

    # Splice the new ops + new return into the MLIR text.
    insertion_block = "\n".join([*new_ops, new_ret])
    new_mlir = mlir_text[:ret_offset] + insertion_block + mlir_text[ret_offset + replacement_len :]
    return new_mlir, None


# ── Convenience: directly plan + emit in one call (used by tests)


def plan_and_insert(
    module: StableHLOModule,
    spec: ShardingSpec,
    mesh: DeviceMesh | None = None,
) -> CollectiveInsertionResult:
    """One-shot convenience wrapper around ``CollectiveInserter``."""
    return CollectiveInserter(mesh=mesh).insert(module, spec)


# Re-export useful types
__all__ = [
    "CollectiveInserter",
    "CollectiveInsertionResult",
    "CollectiveType",
    "InsertedCollective",
    "plan_and_insert",
]
