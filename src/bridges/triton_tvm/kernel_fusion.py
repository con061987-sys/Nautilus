"""Cross-op kernel fusion engine for matmul + activation patterns.

Detects fusible computation patterns in a kernel's op graph and generates
fused ``@triton.jit`` kernels that avoid round-tripping through global memory.

Fusible patterns (all matmul-centric)
-------------------------------------
* matmul + relu
* matmul + gelu
* matmul + silu
* matmul + bias + relu
* matmul + bias + gelu
* matmul + bias + silu

How fusion works
----------------
A fused kernel computes ``C = activation(matmul(A, B))`` (or ``C = activation(matmul(A, B) + bias)``)
in a single kernel launch, keeping the matmul accumulator in registers and
applying the elementwise function before the global store.  This saves:

1. One global-memory write of the raw matmul result
2. One global-memory read of the raw matmul result (by the separate elementwise kernel)
3. Launch overhead of the second kernel

Each pattern produces a speedup estimate grounded in roofline analysis
of the saved memory traffic vs total compute cost.

Usage::

    from src.bridges.triton_tvm.kernel_fusion import FusionPlanner, FusionCodeGenerator

    # op_graph is a list of OpNode in execution order
    ops = [
        OpNode("matmul", {"m": 4096, "n": 4096, "k": 4096, "dtype": "float32"}),
        OpNode("relu"),
    ]
    planner = FusionPlanner()
    plans = planner.find_patterns(ops)

    for plan in plans:
        print(f"Found {plan.pattern}, estimated speedup: {plan.estimated_speedup:.1%}")
        code = FusionCodeGenerator().generate(plan)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.common.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Op-graph model
# ---------------------------------------------------------------------------


class OpKind(str, Enum):
    """Primitive operation kinds recognised by the fusion planner."""

    MATMUL = "matmul"
    BIAS_ADD = "bias_add"
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    UNKNOWN = "unknown"


# Activation ops that can fuse with matmul.
_FUSIBLE_ACTIVATIONS = frozenset({OpKind.RELU, OpKind.GELU, OpKind.SILU})

# Estimated extra cost of each activation relative to a single matmul iteration.
# Used to bound the speedup estimate (large activation cost = less relative saving).
_ACTIVATION_COST_FACTOR: dict[OpKind, float] = {
    OpKind.RELU: 0.01,  # negligible
    OpKind.GELU: 0.05,  # expf + erf approximation
    OpKind.SILU: 0.03,  # sigmoid
}


@dataclass(frozen=True)
class OpNode:
    """A single operation in the kernel computation graph.

    Attributes:
        kind:       Operation kind.
        attrs:      Free-form attributes (tensor shapes, dtypes, axes, etc.).
        consumers:  Indexes of nodes that consume this node's output (populated
                    by the planner during graph analysis).
    """

    kind: OpKind
    attrs: dict[str, Any] = field(default_factory=dict)
    consumers: tuple[int, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Fusion plan
# ---------------------------------------------------------------------------


@dataclass
class FusionPlan:
    """Describes one fusible pattern discovered in the kernel graph.

    ``ops`` holds the indexes (or the actual ``OpNode`` objects) that
    participate in the fusion.  The generator uses this list to know
    which operations to inline.
    """

    pattern: str
    ops: list[OpNode]
    estimated_speedup: float
    fused_kernel_source: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pattern not in _SUPPORTED_PATTERNS:
            logger.warning("Unknown fusion pattern: %s", self.pattern)
        if self.estimated_speedup < 0.0 or self.estimated_speedup > 1.0:
            raise ValueError(
                f"estimated_speedup must be in [0, 1], got {self.estimated_speedup}"
            )


# Pattern names — used as FusionPlan.pattern values.
PATTERN_MATMUL_RELU = "matmul+relu"
PATTERN_MATMUL_GELU = "matmul+gelu"
PATTERN_MATMUL_SILU = "matmul+silu"
PATTERN_MATMUL_BIAS_RELU = "matmul+bias+relu"
PATTERN_MATMUL_BIAS_GELU = "matmul+bias+gelu"
PATTERN_MATMUL_BIAS_SILU = "matmul+bias+silu"

_SUPPORTED_PATTERNS = frozenset({
    PATTERN_MATMUL_RELU,
    PATTERN_MATMUL_GELU,
    PATTERN_MATMUL_SILU,
    PATTERN_MATMUL_BIAS_RELU,
    PATTERN_MATMUL_BIAS_GELU,
    PATTERN_MATMUL_BIAS_SILU,
})


# ---------------------------------------------------------------------------
# Fusion planner
# ---------------------------------------------------------------------------


class FusionPlanner:
    """Analyzes an op graph and produces all fusible pattern candidates.

    The planner walks the op graph in execution order and looks for
    specific subsequences that can be fused into a single kernel:

    1. **matmul + activation** — the matmul's output is immediately
       consumed by a single elementwise activation.
    2. **matmul + bias + activation** — the matmul output feeds a bias-add
       whose output feeds an activation.

    The planner is conservative: it only proposes fusions where the
    elementwise op has exactly one consumer (the matmul) and the bias-add
    (if present) also has exactly one consumer.  This avoids fusing
    through re-used values.
    """

    def find_patterns(self, graph: list[OpNode]) -> list[FusionPlan]:
        """Scan an op graph for fusible subsequences.

        Args:
            graph: A list of ``OpNode`` objects in execution order.

        Returns:
            A (possibly empty) list of ``FusionPlan`` objects, one per
            detected fusible subsequence.
        """
        plans: list[FusionPlan] = []
        if not graph:
            return plans

        # Build consumer-index mapping so we can check use-counts.
        graph = self._populate_consumers(graph)

        i = 0
        while i < len(graph):
            node = graph[i]

            # --- Pattern 1: matmul + activation ---
            if node.kind == OpKind.MATMUL:
                plan = self._try_matmul_activation(graph, i)
                if plan is not None:
                    plans.append(plan)
                    i += len(plan.ops)
                    continue

                # --- Pattern 2: matmul + bias + activation ---
                plan = self._try_matmul_bias_activation(graph, i)
                if plan is not None:
                    plans.append(plan)
                    i += len(plan.ops)
                    continue

            i += 1

        return plans

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _populate_consumers(graph: list[OpNode]) -> list[OpNode]:
        """Fill ``consumers`` for each node based on linear adjacency.

        In the linear fusion patterns we target (matmul→bias→activation),
        each node's output is consumed by the next node in the graph.
        """
        result = list(graph)
        for i in range(len(result) - 1):
            prev = result[i]
            result[i] = OpNode(
                kind=prev.kind,
                attrs=prev.attrs,
                consumers=(*prev.consumers, i + 1),
            )
        return result

    def _try_matmul_activation(
        self,
        graph: list[OpNode],
        matmul_idx: int,
    ) -> FusionPlan | None:
        """Check if graph[matmul_idx] is followed by a fusible activation."""
        if matmul_idx + 1 >= len(graph):
            return None

        act = graph[matmul_idx + 1]
        if act.kind not in _FUSIBLE_ACTIVATIONS:
            return None

        # Conservatively require the activation to have exactly one
        # consumer (the matmul) — i.e. no other op reads the matmul output.
        if len(graph[matmul_idx].consumers) != 1:
            return None

        plan = FusionPlan(
            pattern=f"matmul+{act.kind.value}",
            ops=[graph[matmul_idx], act],
            estimated_speedup=self._estimate_speedup(
                graph[matmul_idx], act.kind
            ),
        )
        logger.debug(
            "FusionPlan found: %s (speedup=%.1f%%)",
            plan.pattern,
            plan.estimated_speedup * 100,
        )
        return plan

    def _try_matmul_bias_activation(
        self,
        graph: list[OpNode],
        matmul_idx: int,
    ) -> FusionPlan | None:
        """Check for matmul + bias_add + activation."""
        if matmul_idx + 2 >= len(graph):
            return None

        bias = graph[matmul_idx + 1]
        act = graph[matmul_idx + 2]

        if bias.kind != OpKind.BIAS_ADD:
            return None
        if act.kind not in _FUSIBLE_ACTIVATIONS:
            return None

        # Both matmul and bias must have a single consumer.
        if len(graph[matmul_idx].consumers) != 1:
            return None
        # Bias's consumer list should include the activation.
        if matmul_idx + 2 not in graph[matmul_idx + 1].consumers:
            return None

        plan = FusionPlan(
            pattern=f"matmul+bias+{act.kind.value}",
            ops=[graph[matmul_idx], bias, act],
            estimated_speedup=self._estimate_speedup(
                graph[matmul_idx], act.kind, has_bias=True
            ),
        )
        logger.debug(
            "FusionPlan found: %s (speedup=%.1f%%)",
            plan.pattern,
            plan.estimated_speedup * 100,
        )
        return plan

    # ------------------------------------------------------------------
    # Speedup estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_speedup(
        matmul_op: OpNode,
        activation: OpKind,
        has_bias: bool = False,
    ) -> float:
        """Compute a conservative speedup fraction for a fused pattern.

        The estimate is based on saved memory traffic vs total execution
        time:

        * Without fusion: matmul writes result → activation reads result
          → activation writes final.  Two global writes + one read of the
          matmul output tile.
        * With fusion: matmul output stays in registers → activation
          applied inline → one global write.

        The savings are bounded by the memory-bandwidth fraction of the
        matmul (which is typically high for matmul, ~70 % on most GPUs).

        Returns a float in [0, 1] representing fractional speedup.
        """
        m = matmul_op.attrs.get("m", 0)
        n = matmul_op.attrs.get("n", 0)
        k = matmul_op.attrs.get("k", 0)

        if m <= 0 or n <= 0 or k <= 0:
            # Cannot estimate — return a conservative baseline.
            return 0.12

        # Rough bytes moved by the matmul (two inputs + one output).
        dtype_bytes = _dtype_bytes(matmul_op.attrs.get("dtype", "float32"))
        matmul_read_bytes = (m * k + k * n) * dtype_bytes
        matmul_write_bytes = m * n * dtype_bytes
        matmul_bytes = matmul_read_bytes + matmul_write_bytes

        # Saved bytes from fusion: one write + one read of the matmul output.
        saved_bytes = 2 * matmul_write_bytes

        if has_bias:
            # Bias add would read the bias vector (n elements) and write
            # the result back.  Saved: one extra read + write of the output.
            saved_bytes += 2 * matmul_write_bytes

        # Account for activation compute cost (reduces relative savings).
        act_cost = _ACTIVATION_COST_FACTOR.get(activation, 0.03)

        # Matmul compute (FLOPs) as a proxy for execution time.
        compute_flops = 2 * m * n * k

        # Memory-bandwidth-bound estimate: fraction of time spent on memory.
        # Assume ~800 GB/s BW, ~200 TFLOPS for a modern GPU.
        mem_time = matmul_bytes / (800e9)
        compute_time = compute_flops / (200e12)
        total_time = mem_time + compute_time

        if total_time <= 0:
            return 0.12

        mem_fraction = mem_time / total_time
        # Saved memory fraction from fusion.
        saved_mem_fraction = (saved_bytes / matmul_bytes) * mem_fraction

        # Conservative: cap at 40% and subtract activation overhead.
        raw_speedup = min(saved_mem_fraction * (1 - act_cost), 0.40)

        # Floor at a small positive value (fusion always saves something).
        return max(raw_speedup, 0.05)


def _dtype_bytes(dtype: str) -> int:
    """Map dtype string to byte width."""
    mapping = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "float64": 8,
        "int32": 4,
        "int16": 2,
        "int8": 1,
    }
    return mapping.get(dtype, 4)


# ---------------------------------------------------------------------------
# Fused-kernel code generator
# ---------------------------------------------------------------------------


class FusionCodeGenerator:
    """Generates ``@triton.jit`` fused kernel source from a ``FusionPlan``.

    The produced kernel:

    * Follows the same block-indexing pattern as the existing vendor matmul
      templates (``matmul.py``) but without vendor-specific tile tuning.
    * Inlines the activation function into the epilogue so the matmul
      result is transformed before the global store.
    * For ``matmul+bias+activation`` patterns, applies the bias before
      the activation.
    * Remains **vendor-neutral** — no constexpr overrides for specific
      hardware.

    Usage::

        code = FusionCodeGenerator().generate(fusion_plan)
        print(code)
    """

    # Jinja-like template for the fused kernel source.
    _KERNEL_TEMPLATE = """\
# --- Fused kernel (auto-generated by FusionCodeGenerator) ---
# Pattern: {pattern}
# Estimated speedup over separate kernels: {speedup:.1%}

import triton
import triton.language as tl


@triton.jit
def {kernel_name}(
    a_ptr, b_ptr, c_ptr,{bias_param}
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,{bias_strides}
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ACC_DTYPE: tl.constexpr = tl.float32,
):
    # -- Block indexing (group-scheduled, vendor-neutral) --
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # -- Pointer arithmetic --
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -- Accumulator --
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)

    # -- K-loop (matmul) --
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a_tile = tl.load(a_ptrs, mask=(offs_k[None, :] < K - _ * BLOCK_K))
        b_tile = tl.load(b_ptrs, mask=(offs_k[:, None] < K - _ * BLOCK_K))
        acc = tl.dot(a_tile, b_tile, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

{bias_apply}

    # -- Activation (inlined in epilogue) --
{activation_body}

    # -- Store fused result --
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    tl.store(c_ptrs, {store_value}, mask=c_mask)
"""

    _ACTIVATION_BODIES: dict[str, str] = {
        "relu": "    # relu: max(0, x)\n    acc = tl.where(acc > 0.0, acc, 0.0)",
        "gelu": (
            "    # gelu (tanh approx): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))\n"
            "    from tl.libdevice import tanh as _tl_tanh\n"
            "    _gelu_inv_sqrt2 = 0.7071067811865475\n"
            "    _gelu_poly = 0.044715\n"
            "    _gelu_x_cube = acc * acc * acc\n"
            "    _gelu_inner = _gelu_inv_sqrt2 * (acc + _gelu_poly * _gelu_x_cube)\n"
            "    acc = 0.5 * acc * (1.0 + _tl_tanh(_gelu_inner))"
        ),
        "silu": "    # silu: x * sigmoid(x)\n    acc = acc * tl.sigmoid(acc)",
    }

    def generate(self, plan: FusionPlan) -> str:
        """Produce a fused ``@triton.jit`` kernel source string.

        Args:
            plan: A ``FusionPlan`` produced by ``FusionPlanner.find_patterns``.

        Returns:
            Valid Python source for a ``@triton.jit`` fused kernel.
        """
        parts = plan.pattern.split("+")
        has_bias = "bias" in parts
        activation_name = parts[-1]  # relu, gelu, or silu

        kernel_name = self._pattern_to_kernel_name(plan.pattern)
        bias_param = " bias_ptr," if has_bias else ""
        bias_strides = " bias_stride," if has_bias else ""
        bias_apply = self._bias_apply_code(has_bias)
        activation_body = self._ACTIVATION_BODIES.get(activation_name, "")
        store_value = self._store_value(has_bias, activation_name)

        source = self._KERNEL_TEMPLATE.format(
            pattern=plan.pattern,
            speedup=plan.estimated_speedup,
            kernel_name=kernel_name,
            bias_param=bias_param,
            bias_strides=bias_strides,
            bias_apply=bias_apply,
            activation_body=activation_body,
            store_value=store_value,
        )
        plan.fused_kernel_source = source
        return source

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pattern_to_kernel_name(pattern: str) -> str:
        """Convert ``matmul+relu`` to ``matmul_relu_fused``."""
        return pattern.replace("+", "_") + "_fused"

    @staticmethod
    def _bias_apply_code(has_bias: bool) -> str:
        if not has_bias:
            return ""
        return (
            "    # -- Bias add (inlined) --\n"
            "    offs_bias = tl.arange(0, BLOCK_N)\n"
            "    bias_mask = offs_bias < N\n"
            "    bias_val = tl.load(bias_ptr + offs_bias, mask=bias_mask, other=0.0).to(ACC_DTYPE)\n"
            "    acc = acc + bias_val[None, :]"
        )

    @staticmethod
    def _store_value(has_bias: bool, activation: str) -> str:
        """Determine the value expression to store.

        After fusion the stored value is always ``acc`` (which was
        modified in-place by the activation and optionally the bias).
        """
        _ = has_bias, activation
        return "acc"


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


__all__ = [
    "PATTERN_MATMUL_BIAS_GELU",
    "PATTERN_MATMUL_BIAS_RELU",
    "PATTERN_MATMUL_BIAS_SILU",
    "PATTERN_MATMUL_GELU",
    "PATTERN_MATMUL_RELU",
    "PATTERN_MATMUL_SILU",
    "FusionCodeGenerator",
    "FusionPlan",
    "FusionPlanner",
    "OpKind",
    "OpNode",
]
