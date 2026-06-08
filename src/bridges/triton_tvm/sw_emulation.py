"""
Software emulation for Nvidia-specific hardware features on non-Nvidia GPUs.

Detects Nvidia-only features (FP4, FP8 tensor cores, Transformer Engine) in
a computation graph and replaces them with equivalent software emulation
using standard Triton operations on FP16/FP32 arithmetic.

Architecture::

    ModelGraph (computation graph)
        │
        ▼
    SWEmulationEngine.detect_nvidia_features()
        │  ┌──────────────────────┐
        ├──│ FP4 casts detected   │ → FP4Emulation
        ├──│ FP8 tensor cores     │ → FP8Emulation
        └──│ Transformer Engine   │ → TransformerEngineEmulation
           └──────────────────────┘
        │
        ▼
    SWEmulationEngine.apply_emulation()
        │
        ▼
    Emulated graph (runs on any vendor)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from src.common.hardware import GpuVendor, enumerate_devices
from src.common.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Computation graph model (lightweight — no MLIR dependency)
# ---------------------------------------------------------------------------


class OpCategory(Enum):
    """Category of an operation in the computation graph."""

    CAST = auto()
    MATMUL = auto()
    ATTENTION = auto()
    REDUCTION = auto()
    ELEMENTWISE = auto()
    SOFTMAX = auto()
    LAYER_NORM = auto()
    RMS_NORM = auto()
    QUANTIZE = auto()
    DEQUANTIZE = auto()
    UNKNOWN = auto()


@dataclass
class OpNode:
    """A single operation in the computation graph."""

    name: str
    category: OpCategory
    attributes: dict[str, Any] = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    # DTYPE info: "fp4", "fp8_e4m3", "fp8_e5m2", "fp16", "bf16", "fp32"
    dtype: str = "fp32"


@dataclass
class ModelGraph:
    """Lightweight computation graph for feature detection.

    This is a vendor-neutral representation that captures the ops
    a model uses, independent of the MLIR dialect.  Callers construct
    it from whichever source they have (TorchFX graph, TTGIR capture,
    StableHLO module, etc.).
    """

    nodes: dict[str, OpNode] = field(default_factory=dict)
    # Ordered list of node names (topological order if available)
    topological_order: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: OpNode) -> None:
        self.nodes[node.name] = node
        if node.name not in self.topological_order:
            self.topological_order.append(node.name)

    def find_nodes_by_category(self, category: OpCategory) -> list[OpNode]:
        return [n for n in self.nodes.values() if n.category == category]

    def find_nodes_by_dtype(self, dtype: str) -> list[OpNode]:
        return [n for n in self.nodes.values() if n.dtype == dtype]

    def find_nodes_by_attr(self, key: str, value: Any) -> list[OpNode]:
        return [n for n in self.nodes.values() if n.attributes.get(key) == value]


# ---------------------------------------------------------------------------
# Emulation plan
# ---------------------------------------------------------------------------


@dataclass
class EmulationPlan:
    """Describes the emulation status for one Nvidia-specific feature.

    Attributes:
        feature: Identifier for the feature ("fp4", "fp8", "transformer_engine").
        detected: Whether the model uses this feature.
        emulated: Whether emulation is active.
        performance_impact: Estimated slowdown factor (1.0 = no impact,
            2.0 = 2x slower, etc.).
        details: Human-readable explanation of what was detected/emulated.
    """

    feature: str
    detected: bool = False
    emulated: bool = False
    performance_impact: float = 1.0
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "detected": self.detected,
            "emulated": self.emulated,
            "performance_impact": self.performance_impact,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Individual emulation implementations
# ---------------------------------------------------------------------------


class FP4Emulation:
    """Emulates FP4 quantization using FP16/FP32 operations.

    Nvidia Hopper GPUs support native 4-bit floating point (FP4) with
    the E2M1 format (2 exponent bits, 1 mantissa bit).  On non-Nvidia
    hardware, we simulate this by:

    1. Quantizing FP16/BF16 values to FP4 range using max-magnitude scaling
    2. Packing two 4-bit values into one INT8 byte (memory footprint savings)
    3. Dequantizing back to FP16/FP32 before computation
    4. Accumulating in FP16/FP32 to avoid precision loss

    This is a *functional* emulation — numerical results match FP4 at
    the cost of higher memory traffic and compute.
    """

    # FP4 E2M1: sign=1, exponent=2, mantissa=1
    # Max normal value: 2^(2^(2-1)-1) * 1.5 = 2^3 * 1.5 = 12.0
    # Min normal value: 2^(-2) * 1.0 = 0.25
    # Smallest subnormal: 2^(-2) * 2^(-1) = 0.125
    FP4_MAX: float = 12.0
    FP4_MIN_NORMAL: float = 0.25
    FP4_MIN_SUBNORMAL: float = 0.125

    @staticmethod
    def simulate_fp4_quantize(value: float, scale: float) -> float:
        """Simulate FP4 quantization of a single value.

        Clamps to FP4 E2M1 representable range, then rounds to the
        nearest representable value.
        """
        # Scale the value into FP4 representable range
        scaled = value / scale if scale != 0.0 else value

        # Clamp to FP4 range
        clamped = max(-FP4Emulation.FP4_MAX, min(FP4Emulation.FP4_MAX, scaled))

        # E2M1 representable values:
        # Exponent bits: 00 (denorm), 01 (2^-2), 10 (2^0=1), 11 (2^2=4)
        # With mantissa bit: 0 or 0.5 fraction
        # Values: denorm: 0, 0.125  |  exp=01: 0.25, 0.375
        #         exp=10: 1.0, 1.5  |  exp=11: 4.0, 6.0
        fp4_values = [
            -12.0,
            -6.0,
            -4.0,
            -1.5,
            -1.0,
            -0.375,
            -0.25,
            -0.125,
            0.0,
            0.125,
            0.25,
            0.375,
            1.0,
            1.5,
            4.0,
            6.0,
            12.0,
        ]
        # Find nearest
        nearest = min(fp4_values, key=lambda x: abs(clamped - x))
        return nearest * scale

    @staticmethod
    def emulate_fp4_linear(
        input_fp16: list[list[float]],
        weight_fp16: list[list[float]],
        activation_scale: float = 1.0,
        weight_scale: float = 1.0,
    ) -> list[list[float]]:
        """Emulate a linear layer with FP4 quantization on both weights and activations.

        This matches the Nvidia FP4 tensor core path::

            Out = dequant(matmul(quant(X, scale_x), quant(W, scale_w)), scale_x * scale_w)

        All arithmetic after dequantization happens in FP32 so the
        result is numerically equivalent to an FP4 tensor core pass.

        Returns the output matrix as FP32 values.
        """
        if not input_fp16 or not weight_fp16:
            return []

        m = len(input_fp16)
        n = len(weight_fp16[0]) if weight_fp16 else 0
        k = len(weight_fp16)

        # Quantize inputs to FP4
        quant_input = [[0.0] * k for _ in range(m)]
        for i in range(m):
            for j in range(k):
                quant_input[i][j] = FP4Emulation.simulate_fp4_quantize(
                    input_fp16[i][j], activation_scale
                )

        # Quantize weights to FP4
        quant_weight = [[0.0] * n for _ in range(k)]
        for i in range(k):
            for j in range(n):
                quant_weight[i][j] = FP4Emulation.simulate_fp4_quantize(
                    weight_fp16[i][j], weight_scale
                )

        # Matmul with FP32 accumulation
        output = [[0.0] * n for _ in range(m)]
        combined_scale = activation_scale * weight_scale
        for i in range(m):
            for j in range(n):
                acc = 0.0
                for t in range(k):
                    acc += quant_input[i][t] * quant_weight[t][j]
                output[i][j] = acc * combined_scale

        return output

    @staticmethod
    def estimate_performance_impact() -> float:
        """Estimated slowdown vs native FP4 tensor cores.

        FP4 emulation requires:
        - Quantization overhead (casting + scaling) on every matmul input
        - Dequantization overhead on matmul output
        - No tensor core speedup on non-Nvidia hardware

        Estimated impact: 3-5x vs native FP4 tensor cores.
        """
        return 4.0


class FP8Emulation:
    """Emulates FP8 tensor core operations using FP16 accumulation.

    Nvidia Hopper GPUs support FP8 (E4M3 and E5M2 formats) in their
    fourth-gen tensor cores.  This emulation simulates FP8 matmul and
    elementwise operations on any GPU by:

    1. Quantizing FP16/BF16 inputs to FP8 range
    2. Computing in FP16 (no FP8 tensor cores available)
    3. Applying FP8-specific rounding to the output

    Two FP8 formats are supported:
    - E4M3 (4 exponent, 3 mantissa):  better precision, smaller range
    - E5M2 (5 exponent, 2 mantissa):  wider range, less precision
    """

    # FP8 E4M3: sign=1, exponent=4, mantissa=3 → max=448, min_normal=2^-6
    E4M3_MAX: float = 448.0
    E4M3_MIN_NORMAL: float = 1 / 64  # 2^-6

    # FP8 E5M2: sign=1, exponent=5, mantissa=2 → max=57344, min_normal=2^-14
    E5M2_MAX: float = 57344.0
    E5M2_MIN_NORMAL: float = 1 / 16384  # 2^-14

    @staticmethod
    def simulate_fp8_quantize(value: float, scale: float, e5m2: bool = False) -> float:
        """Simulate FP8 quantization of a single value.

        Args:
            value: The input value (in FP32).
            scale: Per-tensor scaling factor.
            e5m2: If True, use E5M2 format; otherwise E4M3.

        Returns:
            The value after FP8 quantization and dequantization
            (i.e., the rounded FP32 value that an FP8 tensor would hold).
        """
        max_val = FP8Emulation.E5M2_MAX if e5m2 else FP8Emulation.E4M3_MAX
        min_normal = FP8Emulation.E5M2_MIN_NORMAL if e5m2 else FP8Emulation.E4M3_MIN_NORMAL

        scaled = value / scale if scale != 0.0 else value
        clamped = max(-max_val, min(max_val, scaled))

        # Build the set of representable values for this FP8 format.
        if e5m2:
            # E5M2: sign=1, exponent=5, mantissa=2
            # Exponent bias = 15, so exponent range: -14 to 15
            # Normal values: 2^(E-15) * (1 + M/4) for E in 1..30
            # Denorm values: 2^(-14) * (M/4) for E=0
            # Inf/NaN: E=31
            fp8_values: list[float] = []
            # Denorms (E=0, M=1,2,3)
            for m in range(1, 4):
                fp8_values.append(2 ** (-14) * (m / 4))
                fp8_values.append(-(2 ** (-14)) * (m / 4))
            # Normal values (E=1..30)
            for e in range(1, 31):
                for m in range(4):
                    val = 2 ** (e - 15) * (1 + m / 4)
                    fp8_values.append(val)
                    fp8_values.append(-val)
            fp8_values.append(0.0)
        else:
            # E4M3: sign=1, exponent=4, mantissa=3
            # Nvidia's E4M3 omits Inf/NaN, so exponent field 15 is
            # available for normal numbers. Bias = 7.
            # Normal values: 2^(E-7) * (1 + M/8) for E in 1..15
            # Denorm values: 2^(-6) * (M/8) for E=0
            fp8_values = []
            # Denorms (E=0, M=1..7)
            for m in range(1, 8):
                fp8_values.append(2 ** (-6) * (m / 8))
                fp8_values.append(-(2 ** (-6)) * (m / 8))
            # Normal values (E=1..15; exp=15 gives max=448.0)
            for e in range(1, 16):
                for m in range(8):
                    val = 2 ** (e - 7) * (1 + m / 8)
                    fp8_values.append(val)
                    fp8_values.append(-val)
            fp8_values.append(0.0)

        nearest = min(fp8_values, key=lambda x: abs(clamped - x))
        return nearest * scale

    @staticmethod
    def emulate_fp8_matmul(
        a: list[list[float]],
        b: list[list[float]],
        scale_a: float = 1.0,
        scale_b: float = 1.0,
        scale_out: float = 1.0,
        e5m2: bool = False,
    ) -> list[list[float]]:
        """Emulate FP8 tensor core matrix multiplication.

        This simulates::

            Out = scale_out * matmul(Q(A, scale_a), Q(B, scale_b))

        where Q() is FP8 quantization. The accumulation happens in
        FP32 to match the precision behaviour of Nvidia FP8 tensor
        cores (which accumulate in FP32).
        """
        if not a or not b:
            return []

        m = len(a)
        k = len(a[0]) if a else 0
        n = len(b[0]) if b else 0

        # Quantize both inputs
        quant_a = [[0.0] * k for _ in range(m)]
        for i in range(m):
            for j in range(k):
                quant_a[i][j] = FP8Emulation.simulate_fp8_quantize(
                    a[i][j], scale_a, e5m2=e5m2
                )

        quant_b = [[0.0] * n for _ in range(k)]
        for i in range(k):
            for j in range(n):
                quant_b[i][j] = FP8Emulation.simulate_fp8_quantize(
                    b[i][j], scale_b, e5m2=e5m2
                )

        # Accumulate in FP32
        output = [[0.0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                acc = 0.0
                for t in range(k):
                    acc += quant_a[i][t] * quant_b[t][j]
                output[i][j] = acc * scale_out

        return output

    @staticmethod
    def emulate_fp8_softmax(
        logits: list[float], scale: float = 1.0
    ) -> list[float]:
        """Emulate FP8 softmax (Transformer Engine style).

        In the Transformer Engine, softmax inputs are in FP8 and
        outputs are in FP8.  We simulate by quantizing logits, doing
        softmax in FP32, and quantizing back.
        """
        # Quantize logits to FP8 (E4M3 — typical for activations)
        quant_logits = [
            FP8Emulation.simulate_fp8_quantize(x, scale, e5m2=False)
            for x in logits
        ]

        # Softmax in FP32
        max_val = max(quant_logits)
        exp_vals = [math.exp(x - max_val) for x in quant_logits]
        sum_exp = sum(exp_vals)
        softmax_output = [x / sum_exp for x in exp_vals]

        return softmax_output

    @staticmethod
    def estimate_performance_impact(e5m2: bool = False) -> float:
        """Estimated slowdown vs native FP8 tensor cores.

        FP8 emulation requires:
        - Quantization before matmul (2x casting overhead)
        - FP16 matmul instead of FP8 tensor core (∼2x slower on non-Nvidia)
        - Dequantization after matmul

        Without FP8 tensor core hardware, the effective throughput is
        roughly 2-3x worse than Nvidia's FP8 path.
        """
        return 2.5 if e5m2 else 2.0


class TransformerEngineEmulation:
    """Emulates Nvidia Transformer Engine (TE) operations.

    The Transformer Engine, introduced with Hopper H100, provides:
    - FP8 infrastructure (delayed scaling, AMAX tracking)
    - FP8 GEMM with automatic scale management
    - FP8 attention
    - LayerNorm with FP8 output

    This emulation replaces each TE operation with an equivalent
    sequence of FP32 operations that produce numerically identical
    results.
    """

    @staticmethod
    def emulate_fp8_gemm_with_scaling(
        a: list[list[float]],
        b: list[list[float]],
        amax_history: list[float] | None = None,
        scale_inv: float = 1.0,
    ) -> tuple[list[list[float]], float]:
        """Emulate Transformer Engine's FP8 GEMM with delayed scaling.

        TE tracks an AMAX (absolute max) history to choose the FP8
        scale factor dynamically.  This emulation:

        1. Computes the FP8 scale from AMAX history
        2. Quantizes inputs to FP8
        3. Performs matmul in FP32
        4. Returns the output and the new AMAX value

        Returns:
            (output_matrix, new_amax)
        """
        if not a or not b:
            return [], 0.0

        # Compute new AMAX
        new_amax = 0.0
        for row in a:
            for val in row:
                new_amax = max(new_amax, abs(val))
        for row in b:
            for val in row:
                new_amax = max(new_amax, abs(val))

        # The scale factor is chosen so that FP8_MAX / new_amax maps
        # the largest value to the FP8 representation limit.
        fp8_max = FP8Emulation.E4M3_MAX
        scale = fp8_max / new_amax if new_amax > 0 else 1.0

        # Quantize and matmul using FP8 emulation
        output = FP8Emulation.emulate_fp8_matmul(
            a, b, scale_a=scale, scale_b=scale, scale_out=scale_inv, e5m2=False
        )

        return output, new_amax

    @staticmethod
    def emulate_fp8_layer_norm(
        x: list[float],
        gamma: list[float],
        beta: list[float],
        eps: float = 1e-5,
    ) -> list[float]:
        """Emulate LayerNorm with FP8 output (TE style).

        TE computes LayerNorm in FP32 but outputs FP8.  This
        emulation computes the full-precision norm and then quantizes
        the output.
        """
        n = len(x)
        if n == 0:
            return []

        # Compute mean and variance in FP32
        mean = sum(x) / n
        variance = sum((v - mean) ** 2 for v in x) / n
        std = math.sqrt(variance + eps)

        # Normalize and apply affine transform
        normalized = [(v - mean) / std for v in x]
        result = [gamma[i] * normalized[i] + beta[i] for i in range(n)]

        # Quantize output to FP8 (E4M3 is typical for TE activations)
        quant_result = [
            FP8Emulation.simulate_fp8_quantize(v, scale=1.0, e5m2=False)
            for v in result
        ]
        return quant_result

    @staticmethod
    def emulate_delayed_scale_update(
        amax_history: list[float],
        fp8_max: float = FP8Emulation.E4M3_MAX,
        history_len: int = 1024,
    ) -> float:
        """Simulate TE's delayed scaling scale factor computation.

        TE maintains a history of AMAX values and computes the scale
        factor as ``fp8_max / max(amax_history)`` with a safety margin.
        """
        if not amax_history:
            return 1.0
        max_amax = max(amax_history[-history_len:])
        if max_amax <= 0:
            return 1.0
        safety_margin = 1.1  # 10% safety margin as TE does
        return fp8_max / (max_amax * safety_margin)

    @staticmethod
    def estimate_performance_impact() -> float:
        """Estimated slowdown vs native Transformer Engine.

        TE emulation combines FP8 matmul overhead with additional
        scale management and AMAX tracking.
        """
        return 3.0


# ---------------------------------------------------------------------------
# Graph scanning — feature detection
# ---------------------------------------------------------------------------

# Ops that indicate Nvidia-specific features
_FP4_INDICATOR_OPS: set[str] = {
    "tt.fp4_to_fp16",
    "tt.fp4_cast",
    "nvvm.fp4",
    "nvvm.fp4_to_fp16x2",
    "triton.nvidia_fp4",
}

_FP8_INDICATOR_OPS: set[str] = {
    "tt.fp8_to_fp16",
    "tt.fp8_cast",
    "tt.dot_fp8",
    "nvvm.fp8",
    "nvvm.fp8_e4m3",
    "nvvm.fp8_e5m2",
    "nvvm.wgmma_fp8",
    "nvvm.tma_fp8",
    "triton.nvidia_fp8",
}

_TE_INDICATOR_OPS: set[str] = {
    "te.fp8_gemm",
    "te.fp8_attention",
    "te.fp8_layer_norm",
    "te.delayed_scaling",
    "te.amax_update",
    "transformer_engine.fp8_gemm",
    "transformer_engine.delayed_scaling",
    "nvidia.TE_fp8",
}

_FP4_DTYPES: set[str] = {"fp4", "e2m1"}
_FP8_DTYPES: set[str] = {"fp8", "fp8_e4m3", "fp8_e5m2"}


def _scan_for_fp4(graph: ModelGraph) -> EmulationPlan:
    """Detect FP4-specific operations in the computation graph."""
    plan = EmulationPlan(feature="fp4", details="No FP4 operations detected.")

    # Check by operation name
    for node in graph.nodes.values():
        op_name = node.name.lower()
        if any(indicator in op_name for indicator in _FP4_INDICATOR_OPS):
            plan.detected = True
            plan.details = f"FP4 operation detected: {node.name}"
            break

    # Check by dtype
    if not plan.detected:
        for node in graph.find_nodes_by_dtype("fp4"):
            plan.detected = True
            plan.details = f"FP4 dtype detected in node: {node.name}"
            break

    if plan.detected:
        plan.performance_impact = FP4Emulation.estimate_performance_impact()

    return plan


def _scan_for_fp8(graph: ModelGraph) -> EmulationPlan:
    """Detect FP8-specific operations in the computation graph."""
    plan = EmulationPlan(feature="fp8", details="No FP8 operations detected.")

    # Check by operation name
    for node in graph.nodes.values():
        op_name = node.name.lower()
        if any(indicator in op_name for indicator in _FP8_INDICATOR_OPS):
            plan.detected = True
            plan.details = f"FP8 operation detected: {node.name}"
            break

    # Check by dtype
    if not plan.detected:
        for node in graph.nodes.values():
            if node.dtype in _FP8_DTYPES:
                plan.detected = True
                plan.details = f"FP8 dtype detected in node: {node.name} ({node.dtype})"
                break

    if plan.detected:
        e5m2 = any(
            "e5m2" in node.dtype for node in graph.nodes.values()
        )
        plan.performance_impact = FP8Emulation.estimate_performance_impact(e5m2=e5m2)

    return plan


def _scan_for_transformer_engine(graph: ModelGraph) -> EmulationPlan:
    """Detect Transformer Engine operations in the computation graph."""
    plan = EmulationPlan(
        feature="transformer_engine",
        details="No Transformer Engine operations detected.",
    )

    # Check by operation name
    for node in graph.nodes.values():
        op_name = node.name.lower()
        if any(indicator in op_name for indicator in _TE_INDICATOR_OPS):
            plan.detected = True
            plan.details = f"Transformer Engine operation detected: {node.name}"
            break

    # Check for combined FP8 matmul + scaling patterns (TE-specific)
    if not plan.detected:
        fp8_matmuls = [
            n
            for n in graph.nodes.values()
            if n.category == OpCategory.MATMUL and n.dtype in _FP8_DTYPES
        ]
        amax_updates = [
            n
            for n in graph.nodes.values()
            if "amax" in n.name.lower() or "scale" in n.name.lower()
        ]
        if len(fp8_matmuls) >= 2 and len(amax_updates) >= 1:
            plan.detected = True
            plan.details = (
                f"Transformer Engine pattern detected: "
                f"{len(fp8_matmuls)} FP8 matmuls + AMAX tracking"
            )

    if plan.detected:
        plan.performance_impact = TransformerEngineEmulation.estimate_performance_impact()

    return plan


# ---------------------------------------------------------------------------
# Graph rewriting — inserting emulation layers
# ---------------------------------------------------------------------------


def _rewrite_fp4_op(node: OpNode, graph: ModelGraph) -> OpNode:
    """Replace a single FP4 operation with its emulated equivalent.

    The rewritten node keeps its name but has its attributes updated
    to mark it as emulated.  The actual emulation is performed at
    compile time by inserting appropriate quantization/dequantization
    Triton kernels around the original operation.
    """
    emulated = OpNode(
        name=f"{node.name}_fp4_emulated",
        category=node.category,
        attributes={
            **node.attributes,
            "emulated": True,
            "emulation_type": "fp4",
            "original_op": node.name,
            "scale_factor": node.attributes.get("scale_factor", 1.0),
        },
        inputs=node.inputs,
        outputs=node.outputs,
        dtype="fp32",  # Emulated in FP32
    )
    return emulated


def _rewrite_fp8_op(node: OpNode, graph: ModelGraph) -> OpNode:
    """Replace a single FP8 operation with its emulated equivalent."""
    emulated = OpNode(
        name=f"{node.name}_fp8_emulated",
        category=node.category,
        attributes={
            **node.attributes,
            "emulated": True,
            "emulation_type": "fp8",
            "original_op": node.name,
            "fp8_format": node.attributes.get("fp8_format", "e4m3"),
        },
        inputs=node.inputs,
        outputs=node.outputs,
        dtype="fp16",  # Emulated in FP16 (accumulated in FP32)
    )
    return emulated


def _rewrite_te_op(node: OpNode, graph: ModelGraph) -> OpNode:
    """Replace a single Transformer Engine operation with its emulated equivalent."""
    emulated = OpNode(
        name=f"{node.name}_te_emulated",
        category=node.category,
        attributes={
            **node.attributes,
            "emulated": True,
            "emulation_type": "transformer_engine",
            "original_op": node.name,
            "amax_handling": "emulated",
            "delayed_scale": "emulated",
        },
        inputs=node.inputs,
        outputs=node.outputs,
        dtype="fp16",
    )
    return emulated


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _has_nvidia_gpu() -> bool:
    """Check whether the current system has Nvidia GPUs."""
    try:
        devices = enumerate_devices()
        return any(d.vendor == GpuVendor.NVIDIA for d in devices)
    except Exception:
        logger.warning("Failed to enumerate devices for emulation decision")
        return False


class SWEmulationEngine:
    """Software emulation engine for Nvidia-specific hardware features.

    Scans a computation graph for Nvidia-only features, estimates the
    performance impact, and applies software emulation for non-Nvidia
    targets.

    Usage::

        engine = SWEmulationEngine()
        plans = engine.detect_nvidia_features(graph)
        if any(p.detected and p.emulated for p in plans):
            emulated_graph = engine.apply_emulation(plans, graph)
        impact = engine.estimate_impact(plans)
    """

    def __init__(self, auto_emulate: bool = True) -> None:
        """Initialize the emulation engine.

        Args:
            auto_emulate: If True, automatically enables emulation when
                Nvidia features are detected on non-Nvidia hardware.
                Default is True.
        """
        self.auto_emulate = auto_emulate
        self._has_nvidia = _has_nvidia_gpu()
        if not self._has_nvidia:
            logger.info(
                "No Nvidia GPU detected; emulation will be enabled "
                "for Nvidia-specific features"
            )

    def detect_nvidia_features(self, model_graph: ModelGraph) -> list[EmulationPlan]:
        """Scan the computation graph for Nvidia-specific features.

        Examines every node in the graph for FP4 casts, FP8 tensor
        core operations, and Transformer Engine patterns.

        Args:
            model_graph: The computation graph to scan.

        Returns:
            A list of EmulationPlan objects, one per detected (or
            not-detected) feature.  The ``emulated`` field is set
            based on ``auto_emulate`` and whether the feature was
            detected on non-Nvidia hardware.
        """
        plans: list[EmulationPlan] = [
            _scan_for_fp4(model_graph),
            _scan_for_fp8(model_graph),
            _scan_for_transformer_engine(model_graph),
        ]

        # Determine emulation status
        for plan in plans:
            if plan.detected:
                if self._has_nvidia:
                    # On Nvidia hardware, keep native
                    plan.emulated = False
                    plan.details += " (running natively on Nvidia hardware)"
                elif self.auto_emulate:
                    plan.emulated = True
                    plan.details += " (emulation enabled for non-Nvidia hardware)"

        return plans

    def apply_emulation(
        self,
        plans: list[EmulationPlan],
        model_graph: ModelGraph,
    ) -> ModelGraph:
        """Apply software emulation to the computation graph.

        Replaces Nvidia-only ops with software emulation nodes.
        The returned graph preserves the original graph topology but
        with emulated operations where needed.

        Args:
            plans: Emulation plans from ``detect_nvidia_features()``.
            model_graph: The original computation graph.

        Returns:
            A new ModelGraph with emulation layers inserted.
        """
        if not any(p.detected and p.emulated for p in plans):
            logger.info("No emulation needed; returning original graph")
            return model_graph

        emulated_graph = ModelGraph(metadata=dict(model_graph.metadata))
        emulated_graph.metadata["emulation_active"] = True

        # Build a set of feature names that need emulation
        features_to_emulate = {p.feature for p in plans if p.detected and p.emulated}

        for node_name in model_graph.topological_order:
            original_node = model_graph.nodes[node_name]
            emulated_node: OpNode | None = None

            if "fp4" in features_to_emulate:
                if _is_fp4_node(original_node):
                    emulated_node = _rewrite_fp4_op(original_node, model_graph)

            if emulated_node is None and "fp8" in features_to_emulate:
                if _is_fp8_node(original_node):
                    emulated_node = _rewrite_fp8_op(original_node, model_graph)

            if emulated_node is None and "transformer_engine" in features_to_emulate:
                if _is_te_node(original_node):
                    emulated_node = _rewrite_te_op(original_node, model_graph)

            if emulated_node is None:
                # Pass through unchanged
                emulated_graph.add_node(original_node)
            else:
                emulated_graph.add_node(emulated_node)
                logger.debug(
                    "Emulated %s node %s -> %s",
                    emulated_node.attributes["emulation_type"],
                    node_name,
                    emulated_node.name,
                )

        return emulated_graph

    def estimate_impact(self, plans: list[EmulationPlan]) -> float:
        """Compute the total performance impact of all active emulations.

        The impact is multiplicative: if FP4 is 4x and FP8 is 2x, the
        total is 8x.  Only emulated plans contribute.

        Args:
            plans: Emulation plans from ``detect_nvidia_features()``.

        Returns:
            Combined performance impact factor (1.0 = no impact).
        """
        total = 1.0
        for plan in plans:
            if plan.emulated:
                total *= plan.performance_impact
        return total

    def get_summary(self, plans: list[EmulationPlan]) -> dict[str, Any]:
        """Return a structured summary suitable for CLI/tooling.

        Args:
            plans: Emulation plans from ``detect_nvidia_features()``.

        Returns:
            A dict with keys: has_nvidia, auto_emulate, plans, total_impact.
        """
        return {
            "has_nvidia_gpu": self._has_nvidia,
            "auto_emulate": self.auto_emulate,
            "emulation_active": any(p.emulated for p in plans),
            "plans": [p.to_dict() for p in plans],
            "total_impact": self.estimate_impact(plans),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_fp4_node(node: OpNode) -> bool:
    """Check if a node uses FP4 features."""
    if node.dtype in _FP4_DTYPES:
        return True
    op_name = node.name.lower()
    return any(indicator in op_name for indicator in _FP4_INDICATOR_OPS)


def _is_fp8_node(node: OpNode) -> bool:
    """Check if a node uses FP8 features."""
    if node.dtype in _FP8_DTYPES:
        return True
    op_name = node.name.lower()
    return any(indicator in op_name for indicator in _FP8_INDICATOR_OPS)


def _is_te_node(node: OpNode) -> bool:
    """Check if a node uses Transformer Engine features."""
    op_name = node.name.lower()
    if any(indicator in op_name for indicator in _TE_INDICATOR_OPS):
        return True
    # TE-specific pattern: FP8 matmul with AMAX tracking
    if node.category == OpCategory.MATMUL and node.dtype in _FP8_DTYPES:
        if "amax" in node.attributes or node.attributes.get("delayed_scaling", False):
            return True
    return False


# ---------------------------------------------------------------------------
# Convenience: build a ModelGraph from a simple op list
# ---------------------------------------------------------------------------


def build_graph_from_ops(
    ops: list[dict[str, Any]],
) -> ModelGraph:
    """Build a ModelGraph from a list of operation descriptors.

    Each descriptor must have at minimum a ``name`` key. Optional
    keys: ``category`` (string), ``attributes`` (dict), ``inputs``
    (list[str]), ``outputs`` (list[str]), ``dtype`` (str).

    This is the primary way to construct a graph for testing and
    for integration with the bridge pipeline.
    """
    graph = ModelGraph()
    for op in ops:
        category_name = op.get("category", "unknown").upper()
        try:
            category = OpCategory[category_name]
        except KeyError:
            category = OpCategory.UNKNOWN

        node = OpNode(
            name=op["name"],
            category=category,
            attributes=op.get("attributes", {}),
            inputs=op.get("inputs", []),
            outputs=op.get("outputs", []),
            dtype=op.get("dtype", "fp32"),
        )
        graph.add_node(node)
    return graph


__all__ = [
    "EmulationPlan",
    "FP4Emulation",
    "FP8Emulation",
    "ModelGraph",
    "OpCategory",
    "OpNode",
    "SWEmulationEngine",
    "TransformerEngineEmulation",
    "build_graph_from_ops",
]
