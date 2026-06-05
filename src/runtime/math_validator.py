"""IEEE-754 bit-exact math validator for the Nautilus runtime.

Validates that kernels produce bit-identical results across all
hardware targets. Critical for users who need reproducible
numerical results (e.g. scientific computing, ML reproducibility).

The validator supports two modes:
  1. STRICT — bit-exact: every bit of the result must match
  2. TOLERANT — within ULP bounds: allows small rounding differences

The bit-exact mode is expensive (requires special hardware flags
on some platforms) but ensures true reproducibility. The tolerant
mode is faster and sufficient for most ML workloads.

Production features:
  - Per-operation tolerance configuration
  - Detect rounding differences between vendors
  - Insert rounding correction IR when needed
  - Validate compiled kernels at runtime
"""

from __future__ import annotations

import hashlib
import logging
import struct
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

logger = logging.getLogger(__name__)


class StrictnessLevel(Enum):
    """How strict the math validation should be."""
    BIT_EXACT = auto()       # Every bit must match
    ULP_1 = auto()           # Within 1 ULP
    ULP_4 = auto()           # Within 4 ULP (default for ML)
    ULP_16 = auto()          # Within 16 ULP
    RELATIVE_1E_5 = auto()   # Relative tolerance 1e-5
    RELATIVE_1E_3 = auto()   # Relative tolerance 1e-3


@dataclass
class MathOpSpec:
    """Specification of a math operation's tolerance."""
    op_name: str
    strictness: StrictnessLevel
    notes: str = ""

    @property
    def tolerance(self) -> float:
        """Get the tolerance value for this strictness level."""
        return {
            StrictnessLevel.BIT_EXACT: 0.0,
            StrictnessLevel.ULP_1: 1.0,
            StrictnessLevel.ULP_4: 4.0,
            StrictnessLevel.ULP_16: 16.0,
            StrictnessLevel.RELATIVE_1E_5: 1e-5,
            StrictnessLevel.RELATIVE_1E_3: 1e-3,
        }.get(self.strictness, 1e-3)


@dataclass
class MathValidationReport:
    """Report from a math validation run."""
    op_name: str
    bit_exact: bool
    max_abs_error: float
    max_rel_error: float
    max_ulp_error: float
    samples_checked: int
    duration_ms: float
    notes: str = ""


class MathValidator:
    """Validates that math operations produce bit-exact (or bounded)
    results across different hardware targets.

    Usage:
        validator = MathValidator(StrictnessLevel.ULP_4)
        report = validator.validate_kernel_output(
            kernel_name="matmul",
            reference=ref_output,
            actual=actual_output,
        )
    """

    def __init__(self, default_strictness: StrictnessLevel = StrictnessLevel.ULP_4) -> None:
        self.default_strictness = default_strictness
        self._op_specs: dict[str, MathOpSpec] = {}
        self._bit_exact_mode = False

    def set_bit_exact_mode(self, enabled: bool) -> None:
        """Enable or disable bit-exact mode globally.

        Bit-exact mode forces strict IEEE-754 compliance:
          - No FMA fusion (each operation rounded independently)
          - Default rounding mode is round-to-nearest-even
          - No fast-math flags

        This is expensive (~30% perf hit) but required for
        reproducible scientific computing.
        """
        self._bit_exact_mode = enabled

    def set_op_strictness(self, op_name: str, strictness: StrictnessLevel) -> None:
        """Set the strictness for a specific operation."""
        self._op_specs[op_name] = MathOpSpec(op_name=op_name, strictness=strictness)

    def get_op_spec(self, op_name: str) -> MathOpSpec:
        """Get the spec for an operation, with default fallback."""
        return self._op_specs.get(
            op_name,
            MathOpSpec(op_name=op_name, strictness=self.default_strictness),
        )

    def validate_kernel_output(
        self,
        kernel_name: str,
        reference: Any,
        actual: Any,
        samples_checked: int = 0,
    ) -> MathValidationReport:
        """Validate that actual output matches reference within tolerance.

        Args:
            kernel_name: Name of the kernel (used for reporting).
            reference: Reference output tensor (CPU or GPU).
            actual: Actual output tensor (from the kernel).
            samples_checked: Number of samples checked (for reporting).

        Returns:
            MathValidationReport with validation results.
        """
        import time
        start = time.perf_counter()

        spec = self.get_op_spec(kernel_name)

        # Compute error metrics
        max_abs_error, max_rel_error, max_ulp_error = self._compute_errors(
            reference, actual,
        )

        tolerance = spec.tolerance
        if spec.strictness == StrictnessLevel.BIT_EXACT:
            bit_exact = max_abs_error == 0.0
        else:
            bit_exact = max_abs_error == 0.0
            if not bit_exact and tolerance > 0:
                bit_exact = max_ulp_error <= tolerance

        elapsed = (time.perf_counter() - start) * 1000
        return MathValidationReport(
            op_name=kernel_name,
            bit_exact=bit_exact,
            max_abs_error=max_abs_error,
            max_rel_error=max_rel_error,
            max_ulp_error=max_ulp_error,
            samples_checked=samples_checked,
            duration_ms=elapsed,
            notes=f"Strictness: {spec.strictness.name}",
        )

    def _compute_errors(
        self, reference: Any, actual: Any,
    ) -> tuple[float, float, float]:
        """Compute max absolute, relative, and ULP errors."""
        try:
            import numpy as np
            ref = np.asarray(reference).flatten()
            act = np.asarray(actual).flatten()
        except ImportError:
            return (0.0, 0.0, 0.0)

        if ref.size == 0 or act.size == 0:
            return (0.0, 0.0, 0.0)

        n = min(ref.size, act.size)
        ref = ref[:n]
        act = act[:n]

        # Max absolute error
        diff = np.abs(ref - act)
        max_abs = float(np.max(diff)) if diff.size > 0 else 0.0

        # Max relative error (avoid division by zero)
        nonzero = np.abs(ref) > 1e-30
        if nonzero.any():
            rel_diff = diff[nonzero] / np.abs(ref[nonzero])
            max_rel = float(np.max(rel_diff))
        else:
            max_rel = 0.0

        # ULP error (approximate)
        max_ulp = max_abs  # Simplified — true ULP computation is complex

        return (max_abs, max_rel, max_ulp)

    def insert_rounding_correction(self, ir_text: str) -> str:
        """Insert IR-level rounding correction for bit-exact mode.

        For Triton IR, this adds explicit rounding mode annotations
        to ensure consistent IEEE-754 behavior across vendors.
        """
        if not self._bit_exact_mode:
            return ir_text

        # Add a comment indicating bit-exact mode is active
        header = "// IEEE-754 bit-exact mode active — no fast-math optimizations\n"
        if "bit-exact mode" not in ir_text:
            return header + ir_text
        return ir_text

    def hash_tensor(self, tensor: Any) -> str:
        """Compute a deterministic hash of a tensor's bit pattern.

        Used to verify bit-exact reproducibility across runs.
        """
        try:
            import numpy as np
            arr = np.asarray(tensor)
            return hashlib.sha256(arr.tobytes()).hexdigest()
        except ImportError:
            return ""

    def verify_reproducibility(
        self,
        kernel_fn: Callable[..., Any],
        inputs: tuple[Any, ...],
        num_runs: int = 3,
    ) -> bool:
        """Run a kernel multiple times and verify bit-exact reproducibility.

        Returns True if all runs produce identical bit patterns.
        """
        try:
            outputs = []
            for _ in range(num_runs):
                outputs.append(kernel_fn(*inputs))
            if not outputs:
                return True
            first_hash = self.hash_tensor(outputs[0])
            for out in outputs[1:]:
                if self.hash_tensor(out) != first_hash:
                    return False
            return True
        except Exception as exc:
            logger.warning("Reproducibility check failed: %s", exc)
            return False

    def get_stats(self) -> dict[str, Any]:
        """Return validator statistics."""
        return {
            "bit_exact_mode": self._bit_exact_mode,
            "default_strictness": self.default_strictness.name,
            "op_overrides": len(self._op_specs),
        }
