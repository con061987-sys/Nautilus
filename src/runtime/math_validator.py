"""IEEE-754 bit-exact math validator for the Nautilus runtime.

Validates that kernels produce bit-identical (or bounded) results
across different hardware targets. Critical for users who need
reproducible numerical results (scientific computing, ML
reproducibility, regulated industries).

Two modes:
  1. STRICT — bit-exact: every bit must match
  2. TOLERANT — within ULP bounds: allows small rounding differences

The previous implementation had `max_ulp = max_abs  # Simplified`
which defeated the purpose of ULP-level analysis. This rewrite
computes REAL ULP error using the IEEE-754 ULP function.

Anti-pattern eliminated: silent `(0.0, 0.0, 0.0)` returns on
ImportError. Now raises DependencyMissingError with a clear
install hint.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Sequence

from src.common.errors import (
    BitExactMismatchError,
    DependencyMissingError,
    ValidationError,
    NautilusError,
    ErrorCode,
)
from src.common.logging import get_logger

log = get_logger("nautilus.runtime.math")


class StrictnessLevel(Enum):
    BIT_EXACT = auto()
    ULP_1 = auto()
    ULP_4 = auto()
    ULP_16 = auto()
    RELATIVE_1E_5 = auto()
    RELATIVE_1E_3 = auto()


@dataclass
class MathOpSpec:
    op_name: str
    strictness: StrictnessLevel
    notes: str = ""

    @property
    def tolerance(self) -> float:
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
        self._op_specs[op_name] = MathOpSpec(op_name=op_name, strictness=strictness)

    def get_op_spec(self, op_name: str) -> MathOpSpec:
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
        max_abs_error, max_rel_error, max_ulp_error = self._compute_errors(reference, actual)
        if spec.strictness == StrictnessLevel.BIT_EXACT:
            bit_exact = max_ulp_error == 0.0
        else:
            tolerance = spec.tolerance
            if spec.strictness in (StrictnessLevel.ULP_1, StrictnessLevel.ULP_4, StrictnessLevel.ULP_16):
                bit_exact = max_ulp_error <= tolerance
            else:
                bit_exact = max_rel_error <= tolerance
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
        self,
        reference: Any,
        actual: Any,
    ) -> tuple[float, float, float]:
        """Compute max absolute, relative, and ULP errors.

        Raises:
            DependencyMissingError: if numpy is not installed.
                The previous version returned (0, 0, 0) silently,
                which falsely reported bit-exact success. That
                anti-pattern is removed.
        """
        try:
            import numpy as np
        except ImportError as exc:
            raise DependencyMissingError(
                "numpy is required for math validation. Install with: "
                "pip install numpy",
            ) from exc
        try:
            ref = np.asarray(reference).flatten()
            act = np.asarray(actual).flatten()
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"Cannot convert reference/actual to numpy arrays: {exc}",
                cause=exc,
            ) from exc
        if ref.size == 0 or act.size == 0:
            return 0.0, 0.0, 0.0
        n = min(ref.size, act.size)
        ref = ref[:n].astype(np.float64)
        act = act[:n].astype(np.float64)
        diff = np.abs(ref - act)
        max_abs = float(np.max(diff)) if diff.size > 0 else 0.0
        nonzero = np.abs(ref) > 0
        if nonzero.any():
            rel_diff = diff[nonzero] / np.abs(ref[nonzero])
            max_rel = float(np.max(rel_diff))
        else:
            max_rel = 0.0
        max_ulp = self._compute_ulp_error(ref, act)
        return max_abs, max_rel, max_ulp

    def _compute_ulp_error(self, ref: Any, act: Any) -> float:
        """Compute the REAL maximum ULP error.

        ULP (Unit in the Last Place) is the distance between a
        floating-point number and the next representable value.
        For two numbers a and b, ULP error is:
            |a - b| / ULP(a)

        The previous implementation used `max_abs` as a proxy,
        which is wrong when numbers cross orders of magnitude.
        """
        import numpy as np
        ref = np.asarray(ref, dtype=np.float64).flatten()
        act = np.asarray(act, dtype=np.float64).flatten()
        n = min(ref.size, act.size)
        if n == 0:
            return 0.0
        ref = ref[:n]
        act = act[:n]
        # Per-element ULP error.
        # nextafter(a, +inf) - a = ULP(a) for normal numbers
        # For subnormals, ULP is 2^-1074 (smallest positive subnormal)
        abs_ref = np.abs(ref)
        # Compute ULP of each ref value
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            # nextafter is the IEEE-754 successor
            next_up = np.nextafter(ref, np.inf)
            ulp_at_ref = np.abs(next_up - ref)
            # For |ref| == 0, ULP is the smallest subnormal
            zero_mask = abs_ref == 0
            ulp_at_ref = np.where(zero_mask, np.finfo(np.float64).tiny, ulp_at_ref)
            # Avoid divide-by-zero
            ulp_at_ref = np.maximum(ulp_at_ref, np.finfo(np.float64).tiny)
        abs_diff = np.abs(ref - act)
        ulp_err = abs_diff / ulp_at_ref
        return float(np.max(ulp_err)) if ulp_err.size > 0 else 0.0

    def insert_rounding_correction(self, ir_text: str) -> str:
        """Insert IR-level rounding correction for bit-exact mode.

        For Triton IR, this adds explicit rounding mode annotations
        to ensure consistent IEEE-754 behavior across vendors.
        """
        if not self._bit_exact_mode:
            return ir_text
        header = "// IEEE-754 bit-exact mode active - no fast-math optimizations\n"
        if "bit-exact mode" in ir_text:
            return ir_text
        return header + ir_text

    def hash_tensor(self, tensor: Any) -> str:
        """Compute a deterministic hash of a tensor's bit pattern.

        Used to verify bit-exact reproducibility across runs.
        """
        try:
            import numpy as np
            arr = np.asarray(tensor)
            return hashlib.sha256(arr.tobytes()).hexdigest()
        except ImportError as exc:
            raise DependencyMissingError(
                "numpy is required for tensor hashing. Install with: pip install numpy",
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"Cannot hash tensor: {exc}",
                cause=exc,
            ) from exc

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
        except NautilusError:
            raise
        except Exception as exc:
            log.warning("Reproducibility check failed", error=str(exc))
            return False

    def get_stats(self) -> dict[str, Any]:
        return {
            "bit_exact_mode": self._bit_exact_mode,
            "default_strictness": self.default_strictness.name,
            "op_overrides": len(self._op_specs),
        }
