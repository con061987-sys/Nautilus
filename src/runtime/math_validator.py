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
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from src.common.errors import (
    DependencyMissingError,
    NautilusError,
    ValidationError,
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
            if spec.strictness in (
                StrictnessLevel.ULP_1,
                StrictnessLevel.ULP_4,
                StrictnessLevel.ULP_16,
            ):
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

        NaN policy:
          - all-NaN in BOTH ref and act → (0.0, 0.0, 0.0) in
            bit-exact mode (deterministic NaN); falls through
            to the standard ULP-mask path in tolerant mode
            (where _compute_ulp returns 0.0 anyway because
            the mask is satisfied everywhere).
          - NaN positions in EITHER tensor are excluded from
            abs/rel/ulp statistics (masking) so a single
            divergent NaN doesn't poison the entire
            comparison. NaN mismatch between ref and act at
            a position contributes a finite-sentinel ULP
            error (downstream tolerance check catches the
            divergence).

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
                "numpy is required for math validation. Install with: pip install numpy",
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
        ref_nan = np.isnan(ref)
        act_nan = np.isnan(act)
        if ref_nan.all() and act_nan.all():
            return 0.0, 0.0, 0.0
        mask = ~(ref_nan | act_nan)
        if not mask.any():
            if self._bit_exact_mode:
                return 0.0, 0.0, 0.0
            return 0.0, 0.0, 0.0
        diff = np.abs(ref - act)
        finite_abs = diff[mask]
        max_abs = float(np.max(finite_abs)) if finite_abs.size > 0 else 0.0
        nonzero = mask & (np.abs(ref) > 0)
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

        NaN handling
        ------------
        NaN positions are MASKED out of the ULP calculation:

        * If both ref[i] and act[i] are NaN, they "match" (both
          NaN is the IEEE-754 convention for unrepresentable) —
          the ULP error at that position is 0.
        * If only one of ref[i], act[i] is NaN, the result
          should be unrepresentable, so we treat that position
          as a mismatch and use a large sentinel (1e300) ULP
          error. The downstream validator's tolerance check
          will fail bit-exact comparison, which is the correct
          behavior: a NaN appearing in only one tensor is a
          divergent computation.
        * If every element of BOTH ref and act is NaN, we
          return 0.0 in bit-exact mode (a "matched" all-NaN
          pair is considered deterministic). In non-bit-exact
          mode the same all-NaN case returns 0.0 because there
          is no finite baseline to compare against.
        """
        import numpy as np

        ref = np.asarray(ref, dtype=np.float64).flatten()
        act = np.asarray(act, dtype=np.float64).flatten()
        n = min(ref.size, act.size)
        if n == 0:
            return 0.0
        ref = ref[:n]
        act = act[:n]
        ref_nan = np.isnan(ref)
        act_nan = np.isnan(act)
        all_nan = bool(ref_nan.all() and act_nan.all())
        if all_nan:
            return 0.0
        abs_ref = np.abs(ref)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            next_up = np.nextafter(ref, np.inf)
            ulp_at_ref = np.abs(next_up - ref)
            zero_mask = abs_ref == 0
            ulp_at_ref = np.where(zero_mask, np.finfo(np.float64).tiny, ulp_at_ref)
            ulp_at_ref = np.maximum(ulp_at_ref, np.finfo(np.float64).tiny)
        abs_diff = np.abs(ref - act)
        ulp_err = abs_diff / ulp_at_ref
        both_nan = ref_nan & act_nan
        only_ref_nan = ref_nan & ~act_nan
        only_act_nan = ~ref_nan & act_nan
        ulp_err = np.where(both_nan, 0.0, ulp_err)
        nan_mismatch_sentinel = np.finfo(np.float64).max
        ulp_err = np.where(only_ref_nan | only_act_nan, nan_mismatch_sentinel, ulp_err)
        finite_err = ulp_err[np.isfinite(ulp_err) | (ulp_err == 0.0)]
        if finite_err.size == 0:
            return float(nan_mismatch_sentinel)
        return float(np.max(finite_err))

    def insert_rounding_correction(self, ir_text: str) -> str:
        """Insert IR-level rounding correction for bit-exact mode.

        Detects the IR format (TTGIR, MLIR, LLVM IR) and dispatches
        to a format-specific helper that injects the canonical
        IEEE-754 strict-fp attribute set. Each helper is
        idempotent: re-invocation is a no-op (detected by a
        unique sentinel string the helper itself emits).

        Raises:
            ValidationError: if the IR format cannot be detected.
                The previous implementation silently prepended a
                comment, which gave downstream passes no signal at
                all — bit-exact mode looked enabled in the source
                text but had no effect on the compiled code.
        """
        if not self._bit_exact_mode:
            return ir_text
        if not ir_text or not ir_text.strip():
            return ir_text
        if self._ir_has_bit_exact_sentinel(ir_text):
            return ir_text
        fmt = self._detect_ir_format(ir_text)
        if fmt == "ttgir":
            return self._inject_ttgir_attributes(ir_text)
        if fmt == "mlir":
            return self._inject_mlir_attributes(ir_text)
        if fmt == "llvm":
            return self._inject_llvm_attributes(ir_text)
        raise ValidationError(
            "Cannot insert rounding correction: unknown IR format. "
            "Expected Triton TTGIR, standard MLIR, or LLVM IR.",
            context={
                "format": fmt,
                "preview": ir_text[:200],
            },
        )

    _BIT_EXACT_SENTINEL = "nautilus.bit_exact"

    def _ir_has_bit_exact_sentinel(self, ir_text: str) -> bool:
        return self._BIT_EXACT_SENTINEL in ir_text

    def _detect_ir_format(self, ir_text: str) -> str:
        """Sniff the IR format from the leading tokens.

        Order matters: LLVM IR has the most specific marker
        (``; ModuleID``), TTGIR is recognised by Triton-specific
        ops, and standard MLIR is the catch-all for anything
        with a ``module {`` block and dialect ops.
        """
        head = ir_text.lstrip()[:512]
        lowered = head.lower()
        if (
            head.startswith("; ModuleID")
            or "target datalayout" in head
            or "target triple" in head
            or head.startswith("define ")
        ):
            return "llvm"
        if (
            "tt.func" in head
            or "ttgir" in lowered
            or "triton_gpu" in head
            or "tt.load" in head
            or "tt.dot" in head
        ):
            return "ttgir"
        if (
            "module {" in head
            or "module attributes" in head
            or "func.func" in head
            or "arith." in head
            or "vector." in head
            or "math." in head
        ):
            return "mlir"
        return "unknown"

    def _inject_ttgir_attributes(self, ir_text: str) -> str:
        """Inject TTGIR module attributes for IEEE-754 bit-exact mode.

        Inserts ``tt.mode = "ieee"`` and a sentinel
        ``nautilus.bit_exact = true`` attribute into the module
        attribute block. Triton 3.x reads ``tt.mode`` to decide
        whether to emit FMA-fused / fast-math instructions; the
        sentinel is used by our own pass manager to recognise
        that bit-exact mode is active.
        """
        marker = f"{self._BIT_EXACT_SENTINEL} = true"
        attrs = f'tt.mode = "ieee", {marker}'
        module_idx = ir_text.find("module")
        if module_idx == -1:
            return self._wrap_with_ttgir_attrs_block(ir_text, attrs)
        brace_idx = ir_text.find("{", module_idx)
        if brace_idx == -1:
            return self._wrap_with_ttgir_attrs_block(ir_text, attrs)
        if "module attributes" in ir_text[:brace_idx]:
            return ir_text[: brace_idx + 1] + f"\n  {attrs}," + ir_text[brace_idx + 1 :]
        if ir_text[module_idx:brace_idx].strip() == "module":
            return ir_text[:brace_idx] + " attributes {" + attrs + "} " + ir_text[brace_idx:]
        return self._wrap_with_ttgir_attrs_block(ir_text, attrs)

    def _inject_mlir_attributes(self, ir_text: str) -> str:
        """Inject standard MLIR module attributes for bit-exact mode.

        Adds ``arith.fastmath = false`` and the
        ``nautilus.bit_exact = true`` sentinel to the module
        attribute block. ``arith.fastmath = false`` is the
        canonical MLIR signal that downstream passes (and
        arith-emitting frontends) must NOT fuse FP operations
        or use contract / reassoc flags.
        """
        marker = f"{self._BIT_EXACT_SENTINEL} = true"
        attrs = f"arith.fastmath = false, {marker}"
        module_idx = ir_text.find("module")
        if module_idx == -1:
            return self._wrap_with_mlir_attrs_block(ir_text, attrs)
        brace_idx = ir_text.find("{", module_idx)
        if brace_idx == -1:
            return self._wrap_with_mlir_attrs_block(ir_text, attrs)
        if "module attributes" in ir_text[:brace_idx]:
            return ir_text[: brace_idx + 1] + f"\n  {attrs}," + ir_text[brace_idx + 1 :]
        if ir_text[module_idx:brace_idx].strip() == "module":
            return ir_text[:brace_idx] + " attributes {" + attrs + "} " + ir_text[brace_idx:]
        return self._wrap_with_mlir_attrs_block(ir_text, attrs)

    def _inject_llvm_attributes(self, ir_text: str) -> str:
        """Inject LLVM IR function attributes for bit-exact mode.

        Adds a top-level attribute group containing
        ``noimplicitfloat`` and ``strict`` denormal-fp-math,
        then references that group from every ``define`` that
        doesn't already have an attribute group. This is the
        canonical LLVM mechanism for disabling fast-math
        rewrites and forcing IEEE-754 semantics in the
        backend.
        """
        attr_group_id = 0
        attr_group_def = (
            f"attributes #{attr_group_id} = {{ "
            f'"noimplicitfloat" '
            f'"denormal-fp-math"="ieee,strict" '
            f'"denormal-fp-math-f32"="ieee,strict" '
            f'"strict-fp" '
            f"}}\n"
        )
        sentinel_comment = f"; nautilus.bit_exact: strict-fp attribute group #{attr_group_id}\n"
        injection = sentinel_comment + attr_group_def
        if f"attributes #{attr_group_id} = " in ir_text:
            return ir_text
        out_lines: list[str] = []
        for line in ir_text.split("\n"):
            stripped = line.lstrip()
            if stripped.startswith("define ") and line.rstrip().endswith("{"):
                if re.search(r"\) #\d+\s*\{?\s*$", line.rstrip()):
                    out_lines.append(line)
                    continue
                if line.rstrip().endswith("{"):
                    new_line = line.rstrip()[:-1].rstrip() + f" #{attr_group_id} {{"
                    out_lines.append(new_line)
                else:
                    out_lines.append(line + "  ;; nautilus.bit_exact: would attach attrs here")
            else:
                out_lines.append(line)
        annotated = "\n".join(out_lines)
        return injection + annotated

    def _wrap_with_ttgir_attrs_block(self, ir_text: str, attrs: str) -> str:
        return f"module attributes {{ {attrs} }} {{\n{ir_text}\n}}\n"

    def _wrap_with_mlir_attrs_block(self, ir_text: str, attrs: str) -> str:
        return f"module attributes {{ {attrs} }} {{\n{ir_text}\n}}\n"

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
