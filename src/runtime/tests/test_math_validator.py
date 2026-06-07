"""Tests for src.runtime.math_validator — bit-exact mode + NaN handling.

Covers:
  H-21  insert_rounding_correction is a real IR attribute injector
        with 3 separate helpers (TTGIR, MLIR, LLVM). Idempotent and
        raises on unknown format.
  NaN   All-NaN in both tensors returns (0.0, 0.0, 0.0) in
        bit-exact mode; partial NaN is masked out of the
        per-element ULP calculation; a NaN in only one tensor
        produces a finite-sentinel ULP error so downstream
        tolerance check catches the divergence.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3]))

from src.common.errors import ValidationError
from src.runtime.math_validator import (
    MathValidator,
    StrictnessLevel,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def validator() -> MathValidator:
    return MathValidator()


@pytest.fixture
def bit_exact_validator() -> MathValidator:
    v = MathValidator()
    v.set_bit_exact_mode(True)
    return v


# ---------------------------------------------------------------------------
# NaN handling in _compute_ulp / _compute_errors
# ---------------------------------------------------------------------------


class TestNaNHandling:
    def test_all_nan_bit_exact_returns_zero_tuple(
        self,
        bit_exact_validator: MathValidator,
    ) -> None:
        ref = np.array([float("nan")] * 8)
        act = np.array([float("nan")] * 8)
        abs_e, _, ulp_e = bit_exact_validator._compute_errors(ref, act)
        assert abs_e == 0.0
        assert ulp_e == 0.0

    def test_all_nan_non_bit_exact_falls_through(
        self,
        validator: MathValidator,
    ) -> None:
        """Outside bit-exact mode, all-NaN still produces 0.0
        (no finite baseline to compare against)."""
        ref = np.array([float("nan")] * 4)
        act = np.array([float("nan")] * 4)
        abs_e, _, ulp_e = validator._compute_errors(ref, act)
        assert abs_e == 0.0
        assert ulp_e == 0.0

    def test_partial_nan_masked_out_of_ulp(
        self,
        validator: MathValidator,
    ) -> None:
        """A single NaN position must not poison the ULP max."""
        ref = np.array([1.0, 2.0, float("nan"), 4.0])
        act = np.array([1.0, 2.0, 2.0, 4.0])  # mismatch only at index 2
        abs_e, rel_e, ulp_e = validator._compute_errors(ref, act)
        # The mismatched position is NaN in ref only; the
        # sentinel ULP is finite but huge — bit_exact is off
        # here so the test is "finite, non-NaN".
        assert math.isfinite(ulp_e) or ulp_e == 0.0
        assert math.isfinite(abs_e)

    def test_both_nan_at_position_zero_ulp(
        self,
        validator: MathValidator,
    ) -> None:
        ref = np.array([float("nan"), 1.0, 2.0])
        act = np.array([float("nan"), 1.0, 2.0])
        # Position 0: both NaN → 0 ULP error
        # Positions 1,2: identical → 0 ULP error
        abs_e, rel_e, ulp_e = validator._compute_errors(ref, act)
        assert abs_e == 0.0
        assert rel_e == 0.0
        assert ulp_e == 0.0

    def test_nan_mismatch_yields_finite_sentinel(
        self,
        validator: MathValidator,
    ) -> None:
        """NaN in only one of ref/act at a position is treated
        as a divergence (finite ULP, not inf/NaN, so the
        downstream tolerance check can see it)."""
        ref = np.array([1.0, float("nan"), 3.0])
        act = np.array([1.0, 1.0, 3.0])
        _, _, ulp_e = validator._compute_errors(ref, act)
        assert math.isfinite(ulp_e)
        assert ulp_e > 0.0

    def test_zero_arrays(self, validator: MathValidator) -> None:
        ref = np.zeros(4)
        act = np.zeros(4)
        abs_e, rel_e, ulp_e = validator._compute_errors(ref, act)
        assert abs_e == 0.0
        assert rel_e == 0.0
        assert ulp_e == 0.0

    def test_empty_arrays(self, validator: MathValidator) -> None:
        ref = np.array([])
        act = np.array([])
        assert validator._compute_errors(ref, act) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# H-21: bit-exact mode + IR injection
# ---------------------------------------------------------------------------


class TestBitExactModeGate:
    def test_disabled_passes_through_unchanged(
        self,
        validator: MathValidator,
    ) -> None:
        ir = 'module { "triton_gpu" { } }'
        assert validator.insert_rounding_correction(ir) == ir

    def test_empty_input_passes_through(
        self,
        bit_exact_validator: MathValidator,
    ) -> None:
        assert bit_exact_validator.insert_rounding_correction("") == ""
        assert bit_exact_validator.insert_rounding_correction("   \n") == "   \n"


class TestFormatDetection:
    def test_detect_ttgir(
        self,
        bit_exact_validator: MathValidator,
    ) -> None:
        ir = (
            'module attributes {tt.target = "cuda:0"} {\n'
            "  tt.func @kernel() { tt.load %x : !tt.ptr<f32> }\n"
            "}\n"
        )
        assert bit_exact_validator._detect_ir_format(ir) == "ttgir"

    def test_detect_mlir(
        self,
        bit_exact_validator: MathValidator,
    ) -> None:
        ir = (
            "module {\n"
            "  func.func @kernel() -> f32 {\n"
            "    %x = arith.constant 1.0 : f32\n"
            "    return %x : f32\n"
            "  }\n"
            "}\n"
        )
        assert bit_exact_validator._detect_ir_format(ir) == "mlir"

    def test_detect_llvm(
        self,
        bit_exact_validator: MathValidator,
    ) -> None:
        ir = (
            "; ModuleID = 'kernel.ll'\n"
            'target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"\n'
            'target triple = "x86_64-unknown-linux-gnu"\n'
            "define float @kernel(float %x) #0 {\n"
            "  ret float %x\n"
            "}\n"
            'attributes #0 = { "noimplicitfloat" }\n'
        )
        assert bit_exact_validator._detect_ir_format(ir) == "llvm"

    def test_unknown_raises(
        self,
        bit_exact_validator: MathValidator,
    ) -> None:
        with pytest.raises(ValidationError):
            bit_exact_validator.insert_rounding_correction("just some text")


class TestTTGIRInjection:
    def test_injects_into_existing_module_attrs(
        self,
        bit_exact_validator: MathValidator,
    ) -> None:
        ir = 'module attributes {tt.target = "cuda:0"} {\n  tt.func @kernel() {}\n}\n'
        out = bit_exact_validator.insert_rounding_correction(ir)
        assert 'tt.mode = "ieee"' in out
        assert "nautilus.bit_exact = true" in out
        # Original op preserved
        assert "tt.func @kernel()" in out

    def test_injects_into_plain_module_block(
        self,
        bit_exact_validator: MathValidator,
    ) -> None:
        ir = "module {\n  tt.func @k() {}\n}\n"
        out = bit_exact_validator.insert_rounding_correction(ir)
        assert 'tt.mode = "ieee"' in out
        assert "nautilus.bit_exact = true" in out

    def test_idempotent(
        self,
        bit_exact_validator: MathValidator,
    ) -> None:
        ir = 'module attributes {tt.target = "cuda:0"} {\n  tt.func @kernel() {}\n}\n'
        once = bit_exact_validator.insert_rounding_correction(ir)
        twice = bit_exact_validator.insert_rounding_correction(once)
        # Idempotent: re-running does not duplicate the attribute
        assert once == twice


class TestMLIRInjection:
    def test_injects_arith_fastmath_false(
        self,
        bit_exact_validator: MathValidator,
    ) -> None:
        ir = (
            "module {\n"
            "  func.func @kernel() -> f32 {\n"
            "    %x = arith.constant 1.0 : f32\n"
            "    return %x : f32\n"
            "  }\n"
            "}\n"
        )
        out = bit_exact_validator.insert_rounding_correction(ir)
        assert "arith.fastmath = false" in out
        assert "nautilus.bit_exact = true" in out
        assert "func.func @kernel()" in out

    def test_idempotent(
        self,
        bit_exact_validator: MathValidator,
    ) -> None:
        ir = "module attributes {some.attr = 0} {\n  func.func @k() { return }\n}\n"
        once = bit_exact_validator.insert_rounding_correction(ir)
        twice = bit_exact_validator.insert_rounding_correction(once)
        assert once == twice


class TestLLVMInjection:
    def test_adds_attribute_group_and_references_it(
        self,
        bit_exact_validator: MathValidator,
    ) -> None:
        ir = (
            "; ModuleID = 'kernel.ll'\n"
            'target datalayout = "e-m:e"\n'
            'target triple = "x86_64-unknown-linux-gnu"\n'
            "define float @kernel(float %x) {\n"
            "  ret float %x\n"
            "}\n"
        )
        out = bit_exact_validator.insert_rounding_correction(ir)
        # Attribute group with strict-fp flags present
        assert '"noimplicitfloat"' in out
        assert '"denormal-fp-math"="ieee,strict"' in out
        assert '"strict-fp"' in out
        assert "nautilus.bit_exact" in out
        # define line now references the group
        assert "@kernel(float %x) #0 {" in out
        # The attribute group definition is in the output
        assert "attributes #0 =" in out

    def test_idempotent_on_annotated_ir(
        self,
        bit_exact_validator: MathValidator,
    ) -> None:
        ir = "; ModuleID = 'kernel.ll'\ndefine float @kernel(float %x) {\n  ret float %x\n}\n"
        once = bit_exact_validator.insert_rounding_correction(ir)
        twice = bit_exact_validator.insert_rounding_correction(once)
        assert once == twice

    def test_does_not_clobber_existing_attr_group(
        self,
        bit_exact_validator: MathValidator,
    ) -> None:
        ir = (
            "; ModuleID = 'kernel.ll'\n"
            "define float @kernel(float %x) #5 {\n"
            "  ret float %x\n"
            "}\n"
            'attributes #5 = { "nounwind" }\n'
        )
        out = bit_exact_validator.insert_rounding_correction(ir)
        # Existing attr group preserved
        assert 'attributes #5 = { "nounwind" }' in out
        # The function reference still points to #5 (not 0)
        assert "@kernel(float %x) #5 {" in out
        # The new group was still emitted
        assert "attributes #0 =" in out


# ---------------------------------------------------------------------------
# validate_kernel_output with bit_exact + NaN
# ---------------------------------------------------------------------------


class TestValidateKernelOutput:
    def test_all_nan_in_bit_exact_passes(
        self,
        bit_exact_validator: MathValidator,
    ) -> None:
        report = bit_exact_validator.validate_kernel_output(
            kernel_name="my_kernel",
            reference=np.array([float("nan")] * 4),
            actual=np.array([float("nan")] * 4),
        )
        assert report.bit_exact is True
        assert report.max_abs_error == 0.0
        assert report.max_ulp_error == 0.0

    def test_bit_exact_detects_real_mismatch(
        self,
        bit_exact_validator: MathValidator,
    ) -> None:
        bit_exact_validator.set_op_strictness(
            "my_kernel",
            StrictnessLevel.BIT_EXACT,
        )
        report = bit_exact_validator.validate_kernel_output(
            kernel_name="my_kernel",
            reference=np.array([1.0, 2.0, 3.0]),
            actual=np.array([1.0, 2.0, 3.0000001]),
        )
        assert report.bit_exact is False
        assert report.max_ulp_error > 0.0

    def test_tolerant_passes_within_ulp(
        self,
        validator: MathValidator,
    ) -> None:
        validator.set_op_strictness(
            "k",
            StrictnessLevel.ULP_16,
        )
        ref = np.array([1.0, 2.0, 3.0])
        act = ref + np.finfo(np.float64).eps  # 1 ULP away
        report = validator.validate_kernel_output(
            kernel_name="k",
            reference=ref,
            actual=act,
        )
        assert report.bit_exact is True, (
            f"Expected bit_exact (1 ULP within ULP_16), got max_ulp_error={report.max_ulp_error}"
        )
