"""
NautilusError — Stable error hierarchy for the entire framework.

Every error has:
  - A stable string code (so C-API can round-trip and so logs are
    machine-searchable across versions)
  - A human-readable message
  - A optional `cause` chain
  - Optional structured context (vendor, arch, kernel_name, etc.)

Anti-pattern eliminated: silent fallbacks that return 0 / None / ""
on failure. From now on, every failure is loud, typed, and contains
the context needed to fix it.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """Stable error codes for cross-version and cross-language compatibility."""

    # Compilation (Phase 2)
    COMPILATION_FAILED = "E_COMPILATION_FAILED"
    COMPILATION_TIMEOUT = "E_COMPILATION_TIMEOUT"
    COMPILATION_OUTPUT_MISSING = "E_COMPILATION_OUTPUT_MISSING"
    COMPILATION_INVALID_IR = "E_COMPILATION_INVALID_IR"
    COMPILATION_UNSUPPORTED_ARCH = "E_COMPILATION_UNSUPPORTED_ARCH"
    COMPILATION_UNSUPPORTED_DTYPE = "E_COMPILATION_UNSUPPORTED_DTYPE"

    # Tuning (Phase 1)
    TUNING_FAILED = "E_TUNING_FAILED"
    TUNING_TIMEOUT = "E_TUNING_TIMEOUT"
    TUNING_NO_RECORDS = "E_TUNING_NO_RECORDS"
    TUNING_INVALID_KERNEL = "E_TUNING_INVALID_KERNEL"

    # Sharding (Phase 3)
    SHARDING_FAILED = "E_SHARDING_FAILED"
    SHARDING_INVALID_MESH = "E_SHARDING_INVALID_MESH"
    GSPMD_FAILED = "E_GSPMD_FAILED"
    GSPMD_TIMEOUT = "E_GSPMD_TIMEOUT"
    DTENSOR_APPLY_FAILED = "E_DTENSOR_APPLY_FAILED"

    # Hardware
    HARDWARE_NOT_FOUND = "E_HARDWARE_NOT_FOUND"
    HARDWARE_PROBE_FAILED = "E_HARDWARE_PROBE_FAILED"
    NO_GPU_AVAILABLE = "E_NO_GPU_AVAILABLE"
    MIXED_VENDOR_RUNTIME = "E_MIXED_VENDOR_RUNTIME"

    # Dependencies
    DEPENDENCY_MISSING = "E_DEPENDENCY_MISSING"
    DEPENDENCY_VERSION_MISMATCH = "E_DEPENDENCY_VERSION_MISMATCH"
    LLVM_MISSING = "E_LLVM_MISSING"
    LLD_MISSING = "E_LLD_MISSING"
    AOTRITON_MISSING = "E_AOTRITON_MISSING"
    TORCH_XLA_MISSING = "E_TORCH_XLA_MISSING"
    TRITON_MISSING = "E_TRITON_MISSING"
    TVM_MISSING = "E_TVM_MISSING"

    # Ingestion (Phase 4)
    INGESTION_FAILED = "E_INGESTION_FAILED"
    INGESTION_PARSE_ERROR = "E_INGESTION_PARSE_ERROR"
    INGESTION_UNSUPPORTED_INTRINSIC = "E_INGESTION_UNSUPPORTED_INTRINSIC"
    INGESTION_INVALID_POINTER = "E_INGESTION_INVALID_POINTER"

    # IR conversion
    IR_CONVERSION_FAILED = "E_IR_CONVERSION_FAILED"
    IR_PARSE_ERROR = "E_IR_PARSE_ERROR"
    IR_LOWERING_FAILED = "E_IR_LOWERING_FAILED"
    IR_DIALECT_MISMATCH = "E_IR_DIALECT_MISMATCH"

    # StableHLO / PyTorch
    STABLEHLO_EXPORT_FAILED = "E_STABLEHLO_EXPORT_FAILED"
    GRAPH_CAPTURE_FAILED = "E_GRAPH_CAPTURE_FAILED"
    KERNEL_NOT_FOUND = "E_KERNEL_NOT_FOUND"

    # Linking
    LINKING_FAILED = "E_LINKING_FAILED"
    LINKING_SYMBOL_UNRESOLVED = "E_LINKING_SYMBOL_UNRESOLVED"

    # Validation
    VALIDATION_FAILED = "E_VALIDATION_FAILED"
    VALIDATION_BIT_EXACT_MISMATCH = "E_VALIDATION_BIT_EXACT_MISMATCH"

    # Configuration
    CONFIG_INVALID = "E_CONFIG_INVALID"
    CONFIG_MISSING_KEY = "E_CONFIG_MISSING_KEY"

    # Checkpointing
    CHECKPOINT_FAILED = "E_CHECKPOINT_FAILED"
    CHECKPOINT_CORRUPT = "E_CHECKPOINT_CORRUPT"
    CHECKPOINT_IO = "E_CHECKPOINT_IO"

    # Stage / budget timeouts
    STAGE_TIMEOUT = "E_STAGE_TIMEOUT"
    TOTAL_BUDGET_EXCEEDED = "E_TOTAL_BUDGET_EXCEEDED"

    # Generic
    BRIDGE_ERROR = "E_BRIDGE_ERROR"
    INTERNAL_ERROR = "E_INTERNAL_ERROR"
    NOT_IMPLEMENTED = "E_NOT_IMPLEMENTED"
    CALLBACK_FAILED = "E_CALLBACK_FAILED"


@dataclass
class NautilusError(Exception):
    """Base class for all Nautilus errors."""

    message: str
    code: ErrorCode = ErrorCode.INTERNAL_ERROR
    cause: Exception | None = None
    context: dict[str, Any] = field(default_factory=dict)
    _traceback: str = ""

    def __post_init__(self) -> None:
        super().__init__(self.message)
        # Capture the call-site traceback for debugging
        if not self._traceback:
            self._traceback = "".join(traceback.format_stack(limit=8)[:-1])
        if self.cause is not None and not self.__cause__:
            self.__cause__ = self.cause

    def to_dict(self) -> dict[str, Any]:
        if self.cause is None:
            cause_repr = None
        else:
            try:
                cause_repr = repr(self.cause)
            except Exception:
                cause_repr = "<unprintable>"
        return {
            "type": type(self).__name__,
            "code": self.code.value,
            "message": self.message,
            "cause": cause_repr,
            "context": dict(self.context),
            "traceback": self._traceback,
        }

    def with_context(self, **kwargs: Any) -> NautilusError:
        """Return a copy with additional context. Chainable."""
        new_ctx = dict(self.context)
        new_ctx.update(kwargs)
        return replace(self, context=new_ctx)


# --- Compilation errors ---


@dataclass
class CompilationError(NautilusError):
    code: ErrorCode = ErrorCode.COMPILATION_FAILED


@dataclass
class CompilationTimeoutError(CompilationError):
    code: ErrorCode = ErrorCode.COMPILATION_TIMEOUT


@dataclass
class CompilationOutputMissingError(CompilationError):
    code: ErrorCode = ErrorCode.COMPILATION_OUTPUT_MISSING


@dataclass
class LinkingError(NautilusError):
    code: ErrorCode = ErrorCode.LINKING_FAILED


@dataclass
class BackendTimeoutError(CompilationError):
    code: ErrorCode = ErrorCode.COMPILATION_TIMEOUT


# --- Tuning errors ---


@dataclass
class TuningError(NautilusError):
    code: ErrorCode = ErrorCode.TUNING_FAILED


@dataclass
class TuningTimeoutError(TuningError):
    code: ErrorCode = ErrorCode.TUNING_TIMEOUT


# --- Sharding errors ---


@dataclass
class ShardingError(NautilusError):
    code: ErrorCode = ErrorCode.SHARDING_FAILED


@dataclass
class GSPMDError(ShardingError):
    code: ErrorCode = ErrorCode.GSPMD_FAILED


@dataclass
class GraphCaptureError(ShardingError):
    code: ErrorCode = ErrorCode.GRAPH_CAPTURE_FAILED


@dataclass
class StableHLOExportError(ShardingError):
    code: ErrorCode = ErrorCode.STABLEHLO_EXPORT_FAILED


@dataclass
class DTensorApplyError(ShardingError):
    code: ErrorCode = ErrorCode.DTENSOR_APPLY_FAILED


@dataclass
class KernelNotFoundError(ShardingError):
    code: ErrorCode = ErrorCode.KERNEL_NOT_FOUND


# --- Hardware errors ---


@dataclass
class HardwareNotFoundError(NautilusError):
    code: ErrorCode = ErrorCode.HARDWARE_NOT_FOUND


@dataclass
class HardwareProbeError(NautilusError):
    code: ErrorCode = ErrorCode.HARDWARE_PROBE_FAILED


@dataclass
class NoGPUAvailableError(HardwareNotFoundError):
    code: ErrorCode = ErrorCode.NO_GPU_AVAILABLE


# --- Dependency errors ---


@dataclass
class DependencyMissingError(NautilusError):
    code: ErrorCode = ErrorCode.DEPENDENCY_MISSING


@dataclass
class DependencyVersionMismatchError(NautilusError):
    code: ErrorCode = ErrorCode.DEPENDENCY_VERSION_MISMATCH


@dataclass
class LLVMError(DependencyMissingError):
    code: ErrorCode = ErrorCode.LLVM_MISSING


@dataclass
class LLDError(DependencyMissingError):
    code: ErrorCode = ErrorCode.LLD_MISSING


@dataclass
class AOTritonError(DependencyMissingError):
    code: ErrorCode = ErrorCode.AOTRITON_MISSING


@dataclass
class TorchXLAError(DependencyMissingError):
    code: ErrorCode = ErrorCode.TORCH_XLA_MISSING


@dataclass
class TritonMissingError(DependencyMissingError):
    code: ErrorCode = ErrorCode.TRITON_MISSING


@dataclass
class TVMMissingError(DependencyMissingError):
    code: ErrorCode = ErrorCode.TVM_MISSING


# --- Ingestion errors ---


@dataclass
class IngestionError(NautilusError):
    code: ErrorCode = ErrorCode.INGESTION_FAILED


@dataclass
class IngestionParseError(IngestionError):
    code: ErrorCode = ErrorCode.INGESTION_PARSE_ERROR


@dataclass
class IngestionUnsupportedIntrinsicError(IngestionError):
    code: ErrorCode = ErrorCode.INGESTION_UNSUPPORTED_INTRINSIC


# --- IR conversion errors ---


@dataclass
class IRConversionError(NautilusError):
    code: ErrorCode = ErrorCode.IR_CONVERSION_FAILED


@dataclass
class IRParseError(IRConversionError):
    code: ErrorCode = ErrorCode.IR_PARSE_ERROR


@dataclass
class IRLoweringError(IRConversionError):
    code: ErrorCode = ErrorCode.IR_LOWERING_FAILED


# --- Validation errors ---


@dataclass
class ValidationError(NautilusError):
    code: ErrorCode = ErrorCode.VALIDATION_FAILED


@dataclass
class BitExactMismatchError(ValidationError):
    code: ErrorCode = ErrorCode.VALIDATION_BIT_EXACT_MISMATCH


# --- Configuration errors ---


@dataclass
class ConfigError(NautilusError):
    code: ErrorCode = ErrorCode.CONFIG_INVALID


@dataclass
class CheckpointError(NautilusError):
    code: ErrorCode = ErrorCode.CHECKPOINT_FAILED


@dataclass
class CheckpointCorruptError(CheckpointError):
    code: ErrorCode = ErrorCode.CHECKPOINT_CORRUPT


@dataclass
class CheckpointIOError(CheckpointError):
    code: ErrorCode = ErrorCode.CHECKPOINT_IO


# --- Observability errors ---


@dataclass
class CircuitOpenError(NautilusError):
    code: ErrorCode = ErrorCode.BRIDGE_ERROR
    breaker_name: str = ""
    consecutive_failures: int = 0


@dataclass
class StageTimeoutError(NautilusError):
    code: ErrorCode = ErrorCode.STAGE_TIMEOUT
    stage_name: str = ""
    budget_seconds: float = 0.0


@dataclass
class TotalBudgetExceededError(NautilusError):
    code: ErrorCode = ErrorCode.TOTAL_BUDGET_EXCEEDED
    elapsed_seconds: float = 0.0
    budget_seconds: float = 0.0


# --- Catch-all ---


@dataclass
class BridgeError(NautilusError):
    """Generic bridge error for unspecified failures."""

    code: ErrorCode = ErrorCode.BRIDGE_ERROR


@dataclass
class CallbackError(NautilusError):
    """Raised when a user-supplied callback (e.g. custom reclaim
    callback) raised an exception during execution.

    Distinct from BridgeError so callers can pattern-match on the
    callback failure mode without inspecting messages.
    """

    code: ErrorCode = ErrorCode.CALLBACK_FAILED


# --- Public re-exports for type narrowing ---


__all__ = [
    "AOTritonError",
    "BackendTimeoutError",
    "BitExactMismatchError",
    "BridgeError",
    "CallbackError",
    "CircuitOpenError",
    "CompilationError",
    "CompilationOutputMissingError",
    "CompilationTimeoutError",
    "ConfigError",
    "DTensorApplyError",
    "DependencyMissingError",
    "DependencyVersionMismatchError",
    "ErrorCode",
    "GSPMDError",
    "GraphCaptureError",
    "HardwareNotFoundError",
    "HardwareProbeError",
    "IRConversionError",
    "IRLoweringError",
    "IRParseError",
    "IngestionError",
    "IngestionParseError",
    "IngestionUnsupportedIntrinsicError",
    "KernelNotFoundError",
    "LLDError",
    "LLVMError",
    "LinkingError",
    "NautilusError",
    "NoGPUAvailableError",
    "ShardingError",
    "StableHLOExportError",
    "StageTimeoutError",
    "TVMMissingError",
    "TorchXLAError",
    "TotalBudgetExceededError",
    "TritonMissingError",
    "TuningError",
    "TuningTimeoutError",
    "ValidationError",
]
