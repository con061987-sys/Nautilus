"""
Foundation types for src.common — the leaf of the internal dependency DAG.

This module exists to break the historical cycle:

    types.py → errors.py, result.py
    types.py → logging.py   (deferred inside __post_init__)
    logging.py → types.py

The fix: any type that the *other* common modules need to import
(``Vendor``, ``Arch``, ``ErrorCode``, ``NautilusError``, ``Result``,
``Ok``, ``Err``, ``HardwareTarget``) lives here, with **zero**
imports from sibling common modules. Everything else stays in its
own module and may import freely from ``primitives`` and from
sibling modules that themselves only depend on ``primitives``.

Import rules
------------
* ``primitives.py`` may only import from the Python standard library.
* ``types.py`` may import from ``primitives`` and (for
  bridge-specific conveniences like ``Vendor.from_string``) from
  ``errors``. It MUST NOT import from ``result`` or ``logging``.
* ``errors.py`` may import from ``primitives`` only.
* ``result.py`` may import from ``primitives`` only.
* ``logging.py`` may import from ``primitives`` and ``types``.

These rules are enforced socially (PR review) and by a smoke test
that imports every module in isolation.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Generic, TypeVar, Union

# ---------------------------------------------------------------------------
# Vendor / architecture enums
# ---------------------------------------------------------------------------


class Vendor(str, Enum):
    """Hardware vendor. Single source of truth for all bridges.

    Lives in ``primitives`` because both ``errors`` (when reporting
    vendor-relative failures) and the bridge-specific ``types`` need
    it. The ``from_string`` convenience that raises ``ConfigError`` on
    unknown input is attached by ``types.py`` to keep
    ``primitives`` free of bridge-specific exceptions.
    """

    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    APPLE = "apple"
    UNKNOWN = "unknown"


class Arch(str, Enum):
    """GPU architecture. Vendor-agnostic identifier (e.g. sm_90, gfx942, xe_hpg)."""

    # NVIDIA
    SM_70 = "sm_70"  # V100
    SM_75 = "sm_75"  # Turing
    SM_80 = "sm_80"  # A100
    SM_86 = "sm_86"  # A100
    SM_89 = "sm_89"  # RTX 4090
    SM_90 = "sm_90"  # H100 Hopper
    SM_100 = "sm_100"  # B100 Blackwell
    SM_120 = "sm_120"  # B200 Blackwell
    # AMD
    GFX900 = "gfx900"  # MI50
    GFX906 = "gfx906"  # MI60
    GFX908 = "gfx908"  # MI100
    GFX90A = "gfx90a"  # MI200 / MI250
    GFX942 = "gfx942"  # MI300X
    GFX950 = "gfx950"  # MI325X
    # Intel
    XE = "intel_gpu_xe"
    XE_LP = "intel_gpu_xelp"
    XE_HPG = "intel_gpu_xehpg"  # Arc
    XE_HPC = "intel_gpu_xehpc"  # Ponte Vecchio
    XE2 = "intel_gpu_xe2"  # Lunar Lake / Battlemage
    GAUDI2 = "intel_gaudi2"
    GAUDI3 = "intel_gaudi3"
    # Apple
    APPLE_M1 = "apple_m1"
    APPLE_M2 = "apple_m2"
    APPLE_M3 = "apple_m3"
    APPLE_M4 = "apple_m4"
    # Generic
    GENERIC = "generic"

    @property
    def vendor(self) -> Vendor:
        v = self.value
        if v.startswith("sm_"):
            return Vendor.NVIDIA
        if v.startswith("gfx"):
            return Vendor.AMD
        if v.startswith("intel") or v.startswith("xe") or v.startswith("gaudi"):
            return Vendor.INTEL
        if v.startswith("apple"):
            return Vendor.APPLE
        return Vendor.UNKNOWN


# ---------------------------------------------------------------------------
# Hardware target
# ---------------------------------------------------------------------------


# (vendor, arch) → TVM target alias. Entries here override the
# default "<prefix>/<arch>" rule in HardwareTarget.to_tvm_target.
TVM_TARGET_ALIASES: dict[tuple[Vendor, Arch], str] = {
    (Vendor.NVIDIA, Arch.SM_90): "nvidia/nvidia-h100",
    (Vendor.NVIDIA, Arch.SM_80): "nvidia/nvidia-a100",
    (Vendor.AMD, Arch.GFX942): "rocm/gfx942",
    (Vendor.INTEL, Arch.GAUDI2): "intel/gaudi-2",
}

_TVM_VENDOR_PREFIX: dict[Vendor, str] = {
    Vendor.NVIDIA: "nvidia",
    Vendor.AMD: "rocm",
    Vendor.INTEL: "intel",
}


@dataclass(frozen=True)
class HardwareTarget:
    """A (vendor, arch) pair, possibly with an alias for downstream tools.

    Lives in ``primitives`` because it is the contract every bridge
    receives and every dispatch path needs to inspect. Only depends on
    ``Vendor`` and ``Arch``.
    """

    vendor: Vendor
    arch: Arch
    alias: str = ""  # e.g. "nvidia/nvidia-h100" for TVM

    def to_tvm_target(self) -> str:
        if self.alias:
            return self.alias
        aliased = TVM_TARGET_ALIASES.get((self.vendor, self.arch))
        if aliased is not None:
            return aliased
        prefix = _TVM_VENDOR_PREFIX.get(self.vendor)
        if prefix is None:
            return "cuda"
        return f"{prefix}/{self.arch.value}"

    def to_triton_target(self) -> str:
        if self.vendor == Vendor.NVIDIA:
            return "cuda"
        if self.vendor == Vendor.AMD:
            return "rocm"
        if self.vendor == Vendor.INTEL:
            return "xpu"
        if self.vendor == Vendor.APPLE:
            return "metal"
        return "cuda"


# ---------------------------------------------------------------------------
# Error codes + base error
# ---------------------------------------------------------------------------


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
    """Base class for all Nautilus errors.

    Concrete subclasses (e.g. ``CompilationError``, ``ConfigError``)
    live in ``src.common.errors`` and inherit from this. The class is
    here in ``primitives`` because ``NautilusError`` is the
    type-narrowing root used by ``Result``'s ``Err`` arm and by the
    ``__cause__`` chaining every error participates in.
    """

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


# ---------------------------------------------------------------------------
# Result[T, E] sum type
# ---------------------------------------------------------------------------

T = TypeVar("T")
E = TypeVar("E", bound=BaseException)
U = TypeVar("U")


@dataclass(frozen=True)
class Ok(Generic[T]):
    """Successful Result carrying value of type T."""

    value: T

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
        return False

    def unwrap(self) -> T:
        return self.value

    def unwrap_or(self, default: T) -> T:
        return self.value

    def unwrap_or_else(self, fn: Callable[[], T]) -> T:
        return self.value

    def map(self, fn: Callable[[T], U]) -> Ok[U]:
        return Ok(fn(self.value))

    def map_err(self, fn: Callable[[E], Any]) -> Ok[T]:
        return self

    def and_then(self, fn: Callable[[T], Result[U, E]]) -> Result[U, E]:
        return fn(self.value)

    def or_else(self, fn: Callable[[E], Result[T, Any]]) -> Ok[T]:
        return self

    def __repr__(self) -> str:
        return f"Ok({self.value!r})"


@dataclass(frozen=True)
class Err(Generic[E]):
    """Failed Result carrying error of type E (must be BaseException)."""

    error: E

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True

    def unwrap(self) -> Any:
        raise self.error

    def unwrap_or(self, default: T) -> T:
        return default

    def unwrap_or_else(self, fn: Callable[[E], T]) -> T:
        return fn(self.error)

    def map(self, fn: Callable[[T], U]) -> Err[E]:
        return self

    def map_err(self, fn: Callable[[E], Any]) -> Err:
        return Err(fn(self.error))

    def and_then(self, fn: Callable[[T], Result[U, E]]) -> Err[E]:
        return self

    def or_else(self, fn: Callable[[E], Result[T, Any]]) -> Result[T, Any]:
        return fn(self.error)

    def __repr__(self) -> str:
        return f"Err({self.error!r})"


# Type alias for the union; spelled out so type checkers accept both arms.
Result = Union[Ok[T], Err[E]]


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "TVM_TARGET_ALIASES",
    "Arch",
    "Err",
    # Errors
    "ErrorCode",
    "HardwareTarget",
    "NautilusError",
    "Ok",
    # Result
    "Result",
    # Vendor / arch / target
    "Vendor",
]
