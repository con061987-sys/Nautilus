"""
src.common — Shared types, errors, hardware detection, logging.

This is the foundation module that ALL bridges depend on. It eliminates
the cross-bridge coupling that previously had pytorch_xla importing
triton_tvm internals (circuit_breaker, timeout_manager, structured_logging).

Modules
-------
types
    Vendor-neutral type definitions: Result[T, E], fat-binary section
    records, kernel handles, sharding specs.
errors
    NautilusError hierarchy with stable string codes (so C-API can
    surface them to Python and vice versa).
hardware
    Vendor detection via /dev probing and CPUID (Intel/AMD/Nvidia/Apple).
    No silent "return 0" — every probe either succeeds with evidence or
    raises HardwareNotFoundError.
logging
    Structured JSON logger with stage/span tracking, moved here from
    bridges/triton_tvm/structured_logging.py.
observability
    Circuit breaker, timeout manager — moved here from
    bridges/triton_tvm/. Pytorch_xla now imports from src.common, not
    from src.bridges.triton_tvm.
"""

from __future__ import annotations

from src.common.errors import (
    BackendTimeoutError,
    BridgeError,
    CircuitOpenError,
    CompilationError,
    ConfigError,
    DependencyMissingError,
    GraphCaptureError,
    GSPMDError,
    HardwareNotFoundError,
    IngestionError,
    IRConversionError,
    KernelNotFoundError,
    LinkingError,
    NautilusError,
    ShardingError,
    StableHLOExportError,
    StageTimeoutError,
    TotalBudgetExceededError,
    TuningError,
    ValidationError,
)
from src.common.hardware import (
    CpuVendor,
    DeviceInfo,
    GpuVendor,
    HostInfo,
    detect_gpu_vendors,
    detect_host_vendor,
    enumerate_devices,
    get_device_paths,
    has_amd_gpu,
    has_apple_gpu,
    has_intel_gpu,
    has_nvidia_gpu,
    probe_pcie_for_gpus,
)
from src.common.logging import (
    CompositeLogSink,
    JsonLogSink,
    LogSink,
    NullLogSink,
    Span,
    StageLog,
    StdoutLogSink,
    configure_logging,
    get_logger,
)
from src.common.logging import (
    span as span_context,
)
from src.common.logging import (
    stage as stage_context,
)
from src.common.observability import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    StageBudgets,
    TimeoutManager,
)
from src.common.types import (
    Arch,
    Err,
    FatBinary,
    HardwareTarget,
    IRModule,
    KernelHandle,
    KernelSection,
    LogLevel,
    MeshShape,
    Ok,
    Result,
    SectionFormat,
    ShardingSpecLite,
    SourceLocation,
    SpanRecord,
    StableHLOModule,
    StageRecord,
    TensorShardingLite,
    TuningConfig,
    Vendor,
)

__version__ = "0.1.0"
__all__ = [
    "Arch",
    "BackendTimeoutError",
    "BridgeError",
    # observability
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitOpenError",
    "CircuitState",
    "CompilationError",
    "CompositeLogSink",
    "ConfigError",
    "CpuVendor",
    "DependencyMissingError",
    "DeviceInfo",
    "Err",
    "FatBinary",
    "GSPMDError",
    "GpuVendor",
    "GraphCaptureError",
    "HardwareNotFoundError",
    "HardwareTarget",
    "HostInfo",
    "IRConversionError",
    "IRModule",
    "IngestionError",
    "JsonLogSink",
    "KernelHandle",
    "KernelNotFoundError",
    "KernelSection",
    "LinkingError",
    "LogLevel",
    "LogSink",
    "MeshShape",
    # errors
    "NautilusError",
    "NullLogSink",
    "Ok",
    # types
    "Result",
    "SectionFormat",
    "ShardingError",
    "ShardingSpecLite",
    "SourceLocation",
    "Span",
    "SpanRecord",
    "StableHLOExportError",
    "StableHLOModule",
    "StageBudgets",
    "StageLog",
    "StageRecord",
    "StageTimeoutError",
    "StdoutLogSink",
    "TensorShardingLite",
    "TimeoutManager",
    "TotalBudgetExceededError",
    "TuningConfig",
    "TuningError",
    "ValidationError",
    "Vendor",
    "__version__",
    # logging
    "configure_logging",
    "detect_gpu_vendors",
    # hardware
    "detect_host_vendor",
    "enumerate_devices",
    "get_device_paths",
    "get_logger",
    "has_amd_gpu",
    "has_apple_gpu",
    "has_intel_gpu",
    "has_nvidia_gpu",
    "probe_pcie_for_gpus",
    "span_context",
    "stage_context",
]
