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
    NautilusError,
    CompilationError,
    TuningError,
    ShardingError,
    HardwareNotFoundError,
    DependencyMissingError,
    IngestionError,
    ValidationError,
    ConfigError,
    BridgeError,
    KernelNotFoundError,
    GraphCaptureError,
    StableHLOExportError,
    GSPMDError,
    LinkingError,
    BackendTimeoutError,
    CircuitOpenError,
    StageTimeoutError,
    TotalBudgetExceededError,
    IRConversionError,
)
from src.common.types import (
    Result,
    Ok,
    Err,
    Vendor,
    Arch,
    KernelHandle,
    KernelSection,
    SectionFormat,
    FatBinary,
    ShardingSpecLite,
    TensorShardingLite,
    MeshShape,
    HardwareTarget,
    TuningConfig,
    IRModule,
    StableHLOModule,
    SourceLocation,
    SpanRecord,
    StageRecord,
    LogLevel,
)
from src.common.hardware import (
    detect_host_vendor,
    detect_gpu_vendors,
    get_device_paths,
    has_nvidia_gpu,
    has_amd_gpu,
    has_intel_gpu,
    has_apple_gpu,
    enumerate_devices,
    DeviceInfo,
    HostInfo,
    CpuVendor,
    GpuVendor,
    probe_pcie_for_gpus,
)
from src.common.logging import (
    configure_logging,
    get_logger,
    span as span_context,
    stage as stage_context,
    Span,
    StageLog,
    LogSink,
    JsonLogSink,
    StdoutLogSink,
    NullLogSink,
    CompositeLogSink,
)
from src.common.observability import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    TimeoutManager,
    StageBudgets,
)

__version__ = "0.1.0"
__all__ = [
    "__version__",
    # errors
    "NautilusError",
    "CompilationError",
    "TuningError",
    "ShardingError",
    "HardwareNotFoundError",
    "DependencyMissingError",
    "IngestionError",
    "ValidationError",
    "ConfigError",
    "BridgeError",
    "KernelNotFoundError",
    "GraphCaptureError",
    "StableHLOExportError",
    "GSPMDError",
    "LinkingError",
    "BackendTimeoutError",
    "CircuitOpenError",
    "StageTimeoutError",
    "TotalBudgetExceededError",
    "IRConversionError",
    # types
    "Result",
    "Ok",
    "Err",
    "Vendor",
    "Arch",
    "KernelHandle",
    "KernelSection",
    "SectionFormat",
    "FatBinary",
    "ShardingSpecLite",
    "TensorShardingLite",
    "MeshShape",
    "HardwareTarget",
    "TuningConfig",
    "IRModule",
    "StableHLOModule",
    "SourceLocation",
    "SpanRecord",
    "StageRecord",
    "LogLevel",
    # hardware
    "detect_host_vendor",
    "detect_gpu_vendors",
    "get_device_paths",
    "has_nvidia_gpu",
    "has_amd_gpu",
    "has_intel_gpu",
    "has_apple_gpu",
    "enumerate_devices",
    "DeviceInfo",
    "HostInfo",
    "CpuVendor",
    "GpuVendor",
    "probe_pcie_for_gpus",
    # logging
    "configure_logging",
    "get_logger",
    "span_context",
    "stage_context",
    "Span",
    "StageLog",
    "LogSink",
    "JsonLogSink",
    "StdoutLogSink",
    "NullLogSink",
    "CompositeLogSink",
    # observability
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "TimeoutManager",
    "StageBudgets",
]
