"""Triton ↔ TVM MetaSchedule bridge for the Nautilus project.

This is the Phase 1 bridge that wires Triton (the kernel language) to
TVM MetaSchedule (the auto-tuning engine) for cross-vendor AI
compilation.

Architecture (production-grade):
  Python layer (this package):
    bridge_orchestrator.py — main coordinator, public API
    metadata_extractor.py  — fallback metadata extraction (Python-level)
    ir_capture.py          — real IR capture from Triton pipeline
    ir_classifier.py       — classifies captured IR by op structure
    bounds_extractor.py    — extracts M/N/K from real TTGIR
    tir_template.py        — TIR PrimFunc construction
    metaschedule_adapter.py — TVM MetaSchedule integration
    config_mapper.py       — maps TVM traces → Triton Config
    extern_bridge.py       — tvm.extern integration for tt.dot
    circuit_breaker.py     — per-dependency failure isolation
    timeout_manager.py     — per-stage timeouts
    structured_logging.py  — span/stage tracking for observability

  C++ plugin layer (lib/):
    triton_tvm_plugin.cpp — MLIR pass that dumps IR to disk
    ttgir_capture.cpp     — file-based IR capture implementation

  Triton integration:
    backend/compiler.py   — TVMBackend (out-of-tree Triton backend)
    backend/driver.py     — TVMDriver (runtime driver)
    backend/options.py    — TVMOptions (backend options)
    backend/hooks.py      — pipeline inspection hooks

Discovery: Triton finds this backend via the triton.backends entry
point in pyproject.toml, or via the TRITON_PLUGIN_DIRS env var.
"""

from .bridge_orchestrator import TritonTVMBridge, TuningResult, FallbackTier
from .ir_capture import IRCapture, CapturedKernelIR, KernelKind, IRBounds
from .ir_classifier import IRClassifier
from .bounds_extractor import BoundsExtractor
from .metadata_extractor import MetadataExtractor, KernelMetadata
from .tir_template import TIRTemplateBuilder
from .metaschedule_adapter import MetaScheduleAdapter
from .config_mapper import ConfigMapper, MappedTuningConfig
from .extern_bridge import ExternMatmulBuilder, CompiledMatmul
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    CircuitOpenError,
    get_default_breakers,
)
from .timeout_manager import (
    TimeoutManager,
    StageBudgets,
    StageTimeoutError,
    TotalBudgetExceededError,
)
from .structured_logging import (
    Span,
    StageLog,
    span as span_context,
    stage as stage_context,
    configure_logging,
)
from .ir_to_tir import (
    ConversionPipeline,
    ConversionResult,
    ConversionStatus,
    TTGIRParser,
    TTGIRFunction,
    LowerTensorIdioms,
    RewriteSPMDToLoops,
    ReplacePointersWithMemRefs,
    MaterializeTensorsToTVM,
    TVMScriptEmitter,
    TTDotSplitter,
    SplitResult,
)

__all__ = [
    # Main orchestrator
    "TritonTVMBridge",
    "TuningResult",
    "FallbackTier",
    # IR capture
    "IRCapture",
    "CapturedKernelIR",
    "KernelKind",
    "IRBounds",
    "IRClassifier",
    "BoundsExtractor",
    # Metadata (fallback)
    "MetadataExtractor",
    "KernelMetadata",
    # TIR and MetaSchedule
    "TIRTemplateBuilder",
    "MetaScheduleAdapter",
    "ConfigMapper",
    "MappedTuningConfig",
    # Matmul extern bridge
    "ExternMatmulBuilder",
    "CompiledMatmul",
    "ConversionPipeline",
    "ConversionResult",
    "ConversionStatus",
    "TTGIRParser",
    "TTGIRFunction",
    "LowerTensorIdioms",
    "RewriteSPMDToLoops",
    "ReplacePointersWithMemRefs",
    "MaterializeTensorsToTVM",
    "TVMScriptEmitter",
    "TTDotSplitter",
    "SplitResult",
    # Production infrastructure
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "CircuitOpenError",
    "get_default_breakers",
    "TimeoutManager",
    "StageBudgets",
    "StageTimeoutError",
    "TotalBudgetExceededError",
    "Span",
    "StageLog",
    "span_context",
    "stage_context",
    "configure_logging",
]
