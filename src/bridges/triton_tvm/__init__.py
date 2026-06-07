"""src.bridges.triton_tvm — Triton ↔ TVM MetaSchedule bridge (lazy imports).

The actual classes are imported on first access so that this package
can be imported in environments without torch/triton/tvm (e.g. CI
that only runs the common tests, or the CLI verify command).

The previous design imported everything at module load, which broke
test collection when torch was missing.
"""

from __future__ import annotations

_LAZY_EXPORTS = {
    "TritonTVMBridge": "src.bridges.triton_tvm.bridge_orchestrator",
    "TuningResult": "src.bridges.triton_tvm.bridge_orchestrator",
    "FallbackTier": "src.bridges.triton_tvm.bridge_orchestrator",
    "IRCapture": "src.bridges.triton_tvm.ir_capture",
    "CapturedKernelIR": "src.bridges.triton_tvm.ir_capture",
    "KernelKind": "src.bridges.triton_tvm.ir_capture",
    "IRBounds": "src.bridges.triton_tvm.ir_capture",
    "IRClassifier": "src.bridges.triton_tvm.ir_classifier",
    "BoundsExtractor": "src.bridges.triton_tvm.bounds_extractor",
    "MetadataExtractor": "src.bridges.triton_tvm.metadata_extractor",
    "KernelMetadata": "src.bridges.triton_tvm.metadata_extractor",
    "TIRTemplateBuilder": "src.bridges.triton_tvm.tir_template",
    "MetaScheduleAdapter": "src.bridges.triton_tvm.metaschedule_adapter",
    "FusionAnalyzer": "src.bridges.triton_tvm.fusion_analyzer",
    "PatternMatch": "src.bridges.triton_tvm.fusion_analyzer",
    "ConfigCache": "src.bridges.triton_tvm.config_cache",
    "ConfigMapper": "src.bridges.triton_tvm.config_mapper",
    "MappedTuningConfig": "src.bridges.triton_tvm.config_mapper",
    "ExternMatmulBuilder": "src.bridges.triton_tvm.extern_bridge",
    "CompiledMatmul": "src.bridges.triton_tvm.extern_bridge",
    "ConversionPipeline": "src.bridges.triton_tvm.ir_to_tir",
    "ConversionResult": "src.bridges.triton_tvm.ir_to_tir",
    "ConversionStatus": "src.bridges.triton_tvm.ir_to_tir",
    "TTGIRParser": "src.bridges.triton_tvm.ir_to_tir",
    "TTGIRFunction": "src.bridges.triton_tvm.ir_to_tir",
    "LowerTensorIdioms": "src.bridges.triton_tvm.ir_to_tir",
    "RewriteSPMDToLoops": "src.bridges.triton_tvm.ir_to_tir",
    "ReplacePointersWithMemRefs": "src.bridges.triton_tvm.ir_to_tir",
    "MaterializeTensorsToTVM": "src.bridges.triton_tvm.ir_to_tir",
    "TVMScriptEmitter": "src.bridges.triton_tvm.ir_to_tir",
    "TTDotSplitter": "src.bridges.triton_tvm.ir_to_tir",
    "SplitResult": "src.bridges.triton_tvm.ir_to_tir",
    "SearchStrategy": "src.bridges.triton_tvm.search_strategy",
    "KernelType": "src.bridges.triton_tvm.search_strategy",
    "StrategyRecord": "src.bridges.triton_tvm.search_strategy",
    "get_strategy": "src.bridges.triton_tvm.search_strategy",
    "register_strategy": "src.bridges.triton_tvm.search_strategy",
    "list_strategies": "src.bridges.triton_tvm.search_strategy",
    "save_strategy_report": "src.bridges.triton_tvm.search_strategy",
    "strategy_to_tune_kwargs": "src.bridges.triton_tvm.search_strategy",
    # kernel fusion engine
    "FusionPlanner": "src.bridges.triton_tvm.kernel_fusion",
    "FusionPlan": "src.bridges.triton_tvm.kernel_fusion",
    "FusionCodeGenerator": "src.bridges.triton_tvm.kernel_fusion",
    "OpNode": "src.bridges.triton_tvm.kernel_fusion",
    "OpKind": "src.bridges.triton_tvm.kernel_fusion",
    # observability is now in src.common; keep re-exports for back-compat
    "CircuitBreaker": "src.common.observability",
    "CircuitBreakerConfig": "src.common.observability",
    "CircuitState": "src.common.observability",
    "CircuitOpenError": "src.common.observability",
    "get_default_breakers": "src.common.observability",
    "TimeoutManager": "src.common.observability",
    "StageBudgets": "src.common.observability",
    "StageTimeoutError": "src.common.observability",
    "TotalBudgetExceededError": "src.common.observability",
    "Span": "src.common.logging",
    "StageLog": "src.common.logging",
    "span_context": "src.common.logging",
    "stage_context": "src.common.logging",
    "configure_logging": "src.common.logging",
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        import importlib

        module = importlib.import_module(_LAZY_EXPORTS[name])
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(_LAZY_EXPORTS.keys())


__all__ = list(_LAZY_EXPORTS.keys())
