"""Conversion pipeline orchestrator.

Coordinates the 4 conversion passes, manages error handling, and
produces a final ConversionResult with status, TVMScript text, and
diagnostic information.

This is the public API for the ir_to_tir package. Callers
(bridge_orchestrator, TIRTemplateBuilder) use it to convert captured
Triton IR into TVM-consumable TIR.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from .ttgir_parser import TTGIRFunction, TTGIRParser
from .pass1_lower_tensor_idioms import LowerTensorIdioms
from .pass2_rewrite_spmd import RewriteSPMDToLoops
from .pass3_replace_pointers import ReplacePointersWithMemRefs
from .pass4_materialize_tvm import MaterializeTensorsToTVM
from .tvmscript_emitter import TVMScriptEmitter
from .tt_dot_split import TTDotSplitter, SplitResult

logger = logging.getLogger(__name__)


class ConversionStatus(Enum):
    """Status of a conversion attempt."""
    SUCCESS = auto()              # Full conversion succeeded
    SUCCESS_WITH_DOT = auto()     # Converted with dot split out for extern
    PARTIAL = auto()              # Some passes failed, partial result
    FALLBACK = auto()             # Conversion failed, caller should use template


@dataclass
class ConversionResult:
    """Result of a conversion attempt.

    Carries the TVMScript text, the status, and any split info
    (for kernels with tt.dot that need extern_bridge).
    """
    status: ConversionStatus
    tvmscript_text: str = ""
    split: SplitResult | None = None
    error: str | None = None
    diagnostic: str = ""

    # Per-pass timing in milliseconds
    pass_times: dict[str, float] = field(default_factory=dict)

    @property
    def is_usable(self) -> bool:
        """True if the result can be fed to TVM MetaSchedule."""
        return self.status in (ConversionStatus.SUCCESS, ConversionStatus.SUCCESS_WITH_DOT)

    @property
    def has_dot_split(self) -> bool:
        """True if tt.dot was split out for extern_bridge."""
        return self.split is not None and self.split.has_dot


class ConversionPipeline:
    """Orchestrates the 4-pass conversion.

    Usage:
        pipeline = ConversionPipeline()
        result = pipeline.convert(captured_ir_text)
        if result.is_usable:
            # Feed result.tvmscript_text to TVM
            ...
        else:
            # Fall back to template
            ...
    """

    def __init__(self) -> None:
        self.parser = TTGIRParser()
        self.dot_splitter = TTDotSplitter()
        self.pass1 = LowerTensorIdioms()
        self.pass2 = RewriteSPMDToLoops()
        self.pass3 = ReplacePointersWithMemRefs()
        self.pass4 = MaterializeTensorsToTVM()
        self.emitter = TVMScriptEmitter()

    def convert(self, ir_text: str) -> ConversionResult:
        """Run the full conversion pipeline on IR text.

        Args:
            ir_text: The TTGIR text captured from Triton's pipeline.

        Returns:
            A ConversionResult with the TVMScript text (if successful)
            and any split info for extern_bridge.
        """
        import time

        pass_times: dict[str, float] = {}

        # Stage 1: Parse
        t0 = time.perf_counter()
        try:
            func = self.parser.parse(ir_text)
        except Exception as exc:
            logger.warning("TTGIR parse failed: %s", exc)
            return ConversionResult(
                status=ConversionStatus.FALLBACK,
                error=f"Parse failed: {exc}",
                diagnostic="Could not parse the IR text into a structured AST",
            )
        pass_times["parse"] = (time.perf_counter() - t0) * 1000

        # Stage 2: tt.dot split (before passes — dot is opaque to TIR)
        t0 = time.perf_counter()
        split = self.dot_splitter.split(func)
        has_dot = split.has_dot
        if has_dot:
            # Build a new function with the dot removed
            func = TTGIRFunction(
                name=func.name,
                args=func.args,
                ops=split.remainder_ops,
                module_attrs=func.module_attrs,
            )
        pass_times["dot_split"] = (time.perf_counter() - t0) * 1000

        # Stage 3: Apply the 4 passes
        current_func = func
        for pass_name, pass_impl in [
            ("lower_tensor_idioms", self.pass1),
            ("rewrite_spmd", self.pass2),
            ("replace_pointers", self.pass3),
            ("materialize_tvm", self.pass4),
        ]:
            t0 = time.perf_counter()
            try:
                current_func = pass_impl.run(current_func)
            except Exception as exc:
                logger.warning("Pass %s failed: %s", pass_name, exc)
                return ConversionResult(
                    status=ConversionStatus.FALLBACK,
                    error=f"Pass {pass_name} failed: {exc}",
                    split=split if has_dot else None,
                    diagnostic=f"Conversion pass '{pass_name}' threw an exception",
                )
            pass_times[pass_name] = (time.perf_counter() - t0) * 1000

        # Stage 4: Emit TVMScript
        t0 = time.perf_counter()
        try:
            tvmscript = self.emitter.emit(current_func)
        except Exception as exc:
            logger.warning("TVMScript emission failed: %s", exc)
            return ConversionResult(
                status=ConversionStatus.FALLBACK,
                error=f"Emission failed: {exc}",
                split=split if has_dot else None,
                diagnostic="The emitter could not produce TVMScript text",
            )
        pass_times["emit"] = (time.perf_counter() - t0) * 1000

        status = (
            ConversionStatus.SUCCESS_WITH_DOT if has_dot
            else ConversionStatus.SUCCESS
        )
        result = ConversionResult(
            status=status,
            tvmscript_text=tvmscript,
            split=split if has_dot else None,
            pass_times=pass_times,
        )
        logger.info(
            "Conversion succeeded: status=%s, dot=%s, passes=%s",
            status.name, has_dot, list(pass_times.keys()),
        )
        return result
