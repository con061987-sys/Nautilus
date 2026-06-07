"""TTGIR → TVM TIR conversion pipeline.

This package implements the 4-pass conversion that turns real Triton
GPU IR (TTGIR) into TVM TIR text suitable for MetaSchedule tuning.

The pipeline is the production-grade version of the triton-tvm project
(Yongqi-Zhuo/triton-tvm@254f47c) — a Python implementation that:

  1. Parses real TTGIR text into a structured AST (ttgir_parser.py)
  2. Pass 1: lifts arith/math ops to tensor.generate form
  3. Pass 2: rewrites SPMD primitives to explicit loops
  4. Pass 3: replaces Triton pointer types with memref types
  5. Pass 4: materializes tensors into TVM block/buffer form
  6. Emits TVMScript text the TVM MetaSchedule can consume

Why Python instead of C++: avoids the MLIR C-API dependency, uses
TVM's existing Python TIR API, is unit-testable without a C++ toolchain,
and gives us clearer error messages during development.

For kernels with tt.dot, the pipeline splits them — the matmul portion
goes through ExternMatmulBuilder (preserving Tensor Core performance)
and only the rest of the kernel is converted to TIR.

Modules:
  ttgir_parser.py        - Parse TTGIR text → AST
  pass1_lower_tensor_idioms.py - Lift scalar ops
  pass2_rewrite_spmd.py  - Convert SPMD to loops
  pass3_replace_pointers.py - Pointer → memref
  pass4_materialize_tvm.py  - Tensors → TVM blocks
  tvmscript_emitter.py   - AST → TVMScript text
  tt_dot_split.py        - Split kernels with tt.dot
  conversion_pipeline.py - Orchestrate the 4 passes

Architecture decisions:
  - Each pass is independent and testable in isolation
  - Each pass is pure: takes AST, returns AST (no side effects)
  - The pipeline applies passes in order, with per-pass error handling
  - The emitter is the ONLY module that produces TVM-specific output
  - Failures in any pass trigger a fallback to template-based TIR
"""

from .conversion_pipeline import (
    ConversionPipeline,
    ConversionResult,
    ConversionStatus,
)
from .pass1_lower_tensor_idioms import LowerTensorIdioms
from .pass2_rewrite_spmd import RewriteSPMDToLoops
from .pass3_replace_pointers import ReplacePointersWithMemRefs
from .pass4_materialize_tvm import MaterializeTensorsToTVM
from .tt_dot_split import SplitResult, TTDotSplitter
from .ttgir_parser import TTGIRFunction, TTGIROperation, TTGIRParser, TTGIRType
from .tvmscript_emitter import TVMScriptEmitter

__all__ = [
    # Main pipeline
    "ConversionPipeline",
    "ConversionResult",
    "ConversionStatus",
    # Passes
    "LowerTensorIdioms",
    "MaterializeTensorsToTVM",
    "ReplacePointersWithMemRefs",
    "RewriteSPMDToLoops",
    "SplitResult",
    # tt.dot splitting
    "TTDotSplitter",
    "TTGIRFunction",
    "TTGIROperation",
    # AST
    "TTGIRParser",
    "TTGIRType",
    # Emitter
    "TVMScriptEmitter",
]
