"""CUDA → Triton ingestion bridge (Phase 4 of Nautilus).

This package takes standard CUDA C++ (.cu) kernels and translates
them into Triton Python source code, which can then be compiled
through the Phase 1/2 pipeline to run on any supported hardware
(Nvidia, AMD, Intel, Apple).

Architecture (as of tree-sitter rewrite):
    [CUDA C++ Source (.cu)]
        │
        ▼
    [parser.py — tree-sitter AST] ──► [CudaKernel AST]
        │   Uses tree-sitter-cpp for full C++ grammar
        │   support: templates, namespaces, complex loops
        │   - function signatures + typed parameters
        │   - body statements with AST metadata
        │   - shared memory declarations
        │   - CUDA field expressions (threadIdx/blockIdx/blockDim/gridDim)
        │
        ▼
    [intrinsic_mapper.py] ──► [Mapping Table]
        │   Used as a *lookup table* by the translator,
        │   not for blind text-level replacement.
        │   - __syncthreads → tl.barrier
        │   - atomicAdd → tl.atomic_add
        │   - threadIdx.x → tl.program_id(0)
        │
        ▼
    [shared_memory.py] ──► [Shared Memory Plan]
        │   - __shared__ arrays → tl.zeros allocations
        │   - bank conflict analysis
        │
        ▼
    [pointer_analysis.py] ──► [Pointer Layout]
        │   - pointer arithmetic → block pointers
        │   - boundary checks
        │   - coalescing analysis
        │
        ▼
    [translator.py — AST-level dispatch] ──► [Triton Source]
        │   Dispatches per-statement using CudaStatementType
        │   and AST metadata.  No silent no-ops: unsupported
        │   intrinsics raise IngestionUnsupportedIntrinsicError.
        │   - @triton.jit decorated function
        │   - proper block/thread hierarchy (approximate)
        │   - type-cast insertion for atomic args
        │
        ▼
    [kernel_compiler.py] ──► [Compiled Fat Binary]
        │   - feeds Triton source to Phase 1/2
        │   - compiles for all targets
        │   - produces fat binary

Key changes in the tree-sitter rewrite (2026-06):
  - parser.py now uses `tree-sitter-cpp` for AST-level parsing
    rather than fragile regex patterns
  - Handles template parameters, namespaces, function pointers,
    extern "C" blocks, and complex for-loops natively
  - translator.py uses StmtType + metadata for dispatch instead
    of blind text-level replacement via intrinsic_mapper.transform_text()
  - Unsupported constructs raise typed errors instead of
    silently producing no-ops

Modules:
  parser.py             - CUDA C++ source parser (tree-sitter-based)
  intrinsic_mapper.py   - CUDA intrinsic → Triton mapping table
  shared_memory.py      - Shared memory analysis and translation
  pointer_analysis.py   - Pointer arithmetic analysis
  translator.py         - CUDA AST → Triton Python source (AST dispatch)
  kernel_compiler.py    - End-to-end CUDA kernel compilation
"""

from .intrinsic_mapper import IntrinsicMapper, IntrinsicMapping
from .kernel_compiler import CompilationResult, CudaKernelCompiler
from .parser import CudaKernel, CudaParser, CudaStatement, TreeSitterCudaParser
from .pointer_analysis import PointerAnalyzer, PointerLayout
from .shared_memory import SharedMemoryAnalyzer, SharedMemPlan
from .translator import CudaToTritonTranslator, TranslationResult

__all__ = [
    "CudaParser",
    "CudaKernel",
    "CudaStatement",
    "IntrinsicMapper",
    "IntrinsicMapping",
    "SharedMemoryAnalyzer",
    "SharedMemPlan",
    "PointerAnalyzer",
    "PointerLayout",
    "CudaToTritonTranslator",
    "TranslationResult",
    "CudaKernelCompiler",
    "CompilationResult",
]
