---
name: triton-compiler
description: Deep knowledge of OpenAI Triton compiler internals, TTGIR dialect, kernel compilation pipeline, autotuning interface, and backend architecture. Use when writing Triton kernels, analyzing Triton IR, or working with the compilation pipeline.
---

# Triton Compiler Skill

## Overview

OpenAI Triton compiles Python-based kernel definitions into highly optimized GPU code. Its internal compilation pipeline progresses through several IR levels before emitting target machine code.

## Compilation Pipeline

```
Python Kernel Definition
    │
    ▼
AST (Abstract Syntax Tree) — Parsed from Python source
    │
    ▼
Triton IR (TTGIR) — Triton-specific MLIR dialect
    │  - High-level ops: tt.matmul, tt.add, tt.load, tt.store
    │  - Memory layout annotations: tt.all_shared, tt.all_distributed
    │
    ▼
Triton GPU IR (TTGIR) — Lowered with GPU-specific concerns
    │  - Thread layout: tt.num_programs, tt.num_warps
    │  - Shared memory allocation
    │  - Memory coalescing annotations
    │
    ▼
LLVM IR — Lowered from TTGIR via TritonToLLVM passes
    │  - All vendor-specific syntax removed
    │  - Pure LLVM operations
    │
    ▼
PTX (Nvidia) / ROCm (AMD) — Target-specific machine code
```

## Key Entry Points

### IR Interception
The ideal interception point for auto-tuning is **after TTGIR generation but before LLVM lowering**. At this stage, the computation structure is fully defined but no hardware-specific choices are committed.

```python
from triton.compiler import compile, ASTSource
from triton.backends.nvidia.compiler import NvidiaBackend

def intercept_ir(kernel_fn, **kwargs):
    # Compile to TTGIR (stop before LLVM lowering)
    compiled = compile(
        ASTSource(kernel_fn),
        target="ttgir",  # Intercept at TTGIR level
        options={"debug": True}
    )
    return compiled.module  # MLIR Module in TTGIR dialect
```

### Compiler Options for Tuning
```python
options = {
    "num_warps": 4,           # Default, tunable
    "num_stages": 3,          # Pipeline stages, tunable
    "num_ctas": 1,            # Thread block clusters (H100+)
    "max_num_imprecise_acc": 0,  # FP8 accumulation precision
    "enable_fp_fusion": True, # FMA fusion
}
```

## Critical Knowledge

1. **Triton uses MLIR internally.** The IR is a custom MLIR dialect (`triton` and `triton_gpu`). You can dump it with `--dump-ttgir` or `--dump-llvm-ir` flags.
2. **Autotuning in upstream Triton** uses `@triton.autotune` decorator which brute-forces configs. We replace this with TVM MetaSchedule.
3. **Triton's JIT cache** stores compiled kernels in `~/.triton/cache/`. For AOT compilation, we bypass the cache.
4. **Backend-specific code** lives in `triton/backends/<vendor>/`. Each backend handles its own IR lowering to target assembly.
5. **Triton 3.0+** has experimental AOT compilation support via `triton.compile` with `target="llvm"` or binary output.

## When This Skill Triggers

- Writing or modifying Triton kernels
- Working with the `src/bridges/triton_tvm/` bridge code
- Debugging Triton IR output or compilation errors
- Adding new kernel operations or types
- Patching the Triton JIT/AOT compilation pipeline
