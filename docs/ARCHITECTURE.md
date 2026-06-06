# Nautilus Architecture

This document describes the structure of the Nautilus framework.
For a higher-level overview of the problem and solution, see
[PRD.md](PRD.md) and [TECH_SPEC.md](TECH_SPEC.md).

## Layer overview

```
+------------------------------------------------------------------+
|  User interface layer                                              |
|  CLI (click)  |  Python API  |  PyTorch integration  |  Jupyter   |
+-----------------------------+------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|  Compiler orchestration                                           |
|  +--------------+   +----------------+   +-------------------+   |
|  |  Auto-Tuner  |   |  AOT Packager  |   |  Auto-Sharding     |   |
|  |  (Phase 1)   |-->|  (Phase 2)     |-->|  Engine (Phase 3)  |   |
|  +-------+------+   +-------+--------+   +---------+---------+   |
+----------|------------------|-----------------------|-------------+
           |                  |                       |
           v                  v                       v
+------------------------------------------------------------------+
|  Integration bridge layer                                         |
|  +----------------+  +-----------------+  +------------------+  |
|  |  Triton<->TVM  |  |  AOTriton + LLVM |  |  PyTorch FX<->XLA |  |
|  |  Bridge        |  |  Linker          |  |  Bridge           |  |
|  +----------------+  +-----------------+  +------------------+  |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|  Dependency abstraction                                           |
|  C-API Wrappers  |  Pinned Git Submodules  |  MLIR Normalizer   |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|  Common runtime (src.common)                                      |
|  types | errors | hardware | logging | observability            |
+------------------------------------------------------------------+
```

## Module layout

```
src/
+-- common/                     # Foundation; ALL bridges depend on this
|   +-- types.py                # Vendor, Arch, FatBinary, TuningConfig, ...
|   +-- errors.py               # NautilusError hierarchy with stable codes
|   +-- hardware.py             # /dev probing (real, not return 0)
|   +-- logging.py              # Structured JSON logs with span/stage
|   +-- observability.py        # CircuitBreaker, TimeoutManager
|   +-- result.py               # Result[T, E] for fallible functions
+-- c_api/                      # Version-drift isolation gate
|   +-- triton_c_api.h          # Stable Triton ABI
|   +-- tvm_c_api.h             # Stable TVM MetaSchedule ABI
|   +-- xla_c_api.h             # Stable XLA/StableHLO ABI
|   +-- __init__.py             # Python ctypes bindings
+-- cli/                        # `nautilus` command
|   +-- main.py                 # Top-level click group
|   +-- commands/
|       +-- tune.py             # `nautilus tune`
|       +-- build.py            # `nautilus build`
|       +-- shard.py            # `nautilus shard`
|       +-- verify.py           # `nautilus verify`
+-- runtime/                    # Long-running runtime components
|   +-- memory_reclaimer.py     # Vendor-specific allocator flush
|   +-- async_checkpointer.py   # Atomic, checksummed checkpoints
|   +-- math_validator.py       # Real ULP error analysis
+-- bridges/                    # Cross-system integration code
|   +-- triton_tvm/             # Phase 1: Auto-tuning
|   +-- aot_packager/           # Phase 2: Fat binary
|   +-- pytorch_xla/            # Phase 3: Auto-sharding
|   +-- cuda_ingest/            # Phase 4: CUDA C++ translation
+-- tests/                      # Cross-bridge integration tests
```

## The 4 phases

### Phase 1: Auto-Tuning (Triton ↔ TVM MetaSchedule)

Flow:
```
[Triton kernel]  →  [Triton JIT]  →  [TTGIR]
       ↓
[IR capture via Triton backend plugin]
       ↓
[IR normalization]  →  [TVM TIR]
       ↓
[TVM MetaSchedule RL search]
       ↓
[Best block config]  →  [Triton recompile with tuned params]
```

The bridge_orchestrator.py is the entry point.

### Phase 2: AOT Fat Binary

Flow:
```
[Triton kernel source]
       │
       ├──► [NvidiaBackend] ──► kernel.ptx / kernel.cubin
       │    (uses triton.compiler.compile)
       │
       ├──► [AMDBackend] ──────► kernel.hsaco
       │    (uses AOTriton or triton+amdclang++)
       │
       ├──► [IntelBackend] ────► kernel.spv
       │    (uses triton + llvm-spirv)
       │
       └──► [C runtime stub] ──► runtime_stub.o
            (probes /dev/nvidia*, /dev/kfd, /dev/dri/renderD*)
       │
       └──► [LLVM lld linker] ──► fat_binary.o
            (concatenates per-vendor sections + runtime stub)
```

Fat binary format:
- ELF relocatable object
- Sections: `.nv_kernel` (PTX), `.amd_kernel` (HSACO), `.intel_kernel` (SPV)
- C entry point `nautilus_dispatch()` does runtime vendor detection
- Single file; dispatcher picks the right backend at startup

### Phase 3: Auto-Sharding (PyTorch ↔ XLA GSPMD)

Flow:
```
[PyTorch model]  →  [torch.compile()]  →  [FX graph]
       ↓
[FX → StableHLO export]
       ↓
[GSPMD auto-shard on device mesh]
       ↓
[Sharding spec per tensor + per-op]
       ↓
[DTensorApply → PyTorch DTensor]
       ↓
[Per-shard fat binary from Phase 2]
       ↓
[Per-shard Triton source executed on assigned device]
```

### Phase 4: CUDA Ingestion

Flow:
```
[Legacy .cu file]
       ↓
[Tree-sitter C++ parser] → AST
       ↓
[AST → Triton AST translator]
       ↓
[Standard compile pipeline from Phase 1]
```

## Cross-cutting concerns

### Version drift isolation

The C-API headers (`src/c_api/*.h`) define the ONLY signatures
that Nautilus code calls into upstream libraries. When upstream
Triton, TVM, or XLA change their APIs, only the implementation
in `src/c_api/wrappers/` (or, currently, the Python subprocess
fallback in `__init__.py`) needs to be updated. The bridge code
stays untouched.

### Observability

Every long-running operation in Nautilus uses the
`span`/`stage` context managers from `src.common.logging`. Each
bridge has per-dependency `CircuitBreaker` instances and a
`TimeoutManager` with per-stage budgets. Failures are isolated;
slow stages are diagnosed precisely.

### Error model

Every fallible function either returns `Result[T, E]` (for new
code) or raises a `NautilusError` subclass (for legacy code).
Errors carry:
- A stable `code` (e.g. `E_COMPILATION_FAILED`) for cross-version
  compatibility
- A human-readable `message`
- An optional `cause` chain
- A `context` dict with structured diagnostic info

NEVER silently return a placeholder. The previous version had
`return 0` GPU detectors and `_write_placeholder` AMD ELFs; those
are all removed. If a dependency is missing, raise a clear error.

## Configuration

- `pyproject.toml` — package metadata, pinned deps, CLI entry points
- Environment variables (prefixed `NAUTILUS_`):
  - `NAUTILUS_CACHE_DIR` — where to store compiled kernel cache
  - `NAUTILUS_LOG_LEVEL` — debug/info/warning/error
  - `NAUTILUS_LOG_JSON` — true/false, structured JSON vs human
  - `NAUTILUS_FAT_BINARY_CACHE` — fat binary cache location
  - `NAUTILUS_NVIDIA_CACHE`, `NAUTILUS_AMD_CACHE`, `NAUTILUS_INTEL_CACHE` — per-vendor
  - `NAUTILUS_C_LIB` — path to compiled C-API shared library

## Extensibility

To add a new vendor (e.g. Tenstorrent):

1. Implement `src/bridges/aot_packager/tenstorrent_backend.py`
   following the pattern of `nvidia_backend.py`/`amd_backend.py`
2. Add `TenstorrentArch` enum and `TenstorrentCompilationResult` dataclass
3. Wire into `builder.py:FatBinaryBuilder.__init__`
4. Add `tenstorrent` to `HardwareTarget.to_tvm_target()` mapping
5. Add a C runtime stub detection function in `runtime_stub.c`
6. Add a GitHub Actions self-hosted runner label `tenstorrent`
7. Update `docs/USER_GUIDE.md` with examples
