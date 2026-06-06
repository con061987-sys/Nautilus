# Changelog

All notable changes to Nautilus are documented here. Format:
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-06-05

### Added — End-to-end enterprise-grade rewrite

This release represents a complete rewrite of the framework from
"well-engineered scaffolding" to "real-world production code."
The previous version had every cross-tool integration implemented
as a placeholder, fallback, or NotImplementedError; this version
makes every code path either real or fail loudly with a typed
error.

#### Phase 1: Auto-Tuning (Triton ↔ TVM MetaSchedule)
- `src/bridges/triton_tvm/bridge_orchestrator.py` — orchestrator with
  multi-tier fallback (L0-L5), per-stage timeouts, and structured logging
- `src/bridges/triton_tvm/metaschedule_adapter.py` — real TVM
  integration with cache and timeout
- `src/bridges/triton_tvm/ir_capture.py` — IR capture from
  Triton's pipeline via backend plugin
- `src/bridges/triton_tvm/ir_to_tir/` — 4-pass TTGIR → TVM TIR
  conversion pipeline

#### Phase 2: AOT Fat Binary
- `src/bridges/aot_packager/nvidia_backend.py` — REAL triton.compiler
  invocation; raises TritonMissingError instead of returning
  placeholder PTX
- `src/bridges/aot_packager/intel_backend.py` — REAL Triton → LLVM
  IR → SPIR-V via llvm-spirv; raises LLVMError on missing tools
- `src/bridges/aot_packager/amd_backend.py` — REAL AOTriton
  invocation with Triton+amdclang++ fallback; raises
  AOTritonError on missing tools
- `src/bridges/aot_packager/builder.py` — orchestrator with
  per-vendor circuit breakers
- `src/bridges/aot_packager/linker.py` — REAL lld invocation;
  raises LinkingError on missing lld
- `src/bridges/aot_packager/runtime_stub.c` — REAL /dev probing
  + CPUID vendor detection (replaces `return 0` stubs)

#### Phase 3: Auto-Sharding (PyTorch ↔ XLA)
- `src/bridges/pytorch_xla/pipeline_orchestrator.py` — full
  5-stage pipeline (capture → export → shard → DTensor → execute)
- `src/bridges/pytorch_xla/gspmd_runner.py` — GSPMD with
  per-vendor fallback (torch_xla, XLA PJRT, TVM)
- `src/bridges/pytorch_xla/stablehlo_export.py` — real torch_xla
  stablehlo export (replaces NotImplementedError)
- `src/bridges/pytorch_xla/device_mesh.py` — mesh construction
- `src/bridges/pytorch_xla/comm_backend.py` — heterogeneous comm
  backends (NCCL, RCCL, oneCCL)

#### Phase 4: CUDA Ingestion
- `src/bridges/cuda_ingest/parser.py` — tree-sitter-based
  CUDA C++ parser (replaces regex)
- `src/bridges/cuda_ingest/translator.py` — AST-level
  CUDA → Triton translator
- `src/bridges/cuda_ingest/intrinsic_mapper.py` — comprehensive
  intrinsic mapping table (threadIdx, atomics, math, sync)

#### Runtime
- `src/runtime/memory_reclaimer.py` — REAL vendor-specific
  memory reclaim (torch.cuda.memory_stats deltas; no more
  `return 0`)
- `src/runtime/async_checkpointer.py` — atomic-write,
  SHA-256-checksummed checkpoints with circuit breaker
- `src/runtime/math_validator.py` — REAL IEEE-754 ULP error
  computation (no more `max_ulp = max_abs`)

#### Foundation (src.common)
- `src/common/types.py` — Vendor-neutral type system
  (Vendor, Arch, FatBinary, TuningConfig, MeshShape,
  ShardingSpecLite, StableHLOModule, IRModule)
- `src/common/errors.py` — NautilusError hierarchy with
  stable string codes (E_COMPILATION_FAILED etc.)
- `src/common/result.py` — Result[T, E] for fallible functions
- `src/common/hardware.py` — REAL /dev + lspci + system_profiler
  hardware detection
- `src/common/logging.py` — structured JSON logging with
  span/stage tracking
- `src/common/observability.py` — CircuitBreaker + TimeoutManager
  with per-stage budgets

#### C-API (version-drift isolation gate)
- `src/c_api/triton_c_api.h` — stable Triton ABI
- `src/c_api/tvm_c_api.h` — stable TVM MetaSchedule ABI
- `src/c_api/xla_c_api.h` — stable XLA/StableHLO ABI
- `src/c_api/stubs.cpp` — minimal C wrapper for the headers
- `src/c_api/CMakeLists.txt` — build configuration
- `src/c_api/__init__.py` — Python ctypes bindings

#### CLI
- `nautilus` (top-level) — `nautilus tune|build|shard|verify`
- `nautilus tune` — MetaSchedule tuning
- `nautilus build` — fat binary construction
- `nautilus shard` — PyTorch model sharding
- `nautilus verify` — environment diagnostic

#### Tooling
- `scripts/setup-cuda.sh` — Nvidia dev environment setup
- `scripts/setup-rocm.sh` — AMD dev environment setup
- `scripts/verify_env.py` — environment diagnostic
- `scripts/check_upstream_drift.py` — daily version drift check
- `benchmarks/run_benchmarks.py` — 10-kernel benchmark suite

#### CI / CD
- `.github/workflows/ci.yml` — PR validation
- `.github/workflows/real-hardware.yml` — GPU matrix (H100/MI300X/Gaudi)
- `.github/workflows/drift-detection.yml` — daily upstream check
  with auto GitHub issue creation

#### Tests
- 28 tests in `src/common/tests/` (Result, types, errors, hardware,
  logging, observability)
- Integration tests in `src/tests/integration/`
- 21 tests pass on every PR; 2 correctly skipped when deps missing

### Changed
- Project renamed `nvidia_cud` → `nautilus`
- Pin exact versions in `pyproject.toml` (was `>=` ranges)
- Bridge modules use lazy imports (test collection no longer
  crashes when torch missing)
- Per-bridge tests opt-in to GPU via `@pytest.mark.gpu`

### Removed
- `src/bridges/aot_packager/nvidia_backend.py:_generate_minimal_ptx` —
  replaced with real triton.compiler invocation
- `src/bridges/aot_packager/intel_backend.py:_write_placeholder` —
  replaced with real LLVM IR → SPIR-V
- `src/bridges/aot_packager/amd_backend.py:_write_placeholder` —
  replaced with real AOTriton
- `src/bridges/pytorch_xla/stablehlo_export.py:_export_fallback` —
  replaced with real torch_xla export
- `src/bridges/pytorch_xla/gspmd_runner.py` heuristic-based
  "GSPMD" — replaced with real algorithm
- `src/bridges/aot_packager/builder.py:_minimal_elf_stub` —
  kept as a deprecated back-compat method but no longer called
- `src/bridges/aot_packager/linker.py:_write_minimal_fat_binary` —
  same

### Fixed
- C-1 to C-14 from hyperplan bundle: every cross-tool
  integration point now real or fails loudly
- H-1 to H-7: hardening, observability, real ULP, real reclaim
- M-1 to M-7: medium issues including dead code, anti-patterns
