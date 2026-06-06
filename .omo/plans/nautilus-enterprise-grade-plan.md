# Nautilus Enterprise-Grade Implementation Plan

> **Philosophy:** Fix or REDESIGN. Never remove or merely document limitations. Every stub must become real. Every placeholder must become production code. Every isolated bridge must become a wired pipeline. This plan exists to turn the Nautilus codebase from "80% real per-bridge, 40% goal-achieved end-to-end" into a genuinely enterprise-grade, real-world-ready, cross-vendor AI compilation framework where `pip install nautilus` → `nautilus shard my_model.py` → works on real hardware.

**Goal:** Make the Nautilus codebase a genuine, installable, end-to-end cross-vendor AI compilation framework where a HuggingFace model can be auto-sharded, auto-tuned, AOT-compiled into a fat binary, and executed on a mixed AMD/Intel cluster — all from one command — with numerical correctness verified, benchmark numbers published, and CI failing on regression.

**Architecture:** 7 execution phases, each with verification gates. Phases with same depth can run in parallel. No phase removes code — every phase replaces stubs with real implementations or redesigns the approach.

**Tech Stack:** Python 3.10+, Triton 3.0+, TVM 0.18+, XLA/StableHLO, LLVM lld, AOTriton, ROCm, oneAPI, tree-sitter, CMake.

---

## 0. Pre-Flight: Issue Severity Catalog

Every finding from the 4-agent audit is categorized below. The plan WILL address every item. Items marked `[REDESIGN]` require architectural change; `[FIX]` requires replacement of a stub/placeholder; `[PRODUCTIONIZE]` requires hardening existing real code.

### CRITICAL (P0 — blocks enterprise deployment)

| ID | Issue | File:Line | Type |
|----|-------|-----------|------|
| C-01 | Package doesn't install: `setup.py` doesn't call `setup()` | `setup.py:81-105` | FIX |
| C-02 | `shard.py:_generate_shard_source` emits hand-written matmul template | `src/cli/commands/shard.py:264-309` | REDESIGN |
| C-03 | `hardware_orchestrator.py:_generate_shard_source` duplicate matmul stub | `src/bridges/pytorch_xla/hardware_orchestrator.py:147-186` | REDESIGN |
| C-04 | `IRCapture` key-format mismatch: write keys don't match read keys | `ir_capture.py:138` vs `hooks.py:118` vs `compiler.py:281` | FIX |
| C-05 | C-API is decorative: test asserts `is_available() == False`; production code imports upstream directly | `src/c_api/*`, `__init__.py:80-106` | REDESIGN |
| C-06 | All 4 bridge test directories `--ignore`d from default pytest (260 of 296 tests hidden) | `pyproject.toml:146-150` | FIX |
| C-07 | `ci.yml:128` typo: `output_filename` doesn't exist on `FatBinaryConfig` | `.github/workflows/ci.yml:128` | FIX |
| C-08 | `_minimal_elf_stub` returns 64-byte zero-filled ELF header | `src/bridges/aot_packager/builder.py:423-449` | REDESIGN |
| C-09 | `_write_minimal_fat_binary` produces JSON+concatenated blob, not ELF | `src/bridges/aot_packager/linker.py:362-407` | REDESIGN |
| C-10 | `c_api/stubs.cpp` all functions return `NAUTILUS_ERR_BACKEND_MISSING` | `src/c_api/stubs.cpp:59-143` | REDESIGN |

### HIGH (P1 — serious gap for production)

| ID | Issue | File:Line | Type |
|----|-------|-----------|------|
| H-01 | `extern_bridge._compile_triton_to_binary` returns path to 0-byte touch file | `extern_bridge.py:312-324` | FIX |
| H-02 | `metaschedule_adapter._run_tune_with_timeout` has no actual timeout | `metaschedule_adapter.py:148-176` | FIX |
| H-03 | `gspmd_runner._annotate_stablehlo_with_sharding` adds comments, not MLIR attributes | `gspmd_runner.py:943-1026` | REDESIGN |
| H-04 | `dtensor_apply.apply_to_model` converts DTensor back to local tensor | `dtensor_apply.py:155-187` | FIX |
| H-05 | `comm_backend._estimate_bandwidth` returns hardcoded 900/800/200/64 GB/s | `comm_backend.py:133-141` | REDESIGN |
| H-06 | `gspmd_runner._TVMMetaScheduleSharding` uses CPU `Target("llvm")` not GPU target | `gspmd_runner.py:615-620` | FIX |
| H-07 | `bridge_orchestrator._tune_with_real_ir` falls back to `_synthesize_metadata` (grid=1,1,1) silently | `bridge_orchestrator.py:282-284` | FIX |
| H-08 | `backend/hooks.py:92` env var negation typo: `"0".__eq__("0")` → always False | `backend/hooks.py:92` | FIX |
| H-09 | `metaschedule_adapter.tune()` returns `MappedTuningConfig.defaults()` on bare `except Exception` | `metaschedule_adapter.py:140-142` | FIX |
| H-10 | `memory_reclaimer.reclaim()` returns 0 in 4 paths, docstring claims "Never returns 0" | `memory_reclaimer.py:128-180` | REDESIGN |
| H-11 | `gspmd_runner._CommCostModel.estimate_tensor_bytes` uses per-shard shape, not original | `gspmd_runner.py:196-205` | FIX |
| H-12 | CUDA translator emits `tl.debug_barrier()` not `tl.barrier()` | `cuda_ingest/translator.py` | FIX |
| H-13 | 30 tests blocked by top-level `import torch`/`import triton` at module level | `metadata_extractor.py:17`, `config_mapper.py:16`, `bridge_orchestrator.py:28` | FIX |
| H-14 | `graph_capture._capture_via_compile` unpacks old `dynamo.export` API (broken on PyTorch 2.5+) | `graph_capture.py:205-206` | FIX |
| H-15 | `linker._wrap_section_data` produces ELF with empty string table (section name invalid) | `linker.py:251-315` | REDESIGN |
| H-16 | `runtime_stub.c` uses x86_64 inline ASM only; ARM64 falls through to `return 0` | `runtime_stub.c:75-110` | FIX |
| H-17 | `memory_reclaimer._reclaim_apple` returns 0 (admits the anti-pattern in a comment) | `memory_reclaimer.py:268-282` | FIX |
| H-18 | `async_checkpointer._checkpoint_loop` race: new checkpoint can overwrite pending model during save | `async_checkpointer.py:250-266` | FIX |
| H-19 | `async_checkpointer` uses `weights_only=False` in `torch.load` (security) | `async_checkpointer.py:535-539` | FIX |
| H-20 | `async_checkpointer` falls back to `pickle.loads` (arbitrary code execution) | `async_checkpointer.py:548` | FIX |
| H-21 | `math_validator.insert_rounding_correction` prepends a comment, has zero effect | `math_validator.py:235-246` | REDESIGN |
| H-22 | `stablehlo_export._TVMScriptExporter._emit_stablehlo_like` emits single `stablehlo.identity` | `stablehlo_export.py:464-520` | REDESIGN |
| H-23 | `gspmd_runner._TVMMetaScheduleSharding._build_tir_module` uses hardcoded `shape_k=128` | `gspmd_runner.py:649-738` | FIX |
| H-24 | `benchmarks/run_benchmarks.py` only has matmul; README claims 10 kernels | `benchmarks/run_benchmarks.py:31-38`, `README.md:8-25` | FIX |
| H-25 | README/CONTRIBUTING/CHANGELOG reference wrong GitHub URL (`nvindia-cud/nautilus`) | `README.md:17`, `CONTRIBUTING.md:10`, `CHANGELOG.md:75` | FIX |
| H-26 | `ci.yml` coverage gate (`--cov-fail-under=60`) computed over hidden 36 tests only | `.github/workflows/ci.yml:63` | FIX |
| H-27 | `ir_to_tir/` 4-pass pipeline only sets attribute flags, never transforms AST | `ir_to_tir/pass1-4/*.py` | REDESIGN |
| H-28 | `pipeline_orchestrator` uses `"tvm_tune"` circuit breaker (name collision with Phase 1) | `pipeline_orchestrator.py:263` | FIX |
| H-29 | `hardware_validator._local_validation` returns `passed=True` with no real validation | `hardware_validator.py:144-198` | REDESIGN |
| H-30 | `stablehlo_export._TVMScriptExporter._build_relax_from_fx` silently drops ops not in its 10-op map | `stablehlo_export.py:420-447` | FIX |
| H-31 | `backend/compiler.py:46` module-level `_CAPTURE_BUFFER` dict — no locking, no namespacing | `backend/compiler.py:46` | FIX |
| H-32 | `bridge_orchestrator._build_tir_from_captured` returns `None` silently; orchestrator treats as "use fallback" | `bridge_orchestrator.py:378-439` | FIX |
| H-33 | No end-to-end HuggingFace model → sharded fat binary → hardware verification test | not present | REDESIGN |

### MEDIUM (P2 — notable issue)

| ID | Issue | File:Line | Type |
|----|-------|-----------|------|
| M-01 | `Vendor.from_string` silently coerces invalid input to `UNKNOWN` | `types.py:42-46` | FIX |
| M-02 | `KernelSection.sha256` recomputes SHA-256 on every access (O(N) each time) | `types.py:170-171` | FIX |
| M-03 | `HardwareTarget.to_tvm_target` hardcodes vendor→string mappings (maintenance trap) | `types.py:107-124` | PRODUCTIONIZE |
| M-04 | `StableHLOModule.__post_init__` uses stdlib `logging`, not structured logger | `types.py:484-492` | FIX |
| M-05 | `with_context` passes `code=` to constructor, can strip subclass-specific code | `errors.py:134-144` | FIX |
| M-06 | `StageTimeoutError` and `TotalBudgetExceededError` share `COMPILATION_TIMEOUT` code | `errors.py:365-376` | FIX |
| M-07 | `NautilusError.to_dict` unguarded `repr(self.cause)` can crash | `errors.py:124-132` | FIX |
| M-08 | `enumerate_devices()` called 4× per `nautilus verify` (spawns `lspci` 4×) | `hardware.py:543-572` | FIX |
| M-09 | `lspci -nn -mm` parser fragile: `split('"')` instead of `shlex.split` | `hardware.py:247-290` | FIX |
| M-10 | Intel/AMD device enumeration collides on `/dev/dri/renderD*` | `hardware.py:437-484` | FIX |
| M-11 | `configure_logging()` called at module-level `logging.py:187` (import side-effect) | `logging.py:187` | FIX |
| M-12 | Stdlib log mirroring can double-emit if user configures stdlib handler | `logging.py:395-403` | FIX |
| M-13 | `get_default_breakers()` returns fresh dict on every call (defeats circuit breaker sharing) | `observability.py:223-241` | FIX |
| M-14 | `TimeoutManager.stage` raises from `finally`, masking the original exception | `observability.py:298-320` | FIX |
| M-15 | `CircuitBreaker` catches `BaseException` (too broad) | `observability.py:187-192` | FIX |
| M-16 | `Result[T, E]` is zero-adopted outside `src/common/` | `result.py`, `tune.py:27`, `build.py:25` | PRODUCTIONIZE |
| M-17 | `Err.unwrap` raises error with no unwrap site context | `result.py:87-88` | FIX |
| M-18 | CI grep guardrail forbids legitimate `# TODO: implement` (blocks dev workflows) | `.github/workflows/ci.yml:32-38` | FIX |
| M-19 | `benchmarks/README.md` claims 10 kernels, only matmul exists | `benchmarks/README.md` + `run_benchmarks.py` | FIX |
| M-20 | `pyproject.toml` pins `torch==2.4.1`, `triton==3.0.0` (old) | `pyproject.toml:51-53` | PRODUCTIONIZE |
| M-21 | `session-ses_*.md` files committed in repo root (500KB+ AI session logs) | `session-ses_16a6.md`, `session-ses_166a.md` | FIX |
| M-22 | `third_party/` contains only README (submodule claim false in docs) | `third_party/README.md`, `docs/E.md`, `AGENTS.md` | REDESIGN |
| M-23 | All backend AOT compiles use hardcoded signature `["*fp32"]*3 + ["i32"]*3 + ["constexpr"]*3` | `nvidia_backend.py:284-300`, `amd_backend.py:320-330`, `intel_backend.py:253-262` | FIX |
| M-24 | `extern_bridge._generate_triton_matmul` hardcodes `BLOCK_SIZE=128/128/32` | `extern_bridge.py:263-265` | FIX |
| M-25 | `tir_template.build_from_metadata` uses launch grid × 128, not actual tensor sizes | `tir_template.py:203-206` | FIX |
| M-26 | `config_mapper.map_record` uses fragile heuristics on `decisions` dict keys | `config_mapper.py:144-205` | PRODUCTIONIZE |
| M-27 | `bridge_orchestrator` uses module-level `_stages` dict (no reset between calls) | `bridge_orchestrator.py:133-134` | FIX |
| M-28 | `extern_bridge.generate_tir_extern_call` references `"triton_matmul_run"` by string; runtime has no implementation | `extern_bridge.py:148-193` | FIX |
| M-29 | `nvidia_backend.py:316-319` uses `target="cuda"` for all Nvidia targets (no per-arch PTX) | `nvidia_backend.py:316-319` | FIX |
| M-30 | `intel_backend.py:271` falls back to `target="cuda"` if xpu unavailable → wrong ISA | `intel_backend.py:271` | FIX |
| M-31 | No runtime tests exist for `memory_reclaimer.py`, `async_checkpointer.py`, `math_validator.py` | `src/runtime/*` | FIX |
| M-32 | No test for `stablehlo_export.py` or `graph_capture.py` | `pytorch_xla/tests/` | FIX |
| M-33 | `pipeline_orchestrator.shard()` model parameter path is untested (8 tests pass `model=None`) | `test_pipeline_orchestrator.py` | FIX |
| M-34 | `c_api/__init__.py` library loading not thread-safe | `c_api/__init__.py:80-106` | FIX |
| M-35 | `c_api/__init__.py` docstring references non-existent `triton_c_api` submodules | `c_api/__init__.py:14-27` | FIX |
| M-36 | `is_available()` mutates `_C_LIB_LOAD_ERROR` global but reader doesn't exist | `c_api/__init__.py:290-296` | FIX |
| M-37 | `_CAPTURE_BUFFER` global dict in `compiler.py` is module-level mutable state, no locking | `backend/compiler.py:46` | FIX |
| M-38 | `real-hardware.yml` requires `[self-hosted, gpu, *]` runners that don't exist | `.github/workflows/real-hardware.yml:14-27` | REDESIGN |
| M-39 | `ir_classifier.py:test_classify_attention` fails (real bug: counts only 1 dot, not 2) | `ir_classifier.py` | FIX |
| M-40 | `test_timeout_manager:test_stage_under_budget_succeeds` fails (stage under budget still raises) | `timeout_manager.py` | FIX |

### LOW (P3)

(35 items from audit — see Appendix A for full list)
- L-01 through L-35: Minor bugs, stale version-conditional paths, cosmetic issues, docs typos.

### MINOR (P4)

(10 items — `__pycache__/`, `.opencode/node_modules/`, `prompts.md`, file permissions, etc.)

### NICE-TO-HAVE

(12 items — Prometheus metrics, OpenTelemetry, health checks, web UI, auto-update, plugin system, etc.)

> **Full LOW/MINOR/NICE lists** are in Appendix A (separate section at end). Each is addressed here with at most a referenced task, not an independent TODO section.

---

## Philosophy: The Fix-or-Redesign Rule

Every issue in this plan is classified as one of:

1. **FIX**: The code exists but has a defect (bug, security hole, performance issue). The fix is replacing the implementation while keeping the interface. Example: `cuda_ingest/translator.py` emitting the wrong function name (`tl.debug_barrier()` → `tl.barrier()`).

2. **REDESIGN**: The code is fundamentally wrong (returns placeholder, uses wrong target, architecture cannot achieve the goal). Requires replacing the architecture, not just the implementation. Example: `_generate_shard_source` returns a hand-written matmul template → replace with a real StableHLO→Triton translator module.

3. **PRODUCTIONIZE**: The code is correct but not production-grade (no tests, no metrics, no caching, no isolation). Example: `Result[T, E]` is zero-adopted outside `src/common/`.

**NEVER** used after this point:
- ❌ "Mark as deprecated" — we fix or redesign, never kick the can
- ❌ "Document the limitation" — we remove the limitation
- ❌ "Raise NotImplementedError" — we implement or redesign
- ❌ "Keep for back-compat" — we delete the old and replace with new
- ❌ "For now, just stub" — everything is enterprise-grade from the start

---

## 1. Execution Wave Structure

The plan is organized into 7 execution waves, each with parallel tracks. Waves at the same depth are independent.

```
Wave 0 ── Foundation Repairs (P0: 8 tasks, 2-3 days)
         ├── 0.1: Fix setup.py → pyproject.toml build
         ├── 0.2: Move imports inside function bodies (unblock 30 tests)
         ├── 0.3: Remove --ignore patterns from pyproject.toml
         ├── 0.4: Fix 4 real production bugs (translator, shared_memory, ir_classifier, timeout_manager)
         ├── 0.5: Fix IRCapture key-format mismatch
         ├── 0.6: Fix ci.yml typo (output_filename)
         ├── 0.7: Fix hooks.py env var typo
         └── 0.8: Build: third_party/ submodule pins + verify env

Wave 1 ── End-to-End Pipeline Wiring (P0: 6 tasks, 5-7 days, depends on Wave 0)
         ├── 1.1: Build StableHLO→Triton real translator module
         ├── 1.2: Wire `nautilus shard` through `AutoShardingBridge.shard()`
         ├── 1.3: Replace `hardware_orchestrator._generate_shard_source` with real translator call
         ├── 1.4: Wire `pipeline_orchestrator` fat_binary stage (per-shard FatBinaryBuilder.build())
         ├── 1.5: Build end-to-end HuggingFace demo (scripts/demo_e2e.py)
         └── 1.6: Real GSPMD annotation (attributes, not comments) + real TVM MetaSchedule target

Wave 2 ── Bridge Hardening & Productionization (P0/P1: 10 tasks, 3-5 days, parallel with Wave 3)
         ├── 2.1: triton_tvm: real IR capture + real MetaSchedule timeout + valid TIR templates
         ├── 2.2: aot_packager: real hardware validation (cuModuleLoad, hipModuleLoad, zeModuleCreate)
         ├── 2.3: aot_packager: real `_wrap_section_data` with proper ELF string table + symbols
         ├── 2.4: aot_packager: replace `_minimal_elf_stub` with proper ELF section builder
         ├── 2.5: aot_packager: replace `_write_minimal_fat_binary` with proper ELF linker fallback
         ├── 2.6: aot_packager: per-arch AOT compilation (not just target="cuda" for all)
         ├── 2.7: cuda_ingest: handle all 5 known limitations from architecture doc
         ├── 2.8: pytorch_xla: fix dtensor_apply (don't convert DTensor to local)
         ├── 2.9: pytorch_xla: fix graph_capture for PyTorch 2.5+ API
         └── 2.10: aot_packager: fix hardcoded backend signatures (handle real kernel signatures)

Wave 3 ── Runtime Productionization (P1: 6 tasks, 2-3 days, parallel with Wave 2)
         ├── 3.1: memory_reclaimer: change API to return Result[int, NautilusError], fix 4 return-0 paths
         ├── 3.2: memory_reclaimer: fix Apple _reclaim via torch.mps
         ├── 3.3: async_checkpointer: fix race condition in _checkpoint_loop (capture refs under lock)
         ├── 3.4: async_checkpointer: fix weights_only=True + remove pickle fallback
         ├── 3.5: math_validator: real bit-exact mode (insert TTGIR flags, not comments)
         └── 3.6: math_validator: handle NaN in ULP computation

Wave 4 ── C-API & Version Drift (P0/P1: 5 tasks, 4-6 days, depends on Wave 0)
         ├── 4.1: Build real triton_c_api.so, tvm_c_api.so, xla_c_api.so from pinned submodules
         ├── 4.2: Route nvidia_backend through c_api (first production C-API consumer)
         ├── 4.3: Route tvm_adapter through c_api
         ├── 4.4: Route stablehlo_export through c_api
         └── 4.5: Add drift-detection CI that actually tests C-API compatibility

Wave 5 ── CLI & Distribution (P1/P2: 6 tasks, 2-3 days, parallel with Wave 4)
         ├── 5.1: Add `nautilus inspect` subcommand (read fat binary metadata)
         ├── 5.2: Fix CLI help text for all commands (verify --help is accurate)
         ├── 5.3: Fix README/CONTRIBUTING/CHANGELOG GitHub URLs + honesty pass
         ├── 5.4: Fix benchmarks: implement 9 missing kernel benchmarks
         ├── 5.5: Add Docker image for Nvidia + AMD + Intel
         └── 5.6: Add CI jobs with GitHub-hosted GPU runners (document self-hosted as optional)

Wave 6 ── Testing & CI/CD (P1/P2: 8 tasks, 3-5 days, parallel with Wave 4/5)
         ├── 6.1: Add runtime tests for memory_reclaimer, async_checkpointer, math_validator
         ├── 6.2: Add stablehlo_export + graph_capture tests
         ├── 6.3: Add pipeline_orchestrator happy-path test (real model → shard → fat binary)
         ├── 6.4: Add C-API integration test (build .so, call compile, verify result)
         ├── 6.5: Fix ci.yml coverage gate (include bridge code)
         ├── 6.6: Fix ci.yml grep guardrail (allow legitimate TODOs)
         ├── 6.7: Add real-hardware.yml with real GitHub-hosted GPU runners
         └── 6.8: Remove session-ses_*.md + __pycache__/ from git tracking

Wave 7 ── Enterprise Completeness (P2/P3: 10 tasks, 3-5 days, final)
         ├── 7.1: Structured logging everywhere (no stdlib logging bypass)
         ├── 7.2: Result[T,E] adoption across all bridges
         ├── 7.3: SINGLETON fix for get_default_breakers()
         ├── 7.4: TimeoutManager: fix finally-clause exception masking
         ├── 7.5: CircuitBreaker: change BaseException to Exception + excluded_exceptions
         ├── 7.6: Fix common/types.py: Vendor.from_string strict mode, sha256 cache, TVM target table
         ├── 7.7: Fix common/errors.py: with_context type safety, distinct timeout codes
         ├── 7.8: Fix common/observability.py: enumrate_devices cache, lspci parser robustness
         ├── 7.9: Fix runtime_stub.c for ARM64 (non-x86 CPUID)
         └── 7.10: Remove deprecated stubs (_minimal_elf_stub, _write_minimal_fat_binary) after replacing

---

## 2. Dependency Matrix

```
Wave 0 ──► Wave 1 ──► Wave 2 ──► Wave 7
                 │            │
                 └──► Wave 3 ──┘
                 │
                 └──► Wave 4 ──► Wave 7
                 │
                 └──► Wave 5 ──► Wave 7
                 │
                 └──► Wave 6 ──► Wave 7
```

Wave 0 blocks everything. Wave 1 blocks Waves 2-6. Waves 2-6 all converge to Wave 7.

---

## 3. Per-Issue Fix Designs

### Wave 0: Foundation Repairs

#### 0.1 — Fix setup.py / pyproject.toml build (C-01)
**Type:** FIX
**Files:** `setup.py:81-105`, `pyproject.toml:1-4`
**Fix:** Replace `setup.py` with a minimal shell that calls `setup()`. The `build_cpp_plugin()` command class stays but must be registered with `cmdclass=`. Add `pyproject.toml` `[build-system]` `build-backend = "setuptools.build_meta"` (already present). Remove dead `else` branch class definitions.
**Verification:** `pip install -e .[dev]` → `nautilus --help` shows usage. `python -c "import src.bridges.aot_packager; print('ok')"` → `ok`.
**Rollback:** `git checkout setup.py`

#### 0.2 — Move top-level imports inside function bodies (H-13)
**Type:** FIX
**Files:** `metadata_extractor.py:17`, `config_mapper.py:16`, `bridge_orchestrator.py:28`
**Fix:** Move `import torch` inside `extract_from_call()` body. Move `import triton` inside `map_record()` body. This unblocks 30 tests that fail on import.
**Verification:** `pytest src/bridges/triton_tvm/tests/ --collect-only | grep "collected"` → collected = 121 (not blocked).
**Rollback:** `git checkout metadata_extractor.py config_mapper.py bridge_orchestrator.py`

#### 0.3 — Remove --ignore patterns from pyproject.toml (C-06)
**Type:** FIX
**Files:** `pyproject.toml:146-150`
**Fix:** Remove the 4 `--ignore` lines. Wrap `import torch`/`import triton` in `conftest.py` auto-skip logic. Add `@pytest.mark.requires_deps` to tests that need extra packages.
**Verification:** `pytest src/ --collect-only | tail -5` → `collected 296 items`.
**Rollback:** `git checkout pyproject.toml`

#### 0.4 — Fix 4 real production bugs (H-12, M-39, M-40, Appendix A: shared_memory)
**Type:** FIX
**Files:** `cuda_ingest/translator.py`, `cuda_ingest/shared_memory.py:80`, `triton_tvm/ir_classifier.py:142`, `triton_tvm/timeout_manager.py:95`
**Fix:**
- translator: `tl.debug_barrier()` → `tl.barrier()`
- shared_memory: multiply all dimensions for multi-dimensional `__shared__` arrays
- ir_classifier: increment counter for all dot operations, not just the first
- timeout_manager: fix `stage_under_budget_succeeds` — check `elapsed < budget` not `elapsed <= 0`
**Verification:** `pytest src/bridges/cuda_ingest/tests/ src/bridges/triton_tvm/tests/ -v -k "test_" | grep FAILED | wc -l` → 0 failures.
**Rollback:** `git checkout translator.py shared_memory.py ir_classifier.py timeout_manager.py`

#### 0.5 — Fix IRCapture key-format mismatch (C-04)
**Type:** FIX
**Files:** `ir_capture.py:138`, `backend/hooks.py:118`, `backend/compiler.py:281`
**Fix:** Define a shared key format constant (e.g. `CAPTURE_KEY_FMT = "nautilus:ttgir:{source_hash}:{name}"`) used by all three files. `hooks.py:118` writes `f"nautilus:ttgir:{source_hash[:16]}:{kernel_name}"`. `compiler.py:281` writes the same format. `ir_capture.py:138-156` reads using the same format with exact match.
**Verification:** `python -c "from src.bridges.triton_tvm.ir_capture import CAPTURE_KEY_FMT; from src.bridges.triton_tvm.backend.hooks import CAPTURE_KEY_FMT as HKF; assert CAPTURE_KEY_FMT == HKF"` → True.
**Rollback:** `git checkout ir_capture.py hooks.py compiler.py`

#### 0.6 — Fix ci.yml typo (C-07)
**Type:** FIX
**Files:** `.github/workflows/ci.yml:128`
**Fix:** Change `output_filename` to `output_dir` on `FatBinaryConfig`.
**Verification:** `python -c "from src.bridges.aot_packager.builder import FatBinaryConfig; import dataclasses; assert 'output_dir' in [f.name for f in dataclasses.fields(FatBinaryConfig)]"` → True.
**Rollback:** `git checkout .github/workflows/ci.yml`

#### 0.7 — Fix hooks.py env var typo (H-08)
**Type:** FIX
**Files:** `backend/hooks.py:92`
**Fix:** Change `if not os.environ.get("NVINDIACUD_CAPTURE_DISABLED", "0") == "0":` to `if os.environ.get("NVINDIACUD_CAPTURE_DISABLED", "0") == "1":`
**Verification:** `NVINDIACUD_CAPTURE_DISABLED=1 python -c "from src.bridges.triton_tvm.backend.hooks import stages_inspection_hook; assert stages_inspection_hook is not None; print('ok')"` → hook returns early (no crash).
**Rollback:** `git checkout hooks.py`

#### 0.8 — Build: third_party/ submodule pins + verify env (M-22)
**Type:** REDESIGN
**Files:** `third_party/`, `pyproject.toml`, `scripts/setup-*.sh`
**Fix:** Add pinned git submodules for: `triton @ 3.0.0`, `tvm @ 0.18.0`, `xla @ pinned-commit`, `llvm-project @ 19.1.0`. Update `pyproject.toml` to have both PyPI pin (for quick install) and submodule path (for C-API build). Update `scripts/setup-cuda.sh` and `setup-rocm.sh` to clone submodules.
**Verification:** `git submodule status | wc -l` → 4+.
**Rollback:** `git submodule deinit --all; git checkout third_party/`

---

### Wave 1: End-to-End Pipeline Wiring

#### 1.1 — Build StableHLO→Triton real translator module (C-02 + C-03)
**Type:** REDESIGN
**Files:** NEW `src/bridges/pytorch_xla/stablehlo_to_triton.py`, NEW `tests/test_stablehlo_to_triton.py`
**Contract:**
```python
@dataclass
class TritonSource:
    source: str           # Full Python source of a @triton.jit function
    kernel_name: str
    input_specs: list[tuple[str, tuple[int, ...], str]]  # name, shape, dtype
    output_specs: list[tuple[str, tuple[int, ...], str]]
    op_counts: dict[str, int]  # stablehlo op → occurrences (for auditing)

# Core function
def translate(
    stablehlo_mlir: str,
    *,
    kernel_name: str,
    target_arch: str = "nvidia/sm_90",
) -> TritonSource:
    """Translate a StableHLO MLIR string into a Triton @triton.jit function.
    
    Uses MLIR parser → op dispatcher → Triton code generator.
    Supports: stablehlo.dot, stablehlo.add, stablehlo.multiply, stablehlo.subtract,
    stablehlo.convert (dtype casts), stablehlo.broadcast_in_dim, stablehlo.reshape,
    stablehlo.reduce (with add/max/min), stablehlo.concatenate, stablehlo.slice,
    stablehlo.select, stablehlo.compare, stablehlo.negate.
    
    Raises UnsupportedStableHLOOpError with op name and location string.
    """
```

**Sub-tasks (earn as individual TODO items):**
1.2a: MLIR parser that extracts `stablehlo.func` → op list with typed operands/results (5 synthetic tests)
1.2b: Codegen for elementwise ops (add, mul, sub, convert, compare, select) → 5 tests
1.2c: Codegen for `stablehlo.dot` → `tl.dot` with proper block sizing → 3 tests
1.2d: Codegen for reductions → `tl.sum`/`tl.max`/`tl.min` → 3 tests
1.2e: Codegen for broadcast/reshape/concatenate/slice → 4 tests
1.2f: Glue: `nautilus tune --stablehlo path/to/model.mlir` → Triton + AOT fat binary

**Verification:** `pytest tests/test_stablehlo_to_triton.py -v` → all tests pass. `python scripts/demo_translator.py --input tests/fixtures/simple_linear.mlir --output /tmp/out.py && grep tl.dot /tmp/out.py` → non-empty.
**Rollback:** `rm stablehlo_to_triton.py test_stablehlo_to_triton.py`

#### 1.2 — Wire `nautilus shard` through `AutoShardingBridge.shard()` (C-02 + C-03)
**Type:** REDESIGN
**Files:** `src/cli/commands/shard.py:97-181` (replace `_shard_impl`), `shard.py:264-309` (delete `_generate_shard_source`), `hardware_orchestrator.py:147-186` (replace with translator call)
**Fix:** Rewrite `_shard_impl` to call `AutoShardingBridge.shard(model, example_inputs, device_mesh)` instead of manually wiring `graph_capture → gspmd → _generate_shard_source`. The orchestrator returns `ShardingResult` with per-shard `ShardExecutionResult.fat_binary_result`. Write per-shard `kernel.fat.o` from that.
**Verification:** `python -c "import sys; sys.path.insert(0, '.'); from src.cli.commands.shard import _shard_impl; print('ok')"` → ok. `nautilus shard --help` still works.
**Rollback:** `git checkout src/cli/commands/shard.py src/bridges/pytorch_xla/hardware_orchestrator.py`

#### 1.3 — Wire pipeline_orchestrator fat_binary stage (H-28)
**Type:** REDESIGN
**Files:** `pipeline_orchestrator.py:227-253`, `pipeline_orchestrator.py:263`
**Fix:** In Stage 5 (fat_binary), call `FatBinaryBuilder.build()` per shard using translated Triton source. Pass `FatBinaryConfig(kernel_name=f"shard_{i}", kernel_source=source, ...)`. Store `FatBinaryResult` in `ShardExecutionResult`. Fix circuit breaker name from `"tvm_tune"` to `"gspmd_tune"`.
**Verification:** `pytest src/bridges/pytorch_xla/tests/ -v -k "test_pipeline"` → at least the `test_bridge_init` test passes (previously all 8 failed).
**Rollback:** `git checkout pipeline_orchestrator.py`

#### 1.4 — Build end-to-end HuggingFace demo (H-33)
**Type:** REDESIGN
**Files:** NEW `scripts/demo_e2e.py`
**Fix:** Create a script that:
1. Loads a HuggingFace model (`hf-internal-testing/tiny-llama`)
2. Captures the FX graph
3. Exports to StableHLO
4. Runs GSPMD auto-sharding
5. For each shard, calls `StableHLOToTriton.translate()` → `FatBinaryBuilder.build()`
6. Writes per-shard `kernel.fat.o` to `shards/shard_*/`
7. Verifies each fat binary is a real ELF (`b"\x7fELF"`)
8. Compares numerical output to PyTorch eager reference (`max_abs_diff < 1e-2`)
9. Prints `{"PASS": true/false, "reason": "..."}` JSON

**Verification:** `python scripts/demo_e2e.py --model hf-internal-testing/tiny-llama --mesh 1,1` → exits 0, produces `shards/shard_0000/kernel.fat.o`.
**Rollback:** `rm scripts/demo_e2e.py`

#### 1.5 — Real GSPMD annotations (H-03)
**Type:** REDESIGN
**Files:** `gspmd_runner.py:943-1026`
**Fix:** Replace `_annotate_stablehlo_with_sharding` — instead of adding comment annotations (`// mhlo.sharding = "..."`), insert real MLIR attribute syntax: `func.func() attributes {mhlo.sharding = "..."}`. Use proper `stablehlo.sharding` attribute format per the StableHLO spec.
**Verification:** `python -c "from src.bridges.pytorch_xla.gspmd_runner import _annotate_stablehlo_with_sharding; result = _annotate_stablehlo_with_sharding('func.func @main(...)', {'foo': ShardingSpecLite(...)})"` → output contains `func.func` with `attributes {stablehlo.sharding = "..."}`.
**Rollback:** `git checkout gspmd_runner.py`

#### 1.6 — Real TVM MetaSchedule GPU target (H-06)
**Type:** FIX
**Files:** `gspmd_runner.py:615-620`
**Fix:** Change `target=Target("llvm")` to `target=Target(self._mesh_target(mesh))` where `_mesh_target` maps mesh devices to TVM target strings (e.g. `nvidia/nvidia-a100`, `rocm/gfx942`). Use `device_mesh` to determine the target.
**Verification:** `grep -n "Target(" gspmd_runner.py` → no more `"llvm"` as the target.
**Rollback:** `git checkout gspmd_runner.py`

---

### Wave 2: Bridge Hardening (10 tasks parallel)

#### 2.1 — triton_tvm: real IR capture + MetaSchedule timeout + valid TIR (H-02, H-07, H-32)
**Type:** FIX + PRODUCTIONIZE
**Files:** `metaschedule_adapter.py:148-176`, `bridge_orchestrator.py:378-439`, `ir_to_tir/`
**Fix:**
- `metaschedule_adapter._run_tune_with_timeout`: implement actual timeout via `threading.Thread` + `join(timeout)`, or `signal.SIGALRM` (Unix). Raise `StageTimeoutError` on timeout.
- `bridge_orchestrator._build_tir_from_captured`: when IR conversion fails and fallback builds produce different M/N/K, log the discrepancy AND include it in the result, not just silently fall through.
- `ir_to_tir/`: replace the 4-pass "flag-setting" pipeline with actual AST transformation. Each pass must mutate the parsed `TTGIRFunction`. `pass1_lower_tensor_idioms` must replace `tt.dot` with elementwise operations in the AST, not just set a flag.
**Verification:** `pytest src/bridges/triton_tvm/tests/ -v --ignore-skip` → 0 failures (where previously 12 failed).
**Rollback:** `git checkout metaschedule_adapter.py bridge_orchestrator.py ir_to_tir/`

#### 2.2 — aot_packager: real hardware validation (H-29)
**Type:** REDESIGN
**Files:** `hardware_validator.py:144-198`
**Fix:** Replace `_local_validation` (which returns `passed=True` after `binary_path.exists()`) with real GPU module load:
```python
def _local_validation(self, binary_path: Path, vendor: str) -> ValidationResult:
    try:
        import ctypes
        lib = ctypes.CDLL(str(binary_path))
        # Nvidia: cuModuleLoadData(module, binary)
        # AMD: hipModuleLoadData(module, binary)  
        # Intel: zeModuleCreate(device, binary, module, ...)
        # Apple: Metal...
        if vendor == "nvidia" and cuda_available:
            result = ctypes.cdll.LoadLibrary(None).cuModuleLoadData(
                ctypes.byref(module), binary_path.read_bytes()
            )
            ...
        return ValidationResult(passed=True, ...)
    except Exception as exc:
        return ValidationResult(passed=False, error=str(exc), ...)
```
**Verification:** `python -c "from src.bridges.aot_packager.hardware_validator import HardwareValidator; v = HardwareValidator(); r = v.validate(Path('/nonexistent'), 'nvidia', 'sm_90'); assert not r.passed"` → fails (binary doesn't exist, good).
**Rollback:** `git checkout hardware_validator.py`

#### 2.3 — aot_packager: real ELF section builder (H-15)
**Type:** REDESIGN
**Files:** `linker.py:251-315`
**Fix:** Replace `_wrap_section_data` with a proper ELF builder. Use `struct` to produce:
- Valid ELF64 header with `e_shstrndx` pointing to a real string table
- Section header with correct `sh_name` (index into string table)
- String table containing `.nv_kernel`, `.amd_kernel`, `.intel_kernel` names
- `sh_flags` set to `SHF_ALLOC` for runtime loading
**Verification:** `readelf -S /tmp/test_wrap.o` → sections `.nv_kernel`, `.amd_kernel` visible with correct names.
**Rollback:** `git checkout linker.py`

#### 2.4 — aot_packager: replace deprecated stubs (C-08, C-09)
**Type:** REDESIGN
**Files:** `builder.py:423-449`, `linker.py:362-407`
**Fix:** Delete `_minimal_elf_stub` and `_write_minimal_fat_binary` entirely. The builder must use real `gcc` + `lld` path only; if `lld` is unavailable, raise `LinkingError` with clear install instructions. This is NOT a fallback path — `lld` is a required dependency for enterprise use.
**Verification:** `grep -n "DEPRECATED" builder.py linker.py` → 0 matches.
**Rollback:** `git checkout builder.py linker.py`

#### 2.5 — aot_packager: per-arch AOT compilation (M-29, M-30)
**Type:** FIX
**Files:** `nvidia_backend.py:284-300`, `intel_backend.py:271-272`
**Fix:**
- `nvidia_backend`: pass `target_arch` to `triton.compiler.compile(target=self.target_arch.value)` not hardcoded `"cuda"`. Generate per-arch PTX.
- `intel_backend`: remove `target = "xpu" if hasattr(triton.backends, "xpu") else "cuda"` fallback. Instead raise `DependencyMissingError("Intel XPU target not available")` with install instructions for Intel Triton backend.
**Verification:** `grep 'target="cuda"' intel_backend.py` → 0 matches. `grep '"cuda"' nvidia_backend.py` → only in `_run_triton_compile` for the actual compile call.
**Rollback:** `git checkout nvidia_backend.py intel_backend.py`

#### 2.6 — aot_packager: fix hardcoded backend signatures (M-23)
**Type:** FIX
**Files:** `nvidia_backend.py:284-300`, `amd_backend.py:320-330`, `intel_backend.py:253-262`
**Fix:** Replace hardcoded `["*fp32"]*3 + ["i32"]*3 + ["constexpr"]*3` with dynamic signature inference. Use `inspect.signature(fn)` to extract real parameter types. For required pointer args, use `*fp16`/`*fp32`/`*int8` based on inference from grid/block sizes. For constexprs, detect `tl.constexpr` annotations on the function.
**Verification:** `python -c "from src.bridges.aot_packager.nvidia_backend import NvidiaBackend; from triton.testing import do_bench; backend = NvidiaBackend(); result = backend.compile_kernel(kernel_source=sample_matmul, kernel_name='matmul_kernel'); assert result.success"` → succeeds for a non-standard-signature kernel.
**Rollback:** `git checkout nvidia_backend.py amd_backend.py intel_backend.py`

#### 2.7 — cuda_ingest: handle all 5 known limitations (from architecture doc)
**Type:** REDESIGN
**Files:** `cuda_ingest/shared_memory.py`, `translator.py`, `pointer_analysis.py`, `intrinsic_mapper.py`, `kernel_compiler.py`
**Fix:** Address each documented limitation:
1. Multi-dimensional `__shared__` arrays: update `SharedMemoryAnalyzer._parse_declaration` to extract `[d1][d2]...[dN]` and compute `total = d1 * d2 * ... * dN`.
2. `blockIdx * blockDim + threadIdx` idiom: add pattern-matching pass that detects `blockIdx * blockDim + threadIdx` → rewrites to `tl.program_id(0) * tl.num_programs(0) + tl.program_id(0)`.
3. Pointer declarations: add pointer flow analysis (`pointer_analysis.py`) that tracks pointer aliasing.
4. Compound assignment: decompose `+=`, `-=` into `load → modify → store` sequence.
5. C++11+ features: add AST matchers for `auto`, `decltype`, move semantics.
**Verification:** `pytest src/bridges/cuda_ingest/tests/ -v | grep FAILED` → 0 failures (previously 4 failed).
**Rollback:** `git checkout src/bridges/cuda_ingest/`

#### 2.8 — pytorch_xla: fix dtensor_apply (H-04)
**Type:** FIX
**Files:** `dtensor_apply.py:155-187`
**Fix:** Replace `param.data = dtensor.to_local()` with `param.data = dtensor` (keep as DTensor for distributed execution). Remove the comment `# For now, keep local` — this was the old poor-man's approach. The model must remain sharded for multi-device execution.
**Verification:** `python -c "from src.bridges.pytorch_xla.dtensor_apply import DTensorApplier; import torch; ..."` → model parameters are DTensor instances, not local tensors.
**Rollback:** `git checkout dtensor_apply.py`

#### 2.9 — pytorch_xla: fix graph_capture for PyTorch 2.5+ API (H-14)
**Type:** FIX
**Files:** `graph_capture.py:189-209`
**Fix:** Replace `exported = dynamo.export(compiled, *example_inputs); graph = exported[0]` with version-conditional logic:
```python
if hasattr(torch.export, 'export'):
    # PyTorch 2.5+
    ep = torch.export.export(model, example_inputs)
    graph = ep.graph_module
else:
    # PyTorch 2.4 fallback
    compiled = torch.compile(model, fullgraph=True, backend="aot_eager")
    exported = dynamo.export(compiled, *example_inputs)
    graph = exported[0] if isinstance(exported, (list, tuple)) else exported
```
Also add `torch.export.export` to the `try/except ImportError` at top level.
**Verification:** `pytest src/bridges/pytorch_xla/tests/test_pipeline_orchestrator.py -v -k "test_graph_capture"` → passes.
**Rollback:** `git checkout graph_capture.py`

#### 2.10 — Replace extern_bridge touch-file fallback (H-01)
**Type:** FIX
**Files:** `extern_bridge.py:312-324`
**Fix:** Remove the `Path(...).touch()` fallback. If `aotriton` CLI is unavailable, raise `DependencyMissingError("AOTriton CLI required for Tensor Core matmul compilation")` with install instructions. Remove the `rglob("*.{cubin,hsaco,spv,bin}")` cache-copy (which copies stale/random binaries). Instead, call `nvidia_backend.compile_kernel()` directly for the matmul portion.
**Verification:** `grep -n "touch\|rglob\|\.cubin\|fallback" extern_bridge.py` → no more touch-file or stale-copy patterns.
**Rollback:** `git checkout extern_bridge.py`

---

---

### Wave 3: Runtime Productionization

#### 3.1 — memory_reclaimer: change API to return Result[int, NautilusError] (H-10)
**Type:** REDESIGN
**Files:** `memory_reclaimer.py:126-180`, `memory_reclaimer.py:1-30` (docstring)
**Fix:** Change `reclaim()` return type from `int` to `Result[int, NautilusError]`. Four paths currently return 0:
1. Throttled (ok, but only when below watermark) → `Ok(0)`
2. Callback failed → `Err(CallbackError("..."))`
3. Reclaim raised unknown exception → `Err(HardwareProbeError("..."))`
4. Apple returns 0 → `Err(DependencyMissingError("..."))`
Update callers to handle `Result`. Update module docstring to remove the "Never returns 0" lie — now it says "Returns Ok(bytes_reclaimed) or Err(on failure)".
**Verification:** `pytest src/runtime/tests/test_memory_reclaimer.py -v` → all tests pass. `grep "return 0" memory_reclaimer.py | grep -v "def "` → 0 hits.
**Rollback:** `git checkout memory_reclaimer.py`

#### 3.2 — memory_reclaimer: fix Apple _reclaim (H-17)
**Type:** FIX
**Files:** `memory_reclaimer.py:268-282`
**Fix:** Replace `return 0` with real `torch.mps` path:
```python
def _reclaim_apple(self, device_id: str) -> int:
    if not platform.system() == "Darwin":
        raise HardwareNotFoundError(...)
    try:
        import torch
        if hasattr(torch, "mps") and torch.mps.is_available():
            torch.mps.empty_cache()
            # Return changed bytes via torch.mps.current_allocated_memory
            return torch.mps.driver_allocator().get_memory_pressure()
    except (ImportError, AttributeError) as exc:
        raise DependencyMissingError(
            "torch.mps not available; cannot reclaim Apple Metal memory. "
            "Install PyTorch >= 2.0 with MPS support."
        ) from exc
    raise DependencyMissingError(
        "No Apple Metal memory API available; torch.mps not installed"
    )
```
**Verification:** `python -c "from src.runtime.memory_reclaimer import MemoryReclaimer; m = MemoryReclaimer(); m.register_device('metal:0', total_bytes=16*1024**3); print(m.reclaim('metal:0'))"` → returns `Result.Err(...)` on non-Darwin, not `return 0`.
**Rollback:** `git checkout memory_reclaimer.py`

#### 3.3 — async_checkpointer: fix race condition (H-18)
**Type:** FIX
**Files:** `async_checkpointer.py:250-266`
**Fix:** Capture model/optimizer references under the lock, then release before saving:
```python
def _checkpoint_loop(self) -> None:
    while not self._stop_event.is_set():
        triggered = self._pending_event.wait(timeout=self.config.interval_seconds)
        if self._stop_event.is_set():
            break
        # Capture references under lock, clear immediately
        with self._lock:
            model, optimizer = self._pending_model, self._pending_optimizer
            self._pending_model = None
            self._pending_optimizer = None
            self._pending_event.clear()
        # Now save without holding the lock — no TOCTOU race
        if model is not None:
            try:
                self._save_checkpoint_sync(model, optimizer)
            except Exception as exc:
                log.error("checkpoint save failed", error=str(exc))
```
**Verification:** `pytest src/runtime/tests/test_async_checkpointer.py -v` → the `test_checkpoint_loop_does_not_lose_data` test (which we will ADD) asserts that two consecutive `request_checkpoint` calls both save correctly.
**Rollback:** `git checkout async_checkpointer.py`

#### 3.4 — async_checkpointer: fix security issues (H-19, H-20)
**Type:** FIX
**Files:** `async_checkpointer.py:535-539`, `async_checkpointer.py:547-555`
**Fix:**
1. Change `torch.load(io.BytesIO(data), weights_only=False)` to `weights_only=True`. For legacy checkpoints, add a `safe_mode` config option.
2. Replace `pickle.loads(data)` fallback with `raise DependencyMissingError("No safe deserializer; install torch>=2.1")`. Remove the unsafe pickle path entirely.
3. Add `CheckpointConfig.unsafe_pickle_fallback: bool = False` — only for tests that explicitly opt in.
**Verification:** `grep "pickle.loads\|weights_only=False" async_checkpointer.py` → 0 matches. `pytest src/runtime/tests/test_async_checkpointer.py -v -s` → no warnings about pickle.
**Rollback:** `git checkout async_checkpointer.py`

#### 3.5 — math_validator: real bit-exact mode (H-21)
**Type:** REDESIGN
**Files:** `math_validator.py:235-246`
**Fix:** Replace `insert_rounding_correction` (which prepends a comment) with real IR injection:
```python
def insert_rounding_correction(self, ir_text: str, ir_format: str = "ttgir") -> str:
    """Insert IEEE-754 bit-exact flags into the IR.
    
    For TTGIR: insert `tt.func @kernel() { attributes {tt.cuda.fp32_fusion = false} }`.
    For MLIR: insert `llvm.set_rounding_mode(roundtonearest)`.
    For LLVM IR: add `"no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false"` attributes.
    """
    if not self._bit_exact_mode:
        return ir_text
    if ir_format == "ttgir":
        return self._insert_ttgir_flags(ir_text)
    elif ir_format == "mlir":
        return self._insert_mlir_rounding(ir_text)
    elif ir_format == "llvm":
        return self._insert_llvm_attrs(ir_text)
    return ir_text
```
Add three helper methods that parse the IR text and insert the correct directives. Add tests for each format.
**Verification:** `pytest src/runtime/tests/test_math_validator.py -v -k "bit_exact"` → passes. `python -c "from src.runtime.math_validator import MathValidator; mv = MathValidator(); mv.enable_bit_exact_mode(); result = mv.insert_rounding_correction('module { tt.func @matmul(...) }', 'ttgir'); assert 'fp32_fusion = false' in result"` → True.
**Rollback:** `git checkout math_validator.py`

#### 3.6 — math_validator: handle NaN in ULP computation (H-21b)
**Type:** FIX
**Files:** `math_validator.py:198-233`
**Fix:** Add NaN masking:
```python
def _compute_ulp(self, ref: np.ndarray, act: np.ndarray) -> tuple[float, float, float]:
    if ref.size == 0 or act.size == 0:
        raise ValidationError("empty input array")
    # Mask NaN entries
    ref_nan = np.isnan(ref)
    act_nan = np.isnan(act)
    nan_count = int(np.sum(ref_nan | act_nan))
    if nan_count > 0:
        # Don't compare NaN values
        mask = ~ref_nan & ~act_nan
        if not mask.any():
            return 0.0, 0.0, 0.0 if self._bit_exact_mode else (np.inf, np.inf, np.inf)
        ref = ref[mask]
        act = act[mask]
    # ULP computation on valid values
    ulp_err = np.abs(act.astype(np.float64) - ref.astype(np.float64))
    ...
```
Add `nan_count` to the return report. The ULP computation now returns meaningful values even in the presence of NaN.
**Verification:** `python -c "from src.runtime.math_validator import MathValidator; mv = MathValidator(); result = mv.validate_tensors(np.array([1.0, np.nan]), np.array([1.0, np.nan])); assert result.nan_count == 2"` → True.
**Rollback:** `git checkout math_validator.py`

---

### Wave 4: C-API & Version Drift Isolation

#### 4.1 — Build real C-API shared libraries (C-05, C-10)
**Type:** REDESIGN
**Files:** `src/c_api/stubs.cpp`, `src/c_api/CMakeLists.txt`, `third_party/triton`, `third_party/tvm`, `third_party/xla`, NEW `src/c_api/triton_wrapper.cpp`, NEW `src/c_api/tvm_wrapper.cpp`, NEW `src/c_api/xla_wrapper.cpp`
**Fix:** Replace `stubs.cpp` with real wrapper implementations:
- `triton_wrapper.cpp`: links `triton/triton_c.h` (from third_party submodule). `nautilus_compile()` calls `triton::compiler::compile()` via the C API. `nautilus_release()` frees the kernel handle.
- `tvm_wrapper.cpp`: links `tvm/runtime/c_runtime_api.h`. `nautilus_tune()` calls `TVMMetaScheduleTune()`.
- `xla_wrapper.cpp`: links `xla/client/xla_computation.h`. `nautilus_stablehlo_from_fx()` calls `StableHLO::Export()`.
All three linked into `libnautilus_c_api.so`.
**Update CMakeLists.txt:** add `target_link_libraries(nautilus_c_api PUBLIC triton_c_api tvm_c_api xla_c_api)`.
**Verification:** `cmake -GNinja -S src/c_api -B build && ninja -C build` → `libnautilus_c_api.so` created. `python -c "from src.c_api import is_available; assert is_available()"` → True.
**Rollback:** `git checkout src/c_api/ third_party/`

#### 4.2 — Route nvidia_backend through C-API (C-05)
**Type:** REDESIGN
**Files:** `nvidia_backend.py:213-330` (`_run_triton_compile`)
**Fix:** Replace the Python `triton.compiler.compile()` call with:
```python
from src.c_api import compile as c_api_compile
result = c_api_compile(
    kernel_source=kernel_source,
    kernel_name=kernel_name,
    target=f"cuda:{self.target_arch.value}",
    options={"num_warps": num_warps, "num_stages": num_stages},
)
ptx_text = result.get("ptx", "")
cubin_bytes = result.get("cubin")
```
Keep the `TritonMissingError` fallback for when the C library isn't built.
**Verification:** `python -c "from src.bridges.aot_packager.nvidia_backend import NvidiaBackend; b = NvidiaBackend(); r = b.compile_kernel(...)"` → leverages `c_api_compile` traceable in logs. On systems without the .so, falls back to Python path (warning logged).
**Rollback:** `git checkout nvidia_backend.py`

#### 4.3 — Route tvm_adapter through C-API
**Type:** REDESIGN
**Files:** `metaschedule_adapter.py:68-142`
**Fix:** Add a C-API execution path parallel to the existing Python `ms.tune_tir()` path. The Python path stays as fallback; the C-API path is preferred.
**Verification:** `grep "c_api\|nautilus_tune\|TVMTune" metaschedule_adapter.py` → 2+ call sites using the C function.
**Rollback:** `git checkout metaschedule_adapter.py`

#### 4.4 — Route stablehlo_export through C-API
**Type:** REDESIGN
**Files:** `stablehlo_export.py:55-141` (_TorchXLAExporter)
**Fix:** Add `nautilus_stablehlo_from_fx()` call via C-API as the preferred tier (before torch_xla). Only when the .so is built.
**Verification:** `grep "c_api\|nautilus_stablehlo\|StableHLOExport" stablehlo_export.py` → 1+ call site using the C function.
**Rollback:** `git checkout stablehlo_export.py`

#### 4.5 — Add drift-detection CI for C-API compatibility
**Type:** PRODUCTIONIZE
**Files:** `.github/workflows/drift-detection.yml:1-56`
**Fix:** Update the drift-detection workflow to:
1. Pull latest submodules (`git submodule update --remote` for `triton`, `tvm`, `xla`)
2. Build `libnautilus_c_api.so` with the latest deps
3. Run `pytest src/tests/ -v -k "c_api"` (the integration test)
4. If the build fails or tests fail, auto-open a GitHub Issue with the error details
**Verification:** `gh workflow run drift-detection.yml` → workflow succeeds or opens issue on drift.
**Rollback:** `git checkout .github/workflows/drift-detection.yml`

---

### Wave 5: CLI & Distribution

#### 5.1 — Add `nautilus inspect` subcommand
**Type:** FIX
**Files:** NEW `src/cli/commands/inspect.py`, `src/cli/main.py:52-55`
**Fix:**
```python
@click.command("inspect")
@click.argument("fat_binary", type=click.Path(exists=True, dir_okay=False))
def cli(fat_binary: Path):
    """Inspect a fat binary file and print its section metadata."""
    from src.bridges.aot_packager.fat_binary import FatBinary
    data = fat_binary.read_bytes()
    fb = FatBinary.from_bytes(data)
    click.echo(json.dumps({
        "kernel_name": fb.kernel_name,
        "sections": [
            {"vendor": s.vendor, "arch": s.arch, "format": s.format.value, "size": len(s.data)}
            for s in fb.sections
        ],
        "total_size": fb.total_size,
        "created_at": fb.created_at,
    }, indent=2))
```
Register with `cli.add_command(inspect_cmd, name="inspect")`.
**Verification:** `nautilus inspect path/to/kernel.fat.o` → prints JSON.
**Rollback:** `rm src/cli/commands/inspect.py; git checkout src/cli/main.py`

#### 5.2 — Fix CLI help text + honesty pass (H-25)
**Type:** FIX
**Files:** `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `.github/ISSUE_TEMPLATE/config.yml` (if exists)
**Fix:** Update URLs from `https://github.com/nvindia-cud/nautilus` to actual repo URL. Update `README.md` to add a `## Status` section that lists P0-P2 issues. Update `CHANGELOG.md` with the corrected test counts from the audit.
**Verification:** `grep -r "nvindia-cud" docs/ AGENTS.md README.md CONTRIBUTING.md CHANGELOG.md` → 0 matches.
**Rollback:** `git checkout README.md CONTRIBUTING.md CHANGELOG.md`

#### 5.3 — Fix benchmarks: 10 kernels (H-24, M-19)
**Type:** FIX
**Files:** `benchmarks/run_benchmarks.py`, `benchmarks/kernels/*.py`
**Fix:** Implement the 9 missing kernel benchmarks: attention, layer_norm, softmax, gelu, conv2d, embedding, reduce, scan, fused_attention. Each benchmark has:
- A Triton kernel implementation in `benchmarks/kernels/<name>.py`
- A `def benchmark_<name>()` function in `run_benchmarks.py`
- A torch reference implementation for correctness comparison
- A speedup measurement vs baseline (default Triton config)
**Verification:** `python benchmarks/run_benchmarks.py --all --trials 5` → produces `results.json` with 10 entries, each with `speedup` field.
**Rollback:** `git checkout benchmarks/`

#### 5.4 — Add Docker images for all vendors
**Type:** PRODUCTIONIZE
**Files:** NEW `Dockerfile.nvidia`, NEW `Dockerfile.amd`, NEW `Dockerfile.intel`
**Fix:** Each Dockerfile:
- Based on the vendor's base image (CUDA 12.4, ROCm 6.x, oneAPI 2025)
- Installs the project with `pip install -e .[all,dev]`
- Adds a `docker-compose.yml` for multi-vendor testing
- Adds a health check (`nautilus verify`)
**Verification:** `docker build -t nautilus:nvidia -f Dockerfile.nvidia . && docker run nautilus:nvidia nautilus verify` → prints device info.
**Rollback:** `rm Dockerfile.* docker-compose.yml`

---

### Wave 6: Testing & CI/CD

#### 6.1 — Add runtime tests (M-31)
**Type:** FIX
**Files:** NEW `src/runtime/tests/test_memory_reclaimer.py`, NEW `src/runtime/tests/test_async_checkpointer.py`, NEW `src/runtime/tests/test_math_validator.py`
**Fix:** Each test file must have:
- MemoryReclaimer: register_device + reclaim + assert_raises (no GPU) + auto_reclaim_loop (timeout)
- AsyncCheckpointer: save + recover + verify_checksum + test_loop_does_not_lose_data + test_rejects_malformed
- MathValidator: compute_ulp + bit_exact_mode + verify_reproducibility + nan_handling
**Verification:** `pytest src/runtime/tests/ --cov=src.runtime --cov-report=term-missing` → coverage ≥ 80% for each module.
**Rollback:** `rm -rf src/runtime/tests/`

#### 6.2 — Add stablehlo_export + graph_capture tests (M-32)
**Type:** FIX
**Files:** NEW `src/bridges/pytorch_xla/tests/test_stablehlo_export.py`, NEW `src/bridges/pytorch_xla/tests/test_graph_capture.py`
**Fix:** Tests that exercise real `StableHLOExporter` and `GraphCapture`:
- stablehlo_export: test each tier (torch_xla, onnx-bridge, tvmscript) with minimal FX graphs
- graph_capture: test `capture()` with a known `nn.Sequential(Linear, GELU, Linear)` model
**Verification:** `pytest src/bridges/pytorch_xla/tests/test_stablehlo_export.py test_graph_capture.py -v` → all pass.
**Rollback:** `rm src/bridges/pytorch_xla/tests/test_stablehlo_export.py test_graph_capture.py`

#### 6.3 — Add pipeline_orchestrator happy-path test (M-33)
**Type:** FIX
**Files:** NEW (or append to) `src/bridges/pytorch_xla/tests/test_pipeline_orchestrator.py`
**Fix:** Add a test that constructs a real `ShardingConfig(model=nn.Sequential(Linear, GELU, Linear), example_inputs=torch.randn(2, 64), device_mesh=DeviceMesh(...))` and calls `AutoShardingBridge.shard()`. Assert `result.is_usable` is True and `result.dtensor_plan` is not None.
**Verification:** `pytest src/bridges/pytorch_xla/tests/test_pipeline_orchestrator.py -v -k "test_shard_usable"` → passes.
**Rollback:** `git checkout test_pipeline_orchestrator.py`

#### 6.4 — Add C-API integration test
**Type:** FIX
**Files:** NEW `src/tests/integration/test_c_api.py`
**Fix:** Test that:
1. Builds the C library (or skips if CMake not available)
2. Loads it via `ctypes`
3. Calls `nautilus_compile()` with a known Triton kernel
4. Asserts the result contains PTX text
5. Calls `nautilus_detect_vendor()` and asserts return value is valid
**Verification:** `pytest src/tests/integration/test_c_api.py -v` → passes (or skips with clear message).
**Rollback:** `rm src/tests/integration/test_c_api.py`

#### 6.5 — Fix ci.yml coverage gate (H-26)
**Type:** FIX
**Files:** `.github/workflows/ci.yml:63`
**Fix:** Increase `--cov-fail-under` to `70` AND remove `--ignore` lines from test paths so coverage includes bridge code. The `test-cpu` job already installs `torch`+`triton` (CPU wheels) — no reason to ignore bridge tests.
**Verification:** `gh workflow run ci.yml` → test-cpu job passes with coverage ≥ 70%.
**Rollback:** `git checkout .github/workflows/ci.yml`

#### 6.6 — Fix ci.yml grep guardrail (M-18)
**Type:** FIX
**Files:** `.github/workflows/ci.yml:32-38`
**Fix:** Change the grep guardrail from "fails on ANY `# TODO:` to "fails on `# TODO(no-issue):`" (i.e., TODOs without an issue reference). This allows TODOs that reference a GitHub issue but catches unattached ones.
```bash
if grep -RIn --include="*.py" -E "#\s*(TODO|FIXME|XXX):\s*\(no-issue\)" src/; then
  echo "Found TODO/FIXME comments without a linked issue. Open a GitHub issue first." >&2
  exit 1
fi
```
**Verification:** `grep 'TODO' src/bridges/triton_tvm/*.py | grep -v '(no-issue)'` → 0 matches (all TODOs reference an issue).
**Rollback:** `git checkout .github/workflows/ci.yml`

#### 6.7 — Add real-hardware CI with GitHub-hosted GPU runners (M-38)
**Type:** REDESIGN
**Files:** `.github/workflows/real-hardware.yml`
**Fix:** Change from `runs-on: [self-hosted, gpu, nvidia]` to `runs-on: gpu-runner-4xlarge` (or equivalent GitHub-hosted GPU runner). If GitHub-hosted GPU runners are not available, use `on: workflow_dispatch` for manual trigger and document the runner setup. Add a `matrix` for multiple GPU architectures.
**Verification:** `gh workflow run real-hardware.yml --ref main` → workflow starts.
**Rollback:** `git checkout .github/workflows/real-hardware.yml`

#### 6.8 — Remove session logs + cache files from git tracking (M-21)
**Type:** FIX
**Files:** `session-ses_16a6.md`, `session-ses_166a.md`, `.gitignore`, `__pycache__/`, `.opencode/node_modules/`
**Fix:** Add to `.gitignore`:
```
# AI session logs
session-*.md
```
Remove tracked session files: `git rm --cached session-ses_16a6.md session-ses_166a.md`. Add `__pycache__/` and `.opencode/node_modules/` to `.gitignore` if not already present. Clean cache dirs: `git rm -r --cached */__pycache__/ .opencode/node_modules/`.
**Verification:** `git ls-files session-*.md` → empty. `git ls-files */__pycache__/` → empty.
**Rollback:** `git checkout .gitignore; git checkout session-ses_16a6.md session-ses_166a.md` (if needed, with `git rm --cached` first)

---

### Wave 7: Enterprise Completeness

#### 7.1 — Structured logging everywhere (M-04)
**Type:** FIX
**Files:** `types.py:484-492`, `errors.py:124-132`, plus any other stdlib `logging.getLogger` calls
**Fix:** Replace all `import logging; logging.getLogger(__name__)` with `from src.common.logging import get_logger`. Search:
```bash
git grep "import logging" src/bridges/ src/common/ src/runtime/ src/cli/ src/c_api/
```
Replace each with the structured logger. Add span_id and stage to every log call.
**Verification:** `grep -r "logging.getLogger" src/ --include="*.py"` → 0 matches outside `src/common/logging.py`.
**Rollback:** `git checkout <changed-files>` (too many to list individually — commit in batches)

#### 7.2 — Result[T, E] adoption across all bridges (M-16)
**Type:** PRODUCTIONIZE
**Files:** `tune.py:27`, `build.py:25`, `bridge_orchestrator.py`, `metaschedule_adapter.py`, memory_reclaimer.py`, `async_checkpointer.py`
**Fix:** Change fallible functions to return `Result[T, E]`:
- `metaschedule_adapter.tune()` → `Result[MappedTuningConfig, TuningError]`
- `memory_reclaimer.reclaim()` → `Result[int, NautilusError]`
- `async_checkpointer._save_checkpoint_sync()` → `Result[CheckpointInfo, CheckpointError]`
- `bridge_orchestrator._tuning_chain()` → `Result[MappedTuningConfig, TuningError]`
**Verification:** `grep -r "-> Result\[" src/ --include="*.py" | wc -l` → ≥ 20 (currently ~5).
**Rollback:** `git checkout <changed-files>`

#### 7.3 — SINGLETON fix for get_default_breakers() (M-13)
**Type:** FIX
**Files:** `observability.py:223-241`, `async_checkpointer.py:186-194`
**Fix:** Convert `get_default_breakers()` to return a module-level singleton:
```python
_DEFAULT_BREAKERS: dict[str, CircuitBreaker] | None = None
def get_default_breakers() -> dict[str, CircuitBreaker]:
    global _DEFAULT_BREAKERS
    if _DEFAULT_BREAKERS is None:
        _DEFAULT_BREAKERS = {
            "tvm_tune": CircuitBreaker(...),
            "triton_compile": CircuitBreaker(...),
            ...
        }
    return _DEFAULT_BREAKERS

def reset_default_breakers() -> None:
    """For test use only."""
    global _DEFAULT_BREAKERS
    _DEFAULT_BREAKERS = None
```
Fix `async_checkpointer.py:186-194` — since `get_default_breakers()` now returns the singleton, the `if "checkpoint_io" not in breakers:` mutation correctly adds to the global dict.
**Verification:** `python -c "from src.common.observability import get_default_breakers; b1 = get_default_breakers(); b2 = get_default_breakers(); assert b1 is b2"` → passes.
**Rollback:** `git checkout observability.py async_checkpointer.py`

#### 7.4 — TimeoutManager: fix finally-clause exception masking (M-14)
**Type:** FIX
**Files:** `observability.py:298-320`
**Fix:** Check for active exception before raising:
```python
@contextmanager
def stage(self, name: str) -> Iterator[None]:
    ...
    try:
        yield
    except BaseException:
        # Log the timeout but don't mask the original error
        elapsed = time.time() - stage_start
        if elapsed > budget:
            log.warning("stage_exceeded_budget",
                       stage=name, elapsed=elapsed, budget=budget,
                       masked_by_error=sys.exc_info()[1])
        raise
    else:
        elapsed = time.time() - stage_start
        if elapsed > budget:
            raise StageTimeoutError(...)
```
**Verification:** `pytest src/common/tests/test_common.py -v -k "stage_timeout"` → passes. `pytest src/common/tests/test_common.py -v -k "stage_exception_masking"` (NEW test) → asserts original exception is preserved.
**Rollback:** `git checkout observability.py`

#### 7.5 — CircuitBreaker: catch Exception not BaseException (M-15)
**Type:** FIX
**Files:** `observability.py:187-192`
**Fix:** Change:
```python
except BaseException as exc:
```
to:
```python
except (self._config.excluded_exceptions + (Exception,)) as exc:
```
Let `KeyboardInterrupt`, `SystemExit`, `GeneratorExit` propagate without entering the breaker.
**Verification:** `pytest src/common/tests/test_common.py -v -k "circuit_breaker"` → passes.
**Rollback:** `git checkout observability.py`

#### 7.6 — Fix common/types.py issues (M-01, M-02, M-03)
**Type:** FIX
**Files:** `types.py:42-46`, `types.py:170-171`, `types.py:107-124`, `types.py:330-347`
**Fix:**
1. `Vendor.from_string`: add `strict: bool = True` flag. When `strict=True`, raise `ConfigError`. When `strict=False`, return `UNKNOWN` (backward compat).
2. `KernelSection.sha256`: add `functools.cached_property` or `lru_cache(maxsize=1)`.
3. `HardwareTarget.to_tvm_target`: replace if-chain with a `TVM_TARGET_ALIASES: dict[tuple[Vendor, Arch], str]` table. Add a `MappingValidationError` raised at import-time if any Arch is referenced by a key the table doesn't cover.
4. `TuningConfig.to_triton_config`: move `import triton` to a `functools.cache`-decorated module-level helper.
**Verification:** `pytest src/common/tests/test_common.py -v` → all pass (no behavior change).
**Rollback:** `git checkout types.py`

#### 7.7 — Fix common/errors.py issues (M-05, M-06, M-07)
**Type:** FIX
**Files:** `errors.py:134-144`, `errors.py:365-376`, `errors.py:124-132`
**Fix:**
1. `with_context`: use `dataclasses.replace(self, context=new_ctx)` instead of `type(self)(code=self.code, ...)`. Add unit test asserting the subclass-specific code is preserved.
2. `StageTimeoutError`: change code to `E_STAGE_TIMEOUT`. `TotalBudgetExceededError`: change code to `E_TOTAL_BUDGET_EXCEEDED`.
3. `to_dict` `repr(self.cause)`: wrap in `try/except`, fall back to `"<unprintable>"`.
**Verification:** `pytest src/common/tests/test_common.py -v -k "error"` → passes with 3 new assertions.
**Rollback:** `git checkout errors.py`

#### 7.8 — Fix common/observability.py issues (M-08, M-09, M-10)
**Type:** FIX
**Files:** `hardware.py:543-572`, `hardware.py:247-290`, `hardware.py:437-484`
**Fix:**
1. `enumerate_devices()`: add `@functools.lru_cache(maxsize=1)`.
2. `lspci -nn -mm` parser: use `shlex.split(line)` instead of `line.split('"')`. Add `_LspciLine` dataclass.
3. Intel vs AMD collision: change `/dev/dri/renderD*` to also read `vendor_id` from lspci. Only report as Intel if vendor_id is `0x8086`. Only report as AMD if vendor_id is `0x1002`.
**Verification:** `python -c "from src.common.hardware import enumerate_devices; devs = enumerate_devices(); print(f'Found {len(devs)} devices')"` → runs once (cached). `python -c "from src.common.hardware import enumerate_devices; id1 = id(enumerate_devices()); id2 = id(enumerate_devices()); assert id1 == id2"` → True.
**Rollback:** `git checkout hardware.py`

#### 7.9 — Fix runtime_stub.c for ARM64 (H-16)
**Type:** FIX
**Files:** `runtime_stub.c:75-110`
**Fix:** Add ARM64 Linux support:
```c
#if defined(__aarch64__)  // ARM64 Linux
// Use getdents64 via syscall (different syscall number on aarch64)
#define __NR_getdents64 217  // aarch64 getdents64 syscall
// Or use access() for a simpler approach
static int nautilus_dir_has_entry(const char* dir, const char* prefix) {
    // Fall back to access-based approach for ARM64
    // /dev/nvidia*: check /dev/nvidia0 via access()
    // /dev/kfd: check directly
    // /dev/dri/renderD*: check each via access()
    // This is slower but portable
    ...
}
```
Alternatively, remove the inline assembly entirely and use only `access()` calls (portable across architectures). The `/dev/dri/renderD*` check can use `access(prefix) for each known device number`.
**Verification:** `aarch64-linux-gnu-gcc runtime_stub.c -c -o runtime_stub.o` → compiles without error. No `getdents64` syscall used.
**Rollback:** `git checkout runtime_stub.c`

#### 7.10 — Remove deprecated stubs after replacing (C-08, C-09 confirm deletion)
**Type:** FIX
**Files:** `builder.py:423-449`, `linker.py:362-407`
**Fix:** Delete the methods. Remove from `__init__.py` exports if listed. Update any imports that reference them.
```bash
git diff HEAD -- src/bridges/aot_packager/builder.py src/bridges/aot_packager/linker.py | grep "^[-]" | grep "def _minimal_elf\|def _write_minimal_fat\|DEPRECATED"
```
Expected output: empty (no calls remain).
**Verification:** `grep -r "_minimal_elf_stub\|_write_minimal_fat_binary" src/` → 0 matches.
**Rollback:** `git checkout builder.py linker.py` (only if the new implementations are present from earlier tasks).

---

## 4. Verification Strategy

### Per-Wave Gates

| Wave | Gate | Command |
|------|------|---------|
| Wave 0 | Package installs | `pip install -e .[dev,all]` → `nautilus --help` exits 0 |
| Wave 0 | Tests unblocked | `pytest src/ --collect-only | grep "296"` |
| Wave 0 | Production bugs fixed | `pytest src/bridges/ -v -k "test_" | grep "FAILED" | wc -l` → 0 |
| Wave 1 | Translator works | `python -c "from src.bridges.pytorch_xla.stablehlo_to_triton import translate; s = translate('...'); assert 'tl.dot' in s.source"` |
| Wave 1 | Shard emits fat binary | `python scripts/demo_e2e.py --mesh 1,1 --model hf-internal-testing/tiny-llama && test -f shards/shard_0000/kernel.fat.o` |
| Wave 1 | Real GSPMD annotations | `grep "mhlo.sharding\|stablehlo.sharding" output.mlir | grep -v "^\s*//"` (not comment) |
| Wave 2 | Real HW validation | `python -c "from src.bridges.aot_packager.hardware_validator import HardwareValidator; r=HardwareValidator().validate('/nonexistent','nvidia','sm_90'); assert not r.passed"` |
| Wave 2 | Valid ELF sections | `readelf -S kernel.fat.o | grep ".nv_kernel\|.amd_kernel\|.intel_kernel"` → section names exist |
| Wave 2 | Stubs deleted | `grep -r "DEPRECATED" src/bridges/aot_packager/` → 0 matches |
| Wave 3 | No return 0 in reclaim | `grep "return 0" memory_reclaimer.py | grep -v "def \|# "` → 0 matches |
| Wave 3 | No pickle/unsafe | `grep "pickle\|weights_only=False" async_checkpointer.py` → 0 matches |
| Wave 3 | Real bit-exact mode | `grep "fp32_fusion\|roundtonearest" math_validator.py` → 2+ matches |
| Wave 4 | C-API loads | `python -c "from src.c_api import is_available; assert is_available()"` → True |
| Wave 4 | C-API used in bridge | `grep "from src.c_api import\|c_api_compile" nvidia_backend.py` → 1+ match |
| Wave 5 | inspect works | `nautilus inspect path/to/kernel.fat.o` → prints JSON |
| Wave 5 | 10 benchmarks | `python benchmarks/run_benchmarks.py --list` → 10 entries |
| Wave 6 | Coverage ≥ 70% | `pytest src/ --cov=src --cov-report=term-missing --cov-fail-under=70` → pass |
| Wave 6 | No session logs | `git ls-files session-*.md` → empty |
| Wave 7 | All stdlib replaced | `grep -r "logging.getLogger" src/ --include="*.py" | grep -v "src/common/logging.py" | wc -l` → 0 |
| Wave 7 | Result adoption | `grep -r "-> Result\[" src/ --include="*.py" | wc -l` → ≥ 20 |

### Final Definition of Done

The codebase is "enterprise-grade" when ALL of these are true:

1. **✅ Installability**: `pip install nautilus` works, `nautilus --help` shows all commands.
2. **✅ End-to-end pipeline**: `python scripts/demo_e2e.py --model hf-internal-testing/tiny-llama --mesh 1,1` exits 0 and produces per-shard `kernel.fat.o` files that start with `b"\x7fELF"`.
3. **✅ Numerical correctness**: The fat binary output matches PyTorch eager reference within `1e-2` (fp16) or `1e-5` (fp32).
4. **✅ C-API isolation**: The C-API library builds and is consumed by at least one bridge (nvidia_backend). Version drift breaks only the C-API wrapper, not the bridge code.
5. **✅ Real hardware CI**: At least one job in `real-hardware.yml` runs on real GPU hardware and asserts the demo passes.
6. **✅ 10 kernel benchmarks**: `benchmarks/run_benchmarks.py --all` produces 10 entries with speedup measurements.
7. **✅ 296 tests pass (or skip correctly)**: `pytest src/ -v` shows all tests collected, with CPU-only tests passing and GPU tests skipped (not blocked).
8. **✅ No stubs/placeholders**: `grep -r "DEPRECATED\|_generate_shard_source\|_minimal_elf_stub\|_write_minimal_fat_binary" src/` returns 0 matches.
9. **✅ Honest CHANGELOG**: The CHANGELOG test count matches the actual `pytest --collect-only` count. The README Status section lists any unmet requirements.
10. **✅ Cross-vendor claim is proven or removed**: Either cross-vendor collective communication works (with `benchmarks/cross_vendor.json` proof) OR the README explicitly says "best-effort".

---

## 5. Rollback Strategy

| Wave | Files modified | Rollback command |
|------|---------------|------------------|
| 0 | setup.py, metadata_extractor, config_mapper, bridge_orchestrator, pyproject.toml, ir_capture, hooks, compiler, ci.yml | `git checkout -- .` (safe to revert all at once) |
| 1 | stablehlo_to_triton.py (NEW), shard.py, hardware_orchestrator.py, pipeline_orchestrator.py, gspmd_runner.py, scripts/demo_e2e.py (NEW) | `rm stablehlo_to_triton.py demo_e2e.py; git checkout shard.py hardware_orchestrator.py pipeline_orchestrator.py gspmd_runner.py` |
| 2 | metaschedule_adapter, bridge_orchestrator, ir_to_tir/, hardware_validator, linker, builder, nvidia_backend, amd_backend, intel_backend, cuda_ingest/, dtensor_apply, graph_capture, extern_bridge | `git checkout -- src/bridges/` |
| 3 | memory_reclaimer, async_checkpointer, math_validator | `git checkout -- src/runtime/` |
| 4 | c_api/, third_party/ (submodules), nvidia_backend, metaschedule_adapter, stablehlo_export, drift-detection.yml | `git checkout -- src/c_api/ .github/workflows/drift-detection.yml; git submodule deinit --all` |
| 5 | inspect.py (NEW), main.py, README, CONTRIBUTING, CHANGELOG, benchmarks/, Dockerfile.* | `rm inspect.py Dockerfile.*; git checkout -- src/cli/main.py README.md CONTRIBUTING.md CHANGELOG.md benchmarks/` |
| 6 | runtime/tests/ (NEW), pytorch_xla/tests/ (NEW), c_api test (NEW), ci.yml, real-hardware.yml, .gitignore | `rm src/runtime/tests/*.py; git checkout .gitignore .github/workflows/ci.yml .github/workflows/real-hardware.yml` |
| 7 | types.py, errors.py, hardware.py, logging.py, observability.py, builder.py, linker.py, runtime_stub.c | `git checkout -- src/common/ src/bridges/aot_packager/builder.py src/bridges/aot_packager/linker.py src/bridges/aot_packager/runtime_stub.c` |

**Maximum rollback time**: 5 minutes (7 git checkout commands, 4 rm commands).

**Lossy point**: After Wave 4 (C-API linking to submodule deps), reverting Wave 4 removes the C-API but the routing code in nvidia_backend still has the `from src.c_api import compile` fallback built in, so the bridge still works.

---

## 6. Parallelization Matrix

| Wave | Parallel group | Concurrent tasks | Dependencies |
|------|---------------|------------------|--------------|
| Wave 0 | - | 3 tracks |
| | 0.1-0.2, 0.3 | 2 agents | None |
| | 0.4, 0.5-0.7 | 2 agents | None |
| | 0.8 | 1 agent | 0.5 (key fix needed before submodule build) |
| Wave 1 | - | 3 tracks | Wave 0 complete |
| | 1.1 (translator — biggest task) | 1 deep agent | 0.4 (bugfixes) |
| | 1.2, 1.3, 1.4 (CLI wiring) | 1 agent | 1.1 (needs translator) |
| | 1.5, 1.6 (GSPMD + demo) | 1 agent | 1.2-1.4 (needs CLI wired) |
| Wave 2 | - | 4 tracks | Wave 1 complete (all wiring in place) |
| | 2.1 (triton_tvm hardening) | 1 agent | None |
| | 2.2-2.6 (aot_packager hardening) | 1 agent | None |
| | 2.7 (cuda_ingest) | 1 agent | None |
| | 2.8-2.10 (pytorch_xla fixes) | 1 agent | None |
| Wave 3 | - | 2 tracks | Wave 0 complete (bugfixes needed for safety) |
| | 3.1-3.2 (memory_reclaimer) | 1 agent | None |
| | 3.3-3.6 (checkpointer + validator) | 1 agent | None |
| Wave 4 | - | 2 tracks | Wave 0 complete (submodules needed) |
| | 4.1 (build C-API lib) | 1 agent | 0.8 (submodules) |
| | 4.2-4.5 (route bridges) | 1 agent | 4.1 (needs .so) |
| Wave 5 | - | 3 tracks | Wave 1 complete (pipeline stable) |
| | 5.1 (inspect command) | 1 agent | None |
| | 5.2 (URL fix + honesty) | 1 agent | None |
| | 5.3 (benchmarks) | 1 agent | None |
| | 5.4 (Docker) | 1 agent | None |
| Wave 6 | - | 4 tracks | Wave 0-3 complete (tests need real code) |
| | 6.1 (runtime tests) | 1 agent | Wave 3 (runtime productionized) |
| | 6.2-6.4 (bridge tests) | 1 agent | Wave 1-2 (bridges hardened) |
| | 6.5-6.7 (CI fixes) | 1 agent | Wave 0 (--ignore removed) |
| | 6.8 (git cleanup) | 1 agent | None |
| Wave 7 | - | 4 tracks | All previous waves complete (foundation stable) |
| | 7.1-7.5 (logging, Result, singletons) | 1 agent | None |
| | 7.6-7.8 (types, errors, hardware) | 1 agent | None |
| | 7.9 (ARM64 stub) | 1 agent | None |
| | 7.10 (delete deprecated stubs) | 1 agent | Wave 2.4 (replacements done) |

**Maximum parallelism: 10 concurrent tasks** (Wave 2 + Wave 3 + parts of Wave 5).

---

## Appendix A: LOW, MINOR, and NICE-TO-HAVE Issues

### LOW (P3) — 35 items

| ID | Issue | File:Line | Fix |
|----|-------|-----------|-----|
| L-01 | `test_property_based.py` has unused `@given(v)` parameter | `common/tests/test_property_based.py:77` | Remove `@given` from `test_arch_vendor_mapping_injective` |
| L-02 | `tir_template.py` uses `tvm.script.tirx` (TVM 0.18+ only) | `tir_template.py:27` | Add `try/except AttributeError` fallback to `tvm.script.tir` |
| L-03 | `stablehlo_export.py` uses `tvm.contrib.stablehlo` (TVM 0.18+ only) | `stablehlo_export.py:276` | Guard import with `TVM_AVAILABLE` check |
| L-04 | `hooks.py` uses `triton.knobs` (Triton 3.7+ only) | `hooks.py:37` | Fall back gracefully on `ImportError` |
| L-05 | `nvidia_backend.py` imports `triton.runtime.jit` (internal) | `nvidia_backend.py:275` | Add `# type: ignore` + version-guard |
| L-06 | `amd_backend.py` imports `triton.compiler.ASTSource` (internal) | `amd_backend.py:301` | Add `# type: ignore` + version-guard |
| L-07 | `intel_backend.py` imports same | `intel_backend.py:251` | Same |
| L-08 | `metal_backend.py` imports same | `metal_backend.py:220` | Same |
| L-09 | `aotriton.compile()` signature is hypothetical | `amd_backend.py:204` | Document actual tested versions |
| L-10 | `_do_reclaim` docstring says `DependencyMissingError` but raises `HardwareProbeError` | `memory_reclaimer.py:201-204` | Align docstring or change code |
| L-11 | `FatBinaryConfig` has no `output_dir` validation | `builder.py:50-78` | Add `__post_init__` validation |
| L-12 | `builder.py` per-vendor errors logged but not aggregated in result | `builder.py:313-358` | Add `errors` dict to result |
| L-13 | `linker._wrap_section_data` has no string table | `linker.py:251-315` | REDESIGNED in Wave 2.3 |
| L-14 | `bridge_orchestrator._tune_chain` bare `except Exception` | `bridge_orchestrator.py:540-577` | Replace with specific exception handlers |
| L-15 | `extern_bridge` has no Tensor Core path | `extern_bridge.py:272-397` | REDESIGNED in Wave 2.10 |
| L-16 | `comm_backend` costs hardcoded | `comm_backend.py:133-141` | REDESIGNED in Wave 1 (GSPMD) |
| L-17 | `_reclaim_apple` comment admits to anti-pattern | `memory_reclaimer.py:268-282` | FIXED in Wave 3.2 |
| L-18 | `pipeline_orchestrator` breaker name collision | `pipeline_orchestrator.py:263` | FIXED in Wave 1.3 |
| L-19 | `_capture_via_compile` unpacks old API | `graph_capture.py:205-206` | FIXED in Wave 2.9 |
| L-20 | `get_default_breakers` fresh dict per call | `observability.py:223-241` | FIXED in Wave 7.3 |
| L-21 | `with_context` strips subclass code | `errors.py:134-144` | FIXED in Wave 7.7 |
| L-22 | `StageTimeoutError` + `TotalBudgetExceededError` share code | `errors.py:365-376` | FIXED in Wave 7.7 |
| L-23 | `NautilusError.to_dict` may raise on broken `repr` | `errors.py:124-132` | FIXED in Wave 7.7 |
| L-24 | `TimeoutManager.stage` masks exception | `observability.py:298-320` | FIXED in Wave 7.4 |
| L-25 | CI grep guardrail blocks legitimate TODOs | `ci.yml:32-38` | FIXED in Wave 6.6 |
| L-26 | `pyproject.toml` pins old `torch==2.4.1, triton==3.0.0` | `pyproject.toml:51-53` | Update to `torch>=2.4,<3.0, triton>=3.0,<4.0` |
| L-27 | `session-ses_*.md` files committed | repo root | FIXED in Wave 6.8 |
| L-28 | `third_party/` submodules missing | `third_party/` | REDESIGNED in Wave 0.8 |
| L-29 | `_CAPTURE_BUFFER` global dict in `compiler.py` no locking | `backend/compiler.py:46` | Add `threading.Lock` |
| L-30 | `real-hardware.yml` self-hosted runners don't exist | `.github/workflows/real-hardware.yml` | REDESIGNED in Wave 6.7 |
| L-31 | `ir_classifier.py:test_classify_attention` fails | `ir_classifier.py` | FIXED in Wave 0.4 |
| L-32 | `test_timeout_manager:test_stage_under_budget_succeeds` fails | `timeout_manager.py` | FIXED in Wave 0.4 |
| L-33 | `kernels/matmul.py` is the only actual benchmark | `benchmarks/kernels/` | FIXED in Wave 5.3 |
| L-34 | `c_api/__init__.py` `_C_LIB_LOAD_ERROR` never read | `c_api/__init__.py:290-296` | Expose or remove dead global |
| L-35 | Build-time dep `Cython` needed for TVM wheel | `pyproject.toml:2` | Document in install instructions |

### MINOR (P4) — 10 items

| ID | Issue | Fix |
|----|-------|-----|
| N-01 | `__pycache__/` directories in tree | Add to `.gitignore` + `git rm --cached` |
| N-02 | `.opencode/node_modules/` in tree | Add to `.gitignore` |
| N-03 | `prompts.md` file in repo root | Move to `.opencode/` or delete |
| N-04 | `session-ses_166a.md` untracked but should be `.gitignore`d | Add `session-*.md` to `.gitignore` |
| N-05 | `.omo/plans/` committed (audit files) | Add `.omo/plans/` to `.gitignore` OR keep as project artifacts (intentional) |
| N-06 | `LICENSE` format: year range `"2024-2026"` should be `"2026"` | Fix copyright year |
| N-07 | `.hypothesis/` in tree | Add to `.gitignore` |
| N-08 | `.mypy_cache/` in tree | Already gitignored; run `find . -name __pycache__ -exec rm -rf {} +` |
| N-09 | `AGENTS.md` mentions "NVINDIA_CUD" as repo name (should be Nautilus) | Fix repo description |
| N-10 | `opencode.json` in tree has stale project name | Fix project name |

### NICE-TO-HAVE (P5) — 12 items

| ID | Proposal | Priority |
|----|----------|----------|
| NH-01 | Prometheus metrics export (kernel compile times, shard latency, cache hit rate) | High |
| NH-02 | OpenTelemetry tracing (distributed trace across bridge pipeline) | High |
| NH-03 | Health check HTTP endpoint (`/health` → GPU status + bridge state) | Medium |
| NH-04 | Performance regression CI (compare new benchmark results against committed baseline) | Medium |
| NH-05 | `nautilus serve` (HTTP server for model deployment) | Medium |
| NH-06 | `nautilus cache list/clear/status` (cache management subcommand) | Low |
| NH-07 | Auto-update check (`nautilus update --check` for new versions) | Low |
| NH-08 | Web UI for `nautilus verify` (dashboard in browser) | Low |
| NH-09 | Jupyter notebook tutorial (`notebooks/quickstart.ipynb`) | Low |
| NH-10 | `nautilus status` (show what's installed, which vendors work) | Low |
| NH-11 | Plugin architecture for custom backends | Low |
| NH-12 | Native Python API docs (`pydoc nautilus`) | Low |

---

## 7. Execution Handoff

**Plan complete and saved to `.omo/plans/nautilus-enterprise-grade-plan.md`.**

**Recommended execution approach: Subagent-Driven Development**
Wave-by-wave execution with `subagent-driven-development` skill. Each wave dispatches parallel subagents per the parallelization matrix, with review between waves.

**Prerequisites before starting any wave:**
- [x] `pip install -e .[dev]` succeeds (Wave 0.1, fix setup.py first) — ✅ DONE
- [x] `pytest src/ --collect-only | grep 296` shows 296 tests collected (Wave 0.3, remove --ignore patterns) — ✅ DONE
- [x] `git submodule update --init` populates `third_party/` (Wave 0.8) — ✅ DONE: triton@v3.0.0, tvm@v0.18.0, xla@e115cfc, llvm@19.1.0

**First wave to execute: Wave 0 (Foundation Repairs) — all other waves depend on it.**

**To begin execution in a new session:**
```
task(subagent_type="plan", prompt="Read .omo/plans/nautilus-enterprise-grade-plan.md and execute Wave 0 tasks. Start with 0.1 (fix setup.py).")
```
