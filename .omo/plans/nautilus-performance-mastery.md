# Nautilus Performance Mastery Plan

## TL;DR

> **Quick Summary**: Transform Nautilus from 85/100 (audited state) to genuine best-in-class by implementing the performance superiority stack: expert-guided auto-tuning, kernel fusion, transfer learning, C++ MLIR normalizer, and a performance database. Every fix is a redesign — nothing is removed or documented as "limitation."
>
> **Deliverables**:
> - All 17 pre-existing test failures fixed (F2 review)
> - All 235 ruff violations → 0 (no `# noqa` crutches)
> - All 152 mypy errors → 0 (proper type annotations, not type: ignore)
> - Expert-guided auto-tuning rules (H100/MI300X/Gaudi/Apple)
> - Kernel fusion engine (cross-op + cross-device)
> - Performance database with transfer learning
> - C++ MLIR Vector Dialect normalizer (10-100x faster IR compilation)
> - `nautilus perf` CLI dashboard
> - End-to-end pipeline: PyTorch → shard → tune → build → dispatch → benchmark
>
> **Estimated Effort**: 19 weeks (4-5 months full-time)
> **Parallel Execution**: YES — 6 waves, up to 6 tasks per wave
> **Critical Path**: Wave 1 (foundation) → Wave 2 (expert tuning) → Wave 5 (C++ normalizer) → Wave 6 (E2E wiring)

---

## Context

### Original Request
Make Nautilus the best AI compilation framework in the world. Not "good enough" — genuinely superior. Fix all remaining issues with enterprise production quality, following the philosophy of "fix or redesign, never remove or document as limitation."

### Current State (from assessment)

```
SCORE: 90/100

Strengths:
  ✅ Architecture correct (wiring, not inventing)
  ✅ 4 vendor backends (Nvidia, AMD, Intel, Apple) — real code
  ✅ Auto-tuning bridge with 6-tier fallback
  ✅ Auto-sharding with GSPMD, collective insertion, NCCL↔RCCL
  ✅ Fat binary ELF with .nautilus.index
  ✅ End-to-end pipeline CLI
  ✅ 1,043 tests passing (97.5%)

Remaining Gaps:
  ❌ 17 pre-existing test failures (parser, classifier edge cases)
  ❌ 495 ruff errors (235 after auto-fix)
  ❌ 152 mypy errors (most are pre-existing SDK stubs)
  ❌ 121 files need reformatting
  ❌ Missing SectionFormat.METALLIB/AIR
  ❌ Conftest conflict in aot_packager tests
  ❌ 235 lint violations
  ❌ MLIR Vector Dialect normalizer (the arch gap)
  ❌ Expert-guided tuning rules
  ❌ Cross-device kernel fusion
  ❌ Transfer learning between vendors
  ❌ Hand-optimized kernel templates
```

### Philosophical Foundation

**"Why are you what?"** — Nautilus exists to break Nvidia's monopoly by being **genuinely superior**, not by being "CUDA that runs on AMD." 

Every fix must:
1. **Fix or redesign** — never remove or document as "limitation"
2. **Enterprise production quality** — not MVP, not basic, not "production ready for small things only"
3. **End-to-end wiring** — not isolated implementations
4. **Consistent practices** — zero anti-patterns, zero `# type: ignore` crutches, zero `Any` where typed is possible
5. **Skill-driven** — leverage gpu-architect, compiler-architect, triton-compiler, xla-sharding expertise

### Skill Loading Strategy

For every task, the assigned agent will load skills appropriate to the domain:
- **`gpu-architect`**: H100/MI300X/Gaudi hardware optimization knowledge
- **`compiler-architect`**: MLIR/LLVM/TVM compilation pipeline expertise
- **`triton-compiler`**: Triton JIT/AOT compiler internals
- **`xla-sharding`**: GSPMD, StableHLO, OpenXLA integration
- **`fat-binary-packaging`**: Multi-vendor ELF packaging
- **`distributed-ai-expert`**: Multi-GPU/multi-node communication patterns
- **`performance-analyst`**: Benchmark methodology, roofline analysis
- **`god-mode`**: Holistic review of all 7 dimensions

---

## Work Objectives

### Core Objective
Transform Nautilus from 90/100 to 100/100 by:
1. Eliminating all remaining code quality issues (linting, typing, testing)
2. Implementing the performance superiority stack (expert rules, fusion, transfer learning)
3. Closing the MLIR Vector Dialect arch gap
4. Wiring everything end-to-end through the performance dashboard

### Concrete Deliverables
- [ ] Zero ruff violations
- [ ] Zero mypy errors
- [ ] Zero pre-existing test failures (17 → 0)
- [ ] Expert rules for H100/MI300X/Gaudi/Apple
- [ ] Kernel fusion engine (cross-op + cross-device)
- [ ] Performance database with transfer learning
- [ ] C++ MLIR Vector Dialect normalizer
- [ ] `nautilus perf` CLI dashboard
- [ ] End-to-end pipeline verified with benchmarks
- [ ] 100% test pass rate

### Definition of Done
- [ ] `ruff check src/` → exit 0
- [ ] `mypy src/ --ignore-missing-imports` → exit 0
- [ ] `pytest src/ -v` → 100% pass
- [ ] `nautilus perf` shows benchmark trends
- [ ] `nautilus pipeline` runs end-to-end with auto-tuning on all 4 vendors
- [ ] Performance database persists measurements
- [ ] C++ normalizer integrates with Python pipeline

### Must Have
- Zero pre-existing test failures
- Zero ruff violations (no `# noqa` crutches)
- Zero mypy errors (proper type annotations, not type: ignore)
- Expert-guided tuning rules encoded in code
- Kernel fusion engine (cross-op + cross-device)
- Performance database with transfer learning
- MLIR Vector Dialect normalizer
- End-to-end pipeline performance verification

### Must NOT Have (Guardrails)
- **No `# type: ignore` or `Any` in new/modified code** — proper type annotations
- **No removed features** — every feature stays; broken features get fixed or redesigned
- **No documented limitations** — limitations are fixed, not documented
- **No hardcoded hardware-specific values** — encoded as expert rules, not magic numbers
- **No isolated implementations** — every component must wire into the full pipeline
- **No silent failures** — every error path must be explicit
- **No fake/mock validation** — all validation must use real verification
- **No `# noqa` crutches** — fix the violation properly

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed.

### Test Decision
- **Infrastructure exists**: YES (pytest, hypothesis, coverage)
- **Automated tests**: TDD for all new code
- **Framework**: pytest with hypothesis for property-based testing

### QA Policy
Every task includes agent-executed QA scenarios:
- Python tests: `pytest src/tests/` with coverage
- Integration tests: `pytest src/tests/integration/` with mocked hardware
- Performance tests: `nautilus bench run` and `nautilus perf` CLI
- Linting: `ruff check`, `ruff format`, `mypy`
- Hardware CI: nightly benchmarks on AMD/Intel/Nvidia cloud

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation Hardening — 7 tasks, start immediately):
├── Task 1: Fix all 17 pre-existing test failures [deep]
├── Task 2: Add SectionFormat.METALLIB/AIR + fix conftest conflict [unspecified-high]
├── Task 3: Eliminate all 235 ruff violations (no noqa crutches) [unspecified-high]
├── Task 4: Eliminate all 152 mypy errors (proper type annotations) [unspecified-high]
├── Task 5: Fix the runtime_stub.c include path for cross-directory builds [quick]
├── Task 6: Fix the 495-ruff-error pre-existing codebase debt [unspecified-high]
└── Task 7: Add `nautilus inspect` deep system verification command [quick]

Wave 2 (Expert-Guided Auto-Tuning — 7 tasks, depends on Wave 1):
├── Task 8: Expert rules module — H100/MI300X/Gaudi/Apple [gpu-architect]
├── Task 9: Adaptive search strategy per kernel×vendor [compiler-architect]
├── Task 10: Hand-optimized matmul Triton templates (all 4 vendors) [gpu-architect]
├── Task 11: Hand-optimized attention Triton templates (all 4 vendors) [gpu-architect]
├── Task 12: Hand-optimized layer_norm/softmax/rms_norm templates [gpu-architect]
├── Task 13: Search result caching with kernel signature keying [unspecified-high]
└── Task 14: Expert-guided tuning integration test [integration-tester]

Wave 3 (Kernel Fusion Engine — 6 tasks, depends on Wave 2):
├── Task 15: Cross-op fusion (matmul+activation patterns) [compiler-architect]
├── Task 16: Cross-device fusion (output+communication patterns) [distributed-ai-expert]
├── Task 17: Fusion opportunity analyzer (graph pattern matching) [compiler-architect]
├── Task 18: Fusion code generator (produces fused Triton kernels) [compiler-architect]
├── Task 19: Communication planner for cross-device fusion [distributed-ai-expert]
└── Task 20: Kernel fusion integration test [integration-tester]

Wave 4 (Performance Database + Transfer Learning — 6 tasks):
├── Task 21: Performance database schema and persistence [unspecified-high]
├── Task 22: Benchmark result ingestion pipeline [performance-analyst]
├── Task 23: Transfer learning engine (vendor-to-vendor config mapping) [performance-analyst]
├── Task 24: Community database sync protocol [quick]
├── Task 25: Hardware CI matrix (weekly benchmarks on 3 vendors) [unspecified-high]
└── Task 26: Performance regression detection [performance-analyst]

Wave 5 (C++ MLIR Vector Dialect Normalizer — 5 tasks, depends on Wave 1):
├── Task 27: C++ project structure with pybind11 bindings [compiler-architect]
├── Task 28: TTGIR → MLIR Vector Dialect conversion in C++ [compiler-architect]
├── Task 29: MLIR Vector → TVMScript emission in C++ [compiler-architect]
├── Task 30: Python wrapper for C++ normalizer with feature detection [unspecified-high]
└── Task 31: MLIR normalizer integration test [integration-tester]

Wave 6 (Performance Dashboard + E2E Wiring — 6 tasks, depends on all prior):
├── Task 32: `nautilus perf report` — benchmark trends across vendors [performance-analyst]
├── Task 33: `nautilus perf compare` — version-to-version comparison [performance-analyst]
├── Task 34: `nautilus perf optimize` — auto-tune underperformers [performance-analyst]
├── Task 35: End-to-end pipeline performance verification [integration-tester]
└── Task 36: God Mode systemic review of all changes [god-mode]

Wave FINAL (Verification — 4 parallel reviews):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality + linting review (unspecified-high)
├── Task F3: Real QA — actual hardware benchmarks (unspecified-high)
└── Task F4: Scope fidelity check (deep)
```

### Dependency Matrix
- **1-7**: — 8-32, 1
- **8-14**: 1-7 — 15-26, 2
- **15-20**: 1-7, 8-14 — 27-36, 3
- **21-26**: 1-7, 8-14 — 27-36, 4
- **27-31**: 1-7 — 32-36, 5
- **32-36**: 1-7, 8-31 — F1-F4, 6

---

## TODOs

### Wave 1: Foundation Hardening

- [ ] 1. **Fix all 17 pre-existing test failures**

  **What to do**:
  - **tests/triton_tvm/**: Fix `test_ir_to_tir.py` failures (4 tests) — Pass 1/2/3/4 issues, tt.dot split operand extraction
  - **tests/triton_tvm/**: Fix `test_metadata_extractor.py` failures — missing attributes in test fixtures
  - **tests/cuda_ingest/**: Fix `test_parse_matmul_kernel` — parser regex/dict issues
  - **tests/aot_packager/**: Fix SectionFormat.METALLIB/AIR missing (covered in Task 2)
  - **tests/aot_packager/**: Fix `test_integration.py` hang issues (mock unconfigured, import errors)
  - **tests/pytorch_xla/**: Fix `_TVMMetaScheduleSharding` import error
  - Each fix must be a real code fix, not a test skip, not a `# xfail` marker

  **Must NOT do**:
  - Do not skip tests with `pytest.mark.skip` or `@pytest.mark.xfail`
  - Do not loosen test assertions to make them pass
  - Do not mock away the test to avoid the issue

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: `triton-compiler`, `gpu-architect`

  **Parallelization**: YES, Wave 1 (with Tasks 2-7)
  **Blocks**: Tasks 8-32 depend on green test suite
  **Blocked By**: None

  **References**:
  - `src/bridges/triton_tvm/ir_to_tir/pass1_lower_tensor_idioms.py`
  - `src/bridges/triton_tvm/ir_to_tir/pass2_rewrite_spmd.py`
  - `src/bridges/triton_tvm/ir_to_tir/pass3_replace_pointers.py`
  - `src/bridges/triton_tvm/ir_to_tir/pass4_materialize_tvm.py`
  - `src/bridges/triton_tvm/ir_to_tir/tt_dot_split.py`
  - `src/bridges/cuda_ingest/parser.py`

  **Acceptance Criteria**:
  - [ ] `pytest src/ -v` shows 100% pass rate (1,082/1,082)
  - [ ] No tests skipped without justified reason
  - [ ] All pre-existing failures fixed at root cause (not by weakening tests)

  **QA Scenarios**:
  ```
  Scenario: Run all tests
    Tool: Bash
    Steps: `pytest src/ -v --tb=short`
    Expected Result: All tests pass, 0 failures
    Evidence: .omo/evidence/task-1-full-test-suite.txt
  ```

  **Commit**: YES
  - Message: `fix(tests): resolve all 17 pre-existing test failures`
  - Files: `src/bridges/**/tests/test_*.py`, `src/bridges/**/ir_to_tir/*.py`

- [ ] 2. **Add SectionFormat.METALLIB/AIR + fix conftest conflict**

  **What to do**:
  - **fat_binary.py**: Add `SectionFormat.METALLIB = "metallib"` and `SectionFormat.AIR = "air"` to the enum
  - **aot_packager/builder.py**: Fix `link_fat_binary` signature to accept `apple_metallib` and `apple_air` parameters
  - **builder.py:243**: Fix dead branch — the `else` clause should use a different fallback (likely `MSL` or raise `CompilationError`)
  - **aot_packager/tests/conftest.py**: Remove `pytest_plugins` declaration (deprecated in non-root locations) — move fixtures to root conftest.py
  - **pytorch_xla/tests/conftest.py**: Remove the `--evidence-dir` CLI option redefinition (conflicts with root conftest.py)
  - **c_api/__init__.py:336 release()**: Add null pointer checks and proper error handling to prevent segfault
  - **builder.py:267**: Fix unknown kwarg `apple_metallib` to `link_fat_binary`

  **Must NOT do**:
  - Do not remove the Apple backend features
  - Do not silently swallow errors in release() (causes segfault)
  - Do not use `# type: ignore` to mask the signature mismatch

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `fat-binary-packaging`, `gpu-architect`

  **Parallelization**: YES, Wave 1
  **Blocks**: None (but other tasks may hit these issues)
  **Blocked By**: None

  **References**:
  - `src/bridges/aot_packager/fat_binary.py` — `SectionFormat` enum
  - `src/bridges/aot_packager/builder.py` — `link_fat_binary` signature
  - `src/bridges/aot_packager/tests/conftest.py` — pytest_plugins declaration
  - `src/bridges/pytorch_xla/tests/conftest.py` — evidence_dir redefinition
  - `src/c_api/__init__.py` — release() method

  **Acceptance Criteria**:
  - [ ] `SectionFormat.METALLIB` and `SectionFormat.AIR` exist in enum
  - [ ] `link_fat_binary` accepts apple_metallib and apple_air kwargs
  - [ ] `pytest --co` collects all test files without conflict errors
  - [ ] `c_api` release() doesn't segfault on double-release
  - [ ] Apple backend tests pass

  **QA Scenarios**:
  ```
  Scenario: SectionFormat includes all formats
    Tool: Bash
    Steps: `python -c "from src.bridges.aot_packager.fat_binary import SectionFormat; print(list(SectionFormat))"`
    Expected Result: All formats including METALLIB, AIR
    Evidence: .omo/evidence/task-2-sectionformat.txt

  Scenario: Pytest collection without conflicts
    Tool: Bash
    Steps: `pytest --co -q 2>&1 | head -20`
    Expected Result: No collection errors
    Evidence: .omo/evidence/task-2-pytest-collect.txt
  ```

  **Commit**: YES
  - Message: `fix(aot_packager): add SectionFormat.METALLIB/AIR, resolve conftest conflicts`
  - Files: `src/bridges/aot_packager/fat_binary.py`, `builder.py`, `tests/conftest.py`, `src/bridges/pytorch_xla/tests/conftest.py`, `src/c_api/__init__.py`

- [ ] 3. **Eliminate all 235 ruff violations (no noqa crutches)**

  **What to do**:
  - Run `ruff check src/ --fix` to auto-fix what can be auto-fixed
  - For remaining violations, fix each one properly:
    - `F821` (undefined names): Add proper imports or refactor
    - `F401` (unused imports): Remove the import (don't `# noqa: F401`)
    - `E402` (import position): Move imports to top of file
    - `E741` (ambiguous variable names): Rename variables
    - `F541` (f-string without placeholders): Add placeholders or convert to regular strings
    - `B904` (raise without from): Add `from err` to raise statements
    - `B905` (zip without strict): Add `strict=` parameter
    - `SIM105` (try/except/pass): Use `contextlib.suppress()`
    - `N812` (lowercase import as non-lowercase): Rename aliases consistently
    - `N806` (variable should be lowercase): Rename variables
    - `UP007` (Union syntax): Use `X | Y` instead of `Union[X, Y]`
  - Run `ruff format src/` to fix all formatting issues
  - `ruff check src/ --quiet` must exit 0
  - No `# noqa` comments may be added to suppress violations

  **Must NOT do**:
  - Do not add `# noqa: XXXX` comments to suppress violations
  - Do not use `noqa` as a crutch — fix the actual issue
  - Do not skip linting in CI

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: None (mechanical work)

  **Parallelization**: YES, Wave 1
  **Blocks**: Task 4 (mypy), Task 5 (CI gate)
  **Blocked By**: None

  **References**:
  - `pyproject.toml` — ruff config (lines 173-184)
  - All Python files in `src/`

  **Acceptance Criteria**:
  - [ ] `ruff check src/ --quiet` exits 0
  - [ ] `ruff format --check src/` exits 0
  - [ ] Zero `# noqa` comments in new code
  - [ ] All violations fixed at root cause

  **QA Scenarios**:
  ```
  Scenario: Zero ruff violations
    Tool: Bash
    Steps: `ruff check src/ --quiet; echo "Exit: $?"`
    Expected Result: Exit 0, zero violations
    Evidence: .omo/evidence/task-3-ruff-clean.txt

  Scenario: Zero noqa crutches
    Tool: Bash
    Steps: `grep -rn "noqa" src/ --include="*.py" | wc -l`
    Expected Result: Zero (or only in unavoidable cases with justification)
    Evidence: .omo/evidence/task-3-no-noqa.txt
  ```

  **Commit**: YES
  - Message: `style: eliminate all 235 ruff violations, zero noqa crutches`
  - Files: All `src/**/*.py`

- [ ] 4. **Eliminate all 152 mypy errors (proper type annotations)**

  **What to do**:
  - Remove all `unused-ignore` comments (mypy no longer needs them)
  - Fix `tir_template.py`: `T.Buffer((...))` should be `T.Buffer[...]` (8 instances)
  - Add `SectionFormat.METALLIB` and `SectionFormat.AIR` (covered in Task 2)
  - Fix dict comprehension key type mismatches in `metal_backend.py` and `nvidia_backend.py`
  - Fix `cli/commands/cluster.py:228-229` — list indexed by `str`
  - Fix `src/bridges/cuda_ingest/translator.py:57` — `CudaStatementType` redefined
  - Fix `builder.py:267` — unknown kwarg `apple_metallib` (covered in Task 2)
  - For SDK imports that have no type stubs (tvm, torch_xla, triton, aotriton):
    - Create `src/stubs/` directory with type stubs for these libraries
    - Add to mypy `--explicit-package-bases` config
  - `mypy src/ --ignore-missing-imports` must exit 0
  - No new `# type: ignore` may be added

  **Must NOT do**:
  - Do not add `# type: ignore` to silence errors
  - Do not use `cast()` to bypass type system
  - Do not weaken function signatures with `Any`

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: None (type system work)

  **Parallelization**: YES, Wave 1
  **Blocks**: Task 5 (CI gate)
  **Blocked By**: Task 3 (ruff)

  **References**:
  - `pyproject.toml` — mypy config (lines 159-192)
  - `src/bridges/triton_tvm/tir_template.py`
  - `src/bridges/aot_packager/builder.py`
  - `src/bridges/cuda_ingest/translator.py`

  **Acceptance Criteria**:
  - [ ] `mypy src/ --ignore-missing-imports` exits 0
  - [ ] Zero `# type: ignore` in new code
  - [ ] All SDK stubs created in `src/stubs/`
  - [ ] All type errors fixed at root cause

  **QA Scenarios**:
  ```
  Scenario: Zero mypy errors
    Tool: Bash
    Steps: `mypy src/ --ignore-missing-imports 2>&1 | grep -c "error:"`
    Expected Result: Zero
    Evidence: .omo/evidence/task-4-mypy-clean.txt
  ```

  **Commit**: YES
  - Message: `types: eliminate all 152 mypy errors with proper type annotations`
  - Files: All Python files with type issues, `src/stubs/`

- [ ] 5. **Fix the runtime_stub.c include path for cross-directory builds**

  **What to do**:
  - **Already fixed**: `runtime_stub.c` now uses `#include "triton_c_api.h"` (relies on -I flag)
  - **Already fixed**: `builder.py` adds `-I/workspaces/.../src/c_api` flag
  - **Verify**: Run `nautilus build` with skip_amd/skip_intel/skip_nvidia/skip_apple and confirm stub compilation succeeds
  - **Improve**: Add a static method `FatBinaryBuilder.compile_runtime_stub(stub_source: str) -> bytes` that can be called independently
  - **Improve**: Add a `StaticStubPath` class constant for type safety
  - **Improve**: Cache the compiled runtime stub at module import time

  **Must NOT do**:
  - Do not change the include path back to a relative path
  - Do not remove the -I flag (it was the fix)
  - Do not add `# noqa` to suppress the build error

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `fat-binary-packaging`

  **Parallelization**: YES, Wave 1
  **Blocks**: Task 35 (E2E pipeline performance verification)
  **Blocked By**: None

  **References**:
  - `src/bridges/aot_packager/builder.py` — `_compile_runtime_stub` method
  - `src/bridges/aot_packager/runtime_stub.c` — source file
  - `src/c_api/triton_c_api.h` — header file

  **Acceptance Criteria**:
  - [ ] `nautilus build` with all skip flags succeeds (compiles runtime stub)
  - [ ] Compiled `runtime_stub.o` matches host architecture (`file` command)
  - [ ] No relative include paths in `runtime_stub.c`

  **QA Scenarios**:
  ```
  Scenario: Runtime stub compiles from temp dir
    Tool: Bash
    Steps: `nautilus build test.py --skip-amd --skip-intel --skip-nvidia --skip-apple -o /tmp/test.o`
    Expected Result: Success, valid ELF produced
    Evidence: .omo/evidence/task-5-stub-build.txt
  ```

  **Commit**: YES
  - Message: `fix(aot_packager): improve runtime stub compilation robustness`
  - Files: `src/bridges/aot_packager/builder.py`, `runtime_stub.c`

- [ ] 6. **Fix the 495-ruff-error pre-existing codebase debt**

  **What to do**:
  - This is an extension of Task 3 — after auto-fix, there are still ~235 violations
  - **Strategy**: Fix violations by category in priority order:
    1. **High-impact** (safety, correctness): B904, B905, B017, B008 — fix these first
    2. **Medium-impact** (clarity): E402, E741, N803, N806, N812, N815, N818 — fix consistently
    3. **Low-impact** (style): UP007, UP035, SIM105, SIM110, B904 — fix systematically
  - **No `# noqa`** — every violation gets a real fix
  - **For SDK stub issues** (e.g., `tvm`, `torch_xla` not having type stubs): Create `src/stubs/` directory
  - **For N-prefixed rules** (naming conventions): Update code to use lowercase consistently
  - **For SIM rules** (try/except/with): Refactor to use `contextlib.suppress()` and `with` blocks

  **Must NOT do**:
  - Do not add `# noqa` comments
  - Do not disable ruff rules globally
  - Do not rename private variables to silence naming rules (they're public API)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: None (mechanical refactoring)

  **Parallelization**: YES, Wave 1
  **Blocks**: Task 35 (E2E verification)
  **Blocked By**: None (can run in parallel with Task 3)

  **References**:
  - `pyproject.toml` — ruff config
  - All Python files in `src/`

  **Acceptance Criteria**:
  - [ ] `ruff check src/ --quiet` exits 0
  - [ ] Zero `# noqa` comments in new code
  - [ ] All violations fixed at root cause
  - [ ] Public API names follow Python conventions (lowercase, snake_case)

  **QA Scenarios**:
  ```
  Scenario: Zero ruff violations across entire codebase
    Tool: Bash
    Steps: `ruff check src/ --quiet; echo "Exit: $?"`
    Expected Result: Exit 0
    Evidence: .omo/evidence/task-6-ruff-zero.txt
  ```

  **Commit**: YES (same as Task 3)
  - Message: `style: eliminate all 495 ruff violations, zero noqa crutches`
  - Files: All `src/**/*.py`

- [ ] 7. **Add `nautilus inspect` deep system verification command**

  **What to do**:
  - Create `src/cli/commands/inspect.py` (if not exists) or expand existing:
    - `nautilus inspect topology` — device discovery with bandwidth probes
    - `nautilus inspect toolchain` — check for gcc, lld, TVM, XLA, etc.
    - `nautilus inspect backends` — test each vendor backend availability
    - `nautilus inspect compliance` — IEEE-754 math validation across vendors
    - `nautilus inspect pipeline` — run full pipeline dry-run with timing
  - Add output formats: text, JSON, YAML
  - Add `--detailed` flag for verbose output
  - This is the "doctor command" — the user runs it to verify their setup

  **Must NOT do**:
  - Do not silently skip checks that fail — report them clearly
  - Do not require all hardware to be present — graceful degradation
  - Do not use mocked/fake detection — all checks must be real

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: None (CLI work)

  **Parallelization**: YES, Wave 1
  **Blocks**: None (improves developer experience)
  **Blocked By**: None

  **References**:
  - `src/cli/commands/inspect.py` (existing, may need expansion)
  - `src/common/hardware.py` — topology discovery
  - `src/bridges/aot_packager/` — backend availability checks

  **Acceptance Criteria**:
  - [ ] `nautilus inspect topology` outputs valid device topology JSON
  - [ ] `nautilus inspect toolchain` lists available/fissing tools
  - [ ] `nautilus inspect backends` tests each vendor backend
  - [ ] All inspect subcommands work without hardware (graceful skip)

  **QA Scenarios**:
  ```
  Scenario: Inspect topology output
    Tool: Bash
    Steps: `nautilus inspect topology --json`
    Expected Result: Valid JSON with devices, bandwidth, interconnects
    Evidence: .omo/evidence/task-7-inspect.txt
  ```

  **Commit**: YES
  - Message: `feat(cli): expand nautilus inspect with deep verification commands`
  - Files: `src/cli/commands/inspect.py`

### Wave 2: Expert-Guided Auto-Tuning

- [ ] 8. **Expert rules module — H100/MI300X/Gaudi/Apple**

  **What to do**:
  - Create `src/bridges/triton_tvm/expert_rules.py` with structured expert knowledge
  - For each vendor, encode optimization rules as typed dataclasses:
    ```python
    @dataclass(frozen=True)
    class VendorRules:
        vendor: Vendor
        matmul: MatmulRules
        attention: AttentionRules
        elementwise: ElementwiseRules
        reduction: ReductionRules
        memory: MemoryRules
        occupancy: OccupancyRules
    ```
  - **H100 rules** (from `gpu-architect` skill):
    - Tensor core tile shapes: 64×64, 128×128, 256×64, etc.
    - Warp-specialized pipelining: 3-5 stages
    - TMA (Tensor Memory Accelerator) hints for H100
    - Thread block clusters: 1, 2, 4, 8 (Hopper+)
    - L2 cache residency hints
  - **MI300X rules**:
    - Wavefront size 64 (vs Nvidia 32)
    - Matrix core tile shapes: 16×16×16, 32×32×16
    - LDS bank conflict avoidance (pad with +1)
    - CDNA3 wavefront occupancy rules
    - Infinity Fabric topology hints
  - **Gaudi rules**:
    - SIMD width 8/16/32
    - SLM (Shared Local Memory) bank alignment
    - Thread grouping for matrix engine
    - HBM2e memory bandwidth optimization
  - **Apple Metal rules**:
    - Threadgroup memory size: 16KB-32KB
    - SIMD group width: 32 (Apple GPU)
    - Tile shapes for Apple GPU matrix operations
    - Unified memory architecture hints
  - Each rule is data, not code — MetaSchedule uses it to guide search
  - Integration: `metaschedule_adapter.py` reads rules from expert_rules.py
  - Each rule has a `confidence` score (0.0-1.0) based on `gpu-architect` knowledge

  **Must NOT do**:
  - Do not hardcode magic numbers — use named constants
  - Do not duplicate rules across vendors — each vendor has its own rules
  - Do not add fallback "generic" rules that override vendor-specific knowledge

  **Recommended Agent Profile**:
  - **Category**: `gpu-architect`
  - **Skills**: `gpu-architect` (deep hardware knowledge)

  **Parallelization**: YES, Wave 2 (with Tasks 9-14)
  **Blocks**: Task 14 (expert-guided tuning integration test)
  **Blocked By**: Wave 1 (clean codebase)

  **References**:
  - `gpu-architect` skill: H100/MI300X/Gaudi/Apple architecture knowledge
  - `src/bridges/triton_tvm/metaschedule_adapter.py` — where rules are consumed
  - `docs/PRD.md` — success metrics

  **Acceptance Criteria**:
  - [ ] All 4 vendors have structured rules
  - [ ] Rules are typed dataclasses (not dicts)
  - [ ] Each rule has a confidence score
  - [ ] MetaSchedule uses the rules to guide search
  - [ ] Auto-tuning converges 10x faster on test cases

  **QA Scenarios**:
  ```
  Scenario: Expert rules guide tuning
    Tool: Bash
    Steps: 
      `python -c "
      from src.bridges.triton_tvm.expert_rules import get_vendor_rules
      h100 = get_vendor_rules('nvidia-h100')
      print(h100.matmul.tile_m)
      print(h100.matmul.confidence)
      "`
    Expected Result: [64, 128, 256], 0.95
    Evidence: .omo/evidence/task-8-rules.txt
  ```

  **Commit**: YES
  - Message: `feat(tuning): add expert-guided auto-tuning rules for all 4 vendors`
  - Files: `src/bridges/triton_tvm/expert_rules.py`, `metaschedule_adapter.py`

- [ ] 9. **Adaptive search strategy per kernel×vendor**

  **What to do**:
  - Create `src/bridges/triton_tvm/search_strategy.py` with adaptive strategies
  - Strategy selection based on kernel type × vendor:
    ```python
    def get_strategy(kernel_type: KernelType, vendor: Vendor) -> SearchStrategy:
        if kernel_type == KernelType.MATMUL and vendor == Vendor.NVIDIA:
            # H100: focus on tensor core tile shapes, warp specialization
            return EvolutionaryStrategy(
                population_size=50,
                mutation_rate=0.3,
                crossover_rate=0.5,
                elite_ratio=0.2,
                max_trials=5000,
                early_stop_generations=10,
            )
        elif kernel_type == KernelType.ATTENTION and vendor == Vendor.AMD:
            # MI300X: focus on memory bandwidth, wavefront occupancy
            return EvolutionaryStrategy(
                population_size=30,
                mutation_rate=0.4,
                max_trials=3000,
                memory_bound_heuristic=True,
            )
        # ... other combinations
    ```
  - Strategies are pure data, not control flow
  - Integration: `metaschedule_adapter.py` uses `search_strategy.get_strategy()` to select approach
  - **Adaptive learning**: Track which strategies work best for which kernel×vendor combos
  - Strategy performance is stored in performance DB (Wave 4)

  **Must NOT do**:
  - Do not use the same strategy for all kernels (defeats the purpose)
  - Do not hardcode strategy parameters — make them configurable
  - Do not remove existing strategies — only add new ones

  **Recommended Agent Profile**:
  - **Category**: `compiler-architect`
  - **Skills**: `compiler-architect`, `tvm-meta-schedule`

  **Parallelization**: YES, Wave 2
  **Blocks**: Task 14
  **Blocked By**: Task 8 (expert rules)

  **References**:
  - `src/bridges/triton_tvm/metaschedule_adapter.py`
  - TVM MetaSchedule API
  - `docs/TECH_SPEC.md`

  **Acceptance Criteria**:
  - [ ] `get_strategy()` returns different strategies for different kernel×vendor combos
  - [ ] Strategies are tuned for specific architectures
  - [ ] Strategy selection is data-driven (not random)
  - [ ] Performance regression: strategies converge faster than default

  **QA Scenarios**:
  ```
  Scenario: Different strategies for different combos
    Tool: Bash
    Steps:
      `python -c "
      from src.bridges.triton_tvm.search_strategy import get_strategy
      from src.common.primitives import Vendor
      from src.bridges.triton_tvm.ir_classifier import KernelKind
      s1 = get_strategy(KernelKind.MATMUL, Vendor.NVIDIA)
      s2 = get_strategy(KernelKind.ATTENTION, Vendor.AMD)
      assert s1.population_size != s2.population_size
      print('OK')
      "`
    Expected Result: OK
    Evidence: .omo/evidence/task-9-strategies.txt
  ```

  **Commit**: YES
  - Message: `feat(tuning): add adaptive search strategies per kernel×vendor`
  - Files: `src/bridges/triton_tvm/search_strategy.py`

- [ ] 10. **Hand-optimized matmul Triton templates (all 4 vendors)**

  **What to do**:
  - Create `src/bridges/triton_tvm/kernel_templates/matmul.py`
  - For each vendor, a hand-optimized matmul template:
    - **H100 matmul**: tensor core warp specialization, TMA hints, 3-5 stage pipelining
    - **MI300X matmul**: wavefront-aware tile shapes, mfma instructions, LDS bank padding
    - **Gaudi matmul**: SIMD width selection, SLM bank alignment, matrix engine grouping
    - **Apple matmul**: threadgroup memory, SIMD group width 32, unified memory
  - Each template is a `@triton.jit` function with compile-time tile shape parameters
  - Templates are **fallback defaults** — auto-tuning improves on them
  - **No hardware-specific libraries** — pure Triton only (works on all hardware)
  - **Documented tuning knobs** — each template has a comment block explaining parameters
  - Integration: `bridge_orchestrator.py` uses templates when no cached config exists

  **Must NOT do**:
  - Do not use cuBLAS or any vendor-specific library (Triton only)
  - Do not hardcode the best config — let auto-tuning improve it
  - Do not create templates that only work on one hardware

  **Recommended Agent Profile**:
  - **Category**: `gpu-architect`
  - **Skills**: `gpu-architect`, `triton-compiler`

  **Parallelization**: YES, Wave 2 (with Tasks 11-14)
  **Blocks**: Task 14
  **Blocked By**: Task 8 (expert rules)

  **References**:
  - `gpu-architect` skill: H100/MI300X/Gaudi/Apple optimization patterns
  - Triton documentation: `@triton.jit` decorator
  - `src/bridges/triton_tvm/extern_bridge.py` — existing matmul extern

  **Acceptance Criteria**:
  - [ ] 4 matmul templates (H100, MI300X, Gaudi, Apple)
  - [ ] Each template compiles and runs on target hardware
  - [ ] Templates achieve ≥90% of cuBLAS on Nvidia (verified by benchmark)
  - [ ] Templates are pure Triton (no vendor-specific libraries)

  **QA Scenarios**:
  ```
  Scenario: Matmul template produces correct result
    Tool: Bash
    Steps:
      `python -c "
      from src.bridges.triton_tvm.kernel_templates.matmul import matmul_h100
      import triton
      import torch
      a = torch.randn(128, 128, device='cuda', dtype=torch.float16)
      b = torch.randn(128, 128, device='cuda', dtype=torch.float16)
      # Verify template compiles
      print('Matmul template importable and compilable')
      "`
    Expected Result: Success
    Evidence: .omo/evidence/task-10-matmul-templates.txt
  ```

  **Commit**: YES
  - Message: `feat(templates): add hand-optimized matmul templates for all 4 vendors`
  - Files: `src/bridges/triton_tvm/kernel_templates/matmul.py`, `kernel_templates/__init__.py`

- [ ] 11. **Hand-optimized attention Triton templates (all 4 vendors)**

  **What to do**:
  - Create `src/bridges/triton_tvm/kernel_templates/attention.py`
  - For each vendor, a hand-optimized flash attention template:
    - **H100**: async copy (cp.async), warp specialization, FlashAttention-2 algorithm
    - **MI300X**: wavefront-aware tile shapes, mfma for QK^T, online softmax
    - **Gaudi**: SIMD width 16, SLM for K/V tiles, matrix engine for QK^T
    - **Apple**: threadgroup memory for K/V, SIMD group width 32
  - Each template is a `@triton.jit` function with:
    - Q, K, V tensor parameters
    - Block size parameters (BLOCK_M, BLOCK_N, BLOCK_K)
    - Causal mask parameter
    - Softcap parameter
  - Integration: `bridge_orchestrator.py` uses templates when no cached config exists

  **Must NOT do**:
  - Do not use FlashAttention C++ library (Triton only)
  - Do not create vendor-specific attention variants — pure Triton
  - Do not hardcode the best config

  **Recommended Agent Profile**:
  - **Category**: `gpu-architect`
  - **Skills**: `gpu-architect`, `triton-compiler`

  **Parallelization**: YES, Wave 2
  **Blocks**: Task 14
  **Blocked By**: Task 8 (expert rules)

  **References**:
  - FlashAttention-2 paper (Dao et al.)
  - Triton flash attention tutorial
  - `gpu-architect` skill: memory bandwidth optimization

  **Acceptance Criteria**:
  - [ ] 4 attention templates (one per vendor)
  - [ ] Templates implement FlashAttention-2 algorithm
  - [ ] Each template achieves ≥90% of FlashAttention on target hardware
  - [ ] Templates are pure Triton

  **QA Scenarios**:
  ```
  Scenario: Attention template compiles
    Tool: Bash
    Steps: `python -c "from src.bridges.triton_tvm.kernel_templates.attention import attention_h100; print('OK')"`
    Expected Result: OK
    Evidence: .omo/evidence/task-11-attention.txt
  ```

  **Commit**: YES
  - Message: `feat(templates): add hand-optimized attention templates for all 4 vendors`
  - Files: `src/bridges/triton_tvm/kernel_templates/attention.py`

- [ ] 12. **Hand-optimized layer_norm/softmax/rms_norm templates**

  **What to do**:
  - Create `src/bridges/triton_tvm/kernel_templates/normalization.py`
  - Templates for:
    - **LayerNorm**: mean, variance, normalize
    - **Softmax**: numerically stable exp + sum + divide
    - **RMSNorm**: root mean square normalization (Llama-style)
    - **GroupNorm**: grouped normalization
  - For each vendor, vendor-aware tile shapes and warp layouts
  - Each template is a `@triton.jit` function
  - Integration: `bridge_orchestrator.py` uses templates when no cached config exists

  **Must NOT do**:
  - Do not use vendor-specific libraries
  - Do not skip vendor-specific optimizations

  **Recommended Agent Profile**:
  - **Category**: `gpu-architect`
  - **Skills**: `gpu-architect`, `triton-compiler`

  **Parallelization**: YES, Wave 2
  **Blocks**: Task 14
  **Blocked By**: Task 8 (expert rules)

  **References**:
  - `gpu-architect` skill: warp/wavefront reduction patterns
  - Triton reduction tutorials
  - `src/bridges/triton_tvm/ir_classifier.py` — for kernel kind detection

  **Acceptance Criteria**:
  - [ ] 4 normalization templates (one per vendor)
  - [ ] Templates are numerically stable
  - [ ] Templates are pure Triton
  - [ ] Auto-tuning improves on template defaults

  **QA Scenarios**:
  ```
  Scenario: LayerNorm template compiles
    Tool: Bash
    Steps: `python -c "from src.bridges.triton_tvm.kernel_templates.normalization import layer_norm_h100; print('OK')"`
    Expected Result: OK
    Evidence: .omo/evidence/task-12-normalization.txt
  ```

  **Commit**: YES
  - Message: `feat(templates): add hand-optimized normalization templates for all 4 vendors`
  - Files: `src/bridges/triton_tvm/kernel_templates/normalization.py`

- [ ] 13. **Search result caching with kernel signature keying**

  **What to do**:
  - Create `src/bridges/triton_tvm/config_cache.py`
  - Cache tuning results by kernel signature (hash of IR) × hardware (vendor+arch)
  - **Cache key**: `sha256(kernel_ir_hash + vendor + arch + key_params)`
  - **Cache value**: `TuningConfig` (JSON-serializable)
  - **Cache invalidation**: when kernel source changes, IR hash changes, cache miss
  - **Cache storage**: local file in `~/.cache/nautilus/tuning/`
  - **Cache sharing**: optional remote sync via Performance DB (Wave 4)
  - Integration: `bridge_orchestrator.py` checks cache before invoking auto-tuning

  **Must NOT do**:
  - Do not use cache forever without invalidation — kernels can change
  - Do not store in-process only — restart should preserve cache
  - Do not store secrets in cache

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: None (caching infrastructure)

  **Parallelization**: YES, Wave 2
  **Blocks**: Task 14
  **Blocked By**: Task 8 (expert rules)

  **References**:
  - `src/bridges/triton_tvm/bridge_orchestrator.py` — where cache is consumed
  - `src/bridges/triton_tvm/config_mapper.py` — for serialization

  **Acceptance Criteria**:
  - [ ] Cache hit returns config without invoking auto-tuning
  - [ ] Cache miss invokes auto-tuning and stores result
  - [ ] Cache is persistent across process restarts
  - [ ] Cache invalidation works when kernel source changes

  **QA Scenarios**:
  ```
  Scenario: Cache hit on second run
    Tool: Bash
    Steps:
      1. Run tuning for a kernel (creates cache entry)
      2. Run again (should hit cache)
      3. Verify no auto-tuning was invoked
    Expected Result: Second run is instant
    Evidence: .omo/evidence/task-13-cache.txt
  ```

  **Commit**: YES
  - Message: `feat(tuning): add persistent config cache for auto-tuning results`
  - Files: `src/bridges/triton_tvm/config_cache.py`

- [ ] 14. **Expert-guided tuning integration test**

  **What to do**:
  - Create `src/bridges/triton_tvm/tests/test_expert_guided_tuning.py`
  - Tests:
    - Expert rules are loaded for all 4 vendors
    - Each rule has confidence score > 0.0
    - MetaSchedule search uses rules to guide exploration
    - Tuning converges faster with rules than without
    - Caching works correctly
  - End-to-end: `nautilus tune my_kernel.py --target nvidia-h100` produces optimal config
  - Use real Triton kernel fixtures from existing tests
  - Mock TVM where not available (so tests work in CI without TVM)

  **Must NOT do**:
  - Do not require real hardware
  - Do not mock auto-tuning (must be real)

  **Recommended Agent Profile**:
  - **Category**: `integration-tester`
  - **Skills**: `triton-compiler`, `tvm-meta-schedule`

  **Parallelization**: YES, Wave 2
  **Blocks**: Wave 3-6
  **Blocked By**: Tasks 8-13

  **References**:
  - `src/bridges/triton_tvm/tests/conftest.py` — existing fixtures
  - `src/bridges/triton_tvm/tests/test_integration.py` — existing integration tests

  **Acceptance Criteria**:
  - [ ] All tests pass
  - [ ] Expert rules are exercised
  - [ ] Search strategies are validated
  - [ ] Caching is validated
  - [ ] Templates are compiled

  **QA Scenarios**:
  ```
  Scenario: Expert-guided tuning produces valid config
    Tool: pytest
    Steps: `pytest src/bridges/triton_tvm/tests/test_expert_guided_tuning.py -v --tb=long`
    Expected Result: All pass
    Evidence: .omo/evidence/task-14-expert-tuning.txt
  ```

  **Commit**: YES
  - Message: `test(tuning): add expert-guided auto-tuning integration tests`
  - Files: `src/bridges/triton_tvm/tests/test_expert_guided_tuning.py`

---

## Final Verification Wave (MANDATORY)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have" from the plan: verify implementation exists (read file, run CLI, check test output). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in `.omo/evidence/`. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality + Linting Review** — `unspecified-high` (with `quality-assurance` skill)
  Run `ruff check src/ --quiet` + `mypy src/ --ignore-missing-imports` + `pytest src/ -v --tb=long`. Review all changed files for: `Any`, `# type: ignore`, empty catches, hardcoded values, commented-out code, unused imports. Check AI slop: excessive comments, over-abstraction, generic names.
  Output: `Build [PASS/FAIL] | Lint [PASS/FAIL] | Tests [N pass/N fail] | VERDICT`

- [ ] F3. **Real QA — Actual Hardware Benchmarks** — `unspecified-high` (with `performance-analyst` skill)
  Execute the full pipeline end-to-end and verify it works. Run `nautilus pipeline model.py --dry-run`. Run `nautilus bench run --quick`. Run `nautilus perf report`. Verify expert rules produce results. Verify config cache works. Save all evidence to `.omo/evidence/final-qa/`.
  Output: `Pipeline [PASS/FAIL] | Benchmarks [PASS/FAIL] | Expert Rules [PASS/FAIL] | Cache [PASS/FAIL] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual git diff. Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no scope creep). Check "Must NOT do" compliance — ensure no forbidden patterns were introduced. Detect cross-task contamination: Task N touching Task M's files without justification. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy
- **Task grouping**: every 2-3 related tasks per commit
- **Message format**: `type(scope): desc`
- **Pre-commit gate**: `ruff check` + `mypy` + `pytest` for changed module

### Wave Commit Plan
| Wave | Commits | Estimated Count |
|---|---|---|
| Wave 1 (Foundation) | `fix:`, `refactor:`, `chore:`, `ci:` | 3-4 commits |
| Wave 2 (Expert Tuning) | `feat(triton_tvm):`, `perf:`, `test(triton_tvm):` | 5-7 commits |
| Wave 3 (Kernel Fusion) | `feat(compiler):`, `feat(distributed):`, `test:` | 5-7 commits |
| Wave 4 (Performance DB) | `feat(perf):`, `feat(bench):`, `feat(ci):` | 5-7 commits |
| Wave 5 (MLIR C++) | `feat(mlir):`, `feat(lib):`, `feat(compiler):` | 5-7 commits |
| Wave 6 (E2E + Verify) | `feat(cli):`, `feat(perf):`, `test:`, `review:` | 4-5 commits |
| **Total** | | **~30-40 commits** |

---

## Success Criteria
- [ ] All 17 pre-existing test failures → 0
- [ ] All 235 ruff violations → 0 (no `# noqa` crutches)
- [ ] All 152 mypy errors → 0 (proper type annotations)
- [ ] Expert rules encoded for H100/MI300X/Gaudi/Apple
- [ ] Hand-optimized matmul + attention + layer_norm templates
- [ ] Kernel fusion engine (cross-op + cross-device)
- [ ] Performance database with transfer learning
- [ ] MLIR Vector Dialect normalizer (C++)
- [ ] `nautilus perf` CLI dashboard
- [ ] End-to-end pipeline performance verified
- [ ] God Mode checklist: all 7 dimensions pass
- [ ] Zero documented limitations, zero removed features
- [ ] 100% test pass rate

## Commit Strategy
- Task grouping: every 2-3 related tasks per commit
- Message format: `type(scope): desc`
- Pre-commit gate: `ruff check` + `mypy` + `pytest` for changed module

### Wave Commit Plan
| Wave | Commits | Estimated Count |
|---|---|---|
| Wave 1 (Foundation) | `fix:`, `refactor:`, `chore:`, `ci:` | 3-4 commits |
| Wave 2 (Expert Tuning) | `feat(triton_tvm):`, `perf:`, `test(triton_tvm):` | 5-7 commits |
| Wave 3 (Kernel Fusion) | `feat(compiler):`, `feat(distributed):`, `test:` | 5-7 commits |
| Wave 4 (Performance DB) | `feat(perf):`, `feat(bench):`, `feat(ci):` | 5-7 commits |
| Wave 5 (MLIR C++) | `feat(mlir):`, `feat(lib):`, `feat(compiler):` | 5-7 commits |
| Wave 6 (E2E + Verify) | `feat(cli):`, `feat(perf):`, `test:`, `review:` | 4-5 commits |
| **Total** | | **~30-40 commits** |

---

## Success Criteria
- [ ] All 17 pre-existing test failures → 0
- [ ] All 235 ruff violations → 0 (no `# noqa` crutches)
- [ ] All 152 mypy errors → 0 (proper type annotations)
- [ ] Expert rules encoded for H100/MI300X/Gaudi/Apple
- [ ] Hand-optimized matmul + attention + layer_norm templates
- [ ] Kernel fusion engine (cross-op + cross-device)
- [ ] Performance database with transfer learning
- [ ] MLIR Vector Dialect normalizer (C++)
- [ ] `nautilus perf` CLI dashboard
- [ ] End-to-end pipeline performance verified
- [ ] God Mode checklist: all 7 dimensions pass
- [ ] Zero documented limitations, zero removed features
- [ ] 100% test pass rate
