# Hyperplan Bundle: NVINDIA_CUD (Nautilus) Codebase Audit

**Lead:** Sisyphus (orchestrator)
**Date:** 2026-06-05
**Subject:** Adversarial review of `/workspaces/NVINDIA_CUD` against the goal in `docs/PRD.md` and `docs/TECH_SPEC.md`

> Team-mode was DEGRADED — 3 of 4 adversarial members errored on launch. Lead (this orchestrator) performed all three rounds of analysis directly using deep reads of `docs/`, `src/`, and runtime verification commands.

---

## Goal Restatement

Per `docs/PRD.md` and the user's user-request: Build **Nautilus** — an independent, cross-vendor AI compilation framework. A unified pipeline that takes **standard PyTorch model code**, optimizes it via **an advanced AI cost model** (TVM MetaSchedule), bundles it into a **single "Fat Binary"** (per-vendor ELF sections + C runtime ), and **automatically splits** (GSPMD auto-sharding) to run at maximum speed across a **mixed cluster of AMD and Intel GPUs**. The user demands **REAL WORLD READY** — not MVP, not fake showcasing, not subset implementation.

---

## Round 1 Findings (Severity Ranked)

### CRITICAL — these mean the product is non-functional today

**C-1. Zero runtime dependencies installed in the dev env.**
Verified: `python3 -c "import triton"` → `ModuleNotFoundError`. Same for `tvm`, `aotriton`, `torch`, `torch_xla`. `which lld` returns nothing. Every "fallback" path IS the de facto production path. (`pyproject.toml` declares `triton>=3.0`, `apache-tvm>=0.18`, `torch>=2.0` — but no `pip install` has been run or recorded in any lockfile. There is no `requirements.txt`, no `poetry.lock`, no `Pipfile`.)

**C-2. `src/common/` is empty.** `docs/PATTERNS.md` requires shared `types.py`, `hardware.py`, `logging.py`, `errors.py`. None exist. Bridges pass untyped `Any` around. There is no `Result[T, E]` type even though the style guide mandates it.

**C-3. `src/c_api/` does not exist.** `docs/TECH_SPEC.md §5.1` defines `triton_c_api.h`, `tvm_c_api.h`, `xla_c_api.h` as the "Core Gate" against version drift. Without these, every bridge calls Python APIs of upstream libraries directly — the wiring IS the version-drift liability the spec was designed to prevent.

**C-4. `src/cli/` does not exist.** `pyproject.toml` declares entry points `nautilus-tune`, `nautilus-build`, `nautilus-shard` pointing at `src.cli.commands.{tune,build,shard}` — that package is missing. A user who runs `pip install -e .` gets no CLI. The "developer onboarding time < 1 hour" success metric (`PRD §7`) is unreachable.

**C-5. The Nvidia backend does NOT compile kernels — it generates a 6-line PTX placeholder.**
File: `src/bridges/aot_packager/nvidia_backend.py:280-292`. The `_run_triton_aot()` method imports `triton.compiler.compile`, then IGNORES it and calls `_generate_minimal_ptx()` which writes:
```
.visible .entry placeholder_kernel() { mov %r0, 0; ret; }
```
The comment at line 287 admits: "For now, we just generate a placeholder PTX so the fat binary build can complete." This is a hardcoded stub kernel that does nothing. Any "fat binary" the linker produces contains a kernel that writes zero to a register and returns.

**C-6. The Intel backend does NOT compile kernels — it returns a path without running llvm-spirv.**
File: `src/bridges/aot_packager/intel_backend.py:281-283`. The `_compile_triton_to_llvm()` writes a `tmp_*.ll` path and returns it with the comment `# Placeholder for actual compilation`. The downstream `llvm-spirv` call is then run on an EMPTY file. The `_write_placeholder()` writes a 20-byte SPIR-V header magic number as the "compilation result." This is not a real SPIR-V module — it has zero opcodes, zero capabilities, zero bindings.

**C-7. The AMD backend falls back to a 64-byte placeholder ELF.**
File: `src/bridges/aot_packager/amd_backend.py:359-379`. When AOTriton is unavailable (which it always is in the test env), `_write_placeholder` writes a minimal `b"\x7fELF..."` 64-byte stub. The comment at line 277 says: "The placeholder is a valid (but non-functional) ELF section that links cleanly. It will fail at runtime if loaded on actual AMD hardware." So the AMD code path is documented to produce non-functional binaries.

**C-8. The C runtime stub is also a stub.**
File: `src/bridges/aot_packager/runtime_stub.c:92-110`. The GPU detection functions are explicit no-ops: `nautilus_has_nvidia_gpu() { return 0; }`, same for AMD and Intel. The comment says "In a real implementation, this would: 1. Check /dev/nvidia0... For the stub, we do a simple file existence check." But the file existence check is `return 0` — it doesn't check anything. Any program that calls `nautilus_dispatch()` will always go to `nautilus_kernel_default`, which is also an `extern` undefined symbol — a linker error if you actually try to link the stub. (The embedded stub inside `builder.py:390-401` is a 4-line pure-default that doesn't even attempt vendor detection.)

**C-9. The StableHLO export is explicitly NotImplemented.**
File: `src/bridges/pytorch_xla/stablehlo_export.py:124-128`. The torch_xla path raises `NotImplementedError("torch_xla path requires example inputs")`. The ONNX path falls through to `_export_fallback()` which generates a "minimal but valid MLIR-like representation" — but the file itself admits: "This is NOT a real StableHLO module — it's a fallback representation." (`stablehlo_export.py:220`).

**C-10. GSPMD is a hand-written heuristic, not Google's actual algorithm.**
File: `src/bridges/pytorch_xla/gspmd_runner.py:195-226`. The `_run_gspmd_algorithm` method does not call into XLA/PJRT. It builds sharding specs via a `for input_spec in module.input_specs: if strategy == DATA_PARALLEL: ... else: AUTO` Python conditional, and generates a sharded StableHLO by `lines.append(module.mlir_text)` — i.e. it just appends comments to the original MLIR. No cost model. No comm volume calculation (`spec.estimated_comm_volume_bytes = 0  # Placeholder`, line 293). The class name `GSPMDRunner` is misleading; the function name is `run_gspmd_algorithm` but it does not implement GSPMD.

**C-11. `src/tests/` is missing — no top-level integration test, no full-pipeline test, no cluster test.**
`docs/TECH_SPEC.md §7` specifies `src/tests/test_auto_tuning.py`, `test_fat_binary.py`, `test_sharding.py`, `integration/test_full_pipeline.py`, `integration/test_cluster.py`. None exist. Tests live inside each bridge subdirectory but only test internal classes. There is NO test that wires two bridges together.

**C-12. The CUDA parser is regex-based and the translator is text-substitution.**
File: `src/bridges/cuda_ingest/parser.py:13-20` admits this explicitly: "The parser is intentionally regex/pattern-based rather than using a full C++ parser (like libclang)." This means the parser will fail on legitimate C++ like template parameters, function pointers, complex expressions. The translator (`translator.py`) calls `intrinsic_mapper.transform_text(stmt.raw_text)` for nearly every statement type (line 231, 235, 239, 243, 247, 251, 255, 259) — a regex-based text replacement. CUDA → Triton translation quality is bounded by string substitution.

**C-13. The shard executor generates hardcoded placeholder Triton source.**
File: `src/bridges/pytorch_xla/hardware_orchestrator.py:147-186`. The `_generate_shard_source()` method returns a string literal of a matmul kernel with a comment: "For now, generate a minimal placeholder that the fat binary builder can compile." The shard does NOT come from the sharded StableHLO — it is a copy-pasted matmul that ignores the actual GSPMD output.

**C-14. Pyproject declares Triton backend plugin entry point but the backend/__init__.py is a 7-line shim.**
File: `src/bridges/triton_tvm/backend/__init__.py` (838 bytes). The entry point `[project.entry-points."triton.backends"] tvm = "src.bridges.triton_tvm.backend:TVMBackend"` requires this module to expose `TVMBackend` and `TVMDriver` — but I have not verified it does. Even if it does, the C++ plugin in `lib/` is a CMake project that requires `TRITON_SRC_DIR` and `MLIR_DIR` env vars to build; if those aren't set, `setup.py` silently skips it (lines 30-37 of setup.py). So the actual native IR capture path is opt-in and undocumented.

### HIGH — gaps that prevent "real world ready"

**H-1. The math validator's ULP error is set equal to abs error.**
File: `src/runtime/math_validator.py:198`. Comment: `max_ulp = max_abs  # Simplified — true ULP computation is complex`. The whole point of IEEE-754 bit-exact mode is ULP-level analysis; substituting abs error defeats the purpose. The `_compute_errors` returns `(0.0, 0.0, 0.0)` on ImportError — so when numpy isn't installed, the validator falsely reports bit-exact success.

**H-2. The memory reclaimer's "production path" is `return 0`.**
File: `src/runtime/memory_reclaimer.py:178-186`. Calls `torch.cuda.empty_cache()` then `return 0` — the function lies about what it reclaimed. The bytes accounting (`state.last_reclaim_bytes = reclaimed`) records zero, so observability dashboards show no reclaim activity. The function signature says `Returns the number of bytes reclaimed` but never returns anything but 0.

**H-3. Cross-bridge coupling: pytorch_xla/pipeline_orchestrator.py imports circuit_breaker/timeout_manager/structured_logging from triton_tvm.**
File: `src/bridges/pytorch_xla/pipeline_orchestrator.py:39-52`. This is a hard dependency of Phase 3 on Phase 1 modules. If anyone restructures `triton_tvm`, the sharding bridge breaks. Without `src/common/` to host these shared utilities, there's no proper home.

**H-4. No drift-detection CI.**
`docs/TECH_SPEC.md §5.3` specifies `.github/workflows/drift-detection.yml` with daily upstream builds. The directory `.github/` does not exist. Version drift WILL break the wiring; nothing detects it.

**H-5. No benchmark suite.**
`docs/PRD §7` requires "Auto-tuning speedup vs. default Triton | ≥30% on non-Nvidia targets | Benchmark suite." No `benchmarks/` directory exists at the top level. The PRD claims 10 benchmark kernels in the Phase 1 deliverable — none present.

**H-6. No Apple Metal backend.**
`PRD §F-2` says "SHOULD support Apple Metal backend." `nvidia_backend.py` declares only NvidiaArch. Intel backend declares only IntelTarget. AMD backend declares only AMDArch. No Metal path. The codebase doesn't even stub it.

**H-7. The runtime stub embedded in builder.py is a 4-line default-dispatch.**
File: `src/bridges/aot_packager/builder.py:390-401`. The actual `runtime_stub.c` (120 lines) is overwritten with this embedded 4-line stub on every build. So the real `runtime_stub.c` is dead code in the shipped fat binary build.

### MEDIUM — quality and robustness issues

**M-1. `src/bridges/triton_tvm/backend/hooks.py` (4.4KB) and `options.py` (3.8KB) exist but I didn't see a consumer outside the backend `__init__.py`. Likely dead code in the absence of a real Triton integration test.**

**M-2. 28 of ~50 source files contain `NotImplementedError` or `return None/0/""` patterns.** Counted via grep. The codebase is structured like a forward-declared skeleton: types, configs, and fallback paths are everywhere; real backends are not.

**M-3. The IR→TIR conversion has 4 passes (`pass1_lower_tensor_idioms.py`, `pass2_rewrite_spmd.py`, `pass3_replace_pointers.py`, `pass4_materialize_tvm.py`) but the only test (16KB `test_ir_to_tir.py`) is the largest test file. This suggests the conversion is the most-tested part — but the upstream `IRCapture` that feeds it depends on the Triton C++ plugin which never builds by default.**

**M-4. `bridge_orchestrator.tune_configs_list` (line 207-236) hand-generates "variants" of the best config by halving/doubling block sizes.** This is not a meta-heuristic — it's three fixed multiplications. Not a real autotune exploration.

**M-5. The `_fallback` chain (line 578-592) returns `MappedTuningConfig(block_m=128, block_n=128, block_k=32, num_warps=...)` — Triton defaults. The whole point of the bridge is to BEAT Triton defaults. If the chain falls through, the bridge adds no value.**

**M-6. AsyncCheckpointer: not reviewed in depth but is 13KB. Re-read before claims. Initial read at lines 1-50 suggests it checkpoints to system RAM but no verification of HOW the checkpointing interacts with the rest of the runtime.**

**M-7. The `pyproject.toml` entry point `[project.entry-points."triton.backends"]` uses `src.bridges.triton_tvm.backend:TVMBackend` but the modules `compiler.py`, `driver.py` (in `backend/`) are large (12KB+). If the entry point is not loadable, the entire Triton integration is dead.**

---

## Round 2 (Cross-Attack Simulation)

If `skeptic-high` were here, they would point out: "The C-5 through C-8 findings are damning — the actual compilation pipeline writes placeholders. This is the *core* of Phase 2. A 'fat binary' with placeholder kernels is not a fat binary."

If `ultrabrain-analyst` were here, they would note: "The dependency check (C-1) changes everything. With no Triton, no TVM, no torch, no torch_xla, no lld, the user's environment is exactly the one where every fallback fires. The code is correct given its fallbacks — but its fallbacks are stubs."

If `artistry-rebel` were here, they would observe: "The user can't even `pip install -e .` and run anything. The README is one line: `# NVINDIA_CUD`. There is no `quickstart`, no demo notebook, no working example. 'Real world ready' is impossible when the entry point doesn't exist."

These all converge on the same conclusion.

## Round 3 (Refined Position)

**Verdict: This is high-quality SCAFFOLDING, not a working product.** ~17,135 lines of Python + ~3,700 lines of tests + ~7KB of C++ + a pyproject.toml. The architecture is sound. The types are mostly right. The production observability features (circuit breakers, timeouts, structured logging, per-stage caches) are real and well-engineered. But **every code path that crosses a bridge to an actual external tool** is a placeholder, a fallback, a stub, or a `NotImplementedError`. The product does not achieve the goal.

The user asked: "Is this real-world ready or fake showcasing?"

**Answer: It is closer to real-world ready than a typical MVP — but it is not a working product. It is a well-engineered prototype of the architecture, with stubs standing in for every "last mile" integration that would make the pipeline actually run.**

---

## Bundle for the Plan Agent

### 5 CRITICAL Fixes (must be done to ship v0)

1. **Install real dependencies and prove a real pipeline works end-to-end on a real GPU** (target: H100 or A100 first; AMD MI300X second; Intel Gaudi third). Without this, no other fix matters. Track via CI matrix in `.github/workflows/real-hardware.yml`.

2. **Replace the placeholder generators with real compilation calls** in `nvidia_backend.py` (use `triton.aot.compile`), `intel_backend.py` (use `triton → ll → llvm-spirv` with real `triton.compiler`), `amd_backend.py` (use AOTriton or a Triton-emitted cubin → SPIR-V path). Delete `_generate_minimal_ptx`, fix `_compile_triton_to_llvm` to actually compile, replace `_write_placeholder` with a real error.

3. **Implement real StableHLO export** in `stablehlo_export.py`. Use `torch.export` + `torch_xla.stablehlo.exported_program_to_stablehlo` (or build an FX→StableHLO converter in TVMScript). Delete the `_export_fallback` path or guard it behind an explicit `EXPORT_FALLBACK_OK=1` env var.

4. **Replace heuristic GSPMD with real sharding.** Either call into XLA's GSPMD via PJRT, or use a TVM-driven approach (TVM has its own auto-scheduler for sharding). Compute `estimated_comm_volume_bytes` for real (cost model: bytes = sum(tensor.size * num_devices_along_axis / total_bandwidth)).

5. **Create the missing infrastructure**:
   - `src/common/{types,hardware,logging,errors}.py` — shared types, hardware detection, structured logger, `Result[T, E]` and error hierarchy
   - `src/c_api/{triton,tvm,xla}_c_api.h` + a C++ wrapper for at least one of them
   - `src/cli/commands/{tune,build,shard}.py` — implement the 3 CLI commands referenced in `pyproject.toml`
   - `src/tests/integration/test_full_pipeline.py` — one test that compiles + tunes + packages + shards a model end-to-end

### Top 3 HIGH Fixes

- Fix math validator: real ULP computation, fail loudly on missing numpy/torch
- Fix memory reclaimer: return real bytes from `torch.cuda.memory_stats()`, support ROCm (`torch.cuda.empty_cache` won't work) and Level Zero
- Decouple pytorch_xla from triton_tvm internals: move `circuit_breaker`, `timeout_manager`, `structured_logging` to `src/common/observability/`

### Order of operations (for the plan agent to refine)

1. **Set up real environment**: pin Triton, TVM, XLA, torch, torch_xla, AOTriton in `pyproject.toml` (already declared but not pinned) and add a `setup-cuda.sh` / `setup-rocm.sh` to a `scripts/` dir. Prove `import triton; import tvm; import torch; import torch_xla` works in CI.

2. **Prove one end-to-end kernel** (matmul) on Nvidia: write a `nautilus-tune matmul.py` that takes a Triton matmul, runs the bridge, produces a fat binary, and the user can `load fatbinary.o` and execute it on a real GPU. This single success unlocks the architecture.

3. **Extend to AMD** (one kernel), then **Intel** (one kernel). Each gets real hardware validation in CI.

4. **Add PyTorch → sharded fat binary** for a real model (e.g., a 2-layer transformer).

5. **Add CUDA ingestion demo**: a sample `.cu` file → translated Triton → fat binary → executed on all 3 vendors.

### Out of scope for this bundle (not blocking the goal but worth noting)

- Apple Metal backend
- Bit-exact mode by default (it's an opt-in validator)
- Multi-node cluster (single-node per vendor is sufficient for "real world ready" first cut)
- Drift-detection CI
- Apple-to-Apple performance comparison
