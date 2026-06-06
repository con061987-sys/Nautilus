# Hyperplan Audit Bundle: Nautilus vs. Stated Goal

**Goal under audit (verbatim):** "We are building Nautilus, an independent, cross-vendor AI compilation framework. It is a unified software pipeline that takes standard PyTorch model code, optimizes it using an advanced AI cost model, bundles it into a single 'Fat Binary' file, and automatically splits it to run at maximum speed across a mixed cluster of AMD and Intel GPUs."

**Audit verdict (lead synthesis):** The codebase is **substantially real, not an MVP** for the four core bridges. Real `triton.compiler.compile`, real AOTriton, real `llvm-spirv`, real `lld`, real `/dev` probing + CPUID, real 5-stage PyTorch→StableHLO→GSPMD wiring, real MetaSchedule adapter, real 3-tier GSPMD with cost model. **But** the goal "takes standard PyTorch model code… runs at maximum speed across a mixed cluster of AMD and Intel" is **NOT achieved end-to-end** — three verified stubs break the pipeline, the integration tests skip on the critical GPU paths, and there is no end-to-end HuggingFace-model demo. The codebase is **80% real and 20% structural lie**.

---

## 1. INVENTORY (lead direct audit, file:line)

### 1.1 REALS (production-grade, verified)

| File | Verdict | Evidence |
|---|---|---|
| `src/bridges/aot_packager/nvidia_backend.py:213-330` | **REAL** | Invokes `triton.compiler.compile(src, target="cuda", options=...)` and extracts `asm["ptx"]`/`asm["cubin"]`. Validates PTX for `.version`/`.target`/`.entry` markers (line 332-359). Caches, raises on missing triton, raises on missing output. The "previous bug" of writing 6-line stub PTX is explicitly called out and replaced. |
| `src/bridges/aot_packager/amd_backend.py:192-279` | **REAL** | First tries `from aotriton import compile` Python API. Falls back to `aotriton` CLI via `subprocess.run`. Second fallback to Triton+amdclang++ (line 281-362) with real `--offload-arch=gfx942`. Validates output via `b"\x7fELF"` magic + `amdgcn`/`AMDGPU` string check (line 364-377). |
| `src/bridges/aot_packager/intel_backend.py:176-316` | **REAL** | Real `triton → LLVM IR → llvm-spirv` pipeline. Falls back `target="cuda"` if `triton.backends.xpu` missing. Validates with `spirv-val --target-env spv1.2` (line 318-330). |
| `src/bridges/aot_packager/linker.py:317-360` | **REAL** | Real `lld -r` invocation with per-vendor ELF wrappers. Cache-keyed linking. Raises `LinkingError` with stderr context on failure. |
| `src/bridges/aot_packager/runtime_stub.c:65-237` | **REAL** | Real `access(F_OK)` probing of `/dev/nvidia0`, `/dev/kfd`, `/dev/dri/renderD*`. Real x86_64 CPUID via inline asm checking `GenuineIntel` (0x756e6547) and `AuthenticAMD` (0x68747541). Real `nautilus_dispatch()` that aborts to `nautilus_kernel_default` if no vendor. Thread-safe. |
| `src/bridges/aot_packager/builder.py:158-299` | **REAL** | Orchestrates AMD+Intel+Nvidia in sequence with per-stage timings. `_compile_runtime_stub` (line 360-401) uses real `gcc -c -fPIC`. `_read_runtime_stub_source` (line 402-421) reads from package data, not hardcoded. |
| `src/bridges/triton_tvm/bridge_orchestrator.py:1-741` | **REAL** | 6-tier fallback (`FallbackTier` enum, line 50-57: L0 DB → L5 safe). Real `IRCapture` (line 109-126) wired to backend plugin. Real `MetaScheduleAdapter` invocation via circuit breaker. 4-pass TIR conversion pipeline referenced (ir_to_tir/). |
| `src/bridges/pytorch_xla/pipeline_orchestrator.py:140-253` | **REAL** | 5-stage pipeline: graph_capture → stablehlo_export → gspmd → dtensor_apply → fat_binary execute. Per-stage timing. Circuit breaker for GSPMD (line 255-284). |
| `src/bridges/pytorch_xla/stablehlo_export.py` (per `docs/rewrite-summary.md`) | **REAL** | 3 real tiers: `_TorchXLAExporter` (real `torch_xla.stablehlo.exported_program_to_stablehlo`), `_ONNXBridgeExporter` (real `torch.onnx.export` + `onnx-mlir --stablehlo`), `_TVMScriptExporter` (real `tvm.relax.frontend.torch.from_pytorch`). `is_real_stablehlo` regex check `\bstablehlo\.\w+`. |
| `src/bridges/pytorch_xla/gspmd_runner.py` (per rewrite-summary) | **REAL** | 3 real tiers: `_TorchXLASharding` (real `torch_xla.experimental.sharding_impl.shard_module`), `_XlaClientSharding` (real `OpSharding` protos), `_TVMMetaScheduleSharding` (real `ms.tune_tir`). Real `_CommCostModel` formulas (line 49-58). |
| `src/runtime/memory_reclaimer.py:126-282` | **REAL** | Vendor-specific reclaim with real byte accounting via `torch.cuda.memory_stats()` deltas. ROCm piggybacks on CUDA API. Intel uses `torch.xpu.memory_stats()`. Apple raises on non-Darwin. Replaces previous `return 0` stub (called out in docstring). |
| `src/runtime/async_checkpointer.py` (per CHANGELOG) | **REAL** | Atomic write + SHA-256 checksum. Circuit breaker. |
| `src/runtime/math_validator.py` (per CHANGELOG) | **REAL** | Real IEEE-754 ULP error computation. |
| `src/common/observability.py` | **REAL** | `CircuitBreaker` + `TimeoutManager` with per-stage `StageBudgets`. |
| `src/common/hardware.py` | **REAL** | Real `/dev/*` + `lspci` + `system_profiler` probing. Raises `HardwareNotFoundError` instead of returning empty. |
| `src/common/types.py` (per CHANGELOG) | **REAL** | Vendor-neutral `Vendor`, `Arch`, `FatBinary`, `TuningConfig`, `MeshShape`, `ShardingSpecLite`, `StableHLOModule`, `IRModule`. |
| `src/common/errors.py` | **REAL** | `NautilusError` hierarchy with stable string codes (e.g. `E_COMPILATION_FAILED`). |

### 1.2 FAKES / STUBS / FANTASIES (verified file:line)

| File:Line | Verdict | Evidence |
|---|---|---|
| `src/cli/commands/shard.py:264-309` `_generate_shard_source()` | **STUB** | Explicit comment in source: `"This is a stub of the eventual StableHLO→Triton translator."` Emits a hand-written matmul template regardless of `stablehlo.mlir_text` or `spec`. The per-shard `kernel.py` is therefore NOT generated from the captured model — it's a hardcoded `tl.dot` template. **This breaks the goal**: a HuggingFace model sharded via this CLI does NOT produce a tuned, model-specific Triton kernel. |
| `src/bridges/aot_packager/linker.py:362-407` `_write_minimal_fat_binary()` | **DEPRECATED STUB** | Manual concatenation of section files with a JSON manifest. Kept "for back-compat." CHANGELOG claims it's "no longer called" — verify in builder.py. If anyone bypasses the lld path, this returns a non-ELF blob. |
| `src/bridges/aot_packager/builder.py:423-449` `_minimal_elf_stub()` | **DEPRECATED STUB** | 64-byte ELF header with all-zero fields. `DeprecationWarning` issued but the method still exists. |
| `src/bridges/pytorch_xla/hardware_orchestrator.py` `ShardExecutor` | **UNVERIFIED** | The pipeline orchestrator at line 230-234 calls `self.executor.execute_all_shards(gspmd_result, stablehlo)`. Without reading `hardware_orchestrator.py`, cannot confirm this actually dispatches the per-shard fat binary to real hardware. **This is the link that determines whether "runs at maximum speed across a mixed cluster" is real.** |
| `src/bridges/triton_tvm/bridge_orchestrator.py:282-284` | **SILENT FALLBACK** | When `IRCapture` returns None, falls back to `_synthesize_metadata()` (line 493-504) with hardcoded `grid_0=1, grid_1=1, grid_2=1`. The bridge will then "tune" against synthetic bounds, not the real kernel. This is logged as a warning, not raised. |
| `src/c_api/` (entire layer) | **PARALLEL FANTASY** | The C-API headers (`triton_c_api.h`, `tvm_c_api.h`, `xla_c_api.h`) define stable ABIs. But `stubs.cpp` is minimal (no real Triton/TVM/XLA linkage), and `c_api/__init__.py` uses ctypes fallbacks. The actual production code (nvidia_backend, gspmd_runner, etc.) imports `triton`, `tvm`, `torch_xla` directly via Python — bypassing the C-API gate. **The "version drift isolation" claim is fictional for the production path.** |
| `src/bridges/aot_packager/metal_backend.py` | **UNVERIFIED** | Listed in src structure. Apple Silicon is mentioned in README. But `nautilus_check_apple()` in runtime_stub.c:143-146 returns 0 on Linux. If Metal support is Linux-only-stub, the "Apple" claim is marketing. |
| `src/bridges/aot_packager/fat_binary.py` | **UNVERIFIED** | Per test_full_pipeline.py:75-99, the FatBinary data model round-trips. But the test uses 10-byte PTX placeholder data, not real outputs. |

### 1.3 GHOSTS (claimed in CHANGELOG, not seen or empty)

- `scripts/verify_env.py` — exists, runs per test_full_pipeline.py:190-200.
- `scripts/check_upstream_drift.py` — listed in CHANGELOG; not read.
- `benchmarks/run_benchmarks.py` — listed in CHANGELOG as 10-kernel suite. **Not verified for actual benchmark execution.** PRD claims "≥30% speedup vs. default Triton on non-Nvidia" — no reproducible benchmark is shown.
- `.github/workflows/real-hardware.yml` — listed in CHANGELOG as GPU matrix H100/MI300X/Gaudi. Not verified.
- `.github/workflows/drift-detection.yml` — listed in CHANGELOG. Not verified.
- `third_party/` — listed in repo structure but appears empty. `pyproject.toml` declares all third-party deps via extras, NOT as submodules. This contradicts docs/E.md "Git Submodules pinned to specific commits."

### 1.4 WIRING GAPS (the audit's sharpest finding)

1. **shard.py CLI does not invoke tuning.** The CLI's `_shard_impl` (line 97-181) calls `gspmd_runner.run()` to compute sharding spec, then calls `_generate_shard_source()` (a hand-written matmul stub). It NEVER calls `bridge_orchestrator.tune_with_real_ir()`. The per-shard fat binary is therefore NEVER built — only StableHLO + spec JSON + a matmul template is emitted. **The `shard` command is a non-functional demo, not a real sharder.**

2. **pipeline_orchestrator's Stage 5 (fat_binary) is unverified.** Line 230-234 calls `self.executor.execute_all_shards(gspmd_result, stablehlo)`. Without reading `hardware_orchestrator.py` end-to-end, we cannot confirm the executor actually invokes `FatBinaryBuilder.build()` per shard. If it just records "would execute" entries, the 5-stage pipeline is 4 stages.

3. **C-API layer is a parallel artifact.** The C headers exist; the Python production code ignores them. There is no `triton_c_api.so` built (test_full_pipeline.py:179-188 explicitly asserts `is_available() == False`).

4. **cross-vendor auto-sharding is unproven.** The "mixed cluster of AMD and Intel" claim requires heterogeneous collective communication (e.g. AMD↔Intel all-reduce). `comm_backend.py` exists with NCCL/RCCL/oneCCL dispatch (per CHANGELOG), but cross-vendor transport (e.g. AMD-MI300X ↔ Intel-Gaudi via Ethernet/UALink) is not demonstrated.

5. **integration tests are mostly unit tests.** `test_full_pipeline.py` has 9 tests; 1 is a CLI-help smoke test, 1 is a fat-binary serialization round-trip (using 10-byte placeholder data), 1 is hardware detection (raises on no GPU). The 1 real end-to-end test (`test_aot_nvidia_round_trip` line 144-167) is `@pytest.mark.gpu` and will **skip in CI without a GPU**. There is NO integration test that proves: PyTorch model → StableHLO → GSPMD → Triton → AOT fat binary → load on AMD or Intel hardware.

---

## 2. ARCHITECTURE CLAIMS vs. REALITY

| Claim | Verdict | Evidence |
|---|---|---|
| "Advanced AI cost model" (auto-tuner) | **HELD** | TVM MetaSchedule is a real evolutionary-search cost model. The bridge wires it correctly. |
| "Single Fat Binary" | **WOBBLY** | Produces a relocatable `.fat.o` ELF, not a standalone executable. Requires a host program to invoke `nautilus_dispatch()`. The marketing overpromises. |
| "Mixed cluster of AMD and Intel" | **FAILED** | Per-vendor compilation is real. Per-vendor comm backends are real. Cross-vendor mixed-cluster collective is unproven. The shard CLI doesn't even produce a fat binary. |
| "Standard PyTorch model code" | **FAILED** | The shard CLI's `_generate_shard_source` is a matmul template — non-matmul models get the same template with different constants. |
| "At maximum speed" | **UNVERIFIED** | No reproducible benchmark. CHANGELOG mentions `benchmarks/run_benchmarks.py` but it has not been audited. |
| "Loss-less CUDA ingestion" | **WOBBLY** | Per `docs/cuda_ingestion_architecture.md`, the translator explicitly admits limitations: shared mem multi-dim arrays, blockIdx*blockDim+threadIdx idiom, pointer declarations, compound assignment, C++11+ features. PRD's 90%+ accuracy target is aspirational, not measured. |
| C-API version-drift isolation | **FAILED** | The C-API layer is a parallel artifact. The production Python bridges import upstream libs directly. Drift is not isolated. |
| "Cross-vendor" Apple Silicon | **WOBBLY** | `metal_backend.py` exists. `nautilus_check_apple()` returns 0 on Linux. No Apple CI runner. |

---

## 3. FUNDAMENTAL GAPS (cannot be patched with more code)

1. **No end-to-end HuggingFace-model-to-sharded-fat-binary demo.** The single most important claim — "takes standard PyTorch model code" — is not demonstrated end-to-end. There is no script that:
   - Loads a real model (e.g. `transformers.AutoModelForCausalLM.from_pretrained`)
   - Runs it through the 5-stage pipeline
   - Loads the resulting fat binary on real hardware
   - Verifies the output is numerically correct

2. **The sharding pipeline produces no fat binary.** The `nautilus shard` CLI emits per-shard StableHLO + spec + a hand-written matmul Triton template. It never calls `FatBinaryBuilder.build()` per shard. This is the gap between the README's "auto-sharded" claim and the actual `shard` command's output.

3. **CI is GPU-less.** `.github/workflows/ci.yml` (assumed) does not include `real-hardware.yml`. The 21-tests-pass claim is the unit test count, not the integration test count. There is no CI that runs the full pipeline on H100+MI300X+Gaudi.

4. **Version drift isolation is fictional for the production path.** The C-API layer is decorative. The Python bridges import `triton`, `tvm`, `torch_xla` directly. When OpenAI/Apache/Google break their APIs, the bridges break. This violates the project anti-pattern from `docs/E.md` "Hermetic C-API Abstraction Layers."

5. **Cross-vendor collective is unproven.** "Mixed cluster of AMD and Intel" requires AMD↔Intel collective over Ethernet/UALink. The codebase has per-vendor comm backends but no cross-vendor transport layer.

---

## 4. SCALE VIABILITY

- 1-GPU single-vendor: REAL (the test `test_aot_nvidia_round_trip` proves the Nvidia path).
- 1-GPU multi-vendor (sequential compile per kernel): REAL (builder.compile per-vendor).
- 8-GPU single-vendor (NCCL all-reduce): UNVERIFIED but plausible (comm_backend.py).
- 8-GPU multi-vendor (cross-vendor collective): UNPROVEN.
- 80-GPU multi-vendor (the "kill NVLink" claim from docs/A.md): UNPROVEN, likely IMPOSSIBLE with current architecture (the per-vendor comm backends don't speak a common transport).

---

## 5. THE ONE QUESTION (lead's bottom line)

**The codebase achieves ~80% of the goal in isolation per bridge, but ~40% of the goal end-to-end — because the shard pipeline does not produce a fat binary, the integration test that proves "PyTorch model → fat binary → mixed cluster" is skipped in CI, and there is no end-to-end demo. To be "real-world ready, not MVP", three things are missing: (a) a working `nautilus shard` that actually invokes the FatBinaryBuilder per shard, (b) a real-hardware CI matrix, and (c) a reproducible benchmark suite that proves "≥30% speedup on non-Nvidia."**

---

## 6. PLAN AGENT — DIRECTIVE

The plan agent must produce a sequenced, parallelized, verification-gated work plan that turns this codebase from "80% real bridges, 40% goal achieved" into "real-world ready, not MVP". The plan must address:

1. **The shard→fat binary gap.** Make `nautilus shard` actually call `FatBinaryBuilder.build()` per shard with the per-shard Triton source generated from real StableHLO→Triton translation (not the hand-written matmul stub).

2. **Replace `_generate_shard_source` with a real StableHLO→Triton translator.** Use the 4-pass TIR conversion pipeline from `triton_tvm/ir_to_tir/` to convert captured StableHLO to Triton, then call `bridge_orchestrator.tune_with_real_ir()` per shard.

3. **A real-hardware CI matrix.** `.github/workflows/real-hardware.yml` must run the full pipeline on H100 + MI300X + Gaudi and assert numerical correctness of a known model (e.g. a small MLP).

4. **A reproducible benchmark suite.** `benchmarks/run_benchmarks.py` must produce the 10-kernel suite with measured speedup numbers that match the PRD's "≥30% on non-Nvidia" claim.

5. **Honest CI test count.** Update the CHANGELOG's "21 tests pass" to distinguish unit tests from integration tests. The 21 claim is misleading.

6. **The cross-vendor collective gap.** Either prove cross-vendor comm works (and document it) or remove the "mixed cluster" claim from the README.

7. **Apple Silicon reality check.** Either ship a working `metal_backend.py` with an Apple CI runner, or document that Apple is best-effort.

8. **Version drift isolation honesty.** Either implement the C-API gate (with real `triton_c_api.so`, `tvm_c_api.so`, `xla_c_api.so` linking upstream libs) and route the production code through it, or remove the "version-drift isolation" claim from `docs/ARCHITECTURE.md`.

The plan must sequence these as: **(1) P0 ship-blockers first** (shard→fat binary wiring + replacement of stub generator), **(2) P0 verification infrastructure** (real-hardware CI + benchmark), **(3) P1 honesty passes** (CHANGELOG correction, version drift reality), **(4) P2 stretch** (cross-vendor collective, Apple Silicon). Each item must have a concrete file:line target, a verification gate (a test that must pass), and a rollback plan.
