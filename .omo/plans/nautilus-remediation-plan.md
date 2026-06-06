# Nautilus Remediation Plan — From "80% real bridges, 40% goal achieved" to Real-World Ready

**Plan date:** 2026-06-05
**Source audit:** `.omo/plans/nautilus-audit-bundle.md`
**Author:** plan agent (read-only)
**Execution mode:** sequenced, parallelized where independent, evidence-gated

---

## 0. Sequencing Overview

The audit's three P0 ship-blockers, three P0 verification items, three P1 honesty passes, and two P2 stretches are reorganized into **seven execution phases** with explicit dependency arrows. Phases at the same depth are independent and can run in parallel; phases at different depths are strict prerequisites.

```
PHASE 0 (P1, parallel) ─── Truth Reconciliation
   │
PHASE 1 (P0) ── StableHLO→Triton Real Translator ─────────┐
   │                                                          │
PHASE 2 (P0) ── Wire shard CLI through PipelineOrchestrator ─┴──┐
   │                                                          │
PHASE 3 (P0) ── End-to-End HuggingFace Demo ★ Definition of Done ★
   │
PHASE 4 (P0) ── Real-Hardware CI Matrix (must actually run)
   │
PHASE 5 (P0) ── Reproducible Benchmark Suite (real speedup numbers)
   │
PHASE 6 (P1/P2 parallel) ── Cross-vendor + Apple honesty
   │
PHASE 7 (P1) ── C-API version-drift reality + CHANGELOG fix
```

The **single Definition of Done (DoD)** is **Phase 3**: a one-command demo loads a real PyTorch model from HuggingFace, runs it through capture → StableHLO → GSPMD → per-shard Triton source generation → per-vendor AOT compilation → fat binary per shard → loads the fat binary on the matching hardware → asserts `mean(|output − pytorch_reference|) < 1e-2`. The other phases exist to make Phase 3 pass, and to ensure the result is reproducible, not a one-shot trick.

---

## 0.1 Definition of Done (DoD)

A single, measurable, reproducible artifact. This is what the user means by "real-world ready, not MVP, not fake showcasing."

| # | Requirement | Concrete artifact | Pass criterion |
|---|---|---|---|
| DoD-1 | HuggingFace model → fat binary pipeline runs end-to-end | `scripts/demo_e2e.py` loads `hf-internal-testing/tiny-llama` (or any small `AutoModelForCausalLM`) and produces `shards/shard_*/kernel.fat.o` per device | All shard dirs contain a `kernel.fat.o` ≥ 1 KB; per-shard stablehlo.mlir is non-empty; no NotImplementedError |
| DoD-2 | Per-shard fat binary is model-specific (not a generic matmul) | diff between any two shards' `kernel.py` is non-empty; the Triton source contains the model op (e.g. `tl.softmax`, `tl.gelu`) | `len(set(kernel_sources)) == num_shards` for a model with N>1 distinct ops |
| DoD-3 | Fat binary loads on real hardware | `python -c "import ctypes; ctypes.CDLL('shards/shard_0/kernel.fat.o')"` succeeds; `nautilus_detect_vendor()` returns matching vendor | `ran=True, detected_vendor in {nvidia,amd,intel}` |
| DoD-4 | Numerical correctness vs PyTorch eager | `scripts/demo_e2e.py` runs both fat-binary and reference forward passes; compares outputs | `max_abs_diff < 1e-2` for fp16/bf16, `< 1e-5` for fp32 |
| DoD-5 | Auto-sharding decision is data-dependent, not constant | GSPMD output has `> 1` unique sharding specs across the model | `len({s.partition_shape for s in spec.tensor_shardings.values()}) > 1` |
| DoD-6 | CI matrix actually runs the DoD demo | `.github/workflows/real-hardware.yml` runs `python scripts/demo_e2e.py --model tiny-llama` and exits non-zero on DoD failure | Workflow status badge is "passing" on main |
| DoD-7 | Speedup claim is reproducible | `benchmarks/run_benchmarks.py` produces `results.json` with measured TFLOPs for nvidia + ≥1 other vendor | At least one (kernel, non-Nvidia) cell has `speedup ≥ 1.30` (matches PRD) |

**DoD gate:** All seven rows must be true. If any is false, the codebase is not real-world ready.

---

## PHASE 0 — Truth Reconciliation (P1, ~1 day, parallel)

**Why first:** The codebase's CHANGELOG, README, and audit bundle disagree on what exists. Before adding code, fix the documentation so the team has a single source of truth for what is real and what is stubbed. **No code changes here** — only `git grep` and `sed`.

**Why P1, not P0:** Documentation does not block the goal; it blocks *measuring progress* against the goal. P0 ship-blockers below all need to be measurable, which requires this phase first.

### Step 0.1 — Catalogue real tests vs claimed

**File:line targets:**
- `/workspaces/NVINDIA_CUD/CHANGELOG.md:113-118` (claims "21 tests pass")
- `/workspaces/NVINDIA_CUD/src/tests/integration/test_full_pipeline.py:65-200` (9 tests, 1 GPU-marked)
- `/workspaces/NVINDIA_CUD/src/bridges/pytorch_xla/tests/test_*.py` (8+10+8+14+8 = 48 test fns)
- `/workspaces/NVINDIA_CUD/src/bridges/aot_packager/tests/test_*.py` (~30 test fns)
- `/workspaces/NVINDIA_CUD/src/bridges/triton_tvm/tests/test_*.py` (~30 test fns)

**Work:**
1. `git grep -c "def test_" src/ --include "test_*.py" | sort` → produce authoritative count.
2. Run `python -m pytest src/ -v -m "not gpu and not cuda and not rocm and not intel and not slow" --collect-only -q` → count what actually runs in CI.
3. Run `python -m pytest src/ --collect-only -q` → count total.
4. Produce a `docs/test-inventory.md` (new file) with three columns: `total_tests`, `cpu_only_pass`, `gpu_required_skip`.

**Verification gate:**
```bash
test -f docs/test-inventory.md && \
  grep -qE "^\| [0-9]+ \| [0-9]+ \| [0-9]+ \|$" docs/test-inventory.md
```

**Rollback:** Delete `docs/test-inventory.md`. No other state changed.

### Step 0.2 — Reconcile CHANGELOG count

**File:line targets:** `/workspaces/NVINDIA_CUD/CHANGELOG.md:113-118, 143-145`

**Work:** Replace "21 tests pass on every PR; 2 correctly skipped" with the real number from Step 0.1, e.g. "127 CPU-only tests pass on every PR; 6 GPU-marked tests skip without self-hosted runner; 3 require `pip install -e .[all]`."

**Verification gate:**
```bash
grep -E "tests pass on every PR" CHANGELOG.md | grep -v "21"   # must be empty
test "$(grep -c 'def test_' $(find src -name 'test_*.py') | awk -F: '{s+=$2} END{print s}')" -ge 100
```

**Rollback:** `git checkout CHANGELOG.md`.

### Step 0.3 — Reconcile benchmarks README

**File:line targets:**
- `/workspaces/NVINDIA_CUD/benchmarks/README.md:8-25` (claims 10 kernels)
- `/workspaces/NVINDIA_CUD/benchmarks/run_benchmarks.py:31-38, 158-162` (only matmul + softmax runners)

**Work:** Mark missing kernels (`gelu`, `reduce`, `layer_norm`, `embedding`, `attention`, `conv2d`, `scan`, `fused_attention`) as `[ ]` in README, not `[x]`. Or implement them in Phase 5.

**Verification gate:**
```bash
diff <(grep -E '^\| `[a-z_]+\.py' benchmarks/README.md | wc -l) \
     <(grep -E '^def benchmark_' benchmarks/run_benchmarks.py | wc -l)
# Must be equal
```

**Rollback:** `git checkout benchmarks/README.md benchmarks/run_benchmarks.py`.

### Step 0.4 — Mark stubs explicitly

**File:line targets:**
- `/workspaces/NVINDIA_CUD/src/cli/commands/shard.py:264-309` (`_generate_shard_source`)
- `/workspaces/NVINDIA_CUD/src/bridges/pytorch_xla/hardware_orchestrator.py:147-186` (second copy of the same stub)
- `/workspaces/NVINDIA_CUD/src/bridges/aot_packager/linker.py:362-407` (`_write_minimal_fat_binary`)
- `/workspaces/NVINDIA_CUD/src/bridges/aot_packager/builder.py:423-449` (`_minimal_elf_stub`)

**Work:** Add a `STUB:` comment block at the top of each stub, with a `TODO(issue-N):` reference to the GitHub issue. Do not delete; do not fix yet. This makes `git grep STUB:` surface all known stubs.

**Verification gate:**
```bash
test "$(grep -rE 'STUB: TODO' src/ | wc -l)" -ge 4
```

**Rollback:** `git checkout src/cli/commands/shard.py src/bridges/pytorch_xla/hardware_orchestrator.py src/bridges/aot_packager/linker.py src/bridges/aot_packager/builder.py`.

**Phase 0 exit gate:** Steps 0.1–0.4 all pass, and the audit's three STUB count in `shard.py:264` and `hardware_orchestrator.py:147` is acknowledged in issue tracker. No source code touched.

---

## PHASE 1 — StableHLO→Triton Real Translator (P0, ~5 days)

**This is the biggest ship-blocker.** The audit confirms (lines 37, 57, 92) that `shard.py:264-309` and `hardware_orchestrator.py:147-186` both emit a hand-written matmul template. The 4-pass TTGIR→TIR pipeline at `src/bridges/triton_tvm/ir_to_tir/` exists but is for **Triton → TVM**, not for **StableHLO → Triton**. Building a real StableHLO→Triton translator is the foundation that unblocks everything else.

### Step 1.1 — Survey what's already in the repo for translator work

**File:line targets:**
- `/workspaces/NVINDIA_CUD/src/bridges/triton_tvm/ir_to_tir/ttgir_parser.py` (TTGIR AST)
- `/workspaces/NVINDIA_CUD/src/bridges/triton_tvm/ir_to_tir/pass1-4` (lowering passes)
- `/workspaces/NVINDIA_CUD/src/bridges/triton_tvm/ir_to_tir/tt_dot_split.py` (matmul handling)
- `/workspaces/NVINDIA_CUD/src/bridges/pytorch_xla/stablehlo_export.py` (already produces MLIR text per audit)
- `/workspaces/NVINDIA_CUD/src/bridges/cuda_ingest/translator.py` (existing AST→target translator pattern)

**Work (read-only):** produce a `docs/translator-survey.md` listing:
- StableHLO op coverage required: dot, add, mul, reduce, broadcast, reshape, transpose, concatenate, slice, convert (dtype), convolution (defer), compare, select.
- For each: does an existing parser pass handle it? If not, the missing work.

**Verification gate:** `test -f docs/translator-survey.md` and it contains a table with at least 10 StableHLO ops.

**Rollback:** Delete `docs/translator-survey.md`.

### Step 1.2 — Build `StableHLOToTriton` translator

**File:line targets (new module):**
- New file: `/workspaces/NVINDIA_CUD/src/bridges/pytorch_xla/stablehlo_to_triton.py`
- New directory: `/workspaces/NVINDIA_CUD/src/bridges/pytorch_xla/tests/test_stablehlo_to_triton.py`

**Contract:**
```python
def translate(stablehlo_mlir: str, *, kernel_name: str, target: str) -> TritonSource:
    """Translate a (per-shard) StableHLO MLIR string into a Triton @triton.jit source.

    Returns TritonSource with:
      - source: full Python source of a @triton.jit function
      - kernel_name: matches the requested name
      - input_specs: list of (name, shape, dtype) for placeholders
      - output_specs: list of (name, shape, dtype) for outputs
      - constants: dict of compile-time constants
      - op_to_triton_map: {stablehlo_op_name: triton_helper_used} for auditing

    Raises:
      UnsupportedStableHLOOpError: with the offending op name and line number
    """
```

**Work breakdown (in this order, sub-gated):**
1. **Step 1.2a:** MLIR parser that pulls `stablehlo.func` arguments, `stablehlo.return` outputs, and the body ops into a list of typed op records. Parser only; no codegen. Test on 5 hand-written MLIR snippets (matmul, add, mul, relu, reshape).
2. **Step 1.2b:** Codegen for elementwise/pointwise ops (add, mul, sub, div, convert, compare, select). One Triton `tl.load`+`tl.store` per tensor. Test on 5 synthetic IRs.
3. **Step 1.2c:** Codegen for `stablehlo.dot` → `tl.dot` with appropriate `BLOCK_M/BLOCK_N/BLOCK_K` from the args' shapes.
4. **Step 1.2d:** Codegen for reductions (`stablehlo.reduce` with `add`/`max`/`min`) → `tl.sum`/`tl.max`/`tl.min` with axis arg.
5. **Step 1.2e:** Codegen for `stablehlo.broadcast_in_dim` and `stablehlo.reshape` → `tl.broadcast_to` and reshape assertions.
6. **Step 1.2f:** Codegen for `stablehlo.concatenate` and `stablehlo.slice` → `tl.cat` and slicing.
7. **Step 1.2g:** Glue: load the generated source, execute it via `triton.jit` on a small input, assert `output == reference` for each op family.

**Verification gate (Step 1.2 composite):**
```bash
pytest src/bridges/pytorch_xla/tests/test_stablehlo_to_triton.py -v
# Must pass at least one test per op family (1.2b–1.2f)
# AND
pytest src/bridges/pytorch_xla/tests/test_stablehlo_to_triton.py::test_end_toend_simple_model -v
# Loads a 2-layer MLP's StableHLO, translates, runs, compares to torch eager
```

**Specific assertion in the test (this is the gate that proves the translator is real, not stub):**
```python
# Inside test_end_toend_simple_model
def test_end_toend_simple_model():
    model = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.GELU(), torch.nn.Linear(128, 64))
    mlir = export_to_stablehlo(model, torch.randn(2, 64))
    src = translate(mlir, kernel_name="mlp_fwd", target="nvidia/sm_90")
    # src.source must contain real ops, not the audit's matmul template
    assert "tl.dot" in src.source or "@triton.jit" in src.source
    # Run the kernel via exec/compile
    out = run_triton_source(src, torch.randn(2, 64))
    ref = model(torch.randn(2, 64))
    assert torch.allclose(out, ref, atol=1e-2)
```

**Rollback:** Delete `src/bridges/pytorch_xla/stablehlo_to_triton.py` and the test file. The codebase is in the same state as before Phase 1.2.

### Step 1.3 — Wire `bridge_orchestrator.tune_with_real_ir` for StableHLO source

**File:line targets:**
- `/workspaces/NVINDIA_CUD/src/bridges/triton_tvm/bridge_orchestrator.py:238-349` (`tune_with_real_ir` currently only takes `source_hash`; the audit confirms it falls back to synthetic metadata at lines 282-284 when IRCapture returns None)
- `/workspaces/NVINDIA_CUD/src/bridges/triton_tvm/ir_capture.py:107-166` (IRCapture reads a buffer; needs a new method for StableHLO input)

**Work:** Add `tune_with_stablehlo(stablehlo_mlir: str, target: str) -> TuningResult` that:
1. Calls `StableHLOToTriton.translate(mlir, kernel_name=...)` to get Triton source.
2. Wraps the source in a temp file and runs `triton.compile(source, target=target)` to get a real source_hash.
3. Calls the existing IRCapture path with that hash.
4. Returns the tuned `MappedTuningConfig` (block_m, block_n, block_k, num_warps, num_stages).

**Verification gate:**
```bash
pytest src/bridges/triton_tvm/tests/test_bridge_orchestrator.py::test_tune_with_stablehlo_returns_real_config -v
# Test asserts: returned config has block_m >= 32 AND came from MetaSchedule, not defaults
```

**Rollback:** Revert the new method; `tune_with_real_ir` still works as before.

**Phase 1 exit gate:** Step 1.2 + 1.3 tests pass. A real StableHLO op (e.g. `stablehlo.add`) translates to a Triton source that, when executed, produces the correct tensor. This is the foundation for Phase 2.

---

## PHASE 2 — Wire `nautilus shard` through PipelineOrchestrator with per-shard fat binary (P0, ~3 days, **strictly after Phase 1**)

The audit confirms (lines 57, 230-234) that `shard.py` bypasses `PipelineOrchestrator.shard()` entirely. It calls the lower-level `gspmd_runner.run()` and then the stub `_generate_shard_source()`. The audit's hardware_orchestrator also has a duplicate stub at line 147.

### Step 2.1 — Delete the stubs and route through the orchestrator

**File:line targets:**
- `/workspaces/NVINDIA_CUD/src/cli/commands/shard.py:97-181` (`_shard_impl`) — replace with a call to `AutoShardingBridge.shard()`.
- `/workspaces/NVINDIA_CUD/src/cli/commands/shard.py:264-309` (`_generate_shard_source`) — **delete the function**. The orchestrator handles per-shard source generation.
- `/workspaces/NVINDIA_CUD/src/bridges/pytorch_xla/hardware_orchestrator.py:147-186` (`_generate_shard_source` duplicate) — **delete or rewrite** to call `StableHLOToTriton.translate()` per shard.

**Work:** Rewrite `_shard_impl` so:
```python
def _shard_impl(model_file, mesh_str, strategy, output_dir, example_inputs):
    captured = _capture_model(model_file, example_inputs)
    stablehlo = _export_to_stablehlo(captured)
    mesh = _parse_mesh(mesh_str)
    device_mesh = _build_device_mesh(mesh)
    bridge = AutoShardingBridge()
    config = ShardingConfig(model=..., example_inputs=..., device_mesh=device_mesh,
                            sharding_strategy=..., enable_fat_binary=True)
    result = bridge.shard(...)
    # result.shard_executions[i].fat_binary_result.fat_binary is the per-shard fat binary
    for i, exec_result in enumerate(result.shard_executions):
        shard_dir = output_dir / f"shard_{i:04d}"
        shard_dir.mkdir(exist_ok=True)
        (shard_dir / "kernel.fat.o").write_bytes(exec_result.fat_binary_result.fat_binary.to_bytes())
        (shard_dir / "stablehlo.mlir").write_text(result.stablehlo_module.mlir_text)
        # The actual Triton source is recoverable from the builder result
```

**Verification gate:**
```bash
# This is the critical new test
pytest src/tests/integration/test_full_pipeline.py::test_shard_emits_fat_binary_per_shard -v
# The test must assert:
#   - shards/shard_*/kernel.fat.o exists
#   - file size > 0
#   - the bytes start with b"\x7fELF" (real ELF, not a JSON manifest)
#   - the per-shard Triton source contains ops OTHER than tl.dot (model-specific)
```

**Rollback:** Revert `shard.py` and `hardware_orchestrator.py` to their pre-Phase-2 state. The codebase still runs; `nautilus shard` is back to emitting the matmul template.

### Step 2.2 — Wire `hardware_orchestrator._execute_single_shard` to use `StableHLOToTriton`

**File:line targets:**
- `/workspaces/NVINDIA_CUD/src/bridges/pytorch_xla/hardware_orchestrator.py:107-186`
- `/workspaces/NVINDIA_CUD/src/bridges/pytorch_xla/pipeline_orchestrator.py:227-234`

**Work:** Replace the `_generate_shard_source` matmul template with:
```python
def _per_shard_stablehlo(self, gspmd_result, stablehlo_module, shard_id) -> str:
    """Extract the per-shard StableHLO by filtering ops assigned to this shard
    via the GSPMD sharding annotations."""
    # Use sharding annotations in the MLIR to slice the func body
    # For v1: take the full module; per-shard slicing is a Phase 6 stretch
    return stablehlo_module.mlir_text

def _generate_shard_source(self, shard_id, gspmd_result, stablehlo_module) -> str:
    mlir = self._per_shard_stablehlo(gspmd_result, stablehlo_module, shard_id)
    src = stablehlo_to_triton.translate(
        mlir,
        kernel_name=f"shard_{shard_id}_kernel",
        target=gspmd_result.target or "cuda",
    )
    return src.source
```

**Verification gate:**
```bash
pytest src/bridges/pytorch_xla/tests/test_hardware_orchestrator.py::test_per_shard_source_is_model_specific -v
# Asserts that two shards of a model with multiple distinct ops (e.g. Linear+GELU+Linear)
# produce two DIFFERENT kernel.py files
```

**Rollback:** Revert `_generate_shard_source` to its pre-Phase-2.2 state (the audit's matmul template).

### Step 2.3 — Make `pipeline_orchestrator` and the per-shard executor agree on the source

**File:line targets:**
- `/workspaces/NVINDIA_CUD/src/bridges/pytorch_xla/pipeline_orchestrator.py:227-234`
- `/workspaces/NVINDIA_CUD/src/bridges/pytorch_xla/hardware_orchestrator.py:120-145`

**Work:** Both call sites should use the same per-shard Triton source. Add `shard_kernel_sources: dict[int, str]` to `ShardingResult` and have the pipeline orchestrator call `_generate_shard_source` once per shard and pass the result to `_execute_single_shard` (instead of letting the executor generate its own).

**Verification gate:**
```bash
pytest src/bridges/pytorch_xla/tests/test_pipeline_orchestrator.py::test_pipeline_per_shard_sources_match_executor -v
# Asserts the executor uses the source the pipeline orchestrator computed
```

**Rollback:** Revert pipeline_orchestrator and hardware_orchestrator to pre-Phase-2.3 state.

**Phase 2 exit gate:** `nautilus shard model.py --mesh 2,2 --output-dir ./shards` produces 4 shard dirs, each with a `kernel.fat.o` whose contents include the model-specific op (`tl.gelu`, `tl.softmax`, etc.), not just a matmul.

---

## PHASE 3 — End-to-End HuggingFace Demo + Hardware-Verified Pipeline (P0, ~4 days, **strictly after Phase 2**)

This is the **Definition of Done**. Without this, the codebase is 40% goal-achieved; with this, it is 80%+.

### Step 3.1 — Build `scripts/demo_e2e.py`

**File:line targets:** New file `/workspaces/NVINDIA_CUD/scripts/demo_e2e.py`

**Contract:**
```python
"""End-to-end Nautilus demo: HuggingFace model → sharded fat binaries → verify on hardware.

This script is the single artifact that proves the framework achieves its stated
goal. It runs in CI (real-hardware.yml) and locally (if you have a GPU).

Usage:
    python scripts/demo_e2e.py --model hf-internal-testing/tiny-llama --mesh 1,1
    python scripts/demo_e2e.py --model /path/to/my_model.py --mesh 2,2
"""
```

**Work:** The script does, in order:
1. Load a HuggingFace model (or a local PyTorch model with `EXAMPLE_INPUTS`).
2. Construct a `DeviceMesh` from `nvidia-smi` / `rocm-smi` / `oneapi-smi` output.
3. Call `AutoShardingBridge.shard()` with `enable_fat_binary=True`.
4. For each `result.shard_executions[i]`:
   - Write `kernel.fat.o` to `output_dir/shard_*/`.
   - `ctypes.CDLL(kernel.fat.o)` → assert load succeeds.
   - Call `lib.nautilus_detect_vendor()` → assert matches the shard's assigned vendor.
5. Run the original PyTorch model with example inputs → reference output.
6. (Optional, requires runtime support) Load and call the fat binary's exposed function.
7. Compare numerical output.
8. Print a single JSON report with: `model_name, mesh, num_shards, per_shard_fat_binary_size, per_shard_vendor, numerical_max_diff, PASS/FAIL`.

**Verification gate:**
```bash
# On a host with a real GPU:
python scripts/demo_e2e.py --model hf-internal-testing/tiny-llama --mesh 1,1
# Exit 0 and a JSON report with "PASS": true
# AND
test -f artifacts/demo_report.json && jq '.PASS' artifacts/demo_report.json | grep -q true
```

**Rollback:** Delete `scripts/demo_e2e.py`. The existing tests still pass.

### Step 3.2 — Add integration test that exercises the demo on CPU-only host

**File:line targets:** New file `/workspaces/NVINDIA_CUD/src/tests/integration/test_hf_demo.py`

**Work:** A test that:
1. Mocks the GPU detection (or runs on a host with no GPU and uses `skip_amd=skip_intel=skip_nvidia=True`).
2. Loads `hf-internal-testing/tiny-distilbert` (smaller, faster than llama).
3. Runs the pipeline up to AOT compile; asserts each shard has a non-empty `kernel.fat.o` ≥ 1 KB.
4. Skips with `pytest.mark.skipif(not torch.cuda.is_available() ...)` for the load-and-verify step.

**Verification gate:**
```bash
pytest src/tests/integration/test_hf_demo.py -v -m "not gpu"
# Must pass: model → StableHLO → GSPMD → per-shard Triton source → fat binary
```

**Rollback:** Delete the test file.

### Step 3.3 — Add the demo to the test inventory

**File:line targets:**
- `/workspaces/NVINDIA_CUD/src/tests/integration/__init__.py`
- `/workspaces/NVINDIA_CUD/pyproject.toml:158-160` (markers list)

**Work:** Add `@pytest.mark.requires_hf` marker; ensure the demo's HF-specific tests are tagged.

**Verification gate:**
```bash
pytest --collect-only src/tests/integration/test_hf_demo.py | grep "requires_hf"
```

**Rollback:** `git checkout pyproject.toml`.

**Phase 3 exit gate:** DoD rows 1, 2, 3, 4, 5 are achievable. The demo runs end-to-end on at least one real-hardware runner and produces a PASS report.

---

## PHASE 4 — Real-Hardware CI Matrix (P0, ~2 days, **starts after Phase 1; can run parallel to Phase 2/3**)

The audit (lines 94, 137) flags that the real-hardware workflow is a fiction. The workflow file exists but uses `runs-on: [self-hosted, gpu, ...]` and no self-hosted runner is configured. Either configure runners, or replace with cloud-runner equivalents.

### Step 4.1 — Decide runner strategy

**File:line targets:** `/workspaces/NVINDIA_CUD/.github/workflows/real-hardware.yml:21-97`

**Work:** Add a decision record in `docs/runner-strategy.md` (new file):
- **Option A:** Self-hosted runners with H100/MI300X/Gaudi labels. Pro: full control. Con: requires physical hardware, ongoing maintenance.
- **Option B:** Cloud runners — GitHub-hosted `gpu-runners` (limited), AWS/GCP spot with GPU, or Intel Tiber AI Cloud + AMD Developer Cloud (per the AGENTS.md critical knowledge). Pro: zero hardware cost. Con: $$$, quota.
- **Option C:** Hybrid — Github-hosted for Nvidia (H100 via `runs-on: gpu-runner-4xlarge` if available), AMD/Intel cloud for non-Nvidia.

Pick one. The Phase 4 schedule depends on this.

**Verification gate:** `test -f docs/runner-strategy.md` with a marked `## Decision:` section naming the chosen option.

**Rollback:** Delete the file.

### Step 4.2 — Implement the chosen runner strategy

**File:line targets:** `/workspaces/NVINDIA_CUD/.github/workflows/real-hardware.yml`

**Work:** Rewrite the workflow to:
1. Provision the chosen runner (cloud or self-hosted).
2. Run `pip install -e .[all,dev]`.
3. Run the hardware-specific deps (`apt install lld spirv-tools gcc`, plus ROCm or oneAPI toolkit).
4. Run `python scripts/demo_e2e.py --model hf-internal-testing/tiny-distilbert --mesh 1,1 --vendor ${{ matrix.vendor }}`.
5. Assert `jq '.PASS' artifacts/demo_report.json` is `true`.
6. On cross-vendor job: use AMD+Intel cloud runners, validate a 2-device mixed mesh.

**Verification gate:**
```bash
# On real hardware: the workflow is green
gh run list --workflow real-hardware.yml --limit 5 --json status,conclusion | jq -e '.[] | select(.conclusion != "success") | length == 0'
```

**Rollback:** `git checkout .github/workflows/real-hardware.yml`.

### Step 4.3 — Schedule the workflow daily (drift detection of hardware behavior)

**File:line targets:** `/workspaces/NVINDIA_CUD/.github/workflows/real-hardware.yml:9-13` (on: push/pull_request)

**Work:** Add `schedule: - cron: '0 6 * * *'` (matches `docs/E.md` drift strategy).

**Verification gate:** Workflow appears under Actions → "Real Hardware" with a daily schedule.

**Rollback:** `git checkout .github/workflows/real-hardware.yml`.

**Phase 4 exit gate:** `real-hardware.yml` has at least one green run on a real GPU and is scheduled to run daily.

---

## PHASE 5 — Reproducible Benchmark Suite (P0, ~3 days, **can run parallel to Phase 2/3/4**)

The audit (lines 50, 109, 114) flags that the benchmark suite is incomplete and the speedup claim is unmeasured. CHANGELOG/README claim 10 kernels; only 2 exist.

### Step 5.1 — Implement the missing kernel runners

**File:line targets:** `/workspaces/NVINDIA_CUD/benchmarks/run_benchmarks.py:158-162` (only `matmul` + `softmax` in `runners` dict; README claims 10)

**Work:** Add runners for: `gelu`, `reduce`, `layer_norm`, `embedding`. Each follows the same pattern as `benchmark_matmul` (warmup + timed loop + TFLOPs/GBps + speedup computation). For non-matmul kernels, use `GB/s` as the metric.

**Verification gate:**
```bash
python benchmarks/run_benchmarks.py --kernel gelu --trials 5
# Must run and print a JSON with at least baseline_gbps + tuned_gbps + speedup
# AND
diff <(grep -E '^def benchmark_' benchmarks/run_benchmarks.py | wc -l) \
     <(grep -E '^\| `[a-z_]+\.py' benchmarks/README.md | wc -l)
# Must be 0 (equal)
```

**Rollback:** Revert the runner additions.

### Step 5.2 — Run benchmarks on real hardware, store in `benchmarks/results.json`

**File:line targets:** New file `/workspaces/NVINDIA_CUD/benchmarks/results.json` (committed to repo)

**Work:** On at least one Nvidia + one non-Nvidia host, run:
```bash
python benchmarks/run_benchmarks.py --output benchmarks/results.json --trials 10
```
Commit the JSON.

**Verification gate:**
```bash
python -c "
import json
r = json.load(open('benchmarks/results.json'))
for kernel, data in r.items():
    if 'skipped' in data or 'error' in data:
        continue
    for vendor, m in data['results'].items():
        if 'nvidia' in vendor: continue
        # PRD: ≥30% speedup on non-Nvidia
        assert m.get('speedup', 0) >= 1.0, f'{kernel}/{vendor} regressed: {m}'
"
# Must exit 0
```

**Rollback:** `rm benchmarks/results.json`.

### Step 5.3 — Add a benchmark CI job

**File:line targets:** `/workspaces/NVINDIA_CUD/.github/workflows/ci.yml:91-133` (existing build-fat-binary job)

**Work:** Add a new job `benchmark-regression` that runs on the same hardware as `real-hardware.yml`, executes `benchmarks/run_benchmarks.py`, and asserts the new `results.json` does not regress the committed `benchmarks/results.json` by more than 10%.

**Verification gate:**
```bash
# Locally
python scripts/compare_benchmarks.py benchmarks/results.json benchmarks/results_new.json --threshold 0.10
# Exits 0 if no regression > 10%
```

**Rollback:** Remove the new job from `ci.yml`.

**Phase 5 exit gate:** All 10 README-claimed kernels have runners; committed `results.json` shows real measured numbers; CI catches regressions.

---

## PHASE 6 — Cross-vendor + Apple Honesty (P1, ~2 days, parallel to Phase 4/5)

The audit (lines 63, 98, 108) flags that "mixed cluster of AMD and Intel" is a marketing claim. The `comm_backend.py:143-166` builds a CPU-staging "bridge" that is a 64 GB/s PCIe data structure, not a real transport. Apple has no CI runner.

### Step 6.1 — Honest measurement: 2-host AMD↔Intel all-reduce over TCP

**File:line targets:**
- `/workspaces/NVINDIA_CUD/src/bridges/pytorch_xla/comm_backend.py:143-166`
- `/workspaces/NVINDIA_CUD/scripts/measure_cross_vendor.py` (new file)

**Work:** Build a script that:
1. Spins up a 2-process group: one on AMD, one on Intel (separate machines or VMs).
2. Issues an all-reduce via the chosen transport: GLOO over TCP, or RCCL+oneCCL with explicit host-buffer staging.
3. Measures bandwidth GB/s and latency.
4. Compares to native single-vendor (RCCL-only on AMD, oneCCL-only on Intel).

**Verification gate:**
```bash
python scripts/measure_cross_vendor.py --transport tcp --output benchmarks/cross_vendor.json
# Output must include a measured cross-vendor bandwidth, not a constant
# AND
test $(jq '.cross_vendor_gbps' benchmarks/cross_vendor.json) -gt 0
```

**Rollback:** Delete the script. `comm_backend.py` unchanged.

### Step 6.2 — Either prove cross-vendor, or remove the README claim

**File:line targets:** `/workspaces/NVINDIA_CUD/README.md:7-13`

**Work:** If Step 6.1's measured cross-vendor bandwidth is ≥ 1 GB/s and the code path is real, add a benchmark line to the README. If not, replace "auto-shards them across mixed-vendor clusters" with "supports AMD and Intel targets (per-vendor compilation; cross-vendor transport best-effort)."

**Verification gate:** `grep -E "(mixed|across mixed)" README.md` either:
- Matches AND `benchmarks/cross_vendor.json` has a real measured number, OR
- Does not match (claim removed).

**Rollback:** `git checkout README.md`.

### Step 6.3 — Apple Silicon reality check

**File:line targets:**
- `/workspaces/NVINDIA_CUD/src/bridges/aot_packager/metal_backend.py` (full file audit; currently raises on non-Darwin)
- `/workspaces/NVINDIA_CUD/src/bridges/aot_packager/runtime_stub.c:143-158` (`nautilus_check_apple` returns 0 on non-Darwin)
- `/workspaces/NVINDIA_CUD/README.md:7-13`

**Work (pick one of two):**
- **Option A (ship Apple):** Add a `macos-14` runner to `real-hardware.yml` that runs `nautilus build` with `metal/apple_m2`. Commit a `benchmarks/apple_m2.json`.
- **Option B (defer Apple):** Mark `metal_backend.py` as `STATUS: best-effort, no CI` in the docstring, and remove "Apple" from the README's "any GPU vendor" line.

**Verification gate:**
- Option A: `gh workflow run real-hardware.yml --ref feat/apple` → green status.
- Option B: `grep -E "Apple|apple" README.md` either does not match, OR matches only in a clearly-marked best-effort section.

**Rollback:** `git checkout` the changed files.

**Phase 6 exit gate:** Either cross-vendor collective is measured and documented, or the claim is removed from README. Apple status is unambiguous.

---

## PHASE 7 — C-API Version-Drift Reality + Final CHANGELOG (P1, ~1 day, last)

The audit (lines 42, 61, 96, 136) confirms the C-API is decorative: `is_available()` returns False, the production Python imports `triton`/`tvm`/`torch_xla` directly. The C library has no real linkage to the upstream libraries.

### Step 7.1 — Decide: implement C-API gate, or remove the claim

**File:line targets:**
- `/workspaces/NVINDIA_CUD/src/c_api/stubs.cpp` (only 4 KB; no real linkage)
- `/workspaces/NVINDIA_CUD/src/c_api/__init__.py:64-296` (loads `.so`; if missing raises)
- `/workspaces/NVINDIA_CUD/AGENTS.md` "Critical Knowledge" item 2 ("Version drift breaks everything")
- `/workspaces/NVINDIA_CUD/docs/ARCHITECTURE.md` (if it exists; else `TECH_SPEC.md`)

**Work (pick one):**
- **Option A (implement):** Write a real `stubs.cpp` that compiles `triton-c-api` from third_party pin and exposes the C functions. Update the production Python imports to go through `c_api.compile()` instead of `triton.compiler.compile()` directly. **Estimated: 5 days, may not be feasible in budget.** This is the audit's P1 stretch.
- **Option B (remove claim):** Update `AGENTS.md`, `README.md`, `TECH_SPEC.md`, `CHANGELOG.md` to say: "Version drift isolation is a future goal; the current codebase imports upstream libraries directly. Pinning via `pyproject.toml` extras is the current mitigation."

**Verification gate (Option B):**
```bash
grep -l "version-drift isolation\|version drift isolation" docs/ AGENTS.md README.md CHANGELOG.md 2>/dev/null
# Either: empty (claim removed), OR the file contains a disclaimer
```

**Rollback:** `git checkout` the changed files.

### Step 7.2 — Final CHANGELOG honesty pass

**File:line targets:** `/workspaces/NVINDIA_CUD/CHANGELOG.md` (entire file)

**Work:** Add a new section `## [0.2.0] - <date> ### Reality pass` that lists:
- The Phase 2 stub deletion and replacement.
- The Phase 3 demo and its CI integration.
- The Phase 5 committed benchmark numbers.
- The Phase 6 honesty fixes.
- The Phase 7 C-API decision.
- The corrected test count from Phase 0.

**Verification gate:** CHANGELOG has a new section dated 2026-06-05 or later.

**Rollback:** `git checkout CHANGELOG.md`.

### Step 7.3 — Final "goal achievement" badge

**File:line targets:** `/workspaces/NVINDIA_CUD/README.md` (top of file)

**Work:** Add a one-line statement of the DoD:
```
**Status (0.2.0):** Goal-achieved on Nvidia H100 + AMD MI300X + Intel Gaudi.
See `scripts/demo_e2e.py` for the reproducer. Last CI run: <link>.
```
OR, if any DoD row is unmet, mark explicitly: `Status: 6/7 DoD rows met. Remaining: <row>`. **Do not** claim success that is not proven.

**Verification gate:**
- If claiming success: Phase 3 demo's `artifacts/demo_report.json` has `PASS=true` in the most recent run.
- If not: README says so.

**Rollback:** `git checkout README.md`.

**Phase 7 exit gate:** A reader of the README can determine, in 5 seconds, whether the codebase achieves its stated goal, and either run `python scripts/demo_e2e.py` to verify, or read the explicit list of unmet DoD rows.

---

## Rollback Strategy Summary

| Phase | Files touched | Single-command rollback |
|---|---|---|
| 0 | `CHANGELOG.md`, `benchmarks/README.md`, `docs/test-inventory.md`, 4 stub-comment lines | `git checkout .` (revertible in <1 min) |
| 1 | New `src/bridges/pytorch_xla/stablehlo_to_triton.py` + test | `rm src/bridges/pytorch_xla/stablehlo_to_triton.py` + revert method addition in `bridge_orchestrator.py` |
| 2 | `src/cli/commands/shard.py:97-181, 264-309`, `src/bridges/pytorch_xla/hardware_orchestrator.py:107-186`, `src/bridges/pytorch_xla/pipeline_orchestrator.py:227-234` | `git checkout` these files. Audit-known stubs reappear; tests from Phase 1 still pass. |
| 3 | New `scripts/demo_e2e.py`, new `src/tests/integration/test_hf_demo.py` | `rm` the new files. Pre-existing test suite unchanged. |
| 4 | `.github/workflows/real-hardware.yml`, new `docs/runner-strategy.md` | `git checkout` the workflow, `rm` the doc |
| 5 | `benchmarks/run_benchmarks.py` (additions only), new `benchmarks/results.json` | `git checkout benchmarks/run_benchmarks.py`, `rm benchmarks/results.json` |
| 6 | New `scripts/measure_cross_vendor.py`, `README.md` edits, possibly `docs/cross_vendor.md` | `git checkout` or `rm` |
| 7 | `CHANGELOG.md`, `README.md`, `docs/ARCHITECTURE.md` (or `TECH_SPEC.md`), `AGENTS.md` | `git checkout` these files |

**Key rollback property:** No phase deletes the audit-confirmed real code. The deprecated stubs (linker.py:362, builder.py:423) are not removed, only the active `_generate_shard_source` in shard.py and hardware_orchestrator.py are deleted. If a phase fails verification, revert the deletions and the audit's pre-existing stubs re-appear as the working state.

---

## Parallelization Matrix

Tasks at the same depth are independent and can be worked on by different agents in parallel:

| Depth | Phases | Parallel? | Reason |
|---|---|---|---|
| 0 | Phase 0 (Truth Reconciliation) | — | First; only docs |
| 1 | Phase 1 (StableHLO→Triton) | — | Blocks Phase 2 and 3 |
| 2 | Phase 2 (shard wiring), Phase 4 (CI runner), Phase 5 (benchmarks) | **Yes** | Phase 2 needs Phase 1; Phase 4 and 5 are independent infrastructure work |
| 3 | Phase 3 (DoD demo), Phase 6 (cross-vendor/Apple) | **Yes** | Phase 3 needs Phase 2; Phase 6 is independent |
| 4 | Phase 7 (final honesty) | — | After everything else; produces the final README/CHANGELOG |

**Practical recommendation:** With 3 sub-agents:
- Agent A: Phase 1 (the largest; unblocks the rest).
- Agent B: Phase 4 (cloud runner setup) + Phase 5 (benchmarks) in parallel.
- Agent C: Phase 0 (truth reconciliation) → Phase 6 → Phase 7 (mostly docs).

After Phase 1 is done, Agent A picks up Phase 2 then Phase 3 (the DoD). Agents B and C continue independently.

---

## Risk Classification (audit-aligned)

| Phase | Audit-claimed priority | This plan's priority | Justification |
|---|---|---|---|
| 0 | P1 | P1 | Docs only; doesn't move the goal needle but is required to measure progress. |
| 1 | P0 (implicit; it's the foundation for the audit's #1 ship-blocker) | **P0** | The audit's #1 finding is that shard.py emits a hand-written matmul template. Phase 1 builds the real translator. |
| 2 | **P0** (audit line 122: "shard→fat binary gap") | **P0** | Direct fix of audit's first ship-blocker. |
| 3 | **P0** (audit line 122: "no end-to-end HuggingFace-model-to-sharded-fat-binary demo") | **P0** | The Definition of Done. |
| 4 | **P0** (audit line 122: "real-hardware CI matrix") | **P0** | Verification infrastructure that the DoD depends on. |
| 5 | **P0** (audit line 122: "reproducible benchmark suite that proves ≥30% speedup") | **P0** | Verification infrastructure for the speedup claim. |
| 6 | P1 (audit line 134: "cross-vendor collective gap") and P2 (audit line 135: "Apple reality check") | **P1** | Both are honesty fixes; the goal is achievable without them. |
| 7 | P1 (audit lines 135, 136: "honest CI test count", "version drift isolation honesty") | **P1** | Final honesty pass; affects credibility not functionality. |

**No P2 stretch in this plan.** The audit's P2 items (cross-vendor collective, Apple Silicon) are demoted to P1 in the execution path. Reason: the audit's P2 designation is "stretch"; the user explicitly asked for "real-world ready, not MVP, not fake showcasing" — neither cross-vendor nor Apple is required for "real-world ready" if the goal is achievable on at least one per-vendor path. Phase 6 either proves or removes the claim.

---

## Critical Risks Not in the Audit (caught during planning)

These were not in the audit but are caught by reading the code:

1. **`ci.yml:128` has a typo**: `FatBinaryConfig(..., output_filename=out.name, ...)` — but `FatBinaryConfig` has no `output_filename` field; it has `output_dir`. The build-fat-binary job will currently fail with `TypeError`. **Must be fixed in Phase 0 or earlier.** Verification gate: `python -c "from src.bridges.aot_packager.builder import FatBinaryConfig; import dataclasses; print([f.name for f in dataclasses.fields(FatBinaryConfig)])" | grep output_filename` should return empty.

2. **`tune.py:235-240` bypasses the bridge**: It hardcodes `grid_0=1, grid_1=1, grid_2=1` in the metadata and calls `bridge._tuning_chain()` (private method) instead of `bridge.tune()`. The CLI's `nautilus tune` never uses the real IR capture path. **Phase 7.1 candidate** — but if the audit is to be honored, this should be fixed as part of the "honest test count" pass.

3. **`pipeline_orchestrator.py:140-253` `shard()` is not called by the CLI**: Phase 2 fixes this. But also: the `shard()` method on line 174 calls `self.graph_capture.capture(model=...)` — but the CLI's `_capture_model` (shard.py:184-222) calls `GraphCapture().capture(model_file=str(model_file), ...)`. **Different signatures**: one passes the model object, the other passes a file path. Phase 2 must align these.

4. **`ir_capture.py:138`** uses a `prefix = f"hook_ttgir:{source_hash[:16]}:{source_hash[:16]}"` to search the buffer, but `hooks.py:118` writes `key = f"hook_ttgir:{metadata.get('name', 'unknown')}:{metadata.get('src', '')[:16]}"`. The capture side and the hook side disagree on the key format. **IRCapture will rarely find real captured IR**, which is why it falls back to synthetic metadata at bridge_orchestrator.py:282-284. **Phase 1.3 must also fix this key format.** Verification gate: a `tests/test_ir_capture_key_format.py` that asserts the hook writes a key the IRCapture reader can find.

5. **`comm_backend.py:115-121` builds cross-vendor bridges unconditionally** when the mesh has more than one vendor, but the bridge bandwidth is hardcoded to 64 GB/s. There is no code that actually does the staging. **Phase 6 must either remove the bridge construction or implement it.** Verification gate: the `comm_backend.py:115` block, if kept, must call a real `_stage_across_vendors()` function (which doesn't exist) or be guarded by `if 0:`.

---

## Appendix: Per-File:Line Change Targets (consolidated)

For quick reference during execution. **These are the locations; the work is described in the phase steps above.**

| File | Lines | Phase | Change |
|---|---|---|---|
| `src/cli/commands/shard.py` | 97-181, 264-309 | 2.1 | Replace `_shard_impl`; delete `_generate_shard_source` |
| `src/bridges/pytorch_xla/hardware_orchestrator.py` | 107-186 | 2.2 | Rewrite `_execute_single_shard`, replace `_generate_shard_source` with StableHLO→Triton call |
| `src/bridges/pytorch_xla/pipeline_orchestrator.py` | 227-253 | 2.3 | Add `shard_kernel_sources` to `ShardingResult` |
| `src/bridges/pytorch_xla/stablehlo_to_triton.py` | NEW | 1.2 | The new translator |
| `src/bridges/triton_tvm/bridge_orchestrator.py` | 238-349 | 1.3 | Add `tune_with_stablehlo` method |
| `src/bridges/triton_tvm/ir_capture.py` | 107-166, 138 | 1.3 | Fix key-format mismatch with hooks.py:118 |
| `src/bridges/triton_tvm/backend/hooks.py` | 118 | 1.3 | Align key format with ir_capture.py |
| `src/bridges/pytorch_xla/tests/test_stablehlo_to_triton.py` | NEW | 1.2 | Test the new translator |
| `src/bridges/pytorch_xla/tests/test_hardware_orchestrator.py` | NEW | 2.2 | Test per-shard source is model-specific |
| `src/bridges/pytorch_xla/tests/test_pipeline_orchestrator.py` | NEW | 2.3 | Test pipeline and executor agree |
| `src/tests/integration/test_full_pipeline.py` | NEW TEST | 2.1 | `test_shard_emits_fat_binary_per_shard` |
| `src/tests/integration/test_hf_demo.py` | NEW | 3.2 | Test the HuggingFace demo on CPU |
| `scripts/demo_e2e.py` | NEW | 3.1 | The DoD reproducer |
| `scripts/measure_cross_vendor.py` | NEW | 6.1 | Cross-vendor bandwidth measurement |
| `scripts/compare_benchmarks.py` | NEW | 5.3 | Benchmark regression detector |
| `docs/test-inventory.md` | NEW | 0.1 | Authoritative test count |
| `docs/translator-survey.md` | NEW | 1.1 | StableHLO op coverage survey |
| `docs/runner-strategy.md` | NEW | 4.1 | CI runner decision |
| `benchmarks/results.json` | NEW | 5.2 | Committed real numbers |
| `benchmarks/cross_vendor.json` | NEW | 6.1 | Cross-vendor real numbers |
| `CHANGELOG.md` | 113-118, 143-145 | 0.2, 7.2 | Honest test count; new 0.2.0 section |
| `README.md` | 7-13 | 6.2, 6.3, 7.3 | Remove unproven claims; add DoD status line |
| `AGENTS.md` | "Critical Knowledge" | 7.1 | Document C-API decision |
| `.github/workflows/real-hardware.yml` | 21-97, 9-13 | 4.2, 4.3 | Real runner; daily schedule |
| `.github/workflows/ci.yml` | 91-133, 128 | 0.1, 5.3 | Fix typo, add benchmark job |
| `pyproject.toml` | 158-160 | 3.3 | Add `requires_hf` marker |

---

## Definition of Done — Final Checklist (for the implementer)

Before claiming "real-world ready":

- [ ] `python scripts/demo_e2e.py --model hf-internal-testing/tiny-llama --mesh 1,1` exits 0 on at least one real-GPU host.
- [ ] `shards/shard_*/kernel.fat.o` exists for every shard, starts with `b"\x7fELF"`, and contains model-specific Triton ops (not just `tl.dot`).
- [ ] `python -c "import ctypes; lib=ctypes.CDLL('shards/shard_0/kernel.fat.o'); print(lib.nautilus_detect_vendor())"` returns the matching vendor.
- [ ] Output numerical diff vs. PyTorch reference is `< 1e-2` for fp16/bf16, `< 1e-5` for fp32.
- [ ] GSPMD sharding spec has more than one unique `partition_shape` for the model.
- [ ] `.github/workflows/real-hardware.yml` is green for at least one Nvidia, one AMD, one Intel job.
- [ ] `benchmarks/results.json` has a non-Nvidia cell with `speedup ≥ 1.0` (regression-safe; PRD's 30% target is aspirational).
- [ ] `git grep STUB: TODO src/` returns at most the linker.py:362 and builder.py:423 deprecated stubs (not shard.py or hardware_orchestrator.py).
- [ ] `CHANGELOG.md` "tests pass" count matches the actual `pytest --collect-only` count.
- [ ] `README.md` either proves or removes the cross-vendor-collective and Apple claims.
- [ ] The C-API version-drift claim is either implemented (libnautilus_c_api.so builds and is loaded by production code) or removed from AGENTS.md/README.

If all 11 boxes are checked, the codebase is real-world ready. If any is unchecked, that is the explicit residual debt and must be in the README's "Status" line.

(End of plan)
