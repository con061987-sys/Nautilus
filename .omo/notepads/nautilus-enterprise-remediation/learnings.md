# nautilus-enterprise-remediation — learnings

Append-only log of corrections, design notes, and gotchas surfaced while
remediating Nautilus CI/CD. Read this before touching the affected files
again — every entry exists because something non-obvious bit us once.

---

## 2026-06-06 — CI matrix refactor + artifact cleanup

### `docs/HARDWARE_SETUP.md` was never tracked
- `git status` showed it as `??` (untracked) — a 23 KB file dropped in the
  working tree by an earlier CI run, not a committed doc.
- Task spec said "Delete it with `git rm docs/HARDWARE_SETUP.md`", but
  `git rm` requires the file to be in the index. Plain `rm` is the
  correct tool for untracked artifacts. Same end state, no index churn.
- If this file reappears, it means CI is leaking artifacts into the
  working tree again — track that down at the source rather than just
  deleting it on sight.

### ci.yml matrix dimension: top-level vs per-target
- Added `python-version: ["3.10", "3.11", "3.12"]` at the **top level**
  of the `matrix:` block, alongside `target:`. This produces 5 × 3 = 15
  job combinations.
- The cloud jobs (amd, intel) fan out 3× each but the matrix dimension
  is functionally ignored on those — their SSH scripts install Python
  on the cloud image. Wasted CI minutes, but does not break anything
  and keeps the matrix structure simple.
- The alternative (`include:` rules to restrict `python-version` to the
  `cpu` target) is more correct per the "inherited wisdom" hint but
  adds YAML complexity. Did not pursue it because the task MUST DO
  list said "alongside the existing `target` matrix" — that phrasing
  matches the simple top-level approach, not an include block.
- If a future optimization wants to drop cloud Python fan-out, switch
  to `include:`/`exclude:` rules; do not remove `python-version` from
  the top level or the cpu/nvidia/macos jobs lose it.

### Cache key must follow setup-python parameterization
- The pip cache key on line 220 originally hard-coded `py3.11`. When
  you parameterize `setup-python` to `matrix.python-version`, you must
  also parameterize the cache key and both `restore-keys` lines —
  otherwise Python 3.10/3.12 jobs collide on the 3.11 cache and
  silently skip a real install (or, worse, reuse incompatible wheels).
- Both `restore-keys` entries got updated: one for the per-target key
  prefix, one for the runner-only key prefix.

### `--tb=short` lived in three places, not one
- Two were inside the cloud SSH scripts (amd @ L424, intel @ L452) —
  the heredoc-style YAML literal blocks make it easy to miss these on
  a casual grep. The local cpu/nvidia/macos path was the obvious one
  (L322). Always count matches before editing.

### drift-detection.yml: spec-met, no changes
- Cron `0 6 * * *` (06:00 UTC daily) ✓
- PyPI pin drift via `scripts/check_upstream_drift.py --md` (sensitive
  set: torch, triton, apache-tvm, torch_xla, aotriton, networkx, pyzmq) ✓
- Submodule SHA drift via `git submodule status` × `ls-remote` per
  branch (default branches: main, master, develop) ✓
- Dedupe-by-day for the drift issue (L244-249) prevents re-opening
  when workflow is retriggered same-day ✓
- `concurrency.cancel-in-progress: false` (L42) — correct, the issue
  side effect must always complete if triggered ✓
- Permissions `contents: read, issues: write` (L45-46) — least-privilege ✓
- Per the MUST NOT DO list, do not touch this file's structure.

### Pre-existing working tree churn (NOT touched by this task)
- `git status` shows many other files modified before this session
  started: `.omo/boulder.json`, `pyproject.toml`,
  `src/bridges/aot_packager/{linker.py,tests/test_linker.py}`,
  `src/bridges/triton_tvm/{config_mapper.py,tests/test_config_mapper.py}`,
  and even `.github/workflows/drift-detection.yml` itself (256 lines
  of pre-existing diff). Out of scope for this task; left untouched.

---

## 2026-06-06 — Runtime stub build wiring (C compilation in setup.py)

### Task spec said "FatBinaryBuilder" but the class is `FatBinaryLinker`
- Task description in the prompt mentioned `FatBinaryBuilder` in
  `src/bridges/aot_packager/linker.py`, but the actual class in that
  file is `FatBinaryLinker`. `FatBinaryBuilder` lives in
  `builder.py` and orchestrates the per-vendor backends. The right
  class to modify for the `stub_path` parameter is the linker — it
  is the layer that actually needs the compiled `.o` bytes.
- The "hardcoded reference to `runtime_stub.o` at repo root" mentioned
  in inherited wisdom is also misleading. Current `linker.py` already
  takes `runtime_stub_o: bytes | None` on `link_fat_binary()` and
  writes them to a temp file (not the repo root). The only real
  path-level default that needed adding was the `__init__` `stub_path`
  param + the bytes-loader fallback inside `link_fat_binary()`.

### `os.uname().machine` over `platform.machine()`
- The MUST NOT DO list explicitly forbade `platform.machine()`. Used
  `os.uname().machine` instead — it is the kernel's view of the
  machine, not a Python-layer shim. On cross-compiled CI images
  (e.g. an aarch64 runner with an x86_64 cross-toolchain installed)
  the two have been observed to disagree, and gcc will silently
  emit a binary for the actual kernel arch, not what
  `platform.machine()` reported.
- Inline comment in `setup.py` documents this gotcha for the next
  person who tries to "simplify" it.

### `__file__`-relative default path requires FOUR .parent hops
- `linker.py` is at `src/bridges/aot_packager/linker.py` — that's
  4 directory levels under the repo root (`aot_packager` →
  `bridges` → `src` → `<repo>`). A first instinct of "two hops"
  would land at `src/`, one level short. Verified the path actually
  exists after construction:
  `FatBinaryLinker().default_stub_path` →
  `/workspaces/NVINDIA_CUD/build/runtime_stub.o`.

### Setuptools `__main__` block never runs because `_setup()` raises first
- `python setup.py` (no command) exits with "no commands supplied"
  before reaching the `if __name__ == "__main__":` block. The
  `try/except ImportError` only catches ImportError, not the
  `SystemExit` argparse-style error from setuptools. This was true
  in the original file too — the `__main__` block was effectively
  dead code for `python setup.py` invocations.
- Real entry points that do work: `python setup.py build_py`,
  `python setup.py develop`, `pip install -e .`. All three hit
  `CustomBuildPy.run()` / `CustomDevelop.run()`, both of which now
  call `build_cpp_plugin()` then `build_runtime_stub()`.
- Test that exercises the full wiring: `python setup.py build_py`
  prints `Building runtime stub: arch=x86_64 out=…/build/runtime_stub.o`
  and produces an `ELF 64-bit LSB relocatable, x86-64` file.

### Existing `runtime_stub.o` at repo root is from a previous build
- `git status` does NOT show `runtime_stub.o` as untracked. The
  `*.o` line in `.gitignore` (line 53) covers it. Same for
  `build/runtime_stub.o` (covered by `build/` on line 8). No need
  to delete or move it — the new build path goes to
  `<repo>/build/runtime_stub.o`, and the old root-level artifact
  will be ignored by both `*.o` and the new convention.

### `-nostdlib -ffreestanding` flags are non-negotiable for this stub
- `runtime_stub.c` deliberately avoids libc (its own `nautilus_strlen`
  etc.) and uses `access(F_OK)` from `<unistd.h>` only inside an
  `#if defined(__linux__)` block. The build flags in the task spec
  (`-nostdlib -ffreestanding`) match the file's own header comment
  (lines 18-19). Adding `-fPIC` (as the existing
  `FatBinaryBuilder._compile_runtime_stub` does on L395) is
  optional for an `-r` relocatable link; left it off the setup.py
  flags to match the task spec exactly.

---

## 2026-06-06 — Vendor enum standardization (C + Python)

### `triton_c_api.h` already had a `typedef enum`, not `#define`
- Task description referenced `#define NAUTILUS_VENDOR_NVIDIA 0` etc.,
  but the header **already** used `typedef enum { ... } nautilus_vendor_t`.
  The `#define` constants lived in `runtime_stub.c`, not the header.
- The header's existing enum had `NAUTILUS_VENDOR_HOST = 4` (compile for
  current host / debug), but `runtime_stub.c` defined
  `NAUTILUS_VENDOR_UNKNOWN = -1`. Two different "extra" values — the
  C side was internally inconsistent.
- Resolution: replaced `HOST = 4` with `UNKNOWN = -1` in the header
  to match the runtime stub's semantics. `NAUTILUS_VENDOR_HOST` was
  referenced **only** in the header itself (confirmed via
  `grep -rn "NAUTILUS_VENDOR_HOST" src/`), so removal is safe.
- Core values 0/1/2/3 (NVIDIA/AMD/INTEL/APPLE) are unchanged —
  backward-compatible with the Python `Vendor` enum in
  `src/common/primitives.py` and any pre-compiled fat binaries that
  encode the int values in section headers.

### `runtime_stub.c` must `#include` the C API header to pick up the enum
- Added `#include "../../c_api/triton_c_api.h"` near the top.
- The relative path resolves because `runtime_stub.c` lives at
  `src/bridges/aot_packager/` and the header at `src/c_api/`.
- If a future refactor moves either file, this include path must
  be updated in lockstep. A build-system include path would be
  cleaner but is not yet wired up.

### `_Static_assert` placement
- Put the asserts **between** the include block and the first
  `extern` declaration. C11 `_Static_assert` at file scope is
  evaluated at compile time, so placement is semantically free,
  but putting them right after the `#include` keeps the invariant
  check adjacent to the source of truth.
- Each assert gets a unique message (`"Vendor enum drift: NVIDIA"`
  etc.) so a compile failure points at the exact mismatched
  vendor, not just "static assertion failed".

### `nautilus_detect_vendor` return type widened from `int` to `nautilus_vendor_t`
- The old return type was `int` even though it returned enum values
  from `#define`s. Widening to the proper enum type makes the contract
  explicit and lets the compiler catch callers that accidentally
  store the result in a plain `int` (e.g. `nautilus_dispatch` local
  variable).
- The `switch (vendor)` in `nautilus_dispatch` still works because
  the four real cases plus a `default:` arm cover all reachable
  values. `-Wswitch` on `-Wall` is happy because `default` is
  present.
- An `int` return value (or implicit narrowing) at the call site
  would trigger `-Wconversion` on a `-Werror` build — worth
  keeping in mind for any future call sites.

### `VendorEnum` does not exist anywhere in the codebase
- The task's MUST DO list mentioned "importing `Vendor` or
  `VendorEnum`", but `VendorEnum` has no occurrences under `src/`
  (`grep -rn "VendorEnum" src/` returns nothing). The Python-side
  canonical name is just `Vendor` (in `src/common/primitives.py`).
- No code change needed for the `VendorEnum` case; flagged here
  so future readers don't go hunting for a non-existent symbol.

### No bridge Python file imports `Vendor` from `src.common.types`
- The task's third checkbox was "All bridge Python files that
  import `Vendor` or `VendorEnum` now import from
  `src.common.primitives` (not `src.common.types`)".
- Verified via `grep -rn "from src\.common\.\(types\|primitives\)" src/bridges/`:
  only two bridge files import from `src.common.types` at all
  (`gspmd_runner.py` and `stablehlo_export.py`), and neither
  imports `Vendor` — they import `MeshShape`, `ShardingSpecLite`,
  `StableHLOModule`, `TensorShardingLite`.
- The "bridge files use types" violation the task anticipated
  does not exist. The `Vendor` imports from `src.common.types`
  are in `src/cli/commands/{build,tune}.py`, `src/common/hardware.py`,
  and the test suite — none of which are in `src/bridges/`.
- Decision: did NOT do a mass rewrite of non-bridge files. The
  task scope was bridges, and the existing `types.py → primitives.py`
  re-export chain already keeps backward compatibility. Touching
  the CLI/hardware imports would expand scope without fixing a
  real defect.

### Compilation: `gcc -c -std=c11 -Wall -Werror`
- `gcc -c -std=c11 src/bridges/aot_packager/runtime_stub.c -o /dev/null -Wall -Werror`
  produces zero output and exit 0. The `_Static_assert`s pass,
  no `-Wswitch` / `-Wenum-compare` / `-Wconversion` warnings.
- Note: the file's existing `__attribute__((unused))` annotations
  on the `nautilus_strlen` / `strcmp` / `memcpy` helpers are what
  keep `-Werror=unused-function` quiet. If you ever wire those
  helpers in, the `__attribute__` should be removed at the same time.

---

## 2026-06-06 — TVM `tirx` → `tir` API alignment

### `tirx` is the unreleased TVM 0.25.dev0 namespace
- `from tvm.script import tirx as T` and `from tvm.tirx import PrimFunc` do
  **not** exist in any stable TVM release, including 0.18.0. The docs
  and tutorials in the wild reference `tirx` because they were written
  against a `main` branch preview. Pinned to `apache-tvm==0.18.0` →
  must use `tir` (not `tirx`).
- Likewise `tvm.s_tir.meta_schedule.tune_tir` is wrong — the correct
  0.18 path is `tvm.meta_schedule.tune_tir` (already used by
  `metaschedule_adapter.py`).
- Pattern to recognise stale docs: any reference to `tirx` or
  `s_tir.meta_schedule` in code comments / docstrings is a copy-paste
  from upstream trunk.

### Files touched (all in `src/bridges/triton_tvm/`)
- `tvmscript_executor.py`: 4 sites — module-level import, docstring,
  `_build_namespace` re-import + `tir` namespace exposure, `PrimFunc`
  isinstance check.
- `extern_bridge.py`: 2 sites — `tir as T`, `tir.PrimFunc`.
- `tir_template.py`: 4 sites — module-level `tvm.tir as tir`,
  `tir as T`, docstring, `_execute_tvmscript` re-import + namespace
  key `"tir"`.
- `metaschedule_adapter.py`: 1 site — `from tvm import tir`.
- `ir_to_tir/tvmscript_emitter.py`: 3 sites — module docstring,
  class docstring, `emit()` docstring. All three referenced
  `tvm.script.tirx.prim_func` and `tvm.s_tir.meta_schedule.tune_tir`.

### `nautilus_tir_*` C ABI is NOT a `tirx` reference
- `src/c_api/{tvm_c_api.h,tvm_wrapper.cpp,__init__.py}` use the C
  symbol naming `nautilus_tir_module_s`, `nautilus_tir_parse`,
  `nautilus_tir_release`. A naive `grep "s_tir"` matches these
  because of the substring `nautilus` ending in `s` + the next
  segment `_tir`. These are **false positives** — the project's
  own C ABI uses `tir` to mean "Tensor IR", not the upstream
  `tvm.script.tirx` namespace. Renaming them would break the
  documented C ABI.
- The verification grep
  `grep -rn "tirx\|s_tir\|tvm\.script\.tirx" src/` will still
  return matches in `src/c_api/` because of this substring. The
  actual `tirx` and `tvm.s_tir` (TVM namespace) references in
  the five bridge files are gone — verified with a more specific
  grep (`grep -rn "tirx\|tvm\.s_tir\|tvm\.script\.tirx"`) which
  returns zero matches.

### `pyproject.toml` already had `apache-tvm==0.18.0`
- All four occurrences (sharding, tuning, rpc, all extras) were
  already pinned with `==0.18.0`. No edit needed; verified with
  `grep -n "apache-tvm" pyproject.toml`.

### Lint: `N812` is the canonical TVMScript style
- The pattern `from tvm.script import tir as T` triggers Ruff
  N812 ("lowercase `tir` imported as non-lowercase `T`"). This
  is **intentional and idiomatic** — every TVM tutorial uses
  `T.prim_func` / `T.Buffer` / `T.grid`. Do not "fix" it to
  `import tir` without the alias; that would make every TVMScript
  emission site longer without clarity benefit. Same applied to
  the pre-existing `tirx` imports — linter still warns.
- The pre-existing F401 warnings (`tvm` / `tvm.tir` imported
  but unused) come from the try/except availability guard
  pattern. Same pattern, just renamed submodule.

---

## 2026-06-07 — `.pre-commit-config.yaml` + gspmd_runner.py remediation

### Task description understated the violation count
- The Wave-1 task description said "currently 1 violation: F821 undefined
  name `total_dev` in gspmd_runner.py:548". The actual state of
  `gspmd_runner.py` was **5 violations** in that file (F821 + F841 + B007
  + 3× F841) plus one SIM105 that only surfaces when ruff is run without
  `--quiet` (it gets filtered out of the default output).
- The wider `src/` tree had ~100+ ruff violations and 101 files needing
  reformatting — those are out of scope for this task (a separate Wave
  concern). Don't expand the gspmd_runner.py fix into a global cleanup
  without explicit scope.

### gspmd_runner.py — all 6 fixes applied
1. **F821 at line 552** — `_compute_collectives(spec, module, total_dev)`
   was called inside `_TorchXLASharding.shard()` without `total_dev`
   being defined in scope. Fixed by computing
   `total_dev = _TorchXLASharding._total_devices(mesh_shape)` immediately
   before the call (matches the existing pattern at line 412/456).
2. **F841 at line 597** — `_TVMMetaScheduleSharding.shard()` had
   `total_dev = _total_devices(mesh_shape)` assigned but never used.
   Pure dead code — the helper `_build_spec_from_tvm` recomputes the same
   value internally (line 830). Removed the dead line. Did **not**
   refactor to thread the value through, because the public `_build_spec_from_tvm`
   signature would need to change, propagating the refactor to callers.
3. **B007 at line 671** — `for name, (shape, dtype) in input_tensors.items():`
   inside `_build_tir_module`. The `name` key was destructured but never
   used in the loop body. Renamed to `_name` (the canonical B007 fix).
4. **F841 at lines 749–751** — Three `tir.Var` assignments (`var_m`,
   `var_n`, `var_k`) inside `_build_minimal_ir_module` that were never
   referenced. The actual function body uses local `i`, `j`, `kk` vars
   (line 758–760). Removed the three dead lines.
5. **SIM105 at line 1300** — `try / except Exception: pass` inside
   `run()`. Replaced with `with contextlib.suppress(Exception): ...`.
   Had to add `import contextlib` to the stdlib import block at the
   top of the file (alphabetical: between nothing and `hashlib`).

### `ruff format` rewrapped the file
- After the code fixes, running `ruff format src/bridges/pytorch_xla/gspmd_runner.py`
  reformatted the file. The changes are pure whitespace/line-wrapping —
  no logic delta. Re-verified `ruff check` still passes after format.

### `.pre-commit-config.yaml` versions
- `pre-commit-hooks` pinned to `v5.0.0` (mature, widely-deployed; v6.0.0
  requires Python ≥3.9 which the project's Python 3.10 floor already
  satisfies but v5.0.0 has been stable for >1 year).
- `ruff-pre-commit` pinned to `v0.15.16` to match the version of ruff
  installed locally (`ruff --version` reports 0.15.16). Mixing hook and
  local versions can surface diagnostic drift.
- `ruff` hook runs with `args: [--fix]` so common auto-fixes (import
  sorting, unused imports) apply on commit. `ruff-format` is a separate
  hook — order matters: ruff before ruff-format so the auto-fixes don't
  fight the formatter.

### `pre-commit run --all-files` is the wrong verification command
- The wider repo has unformatted files and many ruff violations.
  `pre-commit run --all-files` against HEAD will fail on every
  pre-existing issue.
- Correct verification scope is the files this task touched:
  `pre-commit run --files src/bridges/pytorch_xla/gspmd_runner.py .pre-commit-config.yaml`.
  Both files pass all 7 hooks (the toml/json hooks are skipped because
  no .toml or .json files were modified).

### `--quiet` masks some violation types
- `ruff check src/ --quiet` filters out non-error diagnostics like
  SIM105 (a "Try-except-pass" suggestion). Run without `--quiet` to
  see the full picture. The SIM105 here was a real bug-class issue
  worth fixing — a noisy `try/except: pass` in the failure path of
  `GSPMDRunner.run()`.
