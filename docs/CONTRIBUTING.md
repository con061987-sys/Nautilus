# Contributing to Nautilus

Thanks for your interest in making AI compilation cross-vendor.
This document describes how to add code, run tests, and ship
changes.

## Setup

```bash
git clone https://github.com/nvindia-cud/NVINDIA_CUD
cd nautilus
./scripts/setup-cuda.sh   # or setup-rocm.sh / setup-intel.sh
pip install -e .[dev]
pre-commit install
```

## Code style

- **Python**: PEP 8, type hints on all public functions, Google-style
  docstrings, 100-column limit. `ruff check` and `ruff format` are
  enforced by pre-commit.
- **C++**: LLVM coding style, `llvm::Expected<T>` for errors, RAII
  for resources, no raw pointers in public APIs.
- **C**: K&R-style, `-Wall -Werror`, nostdlib-compatible.
- **Shell**: `set -euo pipefail`, `readonly` for constants, prefer
  bash over sh for portability.

## Branching & commits

- Branch from `main` with a descriptive name (`feat/intel-spirv-fastpath`)
- One logical change per commit
- Use conventional commit prefixes:
  - `feat:` new feature
  - `fix:` bug fix
  - `refactor:` code motion without behavior change
  - `perf:` performance improvement
  - `test:` test-only change
  - `docs:` documentation only
  - `ci:` CI / workflow change
  - `chore:` tooling / dependency change

## Architecture invariants

These are non-negotiable. CI will fail if violated.

### 1. No silent placeholders

If a function can't do its job due to a missing dependency, it
MUST raise a clear `DependencyMissingError` (or subclass) with an
install hint. NEVER return a stub, empty bytes, zero, or None
as a proxy for "I don't know."

```python
# WRONG
def has_nvidia_gpu() -> bool:
    return 0  # Stub

# RIGHT
def has_nvidia_gpu() -> bool:
    if not Path("/dev/nvidia0").exists():
        return False  # Real probe
    return True
```

```python
# WRONG
def compile_kernel(...) -> bytes:
    return self._write_placeholder()  # 64-byte ELF stub

# RIGHT
def compile_kernel(...) -> bytes:
    if not shutil.which("aotriton"):
        raise AOTritonError("aotriton not found; install with: pip install aotriton")
    return self._run_aotriton(...)
```

### 2. All errors are typed

Use the `src/common/errors.py` hierarchy. Every error has a stable
string code (so it can cross C-Python boundaries, log aggregation
systems, and version boundaries).

### 3. All public types are in `src.common.types`

Bridges don't define their own Vendor/Arch/TuningConfig — they
use the ones in `src.common`. This eliminates the cross-bridge
incompatibility that previously had pytorch_xla importing
triton_tvm internals.

### 4. Cross-bridge coupling is forbidden

If you need `circuit_breaker` or `timeout_manager` from another
bridge, import from `src.common.observability` — not from
`src.bridges.triton_tvm.circuit_breaker`.

### 5. Hardware probing is real

Every hardware detection function (`has_nvidia_gpu`,
`detect_gpu_vendors`, etc.) does actual `/dev` or `lspci` or
`system_profiler` probing. No "Unknown" returns; if the probe
fails, raise `HardwareProbeError` with the actual error.

## Testing

```bash
pytest                              # all tests
pytest src/common/                 # core tests
pytest src/tests/integration/      # cross-bridge tests
pytest -m gpu                       # GPU-required tests
pytest -m "not gpu and not slow"     # fast CI subset
```

### Markers

| Marker | Meaning |
|--------|---------|
| `gpu` | requires real GPU hardware |
| `cuda` / `rocm` / `intel` | vendor-specific |
| `slow` | > 10s runtime |
| `integration` | cross-bridge |
| `requires_deps` | requires optional ML dependencies |

### Test organization

- Unit tests live next to the code: `src/bridges/<x>/tests/`
- Cross-bridge integration tests: `src/tests/integration/`
- Hardware-only tests: opt-in with `@pytest.mark.gpu` so they
  skip on machines without the hardware

## Adding a new backend

To add support for a new vendor (e.g. Tenstorrent, Graphcore IPU):

1. Add a new `VENDOR_xxx` and `ARCH_xxx` enum to `src/common/types.py`
2. Implement the AOT backend in `src/bridges/aot_packager/xxx_backend.py`:
   - Compile Triton → vendor native binary
   - Validate the output (size, format, sanity)
   - Raise `CompilationError` on failure
   - Cache results on disk
3. Add a hardware detection function in `src/common/hardware.py`:
   - Real `/dev` or vendor-specific probe
   - Raise `HardwareNotFoundError` if vendor not present
4. Update `FatBinaryBuilder` in `builder.py` to use the new backend
5. Update the C runtime stub in `runtime_stub.c` with a detection function
6. Update the `nautilus` CLI's `--target` parser
7. Add a CI self-hosted runner label
8. Add an integration test in `src/tests/integration/`
9. Update `docs/USER_GUIDE.md`

## Adding a new CLI command

CLI commands live in `src/cli/commands/<name>.py`. They:

1. Use `@click.command()` decorator
2. Have a docstring with examples
3. Catch `NautilusError` and print a clean error
4. Use `src/common/logging` for structured logs
5. Use `src/common/errors` for error types
6. Are registered in `src/cli/main.py` via `cli.add_command()`

## Reviewing PRs

Each PR should be reviewed by:

- One bridge expert (for bridge changes)
- One infrastructure expert (for C-API, CLI, runtime changes)
- One ML expert (for tuning, GSPMD, math changes)

Use the [requesting-code-review](../.opencode/skills/requesting-code-review/SKILL.md)
skill for the review process.

## Release process

- Bump version in `pyproject.toml`
- Update `CHANGELOG.md`
- Tag `v0.X.Y` and push
- GitHub Actions builds wheels and publishes to PyPI
