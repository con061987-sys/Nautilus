# Troubleshooting

Common issues and how to resolve them.

## "Triton is not installed"

```
TritonMissingError: Triton is not installed. Install with: pip install triton
```

**Fix:**
```bash
pip install triton==3.0.0
# Or, for GPU support:
pip install -e .[nvidia]
```

## "llvm-spirv not found in PATH"

```
LLVMError: llvm-spirv not found in PATH. Install LLVM (apt install llvm-spirv or brew install llvm).
```

**Fix:**
```bash
# Ubuntu / Debian
sudo apt-get install llvm

# macOS
brew install llvm
# Add to PATH
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
```

## "AOTriton not installed"

```
AOTritonError: AOTriton not installed. Install with: pip install aotriton
```

**Fix:**
```bash
pip install aotriton
# Or with the AMD extra:
pip install -e .[amd]
```

## "lld not found in PATH"

```
LinkingError: lld not found in PATH. Install LLVM (apt install lld / brew install llvm).
```

**Fix:**
```bash
# Ubuntu
sudo apt-get install lld

# macOS (Homebrew)
brew install llvm
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
```

## "No GPU detected on this system"

```
HardwareNotFoundError: No nvidia GPU detected on this system
```

**Diagnose:**
```bash
nautilus verify --target cuda
```

**Possible causes:**
- The Nvidia driver isn't installed
- The user doesn't have permission to read `/dev/nvidia*`
- Running in a container without GPU passthrough

**Fix (driver):** Install the Nvidia driver from your distribution.
**Fix (permissions):** Add yourself to the `video` group or run as root.

## "amdclang++ not found"

```
AOTritonError: AMD fallback requires amdclang++ or hipcc in PATH
```

**Fix:**
```bash
# Install ROCm
sudo apt-get install rocm-dev

# Or use AMD's official installer:
# https://rocm.docs.amd.com/en/latest/install/install.html
```

## "StableHLO export failed"

```
StableHLOExportError: torch_xla.stablehlo.exported_program_to_stablehlo is unavailable
```

**Fix:** Install the right torch_xla version:
```bash
pip install torch_xla==2.4.0 -f https://storage.googleapis.com/libtpu-releases/index.html
```

## Tests are skipped with "Missing deps: triton, tvm, torch_xla"

This is the expected behavior. Nautilus's test suite gates
GPU-dependent tests behind dependency detection. To run those
tests, install the missing dependencies.

## Fat binary build is slow the first time

The first build of a kernel pays the full AOT compilation cost
(Nvidia ~5s, AMD ~10s, Intel ~8s). Subsequent builds with
identical inputs are instant (disk cache).

## "Pytest reports 'plugin validation error' for src/bridges/conftest.py"

Fixed in the latest release. The per-bridge conftest.py was
removed; the test runner now uses pytest's `--ignore` flag to
skip per-bridge tests when deps are missing.

## "How do I add support for vendor X?"

See [Contributing](CONTRIBUTING.md) → "Adding a new vendor."
The pattern is:

1. Add `VENDOR_xxx` and `ARCH_xxx` enums to `src/common/types.py`
2. Implement `<x>_backend.py` (follow the pattern of `nvidia_backend.py`)
3. Add a detection function in `src/common/hardware.py`
4. Update the C runtime stub in `runtime_stub.c`
5. Update `FatBinaryBuilder` in `builder.py`
6. Add a CI self-hosted runner

## Getting help

- File an issue: https://github.com/nvindia-cud/NVINDIA_CUD/issues
- Check the docs: [docs/](.)
- Run `nautilus verify` for environment diagnostics
