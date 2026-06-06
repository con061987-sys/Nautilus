#!/usr/bin/env bash
# scripts/setup-cuda.sh — Set up a CUDA development environment for Nautilus.
#
# Idempotent: can be re-run safely. Verifies each step. Does NOT
# install the CUDA driver (that's a system-level concern).
#
# Usage:
#   ./scripts/setup-cuda.sh
#   CUDA_VERSION=12.4 ./scripts/setup-cuda.sh
#   NAUTILUS_VENV=/path/to/venv ./scripts/setup-cuda.sh
#
# Exits 0 on success, non-zero with a clear message on any failure.

set -euo pipefail

readonly NAUTILUS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly CUDA_VERSION="${CUDA_VERSION:-12.4}"
readonly NAUTILUS_VENV="${NAUTILUS_VENV:-${NAUTILUS_ROOT}/.venv}"
readonly PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
readonly NAUTILUS_EXTRAS="${NAUTILUS_EXTRAS:-nvidia tuning sharding}"

log() { echo "[setup-cuda] $*" >&2; }
fail() { log "FATAL: $*"; exit 1; }
require() {
    command -v "$1" >/dev/null 2>&1 || fail "Required tool '$1' not found in PATH"
}

log "Nautilus root: $NAUTILUS_ROOT"
log "CUDA version:  $CUDA_VERSION"
log "Venv path:     $NAUTILUS_VENV"
log "Extras:        $NAUTILUS_EXTRAS"

# 1. Prerequisite checks --------------------------------------------------------
log "Checking prerequisites..."
require python3
require pip3
require git

PYTHON_MINOR="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
log "  Python: $PYTHON_MINOR"
if [[ "$(printf '%s\n' "$PYTHON_MINOR" "3.10" | sort -V | head -1)" != "3.10" ]]; then
    fail "Python >= 3.10 required, got $PYTHON_MINOR"
fi

# 2. CUDA toolkit (driver must already be installed) -----------------------------
if command -v nvcc >/dev/null 2>&1; then
    NVCC_VERSION="$(nvcc --version | tail -1 | awk '{print $NF}')"
    log "  nvcc:      $NVCC_VERSION (CUDA $NVCC_VERSION)"
    case "$NVCC_VERSION" in
        12.*) log "  CUDA 12.x detected — supported" ;;
        11.*) log "  CUDA 11.x detected — supported but not the default target" ;;
        *)    log "  WARNING: CUDA $NVCC_VERSION not explicitly tested" ;;
    esac
else
    log "  WARNING: nvcc not found. If you only run pre-built kernels, this is fine."
    log "  To compile from source, install CUDA toolkit $CUDA_VERSION:"
    log "    https://developer.nvidia.com/cuda-$CUDA_VERSION-download-archive"
fi

# 3. lld (LLVM linker) ----------------------------------------------------------
if command -v lld >/dev/null 2>&1 || command -v ld.lld >/dev/null 2>&1; then
    log "  lld:       found"
else
    log "  lld:       NOT FOUND. Install LLVM:"
    log "    Ubuntu:  sudo apt install lld"
    log "    macOS:   brew install llvm"
fi

# 4. spirv-val (for Intel SPIR-V validation) ------------------------------------
if command -v spirv-val >/dev/null 2>&1; then
    log "  spirv-val: found"
else
    log "  spirv-val: NOT FOUND. Install SPIRV-Tools:"
    log "    Ubuntu:  sudo apt install spirv-tools"
fi

# 5. Python virtualenv ----------------------------------------------------------
if [[ ! -d "$NAUTILUS_VENV" ]]; then
    log "Creating venv at $NAUTILUS_VENV..."
    python3 -m venv "$NAUTILUS_VENV"
fi
# shellcheck disable=SC1091
source "$NAUTILUS_VENV/bin/activate"
log "  venv:      $NAUTILUS_VENV ($(python3 --version))"

# 6. Upgrade pip ----------------------------------------------------------------
log "Upgrading pip..."
pip install --quiet --upgrade pip setuptools wheel

# 7. Install Nautilus + extras ---------------------------------------------------
log "Installing nautilus with extras: $NAUTILUS_EXTRAS"
pip install --quiet -e "$NAUTILUS_ROOT[$NAUTILUS_EXTRAS]"

# 8. Verify environment ---------------------------------------------------------
log "Running environment verification..."
python "$NAUTILUS_ROOT/scripts/verify_env.py" --target cuda

log "Setup complete. Activate with: source $NAUTILUS_VENV/bin/activate"
log "Try: nautilus --help"
