#!/usr/bin/env bash
# scripts/setup-rocm.sh — Set up an AMD ROCm development environment for Nautilus.
#
# Idempotent. Mirrors setup-cuda.sh but for ROCm.

set -euo pipefail

readonly NAUTILUS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROCM_VERSION="${ROCM_VERSION:-6.0}"
readonly NAUTILUS_VENV="${NAUTILUS_VENV:-${NAUTILUS_ROOT}/.venv}"
readonly PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
readonly NAUTILUS_EXTRAS="${NAUTILUS_EXTRAS:-amd tuning sharding}"

log() { echo "[setup-rocm] $*" >&2; }
fail() { log "FATAL: $*"; exit 1; }
require() { command -v "$1" >/dev/null 2>&1 || fail "Required tool '$1' not found in PATH"; }

log "Nautilus root: $NAUTILUS_ROOT"
log "ROCm version:  $ROCM_VERSION"
log "Venv path:     $NAUTILUS_VENV"
log "Extras:        $NAUTILUS_EXTRAS"

log "Checking prerequisites..."
require python3
require pip3
require git

# 1. ROCm installation ---------------------------------------------------------
if [[ -d "/opt/rocm" ]]; then
    if [[ -f "/opt/rocm/.info" ]]; then
        ROCM_INSTALLED="$(cat /opt/rocm/.info)"
        log "  ROCm:      $ROCM_INSTALLED (at /opt/rocm)"
    else
        log "  ROCm:      installed at /opt/rocm (version unknown)"
    fi
else
    log "  WARNING: /opt/rocm not found. Install ROCm $ROCM_VERSION:"
    log "    https://rocm.docs.amd.com/en/docs-$ROCM_VERSION/install/install.html"
fi

# 2. AOTriton (AMD's ahead-of-time Triton compiler) ----------------------------
if python3 -c "import aotriton" 2>/dev/null; then
    log "  aotriton:  $(python3 -c 'import aotriton; print(aotriton.__version__)')"
else
    log "  aotriton:  NOT FOUND. Install with: pip install aotriton"
fi

# 3. lld ------------------------------------------------------------------------
if command -v lld >/dev/null 2>&1 || command -v ld.lld >/dev/null 2>&1; then
    log "  lld:       found"
else
    log "  lld:       NOT FOUND. Install LLVM (lld)."
fi

# 4. spirv-val (Intel; AMD uses hsaco but you still want SPIR-V for tooling) ---
if command -v spirv-val >/dev/null 2>&1; then
    log "  spirv-val: found"
else
    log "  spirv-val: NOT FOUND. Install spirv-tools."
fi

# 5. venv ------------------------------------------------------------------------
if [[ ! -d "$NAUTILUS_VENV" ]]; then
    log "Creating venv at $NAUTILUS_VENV..."
    python3 -m venv "$NAUTILUS_VENV"
fi
source "$NAUTILUS_VENV/bin/activate"
log "  venv:      $NAUTILUS_VENV ($(python3 --version))"

# 6. Install --------------------------------------------------------------------
log "Upgrading pip..."
pip install --quiet --upgrade pip setuptools wheel

log "Installing nautilus with extras: $NAUTILUS_EXTRAS"
pip install --quiet -e "$NAUTILUS_ROOT[$NAUTILUS_EXTRAS]"

# 7. Verify ---------------------------------------------------------------------
log "Running environment verification..."
python "$NAUTILUS_ROOT/scripts/verify_env.py" --target rocm

log "Setup complete. Activate with: source $NAUTILUS_VENV/bin/activate"
log "Try: nautilus --help"
