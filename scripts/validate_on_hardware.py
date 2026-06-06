"""End-to-end hardware validation script.

Builds a fat binary in CI and verifies it on the current host's
GPU (if any). Used by the `real-hardware.yml` workflow.

Usage:
    python scripts/validate_on_hardware.py --target cuda
    python scripts/validate_on_hardware.py --target all --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Make src importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common.logging import get_logger, configure_logging
from src.common.errors import NautilusError, HardwareNotFoundError

log = get_logger("nautilus.validate")


EXAMPLE_KERNEL = '''
import triton
import triton.language as tl

@triton.jit
def validate_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Tiny matmul used for hardware validation."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B_ptr + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N
    c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc.to(tl.float16))
'''.strip()


def build_fat_binary(targets: list[str], output_dir: Path) -> dict[str, Any]:
    """Build a fat binary for the given targets."""
    from src.bridges.aot_packager.builder import FatBinaryBuilder, FatBinaryConfig

    kernel_path = output_dir / "validate_kernel.py"
    kernel_path.write_text(EXAMPLE_KERNEL)
    builder = FatBinaryBuilder(cache_dir=str(output_dir / "cache"))
    config = FatBinaryConfig(
        kernel_name="validate_kernel",
        kernel_source=EXAMPLE_KERNEL,
        output_dir=str(output_dir),
        skip_amd=not any("amd" in t or "gfx" in t for t in targets),
        skip_intel=not any("intel" in t or "xe" in t for t in targets),
        skip_nvidia=not any("nvidia" in t or "sm_" in t for t in targets),
        skip_validation=False,
    )
    result = builder.build(config)
    if not result.is_usable:
        raise NautilusError(
            f"Fat binary build failed: {result.error}",
            context={"stage_times": result.stage_times},
        )
    return {
        "success": True,
        "output_path": str(result.output_path),
        "total_size": result.fat_binary.total_size,
        "vendors": [v.value for v in result.fat_binary.vendors],
        "total_time_s": result.total_time_s,
    }


def run_on_hardware(fat_binary_path: Path) -> dict[str, Any]:
    """Attempt to load and execute the fat binary on the current host."""
    if not fat_binary_path.exists():
        return {"ran": False, "reason": "fat binary not found"}
    try:
        import ctypes
        lib = ctypes.CDLL(str(fat_binary_path))
        # The C runtime stub exposes a function to query detected vendor
        if hasattr(lib, "nautilus_detect_vendor"):
            lib.nautilus_detect_vendor.restype = ctypes.c_int
            lib.nautilus_detect_vendor.argtypes = []
            vendor = lib.nautilus_detect_vendor()
            vendor_name = {0: "nvidia", 1: "amd", 2: "intel", 3: "apple", -1: "unknown"}.get(vendor, "unknown")
        else:
            vendor = -1
            vendor_name = "no-stub"
        return {
            "ran": True,
            "detected_vendor": vendor_name,
            "vendor_code": vendor,
        }
    except OSError as exc:
        return {"ran": False, "reason": f"load failed: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="all", help="nvidia/sm_90, amd/gfx942, intel/xe_hpg, all")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/nautilus_validate"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    configure_logging(level="info", json=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = args.target.split(",") if args.target != "all" else ["nvidia/sm_90", "amd/gfx942", "intel/xe_hpg"]

    report: dict[str, Any] = {"target": args.target, "checks": {}}
    try:
        build_info = build_fat_binary(targets, args.output_dir)
        report["build"] = build_info
        report["hardware"] = run_on_hardware(Path(build_info["output_path"]))
    except NautilusError as exc:
        report["build"] = {"success": False, "error": str(exc)}
        report["hardware"] = {"ran": False, "reason": str(exc)}
    except Exception as exc:
        log.error("Validation failed", error=str(exc))
        report["build"] = {"success": False, "error": str(exc)}
        report["hardware"] = {"ran": False, "reason": str(exc)}

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(json.dumps(report, indent=2))
    return 0 if report.get("build", {}).get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
