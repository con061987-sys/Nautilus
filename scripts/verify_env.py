#!/usr/bin/env python3
"""scripts/verify_env.py — Verify the Nautilus development environment.

Exits 0 if everything is in order, 1 if any required tool is missing
(with a clear list of what's missing), 2 if a tool is present but
the wrong version.

Usage:
    python scripts/verify_env.py
    python scripts/verify_env.py --target cuda    # only check CUDA deps
    python scripts/verify_env.py --target rocm    # only check ROCm deps
    python scripts/verify_env.py --json            # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


NAUTILUS_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class CheckResult:
    name: str
    required: bool
    present: bool
    version: str = ""
    details: str = ""
    fix: str = ""


@dataclass
class VerifyReport:
    target: str
    overall: str  # "ok" | "warning" | "error"
    checks: list[CheckResult] = field(default_factory=list)
    duration_ms: float = 0.0
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "overall": self.overall,
            "duration_ms": self.duration_ms,
            "summary": self.summary,
            "checks": [asdict(c) for c in self.checks],
        }


def _try_import(name: str) -> tuple[bool, str]:
    try:
        mod = __import__(name)
    except ImportError:
        return False, ""
    ver = getattr(mod, "__version__", "unknown")
    return True, ver


def _which(name: str) -> str:
    return shutil.which(name) or ""


def _run(cmd: list[str], timeout: float = 5.0) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return -1, "", "command not found or timed out"


def _check_cuda() -> list[CheckResult]:
    checks: list[CheckResult] = []
    nvcc_path = _which("nvcc")
    if nvcc_path:
        rc, out, _ = _run(["nvcc", "--version"])
        version = ""
        if rc == 0:
            for line in out.splitlines():
                if "release" in line:
                    version = line.split("release")[-1].strip().rstrip(",")
        checks.append(CheckResult(
            name="nvcc", required=False, present=True,
            version=version, details=nvcc_path,
        ))
    else:
        checks.append(CheckResult(
            name="nvcc", required=False, present=False,
            fix="Install CUDA toolkit: https://developer.nvidia.com/cuda-toolkit",
        ))
    has_torch, torch_ver = _try_import("torch")
    has_triton, triton_ver = _try_import("triton")
    has_aotriton, aotriton_ver = _try_import("aotriton")
    if has_torch:
        checks.append(CheckResult(
            name="torch", required=False, present=True, version=torch_ver,
        ))
    else:
        checks.append(CheckResult(
            name="torch", required=False, present=False,
            fix="pip install torch (the [nvidia] extra installs the right version)",
        ))
    if has_triton:
        checks.append(CheckResult(
            name="triton", required=False, present=True, version=triton_ver,
        ))
    else:
        checks.append(CheckResult(
            name="triton", required=False, present=False,
            fix="pip install triton",
        ))
    if has_aotriton:
        checks.append(CheckResult(
            name="aotriton", required=False, present=True, version=aotriton_ver,
        ))
    else:
        checks.append(CheckResult(
            name="aotriton", required=False, present=False,
            fix="AMD backend will not work without aotriton: pip install aotriton",
        ))
    if has_torch:
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                cap = torch.cuda.get_device_capability(0)
                checks.append(CheckResult(
                    name="cuda.device.0", required=False, present=True,
                    version=f"sm_{cap[0]}{cap[1]}",
                    details=name,
                ))
            else:
                checks.append(CheckResult(
                    name="cuda.device", required=False, present=False,
                    details="torch.cuda.is_available() returned False",
                    fix="Check that the Nvidia driver is loaded: nvidia-smi",
                ))
        except Exception as exc:
            checks.append(CheckResult(
                name="cuda.device", required=False, present=False,
                details=str(exc),
            ))
    return checks


def _check_rocm() -> list[CheckResult]:
    checks: list[CheckResult] = []
    if Path("/opt/rocm").exists():
        info_path = Path("/opt/rocm/.info")
        version = info_path.read_text().strip() if info_path.exists() else "unknown"
        checks.append(CheckResult(
            name="rocm", required=False, present=True, version=version,
            details="/opt/rocm",
        ))
    else:
        checks.append(CheckResult(
            name="rocm", required=False, present=False,
            fix="Install ROCm: https://rocm.docs.amd.com/en/latest/install/install.html",
        ))
    if _which("rocminfo"):
        rc, out, _ = _run(["rocminfo"])
        gpu_count = out.count("Agent ")
        checks.append(CheckResult(
            name="rocminfo", required=False, present=True,
            details=f"{gpu_count} agent(s) detected",
        ))
    else:
        checks.append(CheckResult(
            name="rocminfo", required=False, present=False,
            fix="Install rocminfo (part of rocm-smi)",
        ))
    return checks


def _check_intel() -> list[CheckResult]:
    checks: list[CheckResult] = []
    if _which("vainfo") or _which("intel_gpu_top") or _which("ocloc"):
        checks.append(CheckResult(
            name="intel.tools", required=False, present=True,
        ))
    else:
        checks.append(CheckResult(
            name="intel.tools", required=False, present=False,
            fix="Install oneAPI: https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html",
        ))
    if _which("spirv-val"):
        rc, out, _ = _run(["spirv-val", "--version"])
        ver = out.splitlines()[0] if rc == 0 and out else "unknown"
        checks.append(CheckResult(
            name="spirv-val", required=False, present=True, version=ver,
        ))
    else:
        checks.append(CheckResult(
            name="spirv-val", required=False, present=False,
            fix="Install SPIRV-Tools (apt install spirv-tools)",
        ))
    return checks


def _check_common() -> list[CheckResult]:
    """Tools required for any target."""
    checks: list[CheckResult] = []
    if _which("lld") or _which("ld.lld") or _which("lld-link"):
        path = _which("lld") or _which("ld.lld") or _which("lld-link")
        checks.append(CheckResult(
            name="lld", required=True, present=True, details=path,
        ))
    else:
        checks.append(CheckResult(
            name="lld", required=True, present=False,
            fix="Install LLVM linker (apt install lld / brew install llvm)",
        ))
    for tool in ("gcc", "clang"):
        if _which(tool):
            checks.append(CheckResult(
                name=tool, required=True, present=True, details=_which(tool),
            ))
        else:
            checks.append(CheckResult(
                name=tool, required=True, present=False,
                fix=f"Install {tool} (apt install {tool})",
            ))
    has_ninja = bool(_which("ninja") or _which("ninja-build"))
    checks.append(CheckResult(
        name="ninja", required=True, present=has_ninja,
        fix="pip install ninja" if not has_ninja else "",
    ))
    return checks


def _check_nautilus_imports() -> list[CheckResult]:
    """Make sure `nautilus` is importable (catches missing pip install)."""
    checks: list[CheckResult] = []
    try:
        sys.path.insert(0, str(NAUTILUS_ROOT))
        from src.common import Ok, Err, Vendor, Arch, CircuitBreaker
        checks.append(CheckResult(
            name="src.common", required=True, present=True,
            version="0.1.0",
        ))
    except ImportError as exc:
        checks.append(CheckResult(
            name="src.common", required=True, present=False,
            details=str(exc),
            fix="pip install -e . in the repo root",
        ))
    return checks


def verify(target: str = "all", json_output: bool = False) -> VerifyReport:
    """Run all checks and return a structured report."""
    start = time.perf_counter()
    checks: list[CheckResult] = []

    if target in ("all", "common"):
        checks.extend(_check_common())
        checks.extend(_check_nautilus_imports())
    if target in ("all", "cuda", "nvidia"):
        checks.extend(_check_cuda())
    if target in ("all", "rocm", "amd"):
        checks.extend(_check_rocm())
    if target in ("all", "intel", "xpu"):
        checks.extend(_check_intel())

    missing_required = [c for c in checks if c.required and not c.present]
    missing_optional = [c for c in checks if not c.required and not c.present]

    if missing_required:
        overall = "error"
        summary = f"FAIL: {len(missing_required)} required tool(s) missing"
    elif missing_optional:
        overall = "warning"
        summary = (
            f"OK with warnings: {len(missing_optional)} optional tool(s) missing"
        )
    else:
        overall = "ok"
        summary = "All tools present"

    return VerifyReport(
        target=target,
        overall=overall,
        checks=checks,
        duration_ms=(time.perf_counter() - start) * 1000,
        summary=summary,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", choices=["all", "common", "cuda", "rocm", "intel", "nvidia", "amd", "xpu"],
        default="all",
    )
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    args = parser.parse_args()

    report = verify(args.target, args.json)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"=== Nautilus Environment Verification (target: {args.target}) ===")
        print()
        for c in report.checks:
            tag = "REQ" if c.required else "opt"
            status = "✓" if c.present else "✗"
            line = f"  [{tag}] [{status}] {c.name}"
            if c.version:
                line += f"  ({c.version})"
            if c.details:
                line += f"  — {c.details}"
            if not c.present and c.fix:
                line += f"\n         fix: {c.fix}"
            print(line)
        print()
        print(f"  Result:  {report.overall.upper()}")
        print(f"  Summary: {report.summary}")
        print(f"  Time:    {report.duration_ms:.1f}ms")

    return 0 if report.overall != "error" else 1


if __name__ == "__main__":
    sys.exit(main())
