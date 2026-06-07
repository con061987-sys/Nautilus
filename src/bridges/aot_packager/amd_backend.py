"""AMD AOT backend — REAL AOTriton or Triton-emitted compilation, not a placeholder.

The previous design wrote a 64-byte ELF stub with a comment "for now".
This rewrite:

  1. Tries AOTriton first (AMD's official AOT Triton compiler)
  2. Falls back to Triton-emitted cubin -> SPIR-V via amdclang++
  3. Validates the output (size > 100 bytes, real amdgcn target)
  4. Raises clear errors (AOTritonError, CompilationError) on failure
  5. Caches successful compilations
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from src.common.errors import (
    AOTritonError,
    CompilationError,
    CompilationOutputMissingError,
    DependencyMissingError,
)
from src.common.logging import get_logger

from ._signature_inference import build_signature

log = get_logger("nautilus.aot.amd")


class AMDArch(str, Enum):
    GFX900 = "gfx900"
    GFX906 = "gfx906"
    GFX908 = "gfx908"
    GFX90A = "gfx90a"
    GFX942 = "gfx942"
    GFX950 = "gfx950"


@dataclass
class AMDCompilationResult:
    success: bool
    arch: str = "gfx942"
    hsaco_path: Path | None = None
    hsaco_bytes: bytes | None = None
    error: str | None = None
    compilation_time_s: float = 0.0
    cache_hit: bool = False
    used_aotriton: bool = False
    aotriton_version: str = ""

    @property
    def is_usable(self) -> bool:
        return self.success and self.hsaco_bytes is not None and len(self.hsaco_bytes) > 100


class AMDBackend:
    """AOT compilation for AMD ROCm GPUs via AOTriton or Triton+amdclang++."""

    def __init__(
        self,
        target_arch: AMDArch = AMDArch.GFX942,
        cache_dir: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.target_arch = target_arch
        self.cache_dir = Path(
            cache_dir
            or os.environ.get(
                "NAUTILUS_AMD_CACHE",
                str(Path.home() / ".cache" / "nautilus" / "amd"),
            )
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds
        self._aotriton_version = self._detect_aotriton()
        self._amdclangpp_path = shutil.which("amdclang++") or shutil.which("hipcc") or ""
        self._lock = threading.Lock()

    def compile_kernel(
        self,
        kernel_source: str,
        kernel_name: str,
        block_m: int = 128,
        block_n: int = 128,
        block_k: int = 32,
        num_warps: int = 8,
        num_stages: int = 3,
    ) -> AMDCompilationResult:
        """Compile a Triton kernel to HSACO for the target AMD GPU.

        Raises:
            AOTritonError / DependencyMissingError: if AOTriton unavailable.
            CompilationError: if compilation fails.
            CompilationOutputMissingError: if no output is produced.
        """
        start = time.perf_counter()

        cache_key = self._compute_cache_key(
            kernel_source,
            kernel_name,
            block_m,
            block_n,
            block_k,
            num_warps,
            num_stages,
        )
        cached = self._check_cache(cache_key)
        if cached is not None:
            return AMDCompilationResult(
                success=True,
                arch=self.target_arch.value,
                hsaco_path=cached,
                hsaco_bytes=cached.read_bytes(),
                compilation_time_s=time.perf_counter() - start,
                cache_hit=True,
                used_aotriton="aotriton" in self._aotriton_version.lower(),
                aotriton_version=self._aotriton_version,
            )

        used_aotriton = False
        try:
            hsaco_bytes = self._run_aotriton(
                kernel_source,
                kernel_name,
                block_m,
                block_n,
                block_k,
                num_warps,
                num_stages,
            )
            used_aotriton = True
        except DependencyMissingError as exc:
            log.info("AOTriton unavailable, falling back to Triton+amdclang++", error=str(exc))
            hsaco_bytes = None
        except Exception as exc:
            log.warning("AOTriton compile failed; trying Triton fallback", error=str(exc))
            hsaco_bytes = None

        if hsaco_bytes is None:
            try:
                hsaco_bytes = self._run_triton_fallback(
                    kernel_source,
                    kernel_name,
                    block_m,
                    block_n,
                    block_k,
                    num_warps,
                    num_stages,
                )
            except Exception as exc:
                elapsed = time.perf_counter() - start
                log.error(
                    "AMD AOT compilation failed",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                raise CompilationError(
                    f"AMD AOT compilation failed (AOTriton + Triton fallback): {exc}",
                    cause=exc,
                    context={
                        "arch": self.target_arch.value,
                        "kernel": kernel_name,
                    },
                ) from exc

        if not hsaco_bytes or len(hsaco_bytes) <= 100:
            raise CompilationOutputMissingError(
                f"AMD HSACO output is empty or stub-sized for {kernel_name}",
                context={
                    "arch": self.target_arch.value,
                    "size": len(hsaco_bytes) if hsaco_bytes else 0,
                },
            )

        if not self._validate_hsaco(hsaco_bytes):
            raise CompilationError(
                f"HSACO failed validation for {kernel_name} (not a real amdgcn object)",
                context={"arch": self.target_arch.value, "size": len(hsaco_bytes)},
            )

        hsaco_path = self._cache_path_for(cache_key)
        hsaco_path.parent.mkdir(parents=True, exist_ok=True)
        hsaco_path.write_bytes(hsaco_bytes)

        elapsed = time.perf_counter() - start
        log.info(
            "AMD AOT compile complete",
            kernel=kernel_name,
            arch=self.target_arch.value,
            hsaco_size=len(hsaco_bytes),
            used_aotriton=used_aotriton,
            elapsed_s=elapsed,
        )
        return AMDCompilationResult(
            success=True,
            arch=self.target_arch.value,
            hsaco_path=hsaco_path,
            hsaco_bytes=hsaco_bytes,
            compilation_time_s=elapsed,
            used_aotriton=used_aotriton,
            aotriton_version=self._aotriton_version,
        )

    def _run_aotriton(
        self,
        kernel_source: str,
        kernel_name: str,
        block_m: int,
        block_n: int,
        block_k: int,
        num_warps: int,
        num_stages: int,
    ) -> bytes:
        """Use AOTriton's Python API or CLI to compile."""
        # First, try the Python API
        try:
            from aotriton import compile

            with self._lock:
                tmp_dir = Path(
                    tempfile.mkdtemp(prefix="nautilus_aotriton_", dir=str(self.cache_dir))
                )
                try:
                    src_path = tmp_dir / f"{kernel_name}.py"
                    src_path.write_text(kernel_source)
                    out_path = tmp_dir / f"{kernel_name}.hsaco"
                    compile(
                        src_path=src_path,
                        output=out_path,
                        target=self.target_arch.value,
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                    if out_path.exists() and out_path.stat().st_size > 100:
                        return out_path.read_bytes()
                    raise CompilationOutputMissingError(
                        f"AOTriton produced no output for {kernel_name}",
                        context={"out_path": str(out_path)},
                    )
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
        except ImportError as exc:
            # No AOTriton Python — try CLI
            aotriton_cli = shutil.which("aotriton")
            if not aotriton_cli:
                raise AOTritonError(
                    "AOTriton not installed. Install with: pip install aotriton",
                ) from exc
            return self._run_aotriton_cli(
                aotriton_cli,
                kernel_source,
                kernel_name,
                block_m,
                block_n,
                block_k,
                num_warps,
                num_stages,
            )

    def _run_aotriton_cli(
        self,
        cli_path: str,
        kernel_source: str,
        kernel_name: str,
        block_m: int,
        block_n: int,
        block_k: int,
        num_warps: int,
        num_stages: int,
    ) -> bytes:
        with self._lock:
            tmp_dir = Path(
                tempfile.mkdtemp(prefix="nautilus_aotriton_cli_", dir=str(self.cache_dir))
            )
            try:
                src_path = tmp_dir / f"{kernel_name}.py"
                src_path.write_text(kernel_source)
                out_path = tmp_dir / f"{kernel_name}.hsaco"
                cmd = [
                    cli_path,
                    "compile",
                    "--src",
                    str(src_path),
                    "--output",
                    str(out_path),
                    "--target",
                    self.target_arch.value,
                    "--num-warps",
                    str(num_warps),
                    "--num-stages",
                    str(num_stages),
                ]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
                if result.returncode != 0 or not out_path.exists():
                    raise CompilationError(
                        f"aotriton CLI failed: {result.stderr}",
                        context={"stdout": result.stdout, "stderr": result.stderr},
                    )
                if out_path.stat().st_size <= 100:
                    raise CompilationOutputMissingError(
                        f"aotriton CLI produced empty output for {kernel_name}",
                    )
                return out_path.read_bytes()
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def _run_triton_fallback(
        self,
        kernel_source: str,
        kernel_name: str,
        block_m: int,
        block_n: int,
        block_k: int,
        num_warps: int,
        num_stages: int,
    ) -> bytes:
        """Fallback: use Triton to emit LLVM IR, then amdclang++ to compile
        to amdgcn. Requires amdclang++ or hipcc in PATH.
        """
        if not self._amdclangpp_path:
            raise AOTritonError(
                "AMD fallback requires amdclang++ or hipcc in PATH (AOTriton also "
                "unavailable). Install ROCm or AOTriton.",
            )
        try:
            import importlib.util

            import triton
            from triton.compiler import ASTSource
        except ImportError as exc:
            raise DependencyMissingError(
                "Triton not installed; cannot do AMD fallback",
            ) from exc

        with self._lock:
            tmp_dir = Path(tempfile.mkdtemp(prefix="nautilus_amd_fb_", dir=str(self.cache_dir)))
            try:
                src_path = tmp_dir / f"{kernel_name}.py"
                src_path.write_text(kernel_source)
                spec = importlib.util.spec_from_file_location(f"_amd_{kernel_name}", src_path)
                module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
                spec.loader.exec_module(module)  # type: ignore[union-attr]
                fn = getattr(module, kernel_name, None)
                if fn is None:
                    raise CompilationError(f"Function {kernel_name!r} not found")

                signature, constexprs = build_signature(
                    fn,
                    block_size_values={
                        "BLOCK_M": block_m,
                        "BLOCK_N": block_n,
                        "BLOCK_K": block_k,
                    },
                )
                source = ASTSource(
                    fn=fn,
                    constants={},
                    signature=signature,
                    constexprs=constexprs,
                    attrs={"num_warps": num_warps, "num_stages": num_stages},
                )
                options = {"num_warps": num_warps, "num_stages": num_stages}
                compiled = triton.compiler.compile(
                    src=source,
                    target="rocm",
                    options=options,
                )
                asm = compiled.asm
                ll_text = asm.get("ll") or asm.get("llvm")
                if ll_text is None:
                    raise CompilationOutputMissingError(
                        f"Triton fallback produced no LLVM IR for {kernel_name}",
                        context={"asm_keys": list(asm.keys())},
                    )
                ll_path = tmp_dir / f"{kernel_name}.ll"
                ll_path.write_text(ll_text)
                hsaco_path = tmp_dir / f"{kernel_name}.hsaco"
                cmd = [
                    self._amdclangpp_path,
                    "-x",
                    "ir",
                    f"--offload-arch={self.target_arch.value}",
                    "-o",
                    str(hsaco_path),
                    str(ll_path),
                ]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
                if result.returncode != 0 or not hsaco_path.exists():
                    raise CompilationError(
                        f"amdclang++ failed: {result.stderr}",
                        context={"stdout": result.stdout, "stderr": result.stderr},
                    )
                return hsaco_path.read_bytes()
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def _validate_hsaco(self, hsaco_bytes: bytes) -> bool:
        """Validate that the bytes are a real amdgcn object, not a stub."""
        if not hsaco_bytes or len(hsaco_bytes) < 100:
            return False
        # ELF magic
        if hsaco_bytes[:4] != b"\x7fELF":
            return False
        # Look for amdgcn target triple
        text = hsaco_bytes.decode("latin-1", errors="ignore")
        if "amdgcn" not in text and "AMDGPU" not in text:
            log.warning("HSACO does not declare amdgcn target", size=len(hsaco_bytes))
            return False
        return True

    def _compute_cache_key(
        self,
        source: str,
        name: str,
        block_m: int,
        block_n: int,
        block_k: int,
        num_warps: int,
        num_stages: int,
    ) -> str:
        payload = json.dumps(
            {
                "source": source,
                "name": name,
                "arch": self.target_arch.value,
                "block_m": block_m,
                "block_n": block_n,
                "block_k": block_k,
                "num_warps": num_warps,
                "num_stages": num_stages,
                "aotriton_version": self._aotriton_version,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _check_cache(self, cache_key: str) -> Path | None:
        p = self._cache_path_for(cache_key)
        if p.exists() and p.stat().st_size > 100:
            return p
        return None

    def _cache_path_for(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key[:32]}.hsaco"

    def _detect_aotriton(self) -> str:
        try:
            import aotriton

            return getattr(aotriton, "__version__", "unknown")
        except ImportError:
            pass
        cli = shutil.which("aotriton")
        if not cli:
            return "unavailable"
        try:
            result = subprocess.run(
                [cli, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip() or "unknown"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return "unavailable"

    def supports_arch(self, arch: str) -> bool:
        try:
            return AMDArch(arch) == self.target_arch
        except ValueError:
            return False

    def get_version(self) -> str:
        return self._aotriton_version
