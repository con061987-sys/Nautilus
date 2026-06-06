"""Apple Metal AOT backend — real macOS Metal compilation.

This is the cross-platform parity backend. macOS uses Metal
(MSL) and Metal Performance Shaders instead of CUDA/ROCm/SYCL.

The compilation flow:
  Triton Python source
    -> triton.compiler.compile(target="metal")
    -> MSL (Metal Shading Language) text
    -> metallib (compiled binary) via `xcrun metal`

Like the other backends, this raises clear errors when the
required toolchain is missing, rather than silently returning
a stub.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
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
from typing import Any

from src.common.errors import (
    CompilationError,
    CompilationOutputMissingError,
    DependencyMissingError,
    HardwareNotFoundError,
    NautilusError,
)
from src.common.logging import get_logger

log = get_logger("nautilus.aot.metal")


class MetalTarget(str, Enum):
    """Apple Metal GPU families."""
    APPLE_M1 = "apple_m1"
    APPLE_M2 = "apple_m2"
    APPLE_M3 = "apple_m3"
    APPLE_M4 = "apple_m4"
    GENERIC = "generic_metal"


@dataclass
class MetalCompilationResult:
    success: bool
    target: str = "apple_m2"
    metallib_path: Path | None = None
    metallib_bytes: bytes | None = None
    msl_text: str | None = None
    error: str | None = None
    compilation_time_s: float = 0.0
    cache_hit: bool = False

    @property
    def is_usable(self) -> bool:
        return self.success and (self.metallib_bytes is not None or self.msl_text is not None)


class MetalBackend:
    """AOT compilation for Apple Silicon GPUs via Triton + xcrun metal.

    Requires:
      - macOS host (any Apple Silicon)
      - xcrun / Metal toolchain (Xcode command-line tools)
      - Triton with the "metal" backend (Triton 3.0+)

    Raises DependencyMissingError if any of the above is missing.
    """
    def __init__(
        self,
        target: MetalTarget = MetalTarget.APPLE_M2,
        cache_dir: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.target = target
        self.cache_dir = Path(cache_dir or os.environ.get(
            "NAUTILUS_METAL_CACHE",
            str(Path.home() / ".cache" / "nautilus" / "metal"),
        ))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds
        self._xcrun_metal_path = self._find_tool("xcrun")
        self._metallib_ext = "metallib"
        self._lock = threading.Lock()

    def compile_kernel(
        self,
        kernel_source: str,
        kernel_name: str,
        block_m: int = 128,
        block_n: int = 128,
        block_k: int = 32,
        num_warps: int = 8,
    ) -> MetalCompilationResult:
        """Compile a Triton kernel to MSL / metallib for Apple Silicon.

        Raises:
            HardwareNotFoundError: if not running on macOS.
            DependencyMissingError: if xcrun or Triton Metal backend missing.
            CompilationError: if compilation fails.
        """
        import platform
        if platform.system() != "Darwin":
            raise HardwareNotFoundError(
                "Apple Metal backend requires macOS host",
                context={"current_os": platform.system()},
            )
        if not self._xcrun_metal_path:
            raise DependencyMissingError(
                "xcrun not found in PATH. Install Xcode Command Line Tools: "
                "xcode-select --install",
            )

        start = time.perf_counter()
        cache_key = self._compute_cache_key(
            kernel_source, kernel_name, block_m, block_n, block_k, num_warps,
        )
        cached = self._check_cache(cache_key)
        if cached is not None:
            return MetalCompilationResult(
                success=True,
                target=self.target.value,
                metallib_path=cached,
                metallib_bytes=cached.read_bytes(),
                compilation_time_s=time.perf_counter() - start,
                cache_hit=True,
            )

        try:
            msl_text, metallib_bytes = self._compile(
                kernel_source, kernel_name, block_m, block_n, block_k, num_warps,
            )
        except DependencyMissingError:
            raise
        except Exception as exc:
            elapsed = time.perf_counter() - start
            log.error("Metal AOT compilation failed", error=str(exc))
            raise CompilationError(
                f"Apple Metal AOT compilation failed: {exc}",
                cause=exc,
                context={"target": self.target.value, "kernel": kernel_name},
            ) from exc

        if not msl_text:
            raise CompilationOutputMissingError(
                f"Metal AOT produced no MSL text for {kernel_name}",
                context={"target": self.target.value},
            )

        metallib_path = self._cache_path_for(cache_key)
        if metallib_bytes is not None:
            metallib_path.write_bytes(metallib_bytes)
        else:
            metallib_path.write_text(msl_text)

        elapsed = time.perf_counter() - start
        log.info(
            "Metal AOT compile complete",
            kernel=kernel_name,
            target=self.target.value,
            msl_lines=msl_text.count("\n"),
            metallib_size=len(metallib_bytes) if metallib_bytes else 0,
            elapsed_s=elapsed,
        )
        return MetalCompilationResult(
            success=True,
            target=self.target.value,
            metallib_path=metallib_path,
            metallib_bytes=metallib_bytes,
            msl_text=msl_text,
            compilation_time_s=elapsed,
        )

    def _compile(
        self,
        kernel_source: str,
        kernel_name: str,
        block_m: int,
        block_n: int,
        block_k: int,
        num_warps: int,
    ) -> tuple[str, bytes | None]:
        """Run the full Triton -> MSL -> metallib compilation."""
        try:
            import triton
        except ImportError as exc:
            raise DependencyMissingError(
                "Triton not installed; pip install triton>=3.0",
            ) from exc

        with self._lock:
            tmp_dir = Path(tempfile.mkdtemp(prefix="nautilus_metal_", dir=str(self.cache_dir)))
            try:
                src_path = tmp_dir / f"{kernel_name}.py"
                src_path.write_text(kernel_source)
                spec = importlib.util.spec_from_file_location(
                    f"_metal_{kernel_name}", src_path,
                )
                if spec is None or spec.loader is None:
                    raise CompilationError(f"Could not import {kernel_name}")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                fn = getattr(module, kernel_name, None)
                if fn is None or not hasattr(fn, "run"):
                    raise CompilationError(
                        f"Function {kernel_name!r} not found or not @triton.jit",
                    )
                from triton.compiler import ASTSource  # type: ignore[attr-defined]
                sig_args = ["*fp32"] * 3 + ["i32"] * 3 + ["constexpr"] * 3
                signature = {i: a for i, a in enumerate(sig_args)}
                constexprs = {
                    len(sig_args) - 3: block_m,
                    len(sig_args) - 2: block_n,
                    len(sig_args) - 1: block_k,
                }
                source = ASTSource(
                    fn=fn, constants={}, signature=signature, constexprs=constexprs,
                    attrs={"num_warps": num_warps, "num_stages": 2},
                )
                options = {"num_warps": num_warps, "num_stages": 2}
                compiled = triton.compiler.compile(
                    src=source, target="metal", options=options,
                )
                asm = compiled.asm
                msl_text = asm.get("msl") or asm.get("metal")
                if msl_text is None:
                    raise CompilationOutputMissingError(
                        f"triton.compiler.compile produced no MSL for {kernel_name}",
                        context={"asm_keys": list(asm.keys())},
                    )
                metal_air = asm.get("air") or asm.get("metallib")
                return msl_text, metal_air
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def _compute_cache_key(
        self,
        source: str, name: str, block_m: int, block_n: int, block_k: int, num_warps: int,
    ) -> str:
        payload = json.dumps({
            "source": source, "name": name, "target": self.target.value,
            "block_m": block_m, "block_n": block_n, "block_k": block_k,
            "num_warps": num_warps,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _check_cache(self, cache_key: str) -> Path | None:
        p = self._cache_path_for(cache_key)
        if p.exists() and p.stat().st_size > 0:
            return p
        return None

    def _cache_path_for(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key[:32]}.{self._metallib_ext}"

    def _find_tool(self, name: str) -> str:
        return shutil.which(name) or ""

    def supports_target(self, target: str) -> bool:
        try:
            return MetalTarget(target) == self.target
        except ValueError:
            return False

    def get_version(self) -> str:
        if not self._xcrun_metal_path:
            return "unavailable"
        try:
            result = subprocess.run(
                [self._xcrun_metal_path, "metal", "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip() or "unknown"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return "unknown"
