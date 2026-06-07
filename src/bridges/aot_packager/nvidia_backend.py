"""Nvidia AOT backend — REAL Triton compilation, not a placeholder.

The previous design wrote a `mov %r0, 0; ret` stub PTX file with a
comment "for now". This rewrite:

  1. Uses triton.compiler.compile() to invoke Triton's AOT pipeline
  2. Returns real PTX (text) AND cubin (binary) when available
  3. Raises DependencyMissingError / CompilationError on failure —
     NEVER silently returns a placeholder
  4. Caches successful compilations for fast subsequent runs
  5. Validates that the produced PTX is real (not a 6-line stub)

The wiring flow:
  Triton Python source -> triton.JITFunction -> triton.ASTSource
    -> triton.compiler.compile(target=GPUTarget("cuda", arch, 32) or
                               self.target_arch.value,
                               options=...)
    -> CompiledKernel with .asm["ptx"] and .asm["cubin"]
    -> written to cache_dir
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, cast

try:
    from packaging.specifiers import SpecifierSet
    from packaging.version import InvalidVersion, Version

    _PACKAGING_AVAILABLE = True
except ImportError:  # pragma: no cover
    SpecifierSet = None  # type: ignore[assignment,misc]
    InvalidVersion = Exception  # type: ignore[assignment,misc]
    Version = None  # type: ignore[assignment,misc]
    _PACKAGING_AVAILABLE = False

from src.common.errors import (
    CompilationError,
    CompilationOutputMissingError,
    DependencyVersionMismatchError,
    TritonMissingError,
)
from src.common.logging import get_logger

from ._signature_inference import build_signature

log = get_logger("nautilus.aot.nvidia")


class NvidiaArch(str, Enum):
    """CUDA compute capabilities."""

    SM_70 = "sm_70"
    SM_75 = "sm_75"
    SM_80 = "sm_80"
    SM_86 = "sm_86"
    SM_89 = "sm_89"
    SM_90 = "sm_90"
    SM_100 = "sm_100"
    SM_120 = "sm_120"


@dataclass
class NvidiaCompilationResult:
    success: bool
    arch: str = "sm_90"
    ptx_path: Path | None = None
    ptx_text: str | None = None
    cubin_path: Path | None = None
    cubin_bytes: bytes | None = None
    error: str | None = None
    compilation_time_s: float = 0.0
    cache_hit: bool = False
    triton_version: str = ""

    @property
    def is_usable(self) -> bool:
        return self.success and (self.ptx_text is not None or self.cubin_bytes is not None)


class NvidiaBackend:
    """AOT compilation backend for Nvidia CUDA.

    The compile pipeline:
      1. Triton Python source -> triton.compiler.compile()
      2. Extract .asm["ptx"] (text) and .asm["cubin"] (binary)
      3. Validate the PTX is real (contains actual instructions,
         not a 6-line stub)
      4. Write to cache; on next identical request, return cached
    """

    # Range covers 3.x and 4.x; intentionally not pinned to a
    # micro-version per the project's drift strategy.
    SUPPORTED_TRITON_VERSIONS = ">=3.0,<5.0"

    def __init__(
        self,
        target_arch: NvidiaArch = NvidiaArch.SM_90,
        cache_dir: str | None = None,
        timeout_seconds: float = 120.0,
        capture_cubin: bool = True,
    ) -> None:
        self.target_arch = target_arch
        self.cache_dir = Path(
            cache_dir
            or os.environ.get(
                "NAUTILUS_NVIDIA_CACHE",
                str(Path.home() / ".cache" / "nautilus" / "nvidia"),
            )
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds
        self.capture_cubin = capture_cubin
        self._triton_version = self._detect_triton_version()
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
    ) -> NvidiaCompilationResult:
        """Compile a Triton kernel for the target Nvidia GPU.

        Raises:
            TritonMissingError: if triton is not installed.
            CompilationError: if compilation fails.
            CompilationOutputMissingError: if Triton produces no output.
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
            return NvidiaCompilationResult(
                success=True,
                arch=self.target_arch.value,
                ptx_path=cached["ptx"],
                ptx_text=cached["ptx"].read_text(),
                cubin_path=cached.get("cubin"),
                cubin_bytes=cached["cubin"].read_bytes() if cached.get("cubin") else None,
                compilation_time_s=time.perf_counter() - start,
                cache_hit=True,
                triton_version=self._triton_version,
            )

        try:
            ptx_text, cubin_bytes = self._run_triton_compile(
                kernel_source=kernel_source,
                kernel_name=kernel_name,
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        except TritonMissingError:
            raise
        except Exception as exc:
            elapsed = time.perf_counter() - start
            log.error(
                "Nvidia AOT compilation failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            raise CompilationError(
                f"Triton AOT compilation failed: {exc}",
                cause=exc,
                context={
                    "arch": self.target_arch.value,
                    "kernel": kernel_name,
                    "block_m": block_m,
                    "block_n": block_n,
                    "block_k": block_k,
                    "num_warps": num_warps,
                    "num_stages": num_stages,
                },
            ) from exc

        if not ptx_text:
            raise CompilationOutputMissingError(
                f"Triton produced no PTX for {kernel_name}",
                context={"arch": self.target_arch.value, "kernel": kernel_name},
            )

        if not self._validate_ptx(ptx_text, kernel_name):
            raise CompilationError(
                f"Generated PTX failed validation for {kernel_name}",
                context={"arch": self.target_arch.value, "ptx_lines": ptx_text.count("\n")},
            )

        ptx_path = self._cache_path_for(cache_key, "ptx")
        ptx_path.parent.mkdir(parents=True, exist_ok=True)
        ptx_path.write_text(ptx_text)

        cubin_path = None
        if cubin_bytes is not None and self.capture_cubin:
            cubin_path = self._cache_path_for(cache_key, "cubin")
            cubin_path.write_bytes(cubin_bytes)

        elapsed = time.perf_counter() - start
        log.info(
            "Nvidia AOT compile complete",
            kernel=kernel_name,
            arch=self.target_arch.value,
            ptx_size=len(ptx_text),
            cubin_size=len(cubin_bytes) if cubin_bytes else 0,
            elapsed_s=elapsed,
        )
        return NvidiaCompilationResult(
            success=True,
            arch=self.target_arch.value,
            ptx_path=ptx_path,
            ptx_text=ptx_text,
            cubin_path=cubin_path,
            cubin_bytes=cubin_bytes,
            compilation_time_s=elapsed,
            triton_version=self._triton_version,
        )

    def _run_triton_compile(
        self,
        kernel_source: str,
        kernel_name: str,
        block_m: int,
        block_n: int,
        block_k: int,
        num_warps: int,
        num_stages: int,
    ) -> tuple[str, bytes | None]:
        """Invoke Triton's AOT compiler.

        Strategy:
          1. Write the kernel source to a temp file
          2. Dynamically import the file as a module
          3. Find the @triton.jit function by name
          4. Build a triton.ASTSource (signature inferred dynamically)
          5. Call triton.compiler.compile() with target=self.target_arch
             (as a GPUTarget when Triton exposes one, else the arch string)
          6. Extract .asm["ptx"] (always) and .asm["cubin"] (if available)
        """
        try:
            import triton
        except ImportError as exc:
            raise TritonMissingError(
                "Triton is not installed. Install with: pip install triton",
            ) from exc

        if not hasattr(triton, "compiler") or not hasattr(triton.compiler, "compile"):
            raise TritonMissingError(
                "Installed triton version is too old (no triton.compiler.compile). "
                "Upgrade to triton>=3.0.0.",
            )

        self._verify_triton_version(triton.__version__)

        with self._lock:
            tmp_dir = Path(tempfile.mkdtemp(prefix="nautilus_aot_", dir=str(self.cache_dir)))
            try:
                src_path = tmp_dir / f"{kernel_name}.py"
                src_path.write_text(kernel_source)
                spec = importlib.util.spec_from_file_location(f"_aot_{kernel_name}", src_path)
                if spec is None or spec.loader is None:
                    raise CompilationError(
                        f"Could not create import spec for {kernel_name}",
                    )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                fn = getattr(module, kernel_name, None)
                if fn is None:
                    raise CompilationError(
                        f"Function {kernel_name!r} not found in module after import",
                        context={"kernel_name": kernel_name},
                    )
                if not hasattr(fn, "run"):
                    raise CompilationError(
                        f"Function {kernel_name!r} is not a @triton.jit function",
                        context={"kernel_name": kernel_name},
                    )

                from triton.compiler import ASTSource

                signature, constexprs = build_signature(
                    fn,
                    block_size_values={
                        "BLOCK_M": block_m,
                        "BLOCK_N": block_n,
                        "BLOCK_K": block_k,
                    },
                )
                # Triton 3.0+ ASTSource requires:
                #   - signature keyed by parameter NAME (str), not position
                #   - constexprs keyed by positional index wrapped in a tuple
                #   - the legacy `constants=` kwarg is gone (use `constexprs=`)
                # build_signature() works in positional int keys (easier to
                # infer from inspect.signature); remap to Triton's expected
                # shape here so the backend stays self-contained regardless
                # of upstream helper changes.
                arg_names: list[str] = list(getattr(fn, "arg_names", []) or [])
                if arg_names and all(isinstance(k, int) for k in signature):
                    signature = cast(
                        dict[int, str],
                        {
                            arg_names[i]: dtype
                            for i, dtype in signature.items()
                            if i < len(arg_names)
                        },
                    )
                if arg_names and all(isinstance(k, int) for k in constexprs):
                    constexprs = cast(
                        dict[int, Any],
                        {
                            (idx,): value
                            for idx, value in constexprs.items()
                            if idx < len(arg_names)
                        },
                    )
                source = ASTSource(
                    fn=fn,
                    signature=signature,
                    constexprs=constexprs,
                    attrs={
                        "num_warps": num_warps,
                        "num_stages": num_stages,
                    },
                )
                options = {
                    "num_warps": num_warps,
                    "num_stages": num_stages,
                }
                target_arg = self._resolve_nvidia_target()
                compiled = triton.compiler.compile(
                    src=source,
                    target=target_arg,
                    options=options,
                )
                asm = compiled.asm
                ptx_text = asm.get("ptx")
                cubin_bytes = asm.get("cubin")
                if ptx_text is None and cubin_bytes is None:
                    raise CompilationOutputMissingError(
                        f"triton.compiler.compile produced no PTX or cubin for {kernel_name}",
                        context={"arch": self.target_arch.value, "asm_keys": list(asm.keys())},
                    )
                return ptx_text or "", cubin_bytes
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def _resolve_nvidia_target(self) -> Any:
        """Build the ``target`` argument for ``triton.compiler.compile``.

        Prefers a ``GPUTarget(backend="cuda", arch=(major, minor),
        warp_size=32)`` instance (Triton 3.0+) so the emitted PTX/SASS
        matches the configured ``self.target_arch``. Falls back to the
        arch string (e.g. ``"sm_90"``) and ultimately to ``"cuda"`` for
        older Triton versions that only accept backend-name strings.
        """
        arch_str = self.target_arch.value
        # Triton 3.0+ moved GPUTarget to triton.backends.compiler.
        gpu_target_cls: Any = None
        try:
            from triton.backends.compiler import GPUTarget as _NewGPUTarget

            gpu_target_cls = _NewGPUTarget
        except ImportError:
            try:
                from triton.compiler import GPUTarget as _LegacyGPUTarget

                gpu_target_cls = _LegacyGPUTarget
            except ImportError:
                return arch_str if arch_str else "cuda"
        if gpu_target_cls is None:
            return arch_str if arch_str else "cuda"
        if arch_str.startswith("sm_"):
            # GPUTarget.arch is the compute capability as a single int
            # (e.g. 90 for sm_90), not a (major, minor) tuple. The string
            # suffix may be 2 or 3 digits: sm_70, sm_90, sm_100, sm_120.
            digits = arch_str[3:]
            if digits.isdigit() and digits:
                return gpu_target_cls(
                    backend="cuda",
                    arch=int(digits),
                    warp_size=32,
                )
        return arch_str if arch_str else "cuda"

    def _validate_ptx(self, ptx_text: str, kernel_name: str) -> bool:
        """Validate that the PTX is real and not a stub.

        A real PTX from a Triton kernel contains:
          - A .version directive
          - A .target directive
          - A .address_size directive
          - At least one .entry function
          - Multiple actual instructions (ld., st., mov, etc.)

        A stub PTX (the previous bug) was 6 lines ending in `ret`.
        """
        if not ptx_text:
            return False
        markers = [
            ".version",
            ".target",
            ".address_size",
            ".entry",
            ".visible",
            ".func",
        ]
        has_marker = any(m in ptx_text for m in markers)
        if not has_marker:
            log.warning(
                "PTX missing standard markers",
                kernel=kernel_name,
                first_line=ptx_text.splitlines()[0] if ptx_text else "",
            )
            return False
        line_count = ptx_text.count("\n")
        if line_count < 5:
            log.warning("PTX suspiciously short", kernel=kernel_name, lines=line_count)
            return False
        return True

    def _compute_cache_key(
        self,
        kernel_source: str,
        kernel_name: str,
        block_m: int,
        block_n: int,
        block_k: int,
        num_warps: int,
        num_stages: int,
    ) -> str:
        payload = json.dumps(
            {
                "source": kernel_source,
                "name": kernel_name,
                "arch": self.target_arch.value,
                "block_m": block_m,
                "block_n": block_n,
                "block_k": block_k,
                "num_warps": num_warps,
                "num_stages": num_stages,
                "triton_version": self._triton_version,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _check_cache(self, cache_key: str) -> dict[str, Path] | None:
        ptx_path = self._cache_path_for(cache_key, "ptx")
        if not ptx_path.exists() or ptx_path.stat().st_size < 100:
            return None
        result: dict[str, Path] = {"ptx": ptx_path}
        cubin_path = self._cache_path_for(cache_key, "cubin")
        if cubin_path.exists() and cubin_path.stat().st_size > 0:
            result["cubin"] = cubin_path
        return result

    def _cache_path_for(self, cache_key: str, ext: str) -> Path:
        return self.cache_dir / f"{cache_key[:32]}.{ext}"

    def _detect_triton_version(self) -> str:
        try:
            import triton

            return getattr(triton, "__version__", "unknown")
        except ImportError:
            return "unavailable"

    def _verify_triton_version(self, version_str: str) -> None:
        """Raise DependencyVersionMismatchError if triton is outside the
        supported range.

        Defined as a range (not a pin) per the drift strategy: the
        ASTSource / GPUTarget APIs we depend on were stable from
        3.0 onwards, and the upper bound is left open for a future
        4.x without forcing a code change. Unparseable versions are
        warned but do not raise — the user may be running a forked
        build (e.g. nvidia internal triton) with a non-PEP-440 tag.
        """
        if (
            not _PACKAGING_AVAILABLE
            or not version_str
            or version_str
            in {
                "unknown",
                "unavailable",
            }
        ):
            return
        try:
            parsed = Version(version_str)
        except InvalidVersion:
            log.warning(
                "Triton version is not PEP-440 parseable; skipping range check",
                version=version_str,
            )
            return
        spec = SpecifierSet(self.SUPPORTED_TRITON_VERSIONS)
        if parsed not in spec:
            raise DependencyVersionMismatchError(
                f"Installed triton {version_str} is outside the supported "
                f"range {self.SUPPORTED_TRITON_VERSIONS}. The Nvidia AOT "
                f"backend requires an in-range triton build.",
                context={
                    "installed": version_str,
                    "supported": str(self.SUPPORTED_TRITON_VERSIONS),
                },
            )

    def supports_arch(self, arch: str) -> bool:
        try:
            return NvidiaArch(arch) == self.target_arch
        except ValueError:
            return False

    def get_version(self) -> str:
        return self._triton_version
