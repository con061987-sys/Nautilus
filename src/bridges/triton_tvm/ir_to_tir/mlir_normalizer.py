"""MLIR Vector Dialect normalization bridge.

Implements the TTGIR -> MLIR Vector Dialect -> TVM TIR normalization
pipeline described in TECH_SPEC.md Section 2.3 and docs/E.md.

Architecture
------------
The TECH_SPEC prescribes lowering TTGIR through Standard MLIR Vector/Math
Dialect before emitting TVM TIR. This module provides that path.

Two implementations exist:
  1. C++ MLIR pass (recommended, performant) — built from
     src/bridges/triton_tvm/lib/ when MLIR_DIR is configured.
  2. Python fallback (this module) — delegates to the existing 4-pass
     pipeline when MLIR C++ plugin is unavailable.

C++ build:
  mkdir -p src/bridges/triton_tvm/lib/build
  cd src/bridges/triton_tvm/lib/build
  cmake -GNinja \\
    -DTRITON_SRC_DIR=/path/to/triton \\
    -DMLIR_DIR=/path/to/llvm/lib/cmake/mlir \\
    -DCMAKE_BUILD_TYPE=Release \\
    ..
  ninja mlir_vector_normalizer
  # Then: export NAUTILUS_MLIR_PLUGIN=$PWD/libmlir_vector_normalizer.so

Interface
---------
The MLIRNormalizer follows the TTGIRPass protocol (defined in ttgir_parser.py)
so it can be composed into the existing conversion pipeline seamlessly.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from src.common.logging import get_logger

from .ttgir_parser import TTGIRFunction

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

_MLIR_AVAILABLE: bool | None = None
"""Cached result of MLIR availability check."""


def mlir_available() -> bool:
    """Return True if the MLIR C++ plugin or Python bindings are loadable."""
    global _MLIR_AVAILABLE
    if _MLIR_AVAILABLE is not None:
        return _MLIR_AVAILABLE

    # 1) Try the C++ plugin via NAUTILUS_MLIR_PLUGIN env var
    plugin_path = os.environ.get("NAUTILUS_MLIR_PLUGIN")
    if plugin_path and Path(plugin_path).exists():
        try:
            ctypes.CDLL(plugin_path)
            _MLIR_AVAILABLE = True
            logger.info("mlir_normalizer: loaded C++ plugin from %s", plugin_path)
            return True
        except OSError as exc:
            logger.warning("mlir_normalizer: C++ plugin load failed: %s", exc)

    # 2) Try the Python MLIR bindings (pip install mlir, or built from source)
    try:
        import mlir  # noqa: F401
        from mlir.dialects import vector  # noqa: F401
        from mlir.ir import Module as MlirModule  # noqa: F401

        _MLIR_AVAILABLE = True
        logger.info("mlir_normalizer: using MLIR Python bindings")
        return True
    except (ImportError, AttributeError):
        pass

    # 3) Try llvm-spirv (partial MLIR tooling available)
    if shutil_which("mlir-opt") or shutil_which("llvm-spirv"):
        _MLIR_AVAILABLE = True
        logger.info("mlir_normalizer: using MLIR tools via subprocess")
        return True

    _MLIR_AVAILABLE = False
    logger.info(
        "mlir_normalizer: MLIR not available — using Python 4-pass fallback. "
        "Build from source or set NAUTILUS_MLIR_PLUGIN to enable MLIR acceleration."
    )
    return False


def shutil_which(cmd: str) -> str | None:
    """Minimal which() without importing shutil at module level."""
    import shutil
    return shutil.which(cmd)


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------


@dataclass
class NormalizationResult:
    """Result of MLIR normalization."""

    success: bool
    tvmscript: str
    mlir_text: str | None = None
    diagnostics: list[str] = field(default_factory=list)
    backend: str = "python_fallback"
    """Which backend produced the result: 'python_fallback', 'mlir_python', 'mlir_cpp'."""


class MLIRNormalizer:
    """Normalize TTGIR through MLIR Vector Dialect.

    Uses the C++ plugin if available, otherwise falls back to the
    Python 4-pass pipeline. The fallback is semantically equivalent
    but does not go through actual MLIR Vector Dialect.
    """

    def __init__(self, use_mlir: bool | None = None):
        """Initialize normalizer.

        Args:
            use_mlir: Force MLIR path (True), force Python fallback (False),
                or auto-detect (None, default).
        """
        self._force_mlir = use_mlir
        self._backend: str = "python_fallback"

    def normalize(self, ttgir_text: str, kernel_name: str = "kernel") -> NormalizationResult:
        """Normalize TTGIR text to TVMScript.

        This method tries the MLIR path first (if available), falling
        back to the Python 4-pass pipeline.
        """
        diagnostics: list[str] = []

        # Try MLIR path
        use_mlir = self._force_mlir if self._force_mlir is not None else mlir_available()
        if use_mlir:
            result = self._normalize_via_mlir(ttgir_text, kernel_name)
            if result.success:
                self._backend = result.backend
                return result
            diagnostics.append(f"MLIR path failed: {result.diagnostics}")

        # Fallback: Python 4-pass pipeline
        return self._normalize_via_python(ttgir_text, kernel_name, diagnostics)

    def _normalize_via_mlir(self, ttgir_text: str, kernel_name: str) -> NormalizationResult:
        """Attempt MLIR Vector Dialect normalization.

        The C++ plugin implements this pipeline:
          TTGIR (Triton) -> De-Sugar Pass -> Strip vendor-specific layout -> Standard MLIR Vector Dialect -> TVM TIR
        """
        diagnostics: list[str] = []
        backend = "mlir_python"

        # Strategy 1: C++ plugin via ctypes
        plugin_path = os.environ.get("NAUTILUS_MLIR_PLUGIN")
        if plugin_path and Path(plugin_path).exists():
            try:
                lib = ctypes.CDLL(plugin_path)
                lib.normalize_ttgir.argtypes = [ctypes.c_char_p]
                lib.normalize_ttgir.restype = ctypes.c_char_p
                result_bytes = lib.normalize_ttgir(ttgir_text.encode("utf-8"))
                if result_bytes:
                    tvmscript = result_bytes.decode("utf-8")
                    diagnostics.append("C++ MLIR normalizer produced output")
                    return NormalizationResult(
                        success=True,
                        tvmscript=tvmscript,
                        mlir_text="(via C++ plugin)",
                        diagnostics=diagnostics,
                        backend="mlir_cpp",
                    )
                diagnostics.append("C++ plugin returned empty output")
            except (OSError, AttributeError) as exc:
                diagnostics.append(f"C++ plugin error: {exc}")

        # Strategy 2: MLIR Python bindings
        try:
            import mlir  # noqa: F401
            from mlir.dialects import vector  # noqa: F401

            # Placeholder: actual TTGIR -> Vector Dialect conversion
            # would parse TTGIR, lower through Vector Dialect, and emit.
            # The implementation requires MLIR Python bindings built
            # from LLVM source with MLIR_ENABLE_BINDINGS_PYTHON=ON.
            diagnostics.append(
                "MLIR Python bindings available but TTGIR->Vector conversion "
                "requires C++ plugin or LLVM-built bindings with dialect registration. "
                "Falling back to Python 4-pass pipeline."
            )
        except (ImportError, AttributeError) as exc:
            diagnostics.append(f"MLIR Python path failed: {exc}")

        # Strategy 3: mlir-opt subprocess (experimental)
        mlir_opt = shutil_which("mlir-opt")
        if mlir_opt:
            try:
                proc = subprocess.run(
                    [mlir_opt, "--pass-pipeline=builtin.module(lower-affine,convert-vector-to-llvm)"],
                    input=ttgir_text,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if proc.returncode == 0:
                    diagnostics.append("mlir-opt processed input (structural validation)")
                else:
                    diagnostics.append(f"mlir-opt error: {proc.stderr[:200]}")
            except (OSError, subprocess.TimeoutExpired) as exc:
                diagnostics.append(f"mlir-opt failed: {exc}")

        return NormalizationResult(success=False, tvmscript="", diagnostics=diagnostics, backend=backend)

    def _normalize_via_python(
        self,
        ttgir_text: str,
        kernel_name: str,
        prior_diagnostics: list[str] | None = None,
    ) -> NormalizationResult:
        """Fallback: use the Python 4-pass conversion pipeline."""
        diagnostics = list(prior_diagnostics or [])
        diagnostics.append("Using Python 4-pass fallback pipeline")
        self._backend = "python_fallback"

        try:
            from .conversion_pipeline import ConversionPipeline

            pipeline = ConversionPipeline()
            result = pipeline.convert(ttgir_text)
            if result.status.name == "SUCCESS":
                diagnostics.append("Python pipeline succeeded")
                return NormalizationResult(
                    success=True,
                    tvmscript=result.tvmscript_text,
                    diagnostics=diagnostics,
                    backend="python_fallback",
                )
            diagnostics.append(f"Python pipeline failed: {result.error or 'unknown'}")
            return NormalizationResult(
                success=False,
                tvmscript="",
                diagnostics=diagnostics,
                backend="python_fallback",
            )
        except Exception as exc:
            diagnostics.append(f"Python pipeline exception: {exc}")
            return NormalizationResult(
                success=False,
                tvmscript="",
                diagnostics=diagnostics,
                backend="python_fallback",
            )

    @property
    def backend(self) -> str:
        """Return the backend used by the last normalize() call."""
        return self._backend

    # TTGIRPass protocol — allows MLIRNormalizer to be composed into
    # the existing conversion pipeline
    def run(self, func: TTGIRFunction) -> TTGIRFunction:
        """TTGIRPass protocol implementation.

        This is a no-op for the existing pass pipeline (MLIR normalization
        is best applied at the IR text level, not the parsed AST level).
        Returns the function unchanged with a diagnostic note.
        """
        logger.info("mlir_normalizer: TTGIRPass.run() is a no-op; "
                     "call normalize() directly on IR text")
        return func
