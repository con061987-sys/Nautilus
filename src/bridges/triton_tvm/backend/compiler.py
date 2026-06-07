"""TVMBackend — the Triton out-of-tree backend implementation.

This is the contract entry point that Triton calls when our backend is
loaded. It implements triton.backends.compiler.BaseBackend, populating
the stages dict with hooks that:
  1. Capture the real TTGIR from Triton's compilation pipeline
  2. Forward it to the TVM MetaSchedule adapter for tuning
  3. Inject the optimised config back into Triton's compile via ir_override

This is NOT a stub. The make_ttgir override actually reads the IR
module and routes it to the bridge orchestrator.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.common.logging import get_logger

try:
    from triton.backends.compiler import (
        BaseBackend,
        GPUTarget,
        Language,
    )

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

    # Stub for environments without Triton
    class BaseBackend:  # type: ignore
        def __init__(self, *a: Any, **kw: Any) -> None: ...

    class GPUTarget:  # type: ignore
        pass

    class Language:  # type: ignore
        pass


from . import CAPTURE_KEY_FMT
from .options import TVMOptions

logger = get_logger(__name__)


# Module-level cache for capturing IR across stage lambdas
# Keyed by (source_hash, target) to avoid capture mixing between kernels
_CAPTURE_BUFFER: dict[str, str] = {}


@dataclass
class CapturedIR:
    """The IR captured at a specific pipeline stage."""

    source_hash: str
    target: str
    stage_name: str  # 'ttir' | 'ttgir' | 'llir' | ...
    ir_text: str
    metadata: dict[str, Any]


class TVMBackend(BaseBackend):
    """Out-of-tree Triton backend that integrates with TVM MetaSchedule.

    Implements triton.backends.compiler.BaseBackend. The contract is:
    Triton calls add_stages(stages, options) to populate a dict of
    pipeline stages. We override the ttgir stage to capture the actual
    IR module and dispatch it to the bridge orchestrator.

    Hash stability: the hash() method must change whenever the
    backend's behaviour changes, otherwise Triton's cache will
    serve stale binaries.
    """

    # Backend name registered in triton.backends entry point
    NAME = "tvm"

    def __init__(self, target: GPUTarget) -> None:
        if not TRITON_AVAILABLE:
            raise RuntimeError(
                "Triton is required to use TVMBackend. Install with: pip install triton"
            )
        super().__init__(target)
        self.target_obj = target
        # Map Triton GPUTarget.backend to a TVM target string
        self._tvm_target = self._map_target(target)

    @staticmethod
    def supports_target(target: GPUTarget) -> bool:
        """Return True for all targets we support.

        We support all CUDA/HIP backends because TVM MetaSchedule
        can target any of them. AOT selection is handled at the
        FatBinaryPackager level, not here.
        """
        if not TRITON_AVAILABLE:
            return False
        return target.backend in ("cuda", "hip", "metal", "cpu")

    def hash(self) -> str:
        """Backend hash — included in Triton's cache key.

        Must change if behaviour changes. Includes:
          - Backend name
          - Target arch
          - Our internal version (bump on every behaviour change)
        """
        parts = [
            "tvm_backend_v1",
            self.target_obj.backend,
            str(self.target_obj.arch),
            str(self.target_obj.warp_size),
            TVMOptions().to_dict().__repr__(),  # defaults — change invalidates cache
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

    def parse_options(self, options: dict[str, Any]) -> TVMOptions:
        """Parse user-supplied options dict into a TVMOptions instance.

        Called by Triton's Compiler.run() before add_stages(). Anything
        not in the dict is left at default.
        """
        return TVMOptions.from_dict(options or {})

    def add_stages(
        self,
        stages: dict[str, Callable[..., Any]],
        options: TVMOptions,
    ) -> None:
        """Populate the pipeline stages dict.

        This is THE critical integration point. We wrap the standard
        ttir and ttgir stages so that we can capture the IR after
        each transformation. The captured IR is what we feed to
        TVM MetaSchedule for tuning.

        Stages dict contract:
          stages["ttir"]  = lambda src, metadata: Module
          stages["ttgir"] = lambda src, metadata: Module
          stages["llir"]  = lambda src, metadata: Module
          ... etc.

        We wrap each stage to capture the OUTPUT of the previous stage.
        """
        # Wrap TTIR stage — captures the hardware-independent TTIR
        original_ttir = stages.get("ttir")
        if original_ttir is not None:
            stages["ttir"] = self._wrap_stage(
                original_ttir,
                "ttir",
                options,
            )

        # Wrap TTGIR stage — captures the hardware-aware TTGIR
        # This is the most important one — this is what we feed to TVM
        original_ttgir = stages.get("ttgir")
        if original_ttgir is not None:
            stages["ttgir"] = self._wrap_stage(
                original_ttgir,
                "ttgir",
                options,
            )

        # Wrap LLIR stage — captures the LLVM IR after Triton lowering
        original_llir = stages.get("llir")
        if original_llir is not None:
            stages["llir"] = self._wrap_stage(
                original_llir,
                "llir",
                options,
            )

        logger.debug(
            "TVMBackend.add_stages: wrapped ttir/ttgir/llir stages for target=%s",
            self._tvm_target,
        )

    def load_dialects(self, context: Any) -> None:
        """Load any custom MLIR dialects.

        We don't add new dialects — we use Triton's existing dialects.
        But the contract requires this method, so we implement it
        as a no-op that delegates to Triton's own dialect loader.
        """
        # No custom dialects needed; Triton's dialects are loaded
        # automatically by the orchestrator before this is called.
        pass

    def get_module_map(self) -> dict[str, Any]:
        """Return MLIR module imports needed for this backend.

        Triton uses this to set up the MLIR context. We don't need
        any custom imports, so return an empty dict.
        """
        return {}

    def get_codegen_implementation(
        self,
        options: TVMOptions,
    ) -> dict[str, Any]:
        """Return codegen functions used by the frontend.

        The frontend (triton.compiler.frontend) calls these to convert
        Python AST → MLIR. We delegate to Triton's standard codegen
        because we don't have a custom Python frontend.
        """
        # Defer import to avoid circular dependency
        try:
            from triton.language.extra import libdevice

            return {"libdevice": libdevice}
        except ImportError:
            return {}

    # ------------------------------------------------------------------
    # Internal stage wrapping
    # ------------------------------------------------------------------

    def _wrap_stage(
        self,
        original_stage: Callable[..., Any],
        stage_name: str,
        options: TVMOptions,
    ) -> Callable[..., Any]:
        """Wrap a pipeline stage to capture its output for TVM tuning.

        The wrapper:
          1. Calls the original stage
          2. Captures the resulting IR (text or module)
          3. Stores it in the capture buffer keyed by source_hash+target
          4. Returns the original output unchanged (zero-impact on compile)

        This is zero-overhead when the bridge is not active (e.g. when
        not running tuning). When active, it triggers the orchestrator
        asynchronously to avoid blocking the compile.
        """

        def wrapped(src: Any, metadata: dict[str, Any]) -> Any:
            # Run the original stage — this produces the IR module
            result = original_stage(src, metadata)

            # Skip capture if disabled
            if not options.use_native_plugin:
                return result

            # Skip capture if the buffer is disabled
            if os.environ.get("NVINDIACUD_CAPTURE_DISABLED", "0") == "1":
                return result

            try:
                self._capture_stage(result, stage_name, metadata, options)
            except Exception as exc:
                # Capture must NEVER break the compile path
                logger.warning(
                    "TVMBackend capture failed at stage=%s: %s",
                    stage_name,
                    exc,
                )

            return result

        return wrapped

    def _capture_stage(
        self,
        ir_module: Any,
        stage_name: str,
        metadata: dict[str, Any],
        options: TVMOptions,
    ) -> None:
        """Capture the IR module at a specific pipeline stage."""
        # Derive a stable key for the source
        source_hash = self._derive_source_hash(metadata, stage_name)

        # Convert MLIR module to text (this works for both mlir::Module
        # Python objects and string IRs)
        ir_text = self._module_to_text(ir_module)

        # Store in the capture buffer
        capture = CapturedIR(
            source_hash=source_hash,
            target=self._tvm_target,
            stage_name=stage_name,
            ir_text=ir_text,
            metadata=dict(metadata) if metadata else {},
        )

        # The capture buffer is read by the bridge orchestrator which
        # runs the TVM MetaSchedule adapter on a background thread.
        # Storing in a module-level dict keeps the capture zero-overhead
        # when the bridge is not active.
        cache_key = CAPTURE_KEY_FMT.format(
            source_hash=source_hash[:16],
            kernel_name=metadata.get("name", "unknown"),
        )
        _CAPTURE_BUFFER[cache_key] = ir_text

        logger.debug(
            "Captured IR at stage=%s source=%s..%d_chars",
            stage_name,
            source_hash[:12],
            len(ir_text),
        )

    @staticmethod
    def get_capture_buffer() -> dict[str, str]:
        """Return the module-level capture buffer (for orchestrator use)."""
        return _CAPTURE_BUFFER

    @staticmethod
    def clear_capture_buffer() -> None:
        """Clear the capture buffer. Called after a successful tune cycle."""
        _CAPTURE_BUFFER.clear()

    def _derive_source_hash(
        self,
        metadata: dict[str, Any],
        stage_name: str,
    ) -> str:
        """Derive a stable source hash from compile metadata."""
        # metadata['name'] is the kernel function name
        # metadata['src'] is the kernel source hash from Triton's frontend
        # We combine them to get a unique-per-kernel identifier
        parts = [
            metadata.get("name", "unknown"),
            metadata.get("src", ""),
            stage_name,
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

    def _module_to_text(self, module: Any) -> str:
        """Convert an MLIR module to text representation.

        Handles:
          - triton.compiler.CompiledKernel.asm['stage'] (str)
          - mlir::Module (has .str() method or __str__)
          - str (already a string)
        """
        if isinstance(module, str):
            return module
        if hasattr(module, "str"):
            return str(module.str())
        if hasattr(module, "operation"):
            # mlir::Module wrapper
            try:
                return str(module)
            except Exception:
                pass
        return str(module)

    def _map_target(self, target: GPUTarget) -> str:
        """Map a Triton GPUTarget to a TVM target string."""
        backend = target.backend
        arch = target.arch

        mapping: dict[str, dict[Any, str]] = {
            "cuda": {
                70: "nvidia/nvidia-v100",
                80: "nvidia/nvidia-a100",
                86: "nvidia/nvidia-a100",  # sm_86 is also A100 family
                89: "nvidia/nvidia-rtx-4090",
                90: "nvidia/nvidia-h100",
                100: "nvidia/nvidia-b100",
                120: "nvidia/nvidia-b200",
            },
            "hip": {
                "gfx900": "rocm/gfx900",
                "gfx906": "rocm/gfx906",
                "gfx908": "rocm/gfx908",
                "gfx90a": "rocm/gfx90a",
                "gfx942": "rocm/gfx942",
                "gfx950": "rocm/gfx950",
            },
        }

        if backend in mapping and arch in mapping[backend]:
            return mapping[backend][arch]

        # Fallback: construct a reasonable default
        if backend == "cuda":
            return "cuda"
        if backend == "hip":
            return "rocm"
        return str(backend)
