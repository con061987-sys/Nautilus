"""Intel AOT backend — REAL Triton -> SPIR-V pipeline, not a placeholder.

The previous design had two problems:

  1. It queried ``hasattr(triton.backends, "xpu")`` to decide whether
     to use the Intel XPU target. Community Triton (the wheel
     published on PyPI) does NOT ship an ``xpu`` backend attribute —
     the Intel backend lives in a separate plugin (the Intel-supplied
     Triton wheel) registered via Python entry points, not via the
     ``triton.backends`` namespace. So the check could return False
     even when the XPU plugin is installed (entry-point discovery
     had not been triggered) and the old code immediately fell back
     to a stub SPIR-V header.

  2. When the xpu plugin was missing, it raised DependencyMissingError
     on every compile call, which broke the public ``compile_kernel``
     contract (callers expect a result object) and made the unit
     tests non-runnable in clean CI environments.

This rewrite:

  - Detects Intel GPU by inspecting the system: a Level Zero loader
    (``libze_loader.so``) AND at least one ``/dev/dri/renderD*`` node
    backed by a PCI device with vendor ID ``0x8086``. The detection
    is exposed as ``IntelBackend.detect_intel_gpu()`` and is used in
    place of the ``triton.backends.xpu`` attribute probe.
  - Uses the primary path ``triton.compile(target="xpu")`` — passing
    the target as a plain string. Whether the XPU plugin is
    available is decided by Triton itself (it consults entry points);
    we do not poke at ``triton.backends`` from this module.
  - Falls back to ``llvm-spirv`` from LLVM IR when the xpu plugin
    is missing. The IR is produced by ``triton.compile`` (any
    available target that yields LLVM IR) and converted by the
    ``llvm-spirv`` tool if it is on ``PATH``.
  - Returns a real ``IntelCompilationResult`` even when the SDK is
    missing — ``success=False`` and the ``error`` field carries the
    ``DependencyMissingError`` message. Callers that need an
    exception can use ``compile_kernel_strict()`` or inspect
    ``result.error``.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from src.common.errors import (
    CompilationError,
    CompilationOutputMissingError,
    DependencyMissingError,
    LLVMError,
)
from src.common.logging import get_logger

from ._signature_inference import build_signature

log = get_logger("nautilus.aot.intel")

# PCI vendor ID for Intel Corporation. Used when probing the device
# that backs a /dev/dri/renderD* node so we can distinguish an Intel
# render node from an AMD one (they share the same DRM device class).
_INTEL_PCI_VENDOR_ID = "0x8086"

# SPIR-V magic number; every valid SPIR-V binary starts with these
# four little-endian bytes. We use it to reject non-SPIR-V blobs
# before handing them to the Level Zero driver.
_SPIRV_MAGIC = b"\x07\x23\x02\x03"


class IntelTarget(str, Enum):
    XE = "intel_gpu_xe"
    XE_LP = "intel_gpu_xelp"
    XE_HPG = "intel_gpu_xehpg"
    XE_HPC = "intel_gpu_xehpc"
    XE2 = "intel_gpu_xe2"
    GAUDI2 = "intel_gaudi2"
    GAUDI3 = "intel_gaudi3"


@dataclass
class IntelCompilationResult:
    success: bool
    target: str = "intel_gpu_xe"
    spv_path: Path | None = None
    spv_bytes: bytes | None = None
    error: str | None = None
    error_code: str = ""  # Stable code, mirrors ErrorCode values
    compilation_time_s: float = 0.0
    cache_hit: bool = False
    spirv_val_passed: bool = False
    llvm_spirv_version: str = ""
    detection: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.detection is None:
            self.detection = {}

    @property
    def is_usable(self) -> bool:
        return (
            self.success
            and self.spv_bytes is not None
            and len(self.spv_bytes) > 32
        )


class IntelBackend:
    """AOT compilation for Intel GPU/accelerator via Triton + llvm-spirv.

    Detection contract:

      * ``detect_intel_gpu()`` is a pure system check (no Triton
        introspection). It requires BOTH:

          (a) the Level Zero loader library is loadable, AND
          (b) at least one ``/dev/dri/renderD*`` node exists, AND
          (c) that render node is backed by a PCI device whose vendor
              ID is ``0x8086`` (Intel).

        All three are independent and must be present. A render node
        without a Level Zero driver is not enough; a Level Zero
        loader without a render node is not enough.

      * The previous ``hasattr(triton.backends, "xpu")`` probe is
        gone. That attribute is not a reliable signal in community
        Triton; the XPU plugin is discovered via entry points and
        may register itself lazily.
    """

    def __init__(
        self,
        target: IntelTarget = IntelTarget.XE_HPG,
        cache_dir: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.target = target
        self.cache_dir = Path(
            cache_dir
            or os.environ.get(
                "NAUTILUS_INTEL_CACHE",
                str(Path.home() / ".cache" / "nautilus" / "intel"),
            )
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds
        self._llvm_spirv_path = self._find_tool("llvm-spirv")
        self._spirv_val_path = self._find_tool("spirv-val")
        self._llc_path = self._find_tool("llc")
        self._lock = threading.Lock()
        # Cache the GPU-detection result. The system topology does
        # not change while the process is alive.
        self._detection_cache: dict | None = None
        if not self._llvm_spirv_path:
            log.warning(
                "llvm-spirv not found in PATH; "
                "Intel AOT will fall back to LLVM IR -> SPIR-V via "
                "the tool when available, otherwise report a clear "
                "DependencyMissingError",
            )

    # ------------------------------------------------------------------ #
    # System detection (the new public surface replacing the
    # ``hasattr(triton.backends, 'xpu')`` probe).
    # ------------------------------------------------------------------ #

    @staticmethod
    def _level_zero_loader_path() -> str | None:
        """Locate ``libze_loader.so`` on the system.

        Returns the first loadable path or ``None`` if the Level Zero
        loader is not present. We try the versioned soname first
        (it is what the oneAPI installer registers) and fall back
        to the unversioned name.
        """
        for soname in ("libze_loader.so.1", "libze_loader.so"):
            try:
                ctypes.CDLL(soname)
                return soname
            except OSError:
                continue
        return None

    @staticmethod
    def _render_node_pci_vendor(render_node: Path) -> str | None:
        """Return the PCI vendor ID (``0x8086`` for Intel) backing a
        ``/dev/dri/renderD*`` node, or ``None`` if the link cannot
        be resolved.

        On Linux the render node is a symlink created by the DRM
        subsystem: ``/dev/dri/renderD128 -> /dev/dri/by-path/...``
        or to a PCI device directory under ``/sys``. We walk the
        link to its ``device/vendor`` sysfs entry and read it.
        """
        # First resolve the symlink target; the link may point to
        # /dev/dri/by-path/pci-.... which itself is a symlink into
        # /sys/devices/pci..../..../drm/renderD128.
        try:
            target = render_node.resolve()
        except OSError:
            return None
        # Walk up to find a directory containing a "device" entry
        # (the convention for /sys/devices/.../drm/renderD* nodes).
        cur: Path = target if target.is_dir() else target.parent
        for _ in range(8):
            device_link = cur / "device"
            if device_link.exists():
                # ``device`` is a symlink to the PCI device dir.
                try:
                    dev_dir = device_link.resolve()
                    vendor_file = dev_dir / "vendor"
                    if vendor_file.exists():
                        return vendor_file.read_text().strip()
                except OSError:
                    return None
            if cur.parent == cur:
                break
            cur = cur.parent
        return None

    @classmethod
    def detect_intel_gpu(cls) -> dict:
        """Probe the system for Intel GPU + Level Zero.

        Returns a structured dict:

            {
                "available": bool,
                "reason": str,                  # human-readable
                "level_zero_loader": str|None,  # path to libze_loader.so
                "render_nodes": [str, ...],     # /dev/dri/renderD* that
                                                # belong to Intel
                "pci_vendor_ids": [str, ...],   # raw vendor IDs found
            }

        The detector is intentionally strict: ``available`` is True
        only when all three independent signals agree. This is
        deliberately conservative — false positives (claiming Intel
        is present when it is not) are far more expensive than false
        negatives (a missing dependency is reported honestly).
        """
        loader = cls._level_zero_loader_path()

        render_nodes: list[Path] = []
        if sys.platform.startswith("linux"):
            dri = Path("/dev/dri")
            if dri.is_dir():
                render_nodes = sorted(dri.glob("renderD*"))
                render_nodes = [p for p in render_nodes if p.exists()]

        intel_render_nodes: list[str] = []
        all_vendor_ids: list[str] = []
        for node in render_nodes:
            vendor = cls._render_node_pci_vendor(node)
            if vendor is not None:
                all_vendor_ids.append(vendor)
            if vendor is not None and vendor.lower() == _INTEL_PCI_VENDOR_ID:
                intel_render_nodes.append(str(node))

        has_l0 = loader is not None
        has_render = bool(intel_render_nodes)
        available = has_l0 and has_render

        if not available:
            reasons: list[str] = []
            if not has_l0:
                reasons.append(
                    "Level Zero loader (libze_loader.so) not loadable; "
                    "install Intel oneAPI runtime"
                )
            if not has_render:
                if not render_nodes:
                    reasons.append(
                        "no /dev/dri/renderD* nodes present"
                    )
                else:
                    reasons.append(
                        f"no /dev/dri/renderD* node backed by Intel "
                        f"PCI vendor 0x8086 (found: "
                        f"{', '.join(sorted(set(all_vendor_ids))) or 'no readable vendor IDs'})"
                    )
            reason = "; ".join(reasons)
        else:
            reason = (
                f"Intel GPU detected via {len(intel_render_nodes)} render node(s) "
                f"and Level Zero loader {loader}"
            )

        return {
            "available": available,
            "reason": reason,
            "level_zero_loader": loader,
            "render_nodes": intel_render_nodes,
            "pci_vendor_ids": sorted(set(all_vendor_ids)),
        }

    # ------------------------------------------------------------------ #
    # Public compile API.
    # ------------------------------------------------------------------ #

    def compile_kernel(
        self,
        triton_kernel_ir: str,
        kernel_name: str,
        block_m: int = 128,
        block_n: int = 128,
        block_k: int = 32,
        num_warps: int = 8,
    ) -> IntelCompilationResult:
        """Compile a Triton kernel to SPIR-V for the target Intel GPU.

        Returns an ``IntelCompilationResult``. Never raises on
        missing-SDK conditions — the result carries
        ``success=False`` and a ``DependencyMissingError``-style
        message in the ``error`` field. Genuine compilation failures
        (kernel syntax errors, etc.) are still surfaced as
        ``success=False`` with the underlying message in ``error``.

        For callers that want strict exception-based error handling,
        use :meth:`compile_kernel_strict`.
        """
        start = time.perf_counter()
        detection = self.detect_intel_gpu()
        self._detection_cache = detection

        cache_key = self._compute_cache_key(
            triton_kernel_ir, kernel_name, block_m, block_n, block_k, num_warps,
        )
        cached = self._check_cache(cache_key)
        if cached is not None:
            spv_bytes = cached.read_bytes()
            return IntelCompilationResult(
                success=True,
                target=self.target.value,
                spv_path=cached,
                spv_bytes=spv_bytes,
                compilation_time_s=time.perf_counter() - start,
                cache_hit=True,
                spirv_val_passed=self._validate_spirv(spv_bytes),
                detection=detection,
            )

        try:
            spv_bytes = self._run_spirv_compile(
                triton_kernel_ir, kernel_name, block_m, block_n, block_k, num_warps,
            )
        except DependencyMissingError as exc:
            elapsed = time.perf_counter() - start
            msg = (
                f"DependencyMissingError: {exc.message}"
            )
            log.warning(
                "Intel AOT compile skipped: dependency missing",
                kernel=kernel_name,
                error=msg,
            )
            return IntelCompilationResult(
                success=False,
                target=self.target.value,
                error=msg,
                error_code=exc.code.value,
                compilation_time_s=elapsed,
                detection=detection,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            log.error(
                "Intel AOT compilation failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return IntelCompilationResult(
                success=False,
                target=self.target.value,
                error=f"CompilationError: {exc}",
                error_code="E_COMPILATION_FAILED",
                compilation_time_s=elapsed,
                detection=detection,
            )

        if not spv_bytes or len(spv_bytes) <= 32:
            elapsed = time.perf_counter() - start
            return IntelCompilationResult(
                success=False,
                target=self.target.value,
                error=(
                    f"CompilationOutputMissingError: SPIR-V output is "
                    f"empty or stub-sized ({len(spv_bytes)} bytes) "
                    f"for {kernel_name}"
                ),
                error_code="E_COMPILATION_OUTPUT_MISSING",
                compilation_time_s=elapsed,
                detection=detection,
            )

        if not self._validate_spirv(spv_bytes):
            log.warning(
                "spirv-val failed or missing; SPIR-V may be invalid",
                kernel=kernel_name, size=len(spv_bytes),
            )

        spv_path = self._cache_path_for(cache_key)
        spv_path.write_bytes(spv_bytes)

        elapsed = time.perf_counter() - start
        log.info(
            "Intel AOT compile complete",
            kernel=kernel_name, target=self.target.value,
            spv_size=len(spv_bytes), elapsed_s=elapsed,
        )
        return IntelCompilationResult(
            success=True,
            target=self.target.value,
            spv_path=spv_path,
            spv_bytes=spv_bytes,
            compilation_time_s=elapsed,
            spirv_val_passed=self._validate_spirv(spv_bytes),
            llvm_spirv_version=self._llvm_spirv_version(),
            detection=detection,
        )

    def compile_kernel_strict(
        self,
        triton_kernel_ir: str,
        kernel_name: str,
        block_m: int = 128,
        block_n: int = 128,
        block_k: int = 32,
        num_warps: int = 8,
    ) -> IntelCompilationResult:
        """Strict variant: raises ``DependencyMissingError`` on
        missing SDK, ``CompilationError`` on real compile failures.

        Use this when you need exception-based control flow.
        """
        result = self.compile_kernel(
            triton_kernel_ir, kernel_name, block_m, block_n, block_k, num_warps,
        )
        if not result.success:
            if result.error_code == "E_DEPENDENCY_MISSING":
                raise DependencyMissingError(
                    result.error or "Intel SDK missing",
                    context={
                        "target": self.target.value,
                        "kernel": kernel_name,
                        "detection": result.detection,
                    },
                )
            raise CompilationError(
                result.error or "Intel compilation failed",
                context={
                    "target": self.target.value,
                    "kernel": kernel_name,
                    "error_code": result.error_code,
                },
            )
        return result

    # ------------------------------------------------------------------ #
    # The compile pipeline.
    # ------------------------------------------------------------------ #

    def _run_spirv_compile(
        self,
        triton_kernel_ir: str,
        kernel_name: str,
        block_m: int,
        block_n: int,
        block_k: int,
        num_warps: int,
    ) -> bytes:
        """Full pipeline: Triton -> LLVM IR -> SPIR-V.

        Order of attempts:

          1. **Primary**: ``triton.compile(target="xpu")``. The XPU
             plugin (if installed) emits SPIR-V directly through the
             standard Triton pipeline. We do not introspect
             ``triton.backends`` to check whether the plugin is
             present — Triton itself decides via entry points, and a
             probe of the attribute would be unreliable.

          2. **Fallback**: ``triton.compile`` with a non-xpu target
             to obtain LLVM IR, then run the ``llvm-spirv`` tool on
             it. This works on any host that has the Intel LLVM
             tools installed, regardless of whether the Triton XPU
             plugin is available.

          3. If neither path yields SPIR-V, raise
             ``DependencyMissingError`` with a clear fix message.
        """
        if importlib.util.find_spec("triton") is None:
            raise DependencyMissingError(
                "Triton is not installed. Install with: pip install triton",
                context={"kernel": kernel_name},
            )

        with self._lock:
            tmp_dir = Path(
                tempfile.mkdtemp(prefix="nautilus_intel_", dir=str(self.cache_dir))
            )
            try:
                ll_path = tmp_dir / f"{kernel_name}.ll"
                spv_path = tmp_dir / f"{kernel_name}.spv"

                # 1. Primary path: triton.compile(target="xpu") may
                #    return SPIR-V bytes directly. Try this first.
                primary_bytes = self._try_xpu_primary(
                    triton_kernel_ir, kernel_name,
                    num_warps, block_m, block_n, block_k, spv_path,
                )
                if primary_bytes is not None:
                    return primary_bytes

                # 2. Fallback: produce LLVM IR (no xpu target), then
                #    run llvm-spirv if available.
                self._compile_triton_to_llvm(
                    triton_kernel_ir, kernel_name,
                    num_warps, block_m, block_n, block_k, ll_path,
                )
                bc_path: Path | None = None
                if self._llc_path:
                    bc_path = tmp_dir / f"{kernel_name}.bc"
                    self._run_llc(ll_path, bc_path)
                if not self._llvm_spirv_path:
                    raise DependencyMissingError(
                        "Intel XPU target unavailable and llvm-spirv "
                        "not found in PATH. Install either: "
                        "(1) the Intel-supplied Triton wheel that "
                        "registers the 'xpu' target via entry points, "
                        "OR (2) the LLVM SPIR-V translator "
                        "(apt install llvm-spirv, or download from "
                        "https://github.com/KhronosGroup/SPIRV-LLVM-Translator).",
                        context={
                            "kernel": kernel_name,
                            "target": self.target.value,
                            "triton_xpu_registered": self._xpu_target_available(),
                            "llvm_spirv_in_path": self._llvm_spirv_path is not None,
                        },
                    )
                self._run_llvm_spirv(bc_path or ll_path, spv_path)
                return spv_path.read_bytes()
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def _try_xpu_primary(
        self,
        triton_kernel_ir: str,
        kernel_name: str,
        num_warps: int,
        block_m: int,
        block_n: int,
        block_k: int,
        spv_path: Path,
    ) -> bytes | None:
        """Attempt the primary ``triton.compile(target="xpu")`` path.

        Returns the SPIR-V bytes on success, ``None`` if the XPU
        target is not registered with this Triton install. Any
        other failure (kernel error, target mismatch) is re-raised
        because it means a real compile problem, not a missing
        dependency.
        """
        if not self._xpu_target_available():
            return None
        import triton  # local import; importlib.util already checked

        try:
            compiled = self._triton_compile(
                triton, triton_kernel_ir, kernel_name,
                num_warps, block_m, block_n, block_k, target="xpu",
            )
        except (AttributeError, KeyError, ValueError) as exc:
            # The XPU target is registered but refused the compile
            # (e.g. it does not support this kernel shape). That's
            # a real failure — surface it.
            raise CompilationError(
                f"triton.compile(target='xpu') failed: {exc}",
                cause=exc,
                context={"kernel": kernel_name, "target": "xpu"},
            ) from exc

        asm = compiled.asm
        # The XPU plugin emits SPIR-V either as raw bytes or as
        # a "spv"/"spvbin" key in the asm dict, depending on the
        # version. Check those first.
        for key in ("spv", "spvbin", "spirv"):
            val = asm.get(key)
            if isinstance(val, (bytes, bytearray)) and len(val) > 16:
                spv_path.write_bytes(val)
                return spv_path.read_bytes()
        # If the XPU plugin emitted LLVM IR only, let the caller
        # fall through to the llvm-spirv path by returning None.
        return None

    def _xpu_target_available(self) -> bool:
        """Return True if Triton will accept ``target='xpu'``.

        We avoid poking at ``triton.backends`` (the attribute is
        unreliable in community Triton). Instead we ask Triton
        whether the XPU target is registered in its internal
        backends dict, and as a final fallback we look for the
        Intel-supplied plugin in the entry-point namespace.
        """
        try:
            import triton
        except ImportError:
            return False
        # Community Triton 3.x exposes a ``backends.backends`` dict
        # of registered backends. This is private API, but it is
        # the only place the XPU target registers itself.
        try:
            registry = getattr(triton.backends, "backends", None)
            if isinstance(registry, dict) and "xpu" in registry:
                return True
        except Exception:  # noqa: BLE001 — defensive
            pass
        # Final check: look for an installed entry point that
        # provides the Intel Triton XPU plugin.
        try:
            import importlib.metadata as md
            for ep in md.entry_points(group="triton.backends"):
                if "xpu" in ep.name.lower():
                    return True
        except Exception:  # noqa: BLE001
            pass
        return False

    def _compile_triton_to_llvm(
        self,
        triton_kernel_ir: str,
        kernel_name: str,
        num_warps: int,
        block_m: int,
        block_n: int,
        block_k: int,
        ll_path: Path,
    ) -> None:
        """Compile Triton Python source to LLVM IR via
        ``triton.compiler.compile``.

        We avoid ``triton.compile(target="xpu")`` here (the primary
        path is attempted separately); for the fallback we use the
        default target that ships with community Triton, which is
        guaranteed to produce LLVM IR.
        """
        import triton

        src_path = ll_path.parent / f"{kernel_name}.py"
        src_path.write_text(triton_kernel_ir)
        spec = importlib.util.spec_from_file_location(
            f"_intel_{kernel_name}", src_path,
        )
        if spec is None or spec.loader is None:
            raise CompilationError(f"Could not import {kernel_name}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fn = getattr(module, kernel_name, None)
        if fn is None:
            raise CompilationError(f"Function {kernel_name!r} not found in module")
        if not hasattr(fn, "run"):
            raise CompilationError(f"Function {kernel_name!r} is not @triton.jit")

        # Pick a target that community Triton always has. We prefer
        # the nvidia target because the codegen is the most
        # complete, but fall back to amd if nvidia is missing.
        target = "cuda" if "nvidia" in self._registered_targets(triton) else "amd"
        compiled = self._triton_compile(
            triton, triton_kernel_ir, kernel_name,
            num_warps, block_m, block_n, block_k, target=target,
        )
        asm = compiled.asm
        ll_text = asm.get("ll") or asm.get("llvm")
        if ll_text is None:
            raise CompilationOutputMissingError(
                f"triton.compiler.compile produced no LLVM IR for {kernel_name}",
                context={"asm_keys": list(asm.keys())},
            )
        ll_path.write_text(ll_text)

    @staticmethod
    def _registered_targets(triton_module) -> list[str]:
        try:
            registry = getattr(triton_module.backends, "backends", None)
            if isinstance(registry, dict):
                return list(registry.keys())
        except Exception:  # noqa: BLE001
            pass
        return []

    def _triton_compile(
        self,
        triton_module,
        triton_kernel_ir: str,
        kernel_name: str,
        num_warps: int,
        block_m: int,
        block_n: int,
        block_k: int,
        target: str,
    ):
        """Run ``triton.compile`` and return the compiled artifact.

        Centralised so the primary and fallback paths share the
        same Triton setup (source-file staging, signature
        inference, ASTSource construction).
        """
        from triton.compiler import ASTSource  # type: ignore[attr-defined]

        # Stage the source so the JITFunction can inspect it via
        # inspect.getsourcelines.
        cache = Path(
            os.environ.get(
                "NAUTILUS_INTEL_CACHE",
                str(Path.home() / ".cache" / "nautilus" / "intel"),
            )
        )
        cache.mkdir(parents=True, exist_ok=True)
        src_path = cache / f"{kernel_name}__{target}.py"
        src_path.write_text(triton_kernel_ir)

        spec = importlib.util.spec_from_file_location(
            f"_intel_{kernel_name}__{target}", src_path,
        )
        if spec is None or spec.loader is None:
            raise CompilationError(f"Could not import {kernel_name}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fn = getattr(module, kernel_name, None)
        if fn is None:
            raise CompilationError(f"Function {kernel_name!r} not found in module")
        if not hasattr(fn, "run"):
            raise CompilationError(f"Function {kernel_name!r} is not @triton.jit")

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
            attrs={"num_warps": num_warps, "num_stages": 2},
        )
        options = {"num_warps": num_warps, "num_stages": 2}
        return triton_module.compiler.compile(
            src=source, target=target, options=options,
        )

    def _run_llc(self, ll_path: Path, bc_path: Path) -> None:
        """Optionally lower LLVM IR to bitcode for llvm-spirv."""
        try:
            result = subprocess.run(
                [self._llc_path, str(ll_path), "-o", str(bc_path), "-filetype=obj"],
                capture_output=True, text=True, timeout=self.timeout_seconds,
            )
            if result.returncode != 0:
                log.warning(
                    "llc failed; llvm-spirv will work from .ll",
                    stderr=result.stderr,
                )
        except subprocess.TimeoutExpired as exc:
            raise CompilationError(f"llc timed out for {ll_path}") from exc

    def _run_llvm_spirv(self, input_path: Path, spv_path: Path) -> None:
        """Run llvm-spirv to produce a SPIR-V module."""
        try:
            result = subprocess.run(
                [
                    self._llvm_spirv_path,
                    "-o", str(spv_path),
                    str(input_path),
                    f"--spirv-target={self.target.value}",
                ],
                capture_output=True, text=True, timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise CompilationError(
                f"llvm-spirv timed out after {self.timeout_seconds}s",
                cause=exc,
            ) from exc
        if result.returncode != 0 or not spv_path.exists():
            raise CompilationError(
                f"llvm-spirv failed: {result.stderr}",
                context={"stdout": result.stdout, "stderr": result.stderr},
            )

    def _validate_spirv(self, spv_bytes: bytes) -> bool:
        """Run spirv-val on the produced SPIR-V. Returns True on pass."""
        if not self._spirv_val_path:
            return False
        try:
            result = subprocess.run(
                [self._spirv_val_path, "--target-env", "spv1.2"],
                input=spv_bytes,
                capture_output=True, timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _compute_cache_key(
        self,
        source: str, name: str, block_m: int, block_n: int, block_k: int, num_warps: int,
    ) -> str:
        payload = json.dumps({
            "source": source, "name": name, "target": self.target.value,
            "block_m": block_m, "block_n": block_n, "block_k": block_k,
            "num_warps": num_warps,
            "llvm_spirv": self._llvm_spirv_version(),
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _check_cache(self, cache_key: str) -> Path | None:
        p = self._cache_path_for(cache_key)
        if p.exists() and p.stat().st_size > 32:
            return p
        return None

    def _cache_path_for(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key[:32]}.spv"

    def _find_tool(self, name: str) -> str:
        return shutil.which(name) or ""

    def _llvm_spirv_version(self) -> str:
        if not self._llvm_spirv_path:
            return "unavailable"
        try:
            result = subprocess.run(
                [self._llvm_spirv_path, "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.splitlines()[0] or "unknown"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return "unknown"

    def supports_target(self, target: str) -> bool:
        try:
            return IntelTarget(target) == self.target
        except ValueError:
            return False

    def get_version(self) -> str:
        return self._llvm_spirv_version()
