"""
src.c_api — Python ctypes bindings for the C-API headers.

Loads the compiled shared library (libnautilus_c_api.so / .dylib / .dll)
and exposes Pythonic access to the C functions. The C library is built
separately via CMake (see src/c_api/CMakeLists.txt).

The C headers are the SOURCE OF TRUTH for the ABI. This Python module
mirrors them; if you change a header, you must update this file too.

Three sub-APIs are exposed, each backed by a different C wrapper:

* Triton C-API (``triton_c_api.h``) — compile a Triton @triton.jit
  kernel to a vendor-specific binary (PTX, CUBIN, HSACO, SPIR-V,
  Metal AIR).

* TVM C-API (``tvm_c_api.h``) — parse a TIR / TVMScript module and
  run MetaSchedule autotuning to obtain a tuning record.

* XLA C-API (``xla_c_api.h``) — build a StableHLO module from a
  TorchFX graph, create a device mesh, and run GSPMD auto-sharding.

All three wrappers dlopen() the corresponding upstream library at
runtime. The Python layer is the only place that has to know about
the version-drift rules: when an upstream ABI changes, only the
symbol table in the affected C wrapper is updated, never the
Python side.

Usage
-----

    from src.c_api import triton_c_api, tvm_c_api, xla_c_api

    rc, handle = triton_c_api.compile(
        source=triton_source,
        kernel_name="matmul_kernel",
        vendor=triton_c_api.VENDOR_NVIDIA,
        arch=triton_c_api.ARCH_SM_90,
        num_warps=8, num_stages=3,
        block_m=128, block_n=128, block_k=32,
    )
    if rc == 0:
        data, size, fmt = triton_c_api.get_binary(handle)
        ...
        triton_c_api.release(handle)

If the shared library is not built (e.g. you ran `pip install -e .`
without the [cpp] extra), calls raise DependencyMissingError with a
clear message about how to build it.
"""

from __future__ import annotations

import ctypes
import os
from ctypes import (
    CDLL,
    POINTER,
    c_char_p,
    c_int,
    c_int64,
    c_size_t,
    c_uint8,
    c_void_p,
)
from pathlib import Path
from typing import Any

from src.common.errors import (
    DependencyMissingError,
    ErrorCode,
    NautilusError,
)

# --- Library loading ---


_C_LIB_PATHS: list[str] = [
    p for p in [
        os.environ.get("NAUTILUS_C_LIB"),
        str(Path(__file__).parent / "libnautilus_c_api.so"),
        str(Path(__file__).parent / "libnautilus_c_api.dylib"),
        str(Path(__file__).parent / "libnautilus_c_api.dll"),
        "libnautilus_c_api.so",
        "libnautilus_c_api.dylib",
        "libnautilus_c_api.dll",
    ] if p
]

_C_LIB: CDLL | None = None
_C_LIB_LOAD_ERROR: str | None = None


def _load_c_lib() -> CDLL:
    """Load the shared library or raise DependencyMissingError."""
    global _C_LIB, _C_LIB_LOAD_ERROR
    if _C_LIB is not None:
        return _C_LIB
    last_err: str = ""
    for path in _C_LIB_PATHS:
        if not path:
            continue
        try:
            lib = CDLL(path)
            _C_LIB = lib
            _set_function_signatures(lib)
            return lib
        except OSError as exc:
            last_err = f"{path}: {exc}"
    msg = (
        "Could not load libnautilus_c_api. "
        "Build it via `cmake -S src/c_api -B build && cmake --build build` "
        "or set NAUTILUS_C_LIB=/path/to/lib. "
        f"Last error: {last_err}"
    )
    _C_LIB_LOAD_ERROR = msg
    raise DependencyMissingError(
        msg,
        context={"last_load_error": last_err},
    )


def _set_function_signatures(lib: CDLL) -> None:
    """Declare C function signatures for ctypes type-safety.

    Critical: without this, ctypes treats everything as int and we get
    garbage on 64-bit systems. The signatures here MUST match the
    public C headers in src/c_api/*.h — when you change a header,
    update this function.
    """
    # --- Triton C-API ---
    lib.nautilus_compile.argtypes = [
        c_char_p, c_char_p, c_int, c_int,
        c_int, c_int, c_int, c_int, c_int,
        POINTER(c_void_p),
    ]
    lib.nautilus_compile.restype = c_int

    lib.nautilus_get_binary.argtypes = [
        c_void_p, POINTER(POINTER(c_uint8)), POINTER(c_size_t), POINTER(c_char_p),
    ]
    lib.nautilus_get_binary.restype = c_int

    lib.nautilus_release.argtypes = [c_void_p]
    lib.nautilus_release.restype = None

    lib.nautilus_set_tuning_param.argtypes = [c_void_p, c_char_p, c_int64]
    lib.nautilus_set_tuning_param.restype = c_int

    lib.nautilus_last_error_message.argtypes = []
    lib.nautilus_last_error_message.restype = c_char_p

    lib.nautilus_triton_version.argtypes = []
    lib.nautilus_triton_version.restype = c_char_p

    # --- TVM C-API ---
    lib.nautilus_tir_parse.argtypes = [
        c_char_p, c_size_t, c_char_p, POINTER(c_void_p),
    ]
    lib.nautilus_tir_parse.restype = c_int

    lib.nautilus_tir_release.argtypes = [c_void_p]
    lib.nautilus_tir_release.restype = None

    lib.nautilus_tune.argtypes = [
        c_void_p, c_int, c_int, c_int, c_int, POINTER(c_void_p),
    ]
    lib.nautilus_tune.restype = c_int

    lib.nautilus_tuning_record_release.argtypes = [c_void_p]
    lib.nautilus_tuning_record_release.restype = None

    lib.nautilus_record_get_int.argtypes = [c_void_p, c_char_p, POINTER(c_int64)]
    lib.nautilus_record_get_int.restype = c_int

    lib.nautilus_tvm_last_error_message.argtypes = []
    lib.nautilus_tvm_last_error_message.restype = c_char_p

    lib.nautilus_tvm_version.argtypes = []
    lib.nautilus_tvm_version.restype = c_char_p

    # --- XLA C-API ---
    lib.nautilus_stablehlo_from_fx.argtypes = [
        c_char_p, c_size_t, c_char_p, POINTER(c_void_p),
    ]
    lib.nautilus_stablehlo_from_fx.restype = c_int

    lib.nautilus_stablehlo_release.argtypes = [c_void_p]
    lib.nautilus_stablehlo_release.restype = None

    lib.nautilus_stablehlo_get_mlir_text.argtypes = [
        c_void_p, POINTER(c_char_p), POINTER(c_size_t),
    ]
    lib.nautilus_stablehlo_get_mlir_text.restype = c_int

    lib.nautilus_mesh_create.argtypes = [
        POINTER(c_int64), c_size_t, POINTER(c_void_p),
    ]
    lib.nautilus_mesh_create.restype = c_int

    lib.nautilus_mesh_release.argtypes = [c_void_p]
    lib.nautilus_mesh_release.restype = None

    lib.nautilus_gspmd_shard.argtypes = [
        c_void_p, c_void_p, c_int, c_int, POINTER(c_void_p),
    ]
    lib.nautilus_gspmd_shard.restype = c_int

    lib.nautilus_sharding_spec_release.argtypes = [c_void_p]
    lib.nautilus_sharding_spec_release.restype = None

    lib.nautilus_xla_last_error_message.argtypes = []
    lib.nautilus_xla_last_error_message.restype = c_char_p

    lib.nautilus_xla_version.argtypes = []
    lib.nautilus_xla_version.restype = c_char_p


# --- Pythonic wrappers that raise on error ---


class CApiUnavailable(DependencyMissingError):
    """Raised when the C-API shared library is not built."""
    code = ErrorCode.DEPENDENCY_MISSING


def _check_rc(rc: int, op: str, last_error_fn_name: str = "nautilus_last_error_message") -> None:
    """Raise an exception if a C function returned non-zero.

    Each sub-API has its own ``nautilus_*_last_error_message`` slot
    (triton, tvm, xla); the caller passes the right one. The default
    is the triton slot for backward compatibility.

    Backend-missing errors (NAUTILUS_ERR_BACKEND_MISSING /
    NAUTILUS_TUNING_ERR_BACKEND / NAUTILUS_XLA_ERR_*) are translated
    to DependencyMissingError so callers can use a single ``except``
    branch for "I need to install something to make this work".
    """
    if rc == 0:
        return
    try:
        lib = _load_c_lib()
        msg_fn = getattr(lib, last_error_fn_name)
        msg = msg_fn().decode("utf-8", errors="replace")
    except Exception:
        msg = f"unknown error (rc={rc})"
    is_backend_missing = rc in (
        -3,   # NAUTILUS_ERR_BACKEND_MISSING (triton_c_api.h)
        -5,   # NAUTILUS_TUNING_ERR_BACKEND (tvm_c_api.h)
    )
    exc_cls: type[NautilusError] = DependencyMissingError if is_backend_missing else NautilusError
    raise exc_cls(
        f"C-API call {op!r} failed: {msg}",
        context={"operation": op, "return_code": rc},
    )


# =====================================================================
# Triton C-API
# =====================================================================


VENDOR_NVIDIA = 0
VENDOR_AMD = 1
VENDOR_INTEL = 2
VENDOR_APPLE = 3
VENDOR_HOST = 4

ARCH_SM_70 = 70
ARCH_SM_75 = 75
ARCH_SM_80 = 80
ARCH_SM_86 = 86
ARCH_SM_89 = 89
ARCH_SM_90 = 90
ARCH_SM_100 = 100
ARCH_SM_120 = 120
ARCH_GFX900 = 900
ARCH_GFX906 = 906
ARCH_GFX908 = 908
ARCH_GFX90A = 910
ARCH_GFX942 = 942
ARCH_GFX950 = 950
ARCH_XE_LP = 1200
ARCH_XE_HPG = 1201
ARCH_XE_HPC = 1202
ARCH_XE2 = 1203
ARCH_GAUDI2 = 2002
ARCH_GAUDI3 = 2003


class TritonKernelHandle:
    """Python-side handle for a compiled Triton kernel.

    Holds a reference to the C handle; releases it on __exit__ / close().
    """
    def __init__(self, c_handle: int) -> None:
        self._c_handle = c_handle
        self._released = False

    @property
    def c_handle(self) -> int:
        if self._released:
            raise RuntimeError("Kernel handle already released")
        return self._c_handle

    def get_binary(self) -> tuple[bytes, int, str]:
        lib = _load_c_lib()
        out_data = POINTER(c_uint8)()
        out_size = c_size_t()
        out_format = c_char_p()
        rc = lib.nautilus_get_binary(
            c_void_p(self._c_handle),
            ctypes.byref(out_data),
            ctypes.byref(out_size),
            ctypes.byref(out_format),
        )
        _check_rc(rc, "nautilus_get_binary", "nautilus_last_error_message")
        data = bytes(
            ctypes.cast(out_data, POINTER(c_uint8 * out_size.value))[:]
        )
        fmt = out_format.value.decode("utf-8") if out_format.value else ""
        return data, out_size.value, fmt

    def set_tuning_param(self, name: str, value: int) -> None:
        lib = _load_c_lib()
        rc = lib.nautilus_set_tuning_param(
            c_void_p(self._c_handle),
            name.encode("utf-8"),
            int(value),
        )
        _check_rc(rc, "nautilus_set_tuning_param", "nautilus_last_error_message")

    def release(self) -> None:
        if not self._released:
            lib = _load_c_lib()
            lib.nautilus_release(c_void_p(self._c_handle))
            self._released = True

    def __enter__(self) -> TritonKernelHandle:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass


def compile(
    source: str,
    kernel_name: str,
    vendor: int,
    arch: int,
    num_warps: int,
    num_stages: int,
    block_m: int = 0,
    block_n: int = 0,
    block_k: int = 0,
) -> TritonKernelHandle:
    """Compile a Triton kernel. See triton_c_api.h for parameter docs."""
    lib = _load_c_lib()
    out_handle = c_void_p()
    rc = lib.nautilus_compile(
        source.encode("utf-8"),
        kernel_name.encode("utf-8"),
        c_int(vendor), c_int(arch),
        c_int(num_warps), c_int(num_stages),
        c_int(block_m), c_int(block_n), c_int(block_k),
        ctypes.byref(out_handle),
    )
    _check_rc(rc, "nautilus_compile", "nautilus_last_error_message")
    return TritonKernelHandle(out_handle.value or 0)


def triton_version() -> str:
    """Return the version string of the underlying Triton library."""
    lib = _load_c_lib()
    return lib.nautilus_triton_version().decode("utf-8", errors="replace")


# =====================================================================
# TVM C-API (MetaSchedule autotuning)
# =====================================================================


TUNING_AUTO = 0
TUNING_RL_EVOLUTIONARY = 1
TUNING_RL_GRADIENT = 2
TUNING_XGBOOST_COST_MODEL = 3


class TIRModuleHandle:
    """Python-side handle for a parsed TIR module."""
    def __init__(self, c_handle: int) -> None:
        self._c_handle = c_handle
        self._released = False

    @property
    def c_handle(self) -> int:
        if self._released:
            raise RuntimeError("TIR module handle already released")
        return self._c_handle

    def release(self) -> None:
        if not self._released:
            lib = _load_c_lib()
            lib.nautilus_tir_release(c_void_p(self._c_handle))
            self._released = True

    def __enter__(self) -> TIRModuleHandle:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass


class TuningRecordHandle:
    """Python-side handle for a MetaSchedule tuning record.

    Records carry integer parameters (BLOCK_M, BLOCK_N, BLOCK_K,
    num_warps, num_stages, ...). Floats / nested configs are not
    surfaced in the C ABI.
    """
    def __init__(self, c_handle: int) -> None:
        self._c_handle = c_handle
        self._released = False

    @property
    def c_handle(self) -> int:
        if self._released:
            raise RuntimeError("Tuning record handle already released")
        return self._c_handle

    def get_int(self, name: str) -> int:
        lib = _load_c_lib()
        out = c_int64(0)
        rc = lib.nautilus_record_get_int(
            c_void_p(self._c_handle),
            name.encode("utf-8"),
            ctypes.byref(out),
        )
        _check_rc(rc, "nautilus_record_get_int", "nautilus_tvm_last_error_message")
        return int(out.value)

    def release(self) -> None:
        if not self._released:
            lib = _load_c_lib()
            lib.nautilus_tuning_record_release(c_void_p(self._c_handle))
            self._released = True

    def __enter__(self) -> TuningRecordHandle:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass


def tir_parse(text: str, target: str) -> TIRModuleHandle:
    """Parse a TVMScript / TIR module from text.

    Args:
        text: TVMScript source (or TIR text).
        target: Upstream target spec, e.g. ``"nvidia/nvidia-h100"``,
            ``"rocm/gfx942"``, ``"llvm -mcpu=..."``.
    """
    lib = _load_c_lib()
    out_handle = c_void_p()
    encoded = text.encode("utf-8")
    rc = lib.nautilus_tir_parse(
        encoded, c_size_t(len(encoded)),
        target.encode("utf-8"),
        ctypes.byref(out_handle),
    )
    _check_rc(rc, "nautilus_tir_parse", "nautilus_tvm_last_error_message")
    return TIRModuleHandle(out_handle.value or 0)


def tune(
    module: TIRModuleHandle,
    max_trials: int,
    num_trials_per_iter: int = 64,
    strategy: int = TUNING_AUTO,
    timeout_seconds: int = 300,
) -> TuningRecordHandle:
    """Run MetaSchedule autotuning on a TIR module.

    Returns a tuning record whose integer parameters can be read with
    ``record.get_int(name)``.
    """
    lib = _load_c_lib()
    out_handle = c_void_p()
    rc = lib.nautilus_tune(
        c_void_p(module.c_handle),
        c_int(max_trials),
        c_int(num_trials_per_iter),
        c_int(strategy),
        c_int(timeout_seconds),
        ctypes.byref(out_handle),
    )
    _check_rc(rc, "nautilus_tune", "nautilus_tvm_last_error_message")
    return TuningRecordHandle(out_handle.value or 0)


def tvm_version() -> str:
    """Return the version string of the underlying TVM library."""
    lib = _load_c_lib()
    return lib.nautilus_tvm_version().decode("utf-8", errors="replace")


# =====================================================================
# XLA C-API (StableHLO + GSPMD auto-sharding)
# =====================================================================


SHARD_AUTO = 0
SHARD_REPLICATED = 1
SHARD_DATA_PARALLEL = 2
SHARD_MODEL_PARALLEL = 3
SHARD_TENSOR_PARALLEL = 4


class StableHLOHandle:
    """Python-side handle for a StableHLO module."""
    def __init__(self, c_handle: int) -> None:
        self._c_handle = c_handle
        self._released = False

    @property
    def c_handle(self) -> int:
        if self._released:
            raise RuntimeError("StableHLO handle already released")
        return self._c_handle

    def get_mlir_text(self) -> tuple[str, int]:
        lib = _load_c_lib()
        out_text = c_char_p()
        out_len = c_size_t()
        rc = lib.nautilus_stablehlo_get_mlir_text(
            c_void_p(self._c_handle),
            ctypes.byref(out_text),
            ctypes.byref(out_len),
        )
        _check_rc(rc, "nautilus_stablehlo_get_mlir_text", "nautilus_xla_last_error_message")
        # The returned pointer is borrowed from the C side and remains
        # valid for the lifetime of the handle. Make a copy so the
        # Python string is independent of the underlying allocation.
        text = out_text.value.decode("utf-8") if out_text.value else ""
        return text, out_len.value

    def release(self) -> None:
        if not self._released:
            lib = _load_c_lib()
            lib.nautilus_stablehlo_release(c_void_p(self._c_handle))
            self._released = True

    def __enter__(self) -> StableHLOHandle:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass


class MeshHandle:
    """Python-side handle for a device mesh."""
    def __init__(self, c_handle: int) -> None:
        self._c_handle = c_handle
        self._released = False

    @property
    def c_handle(self) -> int:
        if self._released:
            raise RuntimeError("Mesh handle already released")
        return self._c_handle

    def release(self) -> None:
        if not self._released:
            lib = _load_c_lib()
            lib.nautilus_mesh_release(c_void_p(self._c_handle))
            self._released = True

    def __enter__(self) -> MeshHandle:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass


class ShardingSpecHandle:
    """Python-side handle for a GSPMD sharding spec."""
    def __init__(self, c_handle: int) -> None:
        self._c_handle = c_handle
        self._released = False

    @property
    def c_handle(self) -> int:
        if self._released:
            raise RuntimeError("Sharding spec handle already released")
        return self._c_handle

    def release(self) -> None:
        if not self._released:
            lib = _load_c_lib()
            lib.nautilus_sharding_spec_release(c_void_p(self._c_handle))
            self._released = True

    def __enter__(self) -> ShardingSpecHandle:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass


def stablehlo_from_fx(
    fx_graph_json: str, function_name: str
) -> StableHLOHandle:
    """Build a StableHLO module from a serialized TorchFX graph.

    Args:
        fx_graph_json: JSON-serialized ``torch.fx.Graph`` (or
            ``torch.export.ExportedProgram`` text).
        function_name: The entry-point function name in the FX graph.
    """
    lib = _load_c_lib()
    out_handle = c_void_p()
    encoded = fx_graph_json.encode("utf-8")
    rc = lib.nautilus_stablehlo_from_fx(
        encoded, c_size_t(len(encoded)),
        function_name.encode("utf-8"),
        ctypes.byref(out_handle),
    )
    _check_rc(rc, "nautilus_stablehlo_from_fx", "nautilus_xla_last_error_message")
    return StableHLOHandle(out_handle.value or 0)


def mesh_create(axes: list[int]) -> MeshHandle:
    """Create a device mesh with the given axis sizes.

    Example: ``mesh_create([2, 4])`` defines a 2x4 mesh.
    """
    lib = _load_c_lib()
    out_handle = c_void_p()
    arr = (c_int64 * len(axes))(*axes)
    rc = lib.nautilus_mesh_create(
        arr, c_size_t(len(axes)),
        ctypes.byref(out_handle),
    )
    _check_rc(rc, "nautilus_mesh_create", "nautilus_xla_last_error_message")
    return MeshHandle(out_handle.value or 0)


def gspmd_shard(
    module: StableHLOHandle,
    mesh: MeshHandle,
    strategy: int = SHARD_AUTO,
    timeout_seconds: int = 300,
) -> ShardingSpecHandle:
    """Run GSPMD auto-sharding on a StableHLO module."""
    lib = _load_c_lib()
    out_handle = c_void_p()
    rc = lib.nautilus_gspmd_shard(
        c_void_p(module.c_handle),
        c_void_p(mesh.c_handle),
        c_int(strategy),
        c_int(timeout_seconds),
        ctypes.byref(out_handle),
    )
    _check_rc(rc, "nautilus_gspmd_shard", "nautilus_xla_last_error_message")
    return ShardingSpecHandle(out_handle.value or 0)


def xla_version() -> str:
    """Return the version string of the underlying OpenXLA library."""
    lib = _load_c_lib()
    return lib.nautilus_xla_version().decode("utf-8", errors="replace")


# --- Lazy access pattern ---


def is_available() -> bool:
    """Return True if the C library can be loaded."""
    try:
        _load_c_lib()
        return True
    except DependencyMissingError:
        return False


# --- Sub-API namespaces (re-export the per-domain functions as
#     attributes so callers can write
#     ``tvm_c_api.tir_parse(...)`` / ``xla_c_api.mesh_create(...)``).


class _TritonNamespace:
    """Re-exports Triton C-API functions as attributes."""
    compile = staticmethod(compile)
    triton_version = staticmethod(triton_version)
    TritonKernelHandle = TritonKernelHandle

    VENDOR_NVIDIA = VENDOR_NVIDIA
    VENDOR_AMD = VENDOR_AMD
    VENDOR_INTEL = VENDOR_INTEL
    VENDOR_APPLE = VENDOR_APPLE
    VENDOR_HOST = VENDOR_HOST
    ARCH_SM_70 = ARCH_SM_70
    ARCH_SM_75 = ARCH_SM_75
    ARCH_SM_80 = ARCH_SM_80
    ARCH_SM_86 = ARCH_SM_86
    ARCH_SM_89 = ARCH_SM_89
    ARCH_SM_90 = ARCH_SM_90
    ARCH_SM_100 = ARCH_SM_100
    ARCH_SM_120 = ARCH_SM_120
    ARCH_GFX900 = ARCH_GFX900
    ARCH_GFX906 = ARCH_GFX906
    ARCH_GFX908 = ARCH_GFX908
    ARCH_GFX90A = ARCH_GFX90A
    ARCH_GFX942 = ARCH_GFX942
    ARCH_GFX950 = ARCH_GFX950
    ARCH_XE_LP = ARCH_XE_LP
    ARCH_XE_HPG = ARCH_XE_HPG
    ARCH_XE_HPC = ARCH_XE_HPC
    ARCH_XE2 = ARCH_XE2
    ARCH_GAUDI2 = ARCH_GAUDI2
    ARCH_GAUDI3 = ARCH_GAUDI3


class _TvmNamespace:
    """Re-exports TVM MetaSchedule C-API functions as attributes."""
    tir_parse = staticmethod(tir_parse)
    tune = staticmethod(tune)
    tvm_version = staticmethod(tvm_version)
    TIRModuleHandle = TIRModuleHandle
    TuningRecordHandle = TuningRecordHandle

    TUNING_AUTO = TUNING_AUTO
    TUNING_RL_EVOLUTIONARY = TUNING_RL_EVOLUTIONARY
    TUNING_RL_GRADIENT = TUNING_RL_GRADIENT
    TUNING_XGBOOST_COST_MODEL = TUNING_XGBOOST_COST_MODEL


class _XlaNamespace:
    """Re-exports XLA C-API functions as attributes."""
    stablehlo_from_fx = staticmethod(stablehlo_from_fx)
    mesh_create = staticmethod(mesh_create)
    gspmd_shard = staticmethod(gspmd_shard)
    xla_version = staticmethod(xla_version)
    StableHLOHandle = StableHLOHandle
    MeshHandle = MeshHandle
    ShardingSpecHandle = ShardingSpecHandle

    SHARD_AUTO = SHARD_AUTO
    SHARD_REPLICATED = SHARD_REPLICATED
    SHARD_DATA_PARALLEL = SHARD_DATA_PARALLEL
    SHARD_MODEL_PARALLEL = SHARD_MODEL_PARALLEL
    SHARD_TENSOR_PARALLEL = SHARD_TENSOR_PARALLEL


triton_c_api = _TritonNamespace()
tvm_c_api = _TvmNamespace()
xla_c_api = _XlaNamespace()


__all__ = [
    "ARCH_GAUDI2",
    "ARCH_GAUDI3",
    "ARCH_GFX90A",
    "ARCH_GFX900",
    "ARCH_GFX906",
    "ARCH_GFX908",
    "ARCH_GFX942",
    "ARCH_GFX950",
    "ARCH_SM_70",
    "ARCH_SM_75",
    "ARCH_SM_80",
    "ARCH_SM_86",
    "ARCH_SM_89",
    "ARCH_SM_90",
    "ARCH_SM_100",
    "ARCH_SM_120",
    "ARCH_XE2",
    "ARCH_XE_HPC",
    "ARCH_XE_HPG",
    "ARCH_XE_LP",
    # XLA sharding strategy constants
    "SHARD_AUTO",
    "SHARD_DATA_PARALLEL",
    "SHARD_MODEL_PARALLEL",
    "SHARD_REPLICATED",
    "SHARD_TENSOR_PARALLEL",
    # TVM strategy constants
    "TUNING_AUTO",
    "TUNING_RL_EVOLUTIONARY",
    "TUNING_RL_GRADIENT",
    "TUNING_XGBOOST_COST_MODEL",
    "VENDOR_AMD",
    "VENDOR_APPLE",
    "VENDOR_HOST",
    "VENDOR_INTEL",
    # Triton vendor / arch constants
    "VENDOR_NVIDIA",
    # Errors
    "CApiUnavailable",
    "MeshHandle",
    "ShardingSpecHandle",
    # XLA C-API
    "StableHLOHandle",
    # TVM C-API
    "TIRModuleHandle",
    # Triton C-API
    "TritonKernelHandle",
    "TuningRecordHandle",
    "compile",
    "gspmd_shard",
    # Availability probe
    "is_available",
    "mesh_create",
    "stablehlo_from_fx",
    "tir_parse",
    "triton_c_api",
    "triton_version",
    "tune",
    "tvm_c_api",
    "tvm_version",
    "xla_c_api",
    "xla_version",
]
