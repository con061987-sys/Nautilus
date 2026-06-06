"""
src.c_api — Python ctypes bindings for the C-API headers.

Loads the compiled shared library (libnautilus_c_api.so / .dylib / .dll)
and exposes Pythonic access to the C functions. The C library is built
separately via CMake (see src/c_api/CMakeLists.txt).

The C headers are the SOURCE OF TRUTH for the ABI. This Python module
mirrors them; if you change a header, you must update this file too.

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
without the [cpp] extra), calls raise CApiUnavailable with a clear
message about how to build it.
"""

from __future__ import annotations

import ctypes
import os
import platform
import sys
from ctypes import (
    CDLL,
    POINTER,
    Structure,
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
    NautilusError,
    ErrorCode,
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
    garbage on 64-bit systems.
    """
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


# --- Pythonic wrappers that raise on error ---


class CApiUnavailable(DependencyMissingError):
    """Raised when the C-API shared library is not built."""
    code = ErrorCode.DEPENDENCY_MISSING


def _check_rc(rc: int, op: str) -> None:
    """Raise an exception if a C function returned non-zero."""
    if rc == 0:
        return
    try:
        lib = _load_c_lib()
        msg = lib.nautilus_last_error_message().decode("utf-8", errors="replace")
    except Exception:
        msg = f"unknown error (rc={rc})"
    raise NautilusError(
        f"C-API call {op!r} failed: {msg}",
        context={"operation": op, "return_code": rc},
    )


# --- Triton C-API ---


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

    def get_binary(self) -> tuple[bytes, str]:
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
        _check_rc(rc, "nautilus_get_binary")
        data = bytes(
            ctypes.cast(out_data, POINTER(c_uint8 * out_size.value))[:]
        )
        fmt = out_format.value.decode("utf-8") if out_format.value else ""
        return data, fmt

    def set_tuning_param(self, name: str, value: int) -> None:
        lib = _load_c_lib()
        rc = lib.nautilus_set_tuning_param(
            c_void_p(self._c_handle),
            name.encode("utf-8"),
            int(value),
        )
        _check_rc(rc, "nautilus_set_tuning_param")

    def release(self) -> None:
        if not self._released:
            lib = _load_c_lib()
            lib.nautilus_release(c_void_p(self._c_handle))
            self._released = True

    def __enter__(self) -> "TritonKernelHandle":
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
    _check_rc(rc, "nautilus_compile")
    return TritonKernelHandle(out_handle.value or 0)


def triton_version() -> str:
    """Return the version string of the underlying Triton library."""
    lib = _load_c_lib()
    return lib.nautilus_triton_version().decode("utf-8", errors="replace")


# --- Lazy access pattern ---


def is_available() -> bool:
    """Return True if the C library can be loaded."""
    try:
        _load_c_lib()
        return True
    except DependencyMissingError:
        return False


__all__ = [
    "CApiUnavailable",
    "TritonKernelHandle",
    "compile",
    "triton_version",
    "is_available",
    "VENDOR_NVIDIA", "VENDOR_AMD", "VENDOR_INTEL", "VENDOR_APPLE", "VENDOR_HOST",
    "ARCH_SM_70", "ARCH_SM_75", "ARCH_SM_80", "ARCH_SM_86", "ARCH_SM_89",
    "ARCH_SM_90", "ARCH_SM_100", "ARCH_SM_120",
    "ARCH_GFX900", "ARCH_GFX906", "ARCH_GFX908", "ARCH_GFX90A", "ARCH_GFX942", "ARCH_GFX950",
    "ARCH_XE_LP", "ARCH_XE_HPG", "ARCH_XE_HPC", "ARCH_XE2", "ARCH_GAUDI2", "ARCH_GAUDI3",
]
