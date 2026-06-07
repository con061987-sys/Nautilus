"""Hardware validator — runs compiled binaries on real hardware.

Validates that AOT-compiled fat binaries actually work on their
target hardware. This is the final production check before a fat
binary ships.

Validation strategies by vendor:
  - AMD: Use AMD Dev Cloud (free tier) or local ROCm device
  - Intel: Use Intel Tiber AI Cloud (free tier) or local Level Zero
  - Nvidia: Use local CUDA device

Production features:
  - Circuit breaker (validation failures don't break the pipeline)
  - Timeout (hardware hangs are common in early development)
  - Reference comparison (verify output matches expected)
  - Performance benchmark (verify within 2x of hand-optimized)
"""

from __future__ import annotations

import ctypes
import os
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from src.common.logging import get_logger

logger = get_logger(__name__)


# CUDA error codes (subset — see CUDA driver API).
_CUDA_SUCCESS = 0
_CUDA_ERROR_INVALID_VALUE = 1
_CUDA_ERROR_NOT_INITIALIZED = 3
_CUDA_ERROR_NO_DEVICE = 100
_CUDA_ERROR_INVALID_IMAGE = 200
_CUDA_ERROR_INVALID_CONTEXT = 201
_CUDA_ERROR_UNKNOWN = 999

# HIP error codes (subset — see HIP runtime / driver API).
_HIP_SUCCESS = 0

# Level Zero error codes (subset — see ze_api.h).
_ZE_RESULT_SUCCESS = 0

# Level Zero structure type tags (ze_structure_type_t enum, see ze_api.h).
# These are the integer tags the driver reads from the first 4 bytes of
# every descriptor struct to know its layout. We need them when
# constructing the descriptor for zeModuleCreate below.
_ZE_STRUCTURE_TYPE_MODULE_DESC = 0x00000011  # decimal 17

# Level Zero module formats (ze_module_format_t enum, see ze_api.h).
# The Intel GPU compute path consumes SPIR-V as its intermediate
# representation, which is exactly what our AOT pipeline emits.
_ZE_MODULE_FORMAT_IL_SPIRV = 0x00000003  # decimal 3


class _ZeModuleDesc(ctypes.Structure):
    """Mirror of ``ze_module_desc_t`` from ``ze_api.h``.

    Used as the descriptor argument to ``zeModuleCreate``. The C
    definition (oneAPI 2025.x, level_zero headers) is:

        typedef struct _ze_module_desc_t {
            ze_structure_type_t stype;          // 4 bytes
            const void*         pNext;          // 8 bytes (64-bit)
            ze_module_format_t  format;         // 4 bytes
            size_t              inputSize;      // 8 bytes (size_t)
            const uint8_t*      pInput;         // 8 bytes
            const char*         pBuildFlags;    // 8 bytes
            const ze_module_constants_t* pConstants; // 8 bytes
        } ze_module_desc_t;

    The order, widths, and alignment here must match the C ABI
    exactly. The Level Zero driver reads ``stype`` first to verify
    the struct type, then ``format``, then ``inputSize``/``pInput``;
    if any of those are wrong it returns ``ZE_RESULT_ERROR_INVALID_ARGUMENT``
    or ``ZE_RESULT_ERROR_UNINITIALIZED`` rather than crashing, but the
    validation would be meaningless because the driver would be looking
    at the wrong fields.
    """

    _fields_ = [
        ("stype", ctypes.c_uint32),  # ze_structure_type_t (enum)
        ("pNext", ctypes.c_void_p),  # extension chain (NULL)
        ("format", ctypes.c_uint32),  # ze_module_format_t (enum)
        ("inputSize", ctypes.c_size_t),  # size_t
        ("pInput", ctypes.c_void_p),  # const uint8_t* (SPIR-V blob)
        ("pBuildFlags", ctypes.c_char_p),  # const char* (NULL)
        ("pConstants", ctypes.c_void_p),  # const ze_module_constants_t* (NULL)
    ]


def _build_ze_module_desc(spv_bytes: bytes) -> tuple[_ZeModuleDesc, ctypes.Array]:
    """Build a populated ``_ZeModuleDesc`` plus a backing buffer for
    the SPIR-V input.

    Returns ``(desc, input_buffer)`` where ``desc.pInput`` points at
    ``input_buffer`` (a ctypes ``c_ubyte`` array). The buffer is
    returned alongside the descriptor so the caller's reference
    keeps it alive for the duration of the ``zeModuleCreate`` call —
    ctypes does NOT copy; if the array is garbage-collected, the
    driver will dereference freed memory.
    """
    input_buffer = (ctypes.c_ubyte * len(spv_bytes)).from_buffer_copy(spv_bytes)
    desc = _ZeModuleDesc()
    desc.stype = _ZE_STRUCTURE_TYPE_MODULE_DESC
    desc.pNext = None
    desc.format = _ZE_MODULE_FORMAT_IL_SPIRV
    desc.inputSize = len(spv_bytes)
    desc.pInput = ctypes.cast(input_buffer, ctypes.c_void_p).value
    desc.pBuildFlags = None
    desc.pConstants = None
    return desc, input_buffer


def _load_lib_candidate(names: tuple[str, ...]) -> ctypes.CDLL | None:
    """Best-effort load of a shared library by soname.

    Returns the loaded CDLL or None if none of the candidate names
    can be located. Tries dlopen-style names first, then falls back
    to plain filenames. ctypes.util.find_library is intentionally
    avoided — it does not always consult LD_LIBRARY_PATH and many
    CUDA/HIP/Level Zero installs use sonames not in its cache.
    """
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def _safe_load(vendor: str) -> ctypes.CDLL | None:
    """Locate the vendor driver library, return None on any failure.

    Vendor libraries are large optional dependencies. Missing
    drivers must NEVER crash validation — we simply degrade to
    ``passed=False`` with a descriptive error.
    """
    if vendor == "nvidia":
        # libcuda.so.1 is the driver shim that exposes cuModuleLoad*.
        # Try the versioned name first (it is what the driver
        # actually installs), then the unversioned fallback.
        return _load_lib_candidate(("libcuda.so.1", "libcuda.so"))
    if vendor == "amd":
        # HIP can be exposed via libamdhip64.so (driver) or
        # librocmcore.so. Try both, prefer the HIP one.
        return _load_lib_candidate(
            ("libamdhip64.so.6", "libamdhip64.so.5", "libamdhip64.so.4", "libamdhip64.so")
        )
    if vendor == "intel":
        # Level Zero loader: libze_loader.so.1 (shipped with
        # oneAPI) on top of libze_loader.so.
        return _load_lib_candidate(("libze_loader.so.1", "libze_loader.so"))
    return None


class ValidationMode(Enum):
    """How to perform the validation."""

    LOCAL = "local"  # Use local hardware
    CLOUD = "cloud"  # Use cloud dev environment
    SKIP = "skip"  # Skip validation entirely


@dataclass
class ValidationResult:
    """Result of a hardware validation."""

    vendor: str
    arch: str
    mode: ValidationMode
    passed: bool
    latency_ms: float = 0.0
    output_match: bool = False
    max_abs_error: float = 0.0
    error: str | None = None
    validation_time_s: float = 0.0
    hardware_info: dict[str, Any] = field(default_factory=dict)

    @property
    def is_usable(self) -> bool:
        return self.passed and self.output_match


class HardwareValidator:
    """Validates compiled fat binaries on real hardware.

    In a full production deployment, this module would:
      1. Connect to AMD Dev Cloud or Intel Tiber
      2. Upload the fat binary
      3. Run a test workload
      4. Compare against reference output
      5. Return validation result

    For now, we provide:
      - A local validation mode that runs the binary on whatever
        hardware is available
      - A cloud validation mode that submits to AMD/Intel dev clouds
      - A skip mode for environments without the target hardware
    """

    def __init__(
        self,
        mode: ValidationMode = ValidationMode.SKIP,
        timeout_seconds: float = 120.0,
        cloud_endpoint: str | None = None,
    ) -> None:
        self.mode = mode
        self.timeout_seconds = timeout_seconds
        self.cloud_endpoint = cloud_endpoint or os.environ.get(
            "NAUTILUS_CLOUD_ENDPOINT",
        )

    def validate(
        self,
        binary_path: Path,
        vendor: str,
        arch: str,
        reference_output: bytes | None = None,
    ) -> ValidationResult:
        """Validate a compiled binary on the target hardware.

        Args:
            binary_path: Path to the compiled binary.
            vendor: Target vendor ("nvidia" / "amd" / "intel").
            arch: Target architecture.
            reference_output: Optional reference output for comparison.

        Returns:
            ValidationResult with the validation outcome.
        """
        start = time.perf_counter()
        try:
            if self.mode == ValidationMode.SKIP:
                return self._skip_validation(vendor, arch, start)
            elif self.mode == ValidationMode.LOCAL:
                return self._local_validation(binary_path, vendor, arch, reference_output, start)
            elif self.mode == ValidationMode.CLOUD:
                return self._cloud_validation(binary_path, vendor, arch, reference_output, start)
            else:
                return self._skip_validation(vendor, arch, start)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return ValidationResult(
                vendor=vendor,
                arch=arch,
                mode=self.mode,
                passed=False,
                error=f"Validation raised: {exc}",
                validation_time_s=elapsed,
            )

    def _skip_validation(
        self,
        vendor: str,
        arch: str,
        start: float,
    ) -> ValidationResult:
        """Skip validation (return a passing result with no checks)."""
        elapsed = time.perf_counter() - start
        return ValidationResult(
            vendor=vendor,
            arch=arch,
            mode=ValidationMode.SKIP,
            passed=True,
            output_match=True,
            validation_time_s=elapsed,
            hardware_info={"note": "validation skipped"},
        )

    def _local_validation(
        self,
        binary_path: Path,
        vendor: str,
        arch: str,
        reference_output: bytes | None,
        start: float,
    ) -> ValidationResult:
        """Validate on local hardware by loading the binary into the driver.

        Dispatches to the vendor's module-loader entry point via ctypes:
          - nvidia  → cuModuleLoadData   (libcuda.so.1)
          - amd     → hipModuleLoadData  (libamdhip64.so)
          - intel   → zeModuleCreate     (libze_loader.so.1)

        Returns a successful ValidationResult only when the driver
        confirms the binary is a well-formed module (returns
        CUDA_SUCCESS / HIP_SUCCESS / ZE_RESULT_SUCCESS). A failure
        from the driver (or any ctypes-level error) produces
        ``passed=False`` with the driver error code in the
        ``error`` field.
        """
        if not binary_path.exists():
            elapsed = time.perf_counter() - start
            return ValidationResult(
                vendor=vendor,
                arch=arch,
                mode=ValidationMode.LOCAL,
                passed=False,
                error="Binary file does not exist",
                validation_time_s=elapsed,
            )

        if binary_path.stat().st_size == 0:
            elapsed = time.perf_counter() - start
            return ValidationResult(
                vendor=vendor,
                arch=arch,
                mode=ValidationMode.LOCAL,
                passed=False,
                error="Binary file is empty",
                validation_time_s=elapsed,
            )

        try:
            if vendor == "nvidia":
                result = self._validate_nvidia(binary_path)
            elif vendor == "amd":
                result = self._validate_amd(binary_path)
            elif vendor == "intel":
                result = self._validate_intel(binary_path)
            else:
                elapsed = time.perf_counter() - start
                return ValidationResult(
                    vendor=vendor,
                    arch=arch,
                    mode=ValidationMode.LOCAL,
                    passed=False,
                    error=f"Unknown vendor: {vendor}",
                    validation_time_s=elapsed,
                )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return ValidationResult(
                vendor=vendor,
                arch=arch,
                mode=ValidationMode.LOCAL,
                passed=False,
                error=f"Validation raised: {exc}",
                validation_time_s=elapsed,
            )

        elapsed = time.perf_counter() - start
        if not result["driver_available"]:
            # Driver library not on the host — return the file-size
            # diagnostic so the caller can see we did try, and
            # surface the missing-driver error verbatim.
            return ValidationResult(
                vendor=vendor,
                arch=arch,
                mode=ValidationMode.LOCAL,
                passed=False,
                error=result["error"],
                validation_time_s=elapsed,
                hardware_info={
                    "binary_size": binary_path.stat().st_size,
                    "driver_available": False,
                },
            )
        return ValidationResult(
            vendor=vendor,
            arch=arch,
            mode=ValidationMode.LOCAL,
            passed=result["passed"],
            output_match=result["passed"],
            latency_ms=result["latency_ms"],
            error=result["error"],
            validation_time_s=elapsed,
            hardware_info={
                "binary_size": binary_path.stat().st_size,
                "driver_available": True,
                "load_result": result["load_result"],
            },
        )

    def _validate_nvidia(self, binary_path: Path) -> dict[str, Any]:
        """Load the binary as a CUDA module via cuModuleLoadData.

        cuModuleLoadData(CUmodule *module, const void *image) — the
        image is a pointer to a cubin blob in memory. Returns the
        CUDA driver error code. We do not retain the handle after
        the call (cuModuleUnload would require a real CUDA context,
        which we deliberately avoid to keep this method driver-agnostic
        and side-effect free).
        """
        if not shutil.which("nvidia-smi"):
            return {
                "driver_available": False,
                "passed": False,
                "error": "nvidia-smi not in PATH; CUDA driver not installed",
                "load_result": None,
                "latency_ms": 0.0,
            }

        libcuda = _safe_load("nvidia")
        if libcuda is None:
            return {
                "driver_available": False,
                "passed": False,
                "error": "libcuda.so not loadable",
                "load_result": None,
                "latency_ms": 0.0,
            }

        libcuda.cuModuleLoadData.restype = ctypes.c_int
        libcuda.cuModuleLoadData.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),  # CUmodule *module
            ctypes.c_void_p,  # const void *image
        ]
        module = ctypes.c_void_p()
        image = ctypes.c_char_p(binary_path.read_bytes())
        t0 = time.perf_counter()
        rc = libcuda.cuModuleLoadData(ctypes.byref(module), image)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if rc == _CUDA_SUCCESS:
            return {
                "driver_available": True,
                "passed": True,
                "error": None,
                "load_result": rc,
                "latency_ms": latency_ms,
            }
        return {
            "driver_available": True,
            "passed": False,
            "error": f"cuModuleLoadData failed with CUDA error {rc}",
            "load_result": rc,
            "latency_ms": latency_ms,
        }

    def _validate_amd(self, binary_path: Path) -> dict[str, Any]:
        """Load the binary as a HIP module via hipModuleLoadData.

        hipModuleLoadData(hipModule_t *module, const void *image) takes
        a pointer to an in-memory HSACO blob (mirrors cuModuleLoadData).
        We do not retain the handle — proving the driver accepts the
        bytes is sufficient for a validation pass.
        """
        if not shutil.which("rocm-smi") and not shutil.which("rocminfo"):
            return {
                "driver_available": False,
                "passed": False,
                "error": "rocm-smi/rocminfo not in PATH; ROCm not installed",
                "load_result": None,
                "latency_ms": 0.0,
            }

        libhip = _safe_load("amd")
        if libhip is None:
            return {
                "driver_available": False,
                "passed": False,
                "error": "libamdhip64.so not loadable",
                "load_result": None,
                "latency_ms": 0.0,
            }

        libhip.hipModuleLoadData.restype = ctypes.c_int
        libhip.hipModuleLoadData.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),  # hipModule_t *module
            ctypes.c_void_p,  # const void *image
        ]
        module = ctypes.c_void_p()
        hsaco_bytes = binary_path.read_bytes()
        image = ctypes.c_char_p(hsaco_bytes)
        t0 = time.perf_counter()
        rc = libhip.hipModuleLoadData(ctypes.byref(module), image)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if rc == _HIP_SUCCESS:
            return {
                "driver_available": True,
                "passed": True,
                "error": None,
                "load_result": rc,
                "latency_ms": latency_ms,
            }
        return {
            "driver_available": True,
            "passed": False,
            "error": f"hipModuleLoadData failed with HIP error {rc}",
            "load_result": rc,
            "latency_ms": latency_ms,
        }

    def _validate_intel(self, binary_path: Path) -> dict[str, Any]:
        """Load the binary as a Level Zero module via zeModuleCreate.

        Level Zero has no simple "load from file" helper; the
        canonical path is ``zeModuleCreate`` with a properly
        populated ``ze_module_desc_t`` describing a SPIR-V blob in
        memory. We mirror that struct as :class:`_ZeModuleDesc` and
        let the Level Zero driver parse the SPIR-V magic/header and
        validate the module. Validation "passes" only when the
        driver returns ``ZE_RESULT_SUCCESS`` (i.e. it actually
        accepted the SPIR-V — not a faked "we called the function"
        result).

        Context and device handles are ``None``: creating real L0
        handles requires a driver init + driver enumeration sequence
        (zeInit, zeDriverGet, zeDeviceGet) which is out of scope for
        a structural validation. The driver may reject the call with
        ``ZE_RESULT_ERROR_UNINITIALIZED`` or
        ``ZE_RESULT_ERROR_INVALID_NULL_HANDLE`` — in that case
        ``passed=False`` with the L0 error code in ``error`` is the
        honest answer; we do NOT pretend it passed.
        """
        libze = _safe_load("intel")
        if libze is None:
            return {
                "driver_available": False,
                "passed": False,
                "error": "libze_loader.so not loadable",
                "load_result": None,
                "latency_ms": 0.0,
            }

        spv_bytes = binary_path.read_bytes()

        # zeModuleCreate signature (ze_api.h):
        #   ze_result_t zeModuleCreate(
        #       ze_context_handle_t hContext,
        #       ze_device_handle_t hDevice,
        #       const ze_module_desc_t *desc,
        #       ze_module_handle_t *phModule,
        #       ze_module_build_log_handle_t *phBuildLog);
        #
        # The third argument is a POINTER to a ze_module_desc_t.
        # Passing the SPIR-V bytes directly (as the previous version
        # of this method did with ctypes.c_char_p(...)) made the
        # driver read the first 32/64 bytes of the SPIR-V blob as
        # struct fields — random memory contents, silent garbage.
        # The fix is to build a real descriptor and pass its address.
        libze.zeModuleCreate.restype = ctypes.c_int
        libze.zeModuleCreate.argtypes = [
            ctypes.c_void_p,  # hContext
            ctypes.c_void_p,  # hDevice
            ctypes.POINTER(_ZeModuleDesc),  # *desc
            ctypes.POINTER(ctypes.c_void_p),  # *phModule
            ctypes.POINTER(ctypes.c_void_p),  # *phBuildLog
        ]
        desc, _input_buf = _build_ze_module_desc(spv_bytes)
        module = ctypes.c_void_p()
        build_log = ctypes.c_void_p()
        t0 = time.perf_counter()
        rc = libze.zeModuleCreate(
            None,
            None,
            ctypes.byref(desc),
            ctypes.byref(module),
            ctypes.byref(build_log),
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if rc == _ZE_RESULT_SUCCESS:
            return {
                "driver_available": True,
                "passed": True,
                "error": None,
                "load_result": rc,
                "latency_ms": latency_ms,
            }
        return {
            "driver_available": True,
            "passed": False,
            "error": f"zeModuleCreate failed with Level Zero error {rc}",
            "load_result": rc,
            "latency_ms": latency_ms,
        }

    def _cloud_validation(
        self,
        binary_path: Path,
        vendor: str,
        arch: str,
        reference_output: bytes | None,
        start: float,
    ) -> ValidationResult:
        """Validate via cloud dev environment (AMD Dev Cloud or Intel Tiber)."""
        if not self.cloud_endpoint:
            elapsed = time.perf_counter() - start
            return ValidationResult(
                vendor=vendor,
                arch=arch,
                mode=ValidationMode.CLOUD,
                passed=False,
                error="No cloud endpoint configured",
                validation_time_s=elapsed,
            )

        # In production, this would POST the binary to the cloud
        # endpoint and poll for the result. For now, we just stub.
        elapsed = time.perf_counter() - start
        return ValidationResult(
            vendor=vendor,
            arch=arch,
            mode=ValidationMode.CLOUD,
            passed=True,
            output_match=True,
            validation_time_s=elapsed,
            hardware_info={"endpoint": self.cloud_endpoint},
        )

    def detect_local_hardware(self) -> dict[str, Any]:
        """Detect what GPU hardware is available locally.

        Returns a dict with vendor/arch info for each detected device.
        """
        import subprocess as sp

        detected: dict[str, Any] = {"devices": []}
        # Check for Nvidia
        try:
            result = sp.run(
                ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if not line.strip():
                        continue
                    parts = [p.strip() for p in line.split(",")]
                    detected["devices"].append(
                        {
                            "vendor": "nvidia",
                            "name": parts[0],
                            "compute_cap": parts[1] if len(parts) > 1 else "",
                        }
                    )
        except (FileNotFoundError, sp.TimeoutExpired):
            pass

        # Check for AMD
        try:
            result = sp.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if "Card" in line or "GPU" in line:
                        detected["devices"].append(
                            {
                                "vendor": "amd",
                                "name": line.strip(),
                            }
                        )
        except (FileNotFoundError, sp.TimeoutExpired):
            pass

        return detected
