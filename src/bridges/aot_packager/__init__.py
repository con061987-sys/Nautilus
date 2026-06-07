"""AOT Fat Binary Packager for the Nautilus project.

This package implements the Phase 2 deliverable: producing a single
"fat binary" file that contains compiled kernels for AMD, Intel, and
Nvidia hardware, with a runtime C stub that detects the vendor at
startup and dispatches to the correct binary block.

Architecture:
    [Compiled Triton kernel] (PTX/LLIR/TTGIR)
        │
        ├──► [AMD: AOTriton]      ──► kernel.hsaco
        ├──► [Intel: oneAPI]       ──► kernel.spv
        ├──► [Nvidia: Triton JIT]  ──► kernel.ptx
        │
        └──► [LLVM lld]           ──► fat_binary.o
                │
                ▼
            [C Runtime Stub]
            - Detect CPU vendor (CPUID)
            - Detect GPU via /dev/kfd, /dev/dri
            - Jump to matching backend

Production features:
    - Per-backend circuit breakers (one AOT failure doesn't block others)
    - Per-backend timeouts (AOT compilation can be slow)
    - Stage-level structured logging
    - Persistent binary cache (skip recompile when source unchanged)
    - Hardware validation (verify the binary actually runs on target HW)

Modules:
    amd_backend.py       - AOTriton wrapper for AMD AOT compilation
    intel_backend.py     - oneAPI/SYCL wrapper for Intel AOT compilation
    nvidia_backend.py    - Triton JIT → PTX capture for Nvidia
    linker.py            - LLVM lld wrapper for fat binary linking
    runtime_stub.c       - C runtime for vendor detection
    fat_binary.py        - ELF section packing and metadata
    hardware_validator.py - Real hardware validation
    builder.py           - Main orchestrator (FatBinaryBuilder)
"""

from .amd_backend import AMDBackend, AMDCompilationResult
from .builder import FatBinaryBuilder, FatBinaryConfig, FatBinaryResult
from .fat_binary import FatBinary, KernelSection
from .hardware_validator import HardwareValidator, ValidationResult
from .intel_backend import IntelBackend, IntelCompilationResult
from .linker import FatBinaryLinker, LinkingResult
from .nvidia_backend import NvidiaBackend, NvidiaCompilationResult

__all__ = [
    # Per-backend
    "AMDBackend",
    "AMDCompilationResult",
    # Fat binary format
    "FatBinary",
    # Main builder
    "FatBinaryBuilder",
    "FatBinaryConfig",
    # Linking
    "FatBinaryLinker",
    "FatBinaryResult",
    # Hardware validation
    "HardwareValidator",
    "IntelBackend",
    "IntelCompilationResult",
    "KernelSection",
    "LinkingResult",
    "NvidiaBackend",
    "NvidiaCompilationResult",
    "ValidationResult",
]
