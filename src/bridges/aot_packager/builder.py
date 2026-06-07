"""FatBinaryBuilder — main orchestrator for the AOT fat binary pipeline.

Coordinates:
  1. AMD AOT compilation (AOTriton → .hsaco)
  2. Intel AOT compilation (oneAPI → .spv)
  3. Nvidia AOT compilation (Triton JIT → .ptx)
  4. Apple AOT compilation (Triton metal / xcrun → .metallib)
  5. C runtime stub compilation
  6. LLVM lld linking
  7. Hardware validation

Production features:
  - Per-vendor circuit breakers (one vendor's failure doesn't block others)
  - Per-stage timeouts
  - Persistent binary cache across all stages
  - Structured logging
  - Configurable target architectures
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.common.errors import (
    CompilationError,
    DependencyMissingError,
    LinkingError,
    NautilusError,
)
from src.common.logging import get_logger

from .amd_backend import AMDArch, AMDBackend, AMDCompilationResult
from .fat_binary import FatBinary, KernelSection, SectionFormat
from .hardware_validator import HardwareValidator, ValidationMode, ValidationResult
from .intel_backend import IntelBackend, IntelCompilationResult, IntelTarget
from .linker import FatBinaryLinker, LinkingResult
from .metal_backend import MetalBackend, MetalCompilationResult, MetalTarget
from .nvidia_backend import NvidiaArch, NvidiaBackend, NvidiaCompilationResult

logger = get_logger(__name__)


@dataclass
class FatBinaryConfig:
    """Configuration for fat binary construction."""

    kernel_name: str
    kernel_source: str

    # Target architectures (all optional, all compiled by default)
    amd_arch: AMDArch = AMDArch.GFX942
    intel_target: IntelTarget = IntelTarget.XE_HPG
    nvidia_arch: NvidiaArch = NvidiaArch.SM_90
    metal_target: MetalTarget = MetalTarget.APPLE_M2

    # Compilation options
    block_m: int = 128
    block_n: int = 128
    block_k: int = 32
    num_warps: int = 8
    num_stages: int = 3

    # Output
    output_dir: str | None = None

    # Validation
    validation_mode: ValidationMode = ValidationMode.SKIP

    # Build options
    skip_amd: bool = False
    skip_intel: bool = False
    skip_nvidia: bool = False
    skip_apple: bool = False
    skip_validation: bool = False


@dataclass
class FatBinaryResult:
    """Result of the complete fat binary build."""

    success: bool
    fat_binary: FatBinary | None = None
    output_path: Path | None = None
    amd_result: AMDCompilationResult | None = None
    intel_result: IntelCompilationResult | None = None
    nvidia_result: NvidiaCompilationResult | None = None
    apple_result: MetalCompilationResult | None = None
    linking_result: LinkingResult | None = None
    validation_results: list[ValidationResult] = field(default_factory=list)
    error: str | None = None
    total_time_s: float = 0.0
    stage_times: dict[str, float] = field(default_factory=dict)

    @property
    def is_usable(self) -> bool:
        return self.success and self.fat_binary is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "kernel_name": self.fat_binary.kernel_name if self.fat_binary else "",
            "vendors": self.fat_binary.vendors if self.fat_binary else [],
            "total_size": self.fat_binary.total_size if self.fat_binary else 0,
            "output_path": str(self.output_path) if self.output_path else None,
            "total_time_s": self.total_time_s,
            "stage_times": self.stage_times,
        }


class FatBinaryBuilder:
    """Production orchestrator for the AOT fat binary pipeline.

    Usage:
        config = FatBinaryConfig(
            kernel_name="matmul",
            kernel_source=triton_source,
            block_m=128, block_n=128, block_k=32,
            num_warps=8, num_stages=3,
        )
        builder = FatBinaryBuilder()
        result = builder.build(config)
        if result.is_usable:
            # result.fat_binary contains all per-vendor sections
            # result.output_path points to the linked ELF
            ...
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        validation_mode: ValidationMode = ValidationMode.SKIP,
    ) -> None:
        self.cache_dir = Path(
            cache_dir
            or os.environ.get(
                "NAUTILUS_FAT_BINARY_CACHE",
                str(Path.home() / ".cache" / "nautilus" / "fat_binary"),
            )
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize per-vendor backends
        self.amd_backend = AMDBackend(
            target_arch=AMDArch.GFX942,
            cache_dir=str(self.cache_dir / "amd"),
        )
        self.intel_backend = IntelBackend(
            target=IntelTarget.XE_HPG,
            cache_dir=str(self.cache_dir / "intel"),
        )
        self.nvidia_backend = NvidiaBackend(
            target_arch=NvidiaArch.SM_90,
            cache_dir=str(self.cache_dir / "nvidia"),
        )
        self.metal_backend = MetalBackend(
            target=MetalTarget.APPLE_M2,
            cache_dir=str(self.cache_dir / "apple"),
        )
        self.linker = FatBinaryLinker(
            cache_dir=str(self.cache_dir / "link"),
        )
        self.validator = HardwareValidator(mode=validation_mode)

    def build(self, config: FatBinaryConfig) -> FatBinaryResult:
        """Build a fat binary from the given configuration.

        This is the main entry point. It:
          1. Compiles the kernel for each vendor (parallel-friendly)
          2. Compiles the C runtime stub
          3. Links everything via lld
          4. Optionally validates on hardware

        Args:
            config: FatBinaryConfig with kernel source and target info.

        Returns:
            FatBinaryResult with all per-stage results and the final fat binary.
        """
        start = time.perf_counter()
        stage_times: dict[str, float] = {}

        # Create the FatBinary container
        fat_binary = FatBinary(
            kernel_name=config.kernel_name,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

        # Stage 1: Compile for each vendor
        amd_result = None
        if not config.skip_amd:
            t0 = time.perf_counter()
            amd_result = self._build_amd(config)
            stage_times["amd"] = time.perf_counter() - t0
            if amd_result.is_usable and amd_result.hsaco_bytes is not None:
                fat_binary.add_section(
                    KernelSection(
                        vendor="amd",
                        arch=amd_result.arch,
                        format=SectionFormat.HSACO,
                        data=amd_result.hsaco_bytes,
                        metadata={"compilation_time_s": amd_result.compilation_time_s},
                    )
                )

        intel_result = None
        if not config.skip_intel:
            t0 = time.perf_counter()
            intel_result = self._build_intel(config)
            stage_times["intel"] = time.perf_counter() - t0
            if intel_result.is_usable and intel_result.spv_bytes is not None:
                fat_binary.add_section(
                    KernelSection(
                        vendor="intel",
                        arch=intel_result.target,
                        format=SectionFormat.SPV,
                        data=intel_result.spv_bytes,
                        metadata={"compilation_time_s": intel_result.compilation_time_s},
                    )
                )

        nvidia_result = None
        if not config.skip_nvidia:
            t0 = time.perf_counter()
            nvidia_result = self._build_nvidia(config)
            stage_times["nvidia"] = time.perf_counter() - t0
            if nvidia_result.is_usable and nvidia_result.ptx_text is not None:
                fat_binary.add_section(
                    KernelSection(
                        vendor="nvidia",
                        arch=nvidia_result.arch,
                        format=SectionFormat.PTX,
                        data=nvidia_result.ptx_text.encode("utf-8"),
                        metadata={"compilation_time_s": nvidia_result.compilation_time_s},
                    )
                )

        apple_result = None
        if not config.skip_apple:
            t0 = time.perf_counter()
            apple_result = self._build_apple(config)
            stage_times["apple"] = time.perf_counter() - t0
            if apple_result.is_usable and apple_result.output_bytes is not None:
                if apple_result.metallib_bytes is not None:
                    apple_fmt = SectionFormat.METALLIB
                elif apple_result.air_bytes is not None:
                    apple_fmt = SectionFormat.AIR
                else:
                    # Only MSL text is available. The linker only
                    # embeds metallib / air binaries, so we must skip
                    # the Apple section to avoid putting unparseable
                    # source into the fat binary.
                    logger.warning(
                        "Apple backend produced only MSL text; "
                        "no metallib or AIR bytes available. "
                        "Skipping the Apple section. To embed an Apple "
                        "section, install the Xcode Command Line Tools "
                        "and ensure `xcrun metallib` is on PATH.",
                        target=apple_result.target,
                        kernel=config.kernel_name,
                    )
                if apple_result.metallib_bytes is not None or apple_result.air_bytes is not None:
                    fat_binary.add_section(
                        KernelSection(
                            vendor="apple",
                            arch=apple_result.target,
                            format=apple_fmt,
                            data=apple_result.output_bytes,
                            metadata={
                                "compilation_time_s": apple_result.compilation_time_s,
                                "used_triton_metal_target": apple_result.used_triton_metal_target,
                                "xcrun_version": apple_result.xcrun_version,
                            },
                        )
                    )

        # Stage 2: Compile the C runtime stub
        t0 = time.perf_counter()
        runtime_stub_o = self._compile_runtime_stub(config)
        stage_times["runtime_stub"] = time.perf_counter() - t0

        # Stage 3: Link the fat binary
        output_dir = Path(config.output_dir) if config.output_dir else self.cache_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{config.kernel_name}.fat.o"

        t0 = time.perf_counter()
        linking_result = self.linker.link_fat_binary(
            nvidia_ptx=nvidia_result.ptx_text.encode("utf-8")
            if nvidia_result and nvidia_result.ptx_text
            else None,
            nvidia_cubin=nvidia_result.cubin_bytes
            if nvidia_result and nvidia_result.cubin_bytes
            else None,
            amd_hsaco=amd_result.hsaco_bytes if amd_result and amd_result.hsaco_bytes else None,
            intel_spv=intel_result.spv_bytes if intel_result and intel_result.spv_bytes else None,
            apple_metallib=self._apple_link_bytes(apple_result),
            runtime_stub_o=runtime_stub_o,
            kernel_name=config.kernel_name,
            output_path=output_path,
        )
        stage_times["link"] = time.perf_counter() - t0

        # Stage 4: Validation (optional)
        validation_results: list[ValidationResult] = []
        if (
            not config.skip_validation
            and linking_result.is_usable
            and linking_result.output_path is not None
        ):
            t0 = time.perf_counter()
            for section in fat_binary.sections:
                binary_path = linking_result.output_path
                if (
                    (section.format == SectionFormat.HSACO and amd_result)
                    or (section.format == SectionFormat.SPV and intel_result)
                    or (section.format == SectionFormat.PTX and nvidia_result)
                    or (
                        section.format in (SectionFormat.METALLIB, SectionFormat.AIR)
                        and apple_result
                    )
                ):
                    validation_results.append(
                        self.validator.validate(
                            binary_path=binary_path,
                            vendor=section.vendor,
                            arch=section.arch,
                        )
                    )
            stage_times["validation"] = time.perf_counter() - t0

        # Check if at least one vendor compiled
        if not fat_binary.sections:
            elapsed = time.perf_counter() - start
            return FatBinaryResult(
                success=False,
                error="No vendor compilations succeeded",
                total_time_s=elapsed,
                stage_times=stage_times,
            )

        total_elapsed = time.perf_counter() - start
        return FatBinaryResult(
            success=linking_result.is_usable,
            fat_binary=fat_binary,
            output_path=linking_result.output_path if linking_result.is_usable else None,
            amd_result=amd_result,
            intel_result=intel_result,
            nvidia_result=nvidia_result,
            apple_result=apple_result,
            linking_result=linking_result,
            validation_results=validation_results,
            total_time_s=total_elapsed,
            stage_times=stage_times,
        )

    def _build_amd(self, config: FatBinaryConfig) -> AMDCompilationResult:
        """Compile the kernel for AMD."""
        try:
            return self.amd_backend.compile_kernel(
                kernel_source=config.kernel_source,
                kernel_name=config.kernel_name,
                block_m=config.block_m,
                block_n=config.block_n,
                block_k=config.block_k,
                num_warps=config.num_warps,
                num_stages=config.num_stages,
            )
        except Exception as exc:
            logger.error("AMD build failed: %s", exc)
            return AMDCompilationResult(
                success=False,
                arch=config.amd_arch.value,
                error=str(exc),
            )

    def _build_intel(self, config: FatBinaryConfig) -> IntelCompilationResult:
        """Compile the kernel for Intel."""
        try:
            return self.intel_backend.compile_kernel(
                triton_kernel_ir=config.kernel_source,
                kernel_name=config.kernel_name,
                block_m=config.block_m,
                block_n=config.block_n,
                block_k=config.block_k,
                num_warps=config.num_warps,
            )
        except Exception as exc:
            logger.error("Intel build failed: %s", exc)
            return IntelCompilationResult(
                success=False,
                target=config.intel_target.value,
                error=str(exc),
            )

    def _build_nvidia(self, config: FatBinaryConfig) -> NvidiaCompilationResult:
        """Compile the kernel for Nvidia."""
        try:
            return self.nvidia_backend.compile_kernel(
                kernel_source=config.kernel_source,
                kernel_name=config.kernel_name,
                block_m=config.block_m,
                block_n=config.block_n,
                block_k=config.block_k,
                num_warps=config.num_warps,
                num_stages=config.num_stages,
            )
        except Exception as exc:
            logger.error("Nvidia build failed: %s", exc)
            return NvidiaCompilationResult(
                success=False,
                arch=config.nvidia_arch.value,
                error=str(exc),
            )

    def _build_apple(self, config: FatBinaryConfig) -> MetalCompilationResult:
        """Compile the kernel for Apple Silicon (Metal).

        On non-Apple hosts this returns a failed
        ``MetalCompilationResult`` carrying the
        ``E_HARDWARE_NOT_FOUND`` error code — never raises. The
        caller (``build``) treats the failed result as "skip
        this vendor" rather than aborting the whole build.
        """
        try:
            return self.metal_backend.compile_kernel(
                kernel_source=config.kernel_source,
                kernel_name=config.kernel_name,
                block_m=config.block_m,
                block_n=config.block_n,
                block_k=config.block_k,
                num_warps=config.num_warps,
                num_stages=config.num_stages,
            )
        except Exception as exc:
            logger.error("Apple build failed: %s", exc)
            return MetalCompilationResult(
                success=False,
                target=config.metal_target.value,
                error=str(exc),
                error_code="E_COMPILATION_FAILED",
            )

    def _apple_link_bytes(
        self,
        apple_result: MetalCompilationResult | None,
    ) -> bytes | None:
        """Choose which Apple artifact bytes to embed in the fat binary.

        Preference order:

          1. ``metallib_bytes`` (preferred — what the runtime
             loader wants).
          2. ``air_bytes`` (still loadable via
             ``newLibraryWithData:``).
          3. ``msl_text`` bytes (last-resort — the runtime
             loader will refuse; useful for debugging).

        Returns ``None`` if the Apple build failed or produced
        no bytes, in which case the linker is invoked without an
        Apple section.
        """
        if apple_result is None or not apple_result.success:
            return None
        return apple_result.output_bytes

    def _compile_runtime_stub(self, config: FatBinaryConfig) -> bytes:
        """Compile the C runtime stub into an object file.

        The stub is a real C file with /dev probing and CPUID-based
        vendor detection. We compile it with gcc to produce a
        relocatable object file that lld combines with the per-vendor
        kernel sections.

        Hard requirement: BOTH gcc AND lld must be available. Without
        lld, the linker step cannot produce a real fat binary ELF and
        the runtime dispatch cannot work. Raises LinkingError in that
        case — never silently returns a non-functional stub.
        """
        if not self.linker._lld_path:
            raise LinkingError(
                "lld not found in PATH; cannot link fat binary. "
                "Install LLVM (apt install lld / brew install llvm). "
                "The C runtime stub is compiled but useless without lld "
                "to combine it with the per-vendor kernel sections.",
                context={"kernel": config.kernel_name},
            )

        stub_path = self.cache_dir / "runtime_stub.c"
        stub_path.write_text(self._read_runtime_stub_source())

        output_path = self.cache_dir / f"tmp_runtime_stub_{config.kernel_name}.o"

        if not shutil.which("gcc"):
            raise DependencyMissingError(
                "gcc not found in PATH; cannot compile C runtime stub. "
                "Install gcc (apt install gcc / brew install gcc).",
            )
        try:
            # Add -I flag so #include "../../c_api/triton_c_api.h" resolves
            # even when runtime_stub.c is compiled from a temp cache dir.
            c_api_include = str(Path(__file__).resolve().parent.parent.parent / "c_api")
            cmd = [
                "gcc",
                "-c",
                "-fPIC",
                f"-I{c_api_include}",
                "-o",
                str(output_path),
                str(stub_path),
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0 or not output_path.exists():
                raise CompilationError(
                    f"gcc failed to compile runtime_stub.c: {result.stderr}",
                    context={"stdout": result.stdout, "stderr": result.stderr},
                )
            return output_path.read_bytes()
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            raise CompilationError(
                f"Failed to compile runtime_stub.c: {exc}",
                cause=exc,
            ) from exc
        finally:
            stub_path.unlink(missing_ok=True)

    def _read_runtime_stub_source(self) -> str:
        """Read the C runtime stub source from the package data.

        The previous design embedded a 4-line default-dispatch stub as
        a string here, overwriting the real 100+ line runtime_stub.c
        on every build. This implementation reads the real stub from
        the package's bundled .c file via importlib.resources.
        """
        from importlib import resources

        try:
            return (resources.files("src.bridges.aot_packager") / "runtime_stub.c").read_text()
        except (FileNotFoundError, ModuleNotFoundError):
            # Fallback: read relative to this file's location
            here = Path(__file__).parent
            stub = here / "runtime_stub.c"
            if stub.exists():
                return stub.read_text()
            raise NautilusError(
                f"runtime_stub.c not found in package; looked at {here}",
            ) from None
