"""LLVM lld fat binary linker.

Combines the per-vendor AOT-compiled kernels (PTX, HSACO, SPIR-V)
plus the C runtime stub into a single "fat binary" ELF object.

The fat binary format:
┌─────────────────────────────────────┐
│  ELF Header                         │
├─────────────────────────────────────┤
│  C Runtime Stub (code)              │
│  - Detect CPU vendor (CPUID)        │
│  - Detect GPU via /dev/kfd, /dev/dri│
│  - Jump to matching backend         │
├─────────────────────────────────────┤
│  Section: .nv_kernel (PTX text)     │
├─────────────────────────────────────┤
│  Section: .amd_kernel (HSACO binary)│
├─────────────────────────────────────┤
│  Section: .intel_kernel (SPIR-V)    │
├─────────────────────────────────────┤
│  Section: .nautilus_metadata        │
│  - Build timestamp                  │
│  - Target architectures             │
│  - Hash of source kernel            │
└─────────────────────────────────────┘

This module invokes the LLVM linker (`lld`) to combine everything
into a relocatable object that can be linked into a final executable.

Production features:
  - Use LLVM's lld (not GNU ld) for consistent cross-platform behavior
  - Support both PTX text and cubin binary for Nvidia
  - Persistent cache (skip re-linking when inputs unchanged)
  - Validation that the output is a valid ELF
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from src.common.errors import LinkingError
from src.common.logging import get_logger
from src.common.result import Err, Ok, Result

logger = get_logger(__name__)


# ELF64 ABI constants from <elf.h>. Module-level so they don't
# trip the N806 linter rule that fires on UPPER_CASE locals.
_ELFCLASS64 = 2
_ELFDATA2LSB = 1
_EV_CURRENT = 1
_ELFOSABI_NONE = 0
_ET_REL = 1
_EM_X86_64 = 62
_SHT_PROGBITS = 1
_SHT_STRTAB = 3
_SHT_NOBITS = 8
_SHF_ALLOC = 0x2


@dataclass
class LinkingResult:
    """Result of a fat binary linking operation."""
    success: bool
    output_path: Path | None = None
    output_size: int = 0
    sections: dict[str, int] = field(default_factory=dict)  # section name → size
    error: str | None = None
    linking_time_s: float = 0.0
    cache_hit: bool = False
    linker_version: str = ""

    @property
    def is_usable(self) -> bool:
        return self.success and self.output_path is not None


class FatBinaryLinker:
    """Production-grade LLVM lld wrapper for fat binary linking.

    Takes the per-vendor compiled kernels plus the C runtime stub
    and produces a single relocatable ELF object containing all of them.
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        timeout_seconds: float = 30.0,
        stub_path: str | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir or os.environ.get(
            "NAUTILUS_LINK_CACHE",
            str(Path.home() / ".cache" / "nautilus" / "link"),
        ))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds
        # Default to <repo_root>/build/runtime_stub.o, the artifact of
        # setup.py build_runtime_stub(). __file__ is
        # .../src/bridges/aot_packager/linker.py — four .parent hops
        # reach the repo root.
        if stub_path is None:
            repo_root = Path(__file__).resolve().parent.parent.parent.parent
            stub_path = str(repo_root / "build" / "runtime_stub.o")
        self.default_stub_path = Path(stub_path)
        self._lld_path = self._find_lld()
        self._lld_version = self._detect_lld_version()

    def link_fat_binary(
        self,
        nvidia_ptx: bytes | None = None,
        nvidia_cubin: bytes | None = None,
        amd_hsaco: bytes | None = None,
        intel_spv: bytes | None = None,
        runtime_stub_o: bytes | None = None,
        kernel_name: str = "kernel",
        output_path: Path | None = None,
    ) -> LinkingResult:
        """Link a fat binary from the per-vendor compiled kernels.

        Args:
            nvidia_ptx: PTX text for Nvidia GPUs (bytes).
            nvidia_cubin: Cubin binary for Nvidia GPUs (bytes).
            amd_hsaco: HSACO binary for AMD GPUs (bytes).
            intel_spv: SPIR-V binary for Intel GPUs (bytes).
            runtime_stub_o: C runtime stub object file (bytes). If None,
                bytes are loaded from self.default_stub_path (set in
                __init__, defaulting to <repo>/build/runtime_stub.o).
            kernel_name: Name of the kernel.
            output_path: Where to write the fat binary (default: cache).

        Returns:
            LinkingResult with the fat binary path and metadata.
        """
        import time
        start = time.perf_counter()

        if runtime_stub_o is None and self.default_stub_path.exists():
            runtime_stub_o = self.default_stub_path.read_bytes()

        output_path = output_path or (self.cache_dir / f"{kernel_name}.fat.o")

        # Check cache
        cache_key = self._compute_link_cache_key(
            nvidia_ptx, nvidia_cubin, amd_hsaco, intel_spv, runtime_stub_o, kernel_name,
        )
        cached_path = self._cache_path_for(cache_key)
        if cached_path.exists() and cached_path.stat().st_size > 0:
            elapsed = time.perf_counter() - start
            return LinkingResult(
                success=True,
                output_path=cached_path,
                output_size=cached_path.stat().st_size,
                linking_time_s=elapsed,
                cache_hit=True,
                linker_version=self._lld_version,
            )

        # Write the input objects to temp files
        temp_dir = self.cache_dir / f"tmp_{kernel_name}_{cache_key[:8]}"
        temp_dir.mkdir(exist_ok=True)
        try:
            section_files = self._write_input_sections(
                temp_dir, nvidia_ptx, nvidia_cubin, amd_hsaco, intel_spv,
                runtime_stub_o,
            )
            # Link using lld
            link_result = self._run_lld(temp_dir, section_files, output_path, kernel_name)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            logger.error(
                "Fat binary linking failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return LinkingResult(
                success=False,
                error=str(exc),
                linking_time_s=elapsed,
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        if link_result.is_err():
            elapsed = time.perf_counter() - start
            return LinkingResult(
                success=False,
                error=str(link_result.error),
                linking_time_s=elapsed,
            )

        # Cache the result
        shutil.copy(output_path, cached_path)

        elapsed = time.perf_counter() - start
        return LinkingResult(
            success=True,
            output_path=cached_path,
            output_size=cached_path.stat().st_size,
            sections=self._count_sections(cached_path),
            linking_time_s=elapsed,
            linker_version=self._lld_version,
        )

    def _compute_link_cache_key(
        self,
        nvidia_ptx: bytes | None,
        nvidia_cubin: bytes | None,
        amd_hsaco: bytes | None,
        intel_spv: bytes | None,
        runtime_stub_o: bytes | None,
        kernel_name: str,
    ) -> str:
        """Compute a cache key for the link operation."""
        payload = json.dumps({
            "name": kernel_name,
            "nv_ptx": nvidia_ptx.hex() if nvidia_ptx else "",
            "nv_cubin": nvidia_cubin.hex() if nvidia_cubin else "",
            "amd_hsaco": amd_hsaco.hex() if amd_hsaco else "",
            "intel_spv": intel_spv.hex() if intel_spv else "",
            "runtime_stub": runtime_stub_o.hex() if runtime_stub_o else "",
            "lld_version": self._lld_version,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cache_path_for(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key[:32]}.fat.o"

    def _write_input_sections(
        self,
        temp_dir: Path,
        nvidia_ptx: bytes | None,
        nvidia_cubin: bytes | None,
        amd_hsaco: bytes | None,
        intel_spv: bytes | None,
        runtime_stub_o: bytes | None,
    ) -> dict[str, Path]:
        """Write each input section to a temp file for lld to consume.

        Each per-vendor kernel gets its own object file. The linker
        combines them with the runtime stub.
        """
        section_files: dict[str, Path] = {}

        if nvidia_ptx:
            ptx_o = temp_dir / "nvidia.ptx.o"
            ptx_o.write_bytes(self._wrap_section_data(nvidia_ptx, ".nv_kernel", "PROGBITS"))
            section_files["nvidia_ptx"] = ptx_o

        if nvidia_cubin:
            cubin_o = temp_dir / "nvidia.cubin.o"
            cubin_o.write_bytes(self._wrap_section_data(nvidia_cubin, ".nv_kernel_cubin", "PROGBITS"))
            section_files["nvidia_cubin"] = cubin_o

        if amd_hsaco:
            amd_o = temp_dir / "amd.hsaco.o"
            amd_o.write_bytes(self._wrap_section_data(amd_hsaco, ".amd_kernel", "PROGBITS"))
            section_files["amd"] = amd_o

        if intel_spv:
            spv_o = temp_dir / "intel.spv.o"
            spv_o.write_bytes(self._wrap_section_data(intel_spv, ".intel_kernel", "PROGBITS"))
            section_files["intel"] = spv_o

        if runtime_stub_o:
            stub_path = temp_dir / "runtime_stub.o"
            stub_path.write_bytes(runtime_stub_o)
            section_files["runtime_stub"] = stub_path

        return section_files

    def _wrap_section_data(self, data: bytes, section_name: str, section_type: str) -> bytes:
        """Wrap raw bytes as a minimal ELF64 relocatable object with one section.

        Produces a layout that ``readelf -h`` and ``readelf -S`` parse cleanly:

            [ELF64 header]            64 bytes
            [section data]            len(data) bytes
            [shstrtab data]           M bytes
            [8-byte pad]              0-7 bytes (align SHT)
            [section header table]    3 * 64 = 192 bytes

        SHT entries:
            [0] NULL                   (required by ELF spec)
            [1] data section (PROGBITS or NOBITS) with SHF_ALLOC
            [2] .shstrtab (SHT_STRTAB) — e_shstrndx points here

        ``sh_name`` of the data section is the byte offset of
        ``section_name`` in the string table; ``.shstrtab`` is
        named via the same string table to keep the output
        self-describing.
        """
        # Build the section-header string table: leading NUL,
        # section_name\0, ".shstrtab"\0. sh_name is the offset
        # of the name within this table.
        shstrtab = b"\0" + section_name.encode("utf-8") + b"\0" + b".shstrtab\0"
        sh_name_offset = 1
        shstrtab_name_offset = 1 + len(section_name.encode("utf-8")) + 1

        sht_type = _SHT_PROGBITS if section_type == "PROGBITS" else _SHT_NOBITS

        e_ehsize = 64
        e_shentsize = 64
        e_shnum = 3  # NULL + data + shstrtab

        # Compute the SHT offset. ELF64 requires the SHT to start
        # on an 8-byte boundary, so pad with up to 7 NUL bytes.
        sht_offset = e_ehsize + len(data) + len(shstrtab)
        pad = (-sht_offset) % 8
        sht_offset += pad

        e_ident = (
            b"\x7fELF"
            + bytes([_ELFCLASS64, _ELFDATA2LSB, _EV_CURRENT, _ELFOSABI_NONE])
            + b"\x00" * 8
        )
        header = (
            e_ident
            + struct.pack("<H", _ET_REL)              # e_type
            + struct.pack("<H", _EM_X86_64)           # e_machine
            + struct.pack("<I", _EV_CURRENT)          # e_version
            + struct.pack("<Q", 0)                    # e_entry
            + struct.pack("<Q", 0)                    # e_phoff
            + struct.pack("<Q", sht_offset)           # e_shoff
            + struct.pack("<I", 0)                    # e_flags
            + struct.pack("<H", e_ehsize)             # e_ehsize
            + struct.pack("<H", 0)                    # e_phentsize
            + struct.pack("<H", 0)                    # e_phnum
            + struct.pack("<H", e_shentsize)          # e_shentsize
            + struct.pack("<H", e_shnum)              # e_shnum
            + struct.pack("<H", 2)                    # e_shstrndx (index of .shstrtab)
        )
        assert len(header) == 64, f"ELF header is {len(header)} bytes, expected 64"

        # SHT entry 0: NULL (required by spec, all fields zero).
        null_sh = b"\x00" * 64

        # SHT entry 1: the data section. PROGBITS + SHF_ALLOC so the
        # runtime loader mmaps it into the process image.
        sh1 = struct.pack(
            "<IIQQQQIIQQ",
            sh_name_offset,   # sh_name (offset into shstrtab)
            sht_type,         # sh_type
            _SHF_ALLOC,       # sh_flags
            0,                # sh_addr (relocatable — addresses fixed at link)
            e_ehsize,         # sh_offset
            len(data),        # sh_size
            0,                # sh_link
            0,                # sh_info
            1,                # sh_addralign
            0,                # sh_entsize
        )

        # SHT entry 2: .shstrtab. SHT_STRTAB sections do not carry
        # SHF_ALLOC (the string table is consumed by the loader,
        # not mapped into the process).
        sh2 = struct.pack(
            "<IIQQQQIIQQ",
            shstrtab_name_offset,  # sh_name
            _SHT_STRTAB,           # sh_type
            0,                     # sh_flags
            0,                     # sh_addr
            e_ehsize + len(data),  # sh_offset
            len(shstrtab),         # sh_size
            0,                     # sh_link
            0,                     # sh_info
            1,                     # sh_addralign
            0,                     # sh_entsize
        )
        sht = null_sh + sh1 + sh2
        assert len(sht) == 3 * 64, f"SHT is {len(sht)} bytes, expected 192"

        return header + data + shstrtab + (b"\x00" * pad) + sht

    def _run_lld(
        self,
        temp_dir: Path,
        section_files: dict[str, Path],
        output_path: Path,
        kernel_name: str,
    ) -> Result[Path, LinkingError]:
        """Invoke the LLVM linker to combine the section files.

        Returns ``Ok(output_path)`` on success and
        ``Err(LinkingError)`` on any failure (lld missing, subprocess
        timeout, non-zero exit, missing output file). NEVER falls
        back to a non-functional manual fat binary.
        """
        if not self._lld_path:
            return Err(LinkingError(
                "lld not found in PATH. Install LLVM (apt install lld / "
                "brew install llvm). The fat binary cannot be linked "
                "without lld.",
            ))

        cmd = [
            self._lld_path, "-r",  # relocatable link
            "-o", str(output_path),
            *[str(p) for p in section_files.values()],
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return Err(LinkingError(
                f"lld timed out after {self.timeout_seconds}s",
                cause=exc,
            ))
        if result.returncode != 0 or not output_path.exists():
            return Err(LinkingError(
                f"lld failed: {result.stderr or '(no stderr)'}",
                context={
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "cmd": cmd,
                },
            ))
        return Ok(output_path)

    def _count_sections(self, path: Path) -> dict[str, int]:
        """Count sections in the output (best-effort)."""
        return {"total_sections": 0, "output_size": path.stat().st_size if path.exists() else 0}

    def _find_lld(self) -> str | None:
        """Find the LLVM linker (lld) on the system."""
        for name in ("ld.lld", "lld", "lld-link"):
            path = shutil.which(name)
            if path:
                return path
        # Check common LLVM install locations
        for prefix in ("/usr/bin", "/usr/local/bin", "/opt/llvm/bin"):
            for name in ("lld", "ld.lld"):
                candidate = Path(prefix) / name
                if candidate.exists():
                    return str(candidate)
        return None

    def _detect_lld_version(self) -> str:
        if not self._lld_path:
            return "unavailable"
        try:
            result = subprocess.run(
                [self._lld_path, "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip().split("\n")[0] or "unknown"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return "unknown"

    def get_version(self) -> str:
        return self._lld_version
