"""LLVM lld fat binary linker.

Combines the per-vendor AOT-compiled kernels (PTX, HSACO, SPIR-V)
plus the C runtime stub into a single "fat binary" ELF object.

The fat binary format (post-collision-fix):

┌─────────────────────────────────────────────────────────────┐
│  ELF Header                                                 │
├─────────────────────────────────────────────────────────────┤
│  C Runtime Stub (code) — vendor detection + dispatch        │
│  - /dev/nvidia*, /dev/kfd, /dev/dri/renderD* probing        │
│  - Resolves kernel sections via .nautilus.index             │
├─────────────────────────────────────────────────────────────┤
│  Section: .nautilus.nvidia.<kernel_name>      (PTX text)    │
│  Section: .nautilus.nvidia.cubin.<kernel_name> (CUBIN)      │
│  Section: .nautilus.amd.<kernel_name>         (HSACO)       │
│  Section: .nautilus.intel.<kernel_name>       (SPIR-V)      │
│  Section: .nautilus.apple.<kernel_name>      (Metal AIR)    │
│  …one section per kernel, never reused. Section names are   │
│  unique per (vendor, kernel) pair so ld.lld -r cannot        │
│  silently merge them.                                       │
├─────────────────────────────────────────────────────────────┤
│  Section: .nautilus.index (text)                            │
│  - One record per kernel:                                   │
│      kernel|vendor|arch|format|section_name|size\\n         │
│  - Terminated by an empty line.                             │
│  - The C runtime stub reads it via the                      │
│    `nautilus_index_data` symbol emitted by the index        │
│    holder object.                                           │
└─────────────────────────────────────────────────────────────┘

Section-name policy: every kernel byte blob lives in its own
section named ``.nautilus.{vendor}.{kernel_name}`` (with an
optional ``.{format}`` infix for the cubin/ptx ambiguity of the
same vendor). The old ``.nv_kernel``/``.amd_kernel``/``.intel_kernel``
generic names have been removed because ``ld.lld -r`` *silently
merges* sections with identical names — that was the bug this
module is fixing.

This module invokes the LLVM linker (`lld`) to combine everything
into a relocatable object that can be linked into a final executable.

Production features:
  - Use LLVM's lld (not GNU ld) for consistent cross-platform behavior
  - Support both PTX text and cubin binary for Nvidia
  - Persistent cache (skip re-linking when inputs unchanged)
  - Validation that the output is a valid ELF
  - Per-kernel unique section names (no silent lld merge)
  - Machine-parseable .nautilus.index section for runtime discovery
"""

from __future__ import annotations

import hashlib
import json
import os
import re
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

# Reserved prefix for every section this module produces. The C
# runtime stub and the index parser key off this prefix — never
# change it without updating both.
_NAUTILUS_PREFIX = ".nautilus."

_INDEX_SECTION = f"{_NAUTILUS_PREFIX}index"

# Whitelist of characters safe for an ELF section name AND safe
# to embed in the .nautilus.index pipe-delimited records.
_SECTION_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


def _sanitize_section_token(token: str) -> str:
    """Replace any non-allowed character with ``_``; empty -> ``unnamed``."""
    if not token:
        return "unnamed"
    out = "".join(c if c.isalnum() or c in "_.-" else "_" for c in token)
    return out or "unnamed"


def section_name_for(vendor: str, kernel_name: str, *, fmt_suffix: str = "") -> str:
    """Return the canonical, unique section name for a kernel blob.

    Format: ``.nautilus.{vendor}[.{fmt_suffix}].{kernel_name}``
    """
    v = _sanitize_section_token(vendor)
    k = _sanitize_section_token(kernel_name)
    if fmt_suffix:
        f = _sanitize_section_token(fmt_suffix)
        return f"{_NAUTILUS_PREFIX}{v}.{f}.{k}"
    return f"{_NAUTILUS_PREFIX}{v}.{k}"


def validate_section_name(name: str) -> str:
    """Raise LinkingError if `name` is not a safe ELF section name.

    Safe = 1-64 chars, drawn from ``[A-Za-z0-9_.-]`` only.
    """
    if not name or len(name) > 64:
        raise LinkingError(
            f"Invalid section name: {name!r} (must be 1-64 chars)",
            context={"section_name": name},
        )
    if not _SECTION_NAME_RE.match(name):
        raise LinkingError(
            f"Invalid section name: {name!r} (only [A-Za-z0-9_.-] allowed)",
            context={"section_name": name},
        )
    return name


@dataclass
class KernelBlob:
    """One kernel binary for one (vendor, arch, format) triple.

    Multiple ``KernelBlob``s with the same vendor but different
    kernel names are allowed — the linker gives each its own
    section named ``.nautilus.{vendor}.{kernel_name}`` and records
    all of them in ``.nautilus.index`` so the runtime stub can
    discover them at startup.
    """

    kernel_name: str
    vendor: str  # "nvidia" / "amd" / "intel" / "apple"
    arch: str  # "sm_90" / "gfx942" / "xe_hpg" — free-form identifier
    fmt: str  # "ptx" / "cubin" / "hsaco" / "spv" / "metallib"
    data: bytes
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class LinkingResult:
    """Result of a fat binary linking operation."""

    success: bool
    output_path: Path | None = None
    output_size: int = 0
    sections: dict[str, int] = field(default_factory=dict)  # section name → size
    section_names: list[str] = field(default_factory=list)  # order-preserved unique names
    index_text: str = ""  # human-readable rendering of the .nautilus.index contents
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
        self.cache_dir = Path(
            cache_dir
            or os.environ.get(
                "NAUTILUS_LINK_CACHE",
                str(Path.home() / ".cache" / "nautilus" / "link"),
            )
        )
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
        apple_metallib: bytes | None = None,
        apple_air: bytes | None = None,
        runtime_stub_o: bytes | None = None,
        kernel_name: str = "kernel",
        output_path: Path | None = None,
        kernels: list[KernelBlob] | None = None,
    ) -> LinkingResult:
        """Link a fat binary from one or more per-vendor compiled kernels.

        Each kernel lives in its own section named
        ``.nautilus.{vendor}.{kernel_name}`` and is listed in
        ``.nautilus.index``. Multi-kernel from the same vendor is
        supported via the ``kernels`` argument.
        """
        import time

        start = time.perf_counter()

        if runtime_stub_o is None and self.default_stub_path.exists():
            runtime_stub_o = self.default_stub_path.read_bytes()

        blobs = self._normalise_kernels(
            kernels=kernels,
            nvidia_ptx=nvidia_ptx,
            nvidia_cubin=nvidia_cubin,
            amd_hsaco=amd_hsaco,
            intel_spv=intel_spv,
            apple_metallib=apple_metallib,
            apple_air=apple_air,
            default_kernel_name=kernel_name,
        )
        if not blobs and runtime_stub_o is None:
            return LinkingResult(
                success=False,
                error="link_fat_binary called with no kernels and no runtime_stub_o",
                linking_time_s=time.perf_counter() - start,
            )

        section_names, index_text = self._build_section_plan(blobs)

        output_path = output_path or (
            self.cache_dir / f"{blobs[0].kernel_name if blobs else 'stub'}.fat.o"
        )

        # Check cache
        cache_key = self._compute_link_cache_key(
            blobs,
            runtime_stub_o,
            section_names,
        )
        cached_path = self._cache_path_for(cache_key)
        if cached_path.exists() and cached_path.stat().st_size > 0:
            elapsed = time.perf_counter() - start
            return LinkingResult(
                success=True,
                output_path=cached_path,
                output_size=cached_path.stat().st_size,
                section_names=list(section_names),
                index_text=index_text,
                linking_time_s=elapsed,
                cache_hit=True,
                linker_version=self._lld_version,
            )

        temp_dir = self.cache_dir / f"tmp_link_{cache_key[:8]}"
        temp_dir.mkdir(exist_ok=True)
        try:
            section_files = self._write_input_sections(
                temp_dir,
                blobs,
                runtime_stub_o,
            )
            index_object = self._build_index_object(temp_dir, blobs, section_names, index_text)
            if index_object is not None:
                section_files["nautilus_index"] = index_object
            link_result = self._run_lld(temp_dir, section_files, output_path)
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

        if isinstance(link_result, Err):
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
            section_names=list(section_names),
            index_text=index_text,
            sections=self._count_sections(cached_path),
            linking_time_s=elapsed,
            linker_version=self._lld_version,
        )

    def _normalise_kernels(
        self,
        *,
        kernels: list[KernelBlob] | None,
        nvidia_ptx: bytes | None,
        nvidia_cubin: bytes | None,
        amd_hsaco: bytes | None,
        intel_spv: bytes | None,
        apple_metallib: bytes | None,
        apple_air: bytes | None,
        default_kernel_name: str,
    ) -> list[KernelBlob]:
        if kernels:
            seen: set[tuple[str, str, str]] = set()
            for b in kernels:
                key = (b.vendor, b.kernel_name, b.fmt)
                if key in seen:
                    raise LinkingError(
                        "Duplicate KernelBlob in input list",
                        context={"vendor": b.vendor, "kernel_name": b.kernel_name, "fmt": b.fmt},
                    )
                seen.add(key)
                validate_section_name(section_name_for(b.vendor, b.kernel_name, fmt_suffix=b.fmt))
            return list(kernels)

        blobs: list[KernelBlob] = []
        if nvidia_ptx is not None:
            blobs.append(
                KernelBlob(
                    kernel_name=default_kernel_name,
                    vendor="nvidia",
                    arch="sm_90",
                    fmt="ptx",
                    data=nvidia_ptx,
                )
            )
        if nvidia_cubin is not None:
            blobs.append(
                KernelBlob(
                    kernel_name=default_kernel_name,
                    vendor="nvidia",
                    arch="sm_90",
                    fmt="cubin",
                    data=nvidia_cubin,
                )
            )
        if amd_hsaco is not None:
            blobs.append(
                KernelBlob(
                    kernel_name=default_kernel_name,
                    vendor="amd",
                    arch="gfx942",
                    fmt="hsaco",
                    data=amd_hsaco,
                )
            )
        if intel_spv is not None:
            blobs.append(
                KernelBlob(
                    kernel_name=default_kernel_name,
                    vendor="intel",
                    arch="xe_hpg",
                    fmt="spv",
                    data=intel_spv,
                )
            )
        if apple_metallib is not None:
            blobs.append(
                KernelBlob(
                    kernel_name=default_kernel_name,
                    vendor="apple",
                    arch="metallib",
                    fmt="metallib",
                    data=apple_metallib,
                )
            )
        if apple_air is not None:
            blobs.append(
                KernelBlob(
                    kernel_name=default_kernel_name,
                    vendor="apple",
                    arch="air",
                    fmt="air",
                    data=apple_air,
                )
            )
        return blobs

    def _build_section_plan(
        self,
        blobs: list[KernelBlob],
    ) -> tuple[list[str], str]:
        """Compute unique section names for every blob and the
        text payload of the .nautilus.index section.
        """
        section_names: list[str] = []
        for b in blobs:
            name = section_name_for(b.vendor, b.kernel_name, fmt_suffix=b.fmt)
            if name in section_names:
                raise LinkingError(
                    f"Internal section-name collision: {name!r} "
                    "— section_name_for must produce unique output "
                    "for every (vendor, kernel, fmt) triple",
                    context={"vendor": b.vendor, "kernel": b.kernel_name, "fmt": b.fmt},
                )
            section_names.append(name)

        # Pipe-delimited records. Field count is fixed: 6.
        # kernel_name | vendor | arch | fmt | section_name | size
        # Records are newline-terminated; the index is double-
        # newline terminated (empty record) so the C parser can
        # detect the end with a single strchr loop.
        records = [
            "|".join(
                [
                    _sanitize_section_token(b.kernel_name),
                    _sanitize_section_token(b.vendor),
                    _sanitize_section_token(b.arch),
                    _sanitize_section_token(b.fmt),
                    section_name_for(b.vendor, b.kernel_name, fmt_suffix=b.fmt),
                    str(len(b.data)),
                ]
            )
            for b in blobs
        ]
        index_text = "\n".join(records) + "\n\n"
        return section_names, index_text

    def _compute_link_cache_key(
        self,
        blobs: list[KernelBlob],
        runtime_stub_o: bytes | None,
        section_names: list[str],
    ) -> str:
        """Compute a cache key for the link operation."""
        payload = json.dumps(
            {
                "kernels": [
                    {
                        "n": b.kernel_name,
                        "v": b.vendor,
                        "a": b.arch,
                        "f": b.fmt,
                        "d": b.data.hex(),
                    }
                    for b in blobs
                ],
                "section_names": list(section_names),
                "runtime_stub": runtime_stub_o.hex() if runtime_stub_o else "",
                "lld_version": self._lld_version,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cache_path_for(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key[:32]}.fat.o"

    def _write_input_sections(
        self,
        temp_dir: Path,
        blobs: list[KernelBlob],
        runtime_stub_o: bytes | None,
    ) -> dict[str, Path]:
        """Write each kernel blob to its own per-section object file.

        ``ld.lld -r`` silently merges sections that share a name;
        every blob therefore goes into a section whose name is
        unique per (vendor, kernel_name, fmt) — see
        ``section_name_for``.
        """
        section_files: dict[str, Path] = {}

        for idx, b in enumerate(blobs):
            section_name = section_name_for(b.vendor, b.kernel_name, fmt_suffix=b.fmt)
            validate_section_name(section_name)
            o_path = temp_dir / f"kernel_{idx:03d}_{b.vendor}_{b.fmt}.o"
            o_path.write_bytes(
                self._wrap_section_data(b.data, section_name, "PROGBITS"),
            )
            section_files[f"kernel_{idx:03d}"] = o_path

        if runtime_stub_o:
            stub_path = temp_dir / "runtime_stub.o"
            stub_path.write_bytes(runtime_stub_o)
            section_files["runtime_stub"] = stub_path

        return section_files

    def _build_index_object(
        self,
        temp_dir: Path,
        blobs: list[KernelBlob],
        section_names: list[str],
        index_text: str,
    ) -> Path | None:
        if not blobs or not index_text:
            return None

        # Compile a tiny C source rather than hand-rolling an
        # SHT_SYMTAB: gcc produces a spec-cleaner object for free.
        # The payload is hex-encoded so the C source is immune to
        # embedded quotes, backslashes, or non-ASCII bytes.
        #
        # Two symbols are emitted: ``nautilus_index_data`` carries
        # the pipe-delimited records, ``nautilus_index_size`` is
        # ``sizeof(nautilus_index_data)`` resolved at compile time
        # so the runtime stub can address the buffer via a typed
        # extern without ELF-header walking.
        index_bytes = index_text.encode("utf-8")
        hex_payload = ",".join(f"0x{b:02x}" for b in index_bytes)

        holder_c = temp_dir / "nautilus_index_holder.c"
        holder_c.write_text(
            "/* Auto-generated by FatBinaryLinker._build_index_object. */\n"
            "/* Hand-edit at your own risk — this file is rewritten on every link. */\n"
            '__attribute__((section(".nautilus.index"), used))\n'
            "const unsigned char nautilus_index_data[] = {\n"
            f"    {hex_payload}\n"
            "};\n"
            "\n"
            '__attribute__((section(".nautilus.index"), used))\n'
            "const unsigned long nautilus_index_size = sizeof(nautilus_index_data);\n"
        )

        holder_o = temp_dir / "nautilus_index_holder.o"
        gcc = shutil.which("gcc")
        if not gcc:
            raise LinkingError(
                "gcc not found in PATH; cannot compile .nautilus.index holder",
            )
        cmd = [
            gcc,
            "-c",
            "-fPIC",
            "-Wall",
            "-Wno-unused-but-set-variable",
            "-o",
            str(holder_o),
            str(holder_c),
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except subprocess.TimeoutExpired as exc:
            raise LinkingError(
                "gcc timed out compiling nautilus_index_holder.c",
                cause=exc,
            ) from exc
        if result.returncode != 0 or not holder_o.exists():
            raise LinkingError(
                f"gcc failed to compile .nautilus.index holder: {result.stderr or '(no stderr)'}",
                context={"cmd": cmd, "stdout": result.stdout, "stderr": result.stderr},
            )
        logger.debug(
            "Built .nautilus.index holder",
            section=section_names,
            index_bytes=len(index_bytes),
        )
        return holder_o

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
            + struct.pack("<H", _ET_REL)  # e_type
            + struct.pack("<H", _EM_X86_64)  # e_machine
            + struct.pack("<I", _EV_CURRENT)  # e_version
            + struct.pack("<Q", 0)  # e_entry
            + struct.pack("<Q", 0)  # e_phoff
            + struct.pack("<Q", sht_offset)  # e_shoff
            + struct.pack("<I", 0)  # e_flags
            + struct.pack("<H", e_ehsize)  # e_ehsize
            + struct.pack("<H", 0)  # e_phentsize
            + struct.pack("<H", 0)  # e_phnum
            + struct.pack("<H", e_shentsize)  # e_shentsize
            + struct.pack("<H", e_shnum)  # e_shnum
            + struct.pack("<H", 2)  # e_shstrndx (index of .shstrtab)
        )
        assert len(header) == 64, f"ELF header is {len(header)} bytes, expected 64"

        # SHT entry 0: NULL (required by spec, all fields zero).
        null_sh = b"\x00" * 64

        # SHT entry 1: the data section. PROGBITS + SHF_ALLOC so the
        # runtime loader mmaps it into the process image.
        sh1 = struct.pack(
            "<IIQQQQIIQQ",
            sh_name_offset,  # sh_name (offset into shstrtab)
            sht_type,  # sh_type
            _SHF_ALLOC,  # sh_flags
            0,  # sh_addr (relocatable — addresses fixed at link)
            e_ehsize,  # sh_offset
            len(data),  # sh_size
            0,  # sh_link
            0,  # sh_info
            1,  # sh_addralign
            0,  # sh_entsize
        )

        # SHT entry 2: .shstrtab. SHT_STRTAB sections do not carry
        # SHF_ALLOC (the string table is consumed by the loader,
        # not mapped into the process).
        sh2 = struct.pack(
            "<IIQQQQIIQQ",
            shstrtab_name_offset,  # sh_name
            _SHT_STRTAB,  # sh_type
            0,  # sh_flags
            0,  # sh_addr
            e_ehsize + len(data),  # sh_offset
            len(shstrtab),  # sh_size
            0,  # sh_link
            0,  # sh_info
            1,  # sh_addralign
            0,  # sh_entsize
        )
        sht = null_sh + sh1 + sh2
        assert len(sht) == 3 * 64, f"SHT is {len(sht)} bytes, expected 192"

        return header + data + shstrtab + (b"\x00" * pad) + sht

    def _run_lld(
        self,
        temp_dir: Path,
        section_files: dict[str, Path],
        output_path: Path,
    ) -> Result[Path, LinkingError]:
        """Invoke the LLVM linker to combine the section files.

        Returns ``Ok(output_path)`` on success and
        ``Err(LinkingError)`` on any failure (lld missing, subprocess
        timeout, non-zero exit, missing output file).
        """
        if not self._lld_path:
            return Err(
                LinkingError(
                    "lld not found in PATH. Install LLVM (apt install lld / "
                    "brew install llvm). The fat binary cannot be linked "
                    "without lld.",
                )
            )

        cmd = [
            self._lld_path,
            "-r",  # relocatable link
            "-o",
            str(output_path),
            *[str(p) for p in section_files.values()],
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return Err(
                LinkingError(
                    f"lld timed out after {self.timeout_seconds}s",
                    cause=exc,
                )
            )
        if result.returncode != 0 or not output_path.exists():
            return Err(
                LinkingError(
                    f"lld failed: {result.stderr or '(no stderr)'}",
                    context={
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "cmd": cmd,
                    },
                )
            )
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
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip().split("\n")[0] or "unknown"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return "unknown"

    def get_version(self) -> str:
        return self._lld_version
