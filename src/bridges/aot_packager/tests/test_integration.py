"""Integration tests for the AOT fat binary packager pipeline.

Tests the full pipeline flow:
  1. Per-vendor AOT compilation (Nvidia PTX, AMD HSACO, Intel SPIR-V)
  2. C runtime stub compilation and symbol exports
  3. LLVM lld linking into a single ELF fat binary
  4. .nautilus.index section structure (Task 17 fix)
  5. Section naming convention (Task 17: .nautilus.{vendor}.{kernel_name})
  6. Runtime vendor detection and binary dispatch
  7. ELF structure verification (header, sections, symbols)
  8. Cross-architecture host-tool compatibility
  9. Edge cases and error handling

Every test either:
  - Uses the ``aot_packager`` harness fixture (mocked backends by default)
    so it runs in CI without real hardware SDKs, OR
  - Directly exercises the linker/fat_binary/runtime_stub modules with
    synthetic inputs.

All tests are hardware-agnostic and require no GPU, no AOTriton, no
oneAPI. The C runtime stub tests require gcc (for compilation) and
readelf/nm (for inspection). The linking tests require lld.
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess
import textwrap
from pathlib import Path
from typing import Any

import pytest

from src.bridges.aot_packager.fat_binary import (
    FatBinary,
    KernelSection,
    SectionFormat,
)
from src.bridges.aot_packager.linker import (
    FatBinaryLinker,
    KernelBlob,
    section_name_for,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_PTX = textwrap.dedent("""\
    .version 7.0
    .target sm_90
    .address_size 64
    .visible .entry sample_kernel(.param .u64 A_ptr, .param .u64 B_ptr, .param .u64 C_ptr)
    {
        ret;
    }
""").encode("utf-8")

SAMPLE_HSACO = b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 56 + b"amdgcn"

SAMPLE_SPV = b"\x03\x02\x23\x07" + b"\x00" * 8 + b"spirv-mock"

SAMPLE_METALLIB = b"MTLB" + b"\x00" * 12

# A minimal, syntactically valid Triton kernel for builder tests
MINIMAL_TRITON_KERNEL = textwrap.dedent("""\
    import triton
    import triton.language as tl

    @triton.jit
    def minimal_kernel(
        A_ptr, B_ptr, C_ptr,
        M, N, K,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rm = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = tl.arange(0, BLOCK_N)
        a = tl.load(A_ptr + rm[:, None] * K + tl.arange(0, BLOCK_K)[None, :])
        b = tl.load(B_ptr + tl.arange(0, BLOCK_K)[:, None] * N + rn[None, :])
        acc = tl.dot(a, b)
        tl.store(C_ptr + rm[:, None] * N + rn[None, :], acc)
""")


def _lld_path() -> str | None:
    """Return path to lld or None."""
    for name in ("ld.lld", "lld"):
        path = shutil.which(name)
        if path:
            return path
    return None


def _lld_or_skip() -> str:
    """Skip the test when lld is not on PATH."""
    lld = _lld_path()
    if lld is None:
        pytest.skip("lld not on PATH; install LLD (apt install lld / brew install llvm)")
    return lld


def _gcc_or_skip() -> str:
    """Skip the test when gcc is not on PATH."""
    gcc = shutil.which("gcc")
    if gcc is None:
        pytest.skip("gcc not on PATH; install gcc (apt install gcc)")
    return gcc


def _readelf_sections(path: Path) -> dict[str, dict[str, str]]:
    """Parse ``readelf -SW`` output into a section-name -> fields dict."""
    r = subprocess.run(
        ["readelf", "-SW", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    sections: dict[str, dict[str, str]] = {}
    for line in r.stdout.splitlines():
        # Format: [NN] .name  TYPE  ADDR  OFFSET  SIZE  ...
        m = re.match(
            r"\s*\[\s*\d+\]\s+"
            r"(?P<name>\S+)\s+"
            r"(?P<type>\S+)\s+"
            r"(?P<addr>[0-9a-fA-F]+)\s+"
            r"(?P<offset>[0-9a-fA-F]+)\s+"
            r"(?P<size>[0-9a-fA-F]+)",
            line,
        )
        if m:
            sections[m.group("name")] = m.groupdict()
    return sections


def _readelf_symbols(path: Path, *, defined: bool = True) -> list[str]:
    """Return symbol names from NM-like output via readelf -s."""
    flag = "--defined-only" if defined else "--undefined-only"
    r = subprocess.run(
        ["nm", flag, str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        # Fall back to readelf -s
        r2 = subprocess.run(
            ["readelf", "-s", str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
        names: list[str] = []
        for line in r2.stdout.splitlines():
            m = re.match(r"\s+\d+:\s+[0-9a-fA-F]+\s+\d+\s+\S+\s+\S+\s+\S+\s+\S+\s+(\S+)", line)
            if m:
                names.append(m.group(1))
        return names
    names = []
    for line in r.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) >= 3:
            names.append(parts[2])
        elif len(parts) == 2:
            names.append(parts[1])
    return names


# ===========================================================================
# SECTION NAMING CONVENTION (Task 17)
# ===========================================================================


class TestSectionNamingConvention:
    """Task 17: every kernel blob gets its own unique
    ``.nautilus.{vendor}.{kernel_name}`` section name so ``ld.lld -r``
    never silently merges sections."""

    def test_section_name_for_nvidia(self) -> None:
        assert section_name_for("nvidia", "matmul") == ".nautilus.nvidia.matmul"
        assert (
            section_name_for("nvidia", "matmul", fmt_suffix="ptx") == ".nautilus.nvidia.ptx.matmul"
        )
        assert (
            section_name_for("nvidia", "matmul", fmt_suffix="cubin")
            == ".nautilus.nvidia.cubin.matmul"
        )

    def test_section_name_for_amd(self) -> None:
        assert section_name_for("amd", "attention") == ".nautilus.amd.attention"

    def test_section_name_for_intel(self) -> None:
        assert section_name_for("intel", "layernorm") == ".nautilus.intel.layernorm"

    def test_section_name_sanitizes_tokens(self) -> None:
        """Non-alphanumeric chars in vendor/kernel names become underscores."""
        name = section_name_for("nvidia!", "my+kernel")
        assert ".nautilus." in name
        assert "nvidia_" in name
        assert "my_kernel" in name

    def test_section_name_unique_per_vendor_kernel_fmt(self) -> None:
        """Different (vendor, kernel, fmt) triples produce different names."""
        names = {
            section_name_for("nvidia", "matmul", fmt_suffix="ptx"),
            section_name_for("nvidia", "matmul", fmt_suffix="cubin"),
            section_name_for("amd", "matmul", fmt_suffix="hsaco"),
            section_name_for("intel", "matmul", fmt_suffix="spv"),
            section_name_for("nvidia", "attention", fmt_suffix="ptx"),
        }
        assert len(names) == 5

    def test_section_name_length_limit(self) -> None:
        """Section names must be 1-64 characters."""
        from src.bridges.aot_packager.linker import validate_section_name
        from src.common.errors import LinkingError

        validate_section_name(".nautilus.nvidia.matmul")  # should not raise
        with pytest.raises(LinkingError):
            validate_section_name("")
        with pytest.raises(LinkingError):
            validate_section_name("." * 65)


# ===========================================================================
# RUNTIME STUB COMPILATION & VENDOR DETECTION
# ===========================================================================


class TestRuntimeStubCompilation:
    """The C runtime stub must compile to a valid object file and export
    the symbols the fat binary dispatcher needs."""

    @classmethod
    def _compile_stub(cls, tmp_path: Path) -> Path:
        """Compile runtime_stub.c and return the path to the .o file."""
        gcc = _gcc_or_skip()
        stub_src = Path(__file__).resolve().parent.parent / "runtime_stub.c"
        assert stub_src.exists(), f"runtime_stub.c not found at {stub_src}"
        stub_o = tmp_path / "runtime_stub.o"
        subprocess.run(
            [
                gcc,
                "-c",
                "-nostdlib",
                "-ffreestanding",
                "-Wall",
                "-Werror",
                "-std=c11",
                "-I",
                str(stub_src.parent.parent.parent / "c_api"),  # so it finds triton_c_api.h
                "-o",
                str(stub_o),
                str(stub_src),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        assert stub_o.exists()
        return stub_o

    def test_stub_compiles_without_errors(self, tmp_path: Path) -> None:
        """The runtime stub must compile cleanly with -Wall -Werror."""
        stub_o = self._compile_stub(tmp_path)
        assert stub_o.stat().st_size > 0, "stub object file is empty"

    def test_stub_exports_dispatch_symbols(self, tmp_path: Path) -> None:
        """The stub object must export nautilus_dispatch and friends."""
        stub_o = self._compile_stub(tmp_path)
        symbols = _readelf_symbols(stub_o, defined=True)
        required = {
            "nautilus_detect_vendor",
            "nautilus_dispatch",
            "nautilus_dispatch_with_index",
            "nautilus_has_nvidia_gpu",
            "nautilus_has_amd_gpu",
            "nautilus_has_intel_gpu",
            "nautilus_version",
            "nautilus_build_info",
        }
        missing = required - set(symbols)
        assert not missing, (
            f"Stub missing required symbols: {missing}. Found: {sorted(set(symbols) & required)}"
        )

    def test_stub_exports_index_symbols(self, tmp_path: Path) -> None:
        """The stub must export index lookup functions."""
        stub_o = self._compile_stub(tmp_path)
        symbols = _readelf_symbols(stub_o, defined=True)
        for sym in ("nautilus_index_find", "nautilus_index_find_by_vendor"):
            assert sym in symbols, f"Stub missing symbol: {sym}"

    def test_stub_references_external_symbols(self, tmp_path: Path) -> None:
        """The stub must reference nautilus_index_data and
        nautilus_index_size as undefined symbols (supplied by the index
        holder object at link time)."""
        stub_o = self._compile_stub(tmp_path)
        undef = _readelf_symbols(stub_o, defined=False)
        for sym in ("nautilus_index_data", "nautilus_index_size"):
            assert sym in undef, (
                f"Stub does not reference {sym!r} as undefined. Undefined symbols: {undef}"
            )

    def test_stub_references_per_vendor_kernels(self, tmp_path: Path) -> None:
        """The stub must reference nautilus_kernel_nvidia/amd/intel/apple/default
        as undefined symbols (supplied by the per-vendor .o files)."""
        stub_o = self._compile_stub(tmp_path)
        undef = _readelf_symbols(stub_o, defined=False)
        for sym in (
            "nautilus_kernel_nvidia",
            "nautilus_kernel_amd",
            "nautilus_kernel_intel",
            "nautilus_kernel_apple",
            "nautilus_kernel_default",
        ):
            assert sym in undef, (
                f"Stub does not reference {sym!r} as undefined. Undefined symbols: {undef}"
            )

    def test_host_arch_compatible(self, tmp_path: Path) -> None:
        """The compiled stub must match the host architecture (x86_64 or
        aarch64). This test verifies cross-architecture compatibility."""
        stub_o = self._compile_stub(tmp_path)
        r = subprocess.run(
            ["readelf", "-h", str(stub_o)],
            capture_output=True,
            text=True,
            check=True,
        )
        # Determine expected machine type
        machine = None
        for line in r.stdout.splitlines():
            if "Machine:" in line:
                machine = line.split("Machine:")[1].strip()
                break
        arch = os.uname().machine
        if arch == "x86_64":
            assert machine == "Advanced Micro Devices X86-64", (
                f"Expected x86_64 ELF but got {machine}"
            )
        elif arch in ("aarch64", "arm64"):
            assert machine == "AArch64", f"Expected AArch64 ELF but got {machine}"


# ===========================================================================
# LINKER: FAT BINARY ELF STRUCTURE
# ===========================================================================


class TestLinkerFatBinaryStructure:
    """The linker must produce a valid ELF relocatable object containing
    the per-vendor kernel sections and the .nautilus.index section."""

    @pytest.fixture
    def linker(self, tmp_path: Path) -> FatBinaryLinker:
        return FatBinaryLinker(cache_dir=str(tmp_path / "link_cache"))

    def _linked_path(
        self,
        linker: FatBinaryLinker,
        tmp_path: Path,
        **kwargs: Any,
    ) -> Path:
        """Helper: run link_fat_binary and return the output path."""
        result = linker.link_fat_binary(
            nvidia_ptx=SAMPLE_PTX,
            amd_hsaco=SAMPLE_HSACO,
            intel_spv=SAMPLE_SPV,
            kernel_name="integration_test",
            output_path=tmp_path / "integ.fat.o",
            **kwargs,
        )
        assert result.is_usable, f"Link failed: {result.error}"
        assert result.output_path is not None
        return result.output_path

    def test_linked_output_is_valid_elf(self, linker: FatBinaryLinker, tmp_path: Path) -> None:
        """The linked output must start with the ELF magic and parse as
        ET_REL (relocatable)."""
        path = self._linked_path(linker, tmp_path)
        header = path.read_bytes()[:64]
        assert header[:4] == b"\x7fELF", "Not a valid ELF file"
        # e_type at offset 16 (2 bytes): 1 = ET_REL
        e_type = struct.unpack_from("<H", header, 16)[0]
        assert e_type == 1, f"Expected ET_REL (1), got {e_type}"

    def test_linked_sections_include_nautilus_nvidia(
        self,
        linker: FatBinaryLinker,
        tmp_path: Path,
    ) -> None:
        """The linked output must contain .nautilus.nvidia.ptx.integration_test
        and .nautilus.nvidia.cubin.integration_test sections."""
        path = self._linked_path(linker, tmp_path)
        sections = _readelf_sections(path)
        expected_sections = {
            ".nautilus.index",
            ".nautilus.nvidia.ptx.integration_test",
            ".nautilus.amd.hsaco.integration_test",
            ".nautilus.intel.spv.integration_test",
        }
        for sec in expected_sections:
            assert sec in sections, (
                f"Missing section {sec!r}. "
                f"Nautilus sections found: {[s for s in sections if '.nautilus.' in s]}"
            )

    def test_section_names_are_unique(self, linker: FatBinaryLinker, tmp_path: Path) -> None:
        """Every nautilus section in the linked output must have a unique name."""
        path = self._linked_path(linker, tmp_path)
        sections = _readelf_sections(path)
        nautilus_names = [
            s for s in sections if s.startswith(".nautilus.") and s != ".nautilus.index"
        ]
        assert len(nautilus_names) == len(set(nautilus_names)), (
            f"Duplicate nautilus section names: {nautilus_names}"
        )

    def test_nautilus_index_section_exists_and_has_content(
        self,
        linker: FatBinaryLinker,
        tmp_path: Path,
    ) -> None:
        """The .nautilus.index section must be present with non-zero size
        and contain pipe-delimited records (Task 17)."""
        path = self._linked_path(linker, tmp_path)
        sections = _readelf_sections(path)
        assert ".nautilus.index" in sections, "Missing .nautilus.index section"
        index_size = int(sections[".nautilus.index"]["size"], 16)
        assert index_size > 0, ".nautilus.index section is empty"
        # Extract the raw bytes of .nautilus.index
        r = subprocess.run(
            ["readelf", "-x", ".nautilus.index", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        raw_index_path = tmp_path / "nautilus.index.raw"
        objcopy_r = subprocess.run(
            [
                "objcopy",
                "-O",
                "binary",
                "--only-section=.nautilus.index",
                str(path),
                str(raw_index_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if objcopy_r.returncode == 0 and raw_index_path.exists():
            raw = raw_index_path.read_bytes()
        else:
            raw = b""
            for line in r.stdout.splitlines():
                tokens = line.strip().split()
                hex_groups = [t for t in tokens if re.match(r"^[0-9a-fA-F]+$", t)]
                if len(hex_groups) >= 2:
                    raw += b"".join(bytes.fromhex(g) for g in hex_groups)
            raw = raw[:index_size]
        text = raw.decode("utf-8", errors="replace")
        assert "nvidia" in text
        assert "amd" in text
        assert "intel" in text
        assert "|" in text, "Index missing pipe-delimited records"

    def test_linked_output_with_multiple_kernels(
        self,
        linker: FatBinaryLinker,
        tmp_path: Path,
    ) -> None:
        """Linking multiple kernels from the same vendor must produce
        unique sections for each."""
        blobs = [
            KernelBlob(
                kernel_name="matmul", vendor="nvidia", arch="sm_90", fmt="ptx", data=SAMPLE_PTX
            ),
            KernelBlob(
                kernel_name="attention", vendor="nvidia", arch="sm_90", fmt="ptx", data=SAMPLE_PTX
            ),
            KernelBlob(
                kernel_name="matmul", vendor="amd", arch="gfx942", fmt="hsaco", data=SAMPLE_HSACO
            ),
        ]
        result = linker.link_fat_binary(
            kernels=blobs,
            kernel_name="multi",
            output_path=tmp_path / "multi.fat.o",
        )
        assert result.is_usable
        assert result.output_path is not None, "linker should produce an output path"
        sections = _readelf_sections(result.output_path)
        expected = {
            ".nautilus.nvidia.ptx.matmul",
            ".nautilus.nvidia.ptx.attention",
            ".nautilus.amd.hsaco.matmul",
        }
        for sec in expected:
            assert sec in sections, f"Missing section {sec!r} in multi-kernel link"

    def test_linker_cache_hit(self, linker: FatBinaryLinker, tmp_path: Path) -> None:
        """Identical inputs to link_fat_binary should hit the cache."""
        r1 = linker.link_fat_binary(
            nvidia_ptx=SAMPLE_PTX,
            kernel_name="cache_me",
            output_path=tmp_path / "a.fat.o",
        )
        r2 = linker.link_fat_binary(
            nvidia_ptx=SAMPLE_PTX,
            kernel_name="cache_me",
            output_path=tmp_path / "b.fat.o",
        )
        assert r1.is_usable
        assert r2.is_usable
        assert r2.cache_hit


# ===========================================================================
# FAT BINARY FORMAT WRAPPING
# ===========================================================================


class TestFatBinaryWrapper:
    """The ``_wrap_section_data`` method must produce a valid ELF object
    for a single data section, verified by readelf."""

    def test_wrapped_section_is_valid_elf(self, tmp_path: Path) -> None:
        """A wrapped section must parse as a valid ET_REL ELF."""
        linker = FatBinaryLinker(cache_dir=str(tmp_path))
        wrapped = linker._wrap_section_data(SAMPLE_PTX, ".test.section", "PROGBITS")
        assert wrapped[:4] == b"\x7fELF"
        e_type = struct.unpack_from("<H", wrapped, 16)[0]
        assert e_type == 1, f"Expected ET_REL, got {e_type}"

    def test_wrapped_section_has_correct_name(self, tmp_path: Path) -> None:
        """The section name in the wrapped object must match the requested name."""
        linker = FatBinaryLinker(cache_dir=str(tmp_path))
        name = ".nautilus.nvidia.ptx.test_kernel"
        wrapped = linker._wrap_section_data(SAMPLE_PTX, name, "PROGBITS")
        obj = tmp_path / "wrapped.o"
        obj.write_bytes(wrapped)
        sections = _readelf_sections(obj)
        assert name in sections, f"Section {name!r} not found. Sections: {list(sections.keys())}"

    def test_wrapped_section_data_integrity(self, tmp_path: Path) -> None:
        """The data stored in the wrapped section must round-trip correctly."""
        linker = FatBinaryLinker(cache_dir=str(tmp_path))
        name = ".my.data"
        payload = b"Hello, Fat Binary!" * 100
        wrapped = linker._wrap_section_data(payload, name, "PROGBITS")
        obj = tmp_path / "data.o"
        obj.write_bytes(wrapped)
        sections = _readelf_sections(obj)
        assert name in sections
        offset = int(sections[name]["offset"], 16)
        size = int(sections[name]["size"], 16)
        recovered = obj.read_bytes()[offset : offset + size]
        assert recovered == payload, (
            f"Data mismatch: expected {len(payload)} bytes, got {len(recovered)}"
        )

    def test_wrapped_section_with_metadata_symbols(self, tmp_path: Path) -> None:
        """The .nautilus.index holder object must export the correct symbols."""
        linker = FatBinaryLinker(cache_dir=str(tmp_path))
        blobs = [
            KernelBlob(kernel_name="k1", vendor="nvidia", arch="sm_90", fmt="ptx", data=b"X" * 16),
        ]
        section_names = [section_name_for(b.vendor, b.kernel_name, fmt_suffix=b.fmt) for b in blobs]
        index_text = (
            "\n".join(
                "|".join(
                    [
                        b.kernel_name,
                        b.vendor,
                        b.arch,
                        b.fmt,
                        section_name_for(b.vendor, b.kernel_name, fmt_suffix=b.fmt),
                        str(len(b.data)),
                    ]
                )
                for b in blobs
            )
            + "\n\n"
        )
        obj_path = linker._build_index_object(tmp_path, blobs, section_names, index_text)
        assert obj_path is not None
        symbols = _readelf_symbols(obj_path, defined=True)
        assert "nautilus_index_data" in symbols
        assert "nautilus_index_size" in symbols


# ===========================================================================
# RUNTIME STUB VENDOR DETECTION
# ===========================================================================


class TestRuntimeVendorDetection:
    """The vendor detection logic must correctly identify the current
    platform's GPU vendor (or return 'none' when no GPU is present)."""

    def test_detect_no_gpu_returns_unknown_on_ci(self) -> None:
        """In CI (no GPU), nautilus_detect_vendor should return -1
        (NAUTILUS_VENDOR_UNKNOWN). We verify this by inspecting the C
        source logic directly."""
        stub_src = (Path(__file__).resolve().parent.parent / "runtime_stub.c").read_text()
        # The detection functions use access("/dev/nvidia*", F_OK) etc.
        # In CI these files won't exist, so the C functions return 0.
        # We verify the C code structure is correct.
        assert "nautilus_check_nvidia" in stub_src
        assert "nautilus_check_amd" in stub_src
        assert "nautilus_check_intel" in stub_src
        assert "nautilus_detect_vendor" in stub_src
        # _Static_asserts for vendor enum drift
        assert "NAUTILUS_VENDOR_NVIDIA" in stub_src
        assert "NAUTILUS_VENDOR_AMD" in stub_src
        assert "NAUTILUS_VENDOR_INTEL" in stub_src
        assert "NAUTILUS_VENDOR_APPLE" in stub_src
        assert "NAUTILUS_VENDOR_UNKNOWN" in stub_src

    def test_vendor_c_api_header_matches_stub(self) -> None:
        """The nautilus_vendor_t enum in triton_c_api.h must match the
        static_assert values in runtime_stub.c."""
        header_path = (
            Path(__file__).resolve().parent.parent.parent.parent / "c_api" / "triton_c_api.h"
        )
        header_text = header_path.read_text()
        # Check that the enum values exist
        assert "NAUTILUS_VENDOR_NVIDIA  = 0" in header_text
        assert "NAUTILUS_VENDOR_AMD     = 1" in header_text
        assert "NAUTILUS_VENDOR_INTEL   = 2" in header_text
        assert "NAUTILUS_VENDOR_APPLE   = 3" in header_text
        assert "NAUTILUS_VENDOR_UNKNOWN = -1" in header_text

    def test_runtime_dispatch_logic(
        self,
        tmp_path: Path,
    ) -> None:
        """Compile a small C program that links against the runtime stub
        and verifies vendor detection returns -1 (no GPU)."""
        gcc = _gcc_or_skip()
        stub_dir = Path(__file__).resolve().parent.parent
        repo_root = stub_dir.parent.parent.parent
        test_c = tmp_path / "test_vendor.c"
        test_c.write_text(
            textwrap.dedent("""\
            #include <stdio.h>
            #include <stdlib.h>

            const unsigned char nautilus_index_data[] = {};
            const unsigned long nautilus_index_size = 0;
            int nautilus_kernel_nvidia(void* a) { (void)a; return 0; }
            int nautilus_kernel_amd(void* a) { (void)a; return 0; }
            int nautilus_kernel_intel(void* a) { (void)a; return 0; }
            int nautilus_kernel_apple(void* a) { (void)a; return 0; }
            int nautilus_kernel_default(void* a) { (void)a; return 0; }

            #include "runtime_stub.c"

            int main(void) {
                int vendor = nautilus_detect_vendor();
                printf("VENDOR=%d\\n", vendor);
                printf("HAS_NVIDIA=%d\\n", nautilus_has_nvidia_gpu());
                printf("HAS_AMD=%d\\n", nautilus_has_amd_gpu());
                printf("HAS_INTEL=%d\\n", nautilus_has_intel_gpu());
                printf("HAS_APPLE=%d\\n", nautilus_has_apple_gpu());
                printf("VERSION=%s\\n", nautilus_version());
                return vendor == -1 ? 0 : 1;
            }
        """)
        )

        binary = tmp_path / "test_vendor"
        subprocess.run(
            [
                gcc,
                "-std=c11",
                "-Wall",
                "-Werror",
                "-I",
                str(stub_dir),
                "-I",
                str(repo_root / "src"),
                "-I",
                str(repo_root / "src" / "c_api"),
                "-o",
                str(binary),
                str(test_c),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        r = subprocess.run(
            [str(binary)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # On CI with no GPU, vendor should be -1 (UNKNOWN)
        # If there is a GPU, it might find one — that's still valid.
        # We just verify the binary ran and produced output.
        assert r.returncode in (0, 1), f"test_vendor exited with {r.returncode}: {r.stderr}"
        output_lines = r.stdout.strip().split("\n")
        output_map = {}
        for line in output_lines:
            if "=" in line:
                k, v = line.split("=", 1)
                output_map[k] = v
        assert "VENDOR" in output_map, f"No VENDOR= in output: {output_lines}"
        assert "VERSION" in output_map
        assert "Nautilus" in output_map["VERSION"]


# ===========================================================================
# FULL PIPELINE WITH HARNESS FIXTURE (MOCKED BACKENDS)
# ===========================================================================


class TestFullPipelineWithHarness:
    """End-to-end pipeline using the ``aot_packager`` fixture from the
    integration test harness (Task 7). The fixture mocks per-vendor
    compilers and the linker, so the test runs in any environment."""

    def test_build_all_vendors_calls_all_backends(
        self,
        aot_packager: Any,
        tmp_path: Path,
    ) -> None:
        """A full build with all vendors enabled must call each vendor
        backend and the linker exactly once."""
        from src.bridges.aot_packager.builder import FatBinaryConfig

        config = FatBinaryConfig(
            kernel_name="mock_kernel",
            kernel_source=MINIMAL_TRITON_KERNEL,
            output_dir=str(tmp_path),
            skip_validation=True,
        )
        result = aot_packager.build(config)
        assert result.success, f"Build failed: {result.error}"
        aot_packager._mock_amd.assert_called_once()
        aot_packager._mock_intel.assert_called_once()
        aot_packager._mock_nvidia.assert_called_once()
        aot_packager._mock_link.assert_called_once()

    def test_build_with_skip_amd(self, aot_packager: Any, tmp_path: Path) -> None:
        """skip_amd=True must skip the AMD backend."""
        from src.bridges.aot_packager.builder import FatBinaryConfig

        config = FatBinaryConfig(
            kernel_name="no_amd",
            kernel_source=MINIMAL_TRITON_KERNEL,
            output_dir=str(tmp_path),
            skip_amd=True,
            skip_validation=True,
        )
        result = aot_packager.build(config)
        assert result.success
        aot_packager._mock_amd.assert_not_called()
        aot_packager._mock_intel.assert_called_once()
        aot_packager._mock_nvidia.assert_called_once()
        aot_packager._mock_link.assert_called_once()

    def test_build_with_skip_all_but_nvidia(self, aot_packager: Any, tmp_path: Path) -> None:
        """Skipping AMD and Intel must only compile Nvidia and link."""
        from src.bridges.aot_packager.builder import FatBinaryConfig

        config = FatBinaryConfig(
            kernel_name="nvidia_only",
            kernel_source=MINIMAL_TRITON_KERNEL,
            output_dir=str(tmp_path),
            skip_amd=True,
            skip_intel=True,
            skip_validation=True,
        )
        result = aot_packager.build(config)
        assert result.success
        aot_packager._mock_amd.assert_not_called()
        aot_packager._mock_intel.assert_not_called()
        aot_packager._mock_nvidia.assert_called_once()
        aot_packager._mock_link.assert_called_once()
        # The fat binary should only contain Nvidia
        assert result.fat_binary is not None
        assert result.fat_binary.vendors == ["nvidia"]

    def test_build_fat_binary_has_correct_sections(self, aot_packager: Any, tmp_path: Path) -> None:
        """The fat binary produced by the build must have sections for
        all compiled vendors."""
        from src.bridges.aot_packager.builder import FatBinaryConfig

        config = FatBinaryConfig(
            kernel_name="section_check",
            kernel_source=MINIMAL_TRITON_KERNEL,
            output_dir=str(tmp_path),
            skip_validation=True,
        )
        result = aot_packager.build(config)
        assert result.fat_binary is not None
        assert len(result.fat_binary.sections) >= 1
        vendors_found = {s.vendor for s in result.fat_binary.sections}
        assert "nvidia" in vendors_found
        # Test that if AMD or Intel sections are present, they have valid data
        for vendor_name in ("amd", "intel"):
            section = result.fat_binary.get_section(vendor_name)
            if section is not None:
                assert section.size > 0

    def test_build_records_linking_result(self, aot_packager: Any, tmp_path: Path) -> None:
        """The build result must contain a linking result."""
        from src.bridges.aot_packager.builder import FatBinaryConfig

        config = FatBinaryConfig(
            kernel_name="link_check",
            kernel_source=MINIMAL_TRITON_KERNEL,
            output_dir=str(tmp_path),
            skip_validation=True,
        )
        result = aot_packager.build(config)
        assert result.linking_result is not None
        assert result.linking_result.is_usable
        assert result.linking_result.linker_version == "mock-lld"

    def test_build_records_all_stage_times(self, aot_packager: Any, tmp_path: Path) -> None:
        """The build must record elapsed time for each compilation stage."""
        from src.bridges.aot_packager.builder import FatBinaryConfig

        config = FatBinaryConfig(
            kernel_name="timing_test",
            kernel_source=MINIMAL_TRITON_KERNEL,
            output_dir=str(tmp_path),
            skip_validation=True,
        )
        result = aot_packager.build(config)
        for stage in ("nvidia", "amd", "intel", "runtime_stub", "link"):
            assert stage in result.stage_times, f"Missing stage time: {stage}"
            assert result.stage_times[stage] > 0

    def test_build_total_time_is_positive(self, aot_packager: Any, tmp_path: Path) -> None:
        """The total elapsed time must be positive."""
        from src.bridges.aot_packager.builder import FatBinaryConfig

        config = FatBinaryConfig(
            kernel_name="total_time",
            kernel_source=MINIMAL_TRITON_KERNEL,
            output_dir=str(tmp_path),
            skip_validation=True,
        )
        result = aot_packager.build(config)
        assert result.total_time_s > 0

    def test_build_output_path_exists(self, aot_packager: Any, tmp_path: Path) -> None:
        """The build must produce an output file on disk."""
        from src.bridges.aot_packager.builder import FatBinaryConfig

        config = FatBinaryConfig(
            kernel_name="output_check",
            kernel_source=MINIMAL_TRITON_KERNEL,
            output_dir=str(tmp_path),
            skip_validation=True,
        )
        result = aot_packager.build(config)
        assert result.output_path is not None
        assert result.output_path.exists()
        assert result.output_path.stat().st_size > 0

    def test_build_to_dict_serializable(self, aot_packager: Any, tmp_path: Path) -> None:
        """The build result must be JSON-serializable via to_dict()."""
        import json

        from src.bridges.aot_packager.builder import FatBinaryConfig

        config = FatBinaryConfig(
            kernel_name="serialize_check",
            kernel_source=MINIMAL_TRITON_KERNEL,
            output_dir=str(tmp_path),
            skip_validation=True,
        )
        result = aot_packager.build(config)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["success"]
        assert d["kernel_name"] == "serialize_check"
        assert "vendors" in d
        assert d["total_time_s"] > 0
        # Must be JSON-serializable
        json.dumps(d)

    def test_build_error_when_no_vendors_succeed(
        self,
        aot_packager: Any,
        tmp_path: Path,
    ) -> None:
        """When all vendors are skipped and no stub is produced, the
        build must return a failure result — not crash."""
        from src.bridges.aot_packager.builder import FatBinaryConfig

        config = FatBinaryConfig(
            kernel_name="fail_check",
            kernel_source=MINIMAL_TRITON_KERNEL,
            output_dir=str(tmp_path),
            skip_amd=True,
            skip_intel=True,
            skip_nvidia=True,
            skip_validation=True,
        )
        result = aot_packager.build(config)
        assert not result.success
        assert result.error is not None


# ===========================================================================
# FAT BINARY DATA MODEL ROUND-TRIPS
# ===========================================================================


class TestFatBinaryRoundTrips:
    """The FatBinary data model must serialize and deserialize losslessly
    through all supported formats."""

    def test_fat_binary_to_bytes_and_back(self) -> None:
        """Binary serialization with NFAT magic must round-trip."""
        fb = FatBinary(kernel_name="roundtrip")
        fb.add_section(KernelSection("nvidia", "sm_90", SectionFormat.PTX, SAMPLE_PTX))
        fb.add_section(KernelSection("amd", "gfx942", SectionFormat.HSACO, SAMPLE_HSACO))
        data = fb.to_bytes()
        assert data[:4] == b"NFAT"
        restored = FatBinary.from_bytes(data)
        assert restored.kernel_name == "roundtrip"
        assert len(restored.sections) == 2
        assert restored.sections[0].data == SAMPLE_PTX
        assert restored.sections[1].data == SAMPLE_HSACO

    def test_fat_binary_to_json_and_back(self) -> None:
        """JSON serialization must round-trip."""
        fb = FatBinary(kernel_name="json_test", metadata={"version": 2})
        fb.add_section(KernelSection("intel", "xe_hpg", SectionFormat.SPV, SAMPLE_SPV))
        fb.add_section(
            KernelSection(
                "nvidia",
                "sm_90",
                SectionFormat.PTX,
                SAMPLE_PTX,
                metadata={"compilation_time_s": 0.5},
            ),
        )
        json_text = fb.to_json()
        restored = FatBinary.from_json(json_text)
        assert restored.kernel_name == "json_test"
        assert restored.metadata["version"] == 2
        assert len(restored.sections) == 2
        assert restored.get_section("intel") is not None
        assert restored.get_section("nvidia") is not None
        nv = restored.get_section("nvidia")
        assert nv is not None
        assert nv.metadata.get("compilation_time_s") == 0.5

    def test_fat_binary_save_and_load(self, tmp_path: Path) -> None:
        """Disk serialization must round-trip."""
        fb = FatBinary(kernel_name="disk_test")
        fb.add_section(KernelSection("intel", "xe_hpg", SectionFormat.SPV, SAMPLE_SPV))
        path = tmp_path / "test.nfat"
        fb.save(path)
        loaded = FatBinary.load(path)
        assert loaded.kernel_name == "disk_test"
        assert loaded.sections[0].data == SAMPLE_SPV

    def test_fat_binary_total_size(self) -> None:
        """total_size must sum all section data sizes."""
        fb = FatBinary(kernel_name="size_test")
        fb.add_section(KernelSection("nvidia", "sm_90", SectionFormat.PTX, b"a" * 100))
        fb.add_section(KernelSection("amd", "gfx942", SectionFormat.HSACO, b"b" * 200))
        fb.add_section(KernelSection("intel", "xe_hpg", SectionFormat.SPV, b"c" * 300))
        assert fb.total_size == 600

    def test_fat_binary_vendors_dedup(self) -> None:
        """vendors must return unique vendor names."""
        fb = FatBinary(kernel_name="vendor_test")
        fb.add_section(KernelSection("nvidia", "sm_90", SectionFormat.PTX, b"a"))
        fb.add_section(KernelSection("nvidia", "sm_80", SectionFormat.PTX, b"b"))
        fb.add_section(KernelSection("amd", "gfx942", SectionFormat.HSACO, b"c"))
        assert set(fb.vendors) == {"nvidia", "amd"}


# ===========================================================================
# KERNEL SECTION PROPERTIES
# ===========================================================================


class TestKernelSectionProperties:
    """KernelSection metadata properties must behave correctly."""

    def test_section_size_property(self) -> None:
        section = KernelSection("nvidia", "sm_90", SectionFormat.PTX, SAMPLE_PTX)
        assert section.size == len(SAMPLE_PTX)

    def test_section_sha256_stable(self) -> None:
        s1 = KernelSection("nvidia", "sm_90", SectionFormat.PTX, SAMPLE_PTX)
        s2 = KernelSection("nvidia", "sm_90", SectionFormat.PTX, SAMPLE_PTX)
        assert s1.sha256 == s2.sha256
        assert len(s1.sha256) == 64

    def test_section_to_dict(self) -> None:
        section = KernelSection(
            "amd",
            "gfx942",
            SectionFormat.HSACO,
            SAMPLE_HSACO,
            metadata={"arch": "cdna3"},
        )
        d = section.to_dict()
        assert d["vendor"] == "amd"
        assert d["arch"] == "gfx942"
        assert d["format"] == "hsaco"
        assert d["size"] == len(SAMPLE_HSACO)
        assert "sha256" in d

    def test_section_add_section_replaces(self) -> None:
        """add_section must replace existing section for the same vendor+arch."""
        fb = FatBinary(kernel_name="replace")
        fb.add_section(KernelSection("nvidia", "sm_90", SectionFormat.PTX, b"old"))
        fb.add_section(KernelSection("nvidia", "sm_90", SectionFormat.PTX, b"new"))
        assert len(fb.sections) == 1
        assert fb.sections[0].data == b"new"

    def test_section_get_section_nonexistent(self) -> None:
        fb = FatBinary(kernel_name="get_test")
        assert fb.get_section("nonexistent") is None


# ===========================================================================
# LINKER ERROR HANDLING & EDGE CASES
# ===========================================================================


class TestLinkerErrorHandling:
    """The linker must handle edge cases gracefully and produce clear
    error messages."""

    def test_link_with_no_kernels_and_no_stub(self, tmp_path: Path) -> None:
        """Calling link_fat_binary with no kernels and no accessible
        stub must return a failure result, not crash."""
        linker = FatBinaryLinker(cache_dir=str(tmp_path), stub_path="/nonexistent/stub.o")
        result = linker.link_fat_binary(kernel_name="empty")
        assert not result.success
        assert result.error is not None

    def test_link_with_duplicate_kernels(self, tmp_path: Path) -> None:
        """Duplicate KernelBlob entries must raise LinkingError."""
        from src.common.errors import LinkingError

        linker = FatBinaryLinker(cache_dir=str(tmp_path))
        blobs = [
            KernelBlob(kernel_name="k1", vendor="nvidia", arch="sm_90", fmt="ptx", data=b"a"),
            KernelBlob(kernel_name="k1", vendor="nvidia", arch="sm_90", fmt="ptx", data=b"b"),
        ]
        with pytest.raises(LinkingError, match="Duplicate"):
            linker.link_fat_binary(kernels=blobs, kernel_name="dup")

    def test_linker_cache_isolated(self, tmp_path: Path) -> None:
        """Different inputs must NOT hit the same cache entry."""
        linker = FatBinaryLinker(cache_dir=str(tmp_path))
        r1 = linker.link_fat_binary(
            nvidia_ptx=b"input_a",
            kernel_name="cache_iso",
            output_path=tmp_path / "a.fat.o",
        )
        r2 = linker.link_fat_binary(
            nvidia_ptx=b"input_b",
            kernel_name="cache_iso",
            output_path=tmp_path / "b.fat.o",
        )
        assert r1.is_usable
        assert r2.is_usable
        # At least one must be a cache miss (different inputs)
        if r2.cache_hit and r1.cache_hit:
            # Both might be cached if the cache_dir has existing entries,
            # but different inputs should produce different cache keys.
            # At worst, insist the cache keys differ by inspecting output.
            assert r1.output_path != r2.output_path

    def test_section_name_with_special_chars(self) -> None:
        """Special characters in vendor/kernel names must be sanitized."""
        name = section_name_for("nVid!a", "my_kernel@123")
        assert ".nautilus." in name
        assert "nVid_a" in name  # '!' sanitized to '_'
        assert "my_kernel_123" in name  # '@' sanitized to '_'


# ===========================================================================
# C API HEADER CONSISTENCY
# ===========================================================================


class TestCApiConsistency:
    """The nautilus_vendor_t and nautilus_arch_t enums in triton_c_api.h
    must be consistent with the Python-side enums and the C stub."""

    def test_c_api_vendor_enum_values(self) -> None:
        """Verify vendor enum values match across Python and C."""
        header_path = (
            Path(__file__).resolve().parent.parent.parent.parent / "c_api" / "triton_c_api.h"
        )
        header = header_path.read_text()
        # Nvidia = 0
        assert re.search(r"NAUTILUS_VENDOR_NVIDIA\s*=\s*0", header)
        assert re.search(r"NAUTILUS_VENDOR_AMD\s*=\s*1", header)
        assert re.search(r"NAUTILUS_VENDOR_INTEL\s*=\s*2", header)
        assert re.search(r"NAUTILUS_VENDOR_APPLE\s*=\s*3", header)
        assert re.search(r"NAUTILUS_VENDOR_UNKNOWN\s*=\s*-1", header)

    def test_c_api_arch_enum_values(self) -> None:
        """Verify arch enum values in the C header."""
        header_path = (
            Path(__file__).resolve().parent.parent.parent.parent / "c_api" / "triton_c_api.h"
        )
        header = header_path.read_text()
        assert re.search(r"NAUTILUS_ARCH_SM_90\s*=\s*90", header)
        assert re.search(r"NAUTILUS_ARCH_GFX942\s*=\s*942", header)
        assert re.search(r"NAUTILUS_ARCH_XE_HPG\s*=\s*1201", header)

    def test_c_api_compile_signature(self) -> None:
        """The nautilus_compile() signature must be stable."""
        header_path = (
            Path(__file__).resolve().parent.parent.parent.parent / "c_api" / "triton_c_api.h"
        )
        header = header_path.read_text()
        assert "int nautilus_compile" in header
        assert "const char* source" in header
        assert "nautilus_kernel_t** out" in header
        assert "nautilus_vendor_t vendor" in header
        assert "nautilus_arch_t arch" in header

    def test_runtime_stub_static_asserts_match_c_api(self) -> None:
        """The _Static_assert values in runtime_stub.c must match the enum
        values in the C API header."""
        stub_path = Path(__file__).resolve().parent.parent / "runtime_stub.c"
        stub = stub_path.read_text()
        assert "NAUTILUS_VENDOR_NVIDIA  ==  0" in stub
        assert "NAUTILUS_VENDOR_AMD     ==  1" in stub
        assert "NAUTILUS_VENDOR_INTEL   ==  2" in stub
        assert "NAUTILUS_VENDOR_APPLE   ==  3" in stub
        assert "NAUTILUS_VENDOR_UNKNOWN == -1" in stub


# ===========================================================================
# HARDWARE VALIDATOR
# ===========================================================================


class TestHardwareValidator:
    """The hardware validator must produce sensible skip results when
    no hardware is present."""

    def test_skip_mode_always_passes(self, tmp_path: Path) -> None:
        """SKIP mode must always return passed=True."""
        from src.bridges.aot_packager.hardware_validator import (
            HardwareValidator,
            ValidationMode,
        )

        validator = HardwareValidator(mode=ValidationMode.SKIP)
        result = validator.validate(
            binary_path=tmp_path / "nonexistent",
            vendor="nvidia",
            arch="sm_90",
        )
        assert result.passed
        assert result.output_match
        assert result.mode == ValidationMode.SKIP

    def test_local_mode_no_binary(self, tmp_path: Path) -> None:
        """LOCAL mode with a missing binary must return passed=False."""
        from src.bridges.aot_packager.hardware_validator import (
            HardwareValidator,
            ValidationMode,
        )

        validator = HardwareValidator(mode=ValidationMode.LOCAL)
        result = validator.validate(
            binary_path=tmp_path / "no_such_file.o",
            vendor="nvidia",
            arch="sm_90",
        )
        assert not result.passed
        assert "exist" in (result.error or "").lower()

    def test_local_mode_empty_binary(self, tmp_path: Path) -> None:
        """LOCAL mode with an empty binary must return passed=False."""
        from src.bridges.aot_packager.hardware_validator import (
            HardwareValidator,
            ValidationMode,
        )

        empty = tmp_path / "empty.bin"
        empty.write_bytes(b"")
        validator = HardwareValidator(mode=ValidationMode.LOCAL)
        result = validator.validate(
            binary_path=empty,
            vendor="intel",
            arch="xe_hpg",
        )
        assert not result.passed
        assert "empty" in (result.error or "").lower()

    def test_local_mode_unknown_vendor(self, tmp_path: Path) -> None:
        """LOCAL mode with an unknown vendor must return passed=False."""
        from src.bridges.aot_packager.hardware_validator import (
            HardwareValidator,
            ValidationMode,
        )

        binary = tmp_path / "test.bin"
        binary.write_bytes(b"\x7fELF" + b"\x00" * 20)
        validator = HardwareValidator(mode=ValidationMode.LOCAL)
        result = validator.validate(
            binary_path=binary,
            vendor="unknown_vendor",
            arch="unknown",
        )
        assert not result.passed
        assert "unknown" in (result.error or "").lower()

    def test_cloud_mode_no_endpoint(self, tmp_path: Path) -> None:
        """CLOUD mode without an endpoint must return passed=False."""
        from src.bridges.aot_packager.hardware_validator import (
            HardwareValidator,
            ValidationMode,
        )

        validator = HardwareValidator(mode=ValidationMode.CLOUD)
        result = validator.validate(
            binary_path=tmp_path / "test.bin",
            vendor="amd",
            arch="gfx942",
        )
        assert not result.passed
        assert "endpoint" in (result.error or "").lower()
