"""Tests for the fat binary linker."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from src.bridges.aot_packager.linker import (
    FatBinaryLinker,
    KernelBlob,
    LinkingResult,
    section_name_for,
)

SAMPLE_PTX = b""".version 7.0
.target sm_90
.address_size 64
.visible .entry sample_kernel() { ret; }
"""

SAMPLE_HSACO = b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 56

SAMPLE_SPV = b"\x03\x02\x23\x07" + b"\x00" * 20


def _read_section_names(path: Path) -> list[str]:
    """Return every non-empty section name in an ELF object via readelf."""
    r = subprocess.run(
        ["readelf", "-W", "-S", str(path)],
        capture_output=True, text=True, check=True,
    )
    names: list[str] = []
    for line in r.stdout.splitlines():
        m = re.match(r"\s*\[\s*\d+\]\s+(\S+)", line)
        if m and m.group(1):
            names.append(m.group(1))
    return names


def _lld_or_skip() -> None:
    """Skip the test (via pytest.skip) when lld is not on PATH."""
    import pytest
    if shutil.which("ld.lld") is None and shutil.which("lld") is None:
        pytest.skip("lld not installed")


class TestFatBinaryLinker:
    """Tests for the FatBinaryLinker."""

    def test_linker_init(self, tmp_path: Path) -> None:
        """FatBinaryLinker should initialise with sensible defaults."""
        linker = FatBinaryLinker(cache_dir=str(tmp_path / "link"))
        assert linker.timeout_seconds > 0
        assert linker.cache_dir.exists()

    def test_link_fat_binary(self, tmp_path: Path) -> None:
        """link_fat_binary should produce a valid output."""
        linker = FatBinaryLinker(cache_dir=str(tmp_path / "link"))
        result = linker.link_fat_binary(
            nvidia_ptx=SAMPLE_PTX,
            amd_hsaco=SAMPLE_HSACO,
            intel_spv=SAMPLE_SPV,
            kernel_name="test_kernel",
            output_path=tmp_path / "test.fat.o",
        )
        assert isinstance(result, LinkingResult)
        # Should succeed (either via lld or manual fallback)
        assert result.output_path is not None
        assert result.output_path.exists()

    def test_link_fat_binary_caches_result(self, tmp_path: Path) -> None:
        """Subsequent links should hit cache."""
        linker = FatBinaryLinker(cache_dir=str(tmp_path / "link"))
        result1 = linker.link_fat_binary(
            nvidia_ptx=SAMPLE_PTX,
            kernel_name="cache_test",
            output_path=tmp_path / "test1.fat.o",
        )
        result2 = linker.link_fat_binary(
            nvidia_ptx=SAMPLE_PTX,
            kernel_name="cache_test",
            output_path=tmp_path / "test2.fat.o",
        )
        assert result1.is_usable
        assert result2.is_usable
        assert result2.cache_hit is True

    def test_link_with_runtime_stub(self, tmp_path: Path) -> None:
        """A fat binary can include a runtime stub."""
        linker = FatBinaryLinker(cache_dir=str(tmp_path / "link"))
        runtime_stub = linker._wrap_section_data(b"stub_code", ".stub", "PROGBITS")
        result = linker.link_fat_binary(
            nvidia_ptx=SAMPLE_PTX,
            runtime_stub_o=runtime_stub,
            kernel_name="with_stub",
            output_path=tmp_path / "stub.fat.o",
        )
        assert result.is_usable

    def test_link_with_minimal_input(self, tmp_path: Path) -> None:
        """A fat binary with a single minimal section should still produce an output."""
        linker = FatBinaryLinker(cache_dir=str(tmp_path / "link"))
        minimal_o = linker._wrap_section_data(b"test", ".test", "PROGBITS")
        result = linker.link_fat_binary(
            runtime_stub_o=minimal_o,
            kernel_name="empty",
            output_path=tmp_path / "empty.fat.o",
        )
        assert result.is_usable
        assert result.output_size > 0

    def test_get_version(self, tmp_path: Path) -> None:
        """get_version should return a string."""
        linker = FatBinaryLinker(cache_dir=str(tmp_path))
        version = linker.get_version()
        assert isinstance(version, str)


class TestUniqueSectionNaming:
    """Task 17: unique section names per (vendor, kernel_name, fmt)."""

    def test_section_name_format(self) -> None:
        """section_name_for builds ``.nautilus.{vendor}.{kernel}``."""
        assert section_name_for("nvidia", "matmul") == ".nautilus.nvidia.matmul"
        assert section_name_for("amd", "layernorm") == ".nautilus.amd.layernorm"
        assert (
            section_name_for("nvidia", "matmul", fmt_suffix="ptx")
            == ".nautilus.nvidia.ptx.matmul"
        )
        assert (
            section_name_for("nvidia", "matmul", fmt_suffix="cubin")
            == ".nautilus.nvidia.cubin.matmul"
        )

    def test_section_name_is_unique_per_blob(self, tmp_path: Path) -> None:
        """The linker must produce a distinct section name per (vendor, kernel, fmt)."""
        _lld_or_skip()
        linker = FatBinaryLinker(cache_dir=str(tmp_path / "link"))
        blobs = [
            KernelBlob(kernel_name="k1", vendor="nvidia", arch="sm_90", fmt="ptx", data=b"a"),
            KernelBlob(kernel_name="k1", vendor="nvidia", arch="sm_90", fmt="cubin", data=b"b"),
            KernelBlob(kernel_name="k2", vendor="nvidia", arch="sm_90", fmt="ptx", data=b"c"),
            KernelBlob(kernel_name="k1", vendor="amd", arch="gfx942", fmt="hsaco", data=b"d"),
        ]
        result = linker.link_fat_binary(
            kernels=blobs, kernel_name="k1",
            output_path=tmp_path / "uniq.fat.o",
        )
        assert result.is_usable
        assert len(result.section_names) == 4
        assert len(set(result.section_names)) == 4, (
            f"Duplicate section names: {result.section_names}"
        )

    def test_readelf_shows_unique_sections(self, tmp_path: Path) -> None:
        """``readelf -S`` must list every section under a unique name."""
        _lld_or_skip()
        linker = FatBinaryLinker(cache_dir=str(tmp_path / "link"))
        blobs = [
            KernelBlob(kernel_name="matmul", vendor="nvidia", arch="sm_90", fmt="ptx", data=SAMPLE_PTX),
            KernelBlob(kernel_name="attention", vendor="nvidia", arch="sm_90", fmt="ptx", data=SAMPLE_PTX),
            KernelBlob(kernel_name="layernorm", vendor="amd", arch="gfx942", fmt="hsaco", data=SAMPLE_HSACO),
        ]
        result = linker.link_fat_binary(
            kernels=blobs, kernel_name="multi",
            output_path=tmp_path / "readelf.fat.o",
        )
        names = _read_section_names(result.output_path)
        nautilus_sections = [n for n in names if n.startswith(".nautilus.") and n != ".nautilus.index"]
        assert len(nautilus_sections) == 3
        assert len(set(nautilus_sections)) == 3, (
            f"lld silently merged sections: {nautilus_sections}"
        )
        assert ".nautilus.index" in names

    def test_lld_silent_on_multi_kernels(self, tmp_path: Path) -> None:
        """``ld.lld -r`` should not print merge / duplicate-name warnings."""
        _lld_or_skip()
        linker = FatBinaryLinker(cache_dir=str(tmp_path / "link"))
        blobs = [
            KernelBlob(kernel_name="a", vendor="nvidia", arch="sm_90", fmt="ptx", data=b"x"),
            KernelBlob(kernel_name="b", vendor="nvidia", arch="sm_90", fmt="ptx", data=b"y"),
        ]
        result = linker.link_fat_binary(
            kernels=blobs, kernel_name="dup_test",
            output_path=tmp_path / "silent.fat.o",
        )
        assert result.is_usable
        assert result.error is None


class TestNautilusIndex:
    """Task 17: machine-parseable ``.nautilus.index`` section."""

    def test_index_text_contains_all_blobs(self, tmp_path: Path) -> None:
        """Every blob shows up as one record in the index."""
        _lld_or_skip()
        linker = FatBinaryLinker(cache_dir=str(tmp_path / "link"))
        blobs = [
            KernelBlob(kernel_name="alpha", vendor="nvidia", arch="sm_90", fmt="ptx", data=b"12345"),
            KernelBlob(kernel_name="beta",  vendor="amd",    arch="gfx942", fmt="hsaco", data=b"67890"),
        ]
        result = linker.link_fat_binary(
            kernels=blobs, kernel_name="alpha",
            output_path=tmp_path / "idx.fat.o",
        )
        records = result.index_text.strip("\n").split("\n")
        assert len(records) == 2
        assert any(r.startswith("alpha|nvidia|sm_90|ptx|") for r in records)
        assert any(r.startswith("beta|amd|gfx942|hsaco|") for r in records)
        for r in records:
            fields = r.split("|")
            assert len(fields) == 6
            assert int(fields[5]) > 0

    def test_index_record_round_trip(self, tmp_path: Path) -> None:
        """The C runtime stub should be able to look up blobs by (kernel, vendor)."""
        linker = FatBinaryLinker(cache_dir=str(tmp_path / "link"))
        blobs = [
            KernelBlob(kernel_name="hello", vendor="intel", arch="xe_hpg", fmt="spv", data=b"X" * 17),
        ]
        section_names = [section_name_for(b.vendor, b.kernel_name, fmt_suffix=b.fmt) for b in blobs]
        index_text = "\n".join(
            "|".join([
                b.kernel_name, b.vendor, b.arch, b.fmt,
                section_name_for(b.vendor, b.kernel_name, fmt_suffix=b.fmt),
                str(len(b.data)),
            ])
            for b in blobs
        ) + "\n\n"
        nm = linker._build_index_object(tmp_path, blobs, section_names, index_text)
        assert nm is not None and nm.exists()
        r = subprocess.run(["nm", "--defined-only", str(nm)], capture_output=True, text=True)
        assert "nautilus_index_data" in r.stdout
        assert "nautilus_index_size" in r.stdout

    def test_runtime_stub_has_index_api(self) -> None:
        """The runtime stub object exports the index lookup symbols."""
        repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
        stub_path = repo_root / "build" / "runtime_stub.o"
        if not stub_path.exists():
            stub_path.parent.mkdir(parents=True, exist_ok=True)
            src_c = repo_root / "src" / "bridges" / "aot_packager" / "runtime_stub.c"
            subprocess.run(
                ["gcc", "-c", "-nostdlib", "-ffreestanding", "-Wall", "-std=c11",
                 "-o", str(stub_path), str(src_c)],
                check=True,
            )
        r = subprocess.run(["nm", "--defined-only", str(stub_path)], capture_output=True, text=True)
        assert "nautilus_index_find" in r.stdout
        assert "nautilus_index_find_by_vendor" in r.stdout
        assert "nautilus_dispatch_with_index" in r.stdout
        r2 = subprocess.run(["nm", "--undefined", str(stub_path)], capture_output=True, text=True)
        assert "nautilus_index_data" in r2.stdout
        assert "nautilus_index_size" in r2.stdout
