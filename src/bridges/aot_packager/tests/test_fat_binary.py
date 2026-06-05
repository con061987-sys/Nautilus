"""Tests for the fat binary format."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.bridges.aot_packager.fat_binary import (
    FatBinary,
    KernelSection,
    SectionFormat,
)


SAMPLE_PTX = b".version 7.0\n.target sm_90\n"
SAMPLE_HSACO = b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 56
SAMPLE_SPV = b"\x03\x02\x23\x07" + b"\x00" * 20


class TestKernelSection:
    """Tests for the KernelSection class."""

    def test_section_creation(self) -> None:
        """A section should store vendor, arch, format, and data."""
        section = KernelSection(
            vendor="nvidia",
            arch="sm_90",
            format=SectionFormat.PTX,
            data=SAMPLE_PTX,
        )
        assert section.vendor == "nvidia"
        assert section.arch == "sm_90"
        assert section.format == SectionFormat.PTX
        assert section.data == SAMPLE_PTX

    def test_section_size(self) -> None:
        """The size property should return the data length."""
        section = KernelSection(
            vendor="amd",
            arch="gfx942",
            format=SectionFormat.HSACO,
            data=SAMPLE_HSACO,
        )
        assert section.size == len(SAMPLE_HSACO)

    def test_section_sha256(self) -> None:
        """The sha256 property should compute a stable hash."""
        section = KernelSection(
            vendor="intel",
            arch="intel_gpu_xehpg",
            format=SectionFormat.SPV,
            data=SAMPLE_SPV,
        )
        sha = section.sha256
        assert len(sha) == 64  # SHA-256 hex length
        # Same data should produce the same hash
        section2 = KernelSection(
            vendor="intel",
            arch="intel_gpu_xehpg",
            format=SectionFormat.SPV,
            data=SAMPLE_SPV,
        )
        assert section.sha256 == section2.sha256

    def test_section_to_dict(self) -> None:
        """to_dict should serialise the section metadata."""
        section = KernelSection(
            vendor="amd",
            arch="gfx942",
            format=SectionFormat.HSACO,
            data=SAMPLE_HSACO,
            metadata={"compilation_time": 1.5},
        )
        d = section.to_dict()
        assert d["vendor"] == "amd"
        assert d["arch"] == "gfx942"
        assert d["format"] == "hsaco"
        assert d["size"] == len(SAMPLE_HSACO)
        assert d["metadata"]["compilation_time"] == 1.5


class TestFatBinary:
    """Tests for the FatBinary class."""

    def test_fat_binary_creation(self) -> None:
        """A fat binary should be initialised with a name and optional sections."""
        fb = FatBinary(kernel_name="matmul")
        assert fb.kernel_name == "matmul"
        assert fb.sections == []

    def test_add_section(self) -> None:
        """add_section should add a new vendor section."""
        fb = FatBinary(kernel_name="test")
        section = KernelSection(
            vendor="nvidia",
            arch="sm_90",
            format=SectionFormat.PTX,
            data=SAMPLE_PTX,
        )
        fb.add_section(section)
        assert len(fb.sections) == 1
        assert fb.sections[0].vendor == "nvidia"

    def test_add_section_replaces_existing(self) -> None:
        """add_section should replace existing section for the same vendor+arch."""
        fb = FatBinary(kernel_name="test")
        fb.add_section(KernelSection(
            vendor="nvidia", arch="sm_90", format=SectionFormat.PTX, data=b"old",
        ))
        fb.add_section(KernelSection(
            vendor="nvidia", arch="sm_90", format=SectionFormat.PTX, data=b"new",
        ))
        assert len(fb.sections) == 1
        assert fb.sections[0].data == b"new"

    def test_get_section(self) -> None:
        """get_section should return the section for a given vendor."""
        fb = FatBinary(kernel_name="test")
        section = KernelSection(
            vendor="amd",
            arch="gfx942",
            format=SectionFormat.HSACO,
            data=SAMPLE_HSACO,
        )
        fb.add_section(section)
        amd_section = fb.get_section("amd")
        assert amd_section is not None
        assert amd_section.arch == "gfx942"
        assert fb.get_section("nvidia") is None

    def test_total_size(self) -> None:
        """total_size should sum all section sizes."""
        fb = FatBinary(kernel_name="test")
        fb.add_section(KernelSection("nvidia", "sm_90", SectionFormat.PTX, b"a" * 100))
        fb.add_section(KernelSection("amd", "gfx942", SectionFormat.HSACO, b"b" * 200))
        assert fb.total_size == 300

    def test_vendors(self) -> None:
        """vendors should return the unique list of vendors."""
        fb = FatBinary(kernel_name="test")
        fb.add_section(KernelSection("nvidia", "sm_90", SectionFormat.PTX, b""))
        fb.add_section(KernelSection("amd", "gfx942", SectionFormat.HSACO, b""))
        assert set(fb.vendors) == {"nvidia", "amd"}

    def test_to_json_and_back(self) -> None:
        """FatBinary should round-trip through JSON."""
        original = FatBinary(
            kernel_name="test_kernel",
            metadata={"build_host": "ci-server-01"},
        )
        original.add_section(KernelSection(
            vendor="nvidia",
            arch="sm_90",
            format=SectionFormat.PTX,
            data=SAMPLE_PTX,
        ))
        original.add_section(KernelSection(
            vendor="amd",
            arch="gfx942",
            format=SectionFormat.HSACO,
            data=SAMPLE_HSACO,
        ))

        json_text = original.to_json()
        restored = FatBinary.from_json(json_text)
        assert restored.kernel_name == original.kernel_name
        assert len(restored.sections) == 2
        nvidia_section = restored.get_section("nvidia")
        amd_section = restored.get_section("amd")
        assert nvidia_section is not None
        assert amd_section is not None
        assert nvidia_section.data == SAMPLE_PTX
        assert amd_section.data == SAMPLE_HSACO

    def test_to_bytes_and_back(self, tmp_path: Path) -> None:
        """FatBinary should round-trip through the binary format."""
        original = FatBinary(kernel_name="bin_test")
        original.add_section(KernelSection(
            vendor="intel",
            arch="intel_gpu_xehpg",
            format=SectionFormat.SPV,
            data=SAMPLE_SPV,
        ))

        data = original.to_bytes()
        assert data[:4] == b"NFAT"
        restored = FatBinary.from_bytes(data)
        assert restored.kernel_name == "bin_test"
        intel_section = restored.get_section("intel")
        assert intel_section is not None
        assert intel_section.data == SAMPLE_SPV

    def test_save_and_load(self, tmp_path: Path) -> None:
        """save() and load() should round-trip via file."""
        original = FatBinary(kernel_name="disk_test")
        original.add_section(KernelSection(
            vendor="nvidia",
            arch="sm_90",
            format=SectionFormat.PTX,
            data=SAMPLE_PTX,
        ))

        path = tmp_path / "test.fat"
        original.save(path)
        loaded = FatBinary.load(path)
        assert loaded.kernel_name == "disk_test"
        nvidia_section = loaded.get_section("nvidia")
        assert nvidia_section is not None
        assert nvidia_section.data == SAMPLE_PTX

    def test_from_bytes_invalid_magic(self) -> None:
        """from_bytes should reject invalid magic."""
        with pytest.raises(ValueError, match="magic"):
            FatBinary.from_bytes(b"NOPE" + b"\x00" * 100)
