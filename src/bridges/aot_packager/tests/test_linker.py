"""Tests for the fat binary linker."""

from __future__ import annotations

from pathlib import Path

from src.bridges.aot_packager.linker import (
    FatBinaryLinker,
    LinkingResult,
)

SAMPLE_PTX = b""".version 7.0
.target sm_90
.address_size 64
.visible .entry sample_kernel() { ret; }
"""

SAMPLE_HSACO = b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 56

SAMPLE_SPV = b"\x03\x02\x23\x07" + b"\x00" * 20


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
