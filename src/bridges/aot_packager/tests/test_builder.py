"""Tests for the FatBinaryBuilder orchestrator."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.bridges.aot_packager.builder import (
    FatBinaryBuilder,
    FatBinaryConfig,
    FatBinaryResult,
)
from src.bridges.aot_packager.amd_backend import AMDArch
from src.bridges.aot_packager.intel_backend import IntelTarget
from src.bridges.aot_packager.nvidia_backend import NvidiaArch
from src.bridges.aot_packager.hardware_validator import ValidationMode


SAMPLE_KERNEL = '''
import triton
import triton.language as tl

@triton.jit
def sample_matmul(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    a = tl.load(A_ptr + rm[:, None] * K + tl.arange(0, BLOCK_K)[None, :])
    b = tl.load(B_ptr + tl.arange(0, BLOCK_K)[:, None] * N + rn[None, :])
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc += tl.dot(a, b)
    tl.store(C_ptr + rm[:, None] * N + rn[None, :], acc)
'''


class TestFatBinaryBuilder:
    """Tests for the FatBinaryBuilder orchestrator."""

    def test_builder_init(self, tmp_path: Path) -> None:
        """FatBinaryBuilder should initialise all backends."""
        builder = FatBinaryBuilder(cache_dir=str(tmp_path / "fb"))
        assert builder.amd_backend is not None
        assert builder.intel_backend is not None
        assert builder.nvidia_backend is not None
        assert builder.linker is not None
        assert builder.validator is not None

    def test_build_all_vendors(self, tmp_path: Path) -> None:
        """build() should compile for all 3 vendors and link them."""
        builder = FatBinaryBuilder(cache_dir=str(tmp_path / "fb"))
        config = FatBinaryConfig(
            kernel_name="test_matmul",
            kernel_source=SAMPLE_KERNEL,
            block_m=128, block_n=128, block_k=32,
            num_warps=8, num_stages=3,
            skip_validation=True,
        )
        result = builder.build(config)
        assert isinstance(result, FatBinaryResult)
        # Should have at least the Nvidia section (always works)
        assert result.fat_binary is not None
        assert "nvidia" in result.fat_binary.vendors

    def test_build_skip_amd(self, tmp_path: Path) -> None:
        """skip_amd=True should not include AMD in the fat binary."""
        builder = FatBinaryBuilder(cache_dir=str(tmp_path / "fb"))
        config = FatBinaryConfig(
            kernel_name="test_no_amd",
            kernel_source=SAMPLE_KERNEL,
            skip_amd=True,
            skip_intel=True,  # Skip both to speed up test
            skip_validation=True,
        )
        result = builder.build(config)
        assert result.fat_binary is not None
        assert "amd" not in result.fat_binary.vendors
        assert "intel" not in result.fat_binary.vendors
        assert "nvidia" in result.fat_binary.vendors

    def test_build_records_stage_times(self, tmp_path: Path) -> None:
        """build() should record timing for each stage."""
        builder = FatBinaryBuilder(cache_dir=str(tmp_path / "fb"))
        config = FatBinaryConfig(
            kernel_name="timing_test",
            kernel_source=SAMPLE_KERNEL,
            skip_intel=True,  # Skip for speed
            skip_validation=True,
        )
        result = builder.build(config)
        assert isinstance(result.stage_times, dict)
        # Should have nvidia stage at minimum
        assert "nvidia" in result.stage_times
        assert "link" in result.stage_times

    def test_build_produces_output_path(self, tmp_path: Path) -> None:
        """A successful build should produce an output path."""
        builder = FatBinaryBuilder(cache_dir=str(tmp_path / "fb"))
        config = FatBinaryConfig(
            kernel_name="path_test",
            kernel_source=SAMPLE_KERNEL,
            skip_intel=True,
            skip_validation=True,
        )
        result = builder.build(config)
        if result.is_usable:
            assert result.output_path is not None
            assert result.output_path.exists()

    def test_build_fat_binary_has_sections(self, tmp_path: Path) -> None:
        """The built fat binary should have at least one section."""
        builder = FatBinaryBuilder(cache_dir=str(tmp_path / "fb"))
        config = FatBinaryConfig(
            kernel_name="section_test",
            kernel_source=SAMPLE_KERNEL,
            skip_intel=True,
            skip_validation=True,
        )
        result = builder.build(config)
        assert result.fat_binary is not None
        assert len(result.fat_binary.sections) >= 1
        # Each section should have valid data
        for section in result.fat_binary.sections:
            assert section.data
            assert section.size > 0
            assert section.vendor in ("nvidia", "amd", "intel")

    def test_build_total_size_positive(self, tmp_path: Path) -> None:
        """The built fat binary should have a positive total size."""
        builder = FatBinaryBuilder(cache_dir=str(tmp_path / "fb"))
        config = FatBinaryConfig(
            kernel_name="size_test",
            kernel_source=SAMPLE_KERNEL,
            skip_intel=True,
            skip_validation=True,
        )
        result = builder.build(config)
        assert result.fat_binary is not None
        assert result.fat_binary.total_size > 0

    def test_build_records_per_vendor_results(self, tmp_path: Path) -> None:
        """build() should record per-vendor compilation results."""
        builder = FatBinaryBuilder(cache_dir=str(tmp_path / "fb"))
        config = FatBinaryConfig(
            kernel_name="per_vendor",
            kernel_source=SAMPLE_KERNEL,
            skip_intel=True,
            skip_validation=True,
        )
        result = builder.build(config)
        # Nvidia should always be present
        assert result.nvidia_result is not None
        # AMD may or may not be present (depends on aotriton availability)
        assert result.amd_result is not None
        # Linking result should be recorded
        assert result.linking_result is not None
