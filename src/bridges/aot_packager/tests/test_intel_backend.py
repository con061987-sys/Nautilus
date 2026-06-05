"""Tests for the Intel AOT backend (oneAPI/SYCL wrapper)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.bridges.aot_packager.intel_backend import (
    IntelBackend,
    IntelTarget,
    IntelCompilationResult,
)


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


class TestIntelBackend:
    """Tests for the Intel AOT backend."""

    def test_intel_backend_init(self, tmp_path: Path) -> None:
        """IntelBackend should initialise with sensible defaults."""
        backend = IntelBackend(cache_dir=str(tmp_path / "intel"))
        assert backend.target == IntelTarget.XE_HPG
        assert backend.timeout_seconds > 0
        assert backend.cache_dir.exists()

    def test_intel_backend_custom_target(self, tmp_path: Path) -> None:
        """IntelBackend should accept a custom target."""
        backend = IntelBackend(
            target=IntelTarget.XE2,
            cache_dir=str(tmp_path / "intel"),
        )
        assert backend.target == IntelTarget.XE2

    def test_compile_kernel_returns_result(self, tmp_path: Path) -> None:
        """compile_kernel should return an IntelCompilationResult."""
        backend = IntelBackend(cache_dir=str(tmp_path / "intel"))
        result = backend.compile_kernel(
            triton_kernel_ir=SAMPLE_KERNEL,
            kernel_name="test_matmul",
        )
        assert isinstance(result, IntelCompilationResult)
        assert result.target == "intel_gpu_xehpg"

    def test_compile_kernel_caches_result(self, tmp_path: Path) -> None:
        """Subsequent compilations should hit cache."""
        backend = IntelBackend(cache_dir=str(tmp_path / "intel"))
        result1 = backend.compile_kernel(
            triton_kernel_ir=SAMPLE_KERNEL,
            kernel_name="cache_test",
        )
        result2 = backend.compile_kernel(
            triton_kernel_ir=SAMPLE_KERNEL,
            kernel_name="cache_test",
        )
        if result1.is_usable and result2.is_usable:
            assert result2.cache_hit is True

    def test_supports_target(self, tmp_path: Path) -> None:
        """supports_target should match the configured target."""
        backend = IntelBackend(target=IntelTarget.XE_HPG, cache_dir=str(tmp_path))
        assert backend.supports_target("intel_gpu_xehpg") is True
        assert backend.supports_target("intel_gpu_xe2") is False
        assert backend.supports_target("unknown") is False

    def test_get_version(self, tmp_path: Path) -> None:
        """get_version should return a string."""
        backend = IntelBackend(cache_dir=str(tmp_path))
        version = backend.get_version()
        assert isinstance(version, str)
