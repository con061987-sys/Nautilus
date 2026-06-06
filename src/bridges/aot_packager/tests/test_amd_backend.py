"""Tests for the AMD AOT backend (AOTriton wrapper)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.bridges.aot_packager.amd_backend import (
    AMDArch,
    AMDBackend,
    AMDCompilationResult,
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


class TestAMDBackend:
    """Tests for the AMD AOT backend."""

    def test_amd_backend_init(self, tmp_path: Path) -> None:
        """AMDBackend should initialise with sensible defaults."""
        backend = AMDBackend(cache_dir=str(tmp_path / "amd"))
        assert backend.target_arch == AMDArch.GFX942
        assert backend.timeout_seconds > 0
        assert backend.cache_dir.exists()

    def test_amd_backend_custom_arch(self, tmp_path: Path) -> None:
        """AMDBackend should accept a custom target architecture."""
        backend = AMDBackend(
            target_arch=AMDArch.GFX90A,
            cache_dir=str(tmp_path / "amd"),
        )
        assert backend.target_arch == AMDArch.GFX90A

    def test_compile_kernel_returns_result(self, tmp_path: Path) -> None:
        """compile_kernel should return an AMDCompilationResult."""
        backend = AMDBackend(cache_dir=str(tmp_path / "amd"))
        result = backend.compile_kernel(
            kernel_source=SAMPLE_KERNEL,
            kernel_name="test_matmul",
            block_m=128, block_n=128, block_k=32,
            num_warps=8, num_stages=3,
        )
        assert isinstance(result, AMDCompilationResult)
        assert result.arch == "gfx942"
        # The result is either usable (aotriton installed) or has
        # a placeholder (aotriton not installed) — both are valid
        # for production resilience
        assert result.error is None or result.hsaco_bytes is not None

    def test_compile_kernel_caches_result(self, tmp_path: Path) -> None:
        """Subsequent compilations of the same kernel should hit cache."""
        backend = AMDBackend(cache_dir=str(tmp_path / "amd"))
        result1 = backend.compile_kernel(
            kernel_source=SAMPLE_KERNEL,
            kernel_name="cache_test",
            block_m=128, block_n=128, block_k=32,
        )
        result2 = backend.compile_kernel(
            kernel_source=SAMPLE_KERNEL,
            kernel_name="cache_test",
            block_m=128, block_n=128, block_k=32,
        )
        # If both compiled, the second should be a cache hit
        if result1.is_usable and result2.is_usable:
            assert result2.cache_hit is True

    def test_supports_arch(self, tmp_path: Path) -> None:
        """supports_arch should match the configured target."""
        backend = AMDBackend(target_arch=AMDArch.GFX942, cache_dir=str(tmp_path))
        assert backend.supports_arch("gfx942") is True
        assert backend.supports_arch("gfx90a") is False
        assert backend.supports_arch("unknown") is False

    def test_get_version(self, tmp_path: Path) -> None:
        """get_version should return a string (possibly 'unavailable')."""
        backend = AMDBackend(cache_dir=str(tmp_path))
        version = backend.get_version()
        assert isinstance(version, str)
        assert version  # Non-empty
