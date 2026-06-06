"""Tests for the Nvidia AOT backend (Triton JIT PTX capture)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.bridges.aot_packager.nvidia_backend import (
    NvidiaArch,
    NvidiaBackend,
    NvidiaCompilationResult,
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


class TestNvidiaBackend:
    """Tests for the Nvidia AOT backend."""

    def test_nvidia_backend_init(self, tmp_path: Path) -> None:
        """NvidiaBackend should initialise with sensible defaults."""
        backend = NvidiaBackend(cache_dir=str(tmp_path / "nvidia"))
        assert backend.target_arch == NvidiaArch.SM_90
        assert backend.timeout_seconds > 0
        assert backend.cache_dir.exists()

    def test_nvidia_backend_custom_arch(self, tmp_path: Path) -> None:
        """NvidiaBackend should accept a custom architecture."""
        backend = NvidiaBackend(
            target_arch=NvidiaArch.SM_100,
            cache_dir=str(tmp_path / "nvidia"),
        )
        assert backend.target_arch == NvidiaArch.SM_100

    def test_compile_kernel_returns_result(self, tmp_path: Path) -> None:
        """compile_kernel should return a NvidiaCompilationResult."""
        backend = NvidiaBackend(cache_dir=str(tmp_path / "nvidia"))
        result = backend.compile_kernel(
            kernel_source=SAMPLE_KERNEL,
            kernel_name="test_matmul",
        )
        assert isinstance(result, NvidiaCompilationResult)
        assert result.arch == "sm_90"
        # Should always be usable (produces at least minimal PTX)
        assert result.is_usable

    def test_compile_kernel_produces_ptx(self, tmp_path: Path) -> None:
        """The compiled result should contain valid PTX text."""
        backend = NvidiaBackend(cache_dir=str(tmp_path / "nvidia"))
        result = backend.compile_kernel(
            kernel_source=SAMPLE_KERNEL,
            kernel_name="ptx_test",
        )
        if result.ptx_text is not None:
            # Minimal PTX contains .version, .target, .entry
            assert ".version" in result.ptx_text
            assert ".target" in result.ptx_text

    def test_compile_kernel_caches_result(self, tmp_path: Path) -> None:
        """Subsequent compilations should hit cache."""
        backend = NvidiaBackend(cache_dir=str(tmp_path / "nvidia"))
        result1 = backend.compile_kernel(
            kernel_source=SAMPLE_KERNEL,
            kernel_name="cache_test",
        )
        result2 = backend.compile_kernel(
            kernel_source=SAMPLE_KERNEL,
            kernel_name="cache_test",
        )
        assert result1.is_usable
        assert result2.is_usable
        assert result2.cache_hit is True

    def test_supports_arch(self, tmp_path: Path) -> None:
        """supports_arch should match the configured target."""
        backend = NvidiaBackend(target_arch=NvidiaArch.SM_90, cache_dir=str(tmp_path))
        assert backend.supports_arch("sm_90") is True
        assert backend.supports_arch("sm_70") is False
        assert backend.supports_arch("unknown") is False

    def test_get_version(self, tmp_path: Path) -> None:
        """get_version should return a string."""
        backend = NvidiaBackend(cache_dir=str(tmp_path))
        version = backend.get_version()
        assert isinstance(version, str)
