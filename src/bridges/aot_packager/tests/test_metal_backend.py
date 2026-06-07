"""Tests for the Apple Metal AOT backend."""

from __future__ import annotations

import platform
from pathlib import Path

import pytest

from src.bridges.aot_packager.fat_binary import SectionFormat
from src.bridges.aot_packager.linker import section_name_for
from src.bridges.aot_packager.metal_backend import (
    MetalBackend,
    MetalCompilationResult,
    MetalTarget,
)
from src.common.errors import (
    HardwareNotFoundError,
)

SAMPLE_KERNEL = """
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
"""


class TestMetalBackendInit:
    """Basic initialisation contract."""

    def test_metal_backend_init(self, tmp_path: Path) -> None:
        backend = MetalBackend(cache_dir=str(tmp_path / "metal"))
        assert backend.target == MetalTarget.APPLE_M2
        assert backend.timeout_seconds > 0
        assert backend.cache_dir.exists()

    def test_metal_backend_custom_target(self, tmp_path: Path) -> None:
        backend = MetalBackend(
            target=MetalTarget.APPLE_M3,
            cache_dir=str(tmp_path / "metal"),
        )
        assert backend.target == MetalTarget.APPLE_M3

    @pytest.mark.parametrize("target", list(MetalTarget))
    def test_metal_backend_each_target(
        self,
        tmp_path: Path,
        target: MetalTarget,
    ) -> None:
        backend = MetalBackend(target=target, cache_dir=str(tmp_path / "metal"))
        assert backend.target == target
        assert backend.supports_target(target.value)


class TestAppleSiliconDetection:
    """``detect_apple_silicon()`` must report the right state per host."""

    def test_detection_returns_dict(self) -> None:
        info = MetalBackend.detect_apple_silicon()
        assert isinstance(info, dict)
        assert "available" in info
        assert "reason" in info
        assert "os" in info
        assert "machine" in info
        assert "xcrun_path" in info

    def test_detection_consistent_with_platform(self) -> None:
        info = MetalBackend.detect_apple_silicon()
        assert info["os"] == platform.system()
        assert info["machine"] == platform.machine()

    def test_detection_linux_returns_false(self) -> None:
        if platform.system() == "Darwin":
            pytest.skip("Not running on Linux")
        info = MetalBackend.detect_apple_silicon()
        assert info["available"] is False
        assert "Darwin" in info["reason"] or "macOS" in info["reason"] or info["os"] != "Darwin"


class TestMetallibValidation:
    """Magic-byte validation must reject anything that isn't a real metallib."""

    def test_validate_metallib_accepts_real_magic(self) -> None:
        assert MetalBackend._validate_metallib(b"MTLB\x00\x00\x00\x00") is True

    def test_validate_metallib_rejects_short(self) -> None:
        assert MetalBackend._validate_metallib(b"MTL") is False
        assert MetalBackend._validate_metallib(b"") is False

    def test_validate_metallib_rejects_wrong_magic(self) -> None:
        assert MetalBackend._validate_metallib(b"XXXX\x00\x00\x00\x00") is False
        assert MetalBackend._validate_metallib(b"\x7fELF\x00\x00\x00\x00") is False

    def test_validate_air_accepts_real_magic(self) -> None:
        assert MetalBackend._validate_air(b"AIRI\x00\x00\x00\x00") is True

    def test_validate_air_rejects_short(self) -> None:
        assert MetalBackend._validate_air(b"AIR") is False


class TestMetalCompileKernel:
    """``compile_kernel`` must produce a real result or a structured error."""

    def test_compile_kernel_returns_result(self, tmp_path: Path) -> None:
        backend = MetalBackend(cache_dir=str(tmp_path / "metal"))
        result = backend.compile_kernel(
            kernel_source=SAMPLE_KERNEL,
            kernel_name="sample_matmul",
        )
        assert isinstance(result, MetalCompilationResult)
        assert result.target == "apple_m2"

    def test_compile_kernel_fails_on_non_apple_host(self, tmp_path: Path) -> None:
        if platform.system() == "Darwin":
            pytest.skip("Not running on non-Apple host")
        backend = MetalBackend(cache_dir=str(tmp_path / "metal"))
        result = backend.compile_kernel(
            kernel_source=SAMPLE_KERNEL,
            kernel_name="apple_host_test",
        )
        assert result.success is False
        assert result.error is not None
        assert result.error_code == "E_HARDWARE_NOT_FOUND"
        assert "arm64" in result.error or "Darwin" in result.error

    def test_strict_raises_on_non_apple_host(self, tmp_path: Path) -> None:
        if platform.system() == "Darwin":
            pytest.skip("Not running on non-Apple host")
        backend = MetalBackend(cache_dir=str(tmp_path / "metal"))
        with pytest.raises(HardwareNotFoundError):
            backend.compile_kernel_strict(
                kernel_source=SAMPLE_KERNEL,
                kernel_name="strict_test",
            )

    def test_compile_kernel_strict_returns_result_on_success(
        self,
        tmp_path: Path,
    ) -> None:
        backend = MetalBackend(cache_dir=str(tmp_path / "metal"))
        try:
            result = backend.compile_kernel_strict(
                kernel_source=SAMPLE_KERNEL,
                kernel_name="strict_ok",
            )
        except HardwareNotFoundError:
            # Expected on non-Apple hosts; a real failure on macOS.
            assert platform.system() != "Darwin"
            return
        assert result.success
        assert result.is_usable
        assert result.target == "apple_m2"


class TestMetalCompilationResultShape:
    """Result dataclass invariants."""

    def test_output_bytes_prefers_metallib(self) -> None:
        r = MetalCompilationResult(
            success=True,
            msl_text="// msl",
            air_bytes=b"AIRI" + b"\x00" * 4,
            metallib_bytes=b"MTLB" + b"\x00" * 4,
        )
        assert r.output_bytes == b"MTLB" + b"\x00" * 4
        assert r.is_usable

    def test_output_bytes_falls_back_to_air(self) -> None:
        r = MetalCompilationResult(
            success=True,
            msl_text="// msl",
            air_bytes=b"AIRI" + b"\x00" * 4,
            metallib_bytes=None,
        )
        assert r.output_bytes == b"AIRI" + b"\x00" * 4
        assert r.is_usable

    def test_output_bytes_falls_back_to_msl(self) -> None:
        r = MetalCompilationResult(
            success=True,
            msl_text="// msl only",
            air_bytes=None,
            metallib_bytes=None,
        )
        assert r.output_bytes == b"// msl only"
        assert r.is_usable

    def test_output_bytes_none_on_failure(self) -> None:
        r = MetalCompilationResult(success=False)
        assert r.output_bytes is None
        assert r.is_usable is False


class TestFatBinarySectionFormat:
    """METALLIB / AIR section formats must be present in the enum."""

    def test_metallib_format_exists(self) -> None:
        assert SectionFormat.METALLIB.value == "metallib"

    def test_air_format_exists(self) -> None:
        assert SectionFormat.AIR.value == "air"

    def test_section_name_for_apple(self) -> None:
        # The runtime stub looks for sections named exactly
        # .nautilus.apple.<kernel_name> for metallib blobs.
        name = section_name_for("apple", "my_kernel", fmt_suffix="metallib")
        assert name == ".nautilus.apple.metallib.my_kernel"
        # And the format-less variant for the canonical Apple section.
        name = section_name_for("apple", "my_kernel")
        assert name == ".nautilus.apple.my_kernel"


class TestMetalTargetEnum:
    """The MetalTarget enum is the public identifier for Apple Silicon."""

    def test_targets_are_apple_families(self) -> None:
        for t in MetalTarget:
            assert t.value.startswith("apple_") or t.value == "generic_metal"

    def test_target_supports_target(self, tmp_path: Path) -> None:
        backend = MetalBackend(target=MetalTarget.APPLE_M2, cache_dir=str(tmp_path))
        assert backend.supports_target("apple_m2") is True
        assert backend.supports_target("apple_m3") is False
        assert backend.supports_target("unknown") is False


class TestGetVersion:
    """get_version must always return a string (never raise)."""

    def test_get_version_returns_string(self, tmp_path: Path) -> None:
        backend = MetalBackend(cache_dir=str(tmp_path / "metal"))
        version = backend.get_version()
        assert isinstance(version, str)
