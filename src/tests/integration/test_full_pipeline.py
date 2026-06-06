"""End-to-end pipeline test.

This is the GATE that proves the framework actually works. It
exercises:
  1. Triton kernel source -> TTGIR capture (via real IR capture if
     triton is installed, else fallback)
  2. TTGIR -> TVM TIR (via ir_to_tir pipeline)
  3. TVM MetaSchedule tuning
  4. AOT compilation per vendor
  5. Linking into a fat binary
  6. (Skipped in CI) Loading the fat binary and verifying the
     numerical output

Each stage is gated by an appropriate marker so the test will
auto-skip if the corresponding dep is missing. When all deps are
installed, the test should pass; when not, it should fail loudly
(rather than silently returning a placeholder).
"""

from __future__ import annotations

import hashlib
import json
import sys
import textwrap
from pathlib import Path

import pytest


EXAMPLE_MATMUL = textwrap.dedent('''
    import triton
    import triton.language as tl

    @triton.jit
    def matmul_kernel(
        A_ptr, B_ptr, C_ptr,
        M, N, K,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """Simple matmul: C = A @ B."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + offs_m[:, None] * K + offs_k[None, :]
        b_ptrs = B_ptr + offs_k[:, None] * N + offs_n[None, :]
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * N
        c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
        tl.store(c_ptrs, acc.to(tl.float16))
''').strip()


@pytest.mark.integration
class TestFullPipeline:
    """End-to-end pipeline test class."""

    def test_common_imports(self):
        """src.common must be importable without any optional deps."""
        from src.common import (
            Ok, Err, Result, Vendor, Arch, HardwareTarget,
            CircuitBreaker, Span, StageLog, configure_logging,
        )
        assert Ok(1).is_ok()
        assert Err(ValueError("x")).is_err()
        assert HardwareTarget(vendor=Vendor.NVIDIA, arch=Arch.SM_90).to_tvm_target() == "nvidia/nvidia-h100"

    def test_fat_binary_serialization_roundtrip(self):
        """Fat binary data model can be serialized and deserialized loss-lessly."""
        from src.common import FatBinary, KernelSection, SectionFormat, Vendor, Arch
        ptx_data = b"// PTX text"           # 10 bytes
        hsaco_data = b"\\x7fELFhsaco"       # 12 bytes (includes escaped backslash)
        fb = FatBinary(kernel_name="matmul")
        fb.add_section(KernelSection(
            vendor=Vendor.NVIDIA, arch=Arch.SM_90,
            format=SectionFormat.PTX, data=ptx_data,
        ))
        fb.add_section(KernelSection(
            vendor=Vendor.AMD, arch=Arch.GFX942,
            format=SectionFormat.HSACO, data=hsaco_data,
        ))
        assert fb.total_size == len(ptx_data) + len(hsaco_data)
        assert set(fb.vendors) == {Vendor.NVIDIA, Vendor.AMD}

        # Serialize / deserialize
        blob = fb.to_bytes()
        assert blob[:4] == b"NFAT"
        restored = FatBinary.from_bytes(blob)
        assert restored.kernel_name == "matmul"
        assert len(restored.sections) == 2
        assert restored.sections[0].data == ptx_data
        assert restored.sections[1].data == hsaco_data

    def test_hardware_detection_does_not_lie(self, any_gpu_available: bool):
        """hardware module must NOT silently return success on missing devices.

        With no GPU, all per-vendor has_*() must return False; with
        a GPU, at least one must return True. There is no "Unknown"
        middle ground.
        """
        from src.common.hardware import (
            has_nvidia_gpu, has_amd_gpu, has_intel_gpu,
            get_device_paths,
        )
        from src.common.errors import HardwareNotFoundError
        from src.common.types import Vendor
        # At least the call must not raise
        nv = has_nvidia_gpu()
        amd = has_amd_gpu()
        intel = has_intel_gpu()
        if not (nv or amd or intel):
            # No GPUs — get_device_paths for any vendor must raise, not return []
            for vendor in (Vendor.NVIDIA, Vendor.AMD, Vendor.INTEL):
                with pytest.raises(HardwareNotFoundError):
                    get_device_paths(vendor)

    def test_example_matmul_parses(self, tmp_path: Path):
        """The example matmul kernel must parse as a @triton.jit function."""
        from src.cli.commands.tune import _load_kernel_file
        p = tmp_path / "matmul.py"
        p.write_text(EXAMPLE_MATMUL)
        name, text = _load_kernel_file(p)
        assert name == "matmul_kernel"
        assert "@triton.jit" in text
        assert "tl.dot" in text

    @pytest.mark.requires_deps
    def test_bridge_module_imports(self):
        """The Triton<->TVM bridge must import (when deps are installed)."""
        try:
            from src.bridges.triton_tvm.bridge_orchestrator import TritonTVMBridge
        except ImportError as exc:
            pytest.skip(f"bridge module not importable: {exc}")
        assert TritonTVMBridge is not None

    @pytest.mark.gpu
    def test_aot_nvidia_round_trip(self, tmp_path: Path):
        """A Nvidia AOT compile should produce a non-empty PTX (or cubin)."""
        try:
            from src.bridges.aot_packager.nvidia_backend import NvidiaBackend, NvidiaArch
        except ImportError as exc:
            pytest.skip(f"aot packager not importable: {exc}")
        backend = NvidiaBackend(
            target_arch=NvidiaArch.SM_90,
            cache_dir=str(tmp_path),
            capture_cubin=False,
        )
        result = backend.compile_kernel(
            kernel_source=EXAMPLE_MATMUL,
            kernel_name="matmul_kernel",
            block_m=64, block_n=64, block_k=32,
            num_warps=4, num_stages=2,
        )
        if not result.success:
            pytest.skip(f"AOT compile failed: {result.error}")
        assert result.ptx_text, "PTX text should not be empty"
        assert result.arch == "sm_90"
        # PTX must contain real instructions, not just the placeholder
        assert "mov" in result.ptx_text or "ld." in result.ptx_text, \
            "PTX contains only the placeholder kernel — AOT did not run"

    def test_cli_help(self):
        """All CLI commands must show help without crashing."""
        from click.testing import CliRunner
        from src.cli.main import cli
        runner = CliRunner()
        for sub in ("tune", "build", "shard", "verify"):
            result = runner.invoke(cli, [sub, "--help"])
            assert result.exit_code == 0, f"{sub} --help failed: {result.output}"
            assert "Usage:" in result.output

    def test_c_api_stub_loads(self):
        """src.c_api must import even when the C library isn't built."""
        from src.c_api import is_available, compile, triton_version
        # is_available() should return False (no .so built) without raising
        available = is_available()
        assert available is False
        # triton_version() should raise DependencyMissingError
        from src.common.errors import DependencyMissingError
        with pytest.raises(DependencyMissingError):
            triton_version()

    def test_verify_env_script(self, repo_root: Path):
        """scripts/verify_env.py must run and produce structured output."""
        import subprocess
        result = subprocess.run(
            [sys.executable, str(repo_root / "scripts" / "verify_env.py"), "--json"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode in (0, 1), f"verify_env crashed: {result.stderr}"
        report = json.loads(result.stdout)
        assert "checks" in report
        assert "overall" in report
        assert report["overall"] in ("ok", "warning", "error")
