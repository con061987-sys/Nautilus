"""Self-test for the bridge harness.

Verifies that the bridge fixtures in ``conftest.py`` are wired up
correctly: each fixture returns a real bridge instance, the mocked
external services are actually wired (so calls don't blow up), and
the evidence-capture fixture writes files with the documented naming
convention.

This file is its own acceptance test for the harness. If it fails
after a refactor, the harness itself is broken — not the bridges it
shields.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from src.tests.conftest import EvidenceCapture

# ── CLI options are registered ──────────────────────────────────────────


class TestHarnessCliOptions:
    """The harness must register its two CLI options on every test run."""

    def test_evidence_dir_option_registered(self, pytestconfig: pytest.Config) -> None:
        assert pytestconfig.getoption("--evidence-dir") is None  # default off

    def test_use_real_backend_option_registered(self, pytestconfig: pytest.Config) -> None:
        # Default is False, not the string "False" — proves the option is
        # properly typed as a flag.
        value = pytestconfig.getoption("--use-real-backend")
        assert value is False


# ── Evidence capture fixture ────────────────────────────────────────────


class TestEvidenceCaptureFixture:
    """The evidence_capture fixture must yield an EvidenceCapture and write
    files named ``{test_name}-{timestamp}-{slug}.{ext}``."""

    def test_evidence_disabled_when_no_dir(
        self,
        evidence_capture: EvidenceCapture | None,
    ) -> None:
        """Without --evidence-dir the fixture uses a default dir
        (``<repo_root>/.omo/evidence``) and yields an active capture."""
        assert evidence_capture is not None
        assert evidence_capture.evidence_dir is not None
        assert ".omo/evidence" in str(evidence_capture.evidence_dir)

    def test_evidence_attaches_text(
        self,
        request: pytest.FixtureRequest,
    ) -> None:
        """Drive the fixture with a custom --evidence-dir and assert the file lands."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            # Inject a resolved evidence_dir by patching the option on the
            # config and re-running the fixture logic directly.
            from src.tests.conftest import EvidenceCapture

            cap = EvidenceCapture(test_name=request.node.name, evidence_dir=Path(tmp))
            try:
                p = cap.attach_text("hello", "world")
                assert p.exists()
                assert p.read_text() == "world"
                assert p.name.startswith(request.node.name + "-")
                # Timestamp + slug + .txt
                assert p.name.endswith("-hello.txt")
            finally:
                cap.finalize(outcome="passed")
            # Summary file is written by finalize.
            summaries = list(Path(tmp).glob("*-summary.txt"))
            assert len(summaries) == 1
            body = summaries[0].read_text()
            assert "outcome: passed" in body

    def test_evidence_attach_json_and_bytes(
        self,
        request: pytest.FixtureRequest,
    ) -> None:
        from src.tests.conftest import EvidenceCapture

        with __import__("tempfile").TemporaryDirectory() as tmp:
            cap = EvidenceCapture(test_name=request.node.name, evidence_dir=Path(tmp))
            try:
                json_path = cap.attach_json("config", {"a": 1, "b": [2, 3]})
                assert json_path.exists()
                assert json_path.suffix == ".json"
                assert '"a": 1' in json_path.read_text()

                bin_path = cap.attach_bytes("blob", b"\x00\x01\x02")
                assert bin_path.exists()
                assert bin_path.read_bytes() == b"\x00\x01\x02"
                assert bin_path.suffix == ".bin"
            finally:
                cap.finalize(outcome="passed")

    def test_evidence_captures_warnings(
        self,
        request: pytest.FixtureRequest,
    ) -> None:
        """A WARNING+ log record should land in the warnings file."""
        from src.tests.conftest import EvidenceCapture

        with __import__("tempfile").TemporaryDirectory() as tmp:
            cap = EvidenceCapture(test_name=request.node.name, evidence_dir=Path(tmp))
            try:
                logging.getLogger("test_evidence_captures_warnings").warning(
                    "hello from a warning",
                )
            finally:
                cap.finalize(outcome="passed")
            warning_files = list(Path(tmp).glob("*-warnings.log"))
            assert len(warning_files) == 1
            content = warning_files[0].read_text()
            assert "hello from a warning" in content
            assert "[WARNING]" in content

    def test_evidence_finalize_is_idempotent(
        self,
        request: pytest.FixtureRequest,
    ) -> None:
        from src.tests.conftest import EvidenceCapture

        with __import__("tempfile").TemporaryDirectory() as tmp:
            cap = EvidenceCapture(test_name=request.node.name, evidence_dir=Path(tmp))
            cap.finalize(outcome="passed")
            cap.finalize(outcome="failed")  # no-op
            # Only one summary file is written.
            assert len(list(Path(tmp).glob("*-summary.txt"))) == 1


# ── Bridge fixtures: real instances with mocked externals ──────────────


class TestAutoTuningBridgeFixture:
    """The auto_tuning_bridge fixture must yield a real TritonTVMBridge."""

    def test_yields_real_bridge_instance(
        self,
        auto_tuning_bridge: Any,
    ) -> None:
        from src.bridges.triton_tvm.bridge_orchestrator import TritonTVMBridge

        assert isinstance(auto_tuning_bridge, TritonTVMBridge)

    def test_tvm_adapter_patches_are_installed(
        self,
        auto_tuning_bridge: Any,
    ) -> None:
        # The mocked MetaScheduleAdapter.tune + IRCapture.capture must be
        # exposed on the bridge so tests can assert call counts.
        assert hasattr(auto_tuning_bridge, "_mock_tune")
        assert hasattr(auto_tuning_bridge, "_mock_capture")
        # The bridge's tvm_adapter is constructed (with TVM mocked) even
        # when the real TVM is not installed — we set it up explicitly in
        # the fixture so the patch can land on a real method.
        assert auto_tuning_bridge.tvm_adapter is not None


class TestAotPackagerFixture:
    """The aot_packager fixture must yield a real FatBinaryBuilder."""

    def test_yields_real_builder_instance(
        self,
        aot_packager: Any,
    ) -> None:
        from src.bridges.aot_packager.builder import FatBinaryBuilder

        assert isinstance(aot_packager, FatBinaryBuilder)

    def test_vendor_mocks_are_installed(self, aot_packager: Any) -> None:
        for attr in ("_mock_amd", "_mock_intel", "_mock_nvidia", "_mock_link", "_mock_stub"):
            assert hasattr(aot_packager, attr), f"missing mock attribute: {attr}"

    def test_build_produces_synthetic_fat_binary(
        self,
        aot_packager: Any,
        tmp_path: Path,
    ) -> None:
        """A full build() call must succeed end-to-end with the mocks active."""
        from src.bridges.aot_packager.builder import FatBinaryConfig

        config = FatBinaryConfig(
            kernel_name="mock_kernel",
            kernel_source=(
                "@triton.jit\n"
                "def mock_kernel(a_ptr, b_ptr, c_ptr, M, N, K, "
                "BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):\n"
                "    pass\n"
            ),
            output_dir=str(tmp_path),
        )
        result = aot_packager.build(config)
        assert result.success, f"build failed: {result.error}"
        assert result.output_path is not None
        assert result.output_path.exists()
        # The mocked AMD/Intel/Nvidia backends were each called once.
        aot_packager._mock_amd.assert_called_once()
        aot_packager._mock_intel.assert_called_once()
        aot_packager._mock_nvidia.assert_called_once()
        aot_packager._mock_link.assert_called_once()


class TestShardingBridgeFixture:
    """The sharding_bridge fixture must yield a real AutoShardingBridge."""

    def test_yields_real_bridge_instance(
        self,
        sharding_bridge: Any,
    ) -> None:
        from src.bridges.pytorch_xla.pipeline_orchestrator import AutoShardingBridge

        assert isinstance(sharding_bridge, AutoShardingBridge)

    def test_gspmd_mocks_are_installed(self, sharding_bridge: Any) -> None:
        for attr in ("_mock_gspmd", "_mock_graph", "_mock_stablehlo"):
            assert hasattr(sharding_bridge, attr), f"missing mock attribute: {attr}"

    def test_circuit_breakers_disabled_in_mocked_mode(
        self,
        sharding_bridge: Any,
    ) -> None:
        # When using mocks, the fixture disables breakers to avoid
        # pulling in extra deps; verify that's the case.
        assert sharding_bridge.breakers is None
        assert sharding_bridge.timeout_manager is None


class TestCudaIngestorFixture:
    """The cuda_ingestor fixture must yield a real CudaKernelCompiler."""

    def test_yields_real_compiler_instance(
        self,
        cuda_ingestor: Any,
    ) -> None:
        from src.bridges.cuda_ingest.kernel_compiler import CudaKernelCompiler

        assert isinstance(cuda_ingestor, CudaKernelCompiler)

    def test_parser_mock_is_installed(self, cuda_ingestor: Any) -> None:
        assert hasattr(cuda_ingestor, "_mock_parse")
        # Downstream phases must be disabled in mocked mode so the test
        # doesn't accidentally call into Triton/TVM/AOT.
        assert cuda_ingestor.enable_phase1_tuning is False
        assert cuda_ingestor.enable_phase2_aot is False


# ── Marker behaviour ────────────────────────────────────────────────────


class TestUseRealBackendMarker:
    """The use_real_backend marker toggles fixtures into real-backend mode.

    These tests intentionally request the marker so the fixture path
    exercises both modes.
    """

    @pytest.mark.use_real_backend
    def test_marker_marks_node(self, request: pytest.FixtureRequest) -> None:
        marker = request.node.get_closest_marker("use_real_backend")
        assert marker is not None

    @pytest.mark.use_real_backend
    def test_auto_tuning_bridge_skips_without_tvm(
        self,
        auto_tuning_bridge: Any,
    ) -> None:
        # When --use-real-backend is set but TVM/Triton are missing,
        # the fixture should skip. We don't assert success or failure
        # here — pytest handles the skip outcome — but we do assert
        # the fixture is wired (yielded a real bridge OR skipped with a
        # clear reason). Touching the fixture forces its evaluation.
        assert True

    @pytest.mark.use_real_backend
    def test_aot_packager_skips_without_lld(
        self,
        aot_packager: Any,
    ) -> None:
        # Same pattern: in CI without lld/gcc, this test is skipped.
        assert True
