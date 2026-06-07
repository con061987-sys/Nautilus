"""Pytest configuration for integration tests.

Adds fixtures for skipping GPU-dependent tests when no GPU is available,
and for the optional dependency marker.

Bridge fixtures (``auto_tuning_bridge``, ``aot_packager``, ``sharding_bridge``,
``cuda_ingestor``) instantiate the real bridge classes and wrap their external
service calls with ``unittest.mock`` so the test harness can run without the
underlying hardware SDKs (Triton/TVM, AMD AOTriton, Intel oneAPI, OpenXLA,
tree-sitter). The behaviour can be flipped per-fixture to use the real
backends with the ``--use-real-backend`` CLI option or the per-test
``@pytest.mark.use_real_backend`` marker.

Evidence capture: when ``--evidence-dir=<path>`` is passed on the CLI, every
test gets an :class:`EvidenceCapture` fixture that writes test summaries,
captured warnings, and any explicitly attached artifacts to the directory.
Files are named ``{test_name}-{timestamp}-{slug}.{ext}`` with microsecond
ISO-8601 timestamps so re-runs don't clobber each other.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import shutil
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

_COLLECTION_IMPORT_ERRORS: list[tuple[str, str]] = []


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip tests based on missing prerequisites.

    - @pytest.mark.gpu: skip if no nvidia-smi / rocm-smi / lspci
    - @pytest.mark.cuda: skip if no nvcc
    - @pytest.mark.rocm: skip if no /opt/rocm
    - @pytest.mark.intel: skip if no spirv-val
    - @pytest.mark.requires_deps: skip if torch/tvm/triton not importable
    """
    has_cuda = bool(shutil.which("nvidia-smi") or shutil.which("nvcc"))
    has_rocm = Path("/opt/rocm").exists() or bool(shutil.which("rocm-smi"))
    has_intel = bool(shutil.which("spirv-val")) or bool(shutil.which("ocloc"))
    has_gpu = has_cuda or has_rocm or has_intel

    with contextlib.suppress(ImportError):
        import torch  # noqa: F401
    with contextlib.suppress(ImportError):
        import tvm  # noqa: F401
    with contextlib.suppress(ImportError):
        import triton  # noqa: F401
    with contextlib.suppress(ImportError):
        import torch_xla  # noqa: F401

    for item in items:
        markers = {m.name for m in item.iter_markers()}
        if "gpu" in markers and not has_gpu:
            item.add_marker(
                pytest.mark.skip(reason="No GPU detected (nvidia-smi/rocm-smi/lspci missing)")
            )
        if "cuda" in markers and not has_cuda:
            item.add_marker(pytest.mark.skip(reason="CUDA toolkit not installed"))
        if "rocm" in markers and not has_rocm:
            item.add_marker(pytest.mark.skip(reason="ROCm not installed at /opt/rocm"))
        if "intel" in markers and not has_intel:
            item.add_marker(pytest.mark.skip(reason="Intel oneAPI / SPIRV-Tools not installed"))
        requires_deps = item.get_closest_marker("requires_deps")
        if requires_deps:
            for dep in requires_deps.args:
                try:
                    __import__(dep)
                except ImportError:
                    item.add_marker(pytest.mark.skip(reason=f"requires {dep}"))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Path to the Nautilus repo root."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def any_gpu_available() -> bool:
    return bool(
        shutil.which("nvidia-smi") or Path("/opt/rocm").exists() or shutil.which("spirv-val")
    )


@pytest.fixture
def clean_cache(tmp_path: Path) -> Path:
    """Provide an isolated cache directory so tests don't pollute user cache."""
    return tmp_path


# ── CLI options ──────────────────────────────────────────────────────────


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register --evidence-dir and --use-real-backend CLI options."""
    group = parser.getgroup("harness", "Bridge harness options")
    group.addoption(
        "--evidence-dir",
        action="store",
        default=None,
        help=(
            "Directory to write evidence files (test summaries, captured "
            "warnings, attached artifacts). Created if missing. Default: "
            "evidence capture is disabled."
        ),
    )
    group.addoption(
        "--use-real-backend",
        action="store_true",
        default=False,
        help=(
            "Use real (non-mocked) backends in bridge fixtures. "
            "Default: external services are mocked. Can also be enabled "
            "per-test via @pytest.mark.use_real_backend."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers used by the harness fixtures."""
    config.addinivalue_line(
        "markers",
        "use_real_backend: opt this test into real (non-mocked) bridge backends.",
    )
    config.addinivalue_line(
        "markers",
        "evidence: tag a test that intentionally writes evidence artifacts.",
    )


# ── Evidence capture ─────────────────────────────────────────────────────


def _safe_filename_slug(s: str) -> str:
    """Sanitise a string for use in filenames; collapse runs of underscores."""
    return re.sub(r"_+", "_", re.sub(r"[^\w\-.]", "_", str(s))).strip("_") or "x"


@pytest.fixture(scope="session")
def evidence_dir(request: pytest.FixtureRequest) -> Path | None:
    """Resolved ``--evidence-dir`` or ``None`` when evidence capture is off.

    When the CLI option is not given, the harness looks for a default
    location: ``<repo_root>/.omo/evidence``. This keeps evidence capture
    opt-out (set ``--evidence-dir=`` to an empty string to disable)
    while still landing artifacts somewhere reviewable.

    The directory is created (parents included) when set. The fixture is
    session-scoped so the directory exists for the entire test run; the
    per-test timestamp inside :class:`EvidenceCapture` is what makes
    individual files unique.
    """
    raw = request.config.getoption("--evidence-dir")
    if raw is None:
        repo_root = Path(__file__).resolve().parents[2]
        p = (repo_root / ".omo" / "evidence").expanduser().resolve()
    elif raw == "":
        return None
    else:
        p = Path(raw).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class EvidenceCapture:
    """Per-test evidence capture. Writes files to ``evidence_dir``.

    File naming is ``{test_name}-{timestamp}-{slug}.{ext}`` with a
    microsecond-precision ISO-8601 timestamp. The class also installs
    a logging handler that records WARNING+ records so evidence files
    show the structured log output without the test having to attach
    anything explicitly.
    """

    test_name: str
    evidence_dir: Path
    attachments: list[Path] = field(default_factory=list)
    _timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%dT%H%M%S%f"))
    _log_records: list[logging.LogRecord] = field(default_factory=list)
    _log_handler: logging.Handler | None = None
    _finalized: bool = False

    def __post_init__(self) -> None:
        # Capture WARNING+ from any logger. This is intentionally broad:
        # test output is most useful when it includes real warnings from
        # the bridge code under test, not just what the test attached.
        self._log_handler = _CapturingHandler(self._log_records)
        self._log_handler.setLevel(logging.WARNING)
        logging.getLogger().addHandler(self._log_handler)

    def attach(self, name: str, data: Any, ext: str = "txt", *, slug: str | None = None) -> Path:
        """Write a piece of evidence. Returns the path written.

        ``slug`` defaults to ``name``; provide a custom slug when ``name``
        contains characters that need to be sanitised differently.
        """
        if self._finalized:
            # Finalize is idempotent but post-finalize attach is a no-op so
            # fixture teardown order doesn't blow up.
            return (
                self.evidence_dir
                / f"{self.test_name}-{self._timestamp}-{_safe_filename_slug(slug or name)}.{ext}"
            )
        path = self.evidence_dir / (
            f"{self.test_name}-{self._timestamp}-{_safe_filename_slug(slug or name)}.{ext}"
        )
        if isinstance(data, bytes):
            path.write_bytes(data)
        else:
            path.write_text(str(data), encoding="utf-8")
        self.attachments.append(path)
        return path

    def attach_text(self, name: str, data: str) -> Path:
        return self.attach(name, data, ext="txt")

    def attach_json(self, name: str, data: Any) -> Path:
        return self.attach(name, json.dumps(data, indent=2, default=str), ext="json")

    def attach_bytes(self, name: str, data: bytes, ext: str = "bin") -> Path:
        return self.attach(name, data, ext=ext)

    def finalize(self, outcome: str, exc_info: Any = None) -> None:
        """Write a status file plus the captured log records. Idempotent.

        Order matters: the summary and warnings files are attached BEFORE
        ``_finalized`` is set to True, otherwise :meth:`attach` would
        short-circuit and the summary file would never land on disk.
        """
        if self._finalized:
            return
        # Always remove the log handler so subsequent tests don't see it.
        if self._log_handler is not None:
            with contextlib.suppress(Exception):
                logging.getLogger().removeHandler(self._log_handler)
        try:
            summary_lines = [
                f"test: {self.test_name}",
                f"timestamp: {self._timestamp}",
                f"outcome: {outcome}",
                f"attachments: {len(self.attachments)}",
                f"warning_count: {len(self._log_records)}",
            ]
            for a in self.attachments:
                summary_lines.append(f"  - {a.name}")
            if exc_info is not None and exc_info[0] is not None:
                summary_lines.append(f"exception: {exc_info[0].__name__}: {exc_info[1]}")
            self.attach("summary", "\n".join(summary_lines) + "\n", ext="txt", slug="summary")
            if self._log_records:
                logs = "\n".join(
                    f"[{rec.levelname}] {rec.name}: {rec.getMessage()}" for rec in self._log_records
                )
                self.attach("warnings", logs + "\n", ext="log", slug="warnings")
        except Exception:
            # Evidence capture must NEVER fail the test.
            pass
        # Set _finalized last so the attach() calls above actually run.
        self._finalized = True


class _CapturingHandler(logging.Handler):
    def __init__(self, records: list[logging.LogRecord]) -> None:
        super().__init__(logging.WARNING)
        self._records = records

    def emit(self, record: logging.LogRecord) -> None:
        self._records.append(record)


@pytest.fixture
def evidence_capture(
    request: pytest.FixtureRequest,
    evidence_dir: Path | None,
) -> Iterator[EvidenceCapture | None]:
    """Per-test evidence capture fixture. Yields ``None`` when disabled.

    The capture object is stashed on ``request.node._evidence_capture`` so
    the :func:`pytest_runtest_makereport` hook can finalize it with the
    real test outcome (passed/failed/skipped) after pytest determines it.
    """
    if evidence_dir is None:
        yield None
        return
    cap = EvidenceCapture(test_name=request.node.name, evidence_dir=evidence_dir)
    request.node._evidence_capture = cap
    try:
        yield cap
    finally:
        # The hook below will have finalised the capture for normal cases
        # (passed/failed/skipped). If the fixture teardown itself is
        # what's running, fall back to 'unknown'.
        if not cap._finalized:
            cap.finalize(outcome="unknown")


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[Any]) -> Iterator[Any]:
    """Finalise the per-test evidence capture with the real outcome."""
    outcome = yield
    if call.when != "call" or outcome is None:
        return
    cap = getattr(item, "_evidence_capture", None)
    if cap is None or cap._finalized:
        return
    report = outcome.get_result()
    cap.finalize(outcome=report.outcome)


# ── Bridge fixtures ──────────────────────────────────────────────────────


def _use_real_backend(request: pytest.FixtureRequest) -> bool:
    """True if either --use-real-backend is on or the test has the marker."""
    try:
        if request.config.getoption("--use-real-backend"):
            return True
    except ValueError:
        # Option not registered (e.g. when running tests from a subdirectory
        # whose conftest chain doesn't include src/tests/conftest.py)
        pass
    return request.node.get_closest_marker("use_real_backend") is not None


def _try_import(module_path: str, attr: str | None = None) -> Any:
    """Import ``module_path`` (and optionally grab ``attr``) or raise ImportError.

    Used by bridge fixtures so the user sees a clear ``pytest.skip`` with
    the actual import error rather than an opaque ``ImportError`` deep in
    fixture setup.
    """
    import importlib

    mod = importlib.import_module(module_path)
    if attr is None:
        return mod
    return getattr(mod, attr)


# ── auto_tuning_bridge ───────────────────────────────────────────────────


@pytest.fixture
def auto_tuning_bridge(
    request: pytest.FixtureRequest,
    clean_cache: Path,
) -> Iterator[Any]:
    """TritonTVMBridge with TVM MetaSchedule mocked by default.

    Default mode (mocked):
        Creates a real ``TritonTVMBridge`` and patches
        ``MetaScheduleAdapter.tune`` so calls into the TVM-backed search
        return a synthetic :class:`MappedTuningConfig` and never touch
        the real MetaSchedule. ``IRCapture.capture`` is also patched
        so the IR-capture step doesn't require Triton to be installed.

    Real-backend mode (``--use-real-backend`` or the marker):
        Creates the bridge with ``enable_tvm=True`` and skips the test
        when TVM or Triton are not importable.
    """
    if _use_real_backend(request):
        try:
            triton_tvm_bridge_cls = _try_import(
                "src.bridges.triton_tvm.bridge_orchestrator",
                "TritonTVMBridge",
            )
        except ImportError as exc:
            pytest.skip(f"TritonTVMBridge not importable: {exc}")
        try:
            import triton  # noqa: F401
        except ImportError:
            pytest.skip("Triton not installed (required for real backend)")
        try:
            import tvm  # noqa: F401
        except ImportError:
            pytest.skip("TVM not installed (required for real backend)")
        yield triton_tvm_bridge_cls(cache_dir=str(clean_cache), enable_tvm=True)
        return

    # ── Mocked mode ──────────────────────────────────────────────────
    try:
        triton_tvm_bridge_cls = _try_import(
            "src.bridges.triton_tvm.bridge_orchestrator",
            "TritonTVMBridge",
        )
        mapped_tuning_config_cls = _try_import(
            "src.bridges.triton_tvm.config_mapper",
            "MappedTuningConfig",
        )
        meta_schedule_adapter_cls = _try_import(
            "src.bridges.triton_tvm.metaschedule_adapter",
            "MetaScheduleAdapter",
        )
        ir_capture_cls = _try_import(
            "src.bridges.triton_tvm.ir_capture",
            "IRCapture",
        )
    except ImportError as exc:
        pytest.skip(f"triton_tvm bridge not importable: {exc}")

    from src.common.result import Ok

    bridge = triton_tvm_bridge_cls(cache_dir=str(clean_cache), enable_tvm=False)
    bridge.tvm_adapter = meta_schedule_adapter_cls(cache_dir=str(clean_cache))
    synthetic_cfg = mapped_tuning_config_cls()
    fake_tune = MagicMock(return_value=Ok(synthetic_cfg))
    fake_capture = MagicMock(return_value=MagicMock(is_usable=True))

    with ExitStack() as stack:
        stack.enter_context(patch.object(meta_schedule_adapter_cls, "tune", fake_tune))
        # IRCapture's real entry points are capture_for_source /
        # capture_from_text, not capture. Patch both so callers
        # through either path get the mock.
        stack.enter_context(patch.object(ir_capture_cls, "capture_for_source", fake_capture))
        stack.enter_context(patch.object(ir_capture_cls, "capture_from_text", fake_capture))
        # Expose the mocks for tests that want to assert call counts.
        bridge._mock_tune = fake_tune
        bridge._mock_capture = fake_capture
        yield bridge


# ── aot_packager ─────────────────────────────────────────────────────────


@pytest.fixture
def aot_packager(
    request: pytest.FixtureRequest,
    clean_cache: Path,
) -> Iterator[Any]:
    """FatBinaryBuilder with per-vendor compilers and the linker mocked.

    Default mode (mocked):
        Creates a real ``FatBinaryBuilder`` and patches each vendor
        backend's ``compile_kernel`` plus the linker's ``link_fat_binary``
        to return synthetic ``*CompilationResult`` / ``LinkingResult``
        instances. The C runtime stub is also mocked so the test doesn't
        shell out to gcc/lld.

    Real-backend mode:
        Creates the builder with no patches. The test is skipped when
        lld or gcc aren't on PATH (both are required for the link step).
    """
    if _use_real_backend(request):
        try:
            fat_binary_builder_cls = _try_import(
                "src.bridges.aot_packager.builder",
                "FatBinaryBuilder",
            )
        except ImportError as exc:
            pytest.skip(f"FatBinaryBuilder not importable: {exc}")
        if not shutil.which("lld") and not shutil.which("ld.lld"):
            pytest.skip("lld not on PATH (required for real AOT backend)")
        if not shutil.which("gcc"):
            pytest.skip("gcc not on PATH (required for C runtime stub)")
        yield fat_binary_builder_cls(cache_dir=str(clean_cache))
        return

    # ── Mocked mode ──────────────────────────────────────────────────
    try:
        fat_binary_builder_cls = _try_import(
            "src.bridges.aot_packager.builder",
            "FatBinaryBuilder",
        )
        amd_backend_cls = _try_import(
            "src.bridges.aot_packager.amd_backend",
            "AMDBackend",
        )
        intel_backend_cls = _try_import(
            "src.bridges.aot_packager.intel_backend",
            "IntelBackend",
        )
        nvidia_backend_cls = _try_import(
            "src.bridges.aot_packager.nvidia_backend",
            "NvidiaBackend",
        )
        fat_binary_linker_cls = _try_import(
            "src.bridges.aot_packager.linker",
            "FatBinaryLinker",
        )
        amd_compilation_result_cls = _try_import(
            "src.bridges.aot_packager.amd_backend",
            "AMDCompilationResult",
        )
        intel_compilation_result_cls = _try_import(
            "src.bridges.aot_packager.intel_backend",
            "IntelCompilationResult",
        )
        nvidia_compilation_result_cls = _try_import(
            "src.bridges.aot_packager.nvidia_backend",
            "NvidiaCompilationResult",
        )
        linking_result_cls = _try_import(
            "src.bridges.aot_packager.linker",
            "LinkingResult",
        )
    except ImportError as exc:
        pytest.skip(f"aot_packager bridge not importable: {exc}")

    # Tiny synthetic payloads — enough to look like real binaries without
    # being anywhere near the right size.
    fake_hsaco = b"\x7fELF" + b"\x00" * 12 + b"hsaco-mock"
    fake_spv = b"\x07\x23\x02\x03" + b"\x00" * 8 + b"spv-mock"
    # ptx_text is a string in the real backend; the builder does
    # nvidia_result.ptx_text.encode("utf-8") downstream, so a bytes
    # value here blows up the build with AttributeError.
    fake_ptx_text = "// mock ptx\n.visible .entry mock_kernel() { ret; }\n"
    fake_cubin = b"\x7fELF" + b"\x00" * 12 + b"cubin-mock"

    amd_mock_result = amd_compilation_result_cls(
        success=True,
        arch="gfx942",
        hsaco_bytes=fake_hsaco,
        compilation_time_s=0.001,
    )
    intel_mock_result = intel_compilation_result_cls(
        success=True,
        target="xe_hpg",
        spv_bytes=fake_spv,
        compilation_time_s=0.001,
    )
    nvidia_mock_result = nvidia_compilation_result_cls(
        success=True,
        arch="sm_90",
        ptx_text=fake_ptx_text,
        cubin_bytes=fake_cubin,
        compilation_time_s=0.001,
    )
    # LinkingResult.output_path is normally a Path written by the linker; we
    # point it at the clean_cache so the test can verify the file exists if
    # it wants to.
    linking_output = clean_cache / "mock_kernel.fat.o"
    linking_output.write_bytes(b"\x7fELF" + b"\x00" * 16 + b"fat-mock")
    linking_mock_result = linking_result_cls(
        success=True,
        output_path=linking_output,
        output_size=linking_output.stat().st_size,
        linking_time_s=0.001,
        linker_version="mock-lld",
    )

    builder = fat_binary_builder_cls(cache_dir=str(clean_cache))

    amd_compile = MagicMock(return_value=amd_mock_result)
    intel_compile = MagicMock(return_value=intel_mock_result)
    nvidia_compile = MagicMock(return_value=nvidia_mock_result)
    link_call = MagicMock(return_value=linking_mock_result)
    stub_call = MagicMock(return_value=b"\x7fELF" + b"stub-mock")

    with ExitStack() as stack:
        stack.enter_context(patch.object(amd_backend_cls, "compile_kernel", amd_compile))
        stack.enter_context(patch.object(intel_backend_cls, "compile_kernel", intel_compile))
        stack.enter_context(patch.object(nvidia_backend_cls, "compile_kernel", nvidia_compile))
        stack.enter_context(patch.object(fat_binary_linker_cls, "link_fat_binary", link_call))
        stack.enter_context(patch.object(builder, "_compile_runtime_stub", stub_call))
        # Mock validator (no hardware) so the optional validation stage is a no-op.
        stack.enter_context(
            patch.object(
                builder.validator, "validate", MagicMock(return_value=MagicMock(is_usable=True))
            )
        )
        builder._mock_amd = amd_compile
        builder._mock_intel = intel_compile
        builder._mock_nvidia = nvidia_compile
        builder._mock_link = link_call
        builder._mock_stub = stub_call
        yield builder


# ── sharding_bridge ──────────────────────────────────────────────────────


@pytest.fixture
def sharding_bridge(
    request: pytest.FixtureRequest,
) -> Iterator[Any]:
    """AutoShardingBridge with GSPMD / XLA mocked by default.

    Default mode (mocked):
        Creates a real ``AutoShardingBridge`` with circuit breakers
        disabled (which avoids pulling in extra deps) and patches the
        ``gspmd_runner.run`` method to return a synthetic
        :class:`GSPMDResult`. ``GraphCapture.capture`` and
        ``StableHLOExporter.export_from_captured`` are also patched so
        tests don't need a real PyTorch model or FX exporter.

    Real-backend mode:
        Creates the bridge with circuit breakers enabled. The test is
        skipped when torch, torch_xla, or TVM are missing.
    """
    if _use_real_backend(request):
        try:
            auto_sharding_bridge_cls = _try_import(
                "src.bridges.pytorch_xla.pipeline_orchestrator",
                "AutoShardingBridge",
            )
        except ImportError as exc:
            pytest.skip(f"AutoShardingBridge not importable: {exc}")
        for dep in ("torch", "torch_xla", "tvm"):
            try:
                __import__(dep)
            except ImportError:
                pytest.skip(f"{dep} not installed (required for real sharding backend)")
        yield auto_sharding_bridge_cls(enable_circuit_breakers=True)
        return

    # ── Mocked mode ──────────────────────────────────────────────────
    try:
        auto_sharding_bridge_cls = _try_import(
            "src.bridges.pytorch_xla.pipeline_orchestrator",
            "AutoShardingBridge",
        )
        gspmd_runner_cls = _try_import(
            "src.bridges.pytorch_xla.gspmd_runner",
            "GSPMDRunner",
        )
        graph_capture_cls = _try_import(
            "src.bridges.pytorch_xla.graph_capture",
            "GraphCapture",
        )
        stablehlo_exporter_cls = _try_import(
            "src.bridges.pytorch_xla.stablehlo_export",
            "StableHLOExporter",
        )
        _try_import(
            "src.bridges.pytorch_xla.gspmd_runner",
            "GSPMDResult",
        )
        sharding_spec_cls = _try_import(
            "src.bridges.pytorch_xla.gspmd_runner",
            "ShardingSpec",
        )
    except ImportError as exc:
        pytest.skip(f"pytorch_xla bridge not importable: {exc}")

    fake_sharding_spec = (
        MagicMock(spec=sharding_spec_cls)
        if not isinstance(sharding_spec_cls, type)
        else MagicMock()
    )
    fake_gspmd_result = MagicMock()
    fake_gspmd_result.is_usable = True
    fake_gspmd_result.success = True
    fake_gspmd_result.sharding_spec = fake_sharding_spec
    fake_gspmd_result.sharded_stablehlo = "stablehlo-mock"
    fake_gspmd_result.error = None
    fake_gspmd_result.gspmd_time_s = 0.001
    fake_gspmd_result.cache_hit = False
    fake_gspmd_result.diagnostics = {}
    fake_gspmd_result.tier_used = 1

    fake_graph = MagicMock()
    fake_graph.is_usable = True
    fake_graph.model_name = "mock_model"

    fake_stablehlo = MagicMock()
    fake_stablehlo.is_usable = True
    fake_stablehlo.module_text = "stablehlo-mock"

    gspmd_run = MagicMock(return_value=fake_gspmd_result)
    graph_capture = MagicMock(return_value=fake_graph)
    stablehlo_export = MagicMock(return_value=fake_stablehlo)

    with ExitStack() as stack:
        stack.enter_context(patch.object(gspmd_runner_cls, "run", gspmd_run))
        stack.enter_context(patch.object(graph_capture_cls, "capture", graph_capture))
        stack.enter_context(
            patch.object(stablehlo_exporter_cls, "export_from_captured", stablehlo_export)
        )
        bridge = auto_sharding_bridge_cls(enable_circuit_breakers=False)
        bridge._mock_gspmd = gspmd_run
        bridge._mock_graph = graph_capture
        bridge._mock_stablehlo = stablehlo_export
        yield bridge


# ── cuda_ingestor ────────────────────────────────────────────────────────


@pytest.fixture
def cuda_ingestor(
    request: pytest.FixtureRequest,
) -> Iterator[Any]:
    """CudaKernelCompiler with the tree-sitter parser mocked by default.

    Default mode (mocked):
        Creates a real ``CudaKernelCompiler`` and patches the parser
        (``TreeSitterCudaParser.parse`` and the CudaParser facade) to
        return a synthetic :class:`CudaKernel` so the translation step
        has something to work on without parsing real CUDA.

    Real-backend mode:
        Creates the compiler with no parser patches. The test is
        skipped when tree-sitter or tree-sitter-cpp are missing.
    """
    if _use_real_backend(request):
        try:
            cuda_kernel_compiler = _try_import(
                "src.bridges.cuda_ingest.kernel_compiler",
                "CudaKernelCompiler",
            )
        except ImportError as exc:
            pytest.skip(f"CudaKernelCompiler not importable: {exc}")
        for dep in ("tree_sitter", "tree_sitter_cpp"):
            try:
                __import__(dep)
            except ImportError:
                pytest.skip(f"{dep} not installed (required for real CUDA ingest)")
        yield cuda_kernel_compiler()
        return

    # ── Mocked mode ──────────────────────────────────────────────────
    try:
        cuda_kernel_compiler = _try_import(
            "src.bridges.cuda_ingest.kernel_compiler",
            "CudaKernelCompiler",
        )
        cuda_parser = _try_import(
            "src.bridges.cuda_ingest.parser",
            "CudaParser",
        )
        tree_sitter_cuda_parser = _try_import(
            "src.bridges.cuda_ingest.parser",
            "TreeSitterCudaParser",
        )
        cuda_kernel = _try_import(
            "src.bridges.cuda_ingest.parser",
            "CudaKernel",
        )
    except ImportError as exc:
        pytest.skip(f"cuda_ingest bridge not importable: {exc}")

    # Build a minimal synthetic CudaKernel. We don't care about the AST
    # details — the translator's dispatch runs against CudaStatement
    # metadata, but the parser is the seam that produces a kernel from
    # raw .cu text. A MagicMock satisfies the duck-typed surface.
    fake_kernel = MagicMock()
    fake_kernel.name = "mock_kernel"
    fake_kernel.parameters = []
    fake_kernel.body = []
    fake_kernel.shared_mem = []
    fake_kernel.__class__ = cuda_kernel  # best-effort spec, may no-op

    fake_parse = MagicMock(return_value=[fake_kernel])

    with ExitStack() as stack:
        # The parser's real entry points are parse_file / parse_source,
        # not parse. Patch both so callers using either path get the mock.
        stack.enter_context(patch.object(tree_sitter_cuda_parser, "parse_file", fake_parse))
        stack.enter_context(patch.object(tree_sitter_cuda_parser, "parse_source", fake_parse))
        # CudaParser is the public facade the compiler actually calls;
        # patch its methods so the seam is fully covered.
        stack.enter_context(patch.object(cuda_parser, "parse_file", fake_parse))
        stack.enter_context(patch.object(cuda_parser, "parse_source", fake_parse))
        compiler = cuda_kernel_compiler()
        # Disable downstream phases so the test only exercises the parser
        # seam. Both phases call into Triton/TVM/AOT; we don't want that.
        compiler.enable_phase1_tuning = False
        compiler.enable_phase2_aot = False
        compiler._mock_parse = fake_parse
        yield compiler
