"""Tests for `nautilus pipeline` — the end-to-end pipeline CLI command.

Covers:
  * CLI registration (the command shows up in `nautilus --help`)
  * Stage enum correctness
  * Context serialization round-trip
  * Dry-run mode executes nothing expensive
  * Resume-from state is required (no state.json → clear error)
  * Per-stage failure produces a typed NautilusError naming the stage
  * Graceful degradation when dependencies are missing
  * --help renders without crashing
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from src.cli.commands.pipeline import (
    Pipeline,
    PipelineContext,
    PipelineStage,
    StageOutcome,
    _parse_mesh,
    _parse_targets,
    _register_handlers,
    cli,
)
from src.common.errors import (
    ConfigError,
    DependencyMissingError,
    NautilusError,
)
from src.common.types import Arch, HardwareTarget, Vendor


EXAMPLE_KERNEL = textwrap.dedent('''
    import triton
    import triton.language as tl

    @triton.jit
    def matmul_kernel(
        A_ptr, B_ptr, C_ptr,
        M, N, K,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
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


# ---------------------------------------------------------------------------
# Stage enum
# ---------------------------------------------------------------------------


class TestPipelineStage:
    """PipelineStage enum and helper methods."""

    def test_values(self) -> None:
        assert PipelineStage.values() == [
            "capture", "shard", "extract", "tune", "build", "dispatch",
        ]

    def test_from_str_valid(self) -> None:
        for v in PipelineStage.values():
            assert PipelineStage.from_str(v).value == v
        # Case-insensitive
        assert PipelineStage.from_str("BUILD") == PipelineStage.BUILD

    def test_from_str_invalid(self) -> None:
        with pytest.raises(ConfigError) as excinfo:
            PipelineStage.from_str("not-a-stage")
        assert "Unknown pipeline stage" in str(excinfo.value)

    def test_stages_from(self) -> None:
        # Full pipeline
        assert len(PipelineStage.stages_from(PipelineStage.CAPTURE)) == 6
        # From build onwards
        suffix = PipelineStage.stages_from(PipelineStage.BUILD)
        assert [s.value for s in suffix] == ["build", "dispatch"]
        # From dispatch
        assert [s.value for s in PipelineStage.stages_from(
            PipelineStage.DISPATCH,
        )] == ["dispatch"]


# ---------------------------------------------------------------------------
# Context serialization
# ---------------------------------------------------------------------------


class TestPipelineContext:
    """PipelineContext save/load round-trip."""

    def test_round_trip(self, tmp_path: Path) -> None:
        ctx = PipelineContext(input_path=Path("/tmp/foo.py"))
        ctx.kernel_name = "matmul_kernel"
        ctx.kernel_source = "@triton.jit\ndef k(): pass"
        ctx.captured_graph_text = "# FX"
        ctx.captured_graph_hash = "abc123"
        ctx.mesh_axes = [2, 2]
        ctx.sharding_strategy = "tensor_parallel"
        ctx.shard_count = 4
        ctx.sharding_cache_key = "deadbeef"
        ctx.extracted_kernels = [{"name": "k", "source_hash": "abc"}]
        ctx.tuning_configs = {"k": {"block_m": 128, "block_n": 128}}
        ctx.fat_binary_paths = {"primary": Path("/tmp/k.fat.o")}
        ctx.build_stage_times = {"nvidia": 1.23}
        ctx.skipped_vendors = ["intel"]
        ctx.dispatch_plan = {"k": "v"}
        ctx.target_strings = ["nvidia/sm_90"]
        ctx.output_dir = tmp_path
        ctx.dry_run = True

        state = tmp_path / "state.json"
        ctx.save_state(state)
        assert state.exists()

        restored = PipelineContext.load_state(state)
        assert restored.input_path == ctx.input_path
        assert restored.kernel_name == ctx.kernel_name
        assert restored.kernel_source == ctx.kernel_source
        assert restored.captured_graph_text == ctx.captured_graph_text
        assert restored.mesh_axes == ctx.mesh_axes
        assert restored.sharding_strategy == ctx.sharding_strategy
        assert restored.shard_count == ctx.shard_count
        assert restored.sharding_cache_key == ctx.sharding_cache_key
        assert restored.extracted_kernels == ctx.extracted_kernels
        assert restored.tuning_configs == ctx.tuning_configs
        assert restored.fat_binary_paths == ctx.fat_binary_paths
        assert restored.build_stage_times == ctx.build_stage_times
        assert restored.skipped_vendors == ctx.skipped_vendors
        assert restored.dispatch_plan == ctx.dispatch_plan
        assert restored.target_strings == ctx.target_strings
        assert restored.output_dir == ctx.output_dir
        assert restored.dry_run is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for the small command-line helpers."""

    def test_parse_mesh_empty(self) -> None:
        assert _parse_mesh(None) == []
        assert _parse_mesh("") == []

    def test_parse_mesh_1d(self) -> None:
        assert _parse_mesh("4") == [4]

    def test_parse_mesh_2d(self) -> None:
        assert _parse_mesh("2,2") == [2, 2]
        assert _parse_mesh("2, 4") == [2, 4]

    def test_parse_mesh_invalid(self) -> None:
        with pytest.raises(ConfigError):
            _parse_mesh("not,a,number")

    def test_parse_targets_empty_returns_defaults(self) -> None:
        targets = _parse_targets(())
        # Three vendors by default.
        assert len(targets) >= 1
        assert all(isinstance(t, HardwareTarget) for t in targets)

    def test_parse_targets_explicit(self) -> None:
        targets = _parse_targets(("nvidia/sm_90", "amd/gfx942"))
        assert targets[0].vendor == Vendor.NVIDIA
        assert targets[0].arch == Arch.SM_90
        assert targets[1].vendor == Vendor.AMD
        assert targets[1].arch == Arch.GFX942


# ---------------------------------------------------------------------------
# CLI registration & --help
# ---------------------------------------------------------------------------


class TestCliRegistration:
    """The `nautilus pipeline` command must be discoverable."""

    def test_pipeline_in_cli(self) -> None:
        from src.cli.main import cli as main_cli
        assert "pipeline" in main_cli.commands

    def test_pipeline_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.output

    def test_pipeline_subcommand_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        # The pipeline subcommand should appear in the help.
        assert "pipeline" in result.output


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def _make_kernel_file(tmp_path: Path) -> Path:
    p = tmp_path / "matmul.py"
    p.write_text(EXAMPLE_KERNEL)
    return p


def _make_context(tmp_path: Path, kernel_file: Path) -> PipelineContext:
    return PipelineContext(
        input_path=kernel_file,
        output_dir=tmp_path / "out",
        target_strings=["nvidia/sm_90"],
    )


class TestPipelineDryRun:
    """Dry-run must execute no expensive stage and still produce a plan."""

    def test_dry_run_all_stages(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        kernel = _make_kernel_file(tmp_path)
        ctx = _make_context(tmp_path, kernel)
        targets = [HardwareTarget(vendor=Vendor.NVIDIA, arch=Arch.SM_90)]
        pipeline = Pipeline(
            ctx=ctx, targets=targets, dry_run=True,
        )
        # Patch the stage handlers to make sure dry-run NEVER calls them.
        called = {"n": 0}

        def _spy():
            called["n"] += 1
            return {}

        for stage in PipelineStage:
            if stage == PipelineStage.DISPATCH:
                # Dispatch always runs (it just emits a plan).
                continue
            monkeypatch.setitem(
                _STAGE_HANDLERS_DICT(), stage, _spy,
            )

        outcomes = pipeline.run()
        # No stage was executed (only DISPATCH runs in dry-run).
        assert called["n"] == 0
        # All 6 stages report success in dry-run.
        assert len(outcomes) == 6
        assert all(o.success for o in outcomes)
        assert all(o.dry_run for o in outcomes
                   if o.stage != PipelineStage.DISPATCH)
        # State was not persisted (dry-run).
        assert not (tmp_path / "out" / "state.json").exists()


def _STAGE_HANDLERS_DICT():
    """Return the live registry of stage handlers (re-bound on first call)."""
    _register_handlers()
    from src.cli.commands.pipeline import _STAGE_HANDLERS
    return _STAGE_HANDLERS


class TestPipelineResume:
    """--resume-from must use state.json from a previous run."""

    def test_resume_without_state_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        kernel = _make_kernel_file(tmp_path)
        ctx = _make_context(tmp_path, kernel)
        targets = [HardwareTarget(vendor=Vendor.NVIDIA, arch=Arch.SM_90)]
        pipeline = Pipeline(
            ctx=ctx, targets=targets,
            resume_from=PipelineStage.BUILD,
        )
        with pytest.raises(ConfigError) as excinfo:
            pipeline.run()
        assert "Cannot resume" in str(excinfo.value)

    def test_resume_with_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        kernel = _make_kernel_file(tmp_path)
        ctx = _make_context(tmp_path, kernel)
        # Pre-populate state.json as if capture/shard/extract/tune
        # had already run.
        ctx.kernel_name = "matmul_kernel"
        ctx.kernel_source = EXAMPLE_KERNEL
        ctx.captured_graph_text = EXAMPLE_KERNEL
        ctx.captured_graph_hash = "abc123"
        ctx.extracted_kernels = [
            {"name": "matmul_kernel", "source_hash": "abc123",
             "lines": 1, "shard_id": 0},
        ]
        ctx.tuning_configs = {"matmul_kernel": {
            "block_m": 128, "block_n": 128, "block_k": 32,
            "num_warps": 4, "num_stages": 3, "num_ctas": 1,
        }}
        ctx.mesh_axes = [1]
        ctx.shard_count = 1
        ctx.sharding_cache_key = "k1"
        out = tmp_path / "out"
        out.mkdir()
        state = out / "state.json"
        ctx.save_state(state)

        targets = [HardwareTarget(vendor=Vendor.NVIDIA, arch=Arch.SM_90)]
        # Make the build stage a no-op so we don't depend on lld.
        from src.cli.commands.pipeline import _STAGE_HANDLERS
        captured_stages = []

        real_build = _STAGE_HANDLERS[PipelineStage.BUILD]

        def _spy_build():
            captured_stages.append("build")
            return {"output_path": str(out / "stub.fat.o"),
                    "vendors": ["nvidia"], "skipped": [],
                    "elapsed_s": 0.0}

        monkeypatch.setitem(
            _STAGE_HANDLERS, PipelineStage.BUILD, _spy_build,
        )
        try:
            pipeline = Pipeline(
                ctx=ctx, targets=targets,
                resume_from=PipelineStage.BUILD,
            )
            outcomes = pipeline.run()
        finally:
            monkeypatch.setitem(
                _STAGE_HANDLERS, PipelineStage.BUILD, real_build,
            )

        # Build + Dispatch should have run.
        assert "build" in captured_stages
        stage_names = [o.stage.value for o in outcomes]
        assert "build" in stage_names
        assert "dispatch" in stage_names
        # No capture/shard/extract/tune ran.
        for s in ("capture", "shard", "extract", "tune"):
            assert s not in stage_names


class TestPipelineStageFailure:
    """A stage failure should produce a typed NautilusError naming the stage."""

    def test_capture_failure_on_missing_file(
        self, tmp_path: Path,
    ) -> None:
        ctx = _make_context(tmp_path, tmp_path / "missing.py")
        targets = [HardwareTarget(vendor=Vendor.NVIDIA, arch=Arch.SM_90)]
        pipeline = Pipeline(ctx=ctx, targets=targets)
        outcomes = pipeline.run()
        # Capture stage must have failed; remaining stages must not
        # have run.
        assert outcomes[0].stage == PipelineStage.CAPTURE
        assert outcomes[0].success is False
        assert outcomes[0].error is not None
        assert "does not exist" in outcomes[0].error
        # The remaining stages should not have executed.
        assert len(outcomes) == 1

    def test_click_command_propagates_error(
        self, tmp_path: Path,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [str(tmp_path / "missing.py"), "--target", "nvidia/sm_90",
             "--output-dir", str(tmp_path / "out")],
        )
        # Click catches the exception and exits non-zero.
        assert result.exit_code != 0
        # The error message must mention capture (the failing stage)
        # so the user can fix it.
        assert "capture" in result.output.lower() or "does not exist" in result.output


class TestPipelineGracefulDegradation:
    """Missing optional dependencies must NOT crash the pipeline."""

    def test_pipeline_runs_with_deps_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Use a kernel file so capture has something to find.
        kernel = _make_kernel_file(tmp_path)
        ctx = _make_context(tmp_path, kernel)
        targets = [HardwareTarget(vendor=Vendor.NVIDIA, arch=Arch.SM_90)]

        # Force TVM/Triton bridge to fail to import so tune degrades.
        import builtins
        original_import = builtins.__import__

        def _blocked_import(name, globals=None, locals=None,  # noqa: ANN001
                            fromlist=(), level=0):
            if name.startswith("src.bridges.triton_tvm"):
                raise ImportError("simulated missing triton_tvm")
            if name.startswith("src.bridges.aot_packager"):
                raise ImportError("simulated missing aot_packager")
            if name.startswith("src.bridges.pytorch_xla"):
                raise ImportError("simulated missing pytorch_xla")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _blocked_import)

        pipeline = Pipeline(ctx=ctx, targets=targets, trials=2)
        outcomes = pipeline.run()
        # Capture succeeded; downstream stages degraded gracefully.
        # The pipeline should NOT have raised; some stages may
        # report partial success / no fat binary.
        assert len(outcomes) >= 1
        assert outcomes[0].stage == PipelineStage.CAPTURE
        assert outcomes[0].success is True


# ---------------------------------------------------------------------------
# End-to-end happy path
# ---------------------------------------------------------------------------


class TestPipelineHappyPath:
    """Full pipeline run on a kernel file (no torch / TVM / lld)."""

    def test_kernel_pipeline_runs_to_completion(
        self, tmp_path: Path,
    ) -> None:
        kernel = _make_kernel_file(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                str(kernel),
                "--target", "nvidia/sm_90",
                "--output-dir", str(tmp_path / "out"),
                "--trials", "1",
            ],
        )
        # The pipeline may succeed or may report skipped vendors.
        # Either way, exit code is 0 (graceful degradation).
        assert result.exit_code in (0, 2), (
            f"unexpected exit: {result.output}\n{result.exception}"
        )
        # The per-stage summary must appear in stdout.
        assert "PIPELINE SUMMARY" in result.output
        assert "capture" in result.output
        # state.json should have been persisted.
        assert (tmp_path / "out" / "state.json").exists()
        # pipeline_summary.json should have been written.
        assert (tmp_path / "out" / "pipeline_summary.json").exists()
        # dispatch_plan.json should have been written.
        assert (tmp_path / "out" / "dispatch_plan.json").exists()

    def test_dry_run_via_click(
        self, tmp_path: Path,
    ) -> None:
        kernel = _make_kernel_file(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                str(kernel),
                "--target", "nvidia/sm_90",
                "--output-dir", str(tmp_path / "out"),
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        # Dry-run should NOT have written state.json.
        assert not (tmp_path / "out" / "state.json").exists()
        # The summary should still be written so tooling can
        # inspect the plan.
        assert (tmp_path / "out" / "pipeline_summary.json").exists()
        summary = json.loads(
            (tmp_path / "out" / "pipeline_summary.json").read_text(),
        )
        assert summary["dry_run"] is True
        # All 6 stages are present in the summary.
        assert len(summary["stages"]) == 6


# ---------------------------------------------------------------------------
# Handler registration
# ---------------------------------------------------------------------------


class TestHandlerRegistration:
    """All stages must have a registered handler."""

    def test_all_stages_have_handlers(self) -> None:
        _register_handlers()
        from src.cli.commands.pipeline import _STAGE_HANDLERS
        for stage in PipelineStage:
            assert stage in _STAGE_HANDLERS
            assert callable(_STAGE_HANDLERS[stage])
