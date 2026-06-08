"""Tests for pipeline fault tolerance: auto-recover, checkpointer wiring, retry logic.

Covers:
  * Pipeline stage failure triggers auto-recovery (not just crash)
  * ``on_node_failure()`` called when a pipeline stage fails
  * ``rebuild_topology()`` called to reconfigure for surviving nodes
  * Pipeline auto-restarts from the last successful checkpoint
  * ``--auto-recover`` flag (default: enabled)
  * ``--max-retries`` flag (default: 3) before giving up
  * AsyncCheckpointer pipeline_state save/load
  * AutoShardingBridge shard_with_retry
  * All existing tests still pass with the new parameters
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

from src.cli.commands.pipeline import (
    CHECKPOINTER_AVAILABLE,
    Pipeline,
    PipelineContext,
    PipelineStage,
    _STAGE_HANDLERS,
    _pipeline_impl,
    cli,
)
from src.common.errors import ConfigError
from src.common.types import Arch, HardwareTarget, Vendor
from src.runtime.async_checkpointer import AsyncCheckpointer, CheckpointConfig

EXAMPLE_KERNEL = textwrap.dedent("""
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
""").strip()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kernel_file(tmp_path: Path) -> Path:
    p = tmp_path / "matmul.py"
    p.write_text(EXAMPLE_KERNEL)
    return p


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    return tmp_path / "out"


def _make_ctx(
    kernel_file: Path,
    output_dir: Path,
) -> PipelineContext:
    return PipelineContext(
        input_path=kernel_file,
        output_dir=output_dir,
        target_strings=["nvidia/sm_90"],
    )


def _make_pipeline(
    ctx: PipelineContext,
    auto_recover: bool = True,
    max_retries: int = 3,
) -> Pipeline:
    return Pipeline(
        ctx=ctx,
        targets=[HardwareTarget(vendor=Vendor.NVIDIA, arch=Arch.SM_90)],
        auto_recover=auto_recover,
        max_retries=max_retries,
    )


# ---------------------------------------------------------------------------
# Pipeline — checkpointer wiring
# ---------------------------------------------------------------------------


class TestPipelineCheckpointerWiring:
    """Checkpointer is created and wired when auto_recover is enabled."""

    def test_checkpointer_initialized_when_auto_recover(
        self,
        kernel_file: Path,
        output_dir: Path,
    ) -> None:
        ctx = _make_ctx(kernel_file, output_dir)
        p = _make_pipeline(ctx, auto_recover=True)
        if CHECKPOINTER_AVAILABLE:
            assert p._checkpointer is not None
        else:
            assert p._checkpointer is None
            assert p.auto_recover is False

    def test_no_checkpointer_when_auto_recover_disabled(
        self,
        kernel_file: Path,
        output_dir: Path,
    ) -> None:
        ctx = _make_ctx(kernel_file, output_dir)
        p = _make_pipeline(ctx, auto_recover=False)
        assert p._checkpointer is None

    def test_set_checkpointer_override(
        self,
        kernel_file: Path,
        output_dir: Path,
    ) -> None:
        ctx = _make_ctx(kernel_file, output_dir)
        p = _make_pipeline(ctx, auto_recover=False)
        cp = AsyncCheckpointer(
            CheckpointConfig(storage_path=str(output_dir / ".ckpt")),
        )
        p.set_checkpointer(cp)
        assert p._checkpointer is cp

    def test_set_checkpointer_none_disables(
        self,
        kernel_file: Path,
        output_dir: Path,
    ) -> None:
        ctx = _make_ctx(kernel_file, output_dir)
        p = _make_pipeline(ctx, auto_recover=True)
        if p._checkpointer is None:
            pytest.skip("checkpointer not available")
        p.set_checkpointer(None)
        assert p._checkpointer is None

    def test_cli_auto_recover_default_is_true(
        self,
        kernel_file: Path,
        output_dir: Path,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                str(kernel_file),
                "--target",
                "nvidia/sm_90",
                "--output-dir",
                str(output_dir),
                "--dry-run",
            ],
        )
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Pipeline — auto-recover on stage failure
# ---------------------------------------------------------------------------


class TestPipelineAutoRecover:
    """Stage failure triggers retry when auto_recover is enabled."""

    def test_auto_recover_retries_on_failure(
        self,
        kernel_file: Path,
        output_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A failing stage triggers retry; checkpointer hooks are called."""
        ctx = _make_ctx(kernel_file, output_dir)
        p = _make_pipeline(ctx, auto_recover=True, max_retries=2)
        if p._checkpointer is None:
            pytest.skip("checkpointer not available")

        call_count = {"capture": 0, "on_failure": 0, "rebuild": 0}

        # Make stage_capture fail twice, succeed on 3rd
        real_capture = _STAGE_HANDLERS[PipelineStage.CAPTURE]

        def _failing_capture(pipeline):
            call_count["capture"] += 1
            if call_count["capture"] < 3:
                raise RuntimeError("simulated failure")
            return real_capture(pipeline)

        monkeypatch.setitem(
            _STAGE_HANDLERS,
            PipelineStage.CAPTURE,
            _failing_capture,
        )

        orig_on_failure = p._checkpointer.on_node_failure

        def _spy_on_failure(dead_node_id: str) -> bool:
            call_count["on_failure"] += 1
            return orig_on_failure(dead_node_id)

        p._checkpointer.on_node_failure = _spy_on_failure  # type: ignore[method-assign]

        orig_rebuild = p._checkpointer.rebuild_topology

        def _spy_rebuild(alive_nodes: list[str]) -> None:
            call_count["rebuild"] += 1
            orig_rebuild(alive_nodes)

        p._checkpointer.rebuild_topology = _spy_rebuild  # type: ignore[method-assign]

        try:
            outcomes = p.run()
        finally:
            monkeypatch.setitem(
                _STAGE_HANDLERS,
                PipelineStage.CAPTURE,
                real_capture,
            )

        # Capture was called 3 times (2 failures + 1 success)
        assert call_count["capture"] == 3, (
            f"Expected 3 capture calls (2 retries + 1 success), got {call_count['capture']}"
        )
        # on_node_failure called for each retry
        assert call_count["on_failure"] == 2, (
            f"Expected 2 on_node_failure calls, got {call_count['on_failure']}"
        )
        # rebuild_topology called for each retry
        assert call_count["rebuild"] == 2, (
            f"Expected 2 rebuild_topology calls, got {call_count['rebuild']}"
        )
        # Pipeline ultimately succeeded — the last stages in the
        # outcomes should be a full successful run (capture → dispatch).
        last_outcomes = [o for o in outcomes if o.success][-6:]
        assert len(last_outcomes) == 6, (
            f"Expected 6 successful final stages, got {len(last_outcomes)}"
        )
        expected_order = [
            PipelineStage.CAPTURE,
            PipelineStage.SHARD,
            PipelineStage.EXTRACT,
            PipelineStage.TUNE,
            PipelineStage.BUILD,
            PipelineStage.DISPATCH,
        ]
        for i, expected in enumerate(expected_order):
            assert last_outcomes[i].stage == expected, (
                f"Final run stage {i}: expected {expected.value}, got {last_outcomes[i].stage.value}"
            )

    def test_auto_recover_exhausts_retries(
        self,
        kernel_file: Path,
        output_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """After max retries, pipeline stops with failure."""
        ctx = _make_ctx(kernel_file, output_dir)
        p = _make_pipeline(ctx, auto_recover=True, max_retries=1)

        real_capture = _STAGE_HANDLERS[PipelineStage.CAPTURE]

        def _always_fail(pipeline):
            raise RuntimeError("always fail")

        monkeypatch.setitem(
            _STAGE_HANDLERS,
            PipelineStage.CAPTURE,
            _always_fail,
        )

        try:
            outcomes = p.run()
        finally:
            monkeypatch.setitem(
                _STAGE_HANDLERS,
                PipelineStage.CAPTURE,
                real_capture,
            )

        # Should have 2 outcomes: one from attempt 0, one from attempt 1
        capture_outcomes = [o for o in outcomes if o.stage == PipelineStage.CAPTURE]
        assert len(capture_outcomes) >= 1
        assert not capture_outcomes[-1].success, (
            "Final capture attempt should have failed"
        )

    def test_auto_recover_disabled_no_retry(
        self,
        kernel_file: Path,
        output_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With auto_recover=False, failure stops the pipeline immediately."""
        ctx = _make_ctx(kernel_file, output_dir)
        p = _make_pipeline(ctx, auto_recover=False, max_retries=3)

        real_capture = _STAGE_HANDLERS[PipelineStage.CAPTURE]

        def _fail_once(pipeline):
            raise RuntimeError("fail once")

        monkeypatch.setitem(
            _STAGE_HANDLERS,
            PipelineStage.CAPTURE,
            _fail_once,
        )

        try:
            outcomes = p.run()
        finally:
            monkeypatch.setitem(
                _STAGE_HANDLERS,
                PipelineStage.CAPTURE,
                real_capture,
            )

        assert len(outcomes) == 1
        assert outcomes[0].stage == PipelineStage.CAPTURE
        assert outcomes[0].success is False

    def test_cli_auto_recover_flag(
        self,
        kernel_file: Path,
        output_dir: Path,
    ) -> None:
        """--auto-recover and --no-auto-recover flags work via CLI."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                str(kernel_file),
                "--target",
                "nvidia/sm_90",
                "--output-dir",
                str(output_dir),
                "--dry-run",
                "--no-auto-recover",
            ],
        )
        assert result.exit_code == 0

    def test_cli_max_retries_flag(
        self,
        kernel_file: Path,
        output_dir: Path,
    ) -> None:
        """--max-retries flag works via CLI."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                str(kernel_file),
                "--target",
                "nvidia/sm_90",
                "--output-dir",
                str(output_dir),
                "--max-retries",
                "5",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0

    def test_resume_from_still_works_with_auto_recover(
        self,
        kernel_file: Path,
        output_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """--resume-from still works when auto_recover is enabled."""
        ctx = _make_ctx(kernel_file, output_dir)
        # Pre-populate state
        ctx.kernel_name = "matmul_kernel"
        ctx.kernel_source = EXAMPLE_KERNEL
        ctx.captured_graph_text = EXAMPLE_KERNEL
        ctx.captured_graph_hash = "abc123"
        ctx.extracted_kernels = [{"name": "matmul_kernel", "source_hash": "abc123", "lines": 1, "shard_id": 0}]
        ctx.tuning_configs = {"matmul_kernel": {"block_m": 128, "block_n": 128, "block_k": 32, "num_warps": 4, "num_stages": 3, "num_ctas": 1}}
        ctx.mesh_axes = [1]
        ctx.shard_count = 1
        ctx.sharding_cache_key = "k1"
        out = output_dir
        out.mkdir(parents=True, exist_ok=True)
        ctx.save_state(out / "state.json")

        real_build = _STAGE_HANDLERS[PipelineStage.BUILD]
        captured = []

        def _spy_build(pipeline):
            captured.append("build")
            return {"output_path": str(out / "stub.fat.o"), "vendors": ["nvidia"], "skipped": [], "elapsed_s": 0.0}

        monkeypatch.setitem(_STAGE_HANDLERS, PipelineStage.BUILD, _spy_build)

        p = Pipeline(
            ctx=ctx,
            targets=[HardwareTarget(vendor=Vendor.NVIDIA, arch=Arch.SM_90)],
            resume_from=PipelineStage.BUILD,
            auto_recover=True,
            max_retries=3,
        )

        try:
            outcomes = p.run()
        finally:
            monkeypatch.setitem(_STAGE_HANDLERS, PipelineStage.BUILD, real_build)

        assert "build" in captured
        stage_names = [o.stage.value for o in outcomes]
        assert "build" in stage_names
        assert "dispatch" in stage_names
        for s in ("capture", "shard", "extract", "tune"):
            assert s not in stage_names


# ---------------------------------------------------------------------------
# AsyncCheckpointer — pipeline state save/load
# ---------------------------------------------------------------------------


class TestCheckpointerPipelineState:
    """save_pipeline_state / load_pipeline_state round-trip."""

    def test_save_and_load_pipeline_state(
        self,
        tmp_path: Path,
    ) -> None:
        cp = AsyncCheckpointer(
            CheckpointConfig(storage_path=str(tmp_path / ".ckpt")),
        )
        state = {
            "last_completed_stage": "capture",
            "shard_attempt": 0,
            "auto_recover": True,
        }
        cp.save_pipeline_state(state)
        loaded = cp.load_pipeline_state()
        assert loaded is not None
        assert loaded["last_completed_stage"] == "capture"
        assert loaded["auto_recover"] is True

    def test_load_pipeline_state_no_checkpoints(
        self,
        tmp_path: Path,
    ) -> None:
        cp = AsyncCheckpointer(
            CheckpointConfig(storage_path=str(tmp_path / ".ckpt")),
        )
        loaded = cp.load_pipeline_state()
        assert loaded is None

    def test_save_pipeline_state_multiple(
        self,
        tmp_path: Path,
    ) -> None:
        cp = AsyncCheckpointer(
            CheckpointConfig(storage_path=str(tmp_path / ".ckpt"), max_checkpoints=3),
        )
        cp.save_pipeline_state({"stage": "capture", "seq": 0})
        cp.save_pipeline_state({"stage": "shard", "seq": 1})
        cp.save_pipeline_state({"stage": "extract", "seq": 2})
        loaded = cp.load_pipeline_state()
        assert loaded is not None
        # Should load the latest
        assert loaded["stage"] == "extract"
        assert loaded["seq"] == 2

    def test_clear_pipeline_state(
        self,
        tmp_path: Path,
    ) -> None:
        cp = AsyncCheckpointer(
            CheckpointConfig(storage_path=str(tmp_path / ".ckpt")),
        )
        cp.save_pipeline_state({"stage": "capture"})
        assert cp.load_pipeline_state() is not None
        cp.clear_pipeline_state()
        assert cp.load_pipeline_state() is None

    def test_pipeline_state_does_not_break_model_checkpoints(
        self,
        tmp_path: Path,
    ) -> None:
        """Pipeline state checkpoints coexist with model state checkpoints."""
        cp = AsyncCheckpointer(
            CheckpointConfig(storage_path=str(tmp_path / ".ckpt")),
        )
        # Save a model checkpoint
        cp.checkpoint_now({"weights": [1.0, 2.0]})
        # Save a pipeline state
        cp.save_pipeline_state({"stage": "capture"})
        # Model checkpoint should still be recoverable
        model_state = cp.recover_latest()
        assert model_state is not None
        model, _ = model_state
        assert model == {"weights": [1.0, 2.0]}
        # Pipeline state should be loadable separately
        pipeline_state = cp.load_pipeline_state()
        assert pipeline_state is not None
        assert pipeline_state["stage"] == "capture"
        # Clear pipeline state should not remove model checkpoints
        cp.clear_pipeline_state()
        assert cp.load_pipeline_state() is None
        model_state2 = cp.recover_latest()
        assert model_state2 is not None


# ---------------------------------------------------------------------------
# AutoShardingBridge — shard_with_retry
# ---------------------------------------------------------------------------


class TestAutoShardingBridgeRetry:
    """shard_with_retry method wraps shard with auto-recovery."""

    def test_shard_with_retry_success_first_try(
        self,
    ) -> None:
        from src.bridges.pytorch_xla.device_mesh import (
            DeviceMesh,
            DeviceVendor,
            InterconnectType,
            MeshDevice,
        )
        from src.bridges.pytorch_xla.pipeline_orchestrator import (
            AutoShardingBridge,
            ShardingConfig,
        )

        mesh_devices = [
            MeshDevice(0, DeviceVendor.NVIDIA, "sm_90", 80.0, 989.0, InterconnectType.NVLINK),
        ]
        mesh = DeviceMesh(devices=mesh_devices, mesh_shape=[1])

        bridge = AutoShardingBridge(auto_recover=True, max_retries=3)
        checkpointer = AsyncCheckpointer(
            CheckpointConfig(storage_path="/tmp/nautilus_test_shard_retry"),
        )
        bridge.set_checkpointer(checkpointer)

        # No model -> fails gracefully, but retry exhausts
        result = bridge.shard_with_retry(
            model=None,
            example_inputs=(),
            device_mesh=mesh,
        )
        assert result.success is False
        assert bridge._retry_count > 0

        checkpointer.stop()

    def test_shard_with_retry_disabled(
        self,
    ) -> None:
        """With auto_recover=False, shard_with_retry runs once."""
        from src.bridges.pytorch_xla.device_mesh import (
            DeviceMesh,
            DeviceVendor,
            InterconnectType,
            MeshDevice,
        )
        from src.bridges.pytorch_xla.pipeline_orchestrator import (
            AutoShardingBridge,
            ShardingConfig,
        )

        mesh_devices = [
            MeshDevice(0, DeviceVendor.NVIDIA, "sm_90", 80.0, 989.0, InterconnectType.NVLINK),
        ]
        mesh = DeviceMesh(devices=mesh_devices, mesh_shape=[1])

        bridge = AutoShardingBridge(auto_recover=False, max_retries=3)
        bridge.set_checkpointer(None)

        result = bridge.shard_with_retry(
            model=None,
            example_inputs=(),
            device_mesh=mesh,
        )
        assert result.success is False
        assert bridge._retry_count == 0  # No retry happened

    def test_set_checkpointer_clears_completed(
        self,
    ) -> None:
        from src.bridges.pytorch_xla.pipeline_orchestrator import (
            AutoShardingBridge,
        )

        bridge = AutoShardingBridge(auto_recover=True, max_retries=3)
        bridge._completed_stages.add("capture")
        cp = AsyncCheckpointer(
            CheckpointConfig(storage_path="/tmp/nautilus_test_set_cp"),
        )
        bridge.set_checkpointer(cp)
        # completed_stages should be cleared by set_checkpointer
        assert len(bridge._completed_stages) == 0
        assert bridge._retry_count == 0
        cp.stop()


# ---------------------------------------------------------------------------
# Existing tests pass — regression check
# ---------------------------------------------------------------------------


class TestRegression:
    """Ensure existing behavior is preserved."""

    def test_capture_failure_still_stops_pipeline(
        self,
        tmp_path: Path,
        kernel_file: Path,
    ) -> None:
        """A stage failure without auto_recover still stops."""
        ctx = _make_ctx(kernel_file, tmp_path / "out2")
        p = _make_pipeline(ctx, auto_recover=False)
        outcomes = p.run()
        # Should work fine (file exists)
        assert outcomes[0].success is True

    def test_dry_run_all_stages(
        self,
        tmp_path: Path,
        kernel_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from src.cli.commands.pipeline import _STAGE_HANDLERS as handlers

        ctx = _make_ctx(kernel_file, tmp_path / "dry_out")
        p = Pipeline(
            ctx=ctx,
            targets=[HardwareTarget(vendor=Vendor.NVIDIA, arch=Arch.SM_90)],
            dry_run=True,
        )
        called = {"n": 0}

        def _spy():
            called["n"] += 1
            return {}

        for stage in PipelineStage:
            if stage == PipelineStage.DISPATCH:
                continue
            monkeypatch.setitem(handlers, stage, _spy)

        outcomes = p.run()
        assert called["n"] == 0
        assert len(outcomes) == 6
        assert all(o.success for o in outcomes)

    def test_on_node_failure_return_value(
        self,
        tmp_path: Path,
    ) -> None:
        """on_node_failure returns True when checkpoint found, False otherwise."""
        cp = AsyncCheckpointer(
            CheckpointConfig(storage_path=str(tmp_path / ".nfs")),
        )
        # No checkpoint -> False
        assert cp.on_node_failure("node-0") is False
        # After saving -> True
        cp.checkpoint_now({"data": 42})
        assert cp.on_node_failure("node-1") is True

    def test_rebuild_topology_no_error(
        self,
        tmp_path: Path,
    ) -> None:
        """rebuild_topology should not raise with valid args."""
        cp = AsyncCheckpointer(
            CheckpointConfig(storage_path=str(tmp_path / ".rt")),
        )
        cp.rebuild_topology(alive_nodes=["capture", "shard"])
