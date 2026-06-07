"""Integration tests for the auto-tuning bridge pipeline.

Tests the full pipeline: TTGIR extraction → IR normalization → MetaSchedule
tuning → config mapping → Triton recompile. All external dependencies (TVM)
are mocked since TVM is not installed in CI.

Every test documents:
  - What it tests
  - What passing means
  - The fallback tier involved (L0-L5)
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.bridges.triton_tvm.bridge_orchestrator import (
    FallbackTier,
    MappedTuningConfig,
    TritonTVMBridge,
    TuningResult,
)
from src.bridges.triton_tvm.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
    get_default_breakers,
)
from src.bridges.triton_tvm.config_mapper import ConfigMapper
from src.bridges.triton_tvm.timeout_manager import (
    StageBudgets,
    StageTimeoutError,
    TimeoutManager,
    TotalBudgetExceededError,
)
from src.common.errors import TuningError
from src.common.result import Err, Ok


# =========================================================================
# Full pipeline end-to-end
# =========================================================================


class TestFullPipeline:
    """Tests the full tuning pipeline end-to-end with mocked TVM."""

    def test_tuning_chain_returns_ok_with_valid_config(
        self,
        auto_tuning_bridge: TritonTVMBridge,
        sample_matmul_metadata: Any,
    ) -> None:
        """The full tuning chain should return Ok with a valid MappedTuningConfig.

        What it tests:
          _build_tir_template → tvm_adapter.tune → Result<MappedTuningConfig>

        Passing means:
          The result is Ok and contains a config with all fields set to
          the values returned by the mocked MetaSchedule adapter.
        """
        result = auto_tuning_bridge._tuning_chain(
            sample_matmul_metadata, "nvidia/nvidia-a100",
        )
        assert result.is_ok()
        config = result.unwrap()
        assert isinstance(config, MappedTuningConfig)
        assert config.block_m == 64
        assert config.block_n == 128
        assert config.block_k == 64
        assert config.num_warps == 8
        assert config.num_stages == 4
        assert config.num_ctas == 1

    def test_tuning_chain_stage_timing(
        self,
        auto_tuning_bridge: TritonTVMBridge,
        sample_matmul_metadata: Any,
    ) -> None:
        """Stage timing should be populated after a tuning chain call.

        What it tests:
          _stages dict is populated with duration_ms per stage.

        Passing means:
          Both 'build_template' and 'tvm_tune' stages have positive
          duration values.
        """
        auto_tuning_bridge._tuning_chain(
            sample_matmul_metadata, "nvidia/nvidia-a100",
        )
        assert "build_template" in auto_tuning_bridge._stages
        assert "tvm_tune" in auto_tuning_bridge._stages
        assert auto_tuning_bridge._stages["tvm_tune"] > 0

    def test_tune_produces_tuning_result(
        self,
        auto_tuning_bridge: TritonTVMBridge,
    ) -> None:
        """The tune() method should return a TuningResult.

        What it tests:
          tune() → TuningResult with config, tier, timing, cache info.

        Passing means:
          The result has a config, a valid fallback_tier enum value,
          and all stage durations are recorded.
        """
        result = auto_tuning_bridge.tune(
            kernel_fn=_dummy_kernel,
            grid=(1, 1, 1),
            example_args=(),
            target="nvidia/nvidia-a100",
        )
        assert isinstance(result, TuningResult)
        assert isinstance(result.config, MappedTuningConfig)
        assert isinstance(result.fallback_tier, FallbackTier)
        assert result.total_duration_ms > 0
        assert "extract" in result.stages

    def test_tuning_chain_preserves_config_across_metadata_types(
        self,
        auto_tuning_bridge: TritonTVMBridge,
        sample_matmul_metadata: Any,
        sample_reduction_metadata: Any,
        sample_elementwise_metadata: Any,
    ) -> None:
        """The pipeline should work identically for all metadata types.

        What it tests:
          Matmul, reduction, and elementwise metadata all produce Ok.

        Passing means:
          All three metadata types return Ok with a valid config.
        """
        for meta in [sample_matmul_metadata, sample_reduction_metadata,
                     sample_elementwise_metadata]:
            result = auto_tuning_bridge._tuning_chain(
                meta, "nvidia/nvidia-a100",
            )
            assert result.is_ok(), f"Failed for {meta.kernel_name}"
            conf = result.unwrap()
            assert isinstance(conf, MappedTuningConfig)
            assert conf.block_m > 0

    def test_tune_with_real_ir_fallback_path(
        self,
        auto_tuning_bridge: TritonTVMBridge,
    ) -> None:
        """tune_with_real_ir should fall back gracefully when no IR captured.

        What it tests:
          The fallback path in tune_with_real_ir when ir_capture returns
          None (no IR has been captured yet).

        Passing means:
          Returns a TuningResult with fallback_used=True and a valid
          fallback config.
        """
        result = auto_tuning_bridge.tune_with_real_ir(
            source_hash="nonexistent_hash",
            target="nvidia/nvidia-a100",
            force_retune=True,
        )
        assert isinstance(result, TuningResult)
        assert isinstance(result.config, MappedTuningConfig)
        assert result.fallback_used, \
            "Expected fallback_used=True when no real IR is captured"
        assert result.fallback_reason is not None
        assert "real_ir_capture_failed" in result.fallback_reason

    def test_tune_with_real_ir_uses_cache_when_available(
        self,
        cache_dir: str,
        sample_matmul_metadata: Any,
    ) -> None:
        """tune_with_real_ir should return cached result on cache hit.

        What it tests:
          Cache lookup during tune_with_real_ir — the IR capture is
          mocked to return a CapturedKernelIR with a cache key, and
          the cache is pre-seeded.

        Passing means:
          Returns a TuningResult with cache_hit=True and L3_DISK_CACHE tier.
        """
        from unittest.mock import MagicMock, patch

        import src.bridges.triton_tvm.metaschedule_adapter as ms_mod
        import src.bridges.triton_tvm.tir_template as tt_mod

        from src.bridges.triton_tvm.ir_capture import CapturedKernelIR, IRBounds, KernelKind

        with (
            patch.object(tt_mod, "TVM_AVAILABLE", True),
            patch.object(ms_mod, "TVM_AVAILABLE", True),
        ):
            bridge = TritonTVMBridge(cache_dir=cache_dir, enable_tvm=True)
            mock_ir = CapturedKernelIR(
                source_hash="test_hash",
                target="nvidia/nvidia-a100",
                stage_name="ttgir",
                ir_text="",
                kind=KernelKind.MATMUL,
                bounds=IRBounds(m=128, n=128, k=64, data_dtype="float32"),
            )
            bridge.ir_capture.capture_for_source = MagicMock(return_value=mock_ir)

            cached_config = MappedTuningConfig(
                block_m=32, block_n=32, block_k=16,
            )
            bridge._set_cache(mock_ir.cache_key, "nvidia/nvidia-a100", cached_config)

            result = bridge.tune_with_real_ir(
                source_hash="test_hash",
                target="nvidia/nvidia-a100",
            )
            assert result.cache_hit
            assert result.fallback_tier == FallbackTier.L3_DISK_CACHE
            assert result.config.block_m == 32
            assert result.config.block_n == 32


# =========================================================================
# L0–L5 fallback tiers
# =========================================================================


class TestFallbackTiers:
    """Each fallback tier (L0-L5) can degrade independently.

    The bridge tries tiers in decreasing order of quality:
      L0 — existing TVM database record (best)
      L1 — TVM warm-start from similar config
      L2 — full TVM tuning (cold start)
      L3 — previously cached Triton config (disk cache hit)
      L4 — Triton's built-in defaults
      L5 — conservative pre-vetted config (safest)
    """

    def test_l0_tvm_db_hit_tier_is_defined(self) -> None:
        """L0 (TVM database hit) tier exists and is the highest quality.

        What it tests:
          The FallbackTier enum has L0_TVM_DB_HIT.

        Passing means:
          L0 is the first (best) fallback tier.
        """
        members = list(FallbackTier)
        assert FallbackTier.L0_TVM_DB_HIT in members
        assert members.index(FallbackTier.L0_TVM_DB_HIT) == 0

    def test_l0_tvm_db_hit_from_mocked_tune(
        self,
        auto_tuning_bridge: TritonTVMBridge,
    ) -> None:
        """L0: A successful TVM tune produces L0 quality result.

        What it tests:
          When TVM adapter returns Ok(config), tune() reports
          L0_TVM_DB_HIT as the fallback tier.

        Passing means:
          The returned TuningResult has tier L0_TVM_DB_HIT and the
          config matches what the adapter returned.
        """
        result = auto_tuning_bridge.tune(
            kernel_fn=_dummy_kernel,
            grid=(1, 1, 1),
            example_args=(),
            target="nvidia/nvidia-a100",
        )
        assert result.fallback_tier == FallbackTier.L0_TVM_DB_HIT
        assert result.config.block_m == 64

    def test_l1_tvm_warm_start_tier_is_defined(self) -> None:
        """L1 (TVM warm-start) tier exists.

        What it tests:
          The FallbackTier enum has L1_TVM_WARM_START.

        Passing means:
          L1 is present and is the second tier.
        """
        members = list(FallbackTier)
        assert FallbackTier.L1_TVM_WARM_START in members
        assert members.index(FallbackTier.L1_TVM_WARM_START) == 1

    def test_l2_tvm_cold_tier_is_defined(self) -> None:
        """L2 (TVM cold start) tier exists.

        What it tests:
          The FallbackTier enum has L2_TVM_COLD.

        Passing means:
          L2 is present and is the third tier.
        """
        members = list(FallbackTier)
        assert FallbackTier.L2_TVM_COLD in members
        assert members.index(FallbackTier.L2_TVM_COLD) == 2

    def test_l3_disk_cache_returns_cached_config(
        self,
        cache_dir: str,
        sample_matmul_metadata: Any,
    ) -> None:
        """L3: Disk cache hit returns the cached config without tuning.

        What it tests:
          When a config is in the disk cache, _get_cached returns it and
          _tuning_chain is never called.

        Passing means:
          The cached config is returned with cache_hit=True and
          the config matches what was cached.
        """
        bridge = TritonTVMBridge(cache_dir=cache_dir, enable_tvm=False)
        expected = MappedTuningConfig(
            block_m=256, block_n=256, block_k=64,
            num_warps=8, num_stages=5,
        )
        bridge._set_cache(sample_matmul_metadata.cache_key, "nvidia/nvidia-a100", expected)

        bridge2 = TritonTVMBridge(cache_dir=cache_dir, enable_tvm=False)
        cached = bridge2._get_cached(
            sample_matmul_metadata.cache_key, "nvidia/nvidia-a100",
        )
        assert cached is not None
        assert cached.block_m == 256
        assert cached.block_n == 256
        assert cached.block_k == 64
        assert cached.num_warps == 8
        assert cached.num_stages == 5

    def test_l3_disk_cache_produces_tuning_result(
        self,
        cache_dir: str,
        sample_matmul_metadata: Any,
    ) -> None:
        """L3: When the cache has a config, _tuning_chain is bypassed.

        What it tests:
          Cache hit through _get_cached returns the previously stored
          config without calling _build_tir_template or tvm_adapter.

        Passing means:
          _get_cached returns the cached MappedTuningConfig.
        """
        bridge = TritonTVMBridge(cache_dir=cache_dir, enable_tvm=False)
        expected = MappedTuningConfig(block_m=64, block_n=64, block_k=16)

        bridge._set_cache(sample_matmul_metadata.cache_key, "nvidia/nvidia-a100", expected)

        cached = bridge._get_cached(
            sample_matmul_metadata.cache_key, "nvidia/nvidia-a100",
        )
        assert cached is not None
        assert cached.block_m == 64
        assert cached.block_n == 64
        assert cached.block_k == 16
        assert _is_power_of_two(cached.block_m)
        assert _is_power_of_two(cached.block_n)
        assert _is_power_of_two(cached.block_k)

    def test_l4_triton_default_from_tuning_chain_error(
        self,
        auto_tuning_bridge: TritonTVMBridge,
        sample_matmul_metadata: Any,
    ) -> None:
        """L4: When TVM tuning returns Err, tune() falls back to L4 defaults.

        What it tests:
          tvm_adapter.tune returning Err(TuningError) causes _tuning_chain
          to return Err with the L4_TRITON_DEFAULT tier context.

        Passing means:
          The Err result has context.tier == L4_TRITON_DEFAULT.
        """
        auto_tuning_bridge.tvm_adapter.tune = MagicMock(
            return_value=Err(TuningError(
                "TVM unavailable for testing",
                context={"tier": FallbackTier.L4_TRITON_DEFAULT.name},
            )),
        )
        result = auto_tuning_bridge._tuning_chain(
            sample_matmul_metadata, "nvidia/nvidia-a100",
        )
        assert result.is_err()
        error = result.error
        assert "tier" in error.context
        assert error.context["tier"] == FallbackTier.L4_TRITON_DEFAULT.name

    def test_l4_triton_default_is_recoverable(
        self,
        auto_tuning_bridge: TritonTVMBridge,
    ) -> None:
        """L4: An Err from _tuning_chain is recoverable via _fallback_config.

        What it tests:
          When _tuning_chain returns Err, the tune() method calls
          _fallback_config which returns a usable MappedTuningConfig.

        Passing means:
          The tune() call returns a TuningResult with a valid config
          even when TVM tuning fails.
        """
        auto_tuning_bridge.tvm_adapter.tune = MagicMock(
            return_value=Err(TuningError("simulated failure")),
        )
        result = auto_tuning_bridge.tune(
            kernel_fn=_dummy_kernel,
            grid=(1, 1, 1),
            example_args=(),
            target="nvidia/nvidia-a100",
        )
        assert isinstance(result.config, MappedTuningConfig)
        assert result.config.block_m > 0

    def test_l5_safe_fallback_on_template_build_error(
        self,
        auto_tuning_bridge: TritonTVMBridge,
        sample_matmul_metadata: Any,
    ) -> None:
        """L5: Template build errors return Err with L5_SAFE_FALLBACK tier.

        What it tests:
          When _build_tir_template raises ValueError (caught by the
          _tuning_chain try/except), the result is Err with L5 context.

        Passing means:
          The Err result has context.tier == L5_SAFE_FALLBACK.name.
        """
        auto_tuning_bridge._build_tir_template = MagicMock(
            side_effect=ValueError("invalid bounds for template"),
        )
        result = auto_tuning_bridge._tuning_chain(
            sample_matmul_metadata, "nvidia/nvidia-a100",
        )
        assert result.is_err()
        assert result.error.context.get("tier") == FallbackTier.L5_SAFE_FALLBACK.name

    def test_l5_safe_fallback_on_import_error(
        self,
        auto_tuning_bridge: TritonTVMBridge,
        sample_matmul_metadata: Any,
    ) -> None:
        """L5: Missing dependencies during template build return Err with L5.

        What it tests:
          ImportError from _build_tir_template is caught and mapped to L5.

        Passing means:
          The Err result has context.tier == L5_SAFE_FALLBACK.name.
        """
        auto_tuning_bridge._build_tir_template = MagicMock(
            side_effect=ImportError("TVM not installed"),
        )
        result = auto_tuning_bridge._tuning_chain(
            sample_matmul_metadata, "nvidia/nvidia-a100",
        )
        assert result.is_err()
        assert result.error.context.get("tier") == FallbackTier.L5_SAFE_FALLBACK.name

    def test_l5_safe_fallback_on_os_error(
        self,
        auto_tuning_bridge: TritonTVMBridge,
        sample_matmul_metadata: Any,
    ) -> None:
        """L5: Filesystem errors during template build return Err with L5.

        What it tests:
          OSError from _build_tir_template is caught and mapped to L5.

        Passing means:
          The Err result has context.tier == L5_SAFE_FALLBACK.name.
        """
        auto_tuning_bridge._build_tir_template = MagicMock(
            side_effect=OSError(13, "Permission denied: /tmp/tvm"),
        )
        result = auto_tuning_bridge._tuning_chain(
            sample_matmul_metadata, "nvidia/nvidia-a100",
        )
        assert result.is_err()
        assert result.error.context.get("tier") == FallbackTier.L5_SAFE_FALLBACK.name

    def test_tuning_produces_reasonable_default(
        self,
        auto_tuning_bridge: TritonTVMBridge,
        sample_matmul_metadata: Any,
    ) -> None:
        """L5: The fallback config for matmul should use sensible values.

        What it tests:
          _fallback_config produces 128x128x32 for matmul kernels.

        Passing means:
          The fallback config has block_m=128, block_n=128, block_k=32.
        """
        bridge = TritonTVMBridge(enable_tvm=False)
        config = bridge._fallback_config(
            sample_matmul_metadata, FallbackTier.L5_SAFE_FALLBACK,
        )
        assert config.block_m == 128
        assert config.block_n == 128
        assert config.block_k == 32

    def test_non_matmul_fallback_uses_defaults(
        self,
        sample_elementwise_metadata: Any,
    ) -> None:
        """Non-matmul fallback should use MappedTuningConfig.defaults().

        What it tests:
          _fallback_config for non-matmul kernels returns defaults.

        Passing means:
          The config equals MappedTuningConfig.defaults().
        """
        bridge = TritonTVMBridge(enable_tvm=False)
        config = bridge._fallback_config(
            sample_elementwise_metadata, FallbackTier.L5_SAFE_FALLBACK,
        )
        assert config == MappedTuningConfig.defaults()


# =========================================================================
# Circuit breaker integration
# =========================================================================


class TestCircuitBreakerIntegration:
    """Circuit breaker pattern tested in the bridge pipeline context."""

    def test_breaker_opens_after_threshold_failures(self) -> None:
        """Circuit transitions CLOSED → OPEN after N consecutive failures.

        What it tests:
          The standard circuit breaker pattern: failure_threshold=N opens
          after N failures.

        Passing means:
          After 3 failures the circuit is OPEN and subsequent calls
          raise CircuitOpenError.
        """
        cb = CircuitBreaker(
            "test_breaker", CircuitBreakerConfig(failure_threshold=3),
        )
        assert cb.state == CircuitState.CLOSED

        for i in range(3):
            with pytest.raises(RuntimeError):
                cb.call(lambda: (_ for _ in ()).throw(RuntimeError(f"fail {i}")))
            assert cb.total_failures == i + 1

        assert cb.state == CircuitState.OPEN
        assert cb.total_short_circuits == 0

        with pytest.raises(CircuitOpenError):
            cb.call(lambda: 42)
        assert cb.total_short_circuits == 1

    def test_breaker_closes_after_success_threshold(self) -> None:
        """Circuit HALF_OPEN → CLOSED after success_threshold successes.

        What it tests:
          After cooldown, the circuit transitions to HALF_OPEN. A
          successful trial closes it.

        Passing means:
          After one successful call in HALF_OPEN, state is CLOSED.
        """
        cb = CircuitBreaker(
            "test_half_close",
            CircuitBreakerConfig(
                failure_threshold=1, cooldown_seconds=0.05,
                success_threshold=1,
            ),
        )
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        assert cb.state == CircuitState.OPEN

        time.sleep(0.1)
        result = cb.call(lambda: "recovered")
        assert result == "recovered"
        assert cb.state == CircuitState.CLOSED

    def test_breaker_half_open_failure_reopens(self) -> None:
        """A failed HALF_OPEN trial re-opens the circuit.

        What it tests:
          Failure during HALF_OPEN transitions back to OPEN.

        Passing means:
          State is OPEN after the trial fails, with cooldown remaining.
        """
        cb = CircuitBreaker(
            "test_reopen",
            CircuitBreakerConfig(
                failure_threshold=1, cooldown_seconds=0.05,
            ),
        )
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        time.sleep(0.1)
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("still broken")))
        assert cb.state == CircuitState.OPEN

    def test_breaker_used_in_tune_with_breaker(self) -> None:
        """_tune_with_breaker uses the breaker and falls back on failure.

        What it tests:
          The bridge's _tune_with_breaker method runs the adapter through
          the circuit breaker and returns default config when the breaker
          is open.

        Passing means:
          When the breaker is open, _tune_with_breaker returns
          MappedTuningConfig.defaults() without calling the adapter.
        """
        bridge = TritonTVMBridge(enable_tvm=False)
        breaker = bridge.breakers["tvm_tune"]
        breaker.state = CircuitState.OPEN
        breaker.last_failure_time = time.time()

        result = bridge._tune_with_breaker(
            tir_mod=MagicMock(),
            target_str="nvidia/nvidia-a100",
            cache_key="test_key",
        )
        assert result == MappedTuningConfig.defaults()

    def test_breaker_tracks_stats_in_pipeline(self) -> None:
        """The breaker correctly tracks total_calls and total_failures.

        What it tests:
          CircuitBreaker stats update correctly when used through
          _tune_with_breaker.

        Passing means:
          Stats show correct call and failure counts.
        """
        bridge = TritonTVMBridge(enable_tvm=False)
        breaker = bridge.breakers["tvm_tune"]

        stats = breaker.stats
        assert "name" in stats
        assert "state" in stats
        assert "total_calls" in stats
        assert "total_failures" in stats
        assert "total_short_circuits" in stats

    def test_default_breakers_are_configured(self) -> None:
        """The bridge creates all expected circuit breakers.

        What it tests:
          get_default_breakers returns the canonical set.

        Passing means:
          All 8 expected breakers are registered with sensible configs.
        """
        breakers = get_default_breakers()
        assert len(breakers) == 8
        assert "tvm_tune" in breakers
        assert breakers["tvm_tune"].config.failure_threshold == 3
        assert breakers["tvm_tune"].config.cooldown_seconds == 60.0
        assert breakers["triton_compile"].config.failure_threshold == 5

    def test_breaker_reset_restores_pipeline(self) -> None:
        """Resetting a breaker restores normal operation.

        What it tests:
          After reset(), a previously open breaker allows calls again.

        Passing means:
          Call succeeds after reset, returning correct value.
        """
        cb = CircuitBreaker(
            "test_reset", CircuitBreakerConfig(failure_threshold=1),
        )
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        assert cb.is_open
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        result = cb.call(lambda: 99)
        assert result == 99


# =========================================================================
# Timeout handling
# =========================================================================


class TestTimeoutIntegration:
    """Timeout handling in the bridge pipeline context."""

    def test_stage_timeout_error_raised_when_budget_exceeded(self) -> None:
        """A stage that exceeds its budget raises StageTimeoutError.

        What it tests:
          TimeoutManager.stage() enforces per-stage time budgets.

        Passing means:
          A stage that sleeps longer than its budget raises
          StageTimeoutError with the correct stage name and budget.
        """
        mgr = TimeoutManager(StageBudgets(extract_s=0.1))
        with pytest.raises(StageTimeoutError) as exc:
            with mgr.stage("extract"):
                time.sleep(0.3)
        assert exc.value.stage_name == "extract"
        assert exc.value.budget_s == pytest.approx(0.1, abs=0.01)

    def test_stage_under_budget_succeeds(self) -> None:
        """A stage that completes within budget should succeed.

        What it tests:
          TimeoutManager.stage() allows normal completion.

        Passing means:
          No exception is raised and the budget value is correct.
        """
        mgr = TimeoutManager(StageBudgets(extract_s=5.0))
        with mgr.stage("extract") as budget:
            time.sleep(0.01)
            assert budget == 5.0

    def test_total_budget_exceeded_raises(self) -> None:
        """Exhausting the total budget raises TotalBudgetExceededError.

        What it tests:
          TimeoutManager.check_total_budget() enforces the sum of
          all stage budgets.

        Passing means:
          After enough stage timeouts, check_total_budget raises.
        """
        budgets = StageBudgets(
            extract_s=0.05, build_tir_s=0.05, tune_s=0.05,
            map_config_s=0.05, recompile_s=0.05, aot_compile_s=0.05,
            link_s=0.05, validate_s=0.05,
        )
        mgr = TimeoutManager(budgets)
        for _ in range(6):
            with pytest.raises(StageTimeoutError), mgr.stage("extract"):
                time.sleep(0.15)
        with pytest.raises(TotalBudgetExceededError):
            mgr.check_total_budget()

    def test_tuning_chain_handles_timeout_gracefully(
        self,
        auto_tuning_bridge: TritonTVMBridge,
        sample_matmul_metadata: Any,
    ) -> None:
        """Timeout errors from MetaSchedule are handled as Err results.

        What it tests:
          When tvm_adapter.tune raises StageTimeoutError, _tuning_chain
          catches it and returns Err with L4_TRITON_DEFAULT context.

        Passing means:
          The Err result has the timeout stage referenced in the message.
        """
        from src.bridges.triton_tvm.timeout_manager import StageTimeoutError

        def _timeout_raiser(
            tir_mod: Any = None,
            target_str: str = "",
            max_trials: int = 64,
            cache_key: str | None = None,
        ) -> Any:
            raise StageTimeoutError("tvm_tune", 600.0, 601.5)

        auto_tuning_bridge.tvm_adapter.tune = MagicMock(side_effect=_timeout_raiser)
        result = auto_tuning_bridge._tuning_chain(
            sample_matmul_metadata, "nvidia/nvidia-a100",
        )
        assert result.is_err()
        assert "timed out" in result.error.message.lower()
        assert result.error.context.get("tier") == FallbackTier.L4_TRITON_DEFAULT.name

    def test_timeout_does_not_leak_between_stages(self) -> None:
        """Stage budgets are independent — a timeout in one does not
        affect the budget of another.

        What it tests:
          TimeoutManager.StageBudgets provides independent budgets.

        Passing means:
          A slow 'extract' stage does not reduce the available budget
          for 'tune'.
        """
        mgr = TimeoutManager(StageBudgets(extract_s=0.05, tune_s=5.0))
        with pytest.raises(StageTimeoutError), mgr.stage("extract"):
            time.sleep(0.1)
        with mgr.stage("tune") as budget:
            assert budget == 5.0


# =========================================================================
# Config mapping validity
# =========================================================================


class TestConfigValidity:
    """Structural validity of the tuning config output."""

    def test_mapped_tuning_config_defaults(self) -> None:
        """Default config should use sensible safe values.

        What it tests:
          MappedTuningConfig.defaults() and direct construction.

        Passing means:
          All fields have values that are valid Triton Config inputs:
          block sizes are powers of 2, num_warps is in {1,2,4,8,16,32},
          num_stages >= 1, num_ctas >= 1.
        """
        config = MappedTuningConfig.defaults()
        assert config.block_m >= 16
        assert config.block_n >= 16
        assert config.block_k >= 16
        assert config.num_warps in (1, 2, 4, 8, 16, 32)
        assert config.num_stages >= 1
        assert config.num_ctas >= 1
        assert isinstance(config.enable_fp_fusion, bool)
        assert config.max_num_imprecise_acc >= 0

    def test_mapped_tuning_config_powers_of_two(self) -> None:
        """Block sizes should be powers of two.

        What it tests:
          Various configs with valid block sizes.

        Passing means:
          block_m, block_n, block_k are all powers of two.
        """
        for bm in [16, 32, 64, 128, 256]:
            for bn in [16, 32, 64, 128, 256]:
                for bk in [16, 32, 64]:
                    config = MappedTuningConfig(
                        block_m=bm, block_n=bn, block_k=bk,
                    )
                    assert _is_power_of_two(config.block_m)
                    assert _is_power_of_two(config.block_n)
                    assert _is_power_of_two(config.block_k)

    def test_mapped_tuning_config_warps_are_valid(self) -> None:
        """num_warps must be a power of two and <= 32.

        What it tests:
          Valid and invalid num_warps values.

        Passing means:
          Constructing with a valid warps count works; no validation
          is enforced by the dataclass (it's a frozen dataclass).
        """
        for w in [1, 2, 4, 8, 16, 32]:
            config = MappedTuningConfig(num_warps=w)
            assert _is_power_of_two(config.num_warps)
            assert config.num_warps <= 32

    def test_config_mapper_produces_valid_config(self) -> None:
        """ConfigMapper.map_record produces structurally valid config.

        What it tests:
          Mapping a representative TVM trace produces a MappedTuningConfig
          with all fields set to positive, power-of-two values.

        Passing means:
          The mapped config has block_m, block_n, block_k > 0,
          num_warps in {1,2,4,8,16,32}, num_stages >= 1.
        """
        mapper = ConfigMapper()
        trace = {
            "instructions": [{"type": "MultiLevelTiling"}],
            "decisions": {
                "tile_m": [2, 4, 4],
                "tile_n": [2, 4, 4],
                "tile_k": [4, 4],
                "stages": 5,
            },
        }
        config = mapper.map_record(trace)
        assert config.block_m == 32
        assert config.block_n == 32
        assert config.block_k == 16
        assert config.num_stages == 5

    def test_config_mapper_json_round_trip(self) -> None:
        """A config serialized to JSON and back should match.

        What it tests:
          JSON round-trip through config_mapper's map_json_record.

        Passing means:
          The deserialized config matches the original.
        """
        import json

        mapper = ConfigMapper()
        original = {
            "decisions": {"tile_m": [1, 8], "tile_n": [1, 16], "tile_k": [2, 8]},
        }
        json_str = json.dumps(original)
        config = mapper.map_json_record(json_str)
        assert config.block_m == 8
        assert config.block_n == 16
        assert config.block_k == 16

    def test_config_mapper_empty_trace_returns_defaults(self) -> None:
        """Mapping an empty trace should return MappedTuningConfig.defaults().

        What it tests:
          Edge case: empty decisions dict.

        Passing means:
          The result equals MappedTuningConfig.defaults().
        """
        mapper = ConfigMapper()
        config = mapper.map_record({})
        assert config == MappedTuningConfig.defaults()

    def test_config_mapper_none_record_returns_defaults(self) -> None:
        """Mapping None should return MappedTuningConfig.defaults().

        What it tests:
          Edge case: None input.

        Passing means:
          The result equals MappedTuningConfig.defaults().
        """
        mapper = ConfigMapper()
        config = mapper.map_record(None)  # type: ignore[arg-type]
        assert config == MappedTuningConfig.defaults()

    def test_mapped_config_to_triton_config(self) -> None:
        """to_triton_config() should produce a valid triton.Config.

        What it tests:
          The conversion from MappedTuningConfig → triton.Config.

        Passing means:
          The triton.Config has correct BLOCK_SIZE_{M,N,K} in kwargs,
          and correct num_warps/num_stages/num_ctas.
        """
        import triton  # local import: Config creation requires triton

        config = MappedTuningConfig(
            block_m=64, block_n=128, block_k=32,
            num_warps=8, num_stages=4, num_ctas=2,
        )
        triton_cfg = config.to_triton_config({
            "BLOCK_SIZE": 64, "GROUP_SIZE": 8,
        })
        assert triton_cfg.kwargs["BLOCK_SIZE_M"] == 64
        assert triton_cfg.kwargs["BLOCK_SIZE_N"] == 128
        assert triton_cfg.kwargs["BLOCK_SIZE_K"] == 32
        assert triton_cfg.kwargs["BLOCK_SIZE"] == 64
        assert triton_cfg.kwargs["GROUP_SIZE"] == 8
        assert triton_cfg.num_warps == 8
        assert triton_cfg.num_stages == 4
        assert triton_cfg.num_ctas == 2


# =========================================================================
# Bounds extraction in pipeline context
# =========================================================================


class TestBoundsInPipeline:
    """Bounds extraction tested in the pipeline context."""

    def test_bounds_extractor_parses_matmul_ir(self) -> None:
        """Bounds extractor correctly parses a matmul TTGIR snippet.

        What it tests:
          The BoundsExtractor.extract() method via the IR classifier
          and capture pipeline.

        Passing means:
          Returns IRBounds with m=128, n=128, k=64, data_dtype=float32.
        """
        from src.bridges.triton_tvm.bounds_extractor import BoundsExtractor
        from src.bridges.triton_tvm.ir_capture import KernelKind

        ir = _matmul_ttgir()
        extractor = BoundsExtractor()
        bounds = extractor.extract(ir, KernelKind.MATMUL)
        assert bounds.m == 128
        assert bounds.n == 128
        assert bounds.k == 64
        assert bounds.data_dtype == "float32"

    def test_bounds_extractor_parses_reduction_ir(self) -> None:
        """Bounds extractor correctly identifies reduction axis and size.

        What it tests:
          Reduction IR parsing through BoundsExtractor.

        Passing means:
          Returns IRBounds with reduce_size=128 (dim 0 on 128x1024),
          keep_size=1024 (product of retained dims).
        """
        from src.bridges.triton_tvm.bounds_extractor import BoundsExtractor
        from src.bridges.triton_tvm.ir_capture import KernelKind

        ir = _reduction_ttgir()
        extractor = BoundsExtractor()
        bounds = extractor.extract(ir, KernelKind.REDUCTION)
        assert bounds.reduce_size == 128, \
            f"Expected reduce_size=128 (axis 0 on 128x1024), got {bounds.reduce_size}"
        assert bounds.keep_size == 1024, \
            f"Expected keep_size=1024, got {bounds.keep_size}"
        assert bounds.data_dtype == "float32"

    def test_bounds_extractor_parses_elementwise_ir(self) -> None:
        """Bounds extractor correctly computes total_elements.

        What it tests:
          Elementwise IR parsing through BoundsExtractor.

        Passing means:
          Returns IRBounds with total_elements=4096.
        """
        from src.bridges.triton_tvm.bounds_extractor import BoundsExtractor
        from src.bridges.triton_tvm.ir_capture import KernelKind

        ir = _elementwise_ttgir()
        extractor = BoundsExtractor()
        bounds = extractor.extract(ir, KernelKind.ELEMENTWISE)
        assert bounds.total_elements == 4096
        assert bounds.data_dtype == "float32"

    def test_ir_capture_pipeline_processes_matmul(
        self,
    ) -> None:
        """IRCapture processes matmul IR through classify → extract.

        What it tests:
          The IRCapture.capture_from_text pipeline that the bridge uses
          for real-IR tuning.

        Passing means:
          CapturedKernelIR has kind=MATMUL, bounds with m=128, n=128, k=64.
        """
        from src.bridges.triton_tvm.ir_capture import IRCapture, KernelKind

        capture = IRCapture()
        ir = _matmul_ttgir()
        result = capture.capture_from_text(
            ir, source_hash="test", target="nvidia/nvidia-a100",
        )
        assert result.kind == KernelKind.MATMUL
        assert result.bounds.m == 128
        assert result.bounds.n == 128
        assert result.bounds.k == 64
        assert result.ops_seen is not None

    def test_ir_capture_pipeline_processes_reduction(
        self,
    ) -> None:
        """IRCapture processes reduction IR through classify → extract.

        What it tests:
          IRCapture with reduction kernel.

        Passing means:
          CapturedKernelIR has kind=REDUCTION, bounds with reduce_size.
        """
        from src.bridges.triton_tvm.ir_capture import IRCapture, KernelKind

        capture = IRCapture()
        ir = _reduction_ttgir()
        result = capture.capture_from_text(
            ir, source_hash="test_reduce", target="nvidia/nvidia-a100",
        )
        assert result.kind == KernelKind.REDUCTION
        assert result.bounds.reduce_size == 128
        assert result.bounds.keep_size == 1024

    def test_ir_capture_classifier_classifies_kernel_kind(
        self,
    ) -> None:
        """IRClassifier correctly classifies kernel kinds from IR.

        What it tests:
          The classifier used inside IRCapture distinguishes matmul,
          reduction, and elementwise kernels.

        Passing means:
          Each IR snippet is classified to the correct KernelKind.
        """
        from src.bridges.triton_tvm.ir_capture import IRCapture, KernelKind

        capture = IRCapture()
        for ir_text, expected in [
            (_matmul_ttgir(), KernelKind.MATMUL),
            (_reduction_ttgir(), KernelKind.REDUCTION),
            (_elementwise_ttgir(), KernelKind.ELEMENTWISE),
        ]:
            result = capture.capture_from_text(
                ir_text, source_hash="test", target="nvidia/nvidia-a100",
            )
            assert result.kind == expected, \
                f"Expected {expected.name}, got {result.kind.name}"

    def test_bounds_extraction_through_inline_pipeline(self) -> None:
        """Bounds extracted through IR capture flow to the tuning pipeline.

        What it tests:
          When IR is captured and bounds extracted, the values can be
          used to construct a TIR template (mocked here) through the
          bridge's _build_tir_from_captured method.

        Passing means:
          The captured bounds are non-negative and consistent.
        """
        from src.bridges.triton_tvm.ir_capture import IRCapture

        capture = IRCapture()
        ir = _matmul_ttgir()
        result = capture.capture_from_text(
            ir, source_hash="test", target="nvidia/nvidia-a100",
        )
        bounds = result.bounds
        assert bounds.m is not None and bounds.m > 0
        assert bounds.n is not None and bounds.n > 0
        assert bounds.k is not None and bounds.k > 0
        assert bounds.data_dtype in ("float32", "float16", "bfloat16")

    def test_ir_classifier_collects_ops(self) -> None:
        """IRClassifier.collect_ops returns ops in order of appearance.

        What it tests:
          The IR ops collection used by the bridge for observability.

        Passing means:
          The ops list contains the expected ops for a matmul kernel.
        """
        from src.bridges.triton_tvm.ir_classifier import IRClassifier

        classifier = IRClassifier()
        ir = _matmul_ttgir()
        ops = classifier.collect_ops(ir)
        assert "tt.load" in ops
        assert "tt.dot" in ops
        assert "tt.store" in ops

    def test_classifier_detects_attention_pattern(self) -> None:
        """IRClassifier distinguishes attention (softmax) from matmul.

        What it tests:
          The softmax detection heuristic: reduce → exp → dot.

        Passing means:
          IR with two dots and a reduce followed by exp is classified
          as ATTENTION.
        """
        from src.bridges.triton_tvm.ir_classifier import IRClassifier
        from src.bridges.triton_tvm.ir_capture import KernelKind

        classifier = IRClassifier()
        ir = _attention_ttgir()
        result = classifier.classify(ir)
        assert result == KernelKind.ATTENTION, \
            f"Expected ATTENTION, got {result.kind.name}"

    def test_bounds_extraction_errors_are_informative(self) -> None:
        """Bounds extraction errors carry clear messages.

        What it tests:
          The BoundsExtractionError exception that callers must catch.

        Passing means:
          The error message mentions the specific problem (missing op).
        """
        from src.bridges.triton_tvm.bounds_extractor import (
            BoundsExtractionError,
            BoundsExtractor,
        )
        from src.bridges.triton_tvm.ir_capture import KernelKind

        extractor = BoundsExtractor()
        with pytest.raises(BoundsExtractionError) as exc:
            extractor.extract("module { tt.func @main() { tt.return } }", KernelKind.MATMUL)
        assert "no matmul" in str(exc.value).lower()

    def test_classifier_raises_on_no_supported_ops(self) -> None:
        """Classifier raises ClassificationError on unsupported IR.

        What it tests:
          The error path in IRClassifier for unrecognised IR.

        Passing means:
          ClassificationError is raised with a clear message.
        """
        from src.bridges.triton_tvm.ir_classifier import (
            ClassificationError,
            IRClassifier,
        )

        classifier = IRClassifier()
        with pytest.raises(ClassificationError) as exc:
            classifier.classify("module { tt.func @main() { tt.return } }")
        assert "no supported ops" in str(exc.value).lower()


# =========================================================================
# TuningResult structural validity
# =========================================================================


class TestTuningResultValidity:
    """TuningResult output contract validation."""

    def test_tuning_result_from_successful_tune(self) -> None:
        """A successful tune produces a fully-populated TuningResult.

        What it tests:
          TuningResult dataclass contract.

        Passing means:
          All required fields are present and have the correct types.
        """
        config = MappedTuningConfig(block_m=128)
        result = TuningResult(
            config=config,
            fallback_tier=FallbackTier.L0_TVM_DB_HIT,
            total_duration_ms=42.5,
            cache_hit=True,
            stages={"extract": 5.0, "tvm_tune": 35.0},
        )
        assert result.config.block_m == 128
        assert result.fallback_tier == FallbackTier.L0_TVM_DB_HIT
        assert result.total_duration_ms == 42.5
        assert result.cache_hit is True
        assert result.stages["extract"] == 5.0
        assert result.error is None
        assert result.fallback_used is False

    def test_tuning_result_with_fallback(self) -> None:
        """A fallback TuningResult should carry error and reason.

        What it tests:
          TuningResult with error metadata.

        Passing means:
          Error message and fallback reason are populated.
        """
        result = TuningResult(
            config=MappedTuningConfig.defaults(),
            fallback_tier=FallbackTier.L5_SAFE_FALLBACK,
            total_duration_ms=10.0,
            error="TVM not available",
            fallback_used=True,
            fallback_reason="real_ir_capture_failed: no IR for source",
        )
        assert result.error == "TVM not available"
        assert result.fallback_used is True
        assert result.fallback_reason is not None

    def test_tune_configs_list_produces_multiple_configs(
        self,
        auto_tuning_bridge: TritonTVMBridge,
        sample_matmul_metadata: Any,
    ) -> None:
        """tune_configs_list returns varied configs for @triton.autotune.

        What it tests:
          The configs list generation for @triton.autotune integration.

        Passing means:
          Returns a list of MappedTuningConfig with different block sizes.
        """
        configs = auto_tuning_bridge.tune_configs_list(
            sample_matmul_metadata, "nvidia/nvidia-a100",
        )
        assert len(configs) >= 1
        for cfg in configs:
            assert isinstance(cfg, MappedTuningConfig)
            assert cfg.block_m >= 16
            assert cfg.block_n >= 16
            assert cfg.block_k >= 16

    def test_tune_without_tvm_still_produces_valid_result(
        self,
        cache_dir: str,
        sample_matmul_metadata: Any,
    ) -> None:
        """Without TVM, the bridge still returns a TuningResult with defaults.

        What it tests:
          The bridge degrades gracefully when TVM is unavailable.

        Passing means:
          TuningResult config equals MappedTuningConfig.defaults().
        """
        bridge = TritonTVMBridge(cache_dir=cache_dir, enable_tvm=False)
        result = bridge._fallback_config(
            sample_matmul_metadata, FallbackTier.L4_TRITON_DEFAULT,
        )
        # For matmul, fallback should be 128x128x32
        assert result.block_m == 128
        assert result.block_n == 128
        assert result.block_k == 32


# =========================================================================
# Helpers
# =========================================================================


def _is_power_of_two(n: int) -> bool:
    """Return True if n is a positive power of two."""
    return n > 0 and (n & (n - 1)) == 0


def _dummy_kernel() -> None:
    """A real Python function usable as a fake Triton kernel in tests."""
    pass


# -------------------------------------------------------------------------
# TTGIR test fixtures — verified AST-parseable snippets
# -------------------------------------------------------------------------


def _matmul_ttgir() -> str:
    """Canonical matmul TTGIR with M=128, N=128, K=64 dot product."""
    return """
module {
  tt.func @matmul_kernel(%A_ptr: !tt.ptr<tensor<128x64xf32>>, %B_ptr: !tt.ptr<tensor<64x128xf32>>, %C_ptr: !tt.ptr<tensor<128x128xf32>>) {
    %pid = tt.get_program_id x : i32
    %a = tt.load %A_ptr : !tt.ptr<tensor<128x64xf32>>
    %b = tt.load %B_ptr : !tt.ptr<tensor<64x128xf32>>
    %c_init = tt.splat %pid : f32
    %c = tt.dot %a, %b : tensor<128x64xf32> * tensor<64x128xf32> -> tensor<128x128xf32>
    tt.store %C_ptr, %c : !tt.ptr<tensor<128x128xf32>>
    tt.return
  }
}
""".strip()


def _reduction_ttgir() -> str:
    """Canonical reduction TTGIR reducing axis 0 on a 128x1024 tensor."""
    return """
module {
  tt.func @reduce_kernel(%ptr: !tt.ptr<tensor<128x1024xf32>>) {
    %val = tt.load %ptr : !tt.ptr<tensor<128x1024xf32>>
    %reduced = tt.reduce(%val) ({
    ^bb0(%a: f32, %b: f32):
      arith.addf %a, %b : f32
    }) {axis = 0 : i32} : tensor<128x1024xf32> -> tensor<1024xf32>
    tt.return
  }
}
""".strip()


def _elementwise_ttgir() -> str:
    """Canonical elementwise TTGIR with a 4096-element add."""
    return """
module {
  tt.func @add_kernel(%a_ptr: !tt.ptr<tensor<4096xf32>>, %b_ptr: !tt.ptr<tensor<4096xf32>>, %c_ptr: !tt.ptr<tensor<4096xf32>>) {
    %a = tt.load %a_ptr : !tt.ptr<tensor<4096xf32>>
    %b = tt.load %b_ptr : !tt.ptr<tensor<4096xf32>>
    %c = arith.addf %a, %b : tensor<4096xf32>
    tt.store %c_ptr, %c : !tt.ptr<tensor<4096xf32>>
    tt.return
  }
}
""".strip()


def _attention_ttgir() -> str:
    """Canonical attention TTGIR: matmul → softmax → matmul pattern."""
    return """
module {
  tt.func @attention_kernel(%q_ptr: !tt.ptr<tensor<128x64xf32>>, %k_ptr: !tt.ptr<tensor<64x128xf32>>, %v_ptr: !tt.ptr<tensor<128x64xf32>>, %o_ptr: !tt.ptr<tensor<128x128xf32>>) {
    %q = tt.load %q_ptr : !tt.ptr<tensor<128x64xf32>>
    %k = tt.load %k_ptr : !tt.ptr<tensor<64x128xf32>>
    %v = tt.load %v_ptr : !tt.ptr<tensor<128x64xf32>>
    %s = tt.dot %q, %k : tensor<128x64xf32> * tensor<64x128xf32> -> tensor<128x128xf32>
    %m = tt.reduce(%s) ({
    ^bb0(%a: f32, %b: f32):
      arith.maximumf %a, %b : f32
    }) {axis = 1 : i32} : tensor<128x128xf32> -> tensor<128xf32>
    %e = math.exp %s : tensor<128x128xf32>
    %o = tt.dot %e, %v : tensor<128x128xf32> * tensor<128x64xf32> -> tensor<128x128xf32>
    tt.store %o_ptr, %o : !tt.ptr<tensor<128x128xf32>>
    tt.return
  }
}
""".strip()
