"""Tests for the timeout manager."""

from __future__ import annotations

import time

import pytest

from src.bridges.triton_tvm.timeout_manager import (
    StageBudgets,
    StageTimeoutError,
    TimeoutManager,
    TotalBudgetExceededError,
    wallclock_timeout,
)


class TestStageBudgets:
    """Tests for the per-stage budget configuration."""

    def test_default_budgets_sum(self) -> None:
        """Total budget should equal the sum of all stages."""
        budgets = StageBudgets()
        expected = (
            budgets.extract_s
            + budgets.build_tir_s
            + budgets.tune_s
            + budgets.map_config_s
            + budgets.recompile_s
            + budgets.aot_compile_s
            + budgets.link_s
            + budgets.validate_s
        )
        assert budgets.total_s == expected

    def test_for_stage_known(self) -> None:
        """Known stages should return their configured budget."""
        budgets = StageBudgets()
        assert budgets.for_stage("extract") == budgets.extract_s
        assert budgets.for_stage("tune") == budgets.tune_s

    def test_for_stage_unknown_uses_default(self) -> None:
        """Unknown stages should return a default budget."""
        budgets = StageBudgets()
        assert budgets.for_stage("nonexistent_stage") == 60.0


class TestTimeoutManager:
    """Tests for the TimeoutManager class."""

    def test_initial_state(self) -> None:
        """A fresh manager should have no elapsed time."""
        mgr = TimeoutManager(StageBudgets())
        assert mgr.elapsed_s < 1.0
        assert mgr.remaining_s > 0

    def test_stage_under_budget_succeeds(self) -> None:
        """A stage under its budget should complete normally."""
        mgr = TimeoutManager(StageBudgets(extract_s=5.0))
        with mgr.stage("extract") as budget:
            time.sleep(0.1)
            assert budget == 5.0

    def test_stage_over_budget_raises(self) -> None:
        """A stage exceeding its budget should raise StageTimeoutError."""
        mgr = TimeoutManager(StageBudgets(extract_s=0.1))
        with pytest.raises(StageTimeoutError) as exc_info, mgr.stage("extract"):
            time.sleep(0.5)  # Will exceed the 0.1s budget
        assert "extract" in str(exc_info.value)
        assert exc_info.value.budget_s == pytest.approx(0.1, abs=0.01)

    def test_stage_exception_propagates(self) -> None:
        """Exceptions inside a stage should propagate after the stage ends."""
        mgr = TimeoutManager(StageBudgets())
        with pytest.raises(ValueError), mgr.stage("extract"):
            raise ValueError("test error")

    def test_total_budget_exceeded(self) -> None:
        """The total budget should be enforceable independently."""
        # Set tight budgets so we can quickly exhaust
        budgets = StageBudgets(
            extract_s=0.05,
            build_tir_s=0.05,
            tune_s=0.05,
            map_config_s=0.05,
            recompile_s=0.05,
            aot_compile_s=0.05,
            link_s=0.05,
            validate_s=0.05,
        )
        mgr = TimeoutManager(budgets)
        for _ in range(8):
            with pytest.raises(StageTimeoutError), mgr.stage("extract"):
                time.sleep(0.2)
        # Now the total budget should be exhausted
        with pytest.raises(TotalBudgetExceededError):
            mgr.check_total_budget()

    def test_remaining_s_decreases(self) -> None:
        """remaining_s should decrease as time passes."""
        mgr = TimeoutManager(StageBudgets(extract_s=2.0))
        first_remaining = mgr.remaining_s
        time.sleep(0.1)
        second_remaining = mgr.remaining_s
        assert second_remaining < first_remaining


class TestWallclockTimeout:
    """Tests for the SIGALRM-based wallclock timeout."""

    def test_under_timeout_succeeds(self) -> None:
        """A short operation under the timeout should succeed."""
        with wallclock_timeout(seconds=1.0):
            time.sleep(0.05)

    def test_zero_timeout_is_noop(self) -> None:
        """A zero or negative timeout should be a no-op."""
        with wallclock_timeout(seconds=0):
            time.sleep(0.01)
        with wallclock_timeout(seconds=-1):
            time.sleep(0.01)

    def test_over_timeout_raises(self) -> None:
        """An operation exceeding the wallclock timeout should raise."""
        # Note: SIGALRM-based timeout requires Unix and main thread
        import sys

        if sys.platform == "win32":
            pytest.skip("SIGALRM not available on Windows")

        with (
            pytest.raises((StageTimeoutError, TimeoutError)),
            wallclock_timeout(seconds=0.1, label="fast_op"),
        ):
            time.sleep(1.0)  # Will definitely exceed 0.1s
