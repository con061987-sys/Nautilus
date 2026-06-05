"""Per-stage timeout orchestration for the bridge pipeline.

A single global timeout is wrong: a 5-second metadata extraction should
NOT share its budget with a 10-minute TVM tune. Each stage has its own
budget, and the total budget is the sum of all stage budgets.

This module provides:
  - Per-stage timeouts (caller specifies per stage)
  - Hard total budget (the total time the bridge can spend)
  - Graceful timeout behaviour: clean up resources, mark stage as
    timed out, and let the next fallback tier run
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

logger = logging.getLogger(__name__)


class StageTimeoutError(Exception):
    """Raised when a stage exceeds its time budget."""
    def __init__(self, stage_name: str, budget_s: float, elapsed_s: float):
        self.stage_name = stage_name
        self.budget_s = budget_s
        self.elapsed_s = elapsed_s
        super().__init__(
            f"Stage '{stage_name}' timed out after {elapsed_s:.1f}s "
            f"(budget was {budget_s:.1f}s)"
        )


class TotalBudgetExceededError(Exception):
    """Raised when the cumulative elapsed time exceeds the total budget."""
    def __init__(self, budget_s: float, elapsed_s: float):
        self.budget_s = budget_s
        self.elapsed_s = elapsed_s
        super().__init__(
            f"Total bridge budget exhausted: {elapsed_s:.1f}s > {budget_s:.1f}s"
        )


@dataclass
class StageBudgets:
    """Per-stage time budgets.

    Each bridge stage has its own time budget. The total budget is
    the sum of all stage budgets but enforced separately so that
    a fast stage cannot give time to a slow one.
    """
    extract_s: float = 5.0
    build_tir_s: float = 10.0
    tune_s: float = 600.0
    map_config_s: float = 1.0
    recompile_s: float = 30.0
    aot_compile_s: float = 60.0
    link_s: float = 10.0
    validate_s: float = 60.0

    @property
    def total_s(self) -> float:
        """Sum of all stage budgets (for total budget enforcement)."""
        return (
            self.extract_s + self.build_tir_s + self.tune_s +
            self.map_config_s + self.recompile_s + self.aot_compile_s +
            self.link_s + self.validate_s
        )

    def for_stage(self, stage_name: str) -> float:
        """Get the budget for a named stage."""
        attr_name = f"{stage_name}_s"
        if hasattr(self, attr_name):
            return getattr(self, attr_name)
        # Default budget for unknown stages
        return 60.0


class TimeoutManager:
    """Manages per-stage timeouts across a bridge run.

    Usage:
        mgr = TimeoutManager(StageBudgets())

        with mgr.stage("extract") as t:
            metadata = extract_kernel_info(...)

        with mgr.stage("tune") as t:
            config = tvm.tune(...)

        mgr.check_total_budget()  # raises if total budget exceeded
    """

    def __init__(self, budgets: StageBudgets | None = None) -> None:
        self.budgets = budgets or StageBudgets()
        self._stage_start: float = 0.0
        self._run_start: float = time.time()
        self._stage_name: str = ""
        self._total_budget: float = self.budgets.total_s

    @contextmanager
    def stage(self, name: str) -> Iterator[float]:
        """Context manager for a single timed stage.

        Raises StageTimeoutError if the stage exceeds its budget.
        """
        budget = self.budgets.for_stage(name)
        self._stage_name = name
        self._stage_start = time.time()

        # Use a timer thread to enforce the budget
        timed_out = threading.Event()
        exception_holder: list[BaseException] = []

        def _enforcer() -> None:
            time.sleep(budget)
            if not timed_out.is_set():
                timed_out.set()
                logger.warning(
                    "Stage '%s' exceeded budget of %.1fs",
                    name, budget,
                )

        timer = threading.Thread(target=_enforcer, daemon=True)
        timer.start()
        try:
            yield budget
            timed_out.set()
        except Exception as exc:
            exception_holder.append(exc)
            timed_out.set()
            raise
        finally:
            elapsed = time.time() - self._stage_start
            if timed_out.is_set() and not exception_holder:
                raise StageTimeoutError(name, budget, elapsed)
            if elapsed > budget:
                logger.warning(
                    "Stage '%s' took %.1fs (budget was %.1fs)",
                    name, elapsed, budget,
                )

    def check_total_budget(self) -> float:
        """Check if the total budget has been exceeded.

        Returns the remaining time. Raises TotalBudgetExceededError
        if exhausted.
        """
        elapsed = time.time() - self._run_start
        if elapsed > self._total_budget:
            raise TotalBudgetExceededError(self._total_budget, elapsed)
        return self._total_budget - elapsed

    @property
    def elapsed_s(self) -> float:
        """Total elapsed time since the manager was created."""
        return time.time() - self._run_start

    @property
    def remaining_s(self) -> float:
        """Time remaining in the total budget."""
        return max(0.0, self._total_budget - self.elapsed_s)


@contextmanager
def wallclock_timeout(seconds: float, label: str = "operation") -> Iterator[None]:
    """Lower-level timeout context using SIGALRM (Unix only).

    Use TimeoutManager.stage() in normal code. This is for cases
    where SIGALRM-based preemption is required (e.g. blocking C calls).

    Caveats:
      - Unix-only (SIGALRM is not available on Windows)
      - Only works in the main thread
      - Must be inside a function, not at module level
    """
    if not seconds or seconds <= 0:
        yield
        return

    def _handler(signum: int, frame: Any) -> None:
        raise StageTimeoutError(label, seconds, seconds)

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
