"""
Cross-cutting observability primitives — moved here from
src.bridges.triton_tvm so all bridges can share them without
creating cross-bridge coupling.

This module owns:
  - CircuitBreaker: per-dependency failure isolation
  - TimeoutManager / StageBudgets: per-stage timeouts (not a single
    global timeout that hides which stage is slow)
"""

from __future__ import annotations

import enum
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from src.common.errors import (
    CircuitOpenError,
    StageTimeoutError,
    TotalBudgetExceededError,
)
from src.common.logging import get_logger

T = TypeVar("T")
log = get_logger("nautilus.observability")


# --- Circuit Breaker ---


class CircuitState(str, enum.Enum):
    CLOSED = "closed"  # Normal operation
    HALF_OPEN = "half_open"  # Trial: allow one call to test recovery
    OPEN = "open"  # Failing: short-circuit calls


@dataclass
class CircuitBreakerConfig:
    """Tuning knobs for a circuit breaker."""

    name: str = "default"
    failure_threshold: int = 5  # Consecutive failures to open
    reset_timeout_seconds: float = 30.0  # How long to stay open before half-open trial
    half_open_max_trials: int = 1  # How many probes allowed in half-open
    excluded_exceptions: tuple[type[BaseException], ...] = ()


@dataclass
class _BreakerStats:
    state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    total_failures: int = 0
    total_successes: int = 0
    last_failure_time: float = 0.0
    last_state_change: float = 0.0
    half_open_remaining: int = 0
    last_error: str | None = None


class CircuitBreaker:
    """Per-dependency circuit breaker.

    When a dependency (TVM, AOTriton, etc.) starts failing, the
    breaker opens and short-circuits subsequent calls — preventing
    the entire pipeline from blocking on a dead dependency.

    Usage:

        breaker = CircuitBreaker(CircuitBreakerConfig(
            name="tvm_tune",
            failure_threshold=3,
            reset_timeout_seconds=60.0,
        ))

        def tune():
            return breaker.call(tvm.tune_tir, mod, target, ...)

        # Or context manager:
        with breaker:
            do_risky_thing()
    """

    def __init__(self, config: CircuitBreakerConfig) -> None:
        self._config = config
        self._stats = _BreakerStats()
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._maybe_transition()
            return self._stats.state

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            self._maybe_transition()
            return {
                "name": self._config.name,
                "state": self._stats.state.value,
                "consecutive_failures": self._stats.consecutive_failures,
                "total_failures": self._stats.total_failures,
                "total_successes": self._stats.total_successes,
                "last_failure_time": self._stats.last_failure_time,
                "last_state_change": self._stats.last_state_change,
                "last_error": self._stats.last_error,
            }

    def _maybe_transition(self) -> None:
        """If we're OPEN and enough time has passed, transition to HALF_OPEN."""
        if self._stats.state != CircuitState.OPEN:
            return
        now = time.time()
        if now - self._stats.last_state_change >= self._config.reset_timeout_seconds:
            log.info(
                "circuit_breaker half_open",
                name=self._config.name,
                was_open_seconds=now - self._stats.last_state_change,
            )
            self._stats.state = CircuitState.HALF_OPEN
            self._stats.half_open_remaining = self._config.half_open_max_trials
            self._stats.last_state_change = now

    def _record_success(self) -> None:
        with self._lock:
            previous = self._stats.state
            self._stats.total_successes += 1
            if previous == CircuitState.HALF_OPEN:
                log.info("circuit_breaker closed (recovered)", name=self._config.name)
                self._stats.state = CircuitState.CLOSED
                self._stats.consecutive_failures = 0
                self._stats.last_state_change = time.time()
            elif previous == CircuitState.CLOSED:
                self._stats.consecutive_failures = 0

    def _record_failure(self, exc: BaseException) -> None:
        with self._lock:
            self._stats.total_failures += 1
            self._stats.consecutive_failures += 1
            self._stats.last_failure_time = time.time()
            self._stats.last_error = f"{type(exc).__name__}: {exc}"
            if (
                self._stats.consecutive_failures >= self._config.failure_threshold
                and self._stats.state != CircuitState.OPEN
            ):
                log.warning(
                    "circuit_breaker opened",
                    name=self._config.name,
                    consecutive_failures=self._stats.consecutive_failures,
                    last_error=self._stats.last_error,
                )
                self._stats.state = CircuitState.OPEN
                self._stats.last_state_change = time.time()

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Invoke `fn(*args, **kwargs)` under circuit-breaker protection.

        Raises CircuitOpenError if the breaker is open and the call
        should be short-circuited. Re-raises the underlying exception
        otherwise.
        """
        with self._lock:
            self._maybe_transition()
            if self._stats.state == CircuitState.OPEN:
                raise CircuitOpenError(
                    f"Circuit '{self._config.name}' is OPEN; "
                    f"short-circuiting call to {fn.__name__}",
                    context={
                        "breaker": self._config.name,
                        "consecutive_failures": self._stats.consecutive_failures,
                        "last_error": self._stats.last_error,
                    },
                )
            if self._stats.state == CircuitState.HALF_OPEN:
                if self._stats.half_open_remaining <= 0:
                    raise CircuitOpenError(
                        f"Circuit '{self._config.name}' is HALF_OPEN and used all trial slots",
                        context={"breaker": self._config.name},
                    )
                self._stats.half_open_remaining -= 1

        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            if self._config.excluded_exceptions and isinstance(
                exc, self._config.excluded_exceptions
            ):
                # Don't count toward breaker
                raise
            self._record_failure(exc)
            raise
        else:
            self._record_success()
            return result

    def __enter__(self) -> CircuitBreaker:
        self._maybe_transition()
        if self._stats.state == CircuitState.OPEN:
            raise CircuitOpenError(
                f"Circuit '{self._config.name}' is OPEN",
                context={"breaker": self._config.name},
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Literal[False]:
        if exc is None:
            self._record_success()
            return False
        if isinstance(exc, self._config.excluded_exceptions):
            return False
        self._record_failure(exc)
        return False

    def reset(self) -> None:
        """Force the breaker back to CLOSED. For ops/debug only."""
        with self._lock:
            self._stats = _BreakerStats()
            self._stats.last_state_change = time.time()
            log.info("circuit_breaker reset", name=self._config.name)


def _build_default_breakers() -> dict[str, CircuitBreaker]:
    """Construct a fresh dict of framework-standard circuit breakers."""
    return {
        "tvm_tune": CircuitBreaker(
            CircuitBreakerConfig(
                name="tvm_tune",
                failure_threshold=3,
                reset_timeout_seconds=60.0,
            )
        ),
        "triton_compile": CircuitBreaker(
            CircuitBreakerConfig(
                name="triton_compile",
                failure_threshold=5,
                reset_timeout_seconds=30.0,
            )
        ),
        "aotriton_compile": CircuitBreaker(
            CircuitBreakerConfig(
                name="aotriton_compile",
                failure_threshold=3,
                reset_timeout_seconds=60.0,
            )
        ),
        "gspmd": CircuitBreaker(
            CircuitBreakerConfig(
                name="gspmd",
                failure_threshold=3,
                reset_timeout_seconds=30.0,
            )
        ),
        "lld_link": CircuitBreaker(
            CircuitBreakerConfig(
                name="lld_link",
                failure_threshold=5,
                reset_timeout_seconds=15.0,
            )
        ),
    }


_DEFAULT_BREAKERS: dict[str, CircuitBreaker] | None = None


def get_default_breakers() -> dict[str, CircuitBreaker]:
    """Return the framework's default circuit-breaker dict (singleton).

    Subsequent calls return the SAME dict instance, so state
    (open/half-open/closed, failure counts, last error) is shared
    across all callers — exactly what production wiring needs.
    Tests that need a clean slate should call
    :func:`reset_default_breakers` first.
    """
    global _DEFAULT_BREAKERS
    if _DEFAULT_BREAKERS is None:
        _DEFAULT_BREAKERS = _build_default_breakers()
    return _DEFAULT_BREAKERS


def reset_default_breakers() -> dict[str, CircuitBreaker]:
    """Replace the singleton with a fresh dict and return it.

    Intended for tests. After calling this, the next
    :func:`get_default_breakers` returns the new dict. Each breaker
    is also ``reset()`` to CLOSED so any cross-test pollution is
    eliminated.
    """
    global _DEFAULT_BREAKERS
    _DEFAULT_BREAKERS = _build_default_breakers()
    for breaker in _DEFAULT_BREAKERS.values():
        breaker.reset()
    return _DEFAULT_BREAKERS


# --- Timeout Manager ---


@dataclass
class StageBudgets:
    """Per-stage timeouts. NOT a single global timeout."""

    ir_capture_seconds: float = 10.0
    tvm_tune_seconds: float = 600.0
    triton_compile_seconds: float = 120.0
    amd_compile_seconds: float = 300.0
    intel_compile_seconds: float = 300.0
    nvidia_compile_seconds: float = 120.0
    runtime_stub_compile_seconds: float = 30.0
    lld_link_seconds: float = 30.0
    hardware_validation_seconds: float = 60.0
    sharding_seconds: float = 300.0
    stablehlo_export_seconds: float = 60.0
    fat_binary_total_seconds: float = 1800.0  # 30 min overall

    def get(self, stage: str) -> float:
        attr = f"{stage}_seconds"
        if not hasattr(self, attr):
            raise KeyError(f"Unknown stage: {stage!r}")
        return float(getattr(self, attr))


class TimeoutManager:
    """Per-stage timeout enforcement.

    Unlike a single global timeout, this tracks which stage is
    actually slow, and aborts with a typed error.
    """

    def __init__(self, budgets: StageBudgets) -> None:
        self._budgets = budgets
        self._start_time: float | None = None
        self._current_stage: str | None = None

    def start(self) -> None:
        self._start_time = time.time()

    def check_total_budget(self) -> None:
        if self._start_time is None:
            return
        elapsed = time.time() - self._start_time
        if elapsed > self._budgets.fat_binary_total_seconds:
            raise TotalBudgetExceededError(
                f"Total budget exceeded: {elapsed:.1f}s > "
                f"{self._budgets.fat_binary_total_seconds:.1f}s",
                context={
                    "elapsed_seconds": elapsed,
                    "budget_seconds": self._budgets.fat_binary_total_seconds,
                },
            )

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Context manager that enforces the per-stage budget.

        If the stage body raises, the original exception propagates.
        If the budget is exceeded AND no other exception is in flight,
        a :class:`StageTimeoutError` is raised. If the budget is
        exceeded WHILE another exception is propagating, the original
        exception wins — a ``raise`` in ``finally`` would otherwise
        mask the real cause (Python only chains via
        ``__context__``; the original traceback is buried). The
        timeout is still logged so an operator can correlate the
        slow stage with the failure.
        """
        if self._start_time is None:
            self.start()
        budget = self._budgets.get(name)
        self._current_stage = name
        stage_start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - stage_start
            over_budget = elapsed > budget
            # sys.exc_info() returns (None, None, None) when no
            # exception is propagating. If something is in flight,
            # do NOT raise a fresh StageTimeoutError — that would
            # shadow the real cause.
            in_flight_exc = sys.exc_info()[1]
            propagating = in_flight_exc is not None
            if over_budget and not propagating:
                raise StageTimeoutError(
                    f"Stage {name!r} exceeded budget: {elapsed:.1f}s > {budget:.1f}s",
                    context={
                        "stage": name,
                        "elapsed_seconds": elapsed,
                        "budget_seconds": budget,
                    },
                )
            if over_budget and propagating:
                log.warning(
                    "stage_exceeded_budget_with_propagating_exception",
                    stage=name,
                    elapsed_seconds=elapsed,
                    budget_seconds=budget,
                    propagating_exc_type=type(in_flight_exc).__name__,
                )
            self._current_stage = None
            if not propagating:
                self.check_total_budget()


# --- Public re-exports ---


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "StageBudgets",
    "TimeoutManager",
    "get_default_breakers",
    "reset_default_breakers",
]
