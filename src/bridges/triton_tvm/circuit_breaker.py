"""Circuit breaker pattern for per-dependency failure isolation.

In the bridge pipeline, a single failure (e.g. TVM MetaSchedule hang)
should not bring down the whole compilation. The circuit breaker pattern
isolates failures per dependency so:

  - Triton compile failures don't kill TVM tuning
  - TVM tuning failures don't kill Triton fallback
  - Hardware validation failures don't kill the compile
  - Repeated failures trigger an OPEN circuit that fails fast

States (standard circuit breaker pattern):
  - CLOSED:    normal operation, calls pass through
  - OPEN:      too many failures, calls fail fast
  - HALF_OPEN: after cooldown, allow one call to test recovery

Production-grade: tracks failure count, success count, last failure
time, and provides explicit reset() for operators.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, TypeVar

from src.common.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """State of a circuit breaker."""
    CLOSED = auto()      # Normal operation
    OPEN = auto()        # Failing fast
    HALF_OPEN = auto()   # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker instance."""
    # How many consecutive failures before opening
    failure_threshold: int = 3

    # How long to stay OPEN before trying HALF_OPEN (seconds)
    cooldown_seconds: float = 30.0

    # In HALF_OPEN, how many consecutive successes close the circuit
    success_threshold: int = 1

    # Optional timeout for the wrapped call (seconds)
    call_timeout_s: float | None = None

    # Exception types that count as failures (others propagate without tripping)
    expected_exceptions: tuple[type[BaseException], ...] = field(
        default_factory=lambda: (Exception,)
    )


class CircuitOpenError(Exception):
    """Raised when a call is rejected because the circuit is OPEN."""
    def __init__(self, name: str, cooldown_remaining: float):
        self.name = name
        self.cooldown_remaining = cooldown_remaining
        super().__init__(
            f"Circuit '{name}' is OPEN; retry in {cooldown_remaining:.1f}s"
        )


class CircuitBreaker:
    """Per-dependency circuit breaker for the bridge.

    Usage:
        tvm_breaker = CircuitBreaker("tvm_tune", CircuitBreakerConfig(
            failure_threshold=3,
            cooldown_seconds=60.0,
        ))

        try:
            result = tvm_breaker.call(tune_kernel, ...)
        except CircuitOpenError as exc:
            # Fall back to default config
            ...
        except Exception:
            # Real error from the call
            ...
    """

    # Class-level registry of all breakers (for observability)
    _registry: dict[str, "CircuitBreaker"] = {}

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None) -> None:
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: float = 0.0
        self.last_error: BaseException | None = None
        self.total_calls = 0
        self.total_failures = 0
        self.total_short_circuits = 0
        # Register globally
        CircuitBreaker._registry[name] = self

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Call func through the circuit breaker.

        If the circuit is OPEN, raises CircuitOpenError immediately.
        If the call raises an expected exception, transitions to OPEN
        (after threshold failures) or stays in HALF_OPEN (single trial).
        """
        self.total_calls += 1
        self._maybe_transition_to_half_open()

        if self.state == CircuitState.OPEN:
            self.total_short_circuits += 1
            raise CircuitOpenError(
                self.name,
                self._cooldown_remaining(),
            )

        try:
            result = func(*args, **kwargs)
        except self.config.expected_exceptions as exc:
            self._on_failure(exc)
            raise

        # Success path
        self._on_success()
        return result

    def _on_failure(self, exc: BaseException) -> None:
        """Record a failure and possibly trip the circuit."""
        self.failure_count += 1
        self.total_failures += 1
        self.last_failure_time = time.time()
        self.last_error = exc

        if self.state == CircuitState.HALF_OPEN:
            # Trial failed — go back to OPEN
            self.state = CircuitState.OPEN
            logger.warning(
                "Circuit '%s' HALF_OPEN trial failed; back to OPEN for %.1fs",
                self.name, self.config.cooldown_seconds,
            )
        elif self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                "Circuit '%s' TRIPPED OPEN after %d failures; cooldown %.1fs",
                self.name, self.failure_count, self.config.cooldown_seconds,
            )

    def _on_success(self) -> None:
        """Record a success and possibly close the circuit."""
        self.failure_count = 0  # reset on success
        self.success_count += 1

        if self.state == CircuitState.HALF_OPEN:
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.success_count = 0
                logger.info(
                    "Circuit '%s' CLOSED after %d successful trials",
                    self.name, self.config.success_count_threshold
                    if hasattr(self.config, "success_count_threshold")
                    else self.config.success_threshold,
                )

    def _maybe_transition_to_half_open(self) -> None:
        """If we're in OPEN and cooldown has elapsed, go to HALF_OPEN."""
        if self.state != CircuitState.OPEN:
            return
        if time.time() - self.last_failure_time >= self.config.cooldown_seconds:
            self.state = CircuitState.HALF_OPEN
            self.success_count = 0
            logger.info(
                "Circuit '%s' transitioning OPEN → HALF_OPEN",
                self.name,
            )

    def _cooldown_remaining(self) -> float:
        """Time remaining before we exit OPEN state."""
        elapsed = time.time() - self.last_failure_time
        return max(0.0, self.config.cooldown_seconds - elapsed)

    def reset(self) -> None:
        """Force-reset the circuit to CLOSED."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_error = None

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    @property
    def stats(self) -> dict[str, Any]:
        """Return a snapshot of the breaker's state for observability."""
        return {
            "name": self.name,
            "state": self.state.name,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "total_short_circuits": self.total_short_circuits,
            "last_failure_time": self.last_failure_time,
            "last_error": str(self.last_error) if self.last_error else None,
        }

    @classmethod
    def all_stats(cls) -> dict[str, dict[str, Any]]:
        """Return stats for all registered breakers."""
        return {name: cb.stats for name, cb in cls._registry.items()}


# Pre-configured breakers for each bridge dependency
def get_default_breakers() -> dict[str, CircuitBreaker]:
    """Return the canonical set of circuit breakers for the bridge."""
    return {
        "triton_compile": CircuitBreaker(
            "triton_compile",
            CircuitBreakerConfig(failure_threshold=5, cooldown_seconds=10.0),
        ),
        "triton_aot": CircuitBreaker(
            "triton_aot",
            CircuitBreakerConfig(failure_threshold=3, cooldown_seconds=30.0),
        ),
        "tvm_tune": CircuitBreaker(
            "tvm_tune",
            CircuitBreakerConfig(failure_threshold=3, cooldown_seconds=60.0),
        ),
        "tvm_compile": CircuitBreaker(
            "tvm_compile",
            CircuitBreakerConfig(failure_threshold=3, cooldown_seconds=30.0),
        ),
        "aotriton": CircuitBreaker(
            "aotriton",
            CircuitBreakerConfig(failure_threshold=2, cooldown_seconds=120.0),
        ),
        "oneapi": CircuitBreaker(
            "oneapi",
            CircuitBreakerConfig(failure_threshold=2, cooldown_seconds=120.0),
        ),
        "fat_binary_link": CircuitBreaker(
            "fat_binary_link",
            CircuitBreakerConfig(failure_threshold=2, cooldown_seconds=60.0),
        ),
        "hardware_validation": CircuitBreaker(
            "hardware_validation",
            CircuitBreakerConfig(failure_threshold=5, cooldown_seconds=30.0),
        ),
    }
