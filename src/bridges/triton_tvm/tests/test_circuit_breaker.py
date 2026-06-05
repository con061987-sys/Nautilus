"""Tests for the circuit breaker pattern."""

from __future__ import annotations

import time

import pytest

from src.bridges.triton_tvm.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
    get_default_breakers,
)


class TestCircuitBreaker:
    """Tests for the CircuitBreaker class."""

    def test_initial_state_is_closed(self) -> None:
        """A fresh circuit breaker should be CLOSED."""
        cb = CircuitBreaker("test_1", CircuitBreakerConfig(failure_threshold=2))
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_successful_call_keeps_closed(self) -> None:
        """Successful calls should not change the state."""
        cb = CircuitBreaker("test_2", CircuitBreakerConfig(failure_threshold=2))

        def success() -> int:
            return 42

        result = cb.call(success)
        assert result == 42
        assert cb.state == CircuitState.CLOSED
        assert cb.total_calls == 1
        assert cb.total_failures == 0

    def test_failures_below_threshold_stay_closed(self) -> None:
        """Failures below the threshold should not trip the circuit."""
        cb = CircuitBreaker("test_3", CircuitBreakerConfig(failure_threshold=3))

        def failing() -> None:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            cb.call(failing)
        with pytest.raises(RuntimeError):
            cb.call(failing)

        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 2
        assert cb.total_failures == 2

    def test_failures_at_threshold_trip_circuit(self) -> None:
        """Reaching the failure threshold should open the circuit."""
        cb = CircuitBreaker("test_4", CircuitBreakerConfig(failure_threshold=3))

        def failing() -> None:
            raise RuntimeError("boom")

        for _ in range(3):
            with pytest.raises(RuntimeError):
                cb.call(failing)

        assert cb.state == CircuitState.OPEN
        assert cb.failure_count == 3

    def test_open_circuit_short_circuits(self) -> None:
        """An open circuit should raise CircuitOpenError without calling."""
        cb = CircuitBreaker("test_5", CircuitBreakerConfig(failure_threshold=1))

        def failing() -> None:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            cb.call(failing)
        assert cb.state == CircuitState.OPEN

        # Next call should be short-circuited
        def never_called() -> None:
            raise AssertionError("should not be called")

        with pytest.raises(CircuitOpenError) as exc_info:
            cb.call(never_called)
        assert cb.total_short_circuits == 1
        assert "test_5" in str(exc_info.value)

    def test_cooldown_transitions_to_half_open(self) -> None:
        """After cooldown, the circuit should transition to HALF_OPEN."""
        cb = CircuitBreaker(
            "test_6",
            CircuitBreakerConfig(failure_threshold=1, cooldown_seconds=0.1),
        )

        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        assert cb.state == CircuitState.OPEN

        time.sleep(0.15)  # Wait for cooldown
        # Next call should be allowed (HALF_OPEN trial)
        result = cb.call(lambda: "ok")
        assert result == "ok"
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self) -> None:
        """A failure in HALF_OPEN should re-open the circuit."""
        cb = CircuitBreaker(
            "test_7",
            CircuitBreakerConfig(failure_threshold=1, cooldown_seconds=0.05),
        )

        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        time.sleep(0.1)
        # Half-open trial fails
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("still broken")))
        assert cb.state == CircuitState.OPEN

    def test_stats_returns_snapshot(self) -> None:
        """The stats property should return a snapshot of the breaker state."""
        cb = CircuitBreaker("test_8", CircuitBreakerConfig(failure_threshold=2))
        cb.call(lambda: "ok")
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))

        stats = cb.stats
        assert stats["name"] == "test_8"
        assert stats["state"] == "CLOSED"
        assert stats["total_calls"] == 2
        assert stats["total_failures"] == 1
        assert stats["failure_count"] == 1

    def test_reset_returns_to_closed(self) -> None:
        """Reset should return the circuit to CLOSED."""
        cb = CircuitBreaker("test_9", CircuitBreakerConfig(failure_threshold=1))
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_unexpected_exceptions_propagate_without_tripping(self) -> None:
        """Exceptions outside expected_exceptions should not affect circuit."""
        cb = CircuitBreaker(
            "test_10",
            CircuitBreakerConfig(
                failure_threshold=2,
                expected_exceptions=(ValueError,),
            ),
        )

        with pytest.raises(KeyError):
            cb.call(lambda: (_ for _ in ()).throw(KeyError("not counted")))

        # KeyError didn't count — circuit should still be closed
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0


class TestDefaultBreakers:
    """Tests for the pre-configured breakers."""

    def test_default_breakers_include_all_dependencies(self) -> None:
        """All expected breakers should be present."""
        breakers = get_default_breakers()
        expected = {
            "triton_compile", "triton_aot", "tvm_tune", "tvm_compile",
            "aotriton", "oneapi", "fat_binary_link", "hardware_validation",
        }
        assert set(breakers.keys()) == expected

    def test_breakers_have_sensible_configs(self) -> None:
        """Each breaker should have a non-default configuration."""
        breakers = get_default_breakers()
        # tvm_tune should have a longer cooldown than triton_compile
        assert breakers["tvm_tune"].config.cooldown_seconds > breakers["triton_compile"].config.cooldown_seconds

    def test_breakers_are_registered(self) -> None:
        """Each default breaker should be in the global registry."""
        breakers = get_default_breakers()
        all_stats = CircuitBreaker.all_stats()
        for name in breakers:
            assert name in all_stats
