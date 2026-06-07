"""Tests for the circuit breaker pattern."""

from __future__ import annotations

import time
from collections import OrderedDict

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.bridges.triton_tvm.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
    LRUCache,
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


class TestLRUCache:
    """Tests for the LRU cache used by the bridge orchestrator."""

    def test_setitem_and_getitem(self) -> None:
        cache: LRUCache[str, int] = LRUCache(maxsize=3)
        cache["a"] = 1
        cache["b"] = 2
        assert cache["a"] == 1
        assert cache["b"] == 2
        assert len(cache) == 2

    def test_capacity_hard_cap(self) -> None:
        cache: LRUCache[int, int] = LRUCache(maxsize=3)
        for i in range(10):
            cache[i] = i * 10
            assert len(cache) <= 3
        assert len(cache) == 3

    def test_eviction_orders(self) -> None:
        """Canonical LRU bug case: re-accessing 'a' protects it from
        being the first eviction; 'b' is the one evicted when 'd' is
        added to a full cache."""
        cache: LRUCache[str, int] = LRUCache(maxsize=3)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        _ = cache["a"]
        cache["d"] = 4
        assert "b" not in cache, "'b' should have been evicted (it was LRU)"
        assert "a" in cache, "'a' was just accessed; it must NOT be evicted"
        assert "c" in cache
        assert "d" in cache
        assert len(cache) == 3

    def test_get_refreshes_recency(self) -> None:
        cache: LRUCache[str, int] = LRUCache(maxsize=2)
        cache["x"] = 1
        cache["y"] = 2
        assert cache.get("x") == 1
        cache["z"] = 3
        assert "x" in cache
        assert "y" not in cache
        assert "z" in cache

    def test_get_miss_does_not_introduce_key(self) -> None:
        cache: LRUCache[str, int] = LRUCache(maxsize=2)
        cache["a"] = 1
        size_before = len(cache)
        assert cache.get("missing", -1) == -1
        assert len(cache) == size_before
        assert "missing" not in cache

    def test_overwrite_existing_key_refreshes_recency(self) -> None:
        cache: LRUCache[str, int] = LRUCache(maxsize=2)
        cache["a"] = 1
        cache["b"] = 2
        cache["a"] = 99
        cache["c"] = 3
        assert "a" in cache
        assert cache["a"] == 99
        assert "b" not in cache
        assert "c" in cache

    def test_invalid_maxsize_raises(self) -> None:
        with pytest.raises(ValueError):
            LRUCache(maxsize=0)
        with pytest.raises(ValueError):
            LRUCache(maxsize=-1)

    def test_eviction_order_matches_recency(self) -> None:
        cache: LRUCache[str, int] = LRUCache(maxsize=3)
        for k in ("a", "b", "c"):
            cache[k] = ord(k)
        cache["d"] = ord("d")
        assert "a" not in cache
        _ = cache["c"]
        cache["e"] = ord("e")
        assert "b" not in cache
        assert "c" in cache
        assert "d" in cache
        assert "e" in cache

    def test_clear_empties_cache(self) -> None:
        cache: LRUCache[str, int] = LRUCache(maxsize=3)
        cache["a"] = 1
        cache["b"] = 2
        cache.clear()
        assert len(cache) == 0
        assert "a" not in cache


_OPS = st.sampled_from(["set", "get", "touch"])
_KEYS = st.integers(min_value=0, max_value=6)
_VALUES = st.integers(min_value=0, max_value=10_000)
_OPERATIONS = st.lists(st.tuples(_KEYS, _VALUES, _OPS), min_size=0, max_size=60)


def _reference_lru(maxsize: int):
    """Reference LRU used as an oracle in property tests."""
    data: OrderedDict[int, int] = OrderedDict()

    def set_item(k: int, v: int) -> None:
        if k in data:
            data.pop(k)
        data[k] = v
        while len(data) > maxsize:
            data.popitem(last=False)

    def get_item(k: int):
        if k in data:
            data.move_to_end(k)
            return data[k]
        return None

    return data, set_item, get_item, get_item


class TestLRUCacheProperties:
    """Hypothesis-driven property tests for LRU invariants.

    A separate reference implementation acts as an oracle: after every
    operation the property cache and the oracle must agree on:
      - the set of keys present
      - the values for each key
      - the cache size
      - the iteration order (LRU → MRU)
    """

    @given(
        maxsize=st.integers(min_value=1, max_value=8),
        ops=_OPERATIONS,
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_lru_invariants(self, maxsize: int, ops: list[tuple[int, int, str]]) -> None:
        cache: LRUCache[int, int] = LRUCache(maxsize=maxsize)
        ref_data, ref_set, ref_get, ref_touch = _reference_lru(maxsize)

        for k, v, op in ops:
            if op == "set":
                cache[k] = v
                ref_set(k, v)
            elif op == "get":
                got = cache.get(k)
                expected = ref_get(k)
                assert got == expected, (
                    f"get({k!r}) returned {got!r}, oracle returned {expected!r}"
                )
            elif op == "touch":
                try:
                    got = cache[k]
                except KeyError:
                    got = None
                expected = ref_touch(k)
                assert got == expected, (
                    f"touch({k!r}) returned {got!r}, oracle returned {expected!r}"
                )

            assert len(cache) <= maxsize, (
                f"size {len(cache)} > maxsize {maxsize} after {op}({k!r})"
            )
            assert set(cache.keys()) == set(ref_data.keys()), (
                f"key sets diverge: cache={set(cache.keys())}, "
                f"ref={set(ref_data.keys())}"
            )
            for key in ref_data:
                assert cache[key] == ref_data[key], (
                    f"value for {key!r} differs: cache={cache[key]!r}, "
                    f"ref={ref_data[key]!r}"
                )
            assert list(cache.keys()) == list(ref_data.keys()), (
                f"iteration order diverges: cache={list(cache.keys())}, "
                f"ref={list(ref_data.keys())}"
            )

    @given(
        maxsize=st.integers(min_value=2, max_value=6),
        touched_idx=st.integers(min_value=0, max_value=10),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_most_recently_used_is_never_first_eviction(
        self, maxsize: int, touched_idx: int
    ) -> None:
        """In a full cache, touching a key makes it the most-recently-used.
        The next insertion must therefore evict the OLDEST key (the
        actual LRU), not the touched key — that is, the touched key is
        never the *first* one to be evicted."""
        cache: LRUCache[int, int] = LRUCache(maxsize=maxsize)
        for k in range(maxsize):
            cache[k] = k * 10
        victim = touched_idx % maxsize
        _ = cache[victim]
        order_before = list(cache.keys())
        expected_lru = order_before[0]
        cache[100] = 100
        assert victim in cache, (
            f"touched key {victim!r} was evicted first; expected eviction "
            f"of LRU {expected_lru!r}. cache now contains {list(cache.keys())!r}"
        )
        assert expected_lru not in cache, (
            f"LRU key {expected_lru!r} should have been evicted, not the "
            f"touched key {victim!r}"
        )
