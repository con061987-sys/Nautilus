"""Tests for the performance database (PerformanceDB)."""

from __future__ import annotations

import contextlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.bridges.triton_tvm.performance_db import Measurement, PerformanceDB

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    """Temporary SQLite database path for testing."""
    return str(tmp_path / "test_perf.db")


@pytest.fixture
def db(db_path: str) -> PerformanceDB:
    """An empty PerformanceDB backed by a temporary SQLite file."""
    return PerformanceDB(db_path)


@pytest.fixture
def sample_measurement() -> Measurement:
    """A typical matmul tuning result for Nvidia H100."""
    return Measurement(
        kernel_signature="abc123def456",
        vendor="nvidia",
        arch="sm_90",
        config={"block_m": 128, "block_n": 128, "block_k": 64, "num_warps": 8, "num_stages": 4},
        execution_time_ms=1.23,
        bandwidth_gbps=2650.0,
        timestamp=datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        source="auto_tuned",
    )


# ---------------------------------------------------------------------------
# Measurement dataclass
# ---------------------------------------------------------------------------


class TestMeasurement:
    """Measurement construction and field types."""

    def test_default_source(self) -> None:
        """source should default to 'auto_tuned'."""
        m = Measurement(
            kernel_signature="sig",
            vendor="amd",
            arch="gfx942",
            config={},
            execution_time_ms=2.5,
            bandwidth_gbps=1800.0,
            timestamp=datetime.now(timezone.utc),
        )
        assert m.source == "auto_tuned"

    def test_round_trip_dict(self) -> None:
        """Measurement should survive dict round-trip via dataclasses.asdict."""
        from dataclasses import asdict

        m = Measurement(
            kernel_signature="sig1",
            vendor="intel",
            arch="gaudi3",
            config={"block_m": 64},
            execution_time_ms=3.1,
            bandwidth_gbps=1200.0,
            timestamp=datetime(2025, 7, 1, tzinfo=timezone.utc),
            source="template",
        )
        d = asdict(m)
        restored = Measurement(**d)
        assert restored == m


# ---------------------------------------------------------------------------
# PerformanceDB: store / lookup
# ---------------------------------------------------------------------------


class TestStoreAndLookup:
    """Storing measurements and looking up the fastest config."""

    def test_store_and_lookup_fastest(self, db: PerformanceDB, sample_measurement: Measurement) -> None:
        """lookup should return the fastest (lowest execution_time_ms)."""
        # Insert slower entry first
        slow = Measurement(
            kernel_signature=sample_measurement.kernel_signature,
            vendor=sample_measurement.vendor,
            arch=sample_measurement.arch,
            config={"block_m": 64, "block_n": 64, "block_k": 32, "num_warps": 4, "num_stages": 3},
            execution_time_ms=3.50,  # slower
            bandwidth_gbps=1800.0,
            timestamp=sample_measurement.timestamp,
            source="auto_tuned",
        )
        db.store(slow)

        # Insert faster entry second
        fast = Measurement(
            kernel_signature=sample_measurement.kernel_signature,
            vendor=sample_measurement.vendor,
            arch=sample_measurement.arch,
            config={"block_m": 128, "block_n": 128, "block_k": 64, "num_warps": 8, "num_stages": 4},
            execution_time_ms=1.23,  # faster
            bandwidth_gbps=2650.0,
            timestamp=sample_measurement.timestamp,
            source="auto_tuned",
        )
        db.store(fast)

        best = db.lookup(
            sample_measurement.kernel_signature,
            sample_measurement.vendor,
            sample_measurement.arch,
        )
        assert best is not None
        assert best.execution_time_ms == pytest.approx(1.23)
        assert best.config["block_m"] == 128

    def test_lookup_no_match(self, db: PerformanceDB) -> None:
        """lookup should return None for non-existent entries."""
        result = db.lookup("nonexistent", "nvidia", "sm_90")
        assert result is None

    def test_lookup_wrong_vendor(self, db: PerformanceDB, sample_measurement: Measurement) -> None:
        """lookup should not return results for a different vendor."""
        db.store(sample_measurement)
        result = db.lookup(sample_measurement.kernel_signature, "amd", "sm_90")
        assert result is None

    def test_lookup_wrong_arch(self, db: PerformanceDB, sample_measurement: Measurement) -> None:
        """lookup should not return results for a different arch."""
        db.store(sample_measurement)
        result = db.lookup(sample_measurement.kernel_signature, "nvidia", "gfx942")
        assert result is None

    def test_lookup_prefers_fastest_over_most_recent(
        self, db: PerformanceDB, sample_measurement: Measurement
    ) -> None:
        """lookup must return the fastest, not the most recent."""
        # Fast + recent
        fast_recent = Measurement(
            kernel_signature=sample_measurement.kernel_signature,
            vendor=sample_measurement.vendor,
            arch=sample_measurement.arch,
            config={"block_m": 256, "block_n": 256, "block_k": 64, "num_warps": 16, "num_stages": 5},
            execution_time_ms=0.95,
            bandwidth_gbps=3100.0,
            timestamp=datetime(2025, 6, 10, tzinfo=timezone.utc),
            source="auto_tuned",
        )
        # Slower + older
        slow_old = Measurement(
            kernel_signature=sample_measurement.kernel_signature,
            vendor=sample_measurement.vendor,
            arch=sample_measurement.arch,
            config={"block_m": 128, "block_n": 128, "block_k": 32, "num_warps": 4, "num_stages": 3},
            execution_time_ms=2.10,
            bandwidth_gbps=1800.0,
            timestamp=datetime(2025, 5, 1, tzinfo=timezone.utc),
            source="auto_tuned",
        )
        # Even faster + old
        fastest_old = Measurement(
            kernel_signature=sample_measurement.kernel_signature,
            vendor=sample_measurement.vendor,
            arch=sample_measurement.arch,
            config={"block_m": 64, "block_n": 128, "block_k": 128, "num_warps": 8, "num_stages": 4},
            execution_time_ms=0.88,
            bandwidth_gbps=3200.0,
            timestamp=datetime(2025, 4, 1, tzinfo=timezone.utc),
            source="auto_tuned",
        )
        db.store(fast_recent)
        db.store(slow_old)
        db.store(fastest_old)

        best = db.lookup(
            sample_measurement.kernel_signature,
            sample_measurement.vendor,
            sample_measurement.arch,
        )
        assert best is not None
        assert best.execution_time_ms == pytest.approx(0.88)
        assert best.config["block_m"] == 64


# ---------------------------------------------------------------------------
# PerformanceDB: query
# ---------------------------------------------------------------------------


class TestQuery:
    """Filtered queries against the performance database."""

    def _seed(self, db: PerformanceDB) -> None:
        """Insert a variety of measurements for query tests."""
        measurements = [
            Measurement("a", "nvidia", "sm_90", {"bm": 128}, 1.0, 1000.0, datetime(2025, 1, 1, tzinfo=timezone.utc), "auto_tuned"),
            Measurement("a", "amd", "gfx942", {"bm": 64}, 2.0, 800.0, datetime(2025, 2, 1, tzinfo=timezone.utc), "auto_tuned"),
            Measurement("b", "nvidia", "sm_90", {"bm": 256}, 0.5, 2000.0, datetime(2025, 3, 1, tzinfo=timezone.utc), "template"),
            Measurement("b", "nvidia", "sm_80", {"bm": 128}, 1.5, 1200.0, datetime(2025, 4, 1, tzinfo=timezone.utc), "user_submitted"),
            Measurement("c", "intel", "gaudi3", {"bm": 64}, 3.0, 600.0, datetime(2025, 5, 1, tzinfo=timezone.utc), "auto_tuned"),
        ]
        for m in measurements:
            db.store(m)

    def test_query_all(self, db: PerformanceDB) -> None:
        """query() with no filters should return all measurements."""
        self._seed(db)
        results = db.query()
        assert len(results) == 5

    def test_query_by_vendor(self, db: PerformanceDB) -> None:
        """query(vendor=...) should filter by vendor."""
        self._seed(db)
        results = db.query(vendor="nvidia")
        assert len(results) == 3
        assert all(m.vendor == "nvidia" for m in results)

    def test_query_by_arch(self, db: PerformanceDB) -> None:
        """query(arch=...) should filter by architecture."""
        self._seed(db)
        results = db.query(arch="sm_90")
        assert len(results) == 2
        assert all(m.arch == "sm_90" for m in results)

    def test_query_by_vendor_and_arch(self, db: PerformanceDB) -> None:
        """query(vendor=..., arch=...) should AND both filters."""
        self._seed(db)
        results = db.query(vendor="nvidia", arch="sm_90")
        assert len(results) == 2
        assert all(m.vendor == "nvidia" and m.arch == "sm_90" for m in results)

    def test_query_by_kernel_pattern(self, db: PerformanceDB) -> None:
        """query(kernel_pattern=...) should substring-match signatures."""
        self._seed(db)
        results = db.query(kernel_pattern="a")
        assert len(results) == 2
        assert all("a" in m.kernel_signature for m in results)

    def test_query_by_since(self, db: PerformanceDB) -> None:
        """query(since=...) should return measurements on or after the date."""
        self._seed(db)
        since = datetime(2025, 3, 15, tzinfo=timezone.utc)
        results = db.query(since=since)
        assert len(results) == 2
        assert all(m.timestamp >= since for m in results)

    def test_query_empty_db(self, db: PerformanceDB) -> None:
        """query() on an empty database should return an empty list."""
        results = db.query()
        assert results == []

    def test_query_order_is_fastest_first(self, db: PerformanceDB) -> None:
        """query results should be ordered by execution_time_ms ascending."""
        self._seed(db)
        results = db.query(vendor="nvidia")
        times = [m.execution_time_ms for m in results]
        assert times == sorted(times)


# ---------------------------------------------------------------------------
# PerformanceDB: export
# ---------------------------------------------------------------------------


class TestExport:
    """JSON export functionality."""

    def test_export_round_trip(self, db: PerformanceDB, sample_measurement: Measurement, tmp_path: Path) -> None:
        """Exported JSON should be parseable and contain all measurements."""
        db.store(sample_measurement)
        out_path = tmp_path / "export.json"
        db.export(str(out_path))

        assert out_path.exists()
        raw = out_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        assert isinstance(data, list)
        assert len(data) == 1
        entry = data[0]
        assert entry["kernel_signature"] == sample_measurement.kernel_signature
        assert entry["vendor"] == sample_measurement.vendor
        assert entry["arch"] == sample_measurement.arch
        assert entry["execution_time_ms"] == sample_measurement.execution_time_ms
        assert entry["bandwidth_gbps"] == sample_measurement.bandwidth_gbps
        assert entry["source"] == sample_measurement.source
        # Timestamp should be ISO-8601 string
        assert isinstance(entry["timestamp"], str)
        parsed = datetime.fromisoformat(entry["timestamp"])
        assert parsed == sample_measurement.timestamp

    def test_export_empty(self, db: PerformanceDB, tmp_path: Path) -> None:
        """Exporting an empty database should produce an empty JSON array."""
        out_path = tmp_path / "empty.json"
        db.export(str(out_path))
        raw = out_path.read_text(encoding="utf-8")
        assert json.loads(raw) == []

    def test_export_creates_parent_dir(self, db: PerformanceDB, sample_measurement: Measurement, tmp_path: Path) -> None:
        """export() should create parent directories."""
        deep_path = tmp_path / "a" / "b" / "c" / "results.json"
        db.store(sample_measurement)
        db.export(str(deep_path))
        assert deep_path.exists()


# ---------------------------------------------------------------------------
# PerformanceDB: thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Concurrent writes should not corrupt the database."""

    def test_concurrent_stores(self, db_path: str) -> None:
        """Multiple threads should be able to store measurements concurrently."""
        import threading

        n_threads = 8
        measurements_per_thread = 32
        barrier = threading.Barrier(n_threads)
        errors: list[Exception] = []

        def _worker(thread_id: int) -> None:
            local_db = PerformanceDB(db_path)
            with contextlib.suppress(threading.BrokenBarrierError):
                barrier.wait(timeout=10)
            for i in range(measurements_per_thread):
                m = Measurement(
                    kernel_signature=f"thread{thread_id}_kernel{i}",
                    vendor="nvidia",
                    arch="sm_90",
                    config={"bm": 128, "bn": 128, "bk": 32, "warps": 8, "stages": 4},
                    execution_time_ms=float(thread_id * 1000 + i),
                    bandwidth_gbps=2000.0,
                    timestamp=datetime.now(timezone.utc),
                    source="auto_tuned",
                )
                try:
                    local_db.store(m)
                except Exception as exc:
                    errors.append(exc)

        threads = [threading.Thread(target=_worker, args=(tid,)) for tid in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent store raised {len(errors)} error(s): {errors[0]}"

        # Verify all entries were written
        final_db = PerformanceDB(db_path)
        total = final_db.query()
        expected = n_threads * measurements_per_thread
        assert len(total) == expected, f"Expected {expected} measurements, got {len(total)}"


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------


class TestProperties:
    """property tests for invariant validation (using parametrised combos)."""

    @pytest.mark.parametrize("vendor", ["nvidia", "amd", "intel", "apple"])
    @pytest.mark.parametrize("arch", ["sm_90", "gfx942", "gaudi3", "m4"])
    def test_store_and_round_trip_preserves_all_fields(
        self, db: PerformanceDB, vendor: str, arch: str
    ) -> None:
        """A stored measurement should have identical fields when retrieved."""
        config = {"block_m": 128, "block_n": 128, "block_k": 32, "num_warps": 8, "num_stages": 4}
        now = datetime.now(timezone.utc)
        m = Measurement(
            kernel_signature=f"kernel_{vendor}_{arch}",
            vendor=vendor,
            arch=arch,
            config=config,
            execution_time_ms=1.5,
            bandwidth_gbps=2000.0,
            timestamp=now,
            source="auto_tuned",
        )
        db.store(m)

        best = db.lookup(m.kernel_signature, vendor, arch)
        assert best is not None
        assert best.kernel_signature == m.kernel_signature
        assert best.vendor == m.vendor
        assert best.arch == m.arch
        assert best.config == m.config
        assert best.execution_time_ms == pytest.approx(m.execution_time_ms)
        assert best.bandwidth_gbps == pytest.approx(m.bandwidth_gbps)
        assert best.timestamp == m.timestamp
        assert best.source == m.source

    def test_idempotent_repeated_store(self, db: PerformanceDB) -> None:
        """Storing the same measurement multiple times should preserve all entries."""
        config = {"block_m": 64, "num_warps": 4}
        m = Measurement(
            kernel_signature="repeat",
            vendor="nvidia",
            arch="sm_90",
            config=config,
            execution_time_ms=1.0,
            bandwidth_gbps=1500.0,
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
            source="auto_tuned",
        )
        for _ in range(10):
            db.store(m)

        results = db.query(kernel_pattern="repeat")
        assert len(results) == 10

    def test_lookup_on_empty_db_returns_none(self, db: PerformanceDB) -> None:
        """lookup on an empty database must return None."""
        for sig in ("", "anything", "a" * 64):
            assert db.lookup(sig, "nvidia", "sm_90") is None

    def test_measurements_from_different_sources_coexist(
        self, db: PerformanceDB
    ) -> None:
        """Same kernel/vendor/arch with different sources should all be stored."""
        sig = "multi_source_kernel"
        for i, src in enumerate(["auto_tuned", "template", "user_submitted"]):
            db.store(
                Measurement(
                    kernel_signature=sig,
                    vendor="nvidia",
                    arch="sm_90",
                    config={"bm": 128},
                    execution_time_ms=float(1 + i),
                    bandwidth_gbps=2000.0,
                    timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
                    source=src,
                )
            )
        results = db.query(kernel_pattern=sig)
        assert len(results) == 3
        sources = {m.source for m in results}
        assert sources == {"auto_tuned", "template", "user_submitted"}

    def test_lookup_returns_fastest_among_many(
        self, db: PerformanceDB
    ) -> None:
        """Lookup scales correctly with many measurements for the same key."""
        sig = "heavy_kernel"
        # Insert 100 measurements with increasing speed
        for i in range(100):
            t_ms = 100.0 - float(i) * 0.5  # 100.0, 99.5, ..., 50.5
            db.store(
                Measurement(
                    kernel_signature=sig,
                    vendor="nvidia",
                    arch="sm_90",
                    config={"iter": i},
                    execution_time_ms=t_ms,
                    bandwidth_gbps=2000.0,
                    timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
                    source="auto_tuned",
                )
            )
        best = db.lookup(sig, "nvidia", "sm_90")
        assert best is not None
        assert best.execution_time_ms == pytest.approx(50.5)
        assert best.config["iter"] == 99  # The 100th entry (0-indexed 99)

    def test_config_round_trip_complex_types(self, db: PerformanceDB) -> None:
        """Configs with nested dicts and lists should survive store/lookup."""
        complex_config = {
            "block_m": 128,
            "block_n": 128,
            "block_k": 64,
            "num_warps": 8,
            "num_stages": 4,
            "num_ctas": 1,
            "matrix_dims": {"m": 4096, "n": 4096, "k": 4096},
            "prefetch_hints": [2, 4, 8],
        }
        m = Measurement(
            kernel_signature="complex_config_kernel",
            vendor="amd",
            arch="gfx942",
            config=complex_config,
            execution_time_ms=2.5,
            bandwidth_gbps=3500.0,
            timestamp=datetime(2025, 6, 1, tzinfo=timezone.utc),
            source="auto_tuned",
        )
        db.store(m)
        best = db.lookup("complex_config_kernel", "amd", "gfx942")
        assert best is not None
        assert best.config == complex_config
