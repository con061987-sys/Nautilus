"""Benchmark result ingestion pipeline.

Connects benchmark runner output (ResultSet) to the performance database
(PerformanceDB) by converting BenchmarkResult objects into Measurement
objects and persisting them for historical trend analysis and auto-tuning
reference.

Usage::

    from benchmarks.ingestion import BenchmarkIngester
    from src.bridges.triton_tvm.performance_db import PerformanceDB

    db = PerformanceDB()
    ingester = BenchmarkIngester(db)
    count = ingester.ingest_file("benchmarks/results/bench_20250607T094215Z.json")
    print(f"Ingested {count} measurements")

Or via the CLI::

    nautilus bench ingest --latest 5
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.bridges.triton_tvm.performance_db import Measurement, PerformanceDB
from src.common.logging import get_logger

from benchmarks.results import BenchmarkResult, ResultSet

log = get_logger("nautilus.bench.ingestion")


class BenchmarkIngester:
    """Ingest benchmark results into a :class:`PerformanceDB`.

    Converts each :class:`BenchmarkResult` from a :class:`ResultSet` into
    one or more :class:`Measurement` objects and stores them via the
    provided database handle.

    Args:
        db: An open :class:`PerformanceDB` instance.  The caller is
            responsible for deciding the DB path and lifecycle.
    """

    def __init__(self, db: PerformanceDB) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_result_set(self, result_set: ResultSet) -> int:
        """Ingest every result in a :class:`ResultSet`.

        Each ``BenchmarkResult`` is converted to a ``Measurement`` and
        stored in the database.  Results with ``status="skipped"`` are
        skipped entirely (they represent unavailable backends, not real
        measurements).  Error results are recorded with a zero execution
        time and a descriptive note in the extras.

        Args:
            result_set: The result set to ingest.

        Returns:
            The number of measurements actually stored.
        """
        count = 0
        for result in result_set.results.values():
            m = self._convert(result)
            if m is None:
                continue
            self._db.store(m)
            count += 1
        log.info(
            "ingested result set",
            run_id=result_set.run_id,
            measurements=count,
            total_results=len(result_set.results),
        )
        return count

    def ingest_file(self, path: str | Path) -> int:
        """Load a JSON result file and ingest it.

        Args:
            path: Filesystem path to a ``ResultSet`` JSON file as
                written by ``nautilus bench run``.

        Returns:
            The number of measurements stored.
        """
        p = Path(path).expanduser().resolve()
        if not p.exists():
            log.error("result file not found", path=str(p))
            return 0
        rs = ResultSet.read(p)
        return self.ingest_result_set(rs)

    # ------------------------------------------------------------------
    # Internal: BenchmarkResult -> Measurement conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _split_vendor(vendor: str) -> tuple[str, str]:
        """Split ``"nvidia/sm_90"`` into ``("nvidia", "sm_90")``.

        For vendor strings without a ``/`` (e.g. ``"cpu"``), the
        whole string is used as both vendor and arch.
        """
        if "/" in vendor:
            parts = vendor.split("/", 1)
            return (parts[0].strip(), parts[1].strip())
        return (vendor.strip(), vendor.strip())

    def _convert(self, result: BenchmarkResult) -> Measurement | None:
        """Convert a single ``BenchmarkResult`` to a ``Measurement``.

        Returns ``None`` when the result should be skipped (e.g. status
        is ``"skipped"`` or required fields are missing).
        """
        # --- Status handling ---
        if result.status == "skipped":
            return None

        vendor, arch = self._split_vendor(result.vendor)

        # --- Timestamp ---
        ts: datetime
        if result.timestamp:
            try:
                ts = datetime.fromisoformat(result.timestamp)
            except (ValueError, TypeError):
                ts = datetime.now(timezone.utc)
        else:
            ts = datetime.now(timezone.utc)

        # --- Kernel signature (stable identifier) ---
        # Use a hash of the benchmark name so different benchmark
        # suites produce different signatures while staying stable
        # across runs of the same benchmark.
        sig = hashlib.sha256(result.benchmark.encode("utf-8")).hexdigest()

        # --- Config from params ---
        config: dict[str, Any] = dict(result.params or {})

        # --- Execution time (seconds -> milliseconds) ---
        exec_time_ms: float = 0.0
        if result.exec_time_s is not None:
            exec_time_ms = result.exec_time_s * 1000.0

        # --- Memory (MB) and binary size as config extras ---
        if result.memory_mb is not None:
            config["memory_mb"] = result.memory_mb
        if result.binary_size_b is not None:
            config["binary_size_b"] = result.binary_size_b

        # --- Bandwidth (from extras or estimate) ---
        bandwidth_gbps: float = 0.0
        if result.extras:
            bw = result.extras.get("gbs") or result.extras.get("bandwidth_gbps")
            if bw is not None:
                try:
                    bandwidth_gbps = float(bw)
                except (TypeError, ValueError):
                    pass

        # --- Compile time and error info in extras ---
        extras = dict(result.extras or {})
        if result.status == "error":
            extras["original_status"] = "error"
            extras["error_message"] = result.error or "unknown error"
        if result.compile_time_s is not None:
            extras["compile_time_s"] = result.compile_time_s

        # Merge extras into config so they are preserved in the DB.
        # ``config`` is the ``Measurement.config`` dict, which is
        # serialised as JSON by ``PerformanceDB.store``.
        if extras:
            config["extras"] = extras

        return Measurement(
            kernel_signature=sig,
            vendor=vendor,
            arch=arch,
            config=config,
            execution_time_ms=exec_time_ms,
            bandwidth_gbps=bandwidth_gbps,
            timestamp=ts,
            source="benchmark",
        )


__all__ = [
    "BenchmarkIngester",
]
