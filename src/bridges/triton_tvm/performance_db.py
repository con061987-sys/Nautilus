"""Performance database for auto-tuning results.

Stores, queries, and exports historical tuning measurements so the
auto-tuning bridge can learn from past runs and avoid re-tuning kernels
that have already been optimized for a given (vendor, arch) pair.

Storage
-------
* **Local**: SQLite database (default ``~/.cache/nautilus/perf.db``)
  with an indexed ``measurements`` table for efficient lookups.
* **Export**: JSON file via ``export()`` for sharing results across
  machines or CI pipelines.

Thread safety
-------------
SQLite writes are serialised by the database lock.  This module uses
the ``check_same_thread=False`` parameter so that ``store()`` can be
called from any thread without additional synchronisation.  All writes
are wrapped in a transaction (implicit via ``execute``) so concurrent
access from multiple processes is safe as long as they use different
``PerformanceDB`` instances.

Example::

    db = PerformanceDB()
    m = Measurement(
        kernel_signature="abc...",
        vendor="nvidia",
        arch="sm_90",
        config={"block_m": 128, "block_n": 128, "block_k": 32},
        execution_time_ms=1.23,
        bandwidth_gbps=2500.0,
        timestamp=datetime.now(timezone.utc),
        source="auto_tuned",
    )
    db.store(m)
    best = db.lookup("abc...", "nvidia", "sm_90")  # fastest config
    db.export("results.json")
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_DB_PATH = Path.home() / ".cache" / "nautilus" / "perf.db"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS measurements (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    kernel_signature  TEXT    NOT NULL,
    benchmark_name    TEXT    NOT NULL DEFAULT '',
    vendor            TEXT    NOT NULL,
    arch              TEXT    NOT NULL,
    config            TEXT    NOT NULL,  -- JSON-serialised dict
    execution_time_ms REAL    NOT NULL,
    bandwidth_gbps    REAL    NOT NULL,
    timestamp         TEXT    NOT NULL,  -- ISO-8601
    source            TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_perf_lookup
    ON measurements (kernel_signature, vendor, arch, execution_time_ms);

CREATE INDEX IF NOT EXISTS idx_perf_name
    ON measurements (benchmark_name, vendor, arch, execution_time_ms);
"""

_MIGRATE_ADD_BENCHMARK_NAME = """
ALTER TABLE measurements ADD COLUMN benchmark_name TEXT NOT NULL DEFAULT '';
"""

_SELECT_BEST = """
SELECT kernel_signature, benchmark_name, vendor, arch, config,
       execution_time_ms, bandwidth_gbps, timestamp, source
FROM measurements
WHERE kernel_signature = ? AND vendor = ? AND arch = ?
ORDER BY execution_time_ms ASC
LIMIT 1
"""

_INSERT_SQL = """
INSERT INTO measurements
    (kernel_signature, benchmark_name, vendor, arch, config,
     execution_time_ms, bandwidth_gbps, timestamp, source)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class Measurement:
    """A single performance measurement for a tuned kernel.

    Attributes:
        kernel_signature: SHA-256 hash of the kernel IR text.
        benchmark_name: Human-readable benchmark name, e.g. ``"kernels/matmul"``.
        vendor: Hardware vendor string, e.g. ``"nvidia"``, ``"amd"``.
        arch: Architecture identifier, e.g. ``"sm_90"``, ``"gfx942"``.
        config: Tuning configuration (block sizes, warps, stages) as a
            JSON-serialisable dict.  Matches the structure produced by
            ``MappedTuningConfig`` or ``TuningConfig``.
        execution_time_ms: Median execution time in milliseconds.
        bandwidth_gbps: Achieved memory bandwidth in GB/s.
        timestamp: When the measurement was taken.
        source: Provenance of the measurement — one of ``"auto_tuned"``,
            ``"template"``, or ``"user_submitted"``.
    """

    kernel_signature: str
    vendor: str
    arch: str
    config: dict[str, Any]
    execution_time_ms: float
    bandwidth_gbps: float
    timestamp: datetime
    benchmark_name: str = ""
    source: str = "auto_tuned"


# ---------------------------------------------------------------------------
# PerformanceDB
# ---------------------------------------------------------------------------


class PerformanceDB:
    """Persistent store for auto-tuning performance measurements.

    Args:
        db_path: Filesystem path for the SQLite database.  The parent
            directory is created automatically.  Defaults to
            ``~/.cache/nautilus/perf.db``.
    """

    def __init__(self, db_path: str | Path = _DEFAULT_DB_PATH) -> None:
        self._path = Path(db_path).expanduser().resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Thread-local connections so concurrent threads each get their own.
        self._local = threading.local()
        self._init_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(self, measurement: Measurement) -> None:
        """Insert a measurement into the database.

        Duplicates are **not** deduplicated — the caller is expected to
        call ``lookup`` first if they only want to keep the fastest
        config.  This preserves a full history for trend analysis.

        Args:
            measurement: The measurement to persist.
        """
        conn = self._connect()
        conn.execute(
            _INSERT_SQL,
            (
                measurement.kernel_signature,
                measurement.benchmark_name,
                measurement.vendor,
                measurement.arch,
                json.dumps(measurement.config, sort_keys=True) if measurement.config else "{}",
                measurement.execution_time_ms,
                measurement.bandwidth_gbps,
                measurement.timestamp.isoformat(),
                measurement.source,
            ),
        )
        conn.commit()

    def lookup(
        self, kernel_signature: str, vendor: str, arch: str
    ) -> Measurement | None:
        """Return the fastest known config for a kernel on given hardware.

        The fastest config is the one with the lowest
        ``execution_time_ms`` among all measurements matching the
        ``(kernel_signature, vendor, arch)`` triple.

        Args:
            kernel_signature: SHA-256 hash of the kernel IR.
            vendor: Hardware vendor, e.g. ``"nvidia"``.
            arch: Architecture, e.g. ``"sm_90"``.

        Returns:
            The ``Measurement`` with the lowest execution time, or
            ``None`` if no matching measurement exists.
        """
        conn = self._connect()
        cursor = conn.execute(_SELECT_BEST, (kernel_signature, vendor, arch))
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_measurement(row)

    def query(
        self,
        vendor: str | None = None,
        arch: str | None = None,
        kernel_pattern: str | None = None,
        benchmark_name: str | None = None,
        since: datetime | None = None,
    ) -> list[Measurement]:
        """Query measurements with optional filters.

        All filter arguments are ANDed together.  Filters set to
        ``None`` are ignored.

        Args:
            vendor: If set, only measurements for this vendor.
            arch: If set, only measurements for this architecture.
            kernel_pattern: If set, only measurements whose
                ``kernel_signature`` matches ``LIKE '%<pattern>%'``.
                The pattern is matched as a substring (case-sensitive).
            benchmark_name: If set, only measurements whose
                ``benchmark_name`` matches exactly.
            since: If set, only measurements on or after this timestamp.

        Returns:
            A list of matching ``Measurement`` objects, ordered by
            ``timestamp`` ascending (oldest first).
        """
        conn = self._connect()
        conditions: list[str] = []
        params: list[Any] = []

        if vendor is not None:
            conditions.append("vendor = ?")
            params.append(vendor)
        if arch is not None:
            conditions.append("arch = ?")
            params.append(arch)
        if kernel_pattern is not None:
            conditions.append("kernel_signature LIKE ?")
            params.append(f"%{kernel_pattern}%")
        if benchmark_name is not None:
            conditions.append("benchmark_name = ?")
            params.append(benchmark_name)
        if since is not None:
            conditions.append("timestamp >= ?")
            params.append(since.isoformat())

        where = " AND ".join(conditions) if conditions else "1"
        sql = (
            "SELECT kernel_signature, benchmark_name, vendor, arch, config, "
            "       execution_time_ms, bandwidth_gbps, timestamp, source "
            f"FROM measurements WHERE {where} "
            "ORDER BY timestamp ASC"
        )
        cursor = conn.execute(sql, params)
        return [self._row_to_measurement(row) for row in cursor.fetchall()]

    def export(self, output_path: str | Path) -> None:
        """Export all measurements as a JSON array.

        The output is a JSON array of objects, each containing all
        ``Measurement`` fields (with ``config`` already a dict and
        ``timestamp`` as an ISO-8601 string).

        Args:
            output_path: Path for the exported JSON file.  Parent
                directories are created automatically.
        """
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

        all_measurements = self.query()
        data: list[dict[str, Any]] = []
        for m in all_measurements:
            d = asdict(m)
            d["timestamp"] = m.timestamp.isoformat()
            data.append(d)

        output.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create the database and schema if they do not exist."""
        conn = self._connect()
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
        self._migrate_schema(conn)

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        """Apply schema migrations for existing databases."""
        cursor = conn.execute("PRAGMA table_info(measurements)")
        columns = {row[1] for row in cursor.fetchall()}
        if "benchmark_name" not in columns:
            conn.executescript(_MIGRATE_ADD_BENCHMARK_NAME)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        """Get a thread-local SQLite connection.

        Connections are created lazily when first needed in each thread
        and cached in ``self._local``.
        """
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self._path),
                check_same_thread=False,
            )
        return self._local.conn

    @staticmethod
    def _row_to_measurement(row: tuple[Any, ...]) -> Measurement:
        """Convert a SQLite result row to a ``Measurement``."""
        return Measurement(
            kernel_signature=row[0],
            benchmark_name=row[1],
            vendor=row[2],
            arch=row[3],
            config=json.loads(row[4]),
            execution_time_ms=row[5],
            bandwidth_gbps=row[6],
            timestamp=datetime.fromisoformat(row[7]),
            source=row[8],
        )


__all__ = [
    "Measurement",
    "PerformanceDB",
]
