"""Result storage, comparison, and history for the benchmark suite.

A benchmark run produces a :class:`ResultSet` — an immutable snapshot
of every (benchmark, vendor) pair measured in that run. ResultSets are
serialized to JSON and written under :data:`DEFAULT_RESULTS_DIR` so
they can be diffed across commits.

Schema
------
Every :class:`BenchmarkResult` carries four metrics:

  - ``compile_time_s``  : wall-clock time to produce the per-vendor
                         binary (or fat-binary link) from a Triton
                         kernel source.  Used to detect compiler
                         regressions.
  - ``exec_time_s``     : median wall-clock kernel execution time
                         over ``--trials`` runs (after warmup).
                         Used to detect performance regressions.
  - ``memory_mb``       : peak working-set memory used during
                         execution. Read from ``/proc/self/status``
                         and (if available) ``nvidia-smi``.
  - ``binary_size_b``   : size of the per-vendor binary blob inside
                         the fat binary. Used to detect
                         "binary-bloat" regressions.

Regression thresholds (defaults; overridable via CLI flags)::

    REGRESSION_THRESHOLDS = {
        "exec_time_s":    0.05,   #  5% slower
        "compile_time_s": 0.10,   # 10% slower
        "binary_size_b":  0.20,   # 20% larger
        "memory_mb":      0.15,   # 15% more memory
    }

A metric is flagged only if BOTH:
  (a) the absolute delta exceeds the threshold *AND*
  (b) the absolute delta is at least ``min_abs_delta`` for the metric
      (avoids noise on tiny absolute values like 0.02s).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------

# Bump on any breaking change to BenchmarkResult / ResultSet fields.
# The loader refuses to read payloads whose version differs, so a bump
# is a forcing function for migration code.
SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Default location for persisted result sets. Overridable via env so
# CI can write into a temp dir without polluting the repo.
DEFAULT_RESULTS_DIR = Path(
    os.environ.get("NAUTILUS_BENCH_DIR", "benchmarks/results")
).resolve()

# File name pattern. ``run_id`` is a UTC timestamp (e.g. "20250607T094215").
RESULT_FILE_TEMPLATE = "bench_{run_id}.json"

# ---------------------------------------------------------------------------
# Regression thresholds
# ---------------------------------------------------------------------------

# Fractional thresholds. Multiplied against the BASELINE value, so
# 0.05 means "5% worse than baseline".
DEFAULT_REGRESSION_THRESHOLDS: dict[str, float] = {
    "exec_time_s": 0.05,
    "compile_time_s": 0.10,
    "binary_size_b": 0.20,
    "memory_mb": 0.15,
}

# Minimum absolute delta (in the metric's own unit) required to flag
# a regression. Stops 0.5ms noise from looking like a 50% regression
# on a 1ms baseline. Keys match the metric names.
DEFAULT_MIN_ABS_DELTA: dict[str, float] = {
    "exec_time_s": 0.001,    # 1ms
    "compile_time_s": 0.05,  # 50ms
    "binary_size_b": 1024,   # 1 KiB
    "memory_mb": 1.0,        # 1 MiB
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkResult:
    """A single (benchmark, vendor) measurement.

    All metrics are optional so we can record partial runs (e.g. a
    kernel compiled for AMD but not Nvidia). ``status`` records the
    outcome: "ok", "skipped", or "error".
    """

    benchmark: str            # e.g. "kernels/matmul" or "models/resnet50"
    vendor: str               # e.g. "nvidia/sm_90", "amd/gfx942", "cpu"
    status: str = "ok"        # "ok" | "skipped" | "error"
    error: str | None = None

    # The four headline metrics. All in SI units (seconds, bytes, MB).
    compile_time_s: float | None = None
    exec_time_s: float | None = None
    memory_mb: float | None = None
    binary_size_b: int | None = None

    # Extra context (shape, num_warps, dtype, etc.). Free-form JSON.
    params: dict[str, Any] = field(default_factory=dict)
    # Sub-metrics (e.g. "tflops", "gbs", "speedup_vs_default"). The
    # primary 4 stay flat on the dataclass for diffability; extras go here.
    extras: dict[str, Any] = field(default_factory=dict)

    # Wall-clock timestamp for *this measurement* (not the run start).
    timestamp: str = field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict.

        ``frozen=True`` means ``asdict`` is safe (no mutable defaults
        can leak shared state).
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BenchmarkResult:
        """Deserialize. Silently drops unknown keys for forward compat."""
        known = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass(frozen=True)
class RegressionFinding:
    """One flagged metric in a comparison."""

    benchmark: str
    vendor: str
    metric: str
    baseline_value: float
    candidate_value: float
    delta_pct: float          # (candidate - baseline) / baseline * 100
    threshold_pct: float      # the threshold applied (e.g. 5.0 for 5%)
    direction: str            # "regression" | "improvement"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ComparisonReport:
    """Output of :func:`compare_result_sets`."""

    baseline_id: str
    candidate_id: str
    findings: list[RegressionFinding] = field(default_factory=list)
    missing_in_candidate: list[tuple[str, str]] = field(default_factory=list)
    missing_in_baseline: list[tuple[str, str]] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)
    min_abs_deltas: dict[str, float] = field(default_factory=dict)

    @property
    def has_regressions(self) -> bool:
        return any(f.direction == "regression" for f in self.findings)

    @property
    def regression_count(self) -> int:
        return sum(1 for f in self.findings if f.direction == "regression")

    @property
    def improvement_count(self) -> int:
        return sum(1 for f in self.findings if f.direction == "improvement")

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_id": self.baseline_id,
            "candidate_id": self.candidate_id,
            "thresholds": dict(self.thresholds),
            "min_abs_deltas": dict(self.min_abs_deltas),
            "findings": [f.to_dict() for f in self.findings],
            "missing_in_candidate": [list(t) for t in self.missing_in_candidate],
            "missing_in_baseline": [list(t) for t in self.missing_in_baseline],
            "has_regressions": self.has_regressions,
            "regression_count": self.regression_count,
            "improvement_count": self.improvement_count,
        }


@dataclass
class ResultSet:
    """A complete benchmark run, keyed by (benchmark, vendor)."""

    run_id: str
    started_at: str
    finished_at: str
    git_sha: str = ""
    git_branch: str = ""
    host: str = ""
    nautilus_version: str = ""
    schema_version: int = SCHEMA_VERSION
    results: dict[tuple[str, str], BenchmarkResult] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add(self, result: BenchmarkResult) -> None:
        """Add or replace a result keyed by (benchmark, vendor)."""
        self.results[(result.benchmark, result.vendor)] = result

    def get(self, benchmark: str, vendor: str) -> BenchmarkResult | None:
        return self.results.get((benchmark, vendor))

    def filter(self, *, benchmark: str | None = None,
               vendor: str | None = None) -> list[BenchmarkResult]:
        out: list[BenchmarkResult] = []
        for (b, v), r in self.results.items():
            if benchmark is not None and b != benchmark:
                continue
            if vendor is not None and v != vendor:
                continue
            out.append(r)
        return out

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "git_sha": self.git_sha,
            "git_branch": self.git_branch,
            "host": self.host,
            "nautilus_version": self.nautilus_version,
            "schema_version": self.schema_version,
            "results": [
                r.to_dict() for r in sorted(
                    self.results.values(),
                    key=lambda r: (r.benchmark, r.vendor),
                )
            ],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ResultSet:
        sv = int(data.get("schema_version", SCHEMA_VERSION))
        if sv != SCHEMA_VERSION:
            raise ValueError(
                f"ResultSet schema version {sv} != current {SCHEMA_VERSION}; "
                f"refusing to load. Migrate with scripts/migrate_results.py."
            )
        results_raw = data.get("results", [])
        if not isinstance(results_raw, list):
            raise ValueError("ResultSet.results must be a list")
        results: dict[tuple[str, str], BenchmarkResult] = {}
        for entry in results_raw:
            r = BenchmarkResult.from_dict(entry)
            results[(r.benchmark, r.vendor)] = r
        return cls(
            run_id=str(data.get("run_id", "")),
            started_at=str(data.get("started_at", "")),
            finished_at=str(data.get("finished_at", "")),
            git_sha=str(data.get("git_sha", "")),
            git_branch=str(data.get("git_branch", "")),
            host=str(data.get("host", "")),
            nautilus_version=str(data.get("nautilus_version", "")),
            schema_version=sv,
            results=results,
        )

    @classmethod
    def from_json(cls, text: str) -> ResultSet:
        return cls.from_dict(json.loads(text))

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def write(self, directory: Path | None = None) -> Path:
        """Write to ``<dir>/bench_<run_id>.json``. Creates ``dir`` if needed."""
        d = (directory or DEFAULT_RESULTS_DIR).resolve()
        d.mkdir(parents=True, exist_ok=True)
        path = d / RESULT_FILE_TEMPLATE.format(run_id=self.run_id)
        path.write_text(self.to_json())
        return path

    @classmethod
    def read(cls, path: Path) -> ResultSet:
        return cls.from_json(path.read_text())

    @classmethod
    def list_runs(cls, directory: Path | None = None) -> list[Path]:
        """List result-set files in ``directory``, newest first."""
        d = (directory or DEFAULT_RESULTS_DIR).resolve()
        if not d.exists():
            return []
        return sorted(d.glob("bench_*.json"), reverse=True)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _resolve_thresholds(
    thresholds: Mapping[str, float] | None,
    min_abs_deltas: Mapping[str, float] | None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Merge user-supplied thresholds with defaults; user wins on conflict."""
    eff_thr = dict(DEFAULT_REGRESSION_THRESHOLDS)
    if thresholds:
        eff_thr.update(thresholds)
    eff_min = dict(DEFAULT_MIN_ABS_DELTA)
    if min_abs_deltas:
        eff_min.update(min_abs_deltas)
    return eff_thr, eff_min


# Metrics where a HIGHER value is worse. Used to set the "direction" tag
# on a finding. If you add a metric, decide which side of the inequality
# is bad and add it here. (This is a closed set, not user-extensible,
# because getting the direction wrong is a silent footgun.)
_WORSE_WHEN_HIGHER: frozenset[str] = frozenset({
    "exec_time_s",
    "compile_time_s",
    "binary_size_b",
    "memory_mb",
})


def compare_result_sets(
    baseline: ResultSet,
    candidate: ResultSet,
    *,
    thresholds: Mapping[str, float] | None = None,
    min_abs_deltas: Mapping[str, float] | None = None,
    only_regressions: bool = False,
) -> ComparisonReport:
    """Diff two ResultSets and produce a :class:`ComparisonReport`.

    For every (benchmark, vendor) present in *both* sets, every metric
    is checked against the configured threshold. Findings are tagged
    "regression" or "improvement"; "ok" deltas within the noise floor
    are dropped.

    Set ``only_regressions=True`` to skip improvement findings (useful
    for CI gating that only cares about red).
    """
    eff_thr, eff_min = _resolve_thresholds(thresholds, min_abs_deltas)
    findings: list[RegressionFinding] = []

    # Iterate over the union of keys so we can also report missing data.
    base_keys = set(baseline.results.keys())
    cand_keys = set(candidate.results.keys())
    common = base_keys & cand_keys

    for key in sorted(common):
        b_name, v_name = key
        b = baseline.results[key]
        c = candidate.results[key]
        # If either side is non-ok, skip metric diff (no meaningful baseline).
        if b.status != "ok" or c.status != "ok":
            continue
        for metric in DEFAULT_REGRESSION_THRESHOLDS:
            base_v = getattr(b, metric, None)
            cand_v = getattr(c, metric, None)
            if base_v is None or cand_v is None:
                continue
            base_f = float(base_v)
            cand_f = float(cand_v)
            if base_f <= 0:
                # No meaningful baseline (zero, negative). Skip.
                continue
            delta_pct = (cand_f - base_f) / base_f * 100.0
            abs_delta = abs(cand_f - base_f)
            thr_pct = eff_thr[metric] * 100.0
            min_abs = eff_min[metric]
            # Flag only if BOTH fractional and absolute thresholds are
            # exceeded. Either alone is not enough.
            if abs(delta_pct) < thr_pct:
                continue
            if abs_delta < min_abs:
                continue
            # Decide direction.
            higher_is_worse = metric in _WORSE_WHEN_HIGHER
            if higher_is_worse:
                direction = "regression" if delta_pct > 0 else "improvement"
            else:
                # Currently no metric in this set, but be explicit so
                # adding a "lower-is-worse" metric later is a 1-line change.
                direction = "regression" if delta_pct < 0 else "improvement"
            if only_regressions and direction != "regression":
                continue
            findings.append(RegressionFinding(
                benchmark=b_name,
                vendor=v_name,
                metric=metric,
                baseline_value=base_f,
                candidate_value=cand_f,
                delta_pct=delta_pct,
                threshold_pct=thr_pct,
                direction=direction,
            ))

    missing_in_candidate = sorted(base_keys - cand_keys)
    missing_in_baseline = sorted(cand_keys - base_keys)

    return ComparisonReport(
        baseline_id=baseline.run_id or "<ad-hoc>",
        candidate_id=candidate.run_id or "<ad-hoc>",
        findings=findings,
        missing_in_candidate=[(b, v) for (b, v) in missing_in_candidate],
        missing_in_baseline=[(b, v) for (b, v) in missing_in_baseline],
        thresholds=eff_thr,
        min_abs_deltas=eff_min,
    )


# ---------------------------------------------------------------------------
# History (trend over time)
# ---------------------------------------------------------------------------


def load_history(
    directory: Path | None = None,
    *,
    benchmark: str | None = None,
    vendor: str | None = None,
    metric: str = "exec_time_s",
) -> list[tuple[ResultSet, BenchmarkResult, float]]:
    """Walk the results dir, return per-run (set, result, metric_value).

    Sorted oldest-first so callers can render a line chart or a text
    trend table without re-sorting. Entries with a missing or
    non-positive metric value are silently dropped — history is for
    trend, not for completeness audit.
    """
    out: list[tuple[ResultSet, BenchmarkResult, float]] = []
    for path in ResultSet.list_runs(directory):
        try:
            rs = ResultSet.read(path)
        except (ValueError, json.JSONDecodeError):
            # Skip corrupt files but keep going.
            continue
        for (b, v), r in rs.results.items():
            if benchmark is not None and b != benchmark:
                continue
            if vendor is not None and v != vendor:
                continue
            value = getattr(r, metric, None)
            if value is None:
                continue
            try:
                fv = float(value)
            except (TypeError, ValueError):
                continue
            if fv <= 0:
                continue
            out.append((rs, r, fv))
    # list_runs returns newest-first; flip for trend display.
    out.sort(key=lambda t: t[0].started_at)
    return out


def trend_summary(
    points: Iterable[tuple[ResultSet, BenchmarkResult, float]],
    *,
    last_n: int | None = 10,
) -> dict[str, Any]:
    """Reduce a history series to a JSON-friendly summary dict.

    Returns the number of points, first/last values, percent change,
    and mean/median. ``last_n`` truncates to the most recent points
    before computing stats (useful for "last 10 runs").
    """
    pts = list(points)
    if last_n is not None and len(pts) > last_n:
        pts = pts[-last_n:]
    if not pts:
        return {"count": 0}
    values = [v for _, _, v in pts]
    first, last = values[0], values[-1]
    delta_pct = ((last - first) / first * 100.0) if first > 0 else 0.0
    return {
        "count": len(pts),
        "first": first,
        "last": last,
        "delta_pct": delta_pct,
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "first_run_id": pts[0][0].run_id,
        "last_run_id": pts[-1][0].run_id,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def new_run_id(now: _dt.datetime | None = None) -> str:
    """Generate a sortable, filename-safe run id like ``20250607T094215Z``."""
    moment = now or _dt.datetime.now(_dt.timezone.utc)
    return moment.strftime("%Y%m%dT%H%M%SZ")


__all__ = [
    "BenchmarkResult",
    "ComparisonReport",
    "DEFAULT_MIN_ABS_DELTA",
    "DEFAULT_REGRESSION_THRESHOLDS",
    "DEFAULT_RESULTS_DIR",
    "RegressionFinding",
    "RESULT_FILE_TEMPLATE",
    "ResultSet",
    "SCHEMA_VERSION",
    "compare_result_sets",
    "load_history",
    "new_run_id",
    "trend_summary",
]
