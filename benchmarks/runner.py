"""Benchmark execution engine.

A :class:`Benchmark` is a unit of work. The runner takes a list of
benchmarks, executes each against a list of targets, and produces a
:class:`benchmarks.results.ResultSet` with one :class:`BenchmarkResult`
per (benchmark, target) pair.

Each benchmark must implement :class:`BenchmarkProtocol` (duck-typed;
the runner doesn't import an abstract base class for speed). The
mandatory methods are:

  - ``name()``                -> str      (e.g. "kernels/matmul")
  - ``targets()``             -> list[str] (e.g. ["nvidia/sm_90","amd/gfx942","cpu"])
  - ``run(target, ctx)``      -> RawRun   (compile + exec + memory + binary)

Where ``ctx`` is a :class:`RunContext` carrying the global flags
(trials, warmup, etc.) and ``RawRun`` is a dict with optional fields
matching :class:`benchmarks.results.BenchmarkResult`.

Graceful degradation
-------------------
If a backend is missing (no CUDA, no ROCm, etc.) the benchmark's
``run()`` should catch :class:`src.common.DependencyMissingError` and
return ``{"status": "skipped", "error": "..."}`` rather than raising.
The runner never lets one vendor's failure kill the whole suite.
"""

from __future__ import annotations

import datetime as _dt
import importlib
import platform
import resource
import socket
import statistics
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from benchmarks.results import (
    BenchmarkResult,
    ResultSet,
    new_run_id,
)
from src.common.logging import get_logger

log = get_logger("nautilus.bench.runner")

from benchmarks.results import SCHEMA_VERSION  # noqa: E402  (intentional late import)


# ---------------------------------------------------------------------------
# Run context & raw measurement types
# ---------------------------------------------------------------------------


@dataclass
class RunContext:
    """Per-run flags shared across all benchmarks."""

    trials: int = 10
    warmup: int = 3
    timeout_s: float | None = 300.0
    output_dir: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def effective_trials(self, default: int = 10) -> int:
        return self.trials or default


# A raw benchmark run is a dict. Optional keys:
#   "compile_time_s", "exec_time_s", "memory_mb", "binary_size_b",
#   "binary" (bytes; will be measured if not provided),
#   "params" (dict), "extras" (dict), "status", "error".
RawRun = dict[str, Any]


class BenchmarkProtocol(Protocol):
    """Duck-typed interface every benchmark must implement."""

    def name(self) -> str: ...
    def targets(self) -> list[str]: ...
    def run(self, target: str, ctx: RunContext) -> RawRun: ...


# ---------------------------------------------------------------------------
# Host / git metadata
# ---------------------------------------------------------------------------


def _safe_git(*args: str) -> str:
    """Run ``git <args>`` and return stdout trimmed, or "" on any failure."""
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.stdout.strip()


def _collect_host_metadata() -> dict[str, str]:
    """Snapshot the host + git state for the ResultSet header."""
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "git_sha": _safe_git("rev-parse", "HEAD"),
        "git_branch": _safe_git("rev-parse", "--abbrev-ref", "HEAD"),
        "nautilus_version": _nautilus_version(),
    }


def _nautilus_version() -> str:
    """Best-effort version lookup. Empty string on any failure."""
    try:
        from src import __version__  # type: ignore[attr-defined]
        return str(__version__)
    except Exception:  # noqa: BLE001 — last-resort, never raise
        return ""


# ---------------------------------------------------------------------------
# Memory probing
# ---------------------------------------------------------------------------


def _read_rss_mb() -> float:
    """Read peak resident-set size of the current process in MiB.

    Uses :func:`resource.getrusage` on POSIX. Returns 0.0 on platforms
    that don't expose it (e.g. native Windows).
    """
    try:
        # ru_maxrss is kilobytes on Linux, bytes on macOS.
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return rss / (1024.0 * 1024.0)
        return rss / 1024.0
    except (OSError, ValueError):
        return 0.0


def _sample_memory_mb(sampler: Callable[[], float] | None = None) -> float:
    """Take a memory sample using ``sampler`` (or RSS by default)."""
    fn = sampler or _read_rss_mb
    try:
        return float(fn())
    except Exception:  # noqa: BLE001
        return 0.0


# ---------------------------------------------------------------------------
# Timer helper — single source of timing truth
# ---------------------------------------------------------------------------


@dataclass
class Timer:
    """Wall-clock timer. ``stop()`` returns elapsed seconds."""

    start_perf: float = field(default_factory=time.perf_counter)
    end_perf: float | None = None

    def stop(self) -> float:
        if self.end_perf is None:
            self.end_perf = time.perf_counter()
        return self.end_perf - self.start_perf


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------


@dataclass
class RunnerConfig:
    """Top-level runner config."""

    targets_filter: list[str] | None = None      # CLI --target filter
    benchmarks_filter: list[str] | None = None   # CLI --benchmark filter
    output_dir: Path | None = None
    trials: int = 10
    warmup: int = 3
    timeout_s: float | None = 300.0
    fail_fast: bool = False
    # When True, raise on benchmark exceptions instead of recording them.
    raise_on_error: bool = False


class BenchmarkRunner:
    """Discovers benchmarks, executes them, and writes a ResultSet."""

    def __init__(self, config: RunnerConfig | None = None) -> None:
        self.config = config or RunnerConfig()
        self._ctx = RunContext(
            trials=self.config.trials,
            warmup=self.config.warmup,
            timeout_s=self.config.timeout_s,
            output_dir=self.config.output_dir,
        )

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @staticmethod
    def discover(package: str = "benchmarks") -> list[BenchmarkProtocol]:
        """Find every object in ``package`` that looks like a benchmark.

        Convention: a module named ``<name>_bench.py`` containing a
        top-level ``BENCHMARK`` instance with the required interface,
        OR a top-level function ``get_benchmark()`` returning one.
        Returning a module-level BENCHMARK instance is the canonical
        pattern; ``get_benchmark()`` is supported for backwards compat.
        """
        import pkgutil

        benchmarks: list[BenchmarkProtocol] = []
        try:
            mod = importlib.import_module(package)
        except ImportError as exc:
            log.warning("benchmark discovery: import failed", package=package, error=str(exc))
            return benchmarks
        for info in pkgutil.walk_packages(mod.__path__, prefix=f"{package}."):
            if not info.name.endswith("_bench"):
                continue
            try:
                bmod = importlib.import_module(info.name)
            except Exception as exc:  # noqa: BLE001
                log.warning("benchmark import failed", module=info.name, error=str(exc))
                continue
            bench_obj = getattr(bmod, "BENCHMARK", None)
            if bench_obj is None and hasattr(bmod, "get_benchmark"):
                try:
                    bench_obj = bmod.get_benchmark()
                except Exception as exc:  # noqa: BLE001
                    log.warning("get_benchmark() raised", module=info.name, error=str(exc))
                    continue
            if bench_obj is None:
                continue
            if not _looks_like_benchmark(bench_obj):
                log.debug("skipping non-benchmark object", module=info.name)
                continue
            benchmarks.append(bench_obj)
        return benchmarks

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_all(self, benchmarks: Iterable[BenchmarkProtocol] | None = None
                ) -> ResultSet:
        """Execute every (benchmark, target) pair and return a ResultSet."""
        all_benchmarks = list(benchmarks) if benchmarks is not None else self.discover()
        all_benchmarks = self._filter_benchmarks(all_benchmarks)

        meta = _collect_host_metadata()
        rs = ResultSet(
            run_id=new_run_id(),
            started_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
            finished_at="",
            git_sha=meta["git_sha"],
            git_branch=meta["git_branch"],
            host=meta["host"],
            nautilus_version=meta["nautilus_version"],
            schema_version=SCHEMA_VERSION,
        )

        for bench in all_benchmarks:
            name = bench.name()
            targets = self._filter_targets(bench.targets())
            log.info("benchmark start", benchmark=name, targets=targets)
            for target in targets:
                result = self._run_one(bench, target)
                rs.add(result)
                log.info(
                    "benchmark done",
                    benchmark=name,
                    target=target,
                    status=result.status,
                    exec_time_s=result.exec_time_s,
                    compile_time_s=result.compile_time_s,
                    memory_mb=result.memory_mb,
                    binary_size_b=result.binary_size_b,
                )
                if result.status == "error" and self.config.fail_fast:
                    if self.config.raise_on_error:
                        raise RuntimeError(f"{name}@{target} failed: {result.error}")
                    log.error("fail-fast enabled; aborting", benchmark=name, target=target)
                    rs.finished_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
                    rs.write(self.config.output_dir)
                    return rs

        rs.finished_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
        out = rs.write(self.config.output_dir)
        log.info("result set written", path=str(out), count=len(rs.results))
        return rs

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _filter_benchmarks(self, benchmarks: list[BenchmarkProtocol]
                           ) -> list[BenchmarkProtocol]:
        flt = self.config.benchmarks_filter
        if not flt:
            return benchmarks
        wanted = set(flt)
        return [b for b in benchmarks if b.name() in wanted or
                any(b.name().endswith(f"/{w}") or b.name() == w for w in wanted)]

    def _filter_targets(self, targets: list[str]) -> list[str]:
        flt = self.config.targets_filter
        if not flt:
            return list(targets)
        return [t for t in targets if t in flt]

    def _run_one(self, bench: BenchmarkProtocol, target: str) -> BenchmarkResult:
        """Execute one (benchmark, target) and convert RawRun -> BenchmarkResult."""
        name = bench.name()
        pre_mem = _sample_memory_mb()
        try:
            raw = bench.run(target, self._ctx)
        except Exception as exc:  # noqa: BLE001 — surface as a result, not a crash
            if self.config.raise_on_error:
                raise
            log.error(
                "benchmark raised",
                benchmark=name, target=target,
                error=str(exc), traceback=traceback.format_exc(),
            )
            return BenchmarkResult(
                benchmark=name, vendor=target,
                status="error", error=f"{type(exc).__name__}: {exc}",
            )
        post_mem = _sample_memory_mb()
        return _raw_to_result(name, target, raw, fallback_memory_mb=post_mem or pre_mem)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _looks_like_benchmark(obj: Any) -> bool:
    """True if ``obj`` exposes name/targets/run attributes."""
    return all(callable(getattr(obj, m, None)) for m in ("name", "targets", "run"))


def _raw_to_result(
    benchmark: str,
    vendor: str,
    raw: Mapping[str, Any],
    *,
    fallback_memory_mb: float,
) -> BenchmarkResult:
    """Convert a RawRun dict into a :class:`BenchmarkResult`.

    Applies post-processing:
      - If ``binary`` bytes are present but no ``binary_size_b``, measure.
      - If ``exec_time_samples`` is a list, derive median + min + max.
      - Coerce numeric strings to floats so the schema is strict.
      - Default ``status`` to "ok" unless set or an error is present.
    """
    status = str(raw.get("status", "ok"))
    if "error" in raw and not raw.get("status"):
        status = "error"
    err = raw.get("error")
    if err is not None and not isinstance(err, str):
        err = str(err)

    # Numeric coercion. We accept ints as well as floats; bools rejected.
    def _num(key: str) -> float | None:
        v = raw.get(key)
        if v is None:
            return None
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    compile_time_s = _num("compile_time_s")
    exec_time_s = _num("exec_time_s")
    memory_mb = _num("memory_mb")
    binary_size_b = raw.get("binary_size_b")
    if binary_size_b is None and isinstance(raw.get("binary"), (bytes, bytearray)):
        binary_size_b = len(raw["binary"])

    # If a sample list was returned, prefer its median for exec_time_s.
    samples = raw.get("exec_time_samples")
    extras = dict(raw.get("extras") or {})
    if isinstance(samples, (list, tuple)) and samples:
        try:
            f_samples = [float(x) for x in samples]
            extras["exec_time_min_s"] = min(f_samples)
            extras["exec_time_max_s"] = max(f_samples)
            extras["exec_time_stdev_s"] = (
                statistics.pstdev(f_samples) if len(f_samples) > 1 else 0.0
            )
            if exec_time_s is None:
                exec_time_s = statistics.median(f_samples)
        except (TypeError, ValueError):
            pass

    # If memory was not measured inside the benchmark, fall back to the
    # RSS sample taken around the call.
    if memory_mb is None:
        memory_mb = fallback_memory_mb

    return BenchmarkResult(
        benchmark=benchmark,
        vendor=vendor,
        status=status,
        error=err,
        compile_time_s=compile_time_s,
        exec_time_s=exec_time_s,
        memory_mb=memory_mb,
        binary_size_b=int(binary_size_b) if binary_size_b is not None else None,
        params=dict(raw.get("params") or {}),
        extras=extras,
    )


# ---------------------------------------------------------------------------
# Convenience: median-of-N timing
# ---------------------------------------------------------------------------


def time_callable(
    fn: Callable[..., Any],
    args: tuple = (),
    kwargs: Mapping[str, Any] | None = None,
    *,
    trials: int,
    warmup: int,
) -> tuple[float, list[float]]:
    """Run ``fn(*args, **kwargs)`` ``trials`` times and return (median, samples).

    Performs ``warmup`` untimed invocations first. The returned list
    contains every timed sample (length == ``trials``). The median is
    :func:`statistics.median` so single outliers don't dominate.
    """
    kwargs = dict(kwargs or {})
    for _ in range(max(0, warmup)):
        fn(*args, **kwargs)
    samples: list[float] = []
    for _ in range(max(1, trials)):
        t = Timer()
        fn(*args, **kwargs)
        samples.append(t.stop())
    return statistics.median(samples), samples


__all__ = [
    "BenchmarkProtocol",
    "BenchmarkRunner",
    "RawRun",
    "RunContext",
    "RunnerConfig",
    "SCHEMA_VERSION",
    "Timer",
    "time_callable",
]


# ---------------------------------------------------------------------------
# Smoke test (run with: ``python -m benchmarks.runner``)
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import json as _json

    runner = BenchmarkRunner(RunnerConfig(trials=3, warmup=1))
    rs = runner.run_all()
    print(_json.dumps(rs.to_dict(), indent=2, default=str))
