"""`nautilus perf` — benchmark performance report and trend analysis.

Subcommands
-----------
  report    Show a performance summary across vendors with trend analysis.

The ``report`` command reads from the PerformanceDB (SQLite) database
populated via ``nautilus bench ingest``.  It groups measurements by
(benchmark_name, vendor, arch) and displays best / average / worst
execution times along with a trend direction (improving / stable /
regressing) for each group.

Examples
--------

  # Show the default report with the last 10 measurements per group
  nautilus perf report

  # Filter to a specific vendor
  nautilus perf report --vendor nvidia

  # Filter to a specific benchmark (exact benchmark_name match)
  nautilus perf report --kernel kernels/matmul

  # Show measurements since a specific date
  nautilus perf report --since 2025-06-01

  # Use the last 20 measurements per group for trend computation
  nautilus perf report --last 20

  # JSON output
  nautilus perf report --format json

  # Combine filters
  nautilus perf report --vendor amd --kernel kernels/matmul --last 5 --format json
"""

from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click

from src.bridges.triton_tvm.performance_db import PerformanceDB

# ---------------------------------------------------------------------------
# Trend classification threshold
# ---------------------------------------------------------------------------

# Fractional threshold for trend detection.  If the mean of the second
# half of measurements differs from the mean of the first half by more
# than this fraction, the trend is classified as "improving" (faster)
# or "regressing" (slower).
_TREND_THRESHOLD = 0.05  # 5%


# ---------------------------------------------------------------------------
# Trend computation helpers
# ---------------------------------------------------------------------------

def _compute_trend(values_sorted: list[float]) -> str:
    """Classify a sorted list of execution times as improving / stable / regressing.

    Splits the sorted (by time) values into two halves and compares
    their means.  A significant decrease means "improving", a
    significant increase means "regressing".

    Args:
        values_sorted: Execution times in **chronological** order
            (oldest first).

    Returns:
        One of ``"improving"``, ``"stable"``, or ``"regressing"``.
    """
    n = len(values_sorted)
    if n < 4:
        # Too few points for a meaningful trend.
        return "stable"

    mid = n // 2
    first_half = values_sorted[:mid]
    second_half = values_sorted[mid:]

    mean_first = statistics.fmean(first_half)
    mean_second = statistics.fmean(second_half)

    if mean_first <= 0:
        return "stable"

    ratio = mean_second / mean_first

    if ratio < (1.0 - _TREND_THRESHOLD):
        return "improving"
    if ratio > (1.0 + _TREND_THRESHOLD):
        return "regressing"
    return "stable"


# ---------------------------------------------------------------------------
# Report data structures
# ---------------------------------------------------------------------------


def _build_report(
    db: PerformanceDB,
    *,
    vendor: str | None = None,
    benchmark_name: str | None = None,
    since: datetime | None = None,
    last_n: int = 10,
) -> list[dict[str, Any]]:
    """Query the PerformanceDB and build the report data.

    Measurements are grouped by ``(benchmark_name, vendor, arch)``.
    For each group the report contains::

        {
            "benchmark": "kernels/matmul",
            "vendor": "nvidia",
            "arch": "sm_90",
            "count": 15,
            "best_ms": 1.23,
            "avg_ms": 1.45,
            "worst_ms": 1.89,
            "trend": "improving",
            "latest_run": "2025-06-07T10:00:00+00:00",
            "first_run": "2025-06-01T08:00:00+00:00",
        }

    Args:
        db: An open PerformanceDB instance.
        vendor: If set, only measurements for this vendor.
        benchmark_name: If set, only measurements matching this benchmark name.
        since: If set, only measurements on or after this timestamp.
        last_n: Maximum number of recent measurements to consider per
            group (default 10).

    Returns:
        A list of group dicts ordered by (benchmark_name, vendor, arch).
    """
    all_measurements = db.query(
        vendor=vendor,
        benchmark_name=benchmark_name,
        since=since,
    )

    # Group by (benchmark_name, vendor, arch).
    groups: dict[tuple[str, str, str], list[tuple[datetime, float]]] = {}
    for m in all_measurements:
        key = (m.benchmark_name or m.kernel_signature[:12], m.vendor, m.arch)
        if key not in groups:
            groups[key] = []
        groups[key].append((m.timestamp, m.execution_time_ms))

    report: list[dict[str, Any]] = []
    for (b_name, v_name, a_name) in sorted(groups.keys()):
        points = groups[(b_name, v_name, a_name)]

        # Sort by timestamp (oldest first) for trend analysis.
        points.sort(key=lambda x: x[0])

        # Apply --last N to keep only the most recent N points.
        if last_n > 0 and len(points) > last_n:
            points = points[-last_n:]

        timestamps = [p[0] for p in points]
        values = [p[1] for p in points]

        best = min(values)
        avg = statistics.fmean(values)
        worst = max(values)
        trend = _compute_trend(values)

        report.append(
            {
                "benchmark": b_name,
                "vendor": v_name,
                "arch": a_name,
                "count": len(values),
                "best_ms": best,
                "avg_ms": avg,
                "worst_ms": worst,
                "trend": trend,
                "latest_run": timestamps[-1].isoformat() if timestamps else "",
                "first_run": timestamps[0].isoformat() if timestamps else "",
            }
        )

    return report


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _format_text(report: list[dict[str, Any]]) -> str:
    """Pretty-print the report as a human-readable table."""
    if not report:
        return "(no data in performance database — run `nautilus bench ingest` first)"

    lines: list[str] = []
    lines.append("Performance Report")
    lines.append("=" * 80)
    lines.append("")

    header = (
        f"{'benchmark':30s} {'vendor':12s} {'arch':12s} "
        f"{'count':>5s} {'best(ms)':>10s} {'avg(ms)':>10s} "
        f"{'worst(ms)':>10s} {'trend':12s}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for entry in report:
        trend_symbol = {
            "improving": "↓ improving",
            "regressing": "↑ regressing",
            "stable": "→ stable",
        }.get(entry["trend"], entry["trend"])

        lines.append(
            f"{entry['benchmark']:30s} {entry['vendor']:12s} {entry['arch']:12s} "
            f"{entry['count']:>5d} {entry['best_ms']:>10.3f} {entry['avg_ms']:>10.3f} "
            f"{entry['worst_ms']:>10.3f} {trend_symbol:12s}"
        )

    lines.append("")
    lines.append(f"Total groups: {len(report)}")
    return "\n".join(lines)


def _format_json(report: list[dict[str, Any]]) -> str:
    """Render the report as pretty-printed JSON."""
    return json.dumps(report, indent=2, sort_keys=False)


# ---------------------------------------------------------------------------
# CLI command group
# ---------------------------------------------------------------------------


@click.group(
    name="perf",
    short_help="Performance reports from the benchmark database",
    help="""
Query the PerformanceDB and generate performance summaries with
trend analysis for every (benchmark, vendor, arch) combination.

The performance database is populated by ``nautilus bench ingest``.
Each measurement records the execution time (in milliseconds) for a
given benchmark on a specific vendor architecture.

The ``report`` subcommand groups measurements and computes:

  - best / average / worst execution time
  - trend direction (improving / stable / regressing)

Trend is determined by splitting the chronological run list in half
and comparing the means.  A >5% improvement (faster) is "improving",
a >5% regression (slower) is "regressing".
""",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """`nautilus perf` subcommand group."""


# ---------------------------------------------------------------------------
# `nautilus perf report`
# ---------------------------------------------------------------------------


@cli.command(
    "report",
    short_help="Show benchmark performance summary with trends",
    help="""
Read from the PerformanceDB and display a summary of execution times
grouped by (benchmark, vendor, arch).

Filters
-------
  --vendor   Restrict to one hardware vendor (e.g. "nvidia", "amd").
  --kernel   Restrict to one benchmark name (e.g. "kernels/matmul").
  --since    Only consider measurements on or after this date.
  --last     Use only the N most recent measurements per group.

Output formats
--------------
  text       Human-readable table (default).
  json       Machine-readable JSON array.

Examples::

    nautilus perf report
    nautilus perf report --vendor nvidia --format json
    nautilus perf report --kernel kernels/matmul --last 20
    nautilus perf report --since 2025-06-01 --vendor amd
""",
)
@click.option(
    "--vendor",
    default=None,
    help="Filter by hardware vendor (e.g. 'nvidia', 'amd', 'intel').",
)
@click.option(
    "--kernel",
    default=None,
    help="Filter by benchmark name (e.g. 'kernels/matmul').",
)
@click.option(
    "--since",
    default=None,
    help="Show data from this date onwards (ISO-8601, e.g. '2025-06-01').",
)
@click.option(
    "--last",
    type=click.IntRange(min=1, max=100000),
    default=10,
    show_default=True,
    help="Number of recent measurements per group to consider.",
)
@click.option(
    "--db-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to the PerformanceDB SQLite database. "
    "Default: ~/.cache/nautilus/perf.db.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    show_default=True,
    help="Output format.",
)
@click.pass_context
def report_cmd(
    ctx: click.Context,
    vendor: str | None,
    kernel: str | None,
    since: str | None,
    last: int,
    db_path: Path | None,
    fmt: str,
) -> None:
    """Show benchmark performance report."""
    # Parse --since if provided.
    since_dt: datetime | None = None
    if since:
        try:
            # Accept date-only and full ISO-8601.
            if "T" in since:
                since_dt = datetime.fromisoformat(since)
            else:
                since_dt = datetime.strptime(since, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
        except (ValueError, TypeError) as exc:
            click.echo(
                f"nautilus: invalid --since value {since!r}: {exc}", err=True
            )
            sys.exit(2)

    try:
        db = PerformanceDB(db_path) if db_path else PerformanceDB()
    except Exception as exc:
        click.echo(
            f"nautilus: failed to open performance database: {exc}", err=True
        )
        sys.exit(2)

    report_data = _build_report(
        db,
        vendor=vendor,
        benchmark_name=kernel,
        since=since_dt,
        last_n=last,
    )

    if fmt == "json":
        click.echo(_format_json(report_data))
    else:
        click.echo(_format_text(report_data))


if __name__ == "__main__":
    cli()
