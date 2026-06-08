"""`nautilus bench` — run, compare, and browse benchmark results.

Subcommands
-----------
  run       Execute the benchmark suite and write a ResultSet to disk.
  compare   Diff two ResultSets and flag regressions.
  history   Show trend stats (last N runs) for a metric.
  list      List discovered benchmarks (smoke check that discovery works).
  ingest    Load ResultSets into the performance database for trend analysis.

Examples
--------

  # Run all benchmarks, save to benchmarks/results/bench_<id>.json
  nautilus bench run

  # Run only the matmul kernel on two targets
  nautilus bench run --benchmark kernels/matmul --target nvidia/sm_90 --target cpu

  # Compare the most recent two runs
  nautilus bench compare --latest 2

  # Compare an explicit pair
  nautilus bench compare --baseline results/old.json --candidate results/new.json

  # Show the last 10 runs of matmul exec_time
  nautilus bench history --benchmark kernels/matmul --metric exec_time_s --last 10
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

if TYPE_CHECKING:
    import benchmarks.results as _benchmarks_results_mod
    import benchmarks.runner as _benchmarks_runner_mod
    import src.common.errors as _errors_mod
    import src.common.logging as _logging_mod
else:
    _project_root = Path(__file__).resolve().parent.parent.parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    _benchmarks_results_mod = importlib.import_module("benchmarks.results")
    _benchmarks_runner_mod = importlib.import_module("benchmarks.runner")
    _errors_mod = importlib.import_module("src.common.errors")
    _logging_mod = importlib.import_module("src.common.logging")

DEFAULT_REGRESSION_THRESHOLDS = _benchmarks_results_mod.DEFAULT_REGRESSION_THRESHOLDS
DEFAULT_RESULTS_DIR = _benchmarks_results_mod.DEFAULT_RESULTS_DIR
ComparisonReport = _benchmarks_results_mod.ComparisonReport
ResultSet = _benchmarks_results_mod.ResultSet
compare_result_sets = _benchmarks_results_mod.compare_result_sets
load_history = _benchmarks_results_mod.load_history
trend_summary = _benchmarks_results_mod.trend_summary

# Lazy import for RegressionDetector (avoids circular deps).
_RegressionDetector = None

def _get_detector() -> type:
    global _RegressionDetector
    if _RegressionDetector is None:
        import benchmarks.regression as _reg_mod  # noqa: F811
        _RegressionDetector = _reg_mod.RegressionDetector
    return _RegressionDetector

BenchmarkRunner = _benchmarks_runner_mod.BenchmarkRunner
RunnerConfig = _benchmarks_runner_mod.RunnerConfig

_ingestion_mod = importlib.import_module("benchmarks.ingestion")
BenchmarkIngester = _ingestion_mod.BenchmarkIngester
PerformanceDB = importlib.import_module("src.bridges.triton_tvm.performance_db").PerformanceDB

NautilusError = _errors_mod.NautilusError

get_logger = _logging_mod.get_logger
span_context = _logging_mod.span

log = get_logger("nautilus.cli.bench")


# ---------------------------------------------------------------------------
# Shared option helpers
# ---------------------------------------------------------------------------


def _common_threshold_options(f: Any) -> Any:
    """Attach threshold override flags to a command."""

    @click.option(
        "--exec-threshold",
        type=click.FloatRange(min=0.0, max=5.0),
        default=None,
        help=(
            "Override exec_time regression threshold (fraction). "
            f"Default: {DEFAULT_REGRESSION_THRESHOLDS['exec_time_s']:.2f} "
            "(5%)."
        ),
    )
    @click.option(
        "--compile-threshold",
        type=click.FloatRange(min=0.0, max=5.0),
        default=None,
        help=(
            "Override compile_time regression threshold (fraction). "
            f"Default: {DEFAULT_REGRESSION_THRESHOLDS['compile_time_s']:.2f} "
            "(10%)."
        ),
    )
    @click.option(
        "--binary-size-threshold",
        type=click.FloatRange(min=0.0, max=5.0),
        default=None,
        help=(
            "Override binary_size regression threshold (fraction). "
            f"Default: {DEFAULT_REGRESSION_THRESHOLDS['binary_size_b']:.2f} "
            "(20%)."
        ),
    )
    @click.option(
        "--memory-threshold",
        type=click.FloatRange(min=0.0, max=5.0),
        default=None,
        help=(
            "Override memory regression threshold (fraction). "
            f"Default: {DEFAULT_REGRESSION_THRESHOLDS['memory_mb']:.2f} "
            "(15%)."
        ),
    )
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return f(*args, **kwargs)

    # Click's decorator stack is applied bottom-up. We rebuild manually
    # by chaining. (The clean way is the @click.option chain above, but
    # reusing a helper keeps the per-command bodies short.)
    return wrapper


def _collect_threshold_overrides(
    *,
    exec_threshold: float | None,
    compile_threshold: float | None,
    binary_size_threshold: float | None,
    memory_threshold: float | None,
    bandwidth_threshold: float | None = None,
) -> dict[str, float]:
    """Pack CLI flags into a {metric: fraction} dict for the comparator.

    Maps CLI flag names to the metric keys used by both
    :func:`compare_result_sets` and :class:`RegressionDetector`.
    """
    overrides: dict[str, float] = {}
    if exec_threshold is not None:
        overrides["exec_time_s"] = exec_threshold
    if compile_threshold is not None:
        overrides["compile_time_s"] = compile_threshold
    if binary_size_threshold is not None:
        overrides["binary_size_b"] = binary_size_threshold
    if memory_threshold is not None:
        overrides["memory_mb"] = memory_threshold
    if bandwidth_threshold is not None:
        # The RegressionDetector uses "bandwidth_gbps" directly;
        # legacy compare_result_sets doesn't check bandwidth.
        overrides["bandwidth_gbps"] = bandwidth_threshold
    return overrides


# ---------------------------------------------------------------------------
# Top-level command group
# ---------------------------------------------------------------------------


@click.group(
    name="bench",
    short_help="Run, compare, and browse benchmark results",
    help="""
Benchmark suite for Nautilus. The suite measures auto-tuning
speedups, fat-binary compile times, and end-to-end model latency
across vendors.

Each benchmark records four metrics per (benchmark, vendor) pair:
  - compile_time_s   wall-clock compile / fat-binary link time
  - exec_time_s      median kernel or forward-pass time
  - memory_mb        peak RSS during execution
  - binary_size_b    size of the per-vendor binary blob

A regression is flagged when BOTH the fractional threshold (default
5% exec / 10% compile / 20% binary size / 15% memory) AND the
minimum absolute delta are exceeded. This avoids noise on small
absolute values.
""",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """`nautilus bench` subcommand group."""


# ---------------------------------------------------------------------------
# `nautilus bench run`
# ---------------------------------------------------------------------------


@cli.command(
    "run",
    short_help="Run the benchmark suite and write a result set",
    help="""
Discover all benchmarks (any module ending in ``_bench.py`` exporting
a ``BENCHMARK`` instance) and execute each against its supported
targets. Results are written to ``<NAUTILUS_BENCH_DIR>/bench_<id>.json``
(default: ``benchmarks/results/``).

Per-benchmark failures are recorded as ``status="error"`` or
``status="skipped"`` rather than aborting the whole run; the run only
aborts early if ``--fail-fast`` is set.
""",
)
@click.option(
    "--benchmark",
    "-b",
    "benchmarks",
    multiple=True,
    help="Run only these benchmarks (e.g. 'kernels/matmul'). Repeatable.",
)
@click.option(
    "--target",
    "-t",
    "targets",
    multiple=True,
    help="Run only these targets (e.g. 'nvidia/sm_90'). Repeatable.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help=(
        f"Where to write the result set JSON. "
        f"Default: $NAUTILUS_BENCH_DIR or {DEFAULT_RESULTS_DIR}."
    ),
)
@click.option(
    "--trials",
    "-n",
    type=click.IntRange(min=1, max=10000),
    default=10,
    show_default=True,
    help="Number of timed trials per benchmark (after warmup).",
)
@click.option(
    "--warmup",
    type=click.IntRange(min=0, max=100),
    default=3,
    show_default=True,
    help="Untimed warmup runs before the timed trials.",
)
@click.option(
    "--timeout",
    type=click.FloatRange(min=1.0, max=86400.0),
    default=300.0,
    show_default=True,
    help="Per-benchmark timeout in seconds.",
)
@click.option(
    "--fail-fast/--no-fail-fast",
    default=False,
    help="Abort the whole run on the first benchmark error.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Print the result set as JSON to stdout in addition to writing it.",
)
def run_cmd(
    benchmarks: tuple[str, ...],
    targets: tuple[str, ...],
    output_dir: Path | None,
    trials: int,
    warmup: int,
    timeout: float,
    fail_fast: bool,
    as_json: bool,
) -> None:
    """Run benchmarks."""
    try:
        result_set = _run_impl(
            benchmarks=list(benchmarks) or None,
            targets=list(targets) or None,
            output_dir=output_dir,
            trials=trials,
            warmup=warmup,
            timeout=timeout,
            fail_fast=fail_fast,
        )
    except NautilusError as exc:
        click.echo(f"nautilus: {exc.message}", err=True)
        if exc.context:
            click.echo(f"  context: {exc.context}", err=True)
        sys.exit(2)
    except KeyboardInterrupt:
        click.echo("nautilus: interrupted", err=True)
        sys.exit(130)

    if as_json:
        click.echo(result_set.to_json())
    else:
        ok = sum(1 for r in result_set.results.values() if r.status == "ok")
        skipped = sum(1 for r in result_set.results.values() if r.status == "skipped")
        errors = sum(1 for r in result_set.results.values() if r.status == "error")
        click.echo(
            f"\nResultSet {result_set.run_id} written: "
            f"{ok} ok, {skipped} skipped, {errors} errors "
            f"(total {len(result_set.results)})."
        )
        # Default exit code: 0. CI should compare against a baseline
        # via `nautilus bench compare` for actual gating.
        if errors and fail_fast:
            sys.exit(1)


def _run_impl(
    *,
    benchmarks: list[str] | None,
    targets: list[str] | None,
    output_dir: Path | None,
    trials: int,
    warmup: int,
    timeout: float,
    fail_fast: bool,
) -> ResultSet:
    """Implementation of ``bench run``. Raises :class:`NautilusError`."""
    with span_context("bench_run") as sp:
        sp.set(trials=trials, warmup=warmup, timeout=timeout)
        cfg = RunnerConfig(
            benchmarks_filter=benchmarks,
            targets_filter=targets,
            output_dir=output_dir,
            trials=trials,
            warmup=warmup,
            timeout_s=timeout,
            fail_fast=fail_fast,
        )
        runner = BenchmarkRunner(cfg)
        rs = runner.run_all()
    return rs


# ---------------------------------------------------------------------------
# `nautilus bench compare`
# ---------------------------------------------------------------------------


@cli.command(
    "compare",
    short_help="Compare two result sets and flag regressions",
    help=    """
Diff a baseline ResultSet against a candidate ResultSet and report
regressions / improvements.

By default, the ``RegressionDetector`` is used (``--detector``),
which classifies regressions by severity (major/minor/improvement),
gates findings with statistical significance (``--detector-sigma``),
and detects bandwidth (GB/s) regressions. Pass ``--no-detector`` for
the simpler legacy comparator.

At least one of ``--baseline`` / ``--candidate`` / ``--latest`` is
required. If only ``--latest N`` is given, the N most recent runs
under the results dir are used; the (N-1)th becomes the baseline
and the latest is the candidate. Use ``--direction regressions`` to
filter out improvements (handy for CI gating).
""",
)
@click.option(
    "--baseline",
    "-b",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to the baseline result set JSON.",
)
@click.option(
    "--candidate",
    "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to the candidate result set JSON.",
)
@click.option(
    "--latest",
    "-l",
    type=click.IntRange(min=2, max=1000),
    default=None,
    help=(
        "Use the N most recent runs from the results dir; (N-1)th is baseline, Nth is candidate."
    ),
)
@click.option(
    "--results-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help=(f"Where to find result sets. Default: $NAUTILUS_BENCH_DIR or {DEFAULT_RESULTS_DIR}."),
)
@click.option(
    "--direction",
    type=click.Choice(["all", "regressions", "improvements"], case_sensitive=False),
    default="all",
    show_default=True,
    help="Filter findings by direction.",
)
@click.option(
    "--exec-threshold",
    type=click.FloatRange(min=0.0, max=5.0),
    default=None,
    help=(
        "Override exec_time regression threshold. "
        f"Default: {DEFAULT_REGRESSION_THRESHOLDS['exec_time_s']:.2f} (5%)."
    ),
)
@click.option(
    "--compile-threshold",
    type=click.FloatRange(min=0.0, max=5.0),
    default=None,
    help=(
        "Override compile_time regression threshold. "
        f"Default: {DEFAULT_REGRESSION_THRESHOLDS['compile_time_s']:.2f} "
        "(10%)."
    ),
)
@click.option(
    "--binary-size-threshold",
    type=click.FloatRange(min=0.0, max=5.0),
    default=None,
    help=(
        "Override binary_size regression threshold. "
        f"Default: {DEFAULT_REGRESSION_THRESHOLDS['binary_size_b']:.2f} "
        "(20%)."
    ),
)
@click.option(
    "--memory-threshold",
    type=click.FloatRange(min=0.0, max=5.0),
    default=None,
    help=(
        "Override memory regression threshold. "
        f"Default: {DEFAULT_REGRESSION_THRESHOLDS['memory_mb']:.2f} (15%)."
    ),
)
@click.option(
    "--bandwidth-threshold",
    type=click.FloatRange(min=0.0, max=5.0),
    default=None,
    help=(
        "Override bandwidth (GB/s) regression threshold "
        "(RegressionDetector only). "
        "Default: 0.05 (5%)."
    ),
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["table", "json", "markdown"], case_sensitive=False),
    default="table",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--detector/--no-detector",
    default=True,
    show_default=True,
    help=(
        "Use RegressionDetector for statistical-significance gating "
        "(threshold + 2× stdev) and severity classification."
    ),
)
@click.option(
    "--detector-sigma",
    type=click.IntRange(min=0, max=10),
    default=2,
    show_default=True,
    help=(
        "Number of standard deviations beyond the threshold required "
        "to flag a regression (0 disables statistical gating)."
    ),
)
@click.option(
    "--exit-on-regression/--no-exit-on-regression",
    default=False,
    help="Exit with code 1 if any regression is found (for CI gating).",
)
def compare_cmd(
    baseline: Path | None,
    candidate: Path | None,
    latest: int | None,
    results_dir: Path | None,
    direction: str,
    exec_threshold: float | None,
    compile_threshold: float | None,
    binary_size_threshold: float | None,
    memory_threshold: float | None,
    bandwidth_threshold: float | None,
    fmt: str,
    detector: bool,
    detector_sigma: int,
    exit_on_regression: bool,
) -> None:
    """Compare two result sets."""
    try:
        report, _baseline_rs, _candidate_rs = _compare_impl(
            baseline=baseline,
            candidate=candidate,
            latest=latest,
            results_dir=results_dir,
            direction=direction,
            exec_threshold=exec_threshold,
            compile_threshold=compile_threshold,
            binary_size_threshold=binary_size_threshold,
            memory_threshold=memory_threshold,
            bandwidth_threshold=bandwidth_threshold,
            use_detector=detector,
            detector_sigma=detector_sigma,
        )
    except NautilusError as exc:
        click.echo(f"nautilus: {exc.message}", err=True)
        if exc.context:
            click.echo(f"  context: {exc.context}", err=True)
        sys.exit(2)

    # ── Detect whether result is a list[Regression] (detector path) ──
    is_detector_output = isinstance(report, list)

    if fmt == "json":
        if is_detector_output:
            DetectorCls = _get_detector()
            click.echo(
                DetectorCls().report(
                    report, format="json"  # type: ignore[arg-type]
                )
            )
        else:
            click.echo(json.dumps(report.to_dict(), indent=2))
    elif fmt == "markdown":
        if is_detector_output:
            # Convert detector findings to ComparisonReport for markdown.
            from benchmarks.regression import RegressionDetector

            click.echo(
                _format_markdown(
                    RegressionDetector().to_comparison_report(
                        report, _baseline_rs, _candidate_rs  # type: ignore[arg-type]
                    )
                )
            )
        else:
            click.echo(_format_markdown(report))
    else:
        if is_detector_output:
            DetectorCls = _get_detector()
            click.echo(
                DetectorCls().report(
                    report, format="text"  # type: ignore[arg-type]
                )
            )
        else:
            click.echo(_format_table(report, direction=direction))

    # Determine if regressions exist.
    has_regressions = False
    if is_detector_output:
        has_regressions = any(r.severity != "improvement" for r in report)
    elif hasattr(report, "has_regressions"):
        has_regressions = report.has_regressions

    if exit_on_regression and has_regressions:
        sys.exit(1)


def _compare_impl(
    *,
    baseline: Path | None,
    candidate: Path | None,
    latest: int | None,
    results_dir: Path | None,
    direction: str,
    exec_threshold: float | None,
    compile_threshold: float | None,
    binary_size_threshold: float | None,
    memory_threshold: float | None,
    bandwidth_threshold: float | None = None,
    use_detector: bool = True,
    detector_sigma: int = 2,
) -> tuple[Any, ResultSet, ResultSet]:
    """Implementation of ``bench compare``.

    When ``use_detector`` is True, returns a list of
    :class:`benchmarks.regression.Regression` instead of a
    :class:`ComparisonReport`, enabling statistical-significance
    gating, severity classification, and bandwidth detection.
    """
    rs_dir = (results_dir or DEFAULT_RESULTS_DIR).resolve()
    overrides = _collect_threshold_overrides(
        exec_threshold=exec_threshold,
        compile_threshold=compile_threshold,
        binary_size_threshold=binary_size_threshold,
        memory_threshold=memory_threshold,
        bandwidth_threshold=bandwidth_threshold,
    )
    only_regressions = direction.lower() == "regressions"

    if latest is not None:
        runs = ResultSet.list_runs(rs_dir)
        if len(runs) < 2:
            raise NautilusError(
                f"--latest {latest} requested but only {len(runs)} runs exist in {rs_dir}",
                context={"results_dir": str(rs_dir)},
            )
        # list_runs returns newest-first. Baseline = older reference;
        # candidate = newest run that we want to check for regressions.
        baseline = runs[latest - 1]
        candidate = runs[latest - 2]

    if baseline is None or candidate is None:
        raise NautilusError(
            "Must supply --baseline and --candidate, or --latest N",
            context={"baseline": str(baseline), "candidate": str(candidate)},
        )

    if baseline.resolve() == candidate.resolve():
        raise NautilusError(
            "baseline and candidate are the same file; nothing to compare",
            context={"path": str(baseline)},
        )

    baseline_rs = ResultSet.read(baseline)
    candidate_rs = ResultSet.read(candidate)

    # ── RegressionDetector path (statistical significance + severity) ──
    if use_detector:
        DetectorCls = _get_detector()
        detector = DetectorCls(
            thresholds=overrides or None,
            sigma_threshold=detector_sigma,
        )
        regressions = detector.compare(
            baseline_rs,
            candidate_rs,
            only_regressions=only_regressions,
        )
        if direction.lower() == "improvements":
            regressions = [r for r in regressions if r.severity == "improvement"]
        return regressions, baseline_rs, candidate_rs

    # ── Legacy path (ComparisonReport) ──
    report = compare_result_sets(
        baseline_rs,
        candidate_rs,
        thresholds=overrides or None,
        only_regressions=only_regressions,
    )
    if direction.lower() == "improvements":
        report = ComparisonReport(
            baseline_id=report.baseline_id,
            candidate_id=report.candidate_id,
            findings=[f for f in report.findings if f.direction == "improvement"],
            missing_in_candidate=report.missing_in_candidate,
            missing_in_baseline=report.missing_in_baseline,
            thresholds=report.thresholds,
            min_abs_deltas=report.min_abs_deltas,
        )
    return report, baseline_rs, candidate_rs


# ---------------------------------------------------------------------------
# `nautilus bench history`
# ---------------------------------------------------------------------------


@cli.command(
    "history",
    short_help="Show trend over the last N runs of a metric",
    help="""
Walk the results dir and print summary stats (count, first, last,
mean, median, stdev, percent change) for a single metric.

If ``--benchmark`` and/or ``--vendor`` are omitted, the trend is
computed across all matching entries (so you can ask for "all
matmul results across all vendors").

``--last`` truncates the trend to the most recent N points before
computing stats; ``--all`` disables the truncation.
""",
)
@click.option(
    "--benchmark",
    "-b",
    default=None,
    help="Restrict history to one benchmark (e.g. 'kernels/matmul').",
)
@click.option(
    "--vendor",
    "-v",
    default=None,
    help="Restrict history to one vendor (e.g. 'nvidia/sm_90').",
)
@click.option(
    "--metric",
    "-m",
    type=click.Choice(
        ["exec_time_s", "compile_time_s", "binary_size_b", "memory_mb"],
        case_sensitive=False,
    ),
    default="exec_time_s",
    show_default=True,
    help="Which metric to track.",
)
@click.option(
    "--results-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help=(f"Where to find result sets. Default: $NAUTILUS_BENCH_DIR or {DEFAULT_RESULTS_DIR}."),
)
@click.option(
    "--last",
    type=click.IntRange(min=1, max=10000),
    default=10,
    show_default=True,
    help="How many recent runs to consider.",
)
@click.option(
    "--all",
    "use_all",
    is_flag=True,
    default=False,
    help="Use all runs, ignoring --last.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["table", "json"], case_sensitive=False),
    default="table",
    show_default=True,
    help="Output format.",
)
def history_cmd(
    benchmark: str | None,
    vendor: str | None,
    metric: str,
    results_dir: Path | None,
    last: int,
    use_all: bool,
    fmt: str,
) -> None:
    """Show metric trend over time."""
    rs_dir = (results_dir or DEFAULT_RESULTS_DIR).resolve()
    points = load_history(
        rs_dir,
        benchmark=benchmark,
        vendor=vendor,
        metric=metric,
    )
    if not points:
        click.echo(
            f"no history points found for "
            f"benchmark={benchmark!r} vendor={vendor!r} metric={metric!r} "
            f"in {rs_dir}",
            err=True,
        )
        sys.exit(1)
    last_n = None if use_all else last
    summary = trend_summary(points, last_n=last_n)
    full_count = len(points)

    if fmt == "json":
        click.echo(
            json.dumps(
                {
                    "benchmark": benchmark,
                    "vendor": vendor,
                    "metric": metric,
                    "total_points": full_count,
                    "summary": summary,
                },
                indent=2,
            )
        )
    else:
        _print_history_table(
            benchmark=benchmark,
            vendor=vendor,
            metric=metric,
            summary=summary,
            total_points=full_count,
            last_n=last_n,
        )


def _print_history_table(
    *,
    benchmark: str | None,
    vendor: str | None,
    metric: str,
    summary: dict[str, Any],
    total_points: int,
    last_n: int | None,
) -> None:
    """Human-readable trend summary."""
    if summary.get("count", 0) == 0:
        click.echo("(no points)")
        return
    click.echo(
        f"trend for {benchmark or '*'}/{vendor or '*'} on {metric}\n"
        f"  points: {summary['count']} (of {total_points} total"
        + (f", last_n={last_n}" if last_n else "")
        + ")\n"
        f"  first:  {summary['first']:.6f}\n"
        f"  last:   {summary['last']:.6f}\n"
        f"  delta:  {summary['delta_pct']:+.2f}%\n"
        f"  mean:   {summary['mean']:.6f}\n"
        f"  median: {summary['median']:.6f}\n"
        f"  min:    {summary['min']:.6f}\n"
        f"  max:    {summary['max']:.6f}\n"
        f"  stdev:  {summary['stdev']:.6f}\n"
        f"  first_run_id: {summary['first_run_id']}\n"
        f"  last_run_id:  {summary['last_run_id']}\n"
    )


# ---------------------------------------------------------------------------
# `nautilus bench list` — smoke test for discovery
# ---------------------------------------------------------------------------


@cli.command(
    "list",
    short_help="List discovered benchmarks",
    help="""
Print every benchmark found by the runner's discovery mechanism,
along with its declared targets. Useful as a smoke check that a
new ``<name>_bench.py`` module is being picked up.
""",
)
def list_cmd() -> None:
    """List benchmarks."""
    runner = BenchmarkRunner()
    benches = runner.discover()
    if not benches:
        click.echo("(no benchmarks discovered)")
        return
    for b in benches:
        try:
            targets = b.targets()
        except Exception as exc:
            targets = [f"<error: {exc}>"]
        click.echo(f"{b.name():30s}  targets={','.join(targets)}")


# ---------------------------------------------------------------------------
# `nautilus bench ingest`
# ---------------------------------------------------------------------------


@cli.command(
    "ingest",
    short_help="Ingest benchmark results into the performance database",
    help="""
Load one or more ResultSet JSON files (from ``nautilus bench run``) and
ingest every result into the performance database (``PerformanceDB``)
for historical trend analysis and auto-tuning reference.

At least one of ``--latest`` or ``--path`` is required.

Examples::

    # Ingest the 5 most recent runs from the default results dir
    nautilus bench ingest --latest 5

    # Ingest a specific result file
    nautilus bench ingest --path benchmarks/results/bench_20250607T094215Z.json

    # Use a non-default results dir and a custom database path
    nautilus bench ingest --latest 3 --results-dir /tmp/results --db-path /tmp/perf.db
""",
)
@click.option(
    "--latest",
    "-l",
    type=click.IntRange(min=1, max=10000),
    default=None,
    help="Ingest the N most recent result sets from the results dir.",
)
@click.option(
    "--path",
    "-p",
    "paths",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    multiple=True,
    help="Path to a specific result set JSON file. Repeatable.",
)
@click.option(
    "--results-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help=(
        f"Directory containing result sets. "
        f"Default: $NAUTILUS_BENCH_DIR or {DEFAULT_RESULTS_DIR}."
    ),
)
@click.option(
    "--db-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Path to the PerformanceDB SQLite database. "
        "Default: ~/.cache/nautilus/perf.db."
    ),
)
def ingest_cmd(
    latest: int | None,
    paths: tuple[Path, ...],
    results_dir: Path | None,
    db_path: Path | None,
) -> None:
    try:
        total = _ingest_impl(
            latest=latest,
            paths=list(paths) if paths else None,
            results_dir=results_dir,
            db_path=db_path,
        )
    except NautilusError as exc:
        click.echo(f"nautilus: {exc.message}", err=True)
        if exc.context:
            click.echo(f"  context: {exc.context}", err=True)
        sys.exit(2)
    except KeyboardInterrupt:
        click.echo("nautilus: interrupted", err=True)
        sys.exit(130)

    click.echo(f"Ingested {total} measurements from {_ingest_sources_desc(latest, paths)}")


def _ingest_impl(
    *,
    latest: int | None,
    paths: list[Path] | None,
    results_dir: Path | None,
    db_path: Path | None,
) -> int:
    if not latest and not paths:
        raise NautilusError(
            "At least one of --latest or --path is required",
            context={"latest": latest, "paths": paths},
        )

    rs_dir = (results_dir or DEFAULT_RESULTS_DIR).resolve()
    db = PerformanceDB(db_path) if db_path else PerformanceDB()
    ingester = BenchmarkIngester(db)

    total = 0

    files: list[Path] = []
    if latest:
        all_runs = ResultSet.list_runs(rs_dir)
        if len(all_runs) < latest:
            log.warning(
                "fewer runs available than requested",
                requested=latest,
                available=len(all_runs),
                directory=str(rs_dir),
            )
        files.extend(all_runs[:latest])
    if paths:
        files.extend(Path(p).expanduser().resolve() for p in paths)

    seen: set[Path] = set()
    unique_files: list[Path] = []
    for f in files:
        resolved = f.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_files.append(resolved)

    if not unique_files:
        raise NautilusError(
            "No result set files to ingest",
            context={"results_dir": str(rs_dir), "latest": latest, "paths": paths},
        )

    with span_context("bench_ingest") as sp:
        sp.set(file_count=len(unique_files))
        for f in unique_files:
            count = ingester.ingest_file(f)
            total += count
            sp.set(last_file=str(f), last_count=count)

    return total


def _ingest_sources_desc(latest: int | None, paths: tuple[Path, ...] | None) -> str:
    parts: list[str] = []
    if latest:
        parts.append(f"--latest {latest}")
    if paths:
        parts.append(f"--path ({len(paths)} file(s))")
    return " + ".join(parts) if parts else "(empty)"


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------


def _format_table(report: ComparisonReport, *, direction: str) -> str:
    """Plain-text table of findings."""
    lines: list[str] = []
    lines.append(f"Compare {report.baseline_id} → {report.candidate_id} (direction={direction})")
    lines.append("")
    if not report.findings:
        lines.append("(no findings)")
    else:
        lines.append(
            f"{'benchmark':28s} {'vendor':14s} {'metric':14s} "
            f"{'baseline':>12s} {'candidate':>12s} {'delta':>8s} {'thr':>6s} dir"
        )
        lines.append("-" * 110)
        for f in report.findings:
            arrow = "↓" if f.direction == "improvement" else "↑"
            lines.append(
                f"{f.benchmark:28s} {f.vendor:14s} {f.metric:14s} "
                f"{f.baseline_value:>12.4f} {f.candidate_value:>12.4f} "
                f"{f.delta_pct:>+7.2f}% {f.threshold_pct:>5.1f}% "
                f"{arrow} {f.direction}"
            )
    lines.append("")
    lines.append(
        f"regressions: {report.regression_count}  "
        f"improvements: {report.improvement_count}  "
        f"missing_in_candidate: {len(report.missing_in_candidate)}  "
        f"missing_in_baseline: {len(report.missing_in_baseline)}"
    )
    if report.missing_in_candidate:
        lines.append("  missing in candidate:")
        for b, v in report.missing_in_candidate:
            lines.append(f"    - {b} @ {v}")
    if report.missing_in_baseline:
        lines.append("  missing in baseline:")
        for b, v in report.missing_in_baseline:
            lines.append(f"    - {b} @ {v}")
    return "\n".join(lines)


def _format_markdown(report: ComparisonReport) -> str:
    """Markdown table of findings (for PR comments)."""
    lines: list[str] = []
    lines.append(f"## Bench: {report.baseline_id} → {report.candidate_id}")
    lines.append("")
    if not report.findings:
        lines.append("_No findings._")
    else:
        lines.append(
            "| benchmark | vendor | metric | baseline | candidate | delta | threshold | direction |"
        )
        lines.append("|---|---|---|---:|---:|---:|---:|---|")
        for f in report.findings:
            lines.append(
                f"| {f.benchmark} | {f.vendor} | {f.metric} | "
                f"{f.baseline_value:.4f} | {f.candidate_value:.4f} | "
                f"{f.delta_pct:+.2f}% | {f.threshold_pct:.1f}% | "
                f"{f.direction} |"
            )
    lines.append("")
    lines.append(
        f"**regressions: {report.regression_count}** · "
        f"improvements: {report.improvement_count} · "
        f"missing_in_candidate: {len(report.missing_in_candidate)} · "
        f"missing_in_baseline: {len(report.missing_in_baseline)}"
    )
    return "\n".join(lines)


# Silence the unused-helper lint for the threshold decorator. We keep
# the helper around for the day someone wants to attach a compare
# subcommand to a non-comparing command (e.g. "bench plan").
_ = _common_threshold_options


if __name__ == "__main__":
    cli()
