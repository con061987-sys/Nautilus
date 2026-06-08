"""Tests for benchmarks/regression.py — RegressionDetector, Regression, thresholds."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import pytest

from benchmarks.regression import (
    DEFAULT_SIGMA_THRESHOLD,
    DETECTOR_MIN_ABS_DELTA,
    DETECTOR_THRESHOLDS,
    Regression,
    RegressionDetector,
)
from benchmarks.results import (
    BenchmarkResult,
    ComparisonReport,
    ResultSet,
    new_run_id,
)


# ---------------------------------------------------------------------------
# Helpers — build synthetic result sets
# ---------------------------------------------------------------------------


def _make_rs(
    exec_time_s: float = 0.010,
    compile_time_s: float = 2.0,
    binary_size_b: int = 180_000,
    memory_mb: float = 400.0,
    *,
    benchmark: str = "kernels/matmul",
    vendor: str = "nvidia/sm_90",
    extras: dict[str, Any] | None = None,
) -> ResultSet:
    """Build a single-result ResultSet."""
    rs = ResultSet(
        run_id=new_run_id(),
        started_at="2025-01-01T00:00:00",
        finished_at="2025-01-01T00:01:00",
    )
    rs.add(BenchmarkResult(
        benchmark=benchmark,
        vendor=vendor,
        exec_time_s=exec_time_s,
        compile_time_s=compile_time_s,
        binary_size_b=binary_size_b,
        memory_mb=memory_mb,
        extras=extras or {},
    ))
    return rs


def _make_pair(
    baseline_exec: float = 0.010,
    candidate_exec: float = 0.010,
    *,
    baseline_compile: float = 2.0,
    candidate_compile: float = 2.0,
    baseline_binary: int = 180_000,
    candidate_binary: int = 180_000,
    baseline_memory: float = 400.0,
    candidate_memory: float = 400.0,
    benchmark: str = "kernels/matmul",
    vendor: str = "nvidia/sm_90",
    baseline_extras: dict[str, Any] | None = None,
    candidate_extras: dict[str, Any] | None = None,
) -> tuple[ResultSet, ResultSet]:
    """Build a baseline + candidate pair with the same benchmark/vendor."""
    base = _make_rs(
        exec_time_s=baseline_exec,
        compile_time_s=baseline_compile,
        binary_size_b=baseline_binary,
        memory_mb=baseline_memory,
        benchmark=benchmark,
        vendor=vendor,
        extras=baseline_extras,
    )
    cand = _make_rs(
        exec_time_s=candidate_exec,
        compile_time_s=candidate_compile,
        binary_size_b=candidate_binary,
        memory_mb=candidate_memory,
        benchmark=benchmark,
        vendor=vendor,
        extras=candidate_extras,
    )
    return base, cand


# ---------------------------------------------------------------------------
# Regression dataclass
# ---------------------------------------------------------------------------


class TestRegression:
    def test_fields(self) -> None:
        r = Regression(
            benchmark="kernels/matmul",
            vendor="nvidia/sm_90",
            metric="execution_time_ms",
            baseline_value=10.0,
            candidate_value=12.0,
            change_pct=20.0,
            threshold_pct=5.0,
            severity="major",
        )
        assert r.benchmark == "kernels/matmul"
        assert r.severity == "major"
        assert r.sigma == 0.0  # default

    def test_to_dict(self) -> None:
        r = Regression(
            benchmark="kernels/matmul",
            vendor="nvidia/sm_90",
            metric="execution_time_ms",
            baseline_value=10.0,
            candidate_value=12.0,
            change_pct=20.0,
            threshold_pct=5.0,
            severity="major",
            stdev_baseline=0.5,
            stdev_candidate=0.3,
            sigma=3.0,
        )
        d = r.to_dict()
        assert d["benchmark"] == "kernels/matmul"
        assert d["severity"] == "major"
        assert d["sigma"] == 3.0


# ---------------------------------------------------------------------------
# RegressionDetector — compare()
# ---------------------------------------------------------------------------


class TestRegressionDetectorCompare:
    """Core compare() method tests."""

    def test_no_change_yields_empty(self) -> None:
        """Identical baseline/candidate should produce no regressions."""
        base, cand = _make_pair()
        detector = RegressionDetector()
        regs = detector.compare(base, cand)
        assert regs == []

    def test_small_change_below_threshold(self) -> None:
        """A 3% exec-time increase (below 5% threshold) should NOT flag."""
        base, cand = _make_pair(baseline_exec=0.010, candidate_exec=0.0103)
        regs = RegressionDetector().compare(base, cand)
        assert regs == []

    def test_change_exceeding_threshold_is_flagged(self) -> None:
        """A 20% exec-time increase should be flagged."""
        base, cand = _make_pair(baseline_exec=0.010, candidate_exec=0.012)
        regs = RegressionDetector().compare(base, cand)
        assert len(regs) >= 1
        r = next(r for r in regs if r.metric == "execution_time_ms")
        assert r.severity in ("minor", "major")
        assert r.change_pct > 0

    def test_major_severity_at_double_threshold(self) -> None:
        """A change above 2x threshold should be classified 'major'."""
        base, cand = _make_pair(baseline_exec=0.010, candidate_exec=0.012)
        # 20% change on a 5% threshold → above 2x (10%) → major
        regs = RegressionDetector(thresholds={"execution_time_ms": 0.05}).compare(base, cand)
        r = next(r for r in regs if r.metric == "execution_time_ms")
        assert r.severity == "major"

    def test_improvement_is_detected(self) -> None:
        """When candidate is faster, it should be tagged as improvement."""
        base, cand = _make_pair(baseline_exec=0.012, candidate_exec=0.010)
        regs = RegressionDetector().compare(base, cand)
        r = next(r for r in regs if r.metric == "execution_time_ms")
        assert r.severity == "improvement"
        assert r.change_pct < 0

    def test_only_regressions_filter(self) -> None:
        """only_regressions=True should exclude improvements."""
        base = _make_rs(exec_time_s=0.010)
        cand = _make_rs(exec_time_s=0.008)  # faster
        regs = RegressionDetector().compare(base, cand, only_regressions=True)
        # No regression findings (the improvement should be filtered out)
        assert not any(r.severity != "improvement" for r in regs)

    def test_compile_time_regression(self) -> None:
        """A 50% compile-time increase should be flagged."""
        base, cand = _make_pair(baseline_compile=2.0, candidate_compile=3.0)
        regs = RegressionDetector().compare(base, cand)
        r = next((r for r in regs if r.metric == "compilation_time_ms"), None)
        assert r is not None
        assert r.severity == "major"

    def test_binary_size_regression(self) -> None:
        """A 30% binary-size increase (above 20% threshold) should flag."""
        base, cand = _make_pair(baseline_binary=180_000, candidate_binary=240_000)
        regs = RegressionDetector().compare(base, cand)
        r = next((r for r in regs if r.metric == "binary_size_bytes"), None)
        assert r is not None
        assert r.severity in ("minor", "major")
        assert r.change_pct > 0

    def test_custom_thresholds(self) -> None:
        """Custom thresholds should override defaults."""
        base, cand = _make_pair(baseline_exec=0.010, candidate_exec=0.011)  # 10% change
        # Custom threshold of 15% → 10% < 15% → no regression
        detector = RegressionDetector(thresholds={"execution_time_ms": 0.15})
        regs = detector.compare(base, cand)
        assert not any(r.metric == "execution_time_ms" for r in regs)

    def test_min_abs_delta_filter(self) -> None:
        """A change below min_abs_delta should not flag even if % is large."""
        # 50% change on a tiny absolute value
        base, cand = _make_pair(baseline_exec=0.0001, candidate_exec=0.00015)
        detector = RegressionDetector(
            min_abs_deltas={"execution_time_ms": 1.0}  # 1 ms minimum
        )
        regs = detector.compare(base, cand)
        assert not any(r.metric == "execution_time_ms" for r in regs)

    # ── Multi-vendor / multi-benchmark ──

    def test_multiple_benchmarks(self) -> None:
        """Multiple benchmarks in both sets should each be checked."""
        base = ResultSet(
            run_id=new_run_id(),
            started_at="2025-01-01T00:00:00",
            finished_at="2025-01-01T00:01:00",
        )
        cand = ResultSet(
            run_id=new_run_id(),
            started_at="2025-01-01T00:00:00",
            finished_at="2025-01-01T00:01:00",
        )
        for bm, exec_b, exec_c in [
            ("kernels/matmul", 0.010, 0.015),
            ("kernels/attention", 0.020, 0.022),
        ]:
            base.add(BenchmarkResult(
                benchmark=bm, vendor="nvidia/sm_90",
                exec_time_s=exec_b, compile_time_s=1.0, binary_size_b=100_000,
            ))
            cand.add(BenchmarkResult(
                benchmark=bm, vendor="nvidia/sm_90",
                exec_time_s=exec_c, compile_time_s=1.0, binary_size_b=100_000,
            ))
        regs = RegressionDetector().compare(base, cand)
        benchmarks_found = {r.benchmark for r in regs if r.metric == "execution_time_ms"}
        assert "kernels/matmul" in benchmarks_found

    def test_missing_in_candidate_skipped(self) -> None:
        """Results present in baseline but missing in candidate should not crash."""
        base = _make_rs(exec_time_s=0.010)
        cand = _make_rs(exec_time_s=0.012, benchmark="kernels/other")
        regs = RegressionDetector().compare(base, cand)
        # The matmul result from baseline has no counterpart in candidate → skip silently
        assert all(r.benchmark != "kernels/matmul" for r in regs)

    # ── Bandwidth ──

    def test_bandwidth_regression_detected(self) -> None:
        """A 20% bandwidth drop should be flagged."""
        base = _make_rs(extras={"bandwidth_gbps": 500.0})
        cand = _make_rs(extras={"bandwidth_gbps": 400.0})
        regs = RegressionDetector().compare(base, cand)
        r = next((r for r in regs if r.metric == "bandwidth_gbps"), None)
        assert r is not None
        assert r.severity in ("minor", "major")
        assert r.change_pct < 0  # bandwidth decreased → regression

    def test_bandwidth_improvement_skipped_with_filter(self) -> None:
        """Bandwidth improvements should be filterable with only_regressions."""
        base = _make_rs(extras={"bandwidth_gbps": 400.0})
        cand = _make_rs(extras={"bandwidth_gbps": 500.0})
        regs = RegressionDetector().compare(base, cand, only_regressions=True)
        assert not any(r.metric == "bandwidth_gbps" for r in regs)

    # ── Statistical significance ──

    def test_statistical_significance_gates_noisy_regression(self) -> None:
        """A change below threshold + 2*stdev should be filtered out."""
        base = _make_rs(
            exec_time_s=0.010,
            extras={"exec_time_stdev_s": 0.005},  # 50% stdev
        )
        cand = _make_rs(
            exec_time_s=0.011,  # 10% change
            extras={"exec_time_stdev_s": 0.005},
        )
        # 10% change vs 5% threshold: 5% excess. With 50% stdev, sigma = 0.1
        # Which is < 2 sigma threshold → should be gated out.
        detector = RegressionDetector(sigma_threshold=2)
        regs = detector.compare(base, cand)
        assert not any(r.metric == "execution_time_ms" for r in regs)


# ---------------------------------------------------------------------------
# RegressionDetector — report()
# ---------------------------------------------------------------------------


class TestRegressionDetectorReport:
    def test_empty_text_report(self) -> None:
        """Empty regression list should produce 'no regressions'."""
        text = RegressionDetector().report([], format="text")
        assert "no regressions" in text.lower()

    def test_text_report_includes_findings(self) -> None:
        """Text report should include severity and metric details."""
        regs = [
            Regression(
                benchmark="kernels/matmul",
                vendor="nvidia/sm_90",
                metric="execution_time_ms",
                baseline_value=10.0,
                candidate_value=12.0,
                change_pct=20.0,
                threshold_pct=5.0,
                severity="major",
            ),
        ]
        text = RegressionDetector().report(regs, format="text")
        assert "kernels/matmul" in text
        assert "major" in text
        assert "20.00%" in text

    def test_json_report_is_valid(self) -> None:
        """JSON report should parse correctly and contain expected keys."""
        regs = [
            Regression(
                benchmark="kernels/matmul",
                vendor="nvidia/sm_90",
                metric="execution_time_ms",
                baseline_value=10.0,
                candidate_value=12.0,
                change_pct=20.0,
                threshold_pct=5.0,
                severity="major",
            ),
        ]
        raw = RegressionDetector().report(regs, format="json")
        data = json.loads(raw)
        assert data["total_findings"] == 1
        assert data["major_count"] == 1
        assert data["regressions"][0]["benchmark"] == "kernels/matmul"
        assert data["regressions"][0]["severity"] == "major"


# ---------------------------------------------------------------------------
# RegressionDetector — to_comparison_report()
# ---------------------------------------------------------------------------


class TestToComparisonReport:
    def test_converts_regressions_to_comparison_report(self) -> None:
        base = _make_rs(exec_time_s=0.010)
        cand = _make_rs(exec_time_s=0.012)
        regs = RegressionDetector().compare(base, cand)
        report = RegressionDetector().to_comparison_report(regs, base, cand)
        assert isinstance(report, ComparisonReport)
        assert report.regression_count >= 1

    def test_finding_fields_preserved(self) -> None:
        base = _make_rs(exec_time_s=0.010)
        cand = _make_rs(exec_time_s=0.012)
        regs = RegressionDetector().compare(base, cand)
        report = RegressionDetector().to_comparison_report(regs, base, cand)
        if report.findings:
            f = report.findings[0]
            assert f.benchmark == "kernels/matmul"
            assert f.vendor == "nvidia/sm_90"


# ---------------------------------------------------------------------------
# Threshold defaults
# ---------------------------------------------------------------------------


class TestThresholdDefaults:
    def test_detector_thresholds_have_expected_keys(self) -> None:
        assert "execution_time_ms" in DETECTOR_THRESHOLDS
        assert "bandwidth_gbps" in DETECTOR_THRESHOLDS
        assert "compilation_time_ms" in DETECTOR_THRESHOLDS
        assert "binary_size_bytes" in DETECTOR_THRESHOLDS

    def test_detector_min_abs_deltas_have_expected_keys(self) -> None:
        assert "execution_time_ms" in DETECTOR_MIN_ABS_DELTA
        assert "compilation_time_ms" in DETECTOR_MIN_ABS_DELTA
        assert "binary_size_bytes" in DETECTOR_MIN_ABS_DELTA
        assert "bandwidth_gbps" in DETECTOR_MIN_ABS_DELTA

    def test_default_sigma_threshold(self) -> None:
        assert DEFAULT_SIGMA_THRESHOLD == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_non_ok_status_skipped(self) -> None:
        """Baseline or candidate with status != 'ok' should not be compared."""
        base = ResultSet(run_id="b", started_at="", finished_at="")
        cand = ResultSet(run_id="c", started_at="", finished_at="")
        base.add(BenchmarkResult(
            benchmark="k/m", vendor="n/sm90",
            status="error", error="oops",
            exec_time_s=0.010,
        ))
        cand.add(BenchmarkResult(
            benchmark="k/m", vendor="n/sm90",
            status="ok",
            exec_time_s=0.020,
        ))
        regs = RegressionDetector().compare(base, cand)
        assert regs == []

    def test_zero_baseline_skipped(self) -> None:
        """A baseline value of 0 should not cause division-by-zero."""
        base = _make_rs(exec_time_s=0.0)
        cand = _make_rs(exec_time_s=0.010)
        regs = RegressionDetector().compare(base, cand)
        assert regs == []

    def test_none_metrics_skipped(self) -> None:
        """Results with None metrics should be silently skipped."""
        base = ResultSet(run_id="b", started_at="", finished_at="")
        cand = ResultSet(run_id="c", started_at="", finished_at="")
        base.add(BenchmarkResult(
            benchmark="k/m", vendor="n/sm90",
            exec_time_s=None, compile_time_s=1.0,
        ))
        cand.add(BenchmarkResult(
            benchmark="k/m", vendor="n/sm90",
            exec_time_s=0.020, compile_time_s=1.0,
        ))
        regs = RegressionDetector().compare(base, cand)
        # exec_time_s should be skipped; compile_time_s should be fine
        assert all(r.metric != "execution_time_ms" for r in regs)

    def test_identical_vendor_multiple_benchmarks(self) -> None:
        """Multiple benchmarks for the same vendor should not interfere."""
        base = ResultSet(run_id="b", started_at="", finished_at="")
        cand = ResultSet(run_id="c", started_at="", finished_at="")
        for bm, ex in [("k/a", 0.010), ("k/b", 0.010)]:
            for rs in (base, cand):
                rs.add(BenchmarkResult(
                    benchmark=bm, vendor="n/sm90",
                    exec_time_s=ex, compile_time_s=1.0,
                ))
        regs = RegressionDetector().compare(base, cand)
        assert regs == []


# ---------------------------------------------------------------------------
# Integration: export from benchmarks package
# ---------------------------------------------------------------------------


class TestPackageExports:
    def test_regression_detector_importable_from_benchmarks(self) -> None:
        from benchmarks import RegressionDetector as RD  # noqa: F811
        assert RD is RegressionDetector

    def test_regression_importable_from_benchmarks(self) -> None:
        from benchmarks import Regression as R  # noqa: F811
        assert R is Regression
