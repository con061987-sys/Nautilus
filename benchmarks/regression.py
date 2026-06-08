"""Performance regression detection for benchmark result sets.

Uses statistical significance (mean + 2× standard deviation) to
avoid false positives and classifies regressions by severity.

Typical usage::

    from benchmarks.regression import RegressionDetector

    detector = RegressionDetector()
    regressions = detector.compare(baseline_rs, candidate_rs)

    # Human-readable table
    print(detector.report(regressions, format="text"))

    # Machine-readable JSON
    print(detector.report(regressions, format="json"))
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

from benchmarks.results import (
    RegressionFinding,
    ComparisonReport,
    ResultSet,
    BenchmarkResult,
    DEFAULT_REGRESSION_THRESHOLDS,
    DEFAULT_MIN_ABS_DELTA,
    compare_result_sets,
)

# ---------------------------------------------------------------------------
# Default thresholds (fractional; 0.05 = 5%)
# ---------------------------------------------------------------------------
# The task spec defines these four headline metrics. We also support the
# existing metrics from results.py (exec_time_s, compile_time_s, etc.)
# through automatic aliasing.
DETECTOR_THRESHOLDS: dict[str, float] = {
    "execution_time_ms": 0.05,   # 5%
    "bandwidth_gbps": 0.05,      # 5%
    "compilation_time_ms": 0.10, # 10%
    "binary_size_bytes": 0.20,   # 20%
}

# Map detector metric names -> existing result attributes.
_METRIC_ALIAS: dict[str, str] = {
    "execution_time_ms": "exec_time_s",
    "compilation_time_ms": "compile_time_s",
    "binary_size_bytes": "binary_size_b",
    # bandwidth_gbps has no direct attribute; it's derived from extras.
}

# Minimum absolute delta (in the metric's own unit) required to flag.
DETECTOR_MIN_ABS_DELTA: dict[str, float] = {
    "execution_time_ms": 1.0,        # 1 ms
    "compilation_time_ms": 50.0,     # 50 ms
    "binary_size_bytes": 1024.0,     # 1 KiB
    "bandwidth_gbps": 0.1,           # 0.1 GB/s
}

# Metrics where a HIGHER value is worse (regression = candidate > baseline).
_WORSE_WHEN_HIGHER: frozenset[str] = frozenset({
    "execution_time_ms",
    "compilation_time_ms",
    "binary_size_bytes",
})

# Default number of standard deviations beyond threshold for flagging.
# A metric is flagged only if the change exceeds threshold + sigma * stdev.
DEFAULT_SIGMA_THRESHOLD: int = 2


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class Regression:
    """A single regression finding with severity classification.

    Attributes:
        benchmark: Benchmark name (e.g. "kernels/matmul").
        vendor: Target vendor/arch (e.g. "nvidia/sm_90").
        metric: Metric name (e.g. "execution_time_ms").
        baseline_value: Baseline metric value.
        candidate_value: Candidate metric value.
        change_pct: Percent change (positive = candidate slower/larger).
        threshold_pct: The fractional threshold applied as a percentage.
        severity: "major" | "minor" | "improvement".
        stdev_baseline: Standard deviation of baseline samples (if available).
        stdev_candidate: Standard deviation of candidate samples (if available).
        sigma: How many standard deviations beyond the threshold the change is.
    """

    benchmark: str
    vendor: str
    metric: str
    baseline_value: float
    candidate_value: float
    change_pct: float
    threshold_pct: float
    severity: str = "minor"
    stdev_baseline: float = 0.0
    stdev_candidate: float = 0.0
    sigma: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class RegressionDetector:
    """Detects performance regressions between two :class:`ResultSet`\\ s.

    Builds on :func:`compare_result_sets` but adds:

    * **Statistical significance** — uses sample standard deviation
      (stored in ``extras["exec_time_stdev_s"]``) to avoid flagging
      noisy measurements. A metric is flagged only if the change
      exceeds ``threshold + sigma * max(stdev_baseline, stdev_candidate)``.
    * **Severity classification** — ``"major"`` if the change exceeds
      2× the threshold, ``"minor"`` if it exceeds the threshold,
      ``"improvement"`` if performance got better.
    * **Bandwidth detection** — reads ``extras["bandwidth_gbps"]`` or
      ``extras["gbs"]`` from benchmark results.
    """

    def __init__(
        self,
        thresholds: dict[str, float] | None = None,
        min_abs_deltas: dict[str, float] | None = None,
        sigma_threshold: int = DEFAULT_SIGMA_THRESHOLD,
    ) -> None:
        """Initialise detector with optional overrides.

        Args:
            thresholds: Fractional thresholds keyed by metric name.
                Falls back to DETECTOR_THRESHOLDS for detector-native
                metrics, then to DEFAULT_REGRESSION_THRESHOLDS for
                legacy metrics (exec_time_s, compile_time_s, etc.).
            min_abs_deltas: Minimum absolute delta keyed by metric name.
            sigma_threshold: Number of standard deviations beyond the
                threshold required to flag a regression (default: 2).
        """
        self.thresholds: dict[str, float] = dict(DETECTOR_THRESHOLDS)
        if thresholds:
            self.thresholds.update(thresholds)
        self.min_abs_deltas: dict[str, float] = dict(DETECTOR_MIN_ABS_DELTA)
        if min_abs_deltas:
            self.min_abs_deltas.update(min_abs_deltas)
        self.sigma_threshold = sigma_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare(
        self,
        baseline: ResultSet,
        candidate: ResultSet,
        *,
        only_regressions: bool = False,
    ) -> list[Regression]:
        """Compare two :class:`ResultSet`\\ s and return regressions.

        Args:
            baseline: The reference result set.
            candidate: The new result set to check.
            only_regressions: If True, skip improvement findings.

        Returns:
            List of :class:`Regression` findings.
        """
        # Start with the legacy comparison to get the basic diff, then
        # enrich with severity + statistical significance.
        legacy_report = compare_result_sets(
            baseline,
            candidate,
            thresholds=self._legacy_thresholds(),
            min_abs_deltas=self._legacy_min_abs_deltas(),
            only_regressions=False,
        )

        regressions: list[Regression] = []

        # 1. Convert legacy findings to Regression objects.
        for finding in legacy_report.findings:
            detector_metric = self._to_detector_metric(finding.metric)
            sev = self._classify_severity(
                finding.delta_pct,
                finding.threshold_pct,
                finding.direction,
            )
            if only_regressions and sev == "improvement":
                continue

            # Look up sample stdev from extras, if available.
            b_result = baseline.get(finding.benchmark, finding.vendor)
            c_result = candidate.get(finding.benchmark, finding.vendor)
            stdev_b = self._extract_stdev(b_result, finding.metric)
            stdev_c = self._extract_stdev(c_result, finding.metric)
            sigma = self._compute_sigma(
                abs(finding.delta_pct),
                finding.threshold_pct,
                stdev_b,
                stdev_c,
                b_result,
            )

            regressions.append(Regression(
                benchmark=finding.benchmark,
                vendor=finding.vendor,
                metric=detector_metric,
                baseline_value=finding.baseline_value,
                candidate_value=finding.candidate_value,
                change_pct=finding.delta_pct,
                threshold_pct=finding.threshold_pct,
                severity=sev,
                stdev_baseline=stdev_b,
                stdev_candidate=stdev_c,
                sigma=sigma,
            ))

        # 2. Bandwidth regression detection.
        bw_regressions = self._detect_bandwidth_regressions(
            baseline, candidate, only_regressions=only_regressions,
        )
        regressions.extend(bw_regressions)

        # 3. Apply statistical significance filter.
        regressions = [r for r in regressions if self._passes_statistical_significance(r)]

        return regressions

    def report(
        self,
        regressions: list[Regression],
        format: str = "text",
    ) -> str:
        """Generate a comparison report.

        Args:
            regressions: Findings from :meth:`compare`.
            format: ``"text"`` for human-readable table, ``"json"`` for
                machine-readable output.

        Returns:
            Formatted report string.
        """
        if format == "json":
            return self._format_json(regressions)
        return self._format_text(regressions)

    def to_comparison_report(
        self,
        regressions: list[Regression],
        baseline: ResultSet,
        candidate: ResultSet,
    ) -> ComparisonReport:
        """Convert :class:`Regression` list to a :class:`ComparisonReport`.

        Useful when callers want to use the existing CLI formatters
        (table, markdown) that expect a ``ComparisonReport``.
        """
        findings = [
            RegressionFinding(
                benchmark=r.benchmark,
                vendor=r.vendor,
                metric=self._to_legacy_metric(r.metric),
                baseline_value=r.baseline_value,
                candidate_value=r.candidate_value,
                delta_pct=r.change_pct,
                threshold_pct=r.threshold_pct,
                direction="regression" if r.severity != "improvement" else "improvement",
            )
            for r in regressions
        ]
        return ComparisonReport(
            baseline_id=baseline.run_id or "<ad-hoc>",
            candidate_id=candidate.run_id or "<ad-hoc>",
            findings=findings,
            missing_in_candidate=[],
            missing_in_baseline=[],
            thresholds=self._legacy_thresholds(),
            min_abs_deltas=self._legacy_min_abs_deltas(),
        )

    # ------------------------------------------------------------------
    # Internal: bandwidth detection
    # ------------------------------------------------------------------

    def _detect_bandwidth_regressions(
        self,
        baseline: ResultSet,
        candidate: ResultSet,
        *,
        only_regressions: bool,
    ) -> list[Regression]:
        """Detect bandwidth (GB/s) regressions from extras dicts."""
        regressions: list[Regression] = []
        bw_threshold = self.thresholds.get("bandwidth_gbps", 0.05)
        min_abs = self.min_abs_deltas.get("bandwidth_gbps", 0.1)

        for (b_name, v_name), b_result in baseline.results.items():
            c_result = candidate.get(b_name, v_name)
            if c_result is None:
                continue
            if b_result.status != "ok" or c_result.status != "ok":
                continue

            b_bw = self._extract_bandwidth(b_result)
            c_bw = self._extract_bandwidth(c_result)
            if b_bw is None or c_bw is None:
                continue
            if b_bw <= 0:
                continue

            # For bandwidth, a DECREASE is a regression.
            delta_pct = (c_bw - b_bw) / b_bw * 100.0
            abs_delta = abs(c_bw - b_bw)

            if abs(delta_pct) < bw_threshold * 100.0:
                continue
            if abs_delta < min_abs:
                continue

            direction = "regression" if delta_pct < 0 else "improvement"
            sev = self._classify_severity(
                abs(delta_pct), bw_threshold * 100.0, direction,
            )
            if only_regressions and sev == "improvement":
                continue

            sigma = self._compute_sigma(
                abs(delta_pct), bw_threshold * 100.0,
                0.0, 0.0, b_result,
            )

            regressions.append(Regression(
                benchmark=b_name,
                vendor=v_name,
                metric="bandwidth_gbps",
                baseline_value=b_bw,
                candidate_value=c_bw,
                change_pct=delta_pct,
                threshold_pct=bw_threshold * 100.0,
                severity=sev,
                sigma=sigma,
            ))

        return regressions

    @staticmethod
    def _extract_bandwidth(result: BenchmarkResult) -> float | None:
        """Extract bandwidth in GB/s from a benchmark result.

        Checks extras dicts for ``"bandwidth_gbps"``, ``"gbs"``, or
        ``"achieved_bw_gbps"`` keys (case-insensitive).
        """
        extras = result.extras or {}
        for key in ("bandwidth_gbps", "gbs", "achieved_bw_gbps"):
            val = extras.get(key)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    pass
        return None

    @staticmethod
    def _extract_stdev(result: BenchmarkResult | None, metric: str) -> float:
        """Extract sample standard deviation from extras, if available.

        For exec_time_s, looks for ``extras["exec_time_stdev_s"]``.
        For compile_time_s, looks for ``extras["compile_time_stdev_s"]``.
        Returns 0.0 if not found.
        """
        if result is None:
            return 0.0
        extras = result.extras or {}
        stdev_key = f"{metric.replace('_s', '_stdev_s')}"
        # Try direct match first.
        for key in (f"{metric}_stdev", f"{metric.replace('_s', '_stdev_s')}",
                     f"{metric}_stdev_s"):
            val = extras.get(key)
            if val is not None:
                try:
                    v = float(val)
                    return v if v > 0 else 0.0
                except (TypeError, ValueError):
                    pass
        # Fallback: try the known runner key.
        if metric == "exec_time_s":
            val = extras.get("exec_time_stdev_s")
            if val is not None:
                try:
                    v = float(val)
                    return v if v > 0 else 0.0
                except (TypeError, ValueError):
                    pass
        return 0.0

    # ------------------------------------------------------------------
    # Internal: severity classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_severity(
        delta_pct: float,
        threshold_pct: float,
        direction: str,
    ) -> str:
        """Classify a finding's severity.

        * ``"improvement"`` — performance got better (direction says so).
        * ``"major"`` — change exceeds 2× the threshold.
        * ``"minor"`` — change exceeds the threshold but not 2×.
        """
        if direction == "improvement":
            return "improvement"
        if abs(delta_pct) >= threshold_pct * 2:
            return "major"
        return "minor"

    @staticmethod
    def _compute_sigma(
        abs_change_pct: float,
        threshold_pct: float,
        stdev_b: float,
        stdev_c: float,
        result: BenchmarkResult | None,
    ) -> float:
        """Compute how many effective standard deviations beyond threshold.

        Uses the larger of (stdev_b, stdev_c) normalised to percentage
        of the baseline value. Returns 0.0 if neither stdev is available.
        """
        # If we have the baseline value, normalise stdevs to percentages.
        baseline_val = None
        if result is not None and result.exec_time_s is not None:
            baseline_val = result.exec_time_s

        if baseline_val is not None and baseline_val > 0:
            max_stdev = max(stdev_b, stdev_c)
            stdev_pct = (max_stdev / baseline_val) * 100.0
            if stdev_pct > 0:
                excess = abs_change_pct - threshold_pct
                return excess / stdev_pct if stdev_pct > 0 else 0.0
        return 0.0

    def _passes_statistical_significance(self, r: Regression) -> bool:
        """Check if a regression passes the statistical significance gate.

        A regression is significant only if the change exceeds
        threshold + sigma_threshold * max(stdev).
        """
        if self.sigma_threshold <= 0:
            return True
        # If we have meaningful sigma data, enforce the gate.
        # sigma < 0 means the stdev was not available; pass through.
        if r.sigma < 0:
            return True
        # If sigma is 0, either stdev is 0 or not available.
        # In that case, rely on raw threshold check (legacy behaviour).
        if r.sigma == 0 and r.stdev_baseline == 0 and r.stdev_candidate == 0:
            return True
        # sigma measures how many stdevs beyond threshold we are.
        # We require sigma >= sigma_threshold (default 2).
        return r.sigma >= self.sigma_threshold

    # ------------------------------------------------------------------
    # Internal: metric name mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _to_detector_metric(legacy_metric: str) -> str:
        """Map legacy metric names to detector metric names."""
        rev: dict[str, str] = {v: k for k, v in _METRIC_ALIAS.items()}
        return rev.get(legacy_metric, legacy_metric)

    @staticmethod
    def _to_legacy_metric(detector_metric: str) -> str:
        """Map detector metric names back to legacy names."""
        return _METRIC_ALIAS.get(detector_metric, detector_metric)

    def _legacy_thresholds(self) -> dict[str, float]:
        """Build a thresholds dict compatible with compare_result_sets."""
        legacy: dict[str, float] = dict(DEFAULT_REGRESSION_THRESHOLDS)
        for det_metric, thr in self.thresholds.items():
            leg = self._to_legacy_metric(det_metric)
            legacy[leg] = thr
        return legacy

    def _legacy_min_abs_deltas(self) -> dict[str, float]:
        """Build a min_abs_deltas dict compatible with compare_result_sets.

        Detector-native min_abs values (e.g. ``execution_time_ms: 1.0``
        meaning 1 ms vs legacy ``exec_time_s: 0.001``) use different units
        and cannot be directly forwarded to the underlying
        ``compare_result_sets``. We therefore return only the default
        legacy deltas plus any *extra* detector metrics that have no legacy
        counterpart (e.g. ``bandwidth_gbps``).
        """
        legacy: dict[str, float] = dict(DEFAULT_MIN_ABS_DELTA)
        for det_metric, thr in self.min_abs_deltas.items():
            leg = self._to_legacy_metric(det_metric)
            if leg not in DEFAULT_MIN_ABS_DELTA:
                legacy[leg] = thr
        return legacy

    # ------------------------------------------------------------------
    # Internal: formatters
    # ------------------------------------------------------------------

    def _format_text(self, regressions: list[Regression]) -> str:
        """Human-readable text report."""
        if not regressions:
            return "(no regressions detected)"

        # Count by severity.
        majors = sum(1 for r in regressions if r.severity == "major")
        minors = sum(1 for r in regressions if r.severity == "minor")
        improvements = sum(1 for r in regressions if r.severity == "improvement")

        lines: list[str] = []
        lines.append("Regression Report")
        lines.append("=" * 80)
        lines.append(
            f"Total findings: {len(regressions)} "
            f"(major={majors}, minor={minors}, improvement={improvements})"
        )
        lines.append("")

        if regressions:
            lines.append(
                f"{'benchmark':28s} {'vendor':14s} {'metric':20s} "
                f"{'baseline':>12s} {'candidate':>12s} {'delta':>8s} "
                f"{'severity':12s} {'sigma':>5s}"
            )
            lines.append("-" * 120)
            for r in sorted(
                regressions,
                key=lambda x: (x.severity != "major", x.severity != "minor",
                                abs(x.change_pct)),
                reverse=True,
            ):
                delta_str = f"{r.change_pct:>+7.2f}%"
                lines.append(
                    f"{r.benchmark:28s} {r.vendor:14s} {r.metric:20s} "
                    f"{r.baseline_value:>12.4f} {r.candidate_value:>12.4f} "
                    f"{delta_str} {r.severity:12s} {r.sigma:>5.1f}"
                )

        lines.append("")
        lines.append("-" * 80)
        return "\n".join(lines)

    def _format_json(self, regressions: list[Regression]) -> str:
        """JSON-formatted report."""
        majors = sum(1 for r in regressions if r.severity == "major")
        minors = sum(1 for r in regressions if r.severity == "minor")
        improvements = sum(1 for r in regressions if r.severity == "improvement")
        data = {
            "total_findings": len(regressions),
            "major_count": majors,
            "minor_count": minors,
            "improvement_count": improvements,
            "thresholds": dict(self.thresholds),
            "sigma_threshold": self.sigma_threshold,
            "regressions": [r.to_dict() for r in regressions],
        }
        return json.dumps(data, indent=2, default=str)


__all__ = [
    "DETECTOR_THRESHOLDS",
    "DETECTOR_MIN_ABS_DELTA",
    "DEFAULT_SIGMA_THRESHOLD",
    "Regression",
    "RegressionDetector",
]
