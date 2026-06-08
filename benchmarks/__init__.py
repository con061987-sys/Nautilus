"""Nautilus benchmark suite.

This package collects the kernel- and model-level benchmarks used to
measure auto-tuning speedups, compile times, and fat-binary sizes.

Public entry points
-------------------
- :mod:`benchmarks.runner`     — benchmark execution engine
- :mod:`benchmarks.results`    — result storage / compare / history
- :mod:`benchmarks.ingestion`  — result ingestion into performance database
- :mod:`benchmarks.kernels`    — single-kernel benchmarks
- :mod:`benchmarks.models`     — model-level benchmarks

The CLI surface lives in :mod:`src.cli.commands.bench` and is wired
into the top-level ``nautilus`` group as ``nautilus bench``.

Versioning
----------
The on-disk result schema is versioned via :class:`benchmarks.results.SCHEMA_VERSION`.
Any breaking change MUST bump the version and the loader MUST reject
older payloads loudly instead of silently dropping fields.
"""

from __future__ import annotations

# Re-export the schema version + most common types so callers can
# ``from benchmarks import SCHEMA_VERSION``.
from benchmarks.results import (  # noqa: E402
    SCHEMA_VERSION,
    BenchmarkResult,
    ComparisonReport,
    RegressionFinding,
    ResultSet,
)
from benchmarks.regression import (  # noqa: E402
    Regression,
    RegressionDetector,
    DETECTOR_THRESHOLDS,
    DETECTOR_MIN_ABS_DELTA,
    DEFAULT_SIGMA_THRESHOLD,
)

__all__ = [
    "BenchmarkResult",
    "ComparisonReport",
    "DEFAULT_SIGMA_THRESHOLD",
    "DETECTOR_MIN_ABS_DELTA",
    "DETECTOR_THRESHOLDS",
    "Regression",
    "RegressionDetector",
    "RegressionFinding",
    "ResultSet",
    "SCHEMA_VERSION",
]
