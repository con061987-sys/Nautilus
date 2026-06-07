"""Model-level benchmarks.

Model benchmarks differ from kernel benchmarks in two ways:

  1. They exercise the full Nautilus pipeline: graph capture, fat
     binary build, runtime dispatch, and forward pass.
  2. They include both an "eager" baseline and a "compiled" path
     so we can measure the speedup attributable to our toolchain.

Each model benchmark implements :class:`benchmarks.runner.BenchmarkProtocol`
and exposes itself as ``BENCHMARK`` at module level so the runner can
discover it.

Graceful degradation
--------------------
Models are gated by optional dependencies (torch, torchvision,
transformers). The benchmark returns ``{"status": "skipped", "error": ...}``
when a dependency is missing rather than crashing the whole suite.
"""

from __future__ import annotations

import importlib

from benchmarks.runner import RawRun, RunContext
from src.common.logging import get_logger

log = get_logger("nautilus.bench.models")


# ---------------------------------------------------------------------------
# Dependency probe — shared across model benchmarks
# ---------------------------------------------------------------------------


def _have_module(name: str) -> bool:
    """Return True if a top-level module is importable.

    Used to gate model benchmarks on optional heavy deps (torch,
    torchvision, transformers) without paying the import cost when
    they're not installed.
    """
    try:
        importlib.import_module(name)
    except Exception:  # noqa: BLE001 — any failure means "not available"
        return False
    return True


def _skip(dep: str, ctx: RunContext) -> RawRun:  # noqa: ARG001
    """Standard "dependency missing" payload."""
    return {
        "status": "skipped",
        "error": f"{dep} not installed; install with `pip install -e .[{dep}]`",
    }


__all__ = ["_have_module", "_skip"]
