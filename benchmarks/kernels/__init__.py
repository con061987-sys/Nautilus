"""Kernel-level benchmarks.

Each module here exports a ``BENCHMARK`` instance implementing
:mod:`benchmarks.runner.BenchmarkProtocol`. The runner's discovery
finds them automatically.

Legacy modules (without the ``_bench`` suffix) remain importable as
``benchmarks.kernels.<name>`` for backwards compatibility; the new
``<name>_bench.py`` modules are the canonical sources for the
unified benchmark CLI.
"""

from __future__ import annotations

__all__: list[str] = []
