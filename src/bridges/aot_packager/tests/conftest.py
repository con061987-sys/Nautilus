"""Conftest for aot_packager integration tests.

Re-exports the bridge fixtures from the project-level test harness
(``src/tests/conftest.py``) so tests in this directory can use
``aot_packager`` and friends without duplicating definitions.

Pytest auto-discovers ``conftest.py`` files up the tree from each
test file, so fixtures in ``src/tests/conftest.py`` are already
visible here. We do NOT use ``pytest_plugins`` in this conftest —
that mechanism is deprecated for non-root conftest files in pytest
8+ and produces warnings. The explicit imports below make the
dependency obvious to maintainers and the type checker.
"""

from __future__ import annotations

from src.tests.conftest import (
    any_gpu_available,  # noqa: F401
    aot_packager,  # noqa: F401
    auto_tuning_bridge,  # noqa: F401
    clean_cache,  # noqa: F401
    cuda_ingestor,  # noqa: F401
    evidence_capture,  # noqa: F401
    evidence_dir,  # noqa: F401
    repo_root,  # noqa: F401
    sharding_bridge,  # noqa: F401
)
