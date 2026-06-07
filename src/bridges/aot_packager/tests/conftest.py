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

from src.tests.conftest import aot_packager  # noqa: F401
from src.tests.conftest import auto_tuning_bridge  # noqa: F401
from src.tests.conftest import clean_cache  # noqa: F401
from src.tests.conftest import evidence_capture  # noqa: F401
from src.tests.conftest import evidence_dir  # noqa: F401
from src.tests.conftest import repo_root  # noqa: F401
from src.tests.conftest import sharding_bridge  # noqa: F401
from src.tests.conftest import cuda_ingestor  # noqa: F401
from src.tests.conftest import any_gpu_available  # noqa: F401
