"""Pytest conftest for pytorch_xla bridge tests.

Re-exports the ``sharding_bridge`` fixture from the project-level
test harness (``src/tests/conftest.py``).

The ``--evidence-dir`` and ``--use-real-backend`` CLI options plus
the ``use_real_backend`` / ``evidence`` markers are registered by
the project-level conftest. Re-registering them here would cause a
pytest conflict (group.addoption on an already-registered name
fails in strict mode) so we only import the fixture.
"""

from __future__ import annotations

from src.tests.conftest import aot_packager  # noqa: F401
from src.tests.conftest import auto_tuning_bridge  # noqa: F401
from src.tests.conftest import clean_cache  # noqa: F401
from src.tests.conftest import cuda_ingestor  # noqa: F401
from src.tests.conftest import evidence_capture  # noqa: F401
from src.tests.conftest import evidence_dir  # noqa: F401
from src.tests.conftest import repo_root  # noqa: F401
from src.tests.conftest import sharding_bridge  # noqa: E402, F401
from src.tests.conftest import any_gpu_available  # noqa: F401
