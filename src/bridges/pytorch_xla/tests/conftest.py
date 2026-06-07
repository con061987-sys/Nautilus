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
