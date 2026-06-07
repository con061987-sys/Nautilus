"""Pytest conftest for pytorch_xla bridge tests.

Re-exports the ``sharding_bridge`` fixture from the project-level
test harness and registers the CLI options required by those fixtures.
"""
from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register CLI options used by the project-level bridge fixtures."""
    group = parser.getgroup("harness", "Bridge harness options")
    group.addoption(
        "--evidence-dir",
        action="store",
        default=None,
        help="Directory to write evidence files.",
    )
    group.addoption(
        "--use-real-backend",
        action="store_true",
        default=False,
        help="Use real (non-mocked) backends in bridge fixtures.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers used by the harness fixtures."""
    config.addinivalue_line(
        "markers",
        "use_real_backend: opt this test into real (non-mocked) bridge backends.",
    )
    config.addinivalue_line(
        "markers",
        "evidence: tag a test that intentionally writes evidence artifacts.",
    )


# Re-export the sharding_bridge fixture from the project-level harness.
from src.tests.conftest import sharding_bridge  # noqa: E402, F401
