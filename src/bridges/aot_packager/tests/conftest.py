"""Conftest for aot_packager integration tests.

Imports the shared bridge fixtures from the top-level test harness
(``src/tests/conftest.py``) so tests in this directory can use
``aot_packager``, ``auto_tuning_bridge``, etc. without duplicating
the fixture definitions.
"""
from __future__ import annotations

pytest_plugins = ("src.tests.conftest",)
