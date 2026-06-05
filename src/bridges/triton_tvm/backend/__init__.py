"""Triton backend plugin for the Triton ↔ TVM MetaSchedule bridge.

This package provides an out-of-tree Triton backend that hooks into
Triton's compilation pipeline via the TRITON_PLUGIN_DIRS mechanism
and dispatches kernel autotuning through TVM MetaSchedule.

Discovery: Triton discovers this backend via the `triton.backends`
entry point group in pyproject.toml, or via TRITON_PLUGIN_DIRS env var.

Architecture:
    compiler.py   — implements BaseBackend (add_stages, parse_options)
    driver.py     — implements Driver (memory, dispatch)
    options.py    — TVMOptions dataclass
    hooks.py      — pipeline inspection hooks (knobs.runtime.add_stages_inspection_hook)
"""

from .compiler import TVMBackend
from .driver import TVMDriver
from .options import TVMOptions

__all__ = ["TVMBackend", "TVMDriver", "TVMOptions"]
