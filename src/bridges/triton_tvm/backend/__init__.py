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

# Shared capture-buffer key format used by:
#   - backend/compiler.py (writer, in _capture_stage)
#   - backend/hooks.py    (writer, in _wrap_ttgir_for_capture)
#   - ir_capture.py       (reader, in IRCapture.capture_for_source)
# All three MUST use the same format or the bridge can never find what
# was written. The format embeds the project name (for disambiguation
# in shared dicts), the stage, the source hash, and the kernel name.
CAPTURE_KEY_FMT = "nautilus:ttgir:{source_hash}:{kernel_name}"

from .compiler import TVMBackend
from .driver import TVMDriver
from .options import TVMOptions

__all__ = ["TVMBackend", "TVMDriver", "TVMOptions", "CAPTURE_KEY_FMT"]
