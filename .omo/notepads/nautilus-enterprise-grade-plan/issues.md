# Issues Log

## Wave 0 — Foundation Repairs (Blocking everything)
- ~~C-01: setup.py doesn't call setup() — package doesn't install~~ FIXED
  - Added `_setup(**setup_kwargs)` at module level inside the `try/except ImportError` guard
  - Restructured: setup() at module level, `if __name__ == "__main__": build_cpp_plugin()` after it
  - Verified: `pip install -e .[dev]` succeeds; `nautilus --help` shows usage
- C-06: All 4 bridge test dirs ignored from pytest (260 of 296 tests hidden)
- H-13: Top-level `import torch`/`import triton` blocks 30 tests
- H-12: translator emits `tl.debug_barrier()` not `tl.barrier()`
- M-39: ir_classifier counts only 1 dot op, not 2
- M-40: timeout_manager stage_under_budget_succeeds test fails
- C-04: IRCapture key-format mismatch between write/read
- C-07: ci.yml typo `output_filename` vs `output_dir`
- H-08: hooks.py env var negation typo
- M-22: third_party/ submodules missing
