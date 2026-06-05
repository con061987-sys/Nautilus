"""Pipeline inspection hooks for the Triton ↔ TVM bridge.

Triton 3.7+ exposes `knobs.runtime.add_stages_inspection_hook` which
allows us to override any pipeline stage from Python. This is the
cleanest way to inject TVM MetaSchedule into Triton's compile flow
without forking the Triton source.

Two hook patterns are supported:
  1. stages_inspection_hook: receives the full stages dict and can
     replace any stage's lambda. This is the most powerful.
  2. Per-stage hooks: lightweight, can replace individual stages.

This module implements both. The recommended pattern is to use
stages_inspection_hook because it sees all stages at once.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

logger = logging.getLogger(__name__)


def install_stages_inspection_hook() -> bool:
    """Install the bridge's pipeline inspection hook on Triton's runtime.

    The hook intercepts the stages dict populated by add_stages() and
    wraps the ttgir stage with our capture logic.

    Returns:
        True if the hook was installed successfully, False otherwise
        (e.g. if Triton is not installed or the hook API has changed).
    """
    try:
        from triton import knobs
    except ImportError:
        logger.warning("triton.knobs not available; stages hook not installed")
        return False

    hook = _make_stages_inspection_hook()
    if hook is None:
        return False

    try:
        knobs.runtime.add_stages_inspection_hook = hook
        logger.info("Installed Triton pipeline inspection hook for TVM bridge")
        return True
    except (AttributeError, TypeError) as exc:
        # API may have changed between Triton versions
        logger.warning(
            "Failed to install stages_inspection_hook (incompatible Triton?): %s",
            exc,
        )
        return False


def uninstall_stages_inspection_hook() -> None:
    """Remove the bridge's hook, restoring Triton's default behaviour."""
    try:
        from triton import knobs
        if hasattr(knobs.runtime, "add_stages_inspection_hook"):
            del knobs.runtime.add_stages_inspection_hook
    except (ImportError, AttributeError):
        pass


def _make_stages_inspection_hook() -> Callable[..., Any] | None:
    """Build the stages inspection hook function.

    The hook contract (Triton 3.7+):
        def hook(stages=None, options=None, language=None, capability=None):
            # Called twice:
            # 1. With all-None: return (key, hash) for cache invalidation
            # 2. With real args: stages is the dict to modify
    """
    def stages_inspection_hook(
        stages: dict[str, Callable[..., Any]] | None = None,
        options: Any = None,
        language: Any = None,
        capability: Any = None,
    ) -> tuple[str, str]:
        # Case 1: key/hash query for cache invalidation
        if stages is None and options is None:
            return "tvm_bridge_hook", "v1"

        # Case 2: real invocation — wrap ttgir stage for capture
        if stages is None:
            return "tvm_bridge_hook", "v1"

        if not os.environ.get("NVINDIACUD_CAPTURE_DISABLED", "0") == "0":
            return "tvm_bridge_hook", "v1"

        original_ttgir = stages.get("ttgir")
        if original_ttgir is None:
            return "tvm_bridge_hook", "v1"

        stages["ttgir"] = _wrap_ttgir_for_capture(original_ttgir)
        return "tvm_bridge_hook", "v1"

    return stages_inspection_hook


def _wrap_ttgir_for_capture(
    original: Callable[..., Any],
) -> Callable[..., Any]:
    """Wrap a ttgir stage to capture the IR module output."""
    from .compiler import TVMBackend

    def wrapped(src: Any, metadata: dict[str, Any]) -> Any:
        result = original(src, metadata)
        try:
            backend = TVMBackend  # use static methods
            if isinstance(result, str):
                # The TTGIR text is in result. Forward to capture buffer.
                backend.clear_capture_buffer()
                key = f"hook_ttgir:{metadata.get('name', 'unknown')}:{metadata.get('src', '')[:16]}"
                buf = backend.get_capture_buffer()
                buf[key] = result
                logger.debug("Hook captured TTGIR: %d chars", len(result))
        except Exception as exc:
            logger.warning("Hook capture failed: %s", exc)
        return result

    return wrapped
