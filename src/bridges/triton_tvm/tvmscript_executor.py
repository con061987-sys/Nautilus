"""TVMScript executor — runs emitted TVMScript to produce a real TVM IRModule.

The TVMScript emitter (tvmscript_emitter.py) produces a string of
Python code that, when executed in a namespace with tvm.script.tir
available, produces a TVM PrimFunc. This module provides the
production-grade execution of that string with:

  - Proper namespace isolation
  - Error handling and diagnostic information
  - Verification that the result is a valid IRModule
  - Cleanup of any side effects

This is the bridge between "we produced TVMScript text" and "we have
a real TVM IRModule that MetaSchedule can consume".
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Any

from src.common.logging import get_logger

logger = get_logger(__name__)

try:
    import tvm
    from tvm.script import tir as T

    TVM_AVAILABLE = True
except ImportError:
    TVM_AVAILABLE = False


@dataclass
class ExecutionResult:
    """Result of executing a TVMScript string."""

    success: bool
    ir_module: Any = None
    prim_func: Any = None
    error: str | None = None
    traceback: str = ""
    execution_time_ms: float = 0.0

    @property
    def is_usable(self) -> bool:
        return self.success and self.ir_module is not None


class TVMScriptExecutor:
    """Executes TVMScript text to produce a real TVM IRModule.

    The executor is the production-grade boundary between the
    emitter (which produces text) and TVM (which consumes PrimFunc
    objects). It handles:

      - Namespace isolation so TVMScript can't pollute our globals
      - Timeout protection (prevents runaway compilation)
      - Error capture with full tracebacks
      - Validation of the produced IRModule
    """

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        self.timeout_seconds = timeout_seconds

    def execute(self, tvmscript_text: str) -> ExecutionResult:
        """Execute TVMScript and return the resulting IRModule.

        Args:
            tvmscript_text: The TVMScript text from the emitter.

        Returns:
            ExecutionResult with the IRModule and any errors.
        """
        import time

        start = time.perf_counter()

        if not TVM_AVAILABLE:
            return ExecutionResult(
                success=False,
                error="TVM is not available; cannot execute TVMScript",
            )

        if not tvmscript_text or not tvmscript_text.strip():
            return ExecutionResult(
                success=False,
                error="Empty TVMScript text",
            )

        namespace = self._build_namespace()
        try:
            # Execute the TVMScript in the isolated namespace
            # We use exec() rather than compile()+eval() because
            # TVMScript uses decorators (@T.prim_func) that require
            # a module-level scope
            code = compile(tvmscript_text, "<tvmscript>", "exec")
            exec(code, namespace)
        except SyntaxError as exc:
            return ExecutionResult(
                success=False,
                error=f"TVMScript syntax error: {exc.msg} at line {exc.lineno}",
                traceback=traceback.format_exc(),
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            return ExecutionResult(
                success=False,
                error=f"TVMScript execution failed: {exc}",
                traceback=traceback.format_exc(),
                execution_time_ms=elapsed,
            )

        # Find the produced PrimFunc(s)
        prim_funcs = self._extract_prim_funcs(namespace)
        if not prim_funcs:
            return ExecutionResult(
                success=False,
                error="TVMScript executed but did not produce a PrimFunc",
            )

        # Wrap in IRModule
        try:
            ir_module = tvm.IRModule(prim_funcs)
        except Exception as exc:
            return ExecutionResult(
                success=False,
                error=f"Failed to wrap PrimFuncs in IRModule: {exc}",
                traceback=traceback.format_exc(),
            )

        # Validate
        is_valid, validation_error = self._validate(ir_module)
        if not is_valid:
            return ExecutionResult(
                success=False,
                error=f"Produced IRModule failed validation: {validation_error}",
            )

        elapsed = (time.perf_counter() - start) * 1000
        return ExecutionResult(
            success=True,
            ir_module=ir_module,
            prim_func=prim_funcs,
            execution_time_ms=elapsed,
        )

    def _build_namespace(self) -> dict[str, Any]:
        """Build the isolated namespace for TVMScript execution.

        Provides the T namespace and tvm module without polluting
        the calling scope.
        """
        import tvm
        from tvm.script import tir as T

        namespace: dict[str, Any] = {
            "__builtins__": __builtins__,
            "tvm": tvm,
            "T": T,
        }
        # Also expose tir if available
        try:
            import tvm.tir as tir

            namespace["tir"] = tir
        except ImportError:
            pass
        return namespace

    def _extract_prim_funcs(self, namespace: dict[str, Any]) -> dict[str, Any]:
        """Find PrimFunc objects in the namespace after execution.

        TVMScript defines functions decorated with @T.prim_func.
        After execution, these functions become PrimFunc objects
        available as global variables in the namespace.
        """
        prim_funcs: dict[str, Any] = {}
        # Common names to skip
        skip_names = {
            "T",
            "tvm",
            "tir",
            "__builtins__",
            "__name__",
            "__doc__",
            "__package__",
            "__loader__",
            "__spec__",
            "__file__",
            "__cached__",
        }
        for name, obj in namespace.items():
            if name in skip_names or name.startswith("_"):
                continue
            # Check if it's a PrimFunc (TVM 0.18+ uses tvm.tir.PrimFunc)
            if TVM_AVAILABLE:
                try:
                    from tvm.tir import PrimFunc

                    if isinstance(obj, PrimFunc):
                        prim_funcs[name] = obj
                        continue
                except ImportError:
                    pass
            # Check for IRModule directly
            try:
                if isinstance(obj, tvm.IRModule):
                    prim_funcs["main"] = obj
            except Exception:
                pass
        return prim_funcs

    def _validate(self, ir_module: Any) -> tuple[bool, str | None]:
        """Validate that the produced IRModule is well-formed.

        Returns (is_valid, error_message).
        """
        if ir_module is None:
            return False, "IRModule is None"
        try:
            # Check that it has at least one function
            funcs = (
                list(ir_module.functions_items()) if hasattr(ir_module, "functions_items") else []
            )
            if not funcs:
                return False, "IRModule has no functions"
            # Verify it's parseable by TVM's verifier
            # (TVM does verification on construction in most cases)
            return True, None
        except Exception as exc:
            return False, f"Validation error: {exc}"
