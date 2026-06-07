"""TIR template construction from kernel metadata and real captured IR.

This module is the bridge between Triton IR and TVM TIR. It supports
two paths:

  1. Primary: real IR conversion via ConversionPipeline
     Takes captured TTGIR text, runs the 4-pass conversion, produces
     TVMScript text. This is the production path — it preserves the
     actual semantics of the kernel (not generic templates).

  2. Fallback: generic template construction from metadata bounds
     When the real IR is not available (e.g. Python-level fallback),
     builds a generic TIR template from the kernel's mathematical
     bounds. This was the original design and is preserved as a
     safety net.

The philosophy is "fix or redesign, not remove": the template
methods stay, but the primary path is now real IR conversion.
"""

from __future__ import annotations

from typing import Any

from src.common.logging import get_logger

try:
    import tvm
    import tvm.tir as tir
    from tvm.script import tir as T

    TVM_AVAILABLE = True
except ImportError:
    TVM_AVAILABLE = False

    class T:  # type: ignore
        class prim_func:
            pass

        class Buffer:
            pass


logger = get_logger(__name__)


class TIRTemplateBuilder:
    """Constructs TVM TIR PrimFuncs from metadata OR real captured IR.

    Primary path: real IR conversion via ConversionPipeline
    Fallback path: generic templates from bounds (preserved)
    """

    def __init__(self) -> None:
        if not TVM_AVAILABLE:
            raise RuntimeError(
                "TVM is required to build TIR templates. Install with: pip install apache-tvm"
            )
        # Lazy import to avoid circular dependency
        from .ir_to_tir import ConversionPipeline

        self._pipeline = ConversionPipeline()

    def build_from_captured_ir(
        self,
        ir_text: str,
    ) -> tuple[Any, Any]:
        """Build TIR from REAL captured IR (the production path).

        Returns:
            (tvm.IRModule, conversion_result) — the IRModule for TVM
            and the ConversionResult for diagnostics.
        """
        result = self._pipeline.convert(ir_text)
        if not result.is_usable:
            logger.warning(
                "Real IR conversion failed (%s); pipeline returned FALLBACK",
                result.status.name,
            )
            return None, result

        # Execute the TVMScript to get an IRModule
        try:
            ir_module = self._execute_tvmscript(result.tvmscript_text)
        except Exception as exc:
            logger.warning("TVMScript execution failed: %s", exc)
            # Return None for IRModule but keep the result for diagnostics
            return None, result

        return ir_module, result

    def build_matmul(
        self,
        m: int,
        n: int,
        k: int,
        dtype: str = "float32",
    ) -> tvm.IRModule:
        @T.prim_func
        def matmul_kernel(
            A: T.Buffer[(m, k), dtype],
            B: T.Buffer[(k, n), dtype],
            C: T.Buffer[(m, n), dtype],
        ) -> None:
            for i, j, kk in T.grid(m, n, k):
                with T.block("C"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, kk])
                    with T.init():
                        C[vi, vj] = T.cast(T.float32(0), dtype)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

        return tvm.IRModule({"main": matmul_kernel})

    def build_reduction(
        self,
        shape: tuple[int, ...],
        axis: int = 0,
        dtype: str = "float32",
        op: str = "sum",
    ) -> tvm.IRModule:
        assert len(shape) == 2, "Only 2D reduction templates supported"
        d0, d1 = shape

        @T.prim_func
        def reduce_kernel(
            inp: T.Buffer[(d0, d1), dtype],
            out: T.Buffer[(d0,), dtype],
        ) -> None:
            for i in T.grid(d0):
                with T.block("reduce"):
                    vi = T.axis.spatial(d0, i)
                    with T.init():
                        if op == "max":
                            out[vi] = T.cast(T.float32(-1e38), dtype)
                        else:
                            out[vi] = T.cast(T.float32(0), dtype)
                    for k in T.grid(d1):
                        with T.block("reduce_inner"):
                            vk = T.axis.reduce(d1, k)
                            if op == "max":
                                out[vi] = T.max(out[vi], inp[vi, vk])
                            else:
                                out[vi] = out[vi] + inp[vi, vk]

        return tvm.IRModule({"main": reduce_kernel})

    def build_elementwise(
        self,
        shape: tuple[int, ...],
        dtype: str = "float32",
        op: str = "add",
    ) -> tvm.IRModule:
        @T.prim_func
        def elem_kernel(
            A: T.Buffer[shape, dtype],
            B: T.Buffer[shape, dtype],
            C: T.Buffer[shape, dtype],
        ) -> None:
            for idx in T.grid(*shape):
                with T.block("elem"):
                    vi = [T.axis.spatial(s, i) for s, i in zip(shape, idx)]
                    if op == "add":
                        C[vi] = A[vi] + B[vi]
                    elif op == "mul":
                        C[vi] = A[vi] * B[vi]
                    elif op == "max":
                        C[vi] = T.max(A[vi], B[vi])
                    else:
                        C[vi] = A[vi] + B[vi]

        return tvm.IRModule({"main": elem_kernel})

    def build_from_metadata(self, metadata: Any) -> tvm.IRModule:
        """Build a TIR template from extracted kernel metadata.

        This is the FALLBACK path. The primary path is
        build_from_captured_ir which uses real IR conversion.
        """
        dtype = "float32"
        if metadata.arg_dtypes:
            dtype = metadata.arg_dtypes[0]

        if metadata.is_matmul and metadata.matmul_m is not None:
            return self.build_matmul(
                m=metadata.matmul_m,
                n=metadata.matmul_n,
                k=metadata.matmul_k,
                dtype=dtype,
            )

        if metadata.is_reduction and metadata.arg_shapes:
            shape = metadata.arg_shapes[0]
            if len(shape) >= 2:
                return self.build_reduction(
                    shape=shape[:2],
                    axis=0,
                    dtype=dtype,
                    op="sum",
                )

        if metadata.is_elementwise and metadata.arg_shapes:
            shape = metadata.arg_shapes[0]
            return self.build_elementwise(
                shape=shape,
                dtype=dtype,
                op="add",
            )

        m = metadata.grid_0 * 128
        n = metadata.grid_1 * 128
        k = 128
        return self.build_matmul(m=m, n=n, k=k, dtype=dtype)

    def _execute_tvmscript(self, tvmscript_text: str) -> Any:
        """Execute TVMScript text to produce a TVM IRModule.

        This wraps the TVMScript string in a namespace where
        tvm.script.tir is available, then evaluates it. The
        resulting PrimFunc is wrapped in an IRModule.
        """
        import tvm
        from tvm.script import tir as T

        # Build a namespace for the TVMScript to evaluate in
        namespace: dict[str, Any] = {
            "T": T,
            "tvm": tvm,
            "tir": tir if TVM_AVAILABLE else None,
        }

        # Execute the TVMScript in this namespace
        exec(tvmscript_text, namespace)

        # The TVMScript should have defined a function with the
        # kernel's name; find it
        func_names = [
            name
            for name, obj in namespace.items()
            if callable(obj) and not name.startswith("_") and name != "T"
        ]
        if not func_names:
            raise RuntimeError("TVMScript did not define any function")

        # Take the first function (there should be only one PrimFunc)
        func = namespace[func_names[0]]

        # Wrap in IRModule
        return tvm.IRModule({func_names[0]: func})
