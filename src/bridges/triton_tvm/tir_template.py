"""TIR template construction from kernel metadata.

Constructs equivalent TVM TIR PrimFuncs from extracted Triton kernel
metadata. These templates encode the same mathematical computation but
in TVM's Tensor IR format, allowing MetaSchedule to search over
schedule primitives (tiling, binding, vectorization, etc.) that map
directly to Triton's tuning parameters.

This module implements the 'construct_tir_template' step in the
config bridge architecture.
"""

from __future__ import annotations

from typing import Any

# Attempt to import TVM; gracefully degrade if unavailable
try:
    from tvm.script import tirx as T
    import tvm
    import tvm.tirx as tirx

    TVM_AVAILABLE = True
except ImportError:
    TVM_AVAILABLE = False
    # Create placeholder types for when TVM isn't installed
    class T:  # type: ignore
        class prim_func: pass
        class Buffer: pass


class TIRTemplateBuilder:
    """Constructs TVM TIR PrimFunc templates from kernel metadata.

    Each template represents a Triton kernel's computation in TIR
    form, parameterized by the mathematical bounds (M, N, K, etc.)
    extracted from the Triton JIT call. MetaSchedule then tunes these
    templates to find optimal schedule parameters.
    """

    def __init__(self) -> None:
        if not TVM_AVAILABLE:
            raise RuntimeError(
                "TVM is required to build TIR templates. "
                "Install with: pip install apache-tvm"
            )

    def build_matmul(
        self,
        m: int,
        n: int,
        k: int,
        dtype: str = "float32",
    ) -> tvm.IRModule:
        """Build a TIR PrimFunc template for matrix multiplication.

        Args:
            m: M dimension (rows of A, rows of C).
            n: N dimension (cols of B, cols of C).
            k: K dimension (reduction, cols of A, rows of B).
            dtype: Data type string ('float32', 'float16', 'bfloat16').

        Returns:
            tvm.IRModule containing the PrimFunc, ready for MetaSchedule.
        """
        @T.prim_func
        def matmul_kernel(
            A: T.Buffer((m, k), dtype),
            B: T.Buffer((k, n), dtype),
            C: T.Buffer((m, n), dtype),
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
        """Build a TIR PrimFunc template for reduction operations.

        Args:
            shape: Input tensor shape.
            axis: Reduction axis.
            dtype: Data type.
            op: Reduction operation ('sum', 'max', 'min').

        Returns:
            tvm.IRModule containing the PrimFunc.
        """
        # Only 2D reductions are currently supported for simplicity
        assert len(shape) == 2, "Only 2D reduction templates supported"
        d0, d1 = shape

        @T.prim_func
        def reduce_kernel(
            inp: T.Buffer((d0, d1), dtype),
            out: T.Buffer((d0,), dtype),
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
        """Build a TIR PrimFunc template for element-wise operations.

        Args:
            shape: Input/output tensor shape.
            dtype: Data type.
            op: Operation ('add', 'mul', 'max').

        Returns:
            tvm.IRModule containing the PrimFunc.
        """
        @T.prim_func
        def elem_kernel(
            A: T.Buffer(shape, dtype),
            B: T.Buffer(shape, dtype),
            C: T.Buffer(shape, dtype),
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

        Automatically selects the correct template based on kernel
        type classification in the metadata.

        Args:
            metadata: KernelMetadata from metadata_extractor.

        Returns:
            tvm.IRModule ready for MetaSchedule tuning.
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

        # Fallback: build matmul based on grid dimensions
        m = metadata.grid_0 * 128
        n = metadata.grid_1 * 128
        k = 128
        return self.build_matmul(m=m, n=n, k=k, dtype=dtype)
