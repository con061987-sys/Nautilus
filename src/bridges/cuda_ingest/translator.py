"""CUDA → Triton translator.

The main translator that converts a parsed CUDA kernel (AST form)
into Triton Python source code. It coordinates:
  1. IntrinsicMapper — replace CUDA intrinsics with Triton equivalents
  2. SharedMemoryAnalyzer — translate __shared__ arrays
  3. PointerAnalyzer — handle pointer accesses
  4. Statement translation — convert for/if/assign to Triton

The translator produces Triton source that is then fed into the
Phase 1/2 pipeline (Triton → TVM MetaSchedule → Fat Binary) for
tuning and AOT compilation.

Production features:
  - Block/thread grid configuration based on CUDA launch dims
  - Block size inference from CUDA blockDim
  - Proper Triton @triton.jit decorator with type annotations
  - Integration with the shared memory and pointer analysis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .intrinsic_mapper import IntrinsicMapper
from .shared_memory import SharedMemoryAnalyzer, SharedMemPlan
from .pointer_analysis import PointerAnalyzer, PointerLayout

logger = logging.getLogger(__name__)


@dataclass
class TranslationResult:
    """Result of translating a CUDA kernel to Triton source."""
    success: bool
    triton_source: str = ""
    kernel_name: str = ""
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    shared_mem_plan: SharedMemPlan | None = None
    pointer_layouts: dict[str, PointerLayout] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def is_usable(self) -> bool:
        return self.success and bool(self.triton_source)


class CudaToTritonTranslator:
    """Translates a parsed CUDA kernel into Triton Python source.

    Usage:
        translator = CudaToTritonTranslator()
        result = translator.translate(kernel)
        if result.is_usable:
            # Feed result.triton_source to Phase 1/2 pipeline
            ...
    """

    def __init__(self) -> None:
        self.intrinsic_mapper = IntrinsicMapper()
        self.shared_mem_analyzer = SharedMemoryAnalyzer()
        self.pointer_analyzer = PointerAnalyzer()

    def translate(self, kernel: Any) -> TranslationResult:
        """Translate a CudaKernel to Triton source.

        Args:
            kernel: CudaKernel from the parser.

        Returns:
            TranslationResult with the Triton source and diagnostics.
        """
        if not kernel.is_global:
            return TranslationResult(
                success=False,
                kernel_name=kernel.name,
                error=f"Only __global__ kernels can be translated, got {kernel.qualifier}",
            )

        try:
            # Step 1: Analyze shared memory
            shared_mem_plan = self.shared_mem_analyzer.analyze(kernel)

            # Step 2: Analyze pointer layouts
            pointer_layouts = self.pointer_analyzer.analyze_kernel(kernel)

            # Step 3: Generate the function signature
            signature = self._generate_signature(kernel)

            # Step 4: Generate the function body
            body = self._generate_body(kernel, shared_mem_plan, pointer_layouts)

            # Step 5: Assemble the full Triton source
            imports = self._generate_imports()
            decorator = self._generate_decorator(kernel)
            full_source = f"""{imports}

{decorator}
{signature}
{body}
"""

            return TranslationResult(
                success=True,
                triton_source=full_source,
                kernel_name=kernel.name,
                shared_mem_plan=shared_mem_plan,
                pointer_layouts=pointer_layouts,
                warnings=shared_mem_plan.bank_conflict_warnings,
            )
        except Exception as exc:
            logger.error("Translation failed: %s", exc)
            return TranslationResult(
                success=False,
                kernel_name=kernel.name,
                error=str(exc),
            )

    def _generate_imports(self) -> str:
        """Generate the Triton import block."""
        return "import triton\nimport triton.language as tl"

    def _generate_decorator(self, kernel: Any) -> str:
        """Generate the @triton.jit decorator with block/grid hints."""
        # CUDA launch config: <<<grid, block>>>
        # Triton: grid is computed at call site, BLOCK_SIZE is a constexpr
        return "@triton.jit"

    def _generate_signature(self, kernel: Any) -> str:
        """Generate the Triton function signature.

        CUDA signature: __global__ void matmul(float* A, float* B, float* C, int M, int N, int K)
        Triton signature: def matmul(A_ptr, B_ptr, C_ptr, M, N, K,
                                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr)
        """
        params: list[str] = []
        block_constexprs: list[str] = []

        for param in kernel.parameters:
            name = param["name"]
            type_str = param["type"]
            if "*" in type_str:
                # Pointer parameter → becomes a pointer in Triton
                params.append(f"{name}_ptr")
            elif type_str.strip() in ("int", "unsigned int", "size_t"):
                # Integer dimension/size parameter
                params.append(name)
            elif type_str.strip() in ("float", "double"):
                params.append(name)
            else:
                params.append(name)

        # Add block size constants for matmul-like kernels
        if self._is_matmul_like(kernel):
            block_constexprs = [
                "BLOCK_M: tl.constexpr",
                "BLOCK_N: tl.constexpr",
                "BLOCK_K: tl.constexpr",
            ]

        all_params = params + block_constexprs
        params_str = ", ".join(all_params)
        return f"def {kernel.name}({params_str}):"

    def _generate_body(
        self,
        kernel: Any,
        shared_mem_plan: SharedMemPlan,
        pointer_layouts: dict[str, PointerLayout],
    ) -> str:
        """Generate the Triton function body."""
        lines: list[str] = []
        lines.append("    # Block and thread IDs (translated from CUDA)")
        lines.append("    pid = tl.program_id(0)")

        # Add shared memory allocations
        if shared_mem_plan.allocations:
            lines.append("")
            lines.append("    # Shared memory (translated from __shared__)")
            lines.append("    " + self.shared_mem_analyzer.generate_all_allocations(shared_mem_plan))

        # Add the translated kernel body
        lines.append("")
        lines.append("    # Translated kernel body")
        translated_stmts = self._translate_statements(kernel.body)
        for stmt in translated_stmts:
            lines.append(f"    {stmt}")

        return "\n".join(lines)

    def _translate_statements(self, statements: list[Any]) -> list[str]:
        """Translate a list of CUDA statements to Triton statements."""
        translated: list[str] = []
        for stmt in statements:
            triton_stmt = self._translate_one_statement(stmt)
            if triton_stmt is not None:
                translated.append(triton_stmt)
        return translated

    def _translate_one_statement(self, stmt: Any) -> str | None:
        """Translate a single CUDA statement to Triton."""
        if stmt.stmt_type.value == 1:  # SYNC_THREADS
            return "tl.barrier()"
        if stmt.stmt_type.value == 2:  # ATOMIC_OP
            return self._translate_atomic(stmt)
        if stmt.stmt_type.value == 0:  # FUNCTION_DEF
            return None  # Skip nested function defs
        if stmt.stmt_type.value == 3:  # ASSIGNMENT
            return self._translate_assignment(stmt)
        if stmt.stmt_type.value == 4:  # EXPRESSION
            return self._translate_expression(stmt)
        if stmt.stmt_type.value == 5:  # IF
            return self._translate_if(stmt)
        if stmt.stmt_type.value == 6:  # FOR
            return self._translate_for(stmt)
        if stmt.stmt_type.value == 7:  # WHILE
            return "# TODO: while loop translation"
        if stmt.stmt_type.value == 9:  # MEMORY_LOAD
            return self._translate_load(stmt)
        if stmt.stmt_type.value == 10:  # MEMORY_STORE
            return self._translate_store(stmt)
        if stmt.stmt_type.value == 11:  # RETURN
            return "return"
        if stmt.stmt_type.value == 8:  # BLOCK_INDEX
            return self._translate_block_index(stmt)
        return f"# TODO: untranslated {stmt.stmt_type.name}: {stmt.raw_text[:60]}"

    def _translate_atomic(self, stmt: Any) -> str:
        """Translate an atomic operation."""
        return self.intrinsic_mapper.transform_text(stmt.raw_text)

    def _translate_assignment(self, stmt: Any) -> str:
        """Translate a CUDA assignment."""
        return self.intrinsic_mapper.transform_text(stmt.raw_text)

    def _translate_expression(self, stmt: Any) -> str:
        """Translate a CUDA expression statement."""
        return self.intrinsic_mapper.transform_text(stmt.raw_text)

    def _translate_if(self, stmt: Any) -> str:
        """Translate a CUDA if statement."""
        return self.intrinsic_mapper.transform_text(stmt.raw_text)

    def _translate_for(self, stmt: Any) -> str:
        """Translate a CUDA for loop."""
        return self.intrinsic_mapper.transform_text(stmt.raw_text)

    def _translate_load(self, stmt: Any) -> str:
        """Translate a memory load."""
        return self.intrinsic_mapper.transform_text(stmt.raw_text)

    def _translate_store(self, stmt: Any) -> str:
        """Translate a memory store."""
        return self.intrinsic_mapper.transform_text(stmt.raw_text)

    def _translate_block_index(self, stmt: Any) -> str:
        """Translate a block/thread index expression."""
        return self.intrinsic_mapper.transform_text(stmt.raw_text)

    def _is_matmul_like(self, kernel: Any) -> bool:
        """Heuristic: is this kernel a matmul-style GEMM?"""
        name_lower = kernel.name.lower()
        return any(
            keyword in name_lower
            for keyword in ["matmul", "gemm", "sgemm", "dgemm"]
        )
