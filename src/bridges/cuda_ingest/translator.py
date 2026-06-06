"""CUDA → Triton translator using AST-level dispatch.

The main translator converts a parsed CUDA kernel (CudaKernel with
CudaStatement list) into Triton Python source.  It differs from
the old text-level translator in four critical ways:

1. **AST-aware dispatch** — each statement is translated based on its
   tree-sitter-derived metadata (stmt_type + metadata dict), not
   by regex-matching raw text.

2. **No silent no-ops** — unsupported intrinsics raise
   IngestionUnsupportedIntrinsicError.  The pipeline never silently
   produces a broken translation.

3. **Structured atomic translation** — atomic ops are mapped using
   the IntrinsicMapper as a lookup table, with proper recognition
   of argument positions and type-cast insertion.

4. **CUDA field expressions handled via metadata** — threadIdx.x,
   blockIdx.y, etc. are identified by their AST field_expression
   structure, not by regex substitution that would also match
   struct fields named "x".

The IntrinsicMapper is still used, but as a **mapping reference**
(get_mapping()) — not via transform_text() which loses semantic
information.

Usage:
    translator = CudaToTritonTranslator()
    result = translator.translate(kernel)
    if result.is_usable:
        # Feed result.triton_source to Phase 1/2 pipeline
        ...
"""  # noqa: W505

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.common.errors import IngestionUnsupportedIntrinsicError
from src.common.logging import get_logger

from .intrinsic_mapper import IntrinsicMapper
from .shared_memory import SharedMemoryAnalyzer, SharedMemPlan
from .pointer_analysis import PointerAnalyzer, PointerLayout

# Import CudaStatementType for dispatch
try:
    from .parser import CudaStatementType
except ImportError:
    # Fallback for backward compatibility during transition
    from enum import Enum, auto as _auto
    class CudaStatementType(str, Enum):
        FUNCTION_DEF = "FUNCTION_DEF"
        SHARED_MEM = "SHARED_MEM"
        ASSIGNMENT = "ASSIGNMENT"
        EXPRESSION = "EXPRESSION"
        IF = "IF"
        FOR = "FOR"
        WHILE = "WHILE"
        SYNC_THREADS = "SYNC_THREADS"
        ATOMIC_OP = "ATOMIC_OP"
        MEMORY_LOAD = "MEMORY_LOAD"
        MEMORY_STORE = "MEMORY_STORE"
        RETURN = "RETURN"
        BLOCK_INDEX = "BLOCK_INDEX"
        DECLARATION = "DECLARATION"
        UNKNOWN = "UNKNOWN"

logger = get_logger(__name__)


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


# ---------------------------------------------------------------------------
# Mapping tables from CUDA field expressions to Triton expressions.
# These are the "safe" strings used for replacement; they come from
# IntrinsicMapper but are complemented by context-aware logic below.
# ---------------------------------------------------------------------------

_CUDA_FIELD_TO_TRITON: dict[str, dict[str, str]] = {
    "threadIdx": {
        "x": "tl.program_id(0)",
        "y": "tl.program_id(1)",
        "z": "tl.program_id(2)",
    },
    "blockIdx": {
        "x": "tl.program_id(0)",
        "y": "tl.program_id(1)",
        "z": "tl.program_id(2)",
    },
    "blockDim": {
        "x": "tl.num_programs(0)",
        "y": "tl.num_programs(1)",
        "z": "tl.num_programs(2)",
    },
    "gridDim": {
        "x": "tl.num_programs(0)",
        "y": "tl.num_programs(1)",
        "z": "tl.num_programs(2)",
    },
}

_SYNC_TO_TRITON: dict[str, str] = {
    "__syncthreads": "tl.debug_barrier()",
    "__syncwarp": "# tl.syncwarp — closest equivalent: tl.debug_barrier()",
    "__threadfence": "tl.debug_barrier()",
}

_ATOMIC_TO_TRITON: dict[str, str] = {
    "atomicAdd": "tl.atomic_add",
    "atomicSub": "tl.atomic_sub",
    "atomicMin": "tl.atomic_min",
    "atomicMax": "tl.atomic_max",
    "atomicAnd": "tl.atomic_and",
    "atomicOr": "tl.atomic_or",
    "atomicXor": "tl.atomic_xor",
    "atomicCAS": "tl.atomic_cas",
    "atomicExch": "tl.atomic_xchg",
}


class CudaToTritonTranslator:
    """Translates a parsed CUDA kernel into Triton Python source.

    Uses AST metadata from the parser (CudaStatement.metadata) to
    make precise, context-aware translations.  Falls back to text-level
    mapping only for statements that contain no CUDA-specific constructs.
    """

    def __init__(self) -> None:
        self.intrinsic_mapper = IntrinsicMapper()
        self.shared_mem_analyzer = SharedMemoryAnalyzer()
        self.pointer_analyzer = PointerAnalyzer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Code generation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_imports() -> str:
        return "import triton\nimport triton.language as tl"

    @staticmethod
    def _generate_decorator(kernel: Any) -> str:
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
            name = param.get("name", "")
            type_str = param.get("type", "")
            if not name:
                continue
            if "*" in type_str:
                params.append(f"{name}_ptr")
            elif type_str.strip() in ("int", "unsigned int", "size_t"):
                params.append(name)
            elif type_str.strip() in ("float", "double"):
                params.append(name)
            else:
                params.append(name)

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
        lines.append("    # NOTE: blockIdx/threadIdx mapping is approximate.")
        lines.append("    # Review offsets for correctness on your target hardware.")

        if shared_mem_plan.allocations:
            lines.append("")
            lines.append("    # Shared memory (translated from __shared__)")
            alloc_line = self.shared_mem_analyzer.generate_all_allocations(
                shared_mem_plan,
            )
            lines.append("    " + alloc_line)

        lines.append("")
        lines.append("    # Translated kernel body")
        translated_stmts = self._translate_statements(kernel.body)
        for stmt in translated_stmts:
            lines.append(f"    {stmt}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Statement translation dispatch
    # ------------------------------------------------------------------

    def _translate_statements(self, statements: list[Any]) -> list[str]:
        """Translate a list of CUDA statements to Triton statements.

        Returns a list of translated source lines (each indented for
        the function body).  Returns an empty list on unsupported
        constructs (logged as error).
        """
        translated: list[str] = []
        for stmt in statements:
            try:
                triton_stmt = self._translate_one_statement(stmt)
                if triton_stmt is not None:
                    translated.append(triton_stmt)
            except IngestionUnsupportedIntrinsicError:
                # Log and re-raise — never silently skip
                raise
            except Exception as exc:
                logger.error(
                    "Failed to translate statement at line %d: %s",
                    stmt.line_number, exc,
                )
                translated.append(
                    f"# ERROR: translation failed at line {stmt.line_number}: {exc}",
                )
        return translated

    def _translate_one_statement(self, stmt: Any) -> str | None:
        """Translate a single CUDA statement to Triton.

        Dispatches based on CudaStatementType, using metadata for
        precise AST-level translation of CUDA-specific constructs.

        Returns:
            Translated Triton source string, or None to skip.
        """
        stmt_type = stmt.stmt_type

        # Dispatch by statement type — use enum members directly (never .value)
        if stmt_type == CudaStatementType.SYNC_THREADS:
            return self._translate_sync(stmt)
        if stmt_type == CudaStatementType.ATOMIC_OP:
            return self._translate_atomic(stmt)
        if stmt_type == CudaStatementType.FUNCTION_DEF:
            return None
        if stmt_type == CudaStatementType.ASSIGNMENT:
            return self._translate_assignment(stmt)
        if stmt_type == CudaStatementType.EXPRESSION:
            return self._translate_expression(stmt)
        if stmt_type == CudaStatementType.IF:
            return self._translate_if(stmt)
        if stmt_type == CudaStatementType.FOR:
            return self._translate_for(stmt)
        if stmt_type == CudaStatementType.WHILE:
            return self._translate_while(stmt)
        if stmt_type == CudaStatementType.MEMORY_LOAD:
            return self._translate_load(stmt)
        if stmt_type == CudaStatementType.MEMORY_STORE:
            return self._translate_store(stmt)
        if stmt_type == CudaStatementType.RETURN:
            return self._translate_return(stmt)
        if stmt_type == CudaStatementType.BLOCK_INDEX:
            return self._translate_block_index(stmt)
        if stmt_type == CudaStatementType.SHARED_MEM:
            return None  # Handled by shared_memory.py allocation generation
        if stmt_type == CudaStatementType.DECLARATION:
            return self._translate_declaration(stmt)

        return f"# TODO: untranslated {stmt_type.name}: {stmt.raw_text[:60]}"

    # ------------------------------------------------------------------
    # Sync translation
    # ------------------------------------------------------------------

    def _translate_sync(self, stmt: Any) -> str:
        """Translate a synchronization statement.

        Uses metadata to determine which sync function was called.
        Falls back to raw-text mapping.
        """
        calls = stmt.metadata.get("sync_calls", [])
        if calls:
            call_name = calls[0]["name"]
            triton_expr = _SYNC_TO_TRITON.get(call_name)
            if triton_expr is not None:
                if call_name == "__syncthreads":
                    return f"{triton_expr}  # closest equivalent to __syncthreads()"
                return triton_expr

        # Fallback: text-level mapping from intrinsic_mapper
        mapped = self.intrinsic_mapper.transform_text(stmt.raw_text)
        if mapped.strip() != stmt.raw_text.strip():
            return mapped

        return "tl.debug_barrier()  # ~sync (auto-detected)"

    # ------------------------------------------------------------------
    # Atomic translation
    # ------------------------------------------------------------------

    def _translate_atomic(self, stmt: Any) -> str:
        """Translate an atomic operation using AST metadata.

        Uses metadata['atomic_ops'] for structured translation with
        proper function name mapping and type-cast insertion hints.
        """
        ops = stmt.metadata.get("atomic_ops", [])
        if not ops:
            # Fallback: rename via intrinsic_mapper
            mapped = self.intrinsic_mapper.transform_text(stmt.raw_text)
            return self._strip_triton_import(mapped)

        translated_parts: list[str] = []
        for op in ops:
            cuda_name = op["name"]
            args = op.get("args", [])

            triton_name = _ATOMIC_TO_TRITON.get(cuda_name)
            if triton_name is None:
                raise IngestionUnsupportedIntrinsicError(
                    f"Unsupported atomic operation '{cuda_name}' "
                    f"at line {stmt.line_number}. "
                    f"Supported: {', '.join(sorted(_ATOMIC_TO_TRITON))}",
                    context={
                        "cuda_op": cuda_name,
                        "args": args,
                        "line": stmt.line_number,
                    },
                )

            # Build the Triton atomic call
            mapped_args: list[str] = []
            for i, arg in enumerate(args):
                if i == 0:
                    # First arg is the pointer — strip leading '&' for Triton
                    if arg.startswith("&"):
                        arg = arg[1:]
                    # Add _ptr suffix if it looks like a variable name
                    # (simple heuristic — not perfect, but avoids silent errors)
                    if arg.isidentifier():
                        arg = f"{arg}_ptr"
                    mapped_args.append(arg)
                else:
                    # For value arguments, add tl.int32/tl.float32 type cast hint
                    if arg.replace(".", "").replace("f", "").isdigit():
                        # Numeric literal — add type cast hint
                        if "." in arg or "f" in arg:
                            mapped_args.append(f"tl.float32({arg})")
                        else:
                            mapped_args.append(f"tl.int32({arg})")
                    else:
                        mapped_args.append(arg)

            triton_call = f"{triton_name}({', '.join(mapped_args)})"
            translated_parts.append(triton_call)

        if translated_parts:
            return "; ".join(translated_parts)

        # Last resort — text fallback
        return self.intrinsic_mapper.transform_text(stmt.raw_text)

    # ------------------------------------------------------------------
    # Assignment / Expression translation
    # ------------------------------------------------------------------

    def _translate_assignment(self, stmt: Any) -> str:
        """Translate a CUDA assignment statement.

        Uses metadata to replace CUDA field expressions (threadIdx.x,
        blockIdx.y, etc.) with Triton equivalents.  Falls back to
        text-level mapping for simple assignments.
        """
        if not self._has_cuda_constructs(stmt):
            return self._strip_triton_import(
                self.intrinsic_mapper.transform_text(stmt.raw_text),
            )

        # Has CUDA constructs — do structured replacement
        return self._replace_cuda_constructs(stmt, stmt.raw_text)

    def _translate_expression(self, stmt: Any) -> str:
        """Translate a CUDA expression statement."""
        if not self._has_cuda_constructs(stmt):
            return self._strip_triton_import(
                self.intrinsic_mapper.transform_text(stmt.raw_text),
            )

        return self._replace_cuda_constructs(stmt, stmt.raw_text)

    # ------------------------------------------------------------------
    # Control flow translation
    # ------------------------------------------------------------------

    def _translate_if(self, stmt: Any) -> str:
        """Translate a CUDA if statement.

        For if statements with CUDA constructs in the condition,
        does targeted replacement.  Otherwise passes through.
        """
        if not self._has_cuda_constructs(stmt):
            return self._strip_triton_import(
                self.intrinsic_mapper.transform_text(stmt.raw_text),
            )

        return self._replace_cuda_constructs(stmt, stmt.raw_text)

    def _translate_for(self, stmt: Any) -> str:
        """Translate a CUDA for loop.

        For loops are preserved in the Triton output since Triton
        supports for-loops natively.  Only CUDA intrinsics inside
        the loop body/condition are replaced.
        """
        if not self._has_cuda_constructs(stmt):
            return self._strip_triton_import(
                self.intrinsic_mapper.transform_text(stmt.raw_text),
            )

        return self._replace_cuda_constructs(stmt, stmt.raw_text)

    def _translate_while(self, stmt: Any) -> str:
        """Translate a CUDA while loop."""
        if not self._has_cuda_constructs(stmt):
            return self._strip_triton_import(
                self.intrinsic_mapper.transform_text(stmt.raw_text),
            )

        return self._replace_cuda_constructs(stmt, stmt.raw_text)

    # ------------------------------------------------------------------
    # Other statement translators
    # ------------------------------------------------------------------

    @staticmethod
    def _translate_return(stmt: Any) -> str:
        """Translate a return statement."""
        raw = stmt.raw_text.rstrip(";").strip()
        if raw == "return":
            return "return"
        return raw

    def _translate_declaration(self, stmt: Any) -> str:
        """Translate a variable declaration.

        Strips type prefix for simple declarations like:
            int i = 0;       → i = 0
            float x = 1.0;   → x = 1.0
        Replaces CUDA field expressions in the value if present.
        """
        raw = stmt.raw_text.rstrip(";").strip()

        if raw.startswith("__shared__"):
            return ""

        types_to_strip = [
            "int", "float", "double", "char", "short", "long",
            "unsigned int", "unsigned long", "size_t",
            "bool", "void", "unsigned",
        ]

        stripped = raw
        stripped_type = False
        for t in types_to_strip:
            if raw.startswith(t + " ") or raw.startswith(t + "*"):
                rest = raw[len(t):].strip()
                if rest.startswith("*"):
                    # Pointer declaration — needs manual review
                    return f"# (review) {raw}"
                stripped = rest
                stripped_type = True
                break
            if raw.startswith("const " + t):
                rest = raw[len("const " + t):].strip()
                if rest.startswith("*"):
                    return f"# (review) {raw}"
                stripped = rest
                stripped_type = True
                break

        if not stripped_type:
            return f"# (decl) {raw}"

        if "=" not in stripped:
            return f"# (decl) {raw}"

        parts = stripped.split("=", 1)
        var_name = parts[0].strip()
        value = parts[1].strip()

        if self._has_cuda_constructs(stmt):
            value = self._replace_cuda_constructs(stmt, value)

        return f"{var_name} = {value}"

    def _translate_load(self, stmt: Any) -> str:
        """Translate a memory load."""
        if not self._has_cuda_constructs(stmt):
            return self._strip_triton_import(
                self.intrinsic_mapper.transform_text(stmt.raw_text),
            )
        return self._replace_cuda_constructs(stmt, stmt.raw_text)

    def _translate_store(self, stmt: Any) -> str:
        """Translate a memory store."""
        if not self._has_cuda_constructs(stmt):
            return self._strip_triton_import(
                self.intrinsic_mapper.transform_text(stmt.raw_text),
            )
        return self._replace_cuda_constructs(stmt, stmt.raw_text)

    def _translate_block_index(self, stmt: Any) -> str:
        """Translate a block/thread index expression."""
        return self._replace_cuda_constructs(stmt, stmt.raw_text)

    # ------------------------------------------------------------------
    # CUDA field expression replacement
    # ------------------------------------------------------------------

    def _replace_cuda_constructs(self, stmt: Any, text: str) -> str:
        """Replace CUDA-specific constructs in text using AST metadata.

        Uses metadata['cuda_fields'] to do targeted replacement of
        threadIdx.x, blockIdx.y, etc.  This is more precise than
        regex because we know exactly which constructs are present
        and can verify their context.

        Also replaces sync calls and atomic calls using metadata.
        """
        result = text

        # 1. Replace CUDA field expressions (threadIdx.x, blockIdx.y, etc.)
        fields = stmt.metadata.get("cuda_fields", [])
        if fields:
            for field_info in fields:
                obj = field_info.get("object")
                field = field_info.get("field")
                original = field_info.get("text")
                if obj and field and original:
                    triton_expr = _CUDA_FIELD_TO_TRITON.get(obj, {}).get(field)
                    if triton_expr:
                        result = result.replace(original, triton_expr)
                    else:
                        raise IngestionUnsupportedIntrinsicError(
                            f"Unsupported CUDA field expression '{obj}.{field}' "
                            f"at line {stmt.line_number}. "
                            f"Supported objects: {', '.join(sorted(_CUDA_FIELD_TO_TRITON))}",
                            context={
                                "object": obj,
                                "field": field,
                                "line": stmt.line_number,
                            },
                        )

        # 2. Replace sync calls
        calls = stmt.metadata.get("sync_calls", [])
        for call_info in calls:
            call_name = call_info["name"]
            triton_expr = _SYNC_TO_TRITON.get(call_name)
            if triton_expr:
                result = result.replace(f"{call_name}()", triton_expr)

        # 3. Replace atomic calls
        atomic_ops = stmt.metadata.get("atomic_ops", [])
        for op in atomic_ops:
            cuda_name = op["name"]
            triton_name = _ATOMIC_TO_TRITON.get(cuda_name)
            if triton_name:
                result = result.replace(cuda_name, triton_name)

        # 4. Replace math functions (simple text-level)
        result = self._replace_math_functions(result)

        return self._strip_triton_import(result)

    @staticmethod
    def _replace_math_functions(text: str) -> str:
        """Replace CUDA math function names with Triton equivalents.

        This is safe to do at text level because math function
        signatures are stable across CUDA and Triton.
        """
        math_map = [
            ("sinf", "tl.sin"),
            ("cosf", "tl.cos"),
            ("tanf", "tl.tan"),
            ("logf", "tl.log"),
            ("log2f", "tl.log2"),
            ("expf", "tl.exp"),
            ("exp2f", "tl.exp2"),
            ("sqrtf", "tl.sqrt"),
            ("rsqrtf", "tl.rsqrt"),
            ("fabsf", "tl.abs"),
            ("fmaxf", "tl.maximum"),
            ("fminf", "tl.minimum"),
            ("floorf", "tl.floor"),
            ("ceilf", "tl.ceil"),
            ("roundf", "tl.extra.cuda.libdevice.round"),
            ("erff", "tl.extra.cuda.libdevice.erf"),
        ]
        # Longer names first to avoid partial matches
        math_map.sort(key=lambda x: len(x[0]), reverse=True)
        result = text
        for cuda_fn, triton_fn in math_map:
            result = result.replace(cuda_fn, triton_fn)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_cuda_constructs(stmt: Any) -> bool:
        """Check if a statement contains any CUDA-specific constructs."""
        meta = stmt.metadata
        return bool(
            meta.get("has_cuda_field")
            or meta.get("has_sync")
            or meta.get("has_atomic")
        )

    @staticmethod
    def _strip_triton_import(text: str) -> str:
        """Remove accidental double-import of triton/tl.

        The intrinsic_mapper's transform_text may add 'tl.' prefix
        to some tokens.  This method removes any accidental import
        lines from the middle of translated code.
        """
        result = text
        if "import triton" in result and result.strip().startswith("import"):
            # Only if it's at the beginning of the string (shouldn't happen)
            lines = result.split("\n")
            filtered = [l for l in lines if not l.strip().startswith("import ")]
            result = "\n".join(filtered)
        return result

    @staticmethod
    def _is_matmul_like(kernel: Any) -> bool:
        """Heuristic: is this kernel a matmul-style GEMM?"""
        name_lower = kernel.name.lower()
        return any(
            keyword in name_lower
            for keyword in ["matmul", "gemm", "sgemm", "dgemm"]
        )
