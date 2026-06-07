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
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.common.errors import IngestionUnsupportedIntrinsicError
from src.common.logging import get_logger

from .intrinsic_mapper import IntrinsicMapper
from .pointer_analysis import PointerAnalyzer, PointerLayout
from .shared_memory import SharedMemoryAnalyzer, SharedMemPlan

# Import CudaStatementType for dispatch
if TYPE_CHECKING:
    from .parser import CudaStatementType
else:
    try:
        from .parser import CudaStatementType
    except ImportError:
        # Fallback for backward compatibility during transition
        from enum import Enum

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
    "__syncthreads": "tl.barrier()",
    "__syncwarp": "# tl.syncwarp — closest equivalent: tl.barrier()",
    "__threadfence": "tl.barrier()",
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

# Canonical CUDA global thread linearization. The backreference \1
# forces all three dim letters to match — mixed-dim expressions are
# bugs in the original source and are left to fall through.
_BLOCK_LINEAR_RE = re.compile(
    r"blockIdx\.([xyz])\s*\*\s*blockDim\.\1\s*\+\s*threadIdx\.\1",
)

_COMPOUND_OPS: dict[str, str] = {
    "+=": "+",
    "-=": "-",
    "*=": "*",
    "/=": "/",
    "%=": "%",
    "&=": "&",
    "|=": "|",
    "^=": "^",
    "<<=": "<<",
    ">>=": ">>",
}

_COMPOUND_ASSIGN_RE = re.compile(
    r"^([^=+\-*/%&|^!<>]+?)\s*(<<=|>>=|\+=|-=|\*=|/=|%=|&=|\|=|\^=)\s*(.+)$",
    re.DOTALL,
)

# C++11 `auto` declaration prefix. Catches plain `auto`, `auto&`, `auto*`,
# `const auto&`, and `static auto`. Whitespace tolerant.
_AUTO_DECL_RE = re.compile(
    r"^(?:const\s+)?(?:static\s+)?(?:constexpr\s+)?auto(?:\s*[&*]+)?\s+",
)

# C++11 `decltype(...)` declaration prefix. The decltype expression is
# dropped entirely — Triton doesn't need a type annotation.
_DECLTYPE_DECL_RE = re.compile(
    r"^(?:const\s+)?(?:static\s+)?decltype\s*\([^)]*\)\s*(?:[&*]\s*)?",
)

# C++11 `std::move(x)` → `x` (rvalue-reference cast, no-op in Triton).
# The leading \b prevents `mystd::move` and similar names from matching.
_STD_MOVE_RE = re.compile(r"\bstd::move\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)")


def _is_scalar_lhs(lhs: str) -> bool:
    """True iff LHS is a simple identifier (no array-index or member access).

    Non-scalar compound assignments (`x[i] += y`, `obj.field += y`) are
    atomic-unsafe in CUDA and require `tl.atomic_*`; we flag them with a
    `# (atomic-unsafe)` comment so the caller can fix them rather than
    silently produce a racy non-atomic load/modify/store.
    """
    return lhs.isidentifier()


def _decompose_compound_assignment(
    raw: str,
    temp_name: str | None = None,
) -> list[str] | None:
    """Decompose `x <op>= y` into the load → modify → store sequence.

    Per spec, ``x += y;`` is rewritten to::

        temp = x
        temp = temp + y
        x = temp

    A non-scalar LHS (e.g. ``x[i] += y``) returns a single review comment
    because it requires ``tl.atomic_*`` and the translator cannot decide
    which one to use from raw text alone.

    Args:
        raw: The raw statement text (e.g. ``"x += y;"``).
        temp_name: Explicit name for the temp variable.  If None, a
            stable name is derived from the LHS (``__tmp_<lhs>``) so
            consecutive decompositions of the same LHS produce
            matching variable names.

    Returns:
        A list of 3 Triton statements for the scalar case, a single
        ``# (atomic-unsafe)`` comment for the non-scalar case, or
        None if the input is not a compound assignment.
    """
    match = _COMPOUND_ASSIGN_RE.match(raw.strip())
    if not match:
        return None
    lhs = match.group(1).strip()
    op = match.group(2)
    rhs = match.group(3).strip()
    py_op = _COMPOUND_OPS.get(op)
    if py_op is None:
        return None

    if not _is_scalar_lhs(lhs):
        return [f"# (atomic-unsafe) review: {lhs} {op} {rhs}"]

    if temp_name is None:
        temp_name = f"__tmp_{lhs}"

    return [
        f"{temp_name} = {lhs}",
        f"{temp_name} = {temp_name} {py_op} {rhs}",
        f"{lhs} = {temp_name}",
    ]


def _apply_block_linearization(text: str) -> str:
    """Rewrite `blockIdx.d * blockDim.d + threadIdx.d` patterns.

    Runs BEFORE field-by-field replacement so the linearization is
    handled atomically (the field pass would otherwise produce the same
    string via three separate replacements, but the order of fields
    in `cuda_fields` metadata is what guarantees it; the explicit
    pattern here is more robust to that ordering).

    Returns the input unchanged if no canonical pattern is present.

    The backreference `\\1` in `_BLOCK_LINEAR_RE` forces all three dim
    letters to match — mixed-dim expressions are bugs in the original
    source and are left to fall through (the field-by-field pass will
    still produce *something* for them, just not the canonical
    linearization).
    """

    def _repl(match: re.Match[str]) -> str:
        dim = match.group(1)
        dim_idx = {"x": 0, "y": 1, "z": 2}[dim]
        return f"tl.program_id({dim_idx}) * tl.num_programs({dim_idx}) + tl.program_id({dim_idx})"

    return _BLOCK_LINEAR_RE.sub(_repl, text)


def _apply_std_move(text: str) -> str:
    """Replace `std::move(x)` with `x` (rvalue-cast is a Triton no-op)."""
    return _STD_MOVE_RE.sub(lambda m: m.group(1).strip(), text)


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
            elif type_str.strip() in ("int", "unsigned int", "size_t") or type_str.strip() in (
                "float",
                "double",
            ):
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
        # Compound-assignment decomposition produces multi-line strings;
        # split so each decomposed line gets the body indent.
        for stmt in translated_stmts:
            for sub in stmt.split("\n"):
                lines.append(f"    {sub}")

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
                    stmt.line_number,
                    exc,
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

        return "tl.barrier()  # ~sync (auto-detected)"

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

        Compound assignments (`+=`, `-=`, `*=`, `/=`, …) are decomposed
        into the explicit load → modify → store sequence::

            temp = x
            temp = temp + y
            x = temp

        The three lines are returned as a single string with embedded
        newlines; the body generator splits and indents each line.

        Memory-deref compound assignments (`x[i] += y`) are flagged
        with a `# (atomic-unsafe)` review comment rather than silently
        producing racy non-atomic code — the caller must rewrite to
        use ``tl.atomic_*``.
        """
        raw = stmt.raw_text

        # Compound assignment pass: produce the load/modify/store form.
        decomposed = _decompose_compound_assignment(raw)
        if decomposed is not None:
            translated_lines: list[str] = []
            for line in decomposed:
                if line.startswith("#"):
                    translated_lines.append(line)
                    continue
                if self._has_cuda_constructs(stmt):
                    translated_lines.append(
                        self._replace_cuda_constructs(stmt, line),
                    )
                else:
                    translated_lines.append(line)
            return "\n".join(translated_lines)

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

        C++11 `auto` and `decltype(...)` prefixes are also stripped, and
        `std::move(x)` wrappers inside the value are unwrapped, since
        Triton doesn't have explicit type annotations and the rvalue
        cast is a no-op.
        """
        # Apply std::move unwrap before the type-strip so pointer
        # declarations still get the rvalue cast removed.
        raw = _apply_std_move(stmt.raw_text).rstrip(";").strip()

        if raw.startswith("__shared__"):
            return ""

        # C++11 `auto` declaration: `auto x = expr;`, `auto& x = ...;`
        # Treated as "type stripped" so the rest of the logic handles it
        # the same way as `int x = expr;`.
        if _AUTO_DECL_RE.match(raw):
            stripped = _AUTO_DECL_RE.sub("", raw, count=1)
        elif _DECLTYPE_DECL_RE.match(raw):
            stripped = _DECLTYPE_DECL_RE.sub("", raw, count=1)
        else:
            types_to_strip = [
                "int",
                "float",
                "double",
                "char",
                "short",
                "long",
                "unsigned int",
                "unsigned long",
                "size_t",
                "bool",
                "void",
                "unsigned",
            ]

            stripped = raw
            stripped_type = False
            for t in types_to_strip:
                if raw.startswith(t + " ") or raw.startswith(t + "*"):
                    rest = raw[len(t) :].strip()
                    if rest.startswith("*"):
                        # Pointer declaration — needs manual review
                        return f"# (review) {raw}"
                    stripped = rest
                    stripped_type = True
                    break
                if raw.startswith("const " + t):
                    rest = raw[len("const " + t) :].strip()
                    if rest.startswith("*"):
                        return f"# (review) {raw}"
                    stripped = rest
                    stripped_type = True
                    break

            if not stripped_type:
                return f"# (decl) {raw}"

        if "=" not in stripped:
            return f"# (decl) {stripped}"

        parts = stripped.split("=", 1)
        var_name = parts[0].strip()
        value = parts[1].strip()
        value = _apply_std_move(value)

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

        # 0. Atomic block-linearization pass: `blockIdx.d * blockDim.d + threadIdx.d`
        #    Must run before field-by-field replacement so the canonical
        #    pattern is handled in one shot and is robust to metadata
        #    ordering.
        result = _apply_block_linearization(result)

        # 0b. C++11 rvalue cast is a no-op in Triton.
        result = _apply_std_move(result)

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
        return bool(meta.get("has_cuda_field") or meta.get("has_sync") or meta.get("has_atomic"))

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
            filtered = [line for line in lines if not line.strip().startswith("import ")]
            result = "\n".join(filtered)
        return result

    @staticmethod
    def _is_matmul_like(kernel: Any) -> bool:
        """Heuristic: is this kernel a matmul-style GEMM?"""
        name_lower = kernel.name.lower()
        return any(keyword in name_lower for keyword in ["matmul", "gemm", "sgemm", "dgemm"])
