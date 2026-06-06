"""CUDA C++ source parser using tree-sitter AST.

Parses .cu files into a structured AST that the rest of the
ingestion pipeline operates on. Uses tree-sitter (with the C++
grammar) for correct handling of:
  - Template parameters
  - Namespace nesting (any depth)
  - Complex for-loops with multiple init/update statements
  - Nested angle brackets (templates inside templates)
  - Function pointers
  - Lambda expressions
  - CUDA qualifiers (__global__, __device__, __shared__)

The parser preserves the same CudaStatement / CudaKernel dataclasses
as the legacy regex-based parser for backward compatibility. Tests
written against the regex parser will continue to pass after updating
the import to CudaParser (which aliases TreeSitterCudaParser).

Architecture:
  [Source (.cu)] ──► tree-sitter-cpp AST ──► _find_cuda_functions()
                       │                        │
                       ▼                        ▼
                    function_definition      CudaKernel
                    nodes                     └─ body: CudaStatement[]
                                               └─ shared_mem_declarations
                                               └─ parameters
"""  # noqa: W505

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from src.common.errors import IngestionParseError
from src.common.logging import get_logger

logger = get_logger(__name__)


class CudaStatementType(Enum):
    """Types of CUDA statements we recognise."""
    FUNCTION_DEF = auto()        # __global__ / __device__ function
    SHARED_MEM = auto()          # __shared__ declaration
    ASSIGNMENT = auto()          # a = b;
    EXPRESSION = auto()          # standalone expression
    IF = auto()                  # if/else
    FOR = auto()                 # for loop
    WHILE = auto()               # while loop
    SYNC_THREADS = auto()        # __syncthreads()
    ATOMIC_OP = auto()           # atomicAdd, atomicCAS, etc.
    MEMORY_LOAD = auto()         # shared/global load
    MEMORY_STORE = auto()        # shared/global store
    RETURN = auto()              # return
    BLOCK_INDEX = auto()         # blockIdx / threadIdx usage
    DECLARATION = auto()         # variable declaration (non-__shared__)
    UNKNOWN = auto()


@dataclass
class CudaStatement:
    """A single statement in a CUDA kernel."""
    stmt_type: CudaStatementType
    raw_text: str
    line_number: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_control_flow(self) -> bool:
        return self.stmt_type in (
            CudaStatementType.IF,
            CudaStatementType.FOR,
            CudaStatementType.WHILE,
        )


@dataclass
class CudaKernel:
    """A parsed CUDA kernel function."""
    name: str
    qualifier: str  # "__global__" or "__device__"
    return_type: str
    parameters: list[dict[str, str]] = field(default_factory=list)
    body: list[CudaStatement] = field(default_factory=list)
    shared_mem_declarations: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_global(self) -> bool:
        return self.qualifier == "__global__"

    @property
    def is_device(self) -> bool:
        return self.qualifier == "__device__"

    @property
    def num_params(self) -> int:
        return len(self.parameters)

    @property
    def num_statements(self) -> int:
        return len(self.body)


class TreeSitterCudaParser:
    """Parses CUDA C++ source using tree-sitter for AST-level analysis.

    Usage:
        parser = TreeSitterCudaParser()
        kernels = parser.parse_file("path/to/kernel.cu")
        for kernel in kernels:
            # Use kernel.name, kernel.body, etc.
            ...

    Raises:
        IngestionParseError: if tree-sitter is not installed, or
            if the source contains unsupported constructs.
    """

    # Shared parser instance (tree-sitter is thread-safe for parsing)
    _parser: Any = None  # tree_sitter.Parser

    # Regex helpers for simple extractions from CUDA declarations
    _SHARED_MEM_TEXT_RE = re.compile(
        r'__shared__\s+'
        r'(\w[\w\s\*]*?)\s+'   # type
        r'(\w+)\s*'            # name
        r'(.*?)\s*;',          # rest (including array dims)
        re.DOTALL,
    )

    _CUDA_FIELD_NAMES: set[str] = {"threadIdx", "blockIdx", "blockDim", "gridDim"}
    _CUDA_SYNC_NAMES: set[str] = {"__syncthreads", "__syncwarp", "__threadfence"}
    _CUDA_ATOMIC_PREFIX = "atomic"

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Lazy-init the shared tree-sitter parser."""
        if cls._parser is not None:
            return
        try:
            import tree_sitter_cpp as tscpp
            from tree_sitter import Language, Parser  # type: ignore[import-untyped]
            cpp_language = Language(tscpp.language())
            cls._parser = Parser(cpp_language)
        except ImportError as exc:
            raise IngestionParseError(
                "tree-sitter and tree-sitter-cpp are required for CUDA parsing. "
                "Install with:\n"
                "    pip install tree-sitter>=0.20 tree-sitter-cpp>=0.20\n"
                "or (from this project):\n"
                "    pip install -e '.[cuda]'",
                context={"exc": str(exc)},
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_file(self, file_path: str) -> list[CudaKernel]:
        """Parse a .cu file and return all CUDA kernels found."""
        try:
            with open(file_path, encoding="utf-8") as f:
                source = f.read()
        except FileNotFoundError:
            raise IngestionParseError(
                f"CUDA source file not found: {file_path}",
                context={"file_path": file_path},
            )
        return self.parse_source(source)

    def parse_source(self, source: str) -> list[CudaKernel]:
        """Parse CUDA C++ source code and return all kernels."""
        self._ensure_initialized()
        # tree-sitter expects bytes
        source_bytes = source.encode("utf-8")
        tree = self._parser.parse(source_bytes)
        if tree is None:
            raise IngestionParseError(
                "tree-sitter failed to parse the source (returned None).",
            )

        kernels: list[CudaKernel] = []
        self._find_cuda_functions(tree.root_node, source_bytes, kernels)
        return kernels

    # ------------------------------------------------------------------
    # AST walking
    # ------------------------------------------------------------------

    def _find_cuda_functions(
        self,
        node: Any,  # tree_sitter.Node
        source_bytes: bytes,
        kernels: list[CudaKernel],
    ) -> None:
        """Recursively walk the AST, collecting CUDA kernel functions.

        This correctly handles functions nested inside:
          - template declarations
          - namespace definitions (at any depth)
          - extern "C" blocks
          - class/struct definitions
        """
        if node.type == "function_definition":
            qualifier = self._detect_cuda_qualifier(node, source_bytes)
            if qualifier is not None:
                kernel = self._build_kernel(node, source_bytes, qualifier)
                if kernel is not None:
                    kernels.append(kernel)
            # Do NOT recurse into function bodies for more kernel defs.
            return

        # Prune: skip function bodies (cannot contain kernel definitions)
        if node.type in ("compound_statement", "expression_statement"):
            return

        for child in node.children:
            self._find_cuda_functions(child, source_bytes, kernels)

    # ------------------------------------------------------------------
    # CUDA qualifier detection
    # ------------------------------------------------------------------

    def _detect_cuda_qualifier(
        self,
        func_node: Any,
        source_bytes: bytes,
    ) -> str | None:
        """Detect if a function_definition node is __global__ or __device__.

        tree-sitter-cpp does not natively understand CUDA __global__ /
        __device__ qualifiers.  We detect them by scanning the leading
        text of the node, which is robust for all supported cases.
        """
        start = func_node.start_byte
        # Need at most 12 bytes ("__device__" = 10, plus optional space)
        end = min(start + 14, func_node.end_byte)
        prefix = source_bytes[start:end].decode("utf-8", errors="replace").strip()
        if prefix.startswith("__global__"):
            return "__global__"
        if prefix.startswith("__device__"):
            return "__device__"
        return None

    # ------------------------------------------------------------------
    # Kernel construction
    # ------------------------------------------------------------------

    def _build_kernel(
        self,
        func_node: Any,
        source_bytes: bytes,
        qualifier: str,
    ) -> CudaKernel | None:
        """Build a CudaKernel from a function_definition tree-sitter node."""
        # Locate the function_declarator sub-tree
        declarator = self._find_child(func_node, "function_declarator")
        if declarator is None:
            raise IngestionParseError(
                "function_definition node has no function_declarator child",
                context={
                    "qualifier": qualifier,
                    "text": self._node_text(func_node, source_bytes)[:120],
                },
            )

        name = self._node_text(
            self._find_child(declarator, "identifier"), source_bytes,
        )
        if not name:
            raise IngestionParseError(
                "function_declarator has no identifier",
                context={"text": self._node_text(func_node, source_bytes)[:120]},
            )

        return_type = self._extract_return_type(func_node, source_bytes, qualifier)
        parameters = self._extract_parameters(declarator, source_bytes)

        # Locate the compound_statement (function body)
        body_node = self._find_child(func_node, "compound_statement")
        if body_node is None:
            body: list[CudaStatement] = []
            shared_mems: list[dict[str, Any]] = []
        else:
            body = self._extract_body(body_node, source_bytes)
            shared_mems = [
                stmt.metadata
                for stmt in body
                if stmt.stmt_type == CudaStatementType.SHARED_MEM
            ]

        return CudaKernel(
            name=name,
            qualifier=qualifier,
            return_type=return_type,
            parameters=parameters,
            body=body,
            shared_mem_declarations=shared_mems,
        )

    # ------------------------------------------------------------------
    # Return type extraction
    # ------------------------------------------------------------------

    def _extract_return_type(
        self,
        func_node: Any,
        source_bytes: bytes,
        qualifier: str,
    ) -> str:
        """Extract return type from between the CUDA qualifier and function name.

        Example:
            __global__ void kernel(...)  →  "void"
            __device__ int* func(...)    →  "int*"
        """
        declarator = self._find_child(func_node, "function_declarator")
        if declarator is None:
            return "void"

        qual_len = len(qualifier)
        start = func_node.start_byte + qual_len
        end = declarator.start_byte
        text = source_bytes[start:end].decode("utf-8", errors="replace").strip()
        # Strip any leading/trailing junk
        return text

    # ------------------------------------------------------------------
    # Parameter extraction
    # ------------------------------------------------------------------

    def _extract_parameters(
        self,
        declarator: Any,
        source_bytes: bytes,
    ) -> list[dict[str, str]]:
        """Extract typed parameters from a function_declarator node."""
        param_list = self._find_child(declarator, "parameter_list")
        if param_list is None:
            return []

        params: list[dict[str, str]] = []
        for child in param_list.children:
            if child.type == "parameter_declaration":
                parsed = self._parse_parameter(child, source_bytes)
                if parsed is not None:
                    params.append(parsed)
        return params

    def _parse_parameter(
        self,
        node: Any,
        source_bytes: bytes,
    ) -> dict[str, str] | None:
        """Parse a single parameter_declaration node.

        Returns dict with keys:
          - name: parameter name ('' if anonymous)
          - type: full type string (including const, *, &, etc.)
        """
        full_text = self._node_text(node, source_bytes)
        name = ""

        # Find the identifier (rightmost identifier that is a name, not a type)
        name_candidates: list[str] = []
        self._collect_identifiers(node, source_bytes, name_candidates)

        if name_candidates:
            name = name_candidates[-1]

        # Strip trailing array declarators like [16] from the type
        type_str = full_text
        if name and name in type_str:
            idx = type_str.rfind(name)
            type_str = type_str[:idx].strip()

        # Clean up the type string
        type_str = self._clean_type_string(type_str)

        return {"name": name, "type": type_str}

    @staticmethod
    def _collect_identifiers(
        node: Any,
        source_bytes: bytes,
        out: list[str],
    ) -> None:
        """Collect all identifier texts from a node subtree (recursive)."""
        if node.type == "identifier":
            text = source_bytes[node.start_byte:node.end_byte].decode(
                "utf-8", errors="replace",
            )
            out.append(text)
        for child in node.children:
            TreeSitterCudaParser._collect_identifiers(child, source_bytes, out)

    @staticmethod
    def _clean_type_string(raw: str) -> str:
        """Normalise a parameter type string.

        Strips function-level CUDA qualifiers (``__device__``,
        ``__global__``, ``__constant__``) and collapses whitespace.
        ``__restrict__`` is PRESERVED because downstream
        ``PointerAnalyzer`` uses it to mark the parameter as disjoint
        from every other restrict pointer (the C99 restrict contract).
        """
        cleaned = raw
        for q in ("__device__", "__global__", "__constant__"):
            cleaned = cleaned.replace(q, "")
        cleaned = " ".join(cleaned.split())
        return cleaned

    # ------------------------------------------------------------------
    # Body extraction
    # ------------------------------------------------------------------

    def _extract_body(
        self,
        body_node: Any,
        source_bytes: bytes,
    ) -> list[CudaStatement]:
        """Walk compound_statement children to produce CudaStatement list.

        Skips comments and empty nodes.  Each child is classified
        by type and annotated with semantic metadata used by the
        translator.
        """
        statements: list[CudaStatement] = []

        for child in body_node.children:
            # Skip non-named / anonymous children ("{", "}", ";")
            if not child.is_named:
                continue

            stmt_type, metadata = self._classify_body_node(child, source_bytes)

            # Skip comments
            if stmt_type is None:
                continue

            line_number = source_bytes[: child.start_byte].count(b"\n") + 1
            raw_text = self._node_text(child, source_bytes)

            stmt = CudaStatement(
                stmt_type=stmt_type,
                raw_text=raw_text,
                line_number=line_number,
                metadata=metadata,
            )
            statements.append(stmt)

        return statements

    def _classify_body_node(
        self,
        node: Any,
        source_bytes: bytes,
    ) -> tuple[CudaStatementType | None, dict[str, Any]]:
        """Classify a compound_statement child into a CudaStatementType.

        Returns (stmt_type, metadata) where:
          - stmt_type is None for nodes to skip (comments, etc.)
          - metadata contains AST-level info used by the translator.
        """
        metadata: dict[str, Any] = {
            "ast_type": node.type,
            "has_sync": False,
            "has_atomic": False,
            "has_cuda_field": False,
            "cuda_fields": [],
            "sync_calls": [],
            "atomic_ops": [],
        }

        if node.type == "declaration":
            # Check for __shared__ declaration
            node_text = self._node_text(node, source_bytes)
            if node_text.startswith("__shared__"):
                return CudaStatementType.SHARED_MEM, self._build_shared_mem_meta(
                    node, source_bytes,
                )
            # Scan for CUDA constructs inside non-shared declarations
            self._scan_for_cuda_constructs(node, source_bytes, metadata)
            return CudaStatementType.DECLARATION, metadata

        if node.type == "expression_statement":
            return self._classify_expression_statement(node, source_bytes, metadata)

        if node.type in ("if_statement",):
            # Scan the condition and body for CUDA constructs
            self._scan_for_cuda_constructs(node, source_bytes, metadata)
            return CudaStatementType.IF, metadata

        if node.type in ("for_statement",):
            self._scan_for_cuda_constructs(node, source_bytes, metadata)
            return CudaStatementType.FOR, metadata

        if node.type in ("while_statement",):
            self._scan_for_cuda_constructs(node, source_bytes, metadata)
            return CudaStatementType.WHILE, metadata

        if node.type == "return_statement":
            self._scan_for_cuda_constructs(node, source_bytes, metadata)
            return CudaStatementType.RETURN, metadata

        # Skip comments
        if node.type == "comment":
            return None, {}

        # Treat anything else as UNKNOWN expression
        self._scan_for_cuda_constructs(node, source_bytes, metadata)
        return CudaStatementType.UNKNOWN, metadata

    def _classify_expression_statement(
        self,
        node: Any,
        source_bytes: bytes,
        metadata: dict[str, Any],
    ) -> tuple[CudaStatementType, dict[str, Any]]:
        """Classify an expression_statement by examining its inner expression."""
        # The inner expression is usually the first named child
        expr = None
        for child in node.children:
            if child.is_named:
                expr = child
                break
        if expr is None:
            return CudaStatementType.EXPRESSION, metadata

        if expr.type == "call_expression":
            return self._classify_call_expression(expr, source_bytes, metadata)

        if expr.type == "assignment_expression":
            self._scan_for_cuda_constructs(expr, source_bytes, metadata)
            return CudaStatementType.ASSIGNMENT, metadata

        # Other expression types (binary, unary, etc.)
        self._scan_for_cuda_constructs(expr, source_bytes, metadata)
        # Check if it looks like an assignment (compound: +=, -=, etc.)
        expr_text = self._node_text(expr, source_bytes)
        if any(op in expr_text for op in ("+=", "-=", "*=", "/=")):
            return CudaStatementType.ASSIGNMENT, metadata
        return CudaStatementType.EXPRESSION, metadata

    def _classify_call_expression(
        self,
        expr: Any,
        source_bytes: bytes,
        metadata: dict[str, Any],
    ) -> tuple[CudaStatementType, dict[str, Any]]:
        """Classify a call_expression (sync, atomic, or other)."""
        call_name = self._extract_call_name(expr, source_bytes)

        if call_name in self._CUDA_SYNC_NAMES:
            metadata["has_sync"] = True
            metadata["sync_calls"].append({
                "name": call_name,
                "start_byte": expr.start_byte,
                "end_byte": expr.end_byte,
            })
            return CudaStatementType.SYNC_THREADS, metadata

        if call_name.startswith(self._CUDA_ATOMIC_PREFIX):
            metadata["has_atomic"] = True
            args = self._extract_call_args(expr, source_bytes)
            metadata["atomic_ops"].append({
                "name": call_name,
                "args": args,
            })
            metadata["cuda_fields"] = self._collect_field_expressions(
                expr, source_bytes,
            )
            return CudaStatementType.ATOMIC_OP, metadata

        # Regular function call — scan for CUDA constructs
        self._scan_for_cuda_constructs(expr, source_bytes, metadata)
        return CudaStatementType.EXPRESSION, metadata

    # ------------------------------------------------------------------
    # Shared memory metadata
    # ------------------------------------------------------------------

    def _build_shared_mem_meta(
        self,
        node: Any,
        source_bytes: bytes,
    ) -> dict[str, Any]:
        """Build metadata for a __shared__ declaration.

        Extracts type, name, and array dimensions from the raw text
        since tree-sitter-cpp may not parse CUDA declarations fully.
        """
        text = self._node_text(node, source_bytes)
        meta: dict[str, Any] = {
            "ast_type": "shared_mem",
            "raw": text,
            "type": "",
            "name": "",
            "dims": [],
        }

        match = self._SHARED_MEM_TEXT_RE.match(text)
        if match:
            type_str = match.group(1).strip()
            name = match.group(2).strip()
            rest = match.group(3).strip()
            meta["type"] = type_str
            meta["name"] = name
            # Extract array dimensions: [16][16] → ["16", "16"]
            dims = re.findall(r'\[([^\]]*)\]', rest)
            meta["dims"] = dims

        return meta

    # ------------------------------------------------------------------
    # CUDA construct scanning
    # ------------------------------------------------------------------

    def _scan_for_cuda_constructs(
        self,
        node: Any,
        source_bytes: bytes,
        metadata: dict[str, Any],
    ) -> None:
        """Walk a subtree looking for CUDA-specific constructs.

        Populates metadata dict with:
          - has_sync, sync_calls
          - has_atomic, atomic_ops
          - has_cuda_field, cuda_fields
        """
        if node.type == "call_expression":
            call_name = self._extract_call_name(node, source_bytes)
            if call_name in self._CUDA_SYNC_NAMES:
                metadata["has_sync"] = True
                metadata["sync_calls"].append({
                    "name": call_name,
                    "start_byte": node.start_byte,
                    "end_byte": node.end_byte,
                })
            elif call_name.startswith(self._CUDA_ATOMIC_PREFIX):
                metadata["has_atomic"] = True
                args = self._extract_call_args(node, source_bytes)
                metadata["atomic_ops"].append({
                    "name": call_name,
                    "args": args,
                })

        if node.type == "field_expression":
            field_info = self._extract_field_expression(node, source_bytes)
            if field_info is not None:
                metadata["has_cuda_field"] = True
                metadata["cuda_fields"].append(field_info)

        for child in node.children:
            self._scan_for_cuda_constructs(child, source_bytes, metadata)

    def _collect_field_expressions(
        self,
        node: Any,
        source_bytes: bytes,
    ) -> list[dict[str, Any]]:
        """Collect all CUDA field expressions from a subtree."""
        fields: list[dict[str, Any]] = []
        self._collect_field_exprs_recursive(node, source_bytes, fields)
        return fields

    def _collect_field_exprs_recursive(
        self,
        node: Any,
        source_bytes: bytes,
        out: list[dict[str, Any]],
    ) -> None:
        if node.type == "field_expression":
            info = self._extract_field_expression(node, source_bytes)
            if info is not None:
                out.append(info)
        for child in node.children:
            self._collect_field_exprs_recursive(child, source_bytes, out)

    # ------------------------------------------------------------------
    # Field expression helpers (threadIdx.x, blockIdx.y, etc.)
    # ------------------------------------------------------------------

    def _extract_field_expression(
        self,
        node: Any,
        source_bytes: bytes,
    ) -> dict[str, Any] | None:
        """Extract info from a field_expression node.

        Returns None if this is not a CUDA built-in field expression
        (e.g. it's a regular struct member access).
        """
        text = self._node_text(node, source_bytes)
        parts = text.split(".", 1)
        if len(parts) != 2:
            return None
        obj, field = parts
        if obj not in self._CUDA_FIELD_NAMES:
            return None
        return {
            "object": obj,
            "field": field,
            "text": text,
            "start_byte": node.start_byte,
            "end_byte": node.end_byte,
        }

    # ------------------------------------------------------------------
    # Call expression helpers (atomicAdd, __syncthreads, etc.)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_call_name(
        expr_node: Any,
        source_bytes: bytes,
    ) -> str:
        """Extract the called function name from a call_expression.

        The first child is either:
          - an "identifier" node for simple calls:  foo()
          - a "field_expression" node for method calls:  obj.foo()
        """
        for child in expr_node.children:
            if child.type == "identifier":
                return source_bytes[child.start_byte: child.end_byte].decode(
                    "utf-8", errors="replace",
                )
            if child.type == "field_expression":
                return source_bytes[child.start_byte: child.end_byte].decode(
                    "utf-8", errors="replace",
                )
        return source_bytes[expr_node.start_byte: expr_node.end_byte].decode(
            "utf-8", errors="replace",
        ).split("(")[0].strip()

    @staticmethod
    def _extract_call_args(
        expr_node: Any,
        source_bytes: bytes,
    ) -> list[str]:
        """Extract comma-separated argument texts from a call_expression.

        Respects nested parentheses and angle brackets.
        """
        arg_list = None
        for child in expr_node.children:
            if child.type == "argument_list":
                arg_list = child
                break
        if arg_list is None:
            return []

        args: list[str] = []
        current: list[str] = []
        depth = 0

        for child in arg_list.children:
            if not child.is_named and child.type in ("(", ")"):
                continue
            text = source_bytes[child.start_byte: child.end_byte].decode(
                "utf-8", errors="replace",
            )
            if child.type == "," and depth == 0:
                args.append("".join(current).strip())
                current = []
            else:
                # Track nesting for template/generic args
                for ch in text:
                    if ch in "<(":
                        depth += 1
                    elif ch in ">)":
                        depth -= 1
                current.append(text)

        if current:
            args.append("".join(current).strip())
        return args

    # ------------------------------------------------------------------
    # Tree-sitter helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_child(node: Any, child_type: str) -> Any | None:
        """Find the first child of a node with the given type."""
        for child in node.children:
            if child.type == child_type:
                return child
        return None

    @staticmethod
    def _node_text(node: Any, source_bytes: bytes) -> str:
        """Get the source text for a tree-sitter node."""
        return source_bytes[node.start_byte: node.end_byte].decode(
            "utf-8", errors="replace",
        )


# ------------------------------------------------------------------
# Backward-compatible alias
# ------------------------------------------------------------------

CudaParser = TreeSitterCudaParser
"""Alias for backward compatibility.

Legacy code importing `CudaParser` from this module will get
`TreeSitterCudaParser`, which produces identical dataclass types
but uses tree-sitter AST parsing instead of regex.
"""
