"""TTGIR parser — turns real Triton GPU IR text into a structured AST.

The AST is the canonical representation that all 4 conversion passes
operate on. It's a Python dataclass tree, NOT MLIR objects, so the
passes are pure and unit-testable without MLIR.

The parser uses regex-based tokenization since the goal is to extract
semantic information, not parse every MLIR construct. For unsupported
constructs, the parser falls back to OpKind.UNKNOWN so the pipeline
can route the kernel to the template-based fallback.

Supported ops (Pass 1's working set):
  - tt.load, tt.store
  - arith.addf, arith.mulf, arith.divf, arith.subf
  - arith.addi, arith.muli, arith.subi
  - math.exp, math.log, math.sqrt, math.rsqrt, math.tanh, math.cos
  - arith.constant
  - tt.dot (split out for extern_bridge)
  - tt.reduce, tt.broadcast, tt.reshape
  - scf.for, scf.if
  - tt.get_program_id, tt.get_num_programs
  - tt.addptr (pointer arithmetic)
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum, auto

from src.common.logging import get_logger

logger = get_logger(__name__)


class OpKind(Enum):
    """Categorisation of ops for the 4-pass conversion."""
    # Triton-specific
    LOAD = auto()
    STORE = auto()
    DOT = auto()                # tt.dot / tt.dot_scaled — split out
    REDUCE = auto()
    BROADCAST = auto()
    RESHAPE = auto()
    TRANSPOSE = auto()          # tt.trans — permute last two dims
    MAKE_TENSOR_PTR = auto()
    ADVANCE = auto()
    GET_PROGRAM_ID = auto()
    GET_NUM_PROGRAMS = auto()
    ADDPTR = auto()
    MAX = auto()
    MIN = auto()
    RETURN = auto()             # tt.return — kernel terminator
    # Standard arith
    ADDF = auto()
    SUBF = auto()
    MULF = auto()
    DIVF = auto()
    ADDI = auto()
    SUBI = auto()
    MULI = auto()
    CONSTANT = auto()
    # Standard math
    EXP = auto()
    LOG = auto()
    SQRT = auto()
    RSQRT = auto()
    TANH = auto()
    COS = auto()
    SIN = auto()
    # Control flow
    FOR_LOOP = auto()
    IF_STATEMENT = auto()
    YIELD = auto()
    # Pass-4 materialization targets
    TVM_BLOCK = auto()
    TVM_INIT = auto()
    ALLOC_BUFFER = auto()
    # Everything else
    UNKNOWN = auto()


@dataclass
class TTGIRType:
    """Parsed type from TTGIR.

    Examples:
      !tt.ptr<tensor<128x32xf32>>  → pointer to 2D float32 tensor
      tensor<128x32xf32>           → 2D float32 tensor
      f32                          → scalar float32
      i32                          → scalar int32
    """
    raw: str
    is_pointer: bool = False
    is_tensor: bool = False
    element_dtype: str = "float32"
    shape: tuple[int, ...] = field(default_factory=tuple)
    address_space: int = 0  # 0 = global, 1 = shared, etc.

    def __post_init__(self) -> None:
        if self.element_dtype in ("f32",):
            self.element_dtype = "float32"
        elif self.element_dtype in ("f16",):
            self.element_dtype = "float16"
        elif self.element_dtype in ("bf16",):
            self.element_dtype = "bfloat16"
        elif self.element_dtype in ("i32",):
            self.element_dtype = "int32"
        elif self.element_dtype in ("i64",):
            self.element_dtype = "int64"


@dataclass
class TTGIROperation:
    """A single operation parsed from TTGIR.

    The AST is a flat list of operations; control flow (loops, ifs) is
    represented with parent_block / nesting. The parser captures the
    textual operands as raw strings — the conversion passes interpret
    them using the operation kind.
    """
    kind: OpKind
    raw_text: str
    name: str = ""                 # e.g. "tt.dot" or "arith.addf"
    result_name: str = ""          # e.g. "%result" — the SSA result
    operands: list[str] = field(default_factory=list)
    types: list[TTGIRType] = field(default_factory=list)
    attributes: dict[str, str] = field(default_factory=dict)
    # For control flow: indices into the parent function's ops list
    nested_ops: list[TTGIROperation] = field(default_factory=list)
    parent_idx: int = -1


@dataclass
class TTGIRFunction:
    """A parsed Triton kernel function."""
    name: str
    args: list[tuple[str, TTGIRType]] = field(default_factory=list)
    ops: list[TTGIROperation] = field(default_factory=list)

    # Module-level attributes (e.g. ttg.num-warps, ttg.num-stages)
    module_attrs: dict[str, str] = field(default_factory=dict)

    # Convenience accessors
    def has_dot(self) -> bool:
        """True if this function contains a tt.dot op."""
        return any(op.kind == OpKind.DOT for op in self.iter_all_ops())

    def iter_all_ops(self) -> Iterator[TTGIROperation]:
        """Iterate all ops including those nested in loops/ifs."""
        def _walk(ops: list[TTGIROperation]) -> Iterator[TTGIROperation]:
            for op in ops:
                yield op
                yield from _walk(op.nested_ops)
        yield from _walk(self.ops)

    def op_count(self) -> int:
        """Count all ops including nested ones."""
        return sum(1 for _ in self.iter_all_ops())


class TTGIRParser:
    """Parse real TTGIR text into a structured AST."""

    # Op name → OpKind mapping
    OP_KIND_MAP: dict[str, OpKind] = {
        "tt.load": OpKind.LOAD,
        "tt.store": OpKind.STORE,
        "tt.dot": OpKind.DOT,
        "tt.dot_scaled": OpKind.DOT,
        "tt.reduce": OpKind.REDUCE,
        "tt.broadcast": OpKind.BROADCAST,
        "tt.reshape": OpKind.RESHAPE,
        "tt.trans": OpKind.TRANSPOSE,
        "tt.return": OpKind.RETURN,
        "tt.make_tensor_ptr": OpKind.MAKE_TENSOR_PTR,
        "tt.advance": OpKind.ADVANCE,
        "tt.get_program_id": OpKind.GET_PROGRAM_ID,
        "tt.get_num_programs": OpKind.GET_NUM_PROGRAMS,
        "tt.addptr": OpKind.ADDPTR,
        "arith.addf": OpKind.ADDF,
        "arith.subf": OpKind.SUBF,
        "arith.mulf": OpKind.MULF,
        "arith.divf": OpKind.DIVF,
        "arith.addi": OpKind.ADDI,
        "arith.subi": OpKind.SUBI,
        "arith.muli": OpKind.MULI,
        "arith.constant": OpKind.CONSTANT,
        "math.exp": OpKind.EXP,
        "math.log": OpKind.LOG,
        "math.sqrt": OpKind.SQRT,
        "math.rsqrt": OpKind.RSQRT,
        "math.tanh": OpKind.TANH,
        "math.cos": OpKind.COS,
        "math.sin": OpKind.SIN,
        "scf.for": OpKind.FOR_LOOP,
        "scf.if": OpKind.IF_STATEMENT,
        "scf.yield": OpKind.YIELD,
    }

    # Regex patterns
    FUNC_DEF_RE = re.compile(
        r'tt\.func\s+(?:public\s+)?@(\w+)\s*\(([^)]*)\)\s*(?:->\s*[^{]*)?\s*\{',
    )
    OP_RE = re.compile(
        r'(%\w+)\s*=\s*([\w.]+)\s+(.*?)(?=\n\s+%\w+\s*=|\n\s*\}|\Z)',
        re.DOTALL,
    )
    MODULE_ATTR_RE = re.compile(r'(\w[\w.]+)\s*=\s*([^}\n]+)')
    POINTER_TYPE_RE = re.compile(
        r'!tt\.ptr<((?:tensor<[^>]+>)|(?:\w+))(?:,\s*(\d+))?>'
    )
    TENSOR_TYPE_RE = re.compile(r'tensor<([^>]+)>')
    DTYPE_RE = re.compile(r'(f(?:32|16|64)|bf16|i(?:32|64|8)|u(?:32|64|8))')

    def parse(self, ir_text: str) -> TTGIRFunction:
        """Parse real TTGIR text into a structured AST.

        Args:
            ir_text: The TTGIR text captured by the Triton backend plugin.

        Returns:
            A TTGIRFunction with the parsed ops and types.

        Raises:
            ValueError: If the IR cannot be parsed at all (malformed).
        """
        # Extract module-level attributes
        module_attrs = self._extract_module_attrs(ir_text)

        # Find the function definition
        func_match = self.FUNC_DEF_RE.search(ir_text)
        if not func_match:
            raise ValueError("No tt.func definition found in IR")

        func_name = func_match.group(1)
        args_str = func_match.group(2)

        # Parse function arguments
        args = self._parse_args(args_str)

        # Extract the function body
        body_start = func_match.end()
        body_end = self._find_matching_brace(ir_text, body_start - 1)
        body_text = ir_text[body_start:body_end]

        # Parse ops in the body
        ops = self._parse_ops(body_text)

        return TTGIRFunction(
            name=func_name,
            args=args,
            ops=ops,
            module_attrs=module_attrs,
        )

    def _extract_module_attrs(self, ir_text: str) -> dict[str, str]:
        """Extract ttg.* module-level attributes like num_warps, num_stages."""
        attrs: dict[str, str] = {}
        for m in self.MODULE_ATTR_RE.finditer(ir_text):
            key = m.group(1).strip()
            value = m.group(2).strip().rstrip(",")
            # We only care about ttg.* and tt.* module attributes
            if key.startswith("ttg.") or key.startswith("tt."):
                attrs[key] = value
        return attrs

    def _parse_args(self, args_str: str) -> list[tuple[str, TTGIRType]]:
        """Parse function arguments like '%A: !tt.ptr<tensor<128x32xf32>>'."""
        args: list[tuple[str, TTGIRType]] = []
        # Split on top-level commas
        parts = self._split_args(args_str)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                continue
            name, type_str = part.split(":", 1)
            name = name.strip()
            type_str = type_str.strip()
            type_obj = self._parse_type(type_str)
            args.append((name, type_obj))
        return args

    def _parse_type(self, type_str: str) -> TTGIRType:
        """Parse a type string into a TTGIRType."""
        type_str = type_str.strip()
        is_pointer = type_str.startswith("!tt.ptr")
        is_tensor = "tensor<" in type_str

        address_space = 0
        if is_pointer:
            m = self.POINTER_TYPE_RE.match(type_str)
            if m:
                # Strip the pointer wrapper to get the element type
                inner = m.group(1)
                if m.group(2):
                    address_space = int(m.group(2))
            else:
                inner = type_str[len("!tt.ptr<"):-1]
        else:
            inner = type_str

        # Parse element type
        element_dtype = "float32"
        shape: tuple[int, ...] = ()

        if inner.startswith("tensor<"):
            tensor_m = self.TENSOR_TYPE_RE.match(inner)
            if tensor_m:
                shape_str, dtype_str = self._split_tensor_type(tensor_m.group(1))
                shape = self._parse_shape(shape_str)
                element_dtype = self._normalize_dtype(dtype_str)
        else:
            element_dtype = self._normalize_dtype(inner)

        return TTGIRType(
            raw=type_str,
            is_pointer=is_pointer,
            is_tensor=is_tensor,
            element_dtype=element_dtype,
            shape=shape,
            address_space=address_space,
        )

    def _parse_shape(self, shape_str: str) -> tuple[int, ...]:
        """Parse '128x32' into (128, 32). '?' becomes -1."""
        parts = shape_str.split("x")
        result: list[int] = []
        for p in parts:
            p = p.strip()
            if p in ("?", "-1"):
                result.append(-1)
            else:
                try:
                    result.append(int(p))
                except ValueError:
                    result.append(-1)
        return tuple(result)

    def _split_tensor_type(self, content: str) -> tuple[str, str]:
        """Split '128x32xf32' into ('128x32', 'f32')."""
        # Find the last 'x' that introduces a dtype
        # The dtype is always a single segment at the end
        idx = content.rfind("x")
        if idx == -1:
            return ("", content)
        shape_part = content[:idx]
        dtype_part = content[idx+1:]
        return (shape_part, dtype_part)

    def _normalize_dtype(self, dtype: str) -> str:
        """Normalize MLIR dtype names to canonical Python forms."""
        mapping = {
            "f32": "float32", "f16": "float16", "f64": "float64",
            "bf16": "bfloat16",
            "i32": "int32", "i64": "int64", "i8": "int8",
            "u32": "uint32", "u64": "uint64", "u8": "uint8",
            "i1": "bool",
        }
        return mapping.get(dtype, dtype)

    def _split_args(self, args_str: str) -> list[str]:
        """Split arguments on top-level commas (not inside <>)."""
        parts: list[str] = []
        depth = 0
        current: list[str] = []
        for char in args_str:
            if char == "<":
                depth += 1
                current.append(char)
            elif char == ">":
                depth -= 1
                current.append(char)
            elif char == "," and depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(char)
        if current:
            parts.append("".join(current))
        return parts

    def _find_matching_brace(self, text: str, start: int) -> int:
        """Find the matching closing brace for an opening at start."""
        if text[start] != "{":
            raise ValueError(f"Expected '{{' at position {start}, got '{text[start]}'")
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return i + 1
        raise ValueError("Unmatched brace in IR")

    def _parse_ops(self, body_text: str) -> list[TTGIROperation]:
        """Parse ops from the function body.

        Uses a recursive descent approach with brace tracking.
        """
        ops: list[TTGIROperation] = []
        i = 0
        n = len(body_text)

        while i < n:
            # Skip whitespace
            while i < n and body_text[i] in " \t\n\r":
                i += 1
            if i >= n:
                break

            # Match an op definition. <op> is a dotted name (e.g. "tt.dot")
            # or a quoted generic op (e.g. "tt.reduce"); the trailing
            # space is optional because generic ops are immediately
            # followed by an argument list: "tt.reduce"(...).
            op_match = re.match(
                r'(%\w+)\s*=\s*("(?:[^"]+)"|[\w.]+)\s*',
                body_text[i:],
            )
            if op_match:
                result_name = op_match.group(1)
                op_name = op_match.group(2).strip('"')
                op_kind = self.OP_KIND_MAP.get(op_name, OpKind.UNKNOWN)

                # Capture the raw text of this op (until next op or closing brace)
                raw_end = self._find_op_end(body_text, i + op_match.end() - 1)
                raw_text = body_text[i:raw_end].strip()

                op = TTGIROperation(
                    kind=op_kind,
                    raw_text=raw_text,
                    name=op_name,
                    result_name=result_name,
                )
                # Parse operands and attributes
                op.operands, op.attributes = self._parse_operands(
                    body_text[i + op_match.end() - 1:raw_end],
                )

                # For control flow ops, recursively parse the body
                if op_kind in (OpKind.FOR_LOOP, OpKind.IF_STATEMENT):
                    brace_start = raw_end - 1
                    while brace_start > i + op_match.end() and body_text[brace_start] != "{":
                        brace_start -= 1
                    if body_text[brace_start] == "{":
                        body_end = self._find_matching_brace(body_text, brace_start)
                        inner = body_text[brace_start + 1:body_end - 1]
                        op.nested_ops = self._parse_ops(inner)

                # Reduce bodies use `^bb0(...)` block syntax the
                # line-oriented parser doesn't recurse into; surface
                # the combine op as an attribute instead.
                if op_kind == OpKind.REDUCE:
                    combine_op = self._extract_combine_op(
                        body_text[i + op_match.end() - 1:raw_end],
                    )
                    if combine_op is not None:
                        op.attributes["combine_op"] = combine_op

                op.types = self._extract_op_types(
                    body_text[i + op_match.end() - 1:raw_end],
                )

                ops.append(op)
                i = raw_end
                continue

            # Try to match a control-flow-only statement: scf.for ... { ... }
            scf_match = re.match(
                r'(scf\.\w+)\s+(.*?)\{',
                body_text[i:],
                re.DOTALL,
            )
            if scf_match:
                op_name = scf_match.group(1)
                op_kind = self.OP_KIND_MAP.get(op_name, OpKind.UNKNOWN)
                brace_pos = i + scf_match.end() - 1
                body_end = self._find_matching_brace(body_text, brace_pos)
                op = TTGIROperation(
                    kind=op_kind,
                    raw_text=body_text[i:body_end].strip(),
                    name=op_name,
                )
                if op_kind in (OpKind.FOR_LOOP, OpKind.IF_STATEMENT):
                    inner = body_text[brace_pos + 1:body_end - 1]
                    op.nested_ops = self._parse_ops(inner)
                ops.append(op)
                i = body_end
                continue

            # Statement-style op: no ``%result =`` prefix. Used for
            # ``tt.return`` / ``tt.store``, which occupy a single line.
            stmt_match = re.match(
                r'(tt\.\w+)\s+([^\n]*)',
                body_text[i:],
            )
            if stmt_match:
                op_name = stmt_match.group(1)
                op_kind = self.OP_KIND_MAP.get(op_name, OpKind.UNKNOWN)
                line_text = stmt_match.group(0)
                op = TTGIROperation(
                    kind=op_kind,
                    raw_text=line_text.strip(),
                    name=op_name,
                )
                rest = line_text[len(op_name):]
                op.operands, op.attributes = self._parse_operands(rest)
                ops.append(op)
                i += len(line_text)
                continue

            # Skip unrecognized content (one character at a time)
            i += 1

        return ops

    def _find_op_end(self, text: str, start: int) -> int:
        """Find the end of an op definition.

        An op ends at the next top-level newline followed by either a
        ``%``-prefixed op definition, a statement-style op
        (``tt.return``, ``tt.store``), or the closing brace of the
        enclosing block.
        """
        depth = 0
        n = len(text)
        i = start
        while i < n:
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                if depth == 0:
                    return i
                depth -= 1
            elif c == "\n" and depth == 0:
                j = i + 1
                while j < n and text[j] in " \t":
                    j += 1
                if j >= n:
                    return n
                if text[j] == "%":
                    rest = text[j:]
                    if (
                        re.match(r'%\w+\s*=\s*"(?:[^"]+)"\s*', rest)
                        or re.match(r'%\w+\s*=\s*[\w.]+\s*', rest)
                        or re.match(r'%\w+\s*=\s*scf\.', rest)
                    ):
                        return i
                elif re.match(
                    r'(tt|arith|math|scf|memref|llvm|affine)\.\w+',
                    text[j:],
                ):
                    return i
            i += 1
        return n

    def _parse_operands(
        self, op_text: str,
    ) -> tuple[list[str], dict[str, str]]:
        """Parse operands and attributes from op text.

        Returns (operands, attributes) where operands are SSA value
        references (%name) and attributes are key = value pairs.
        """
        operands: list[str] = []
        attrs: dict[str, str] = {}

        # Extract SSA value references
        for m in re.finditer(r'(%[\w.]+)', op_text):
            operands.append(m.group(1))

        # Extract attribute key = value pairs (e.g. eviction_policy = "evict_last")
        for m in re.finditer(r'(\w+)\s*=\s*("[^"]*"|\w+)', op_text):
            key = m.group(1)
            value = m.group(2).strip('"')
            attrs[key] = value

        return operands, attrs

    def _extract_combine_op(self, op_text: str) -> str | None:
        """Extract the combine op name from a tt.reduce's body region."""
        m = re.search(r'\^bb\d*\([^)]*\)\s*:\s*([\w.]+)', op_text)
        if m:
            return m.group(1)
        m = re.search(r'\b(arith|math|tt)\.\w+', op_text)
        if m:
            return m.group(0)
        return None

    def _extract_op_types(self, op_text: str) -> list[TTGIRType]:
        """Extract tensor types from a single op's text."""
        types: list[TTGIRType] = []
        for m in re.finditer(r'tensor<([^>]+)>', op_text):
            try:
                types.append(self._parse_type(f"tensor<{m.group(1)}>"))
            except Exception:
                continue
        return types
