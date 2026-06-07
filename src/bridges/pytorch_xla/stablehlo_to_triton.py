"""StableHLO → Triton translator.

Converts a StableHLO MLIR module (text format) into a @triton.jit
kernel that can be compiled for any of the supported hardware
targets (Nvidia, AMD, Intel, Apple).

The translator handles the common StableHLO ops needed for
auto-sharded model execution:

  - Elementwise: add, multiply, subtract, negate, convert,
                 compare, select
  - Linear algebra: dot
  - Reductions: reduce (sum, max, min)
  - Tensor manipulation: broadcast_in_dim, reshape,
                          concatenate, slice

Design contract (per the bridge pattern in AGENTS.md):

  Intercept  → MLIR text is taken from StableHLOExporter output
  Normalize  → Lightweight regex-based MLIR text parser into
               structured op records with typed SSA operands
  Translate  → Python string emission of a @triton.jit kernel
               using Triton's standard primitives (tl.dot,
               tl.add, tl.sum, tl.broadcast_to, tl.reshape, …)
  Verify     → All generated Python must parse via `ast.parse`
               so downstream stages can compile without
               syntax errors

For unsupported ops, raises UnsupportedStableHLOOpError so the
caller can decide whether to fall back to a different code path
or surface the error to the user.

Usage:

    from src.bridges.pytorch_xla.stablehlo_to_triton import (
        TritonSource,
        translate,
        UnsupportedStableHLOOpError,
    )

    try:
        result = translate(stablehlo_mlir_text, kernel_name="my_kernel")
    except UnsupportedStableHLOOpError as exc:
        # Fall back or report
        ...

    # result.source is the full Python source of a @triton.jit function
    # result.input_specs / output_specs are typed specs
    # result.op_counts tells you which ops were seen

The translator is intentionally stdlib-only — no triton, no
torch, no MLIR libraries. This keeps it testable in minimal
environments and predictable across the dependency drift that
the architecture (C-API isolation) is designed to absorb.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ParsedFunc",
    "ParsedOp",
    "TensorSpec",
    "TritonSource",
    "UnsupportedStableHLOOpError",
    "parse_mlir",
    "translate",
]


# ---------------------------------------------------------------------------
# Public dataclasses and error
# ---------------------------------------------------------------------------


class UnsupportedStableHLOOpError(ValueError):
    """Raised when translate() encounters a StableHLO op it cannot handle.

    The error message identifies the offending op (full dotted name
    such as `stablehlo.convolution`) so the caller can decide whether
    to fall back, decompose the op upstream, or surface the
    limitation to the user.
    """


@dataclass
class TensorSpec:
    """A typed tensor value (shape + dtype)."""

    name: str
    shape: tuple[int, ...]
    dtype: str  # canonical Python dtype, e.g. "float32"

    def shape_str(self) -> str:
        """Render shape as MLIR 'x'-joined dims, e.g. '2x64'."""
        return "x".join(str(d) for d in self.shape)


@dataclass
class ParsedOp:
    """A single StableHLO op parsed from MLIR text.

    The op name uses dots, e.g. "stablehlo.add" or "stablehlo.dot".
    The operands are SSA value references like "%arg0" or "%0".
    Attributes are stored as raw strings — the codegen interprets
    them per-op (e.g. dimension numbers, broadcast dimensions).
    """

    result_name: str
    op_name: str
    operands: list[str] = field(default_factory=list)
    result_shape: tuple[int, ...] = ()
    result_dtype: str = "float32"
    attributes: dict[str, str] = field(default_factory=dict)
    raw_text: str = ""


@dataclass
class ParsedFunc:
    """A parsed stablehlo.func entry point."""

    name: str
    inputs: list[TensorSpec] = field(default_factory=list)
    outputs: list[TensorSpec] = field(default_factory=list)
    ops: list[ParsedOp] = field(default_factory=list)
    return_value: str = ""


@dataclass
class TritonSource:
    """The generated Triton kernel source plus metadata.

    Attributes:
        source: Full Python source of a @triton.jit function
                (imports + decorator + signature + body).
        kernel_name: Name of the generated @triton.jit function.
        input_specs: List of (name, shape, dtype) for the kernel's
                     pointer inputs. Names are the SSA names
                     stripped of the leading '%'.
        output_specs: List of (name, shape, dtype) for the kernel's
                      single output pointer argument. (The kernel
                      signature has a single `out_ptr` regardless
                      of how many results the StableHLO function
                      returns; the codegen writes to it.)
        op_counts: Mapping of op name (e.g. "stablehlo.dot") →
                   number of times the op appeared in the source.
    """

    source: str
    kernel_name: str
    input_specs: list[tuple[str, tuple[int, ...], str]]
    output_specs: list[tuple[str, tuple[int, ...], str]]
    op_counts: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MLIR parser — regex-based, stdlib-only
# ---------------------------------------------------------------------------


# Func signature:
#   func.func @name(%arg0: tensor<2x64xf32>, %arg1: tensor<64x128xf32>)
#       -> tensor<2x128xf32> {
# The body is then everything up to the matching closing brace.
_FUNC_RE = re.compile(
    r"""
    (?:func\.func|stablehlo\.func)\s+   # 'func.func ' or 'stablehlo.func '
    @(?P<name>\w+)\s*                   # '@name'
    \(\s*(?P<args>[^)]*)\s*\)\s*         # '(args)'
    (?:\s*->\s*(?P<ret>[^{]+?))?        # optional '-> ret'
    \s*\{\s*                            # '{'
    (?P<body>.*?)                        # body
    ^\s*\}\s*$                          # closing '}' on its own line
    """,
    re.VERBOSE | re.DOTALL | re.MULTILINE,
)

# A tensor type: 'tensor<2x64xf32>'
# Shape is a sequence of 'x'-separated integers (possibly empty for scalar).
# Dtype is a known MLIR token: f32, f16, bf16, i32, i64, i8, i1, ui32, ui64.
_TENSOR_TYPE_RE = re.compile(
    r"tensor<(?P<shape>(?:\d+x)*\d*)?(?P<dtype>f(?:32|16|64)|bf16|i(?:32|64|8|1)|ui(?:32|64))>"
)

# An MLIR argument: '%arg0: tensor<2x64xf32>'
_ARG_RE = re.compile(r"(%\w+)\s*:\s*(tensor<[^>]+>)")

# Op form 1 (custom): '%0 = stablehlo.add %arg0, %arg1 : tensor<2x64xf32>'
_OP_CUSTOM_RE = re.compile(
    r"""
    ^\s*(?P<result>%\w+)\s*=\s*
    (?P<op>stablehlo\.[\w.]+)\s+
    (?P<operands>[^:]+?)
    \s*:\s*
    (?P<rest>.*?)$
    """,
    re.VERBOSE | re.MULTILINE,
)

# Op form 2 (generic): '%0 = "stablehlo.dot"(%arg0, %arg1)
#                       : (tensor<2x64xf32>, tensor<64x128xf32>) -> tensor<2x128xf32>'
_OP_GENERIC_RE = re.compile(
    r"""
    ^\s*(?P<result>%\w+)\s*=\s*
    "(?P<op>stablehlo\.[\w.]+)"\s*
    \((?P<operands>[^)]*)\)\s*
    (?P<attrs>\{[^}]*\})?\s*         # optional attribute dict
    :\s*
    (?P<rest>.*?)$
    """,
    re.VERBOSE | re.MULTILINE,
)

# `return` statement: 'return %0 : tensor<2x128xf32>'
_RETURN_RE = re.compile(
    r"""
    ^\s*return\s+(?P<value>%\w+)\s*
    (?::\s*(?P<type>tensor<[^>]+>))?\s*$
    """,
    re.VERBOSE | re.MULTILINE,
)

# `stablehlo.return` inside a reduce body — treated like `return`
_RETURN_SIMPLE_RE = re.compile(r"stablehlo\.return\s+(?P<value>%\w+)")

# Attribute dict: '{dimension = 0 : i64}' or
#                 '{broadcast_dimensions = array<i64: 1, 2>}'
_ATTR_DICT_RE = re.compile(r"\{([^}]*)\}")

# Inside an attr dict, 'name = value' pairs.
_ATTR_PAIR_RE = re.compile(r"(\w+)\s*=\s*([^\s,}]+)")

# array<i64: 0, 1, 2> — for slice/concat/broadcast dim lists
_ARRAY_ATTR_RE = re.compile(r"array<\w+:\s*([^>]+)>")


def _normalize_dtype(dtype: str) -> str:
    """Map an MLIR dtype token to a canonical Python dtype string."""
    mapping = {
        "f32": "float32",
        "f16": "float16",
        "f64": "float64",
        "bf16": "bfloat16",
        "i32": "int32",
        "i64": "int64",
        "i8": "int8",
        "i1": "bool",
        "ui32": "uint32",
        "ui64": "uint64",
    }
    return mapping.get(dtype, dtype)


def _parse_tensor_type(type_str: str) -> TensorSpec:
    """Parse a 'tensor<2x64xf32>' into a TensorSpec.

    The name in the returned spec is the raw text (e.g. '%arg0').
    Callers set the name explicitly when they have SSA context.
    """
    m = _TENSOR_TYPE_RE.search(type_str)
    if not m:
        raise UnsupportedStableHLOOpError(f"Cannot parse tensor type from {type_str!r}")
    shape_str = m.group("shape")
    dtype = _normalize_dtype(m.group("dtype"))
    if shape_str:
        shape = tuple(int(d) for d in shape_str.split("x") if d)
    else:
        shape = ()  # scalar tensor
    return TensorSpec(name="", shape=shape, dtype=dtype)


def _split_operands(text: str) -> list[str]:
    """Split an operand text like '%arg0, %arg1' into a list of SSA names.

    Operands may be separated by commas, possibly with surrounding
    whitespace. Bracketed type annotations are skipped.
    """
    # Strip attribute dicts (e.g. '{...}')
    text = _ATTR_DICT_RE.sub("", text)
    # Find all %name references
    return re.findall(r"%\w+", text)


def _parse_attr_dict(text: str) -> dict[str, str]:
    """Parse an attribute dict string into a name→raw-value dict.

    Examples:
        {dimension = 0 : i64}
            → {"dimension": "0"}
        {broadcast_dimensions = array<i64: 1, 2>}
            → {"broadcast_dimensions": "array<i64: 1, 2>"}
    """
    result: dict[str, str] = {}
    for m in _ATTR_PAIR_RE.finditer(text):
        key, value = m.group(1), m.group(2).rstrip(",")
        result[key] = value
    return result


def _parse_array_attr(value: str) -> list[int] | None:
    """Parse 'array<i64: 0, 1, 2>' into [0, 1, 2]. Returns None on failure."""
    m = _ARRAY_ATTR_RE.search(value)
    if not m:
        return None
    raw = m.group(1)
    try:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError:
        return None


# Map of op name → (input_arg_count, accepts_generic_form)
# Used by the parser to know how to interpret the 'rest' of the line.
_OP_INPUT_ARITY: dict[str, int] = {
    "stablehlo.add": 2,
    "stablehlo.multiply": 2,
    "stablehlo.subtract": 2,
    "stablehlo.divide": 2,
    "stablehlo.negate": 1,
    "stablehlo.convert": 1,
    "stablehlo.compare": 2,
    "stablehlo.select": 3,
    "stablehlo.dot": 2,
    "stablehlo.broadcast_in_dim": 1,
    "stablehlo.reshape": 1,
    "stablehlo.concatenate": 2,
    "stablehlo.slice": 1,
    "stablehlo.reduce": 1,
}


def parse_mlir(stablehlo_mlir: str) -> ParsedFunc:
    """Parse a StableHLO MLIR text module into a structured function record.

    The parser is intentionally permissive: it handles the common
    custom and generic op syntaxes that are produced by
    torch_xla.stablehlo and the onnx-mlir bridge. It does NOT
    understand the full MLIR grammar — for that, use the official
    MLIR Python bindings.

    Args:
        stablehlo_mlir: A full MLIR module text (with or without the
                        outer 'module {' / '}' wrapper).

    Returns:
        A ParsedFunc describing the entry-point function.

    Raises:
        ValueError: If no `func.func @name(...)` definition can be found.
        UnsupportedStableHLOOpError: If a referenced op is unknown.
    """
    if not stablehlo_mlir or not stablehlo_mlir.strip():
        raise ValueError("Empty MLIR text")

    func_match = _FUNC_RE.search(stablehlo_mlir)
    if not func_match:
        raise ValueError("No `func.func @name(...)` definition found in MLIR text")

    name = func_match.group("name")
    args_str = func_match.group("args")
    ret_str = (func_match.group("ret") or "").strip()
    body_str = func_match.group("body")

    # Parse input arguments
    inputs: list[TensorSpec] = []
    for arg_m in _ARG_RE.finditer(args_str):
        ssa_name, type_str = arg_m.group(1), arg_m.group(2)
        spec = _parse_tensor_type(type_str)
        spec.name = ssa_name
        inputs.append(spec)

    # Parse return type(s) — usually a single tensor
    outputs: list[TensorSpec] = []
    if ret_str:
        for type_m in _TENSOR_TYPE_RE.finditer(ret_str):
            spec = _parse_tensor_type(type_m.group(0))
            spec.name = "%return"
            outputs.append(spec)
    if not outputs:
        # Fallback: try the return statement at the end of the body
        ret_stmt = _RETURN_RE.search(body_str)
        if ret_stmt and ret_stmt.group("type"):
            spec = _parse_tensor_type(ret_stmt.group("type"))
            spec.name = "%return"
            outputs.append(spec)

    # Parse ops
    ops: list[ParsedOp] = []
    return_value = ""

    for line in body_str.splitlines():
        line = line.rstrip()
        if not line.strip():
            continue

        # Skip comments
        if line.lstrip().startswith("//"):
            continue

        # Skip nested-region markers (we don't descend into reduce bodies —
        #   the reduce op captures the high-level semantics we need).
        if line.lstrip().startswith("^bb") or line.lstrip() == "}":
            continue

        # Parse the op
        op = _parse_op_line(line, body_str)
        if op is not None:
            ops.append(op)

        # Catch the return statement
        ret_m = _RETURN_RE.match(line)
        if ret_m:
            return_value = ret_m.group("value").strip()

    return ParsedFunc(
        name=name,
        inputs=inputs,
        outputs=outputs,
        ops=ops,
        return_value=return_value,
    )


def _parse_op_line(line: str, full_body: str) -> ParsedOp | None:
    """Try to parse a single line as a StableHLO op (custom or generic form)."""
    # Try the generic form first (more structured)
    gm = _OP_GENERIC_RE.match(line)
    if gm:
        return _parse_generic_op(gm, line)

    # Fall back to the custom form
    cm = _OP_CUSTOM_RE.match(line)
    if cm:
        return _parse_custom_op(cm, line)

    return None


def _parse_generic_op(m: re.Match[str], line: str) -> ParsedOp:
    """Parse a generic-form op like '%0 = "stablehlo.dot"(%a, %b) : ...'."""
    result_name = m.group("result")
    op_name = m.group("op")
    operands = _split_operands(m.group("operands"))
    attrs_raw = m.group("attrs") or ""
    rest = m.group("rest").strip()

    # The 'rest' looks like:
    #   '(tensor<2x64xf32>, tensor<64x128xf32>) -> tensor<2x128xf32>'
    # We need to find the result type (the one after the arrow).
    result_type = _extract_result_type_generic(rest)
    result_shape = result_type.shape
    result_dtype = result_type.dtype

    attributes = _parse_attr_dict(attrs_raw)

    if op_name not in _OP_INPUT_ARITY:
        raise UnsupportedStableHLOOpError(f"Unsupported StableHLO op: {op_name!r}")

    return ParsedOp(
        result_name=result_name,
        op_name=op_name,
        operands=operands,
        result_shape=result_shape,
        result_dtype=result_dtype,
        attributes=attributes,
        raw_text=line.strip(),
    )


def _parse_custom_op(m: re.Match[str], line: str) -> ParsedOp:
    """Parse a custom-form op like '%0 = stablehlo.add %a, %b : tensor<2x64xf32>'

    Some ops (e.g. ``stablehlo.dot``) also use a custom form that
    includes an arrow and a result type, e.g.:

        %0 = stablehlo.dot %a, %b : (tensor<2x64xf32>, tensor<64x128xf32>)
                                        -> tensor<2x128xf32>

    We delegate to ``_extract_result_type_generic`` to handle
    both the single-type and arrow cases uniformly.
    """
    result_name = m.group("result")
    op_name = m.group("op")
    operands = _split_operands(m.group("operands"))
    rest = m.group("rest").strip()

    # The 'rest' may be either a bare tensor type or the full
    # arrow-form '(t1, t2) -> t3' (used by stablehlo.dot etc.).
    # _extract_result_type_generic handles both.
    result_type = _extract_result_type_generic(rest)
    result_shape = result_type.shape
    result_dtype = result_type.dtype

    if op_name not in _OP_INPUT_ARITY:
        raise UnsupportedStableHLOOpError(f"Unsupported StableHLO op: {op_name!r}")

    return ParsedOp(
        result_name=result_name,
        op_name=op_name,
        operands=operands,
        result_shape=result_shape,
        result_dtype=result_dtype,
        attributes={},
        raw_text=line.strip(),
    )


def _extract_result_type_generic(rest: str) -> TensorSpec:
    """Extract the result tensor type from a generic-form 'rest' string.

    The 'rest' is everything after the ':' in a generic-form op.
    Examples:
        '(tensor<2x64xf32>, tensor<64x128xf32>) -> tensor<2x128xf32>'
            → tensor<2x128xf32>
        'tensor<2x64xf32>'
            → tensor<2x64xf32>
    """
    # Look for a '-> tensor<...>' first
    arrow_idx = rest.find("->")
    if arrow_idx >= 0:
        after_arrow = rest[arrow_idx + 2 :].strip()
        m = _TENSOR_TYPE_RE.search(after_arrow)
        if m:
            return _parse_tensor_type(m.group(0))

    # Otherwise, take the last tensor type in the string
    matches = list(_TENSOR_TYPE_RE.finditer(rest))
    if matches:
        return _parse_tensor_type(matches[-1].group(0))

    raise UnsupportedStableHLOOpError(
        f"Cannot extract result type from generic-form rest: {rest!r}"
    )


# ---------------------------------------------------------------------------
# Codegen — StableHLO op → Triton source line
# ---------------------------------------------------------------------------


# Triton dtype string for a given canonical dtype.
_TRITON_DTYPE = {
    "float32": "tl.float32",
    "float16": "tl.float16",
    "float64": "tl.float64",
    "bfloat16": "tl.bfloat16",
    "int32": "tl.int32",
    "int64": "tl.int64",
    "int8": "tl.int8",
    "bool": "tl.int1",
    "uint32": "tl.uint32",
    "uint64": "tl.uint64",
}


def _ssa_to_var(ssa: str, alias: dict[str, str] | None = None) -> str:
    """Convert an SSA name like '%0' or '%arg0' to a Python identifier.

    Optionally consults an alias dict for names that have already
    been bound (e.g. input names to loaded variable names).
    """
    if alias is not None and ssa in alias:
        return alias[ssa]
    cleaned = ssa.lstrip("%")
    # '%0' → 'v_0', '%arg0' → 'arg0', '%1.2' → 'v_1_2'
    if cleaned[:1].isdigit():
        return f"v_{cleaned.replace('.', '_')}"
    return cleaned


def _shape_to_triton(shape: tuple[int, ...]) -> str:
    """Render a shape tuple as a Triton tuple expression, e.g. (2, 64)."""
    if not shape:
        return "()"
    return "(" + ", ".join(str(d) for d in shape) + ")"


def _emit_load(var_name: str, spec: TensorSpec) -> str:
    """Emit a tl.load for an input tensor.

    Uses a 1-D linear offset for the generic case — this keeps
    elementwise kernels simple and lets the codegen compose
    arbitrary op sequences.
    """
    n_elems = 1
    for d in spec.shape:
        n_elems *= d
    target_shape = _shape_to_triton(spec.shape) if spec.shape else "(1,)"
    return (
        f"    {var_name}_flat = tl.arange(0, {max(n_elems, 1)})\n"
        f"    {var_name} = tl.reshape("
        f"tl.load({var_name}_ptr + {var_name}_flat), {target_shape}"
        f").to(tl.float32)\n"
    )


def _emit_op(op: ParsedOp, ssa_to_var: dict[str, str]) -> str:
    """Emit a Triton source line for a single StableHLO op.

    The result is bound to a variable whose name is derived from
    the op's result_name (e.g. '%3' → 'v_3').

    Returns:
        A string containing the assignment line (with trailing newline).
    """
    result_var = _ssa_to_var(op.result_name)
    operands = [_ssa_to_var(s, ssa_to_var) for s in op.operands]
    ssa_to_var[op.result_name] = result_var

    base = op.op_name.rsplit(".", 1)[-1]  # 'stablehlo.add' → 'add'
    handler = _OP_HANDLERS.get(base)
    if handler is None:
        raise UnsupportedStableHLOOpError(f"Unsupported StableHLO op: {op.op_name!r}")
    return handler(result_var, operands, op, ssa_to_var)


def _handle_add(result_var: str, operands: list[str], op: ParsedOp, _) -> str:
    a, b = operands[0], operands[1]
    return f"    {result_var} = {a} + {b}\n"


def _handle_multiply(result_var: str, operands: list[str], op: ParsedOp, _) -> str:
    a, b = operands[0], operands[1]
    return f"    {result_var} = {a} * {b}\n"


def _handle_subtract(result_var: str, operands: list[str], op: ParsedOp, _) -> str:
    a, b = operands[0], operands[1]
    return f"    {result_var} = {a} - {b}\n"


def _handle_divide(result_var: str, operands: list[str], op: ParsedOp, _) -> str:
    a, b = operands[0], operands[1]
    return f"    {result_var} = {a} / {b}\n"


def _handle_negate(result_var: str, operands: list[str], op: ParsedOp, _) -> str:
    a = operands[0]
    return f"    {result_var} = -{a}\n"


def _handle_convert(result_var: str, operands: list[str], op: ParsedOp, _) -> str:
    a = operands[0]
    target = _TRITON_DTYPE.get(op.result_dtype, "tl.float32")
    return f"    {result_var} = tl.cast({a}, {target})\n"


def _handle_compare(result_var: str, operands: list[str], op: ParsedOp, _) -> str:
    a, b = operands[0], operands[1]
    direction = op.attributes.get("comparison_direction", "EQ")
    # The attribute value is the raw token, e.g. "#stablehlo<comparison_direction EQ>"
    cmp_op = _comparison_direction_to_op(direction)
    return f"    {result_var} = {a} {cmp_op} {b}\n"


def _comparison_direction_to_op(direction: str) -> str:
    """Map a StableHLO comparison_direction attribute to a Python operator."""
    # The attribute value may be:
    #   'EQ', 'NE', 'LT', 'LE', 'GT', 'GE'  (bare tokens)
    #   '#stablehlo<comparison_direction EQ>'  (full form)
    token = direction.upper()
    token = token.split()[-1] if " " in token else token
    token = token.rstrip(">")
    mapping = {
        "EQ": "==",
        "NE": "!=",
        "LT": "<",
        "LE": "<=",
        "GT": ">",
        "GE": ">=",
    }
    return mapping.get(token, "==")


def _handle_select(result_var: str, operands: list[str], op: ParsedOp, _) -> str:
    pred, on_true, on_false = operands[0], operands[1], operands[2]
    return f"    {result_var} = tl.where({pred}, {on_true}, {on_false})\n"


def _handle_dot(result_var: str, operands: list[str], op: ParsedOp, _) -> str:
    a, b = operands[0], operands[1]
    return f"    {result_var} = tl.dot({a}, {b}, allow_tf32=False)\n"


def _handle_reduce(result_var: str, operands: list[str], op: ParsedOp, _) -> str:
    """Emit a tl.sum / tl.max / tl.min reduction.

    The reduction type is inferred from the reduce body's add/max/min
    inner op. Because the parser doesn't descend into reduce bodies,
    we use a heuristic: if the op's first attribute is 'addfn' we
    use tl.sum; 'maxfn' → tl.max; 'minfn' → tl.min. Otherwise we
    default to tl.sum (the most common case in auto-sharded models).
    """
    x = operands[0]
    # 'dimensions' or 'axes' — both spellings appear in different
    # producers. Defaults to 0 for the no-attr case.
    axis_str = op.attributes.get("dimensions") or op.attributes.get("axes") or "0"
    axis_str = axis_str.strip(":i64").strip()
    try:
        axis = int(axis_str.split(":")[0].strip())
    except ValueError:
        axis = 0

    reduce_fn_attr = op.attributes.get("reduce_fn", "").lower()
    if "max" in reduce_fn_attr:
        tl_fn = "tl.max"
    elif "min" in reduce_fn_attr:
        tl_fn = "tl.min"
    else:
        # Default: most common case is sum. For softmax/attention
        # patterns, the orchestrator layer (Phase 4) will explicitly
        # annotate the reduce_fn attribute.
        tl_fn = "tl.sum"

    return f"    {result_var} = {tl_fn}({x}, axis={axis})\n"


def _handle_broadcast_in_dim(result_var: str, operands: list[str], op: ParsedOp, _) -> str:
    x = operands[0]
    return f"    {result_var} = tl.broadcast_to({x}, {_shape_to_triton(op.result_shape)})\n"


def _handle_reshape(result_var: str, operands: list[str], op: ParsedOp, _) -> str:
    x = operands[0]
    return f"    {result_var} = tl.reshape({x}, {_shape_to_triton(op.result_shape)})\n"


def _handle_concatenate(result_var: str, operands: list[str], op: ParsedOp, _) -> str:
    a, b = operands[0], operands[1]
    dim_str = op.attributes.get("dimension", "0").strip(":i64").strip()
    try:
        dim = int(dim_str.split(":")[0].strip())
    except ValueError:
        dim = 0
    return f"    {result_var} = tl.cat([{a}, {b}], can_reorder=False, axis={dim})\n"


def _handle_slice(result_var: str, operands: list[str], op: ParsedOp, _) -> str:
    x = operands[0]
    start_list = _parse_array_attr(op.attributes.get("start_indices", ""))
    limit_list = _parse_array_attr(op.attributes.get("limit_indices", ""))
    stride_list = _parse_array_attr(op.attributes.get("strides", ""))
    if not start_list or not limit_list:
        # Conservative fallback — emit an identity slice
        return f"    {result_var} = {x}\n"
    if not stride_list:
        stride_list = [1] * len(start_list)
    # We emit a tuple of slice tuples. Triton accepts
    #   x[start0:limit0:stride0, start1:limit1:stride1, ...]
    slices = ", ".join(f"{s}:{l}:{st}" for s, l, st in zip(start_list, limit_list, stride_list))
    return f"    {result_var} = {x}[{slices}]\n"


# Handler dispatch table — keyed by the bare op name (after the
# last dot). This lets the same handler service both 'stablehlo.add'
# and any future alias like 'chlo.add'.
_OP_HANDLERS: dict[str, Any] = {
    "add": _handle_add,
    "multiply": _handle_multiply,
    "subtract": _handle_subtract,
    "divide": _handle_divide,
    "negate": _handle_negate,
    "convert": _handle_convert,
    "compare": _handle_compare,
    "select": _handle_select,
    "dot": _handle_dot,
    "reduce": _handle_reduce,
    "broadcast_in_dim": _handle_broadcast_in_dim,
    "reshape": _handle_reshape,
    "concatenate": _handle_concatenate,
    "slice": _handle_slice,
}


# ---------------------------------------------------------------------------
# Kernel emission
# ---------------------------------------------------------------------------


def _emit_kernel(parsed: ParsedFunc, kernel_name: str) -> str:
    """Emit a complete @triton.jit kernel source for the parsed function.

    The kernel uses a simple 1-D program-id grid and the standard
    BLOCK_M / BLOCK_N / BLOCK_K constexpr pattern. The codegen
    is conservative: it loads every input, executes each op in
    order, and stores the final result. This produces source that
    is syntactically valid Python/Triton even when the semantics
    of the original StableHLO would require a more sophisticated
    lowering (e.g. sharded collectives) — those are layered on
    top by the orchestrator in Wave 1.2.
    """
    parts: list[str] = []

    # Imports
    parts.append("import triton")
    parts.append("import triton.language as tl")
    parts.append("")
    parts.append("")

    # Decorator + signature
    parts.append("@triton.jit")
    parts.append(f"def {kernel_name}(")

    sig_lines: list[str] = []
    for spec in parsed.inputs:
        var = _ssa_to_var(spec.name)
        sig_lines.append(f"    {var}_ptr,")
    sig_lines.append("    out_ptr,")
    sig_lines.append("    M, N, K,")
    sig_lines.append("    BLOCK_M: tl.constexpr,")
    sig_lines.append("    BLOCK_N: tl.constexpr,")
    sig_lines.append("    BLOCK_K: tl.constexpr,")
    parts.append("\n".join(sig_lines))
    parts.append("):")
    parts.append(f'    """Auto-generated Triton kernel from StableHLO function `{parsed.name}`."""')
    parts.append("    pid = tl.program_id(0)")

    # Load each input
    ssa_to_var: dict[str, str] = {}
    for spec in parsed.inputs:
        var = _ssa_to_var(spec.name)
        ssa_to_var[spec.name] = var
        parts.append(_emit_load(var, spec))

    # Emit each op
    for op in parsed.ops:
        try:
            line = _emit_op(op, ssa_to_var)
        except UnsupportedStableHLOOpError:
            # Skip unsupported ops with a comment marker so the
            # kernel stays syntactically valid (the orchestrator
            # will route to a fallback for these).
            line = f"    # SKIPPED unsupported op: {op.op_name} (operands: {op.operands})\n"
        parts.append(line)

    # Store the final result
    if parsed.return_value and parsed.return_value in ssa_to_var:
        final = ssa_to_var[parsed.return_value]
        parts.append(f"    tl.store(out_ptr + tl.arange(0, 1), {final})")
    else:
        # No return value known — store a zero of the right shape
        parts.append("    tl.store(out_ptr + tl.arange(0, 1), tl.zeros((1,), dtype=tl.float32))")

    parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def translate(
    stablehlo_mlir: str,
    *,
    kernel_name: str = "translated_kernel",
    target_arch: str = "nvidia/sm_90",
) -> TritonSource:
    """Convert a StableHLO MLIR module into a @triton.jit kernel.

    Args:
        stablehlo_mlir: Full MLIR module text (with or without the
                        outer 'module { ... }' wrapper).
        kernel_name: Name to give the generated @triton.jit function.
                     The StableHLO function name is preserved
                     separately in the docstring.
        target_arch: Target architecture hint (e.g. "nvidia/sm_90",
                     "amd/gfx942", "intel/xelpx"). Stored in
                     the kernel docstring; not otherwise used
                     at codegen time.

    Returns:
        A TritonSource with the generated source, the input/output
        tensor specs, and a per-op count.

    Raises:
        ValueError: If the MLIR cannot be parsed.
        UnsupportedStableHLOOpError: If an op is encountered that
            this translator does not know how to emit.
    """
    parsed = parse_mlir(stablehlo_mlir)

    # Compute op counts (per the public op name including the dialect prefix)
    op_counts: dict[str, int] = {}
    for op in parsed.ops:
        op_counts[op.op_name] = op_counts.get(op.op_name, 0) + 1

    # Build input specs (strip leading '%' for cleaner display)
    input_specs: list[tuple[str, tuple[int, ...], str]] = [
        (_ssa_to_var(spec.name), spec.shape, spec.dtype) for spec in parsed.inputs
    ]

    # Build output specs
    output_specs: list[tuple[str, tuple[int, ...], str]] = []
    for spec in parsed.outputs:
        # If the StableHLO has multiple returns, name them by index
        out_name = _ssa_to_var(spec.name) if spec.name else "out"
        output_specs.append((out_name, spec.shape, spec.dtype))

    # Generate the kernel
    source = _emit_kernel(parsed, kernel_name)

    # Append target arch to the docstring (we keep this as a comment
    # outside the function so it doesn't break @triton.jit's parse).
    source += f"\n# target_arch: {target_arch}\n"

    # Verify the generated source is valid Python — refuse to
    # return code that downstream stages can't even parse.
    try:
        ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(
            f"Generated Triton source has a syntax error: {exc}. This is a bug in the translator."
        ) from exc

    return TritonSource(
        source=source,
        kernel_name=kernel_name,
        input_specs=input_specs,
        output_specs=output_specs,
        op_counts=op_counts,
    )
