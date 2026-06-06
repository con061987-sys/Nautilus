"""Dynamic signature inference for @triton.jit functions.

The legacy approach in the aot_packager backends hardcoded the signature
pattern (3 *fp32 pointers, 3 i32 ints, 3 constexprs). That worked for the
sample matmul kernel but fails for any kernel with a different shape:

  - Kernels that take fp16/bf16/fp64 pointers
  - Kernels with more or fewer scalar args
  - Kernels where constexprs are not the last 3 params
  - Kernels with int64 (i64) dims
  - Kernels with NO constexprs at all (e.g. simple pointwise)

This module introspects the @triton.jit function via inspect.signature and
builds a Triton ASTSource signature that matches the kernel's real
parameter list. Falls back to the legacy hardcoded pattern if introspection
fails for any reason, so the change is fully backward-compatible.

Public API:
    infer_kernel_signature(fn) -> (sig_args, constexpr_by_idx)
    build_signature(fn, block_size_values=...) -> (signature_dict, constexprs_dict)

The `build_signature` helper is what the backends call. It performs the
full inference + value assignment + legacy fallback in one shot.
"""

from __future__ import annotations

import inspect
import re
from typing import Any

# --- Heuristics for type inference ---------------------------------------

# Name suffixes that suggest a tensor pointer arg
_POINTER_NAME_SUFFIXES: tuple[str, ...] = (
    "_ptr", "ptr", "_buf", "_tensor", "_data", "_buffer",
)

# Known scalar-int arg names (case-sensitive for short names, case-insensitive
# fallback handled by the caller via .upper() comparison where needed).
_INT_NAMES: frozenset[str] = frozenset({
    "M", "N", "K", "dim", "size", "idx", "count", "rows", "cols",
    "head_dim", "seq_len", "batch", "num", "total",
    # common stride names
    "stride_am", "stride_ak", "stride_bk", "stride_bn",
    "stride_cm", "stride_cn", "stride_xm", "stride_xk",
    "stride_yk", "stride_yn",
    # constants that are sometimes passed as int
    "GROUP_M",
})

# Constexpr-style names: ALL_UPPER_CASE_OR_UNDERSCORE with at least one digit/letter
# Examples: BLOCK_M, NUM_WARPS, SPLIT_K, EVEN_N
_CONSTEXX_NAME_PATTERN = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$")

# Plain (single-word) constexpr names that match the pattern above
_PLAIN_CONSTEXX_NAMES: frozenset[str] = frozenset({
    "BLOCK_M", "BLOCK_N", "BLOCK_K", "BLOCK", "SIZE",
    "NUM_WARPS", "NUM_STAGES", "EVEN_N", "EVEN_K", "SPLIT_K",
    "DTYPE", "PRECISION", "GROUP_M", "INSTR_CTA",
    "BLOCK_H", "BLOCK_W", "BLOCK_D",
})


# --- Unwrapping the @triton.jit decorator --------------------------------


def _unwrap_jit(fn: Any) -> Any:
    """Get the underlying Python callable from a @triton.jit function.

    Triton 3.0+ wraps the original Python function in a JITFunction and
    stashes it at `.fn`. Older versions used `.run`. Fall back to fn
    itself if neither is present (the user may have passed the raw fn).
    """
    if hasattr(fn, "fn") and callable(getattr(fn, "fn", None)):
        return fn.fn
    if hasattr(fn, "run") and callable(getattr(fn, "run", None)):
        return fn.run
    return fn


# --- Annotation / name classifiers --------------------------------------


def _is_constexpr_annotation(annotation: Any) -> bool:
    """True if the parameter annotation indicates a constexpr.

    Detects:
      - `tl.constexpr` (exact class match)
      - any annotation whose name contains 'constexpr' (covers
        `_ConstexprType`, `JITConstexpr`, etc. across Triton versions)
      - string forward-refs like "tl.constexpr"
    """
    if annotation is inspect.Parameter.empty:
        return False
    name = getattr(annotation, "__name__", "") or ""
    mod = getattr(annotation, "__module__", "") or ""
    if "constexpr" in name.lower():
        return True
    if "triton" in mod and "constexpr" in name.lower():
        return True
    # String-form annotations (forward refs or repr)
    text = repr(annotation)
    return "constexpr" in text.lower()


def _is_pointer_like(name: str) -> bool:
    lower = name.lower()
    return any(lower.endswith(s) for s in _POINTER_NAME_SUFFIXES)


def _is_int_like(name: str) -> bool:
    if name in _INT_NAMES:
        return True
    if name.upper() in _INT_NAMES:
        return True
    return False


def _is_constexpr_name(name: str) -> bool:
    if name in _PLAIN_CONSTEXX_NAMES:
        return True
    return bool(_CONSTEXX_NAME_PATTERN.match(name))


# --- Public API ---------------------------------------------------------


def infer_kernel_signature(
    fn: Any,
    default_pointer_dtype: str = "*fp32",
    default_int_dtype: str = "i32",
) -> tuple[list[str], dict[int, str]]:
    """Infer a @triton.jit function's Triton ASTSource signature.

    Args:
        fn: a @triton.jit function (or its unwrapped form).
        default_pointer_dtype: Triton sig used for pointer args without
            explicit dtype annotation (default "*fp32").
        default_int_dtype: Triton sig used for int args without explicit
            dtype annotation (default "i32").

    Returns:
        sig_args: list of Triton type strings, in parameter order.
        constexpr_by_idx: {positional_index: parameter_name} for all
            params detected as constexprs. Empty if none.

    Raises:
        ValueError: if the function cannot be introspected (no
            parameters, or inspect.signature fails).
    """
    inner = _unwrap_jit(fn)
    try:
        sig = inspect.signature(inner)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Cannot inspect signature of {fn!r}: {exc}"
        ) from exc

    sig_args: list[str] = []
    constexpr_by_idx: dict[int, str] = {}

    for name, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            # *args / **kwargs are not expected in Triton kernels
            continue

        if (
            _is_constexpr_annotation(param.annotation)
            or _is_constexpr_name(name)
        ):
            sig_args.append("constexpr")
            constexpr_by_idx[len(sig_args) - 1] = name
        elif _is_pointer_like(name):
            sig_args.append(default_pointer_dtype)
        elif _is_int_like(name):
            sig_args.append(default_int_dtype)
        else:
            # Unknown shape — default to pointer dtype. Triton will emit
            # a clear error at AOT time if the param is actually a scalar,
            # which is safer than silently guessing wrong.
            sig_args.append(default_pointer_dtype)

    if not sig_args:
        raise ValueError(f"No parameters found in {fn!r}")

    return sig_args, constexpr_by_idx


def build_signature(
    fn: Any,
    block_size_values: dict[str, int] | None = None,
    default_pointer_dtype: str = "*fp32",
    default_int_dtype: str = "i32",
) -> tuple[dict[int, str], dict[int, Any]]:
    """Build (signature, constexprs) dicts for triton.ASTSource.

    Args:
        fn: a @triton.jit function.
        block_size_values: name -> value mapping for constexprs, e.g.
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}. Names
            not present in the kernel's constexprs are ignored.
        default_pointer_dtype: see infer_kernel_signature.
        default_int_dtype: see infer_kernel_signature.

    Returns:
        (signature, constexprs) tuple ready to pass to
        ``triton.compiler.ASTSource(signature=..., constexprs=...)``.

    Backward compat: if introspection fails OR the kernel has no
    constexprs, the legacy hardcoded pattern is used::

        ["*fp32"] * 3 + ["i32"] * 3 + ["constexpr"] * 3

    with block_size_values applied positionally to the last 3
    constexprs (in the order the values appear in the dict).
    """
    block_size_values = dict(block_size_values or {})

    try:
        sig_args, constexpr_by_idx = infer_kernel_signature(
            fn, default_pointer_dtype, default_int_dtype,
        )
    except (ValueError, TypeError):
        return _legacy_signature(block_size_values)

    if not sig_args:
        return _legacy_signature(block_size_values)

    # Try name-based mapping first (preferred: kernels may have
    # constexprs in any position, not just the end).
    constexprs: dict[int, Any] = {}
    for idx, name in constexpr_by_idx.items():
        if name in block_size_values:
            constexprs[idx] = block_size_values[name]

    # Backward-compat: if name-based mapping failed but we have the
    # legacy 3-tuple of block sizes, apply them positionally to the
    # last 3 constexprs. This preserves behavior for kernels whose
    # constexpr order is BLOCK_M, BLOCK_N, BLOCK_K.
    if not constexprs and len(constexpr_by_idx) >= 3 and len(block_size_values) == 3:
        indices = sorted(constexpr_by_idx.keys())
        values = list(block_size_values.values())
        for i, idx in enumerate(indices[-3:]):
            constexprs[idx] = values[i]

    # If we have no constexprs at all but the caller passed block sizes,
    # the kernel is probably the legacy 3-ptr shape — fall back to
    # the hardcoded pattern.
    if not constexprs and not constexpr_by_idx and block_size_values:
        return _legacy_signature(block_size_values)

    signature = {i: a for i, a in enumerate(sig_args)}
    return signature, constexprs


def _legacy_signature(
    block_size_values: dict[str, int],
) -> tuple[dict[int, str], dict[int, Any]]:
    """Return the legacy hardcoded signature pattern.

    Used as a fallback when introspection is unavailable or the kernel
    has no detectable constexprs. The pattern matches the matmul-shaped
    sample kernels the suite ships with: 3 fp32 ptrs + 3 i32 dims +
    3 constexpr block sizes.
    """
    sig_args = ["*fp32"] * 3 + ["i32"] * 3 + ["constexpr"] * 3
    signature = {i: a for i, a in enumerate(sig_args)}
    values = list(block_size_values.values()) if block_size_values else [128, 128, 32]
    if len(values) < 3:
        values = (values + [128, 128, 32])[:3]
    constexprs = {
        len(sig_args) - 3: values[0],
        len(sig_args) - 2: values[1],
        len(sig_args) - 1: values[2],
    }
    return signature, constexprs


__all__ = [
    "infer_kernel_signature",
    "build_signature",
]
