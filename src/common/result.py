"""
Result[T, E] — Rust-style sum type for fallible operations.

Replaces the implicit "return None on failure" anti-pattern that
plagues the existing bridges. Every fallible function returns
Result[T, E] and the caller MUST explicitly handle both Ok and Err
variants (mypy --strict can enforce this).

Usage
-----

    from src.common.types import Result, Ok, Err
    from src.common.errors import TuningError

    def tune(kernel: str) -> Result[Mapping, TuningError]:
        if not kernel:
            return Err(TuningError("kernel source is empty"))
        ...
        return Ok({"block_m": 128, "num_warps": 8})

    match tune(src):
        case Ok(config):
            apply(config)
        case Err(e):
            log.error("tuning failed: %s", e)
            return Err(e)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")
E = TypeVar("E", bound=BaseException)
U = TypeVar("U")


@dataclass(frozen=True)
class Ok(Generic[T]):
    """Successful Result carrying value of type T."""

    value: T

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
        return False

    def unwrap(self) -> T:
        return self.value

    def unwrap_or(self, default: T) -> T:
        return self.value

    def unwrap_or_else(self, fn: Callable[[], T]) -> T:
        return self.value

    def map(self, fn: Callable[[T], U]) -> Ok[U]:
        return Ok(fn(self.value))

    def map_err(self, fn: Callable[[E], Any]) -> Ok[T]:
        return self

    def and_then(self, fn: Callable[[T], Result[U, E]]) -> Result[U, E]:
        return fn(self.value)

    def or_else(self, fn: Callable[[E], Result[T, Any]]) -> Ok[T]:
        return self

    def __repr__(self) -> str:
        return f"Ok({self.value!r})"


@dataclass(frozen=True)
class Err(Generic[E]):
    """Failed Result carrying error of type E (must be BaseException)."""

    error: E

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True

    def unwrap(self) -> Any:
        raise self.error

    def unwrap_or(self, default: T) -> T:
        return default

    def unwrap_or_else(self, fn: Callable[[E], T]) -> T:
        return fn(self.error)

    def map(self, fn: Callable[[T], U]) -> Err[E]:
        return self

    def map_err(self, fn: Callable[[E], Any]) -> Err:
        return Err(fn(self.error))

    def and_then(self, fn: Callable[[T], Result[U, E]]) -> Err[E]:
        return self

    def or_else(self, fn: Callable[[E], Result[T, Any]]) -> Result[T, Any]:
        return fn(self.error)

    def __repr__(self) -> str:
        return f"Err({self.error!r})"


# Type alias for the union; spelled out so type checkers accept both arms.
Result = Ok[T] | Err[E]


def is_ok(r: Result[T, E]) -> bool:
    """Type-narrowing helper."""
    return isinstance(r, Ok)


def is_err(r: Result[T, E]) -> bool:
    """Type-narrowing helper."""
    return isinstance(r, Err)


def try_catch(fn: Callable[..., T], *args: Any, **kwargs: Any) -> Result[T, Exception]:
    """Wrap a function call in a Result, catching any exception.

    Use sparingly. Prefer explicit Result returns from the callee when
    the callee can fail in a known way.
    """
    try:
        return Ok(fn(*args, **kwargs))
    except Exception as exc:
        return Err(exc)
