from __future__ import annotations

from typing import Any

__version__: str


class Config:
    def __init__(self, kwargs: dict[str, Any] | Any = ..., num_warps: int = ..., num_stages: int = ..., **extra: Any) -> None: ...


class compiler:
    @staticmethod
    def compile(
        src: Any,
        target: str | Any = ...,
        options: dict[str, Any] | None = ...,
    ) -> Any: ...

class JITFunction: ...

class backends:
    backends: dict[str, Any] = ...

class GPUTarget:
    def __init__(self, backend: str, arch: int, warp_size: int) -> None: ...

class knobs:
    runtime: type[runtime_knobs]

    @staticmethod
    def get(key: str) -> Any: ...
    @staticmethod
    def set(key: str, value: Any) -> None: ...

class runtime_knobs:
    add_stages_inspection_hook: Any

def jit(fn: Any) -> JITFunction: ...
