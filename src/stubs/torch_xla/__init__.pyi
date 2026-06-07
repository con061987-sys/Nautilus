from __future__ import annotations

from typing import Any

class device: ...


class stablehlo:
    class StableHLOProgram: ...
    def export(model: Any, args: tuple[Any, ...]) -> StableHLOProgram: ...


def save_as_stablehlo(model: Any, path: str) -> Any: ...


class contrib:
    class xla:
        @staticmethod
        def save_as_stablehlo(
            model: Any,
            args: tuple[Any, ...],
            path: str,
            *,
            undefok: str = ...,
        ) -> Any: ...
